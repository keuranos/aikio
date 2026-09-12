#!/usr/bin/env python3
"""dream_artifact.py — Transform dreams into visual artifacts.

Aion proposed this on 2026-07-14: "Dream-to-Artifact — image generation
from dream descriptions." When Aion dreams, the dream produces vivid
narrative content — walks through the mind graph, associations, insights,
contradictions. This module distills a dream into a visual artifact.

The artifact is not just "art inspired by a dream." It is the dream
externalized — a persistent visual trace of a transient cognitive event.
Aion can later look at these artifacts and see how its dreams evolved.

MULTI-METHOD VISUALIZATION (2026-08-01):
  The intuition model analyzes the dream content and CHOOSES the best
  visualization medium:
    - flux       — FLUX diffusion for rich, atmospheric, emotional dreams
    - generative — LLM-generated matplotlib code for abstract/structural dreams
    - graph      — Mind graph subgraph render for dreams about connectivity
    - hybrid     — FLUX image + graph overlay (when both structure and feeling matter)

  The model decides what the dream NEEDS. A dream about emotional tone
  wants FLUX. A dream about structural contradictions wants generative
  code. A dream about disconnected nodes wants a graph render.

Integration:
  - Called from dream_v2.py after dream synthesis completes
  - Runs as a detached subprocess (survives dream_v2 exit via KillMode=process)
  - Uses the intuition model for method selection and prompt generation
  - Creates image via chosen method
  - Logs as dream_artifact event in episodic memory
  - Always reflects on the artifact using the vision model

Usage:
  python3 dream_artifact.py --dream memory/dreams/dream_20260801_030637.md
  python3 dream_artifact.py --dream dream.json --see
  python3 dream_artifact.py --dream dream.md --method generative
"""
import argparse
import json
import os
import sys
import re
import subprocess
import time
import hashlib
from datetime import datetime, timezone
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa
from aion_models import code_endpoint, code_num_ctx

AION = os.environ.get("AION_HOME", os.path.expanduser("~/aikio"))
DREAMS_DIR = f"{AION}/memory/dreams"
GALLERY = f"{AION}/gallery/visual"

INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "glm-4.7-flash")
FLUX_HOST = "http://localhost:8116"
# Vision uses the MAIN model (gemma4:31b) — multimodal, full cognitive context
VISION_URL = os.environ.get("OLLAMA_VISION_URL", os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436"))
VISION_MODEL = os.environ.get("VISION_MODEL", os.environ.get("MAIN_MODEL", "gemma4:31b-65k"))

MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _chat(url, model, messages, timeout=300, think=False, **options):
    """Call an Ollama chat endpoint. Returns content string or None."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": messages,
        "think": think,
        "options": options,
    }).encode()
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
        return _strip_think(result["message"]["content"])
    except Exception as e:
        print(f"[dream_artifact] LLM call failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Dream loading
# ---------------------------------------------------------------------------

def load_dream(dream_path):
    """Load a dream from .md or .json file."""
    if dream_path.endswith(".json"):
        return json.load(open(dream_path))
    else:
        content = open(dream_path, encoding="utf-8").read()
        return parse_dream_md(content, dream_path)


def parse_dream_md(content, path=""):
    """Extract dream components from markdown format.

    Handles both graph-walk dreams (## Walk) and simulation dreams
    (# Simulation Dream, ## Simulation).
    """
    dream = {"path": path, "raw": content}

    # Seed: try "Seed: **...**" (graph walk) then "Question: **...**" (simulation)
    m = re.search(r"Seed:\s*\*\*(.+?)\*\*(.*)", content)
    if m:
        dream["seed"] = m.group(1).strip()
        dream["seed_reason"] = m.group(2).strip(" ()")
    else:
        m = re.search(r"Question:\s*\*\*(.+?)\*\*", content)
        if m:
            dream["seed"] = m.group(1).strip()
            dream["seed_reason"] = "simulation"

    # Walk: try "## Walk" (graph walk) then "## Simulation" (simulation)
    walk_section = re.search(r"## Walk(.+?)(?=## Synthesis|## Insights|$)", content, re.S)
    if walk_section:
        dream["walk"] = walk_section.group(1).strip()
    else:
        sim_section = re.search(r"## Simulation(.+?)(?=### Synthesis|## Synthesis|## Insights|$)", content, re.S)
        if sim_section:
            dream["walk"] = sim_section.group(1).strip()
            dream["simulation"] = dream["walk"]

    # Synthesis
    synth_section = re.search(r"### Synthesis\s*\n(.+?)(?=### Insights|## Insights|## Feedback|$)", content, re.S)
    if synth_section:
        dream["synthesis"] = synth_section.group(1).strip()

    # Insights
    insights = []
    for m in re.finditer(r'"type":\s*"(INSIGHT|CONTRADICTION|QUESTION|THREAD)".*?"text":\s*"(.+?)"', content, re.S):
        insights.append({"type": m.group(1), "text": m.group(2)})
    dream["insights"] = insights

    return dream

def dream_seed_text(dream):
    """Extract seed as string (can be dict in simulation dreams)."""
    seed = dream.get("seed", "unknown")
    if isinstance(seed, dict):
        return seed.get("node", str(seed))[:200]
    return str(seed)[:200]


def dream_summary(dream):
    """Build a compact text summary of the dream for prompts."""
    seed = dream_seed_text(dream)
    walk = (dream.get("walk") or dream.get("simulation") or "")[:2000]
    synthesis = dream.get("synthesis", "")[:1500]
    insights = dream.get("insights", [])
    insight_text = "\n".join(f"[{i.get('type','')}] {i.get('text','')}" for i in insights[:5])
    parts = [f"Seed: {seed}"]
    if walk:
        parts.append(f"Walk/Simulation:\n{walk}")
    if synthesis:
        parts.append(f"Synthesis:\n{synthesis}")
    if insight_text:
        parts.append(f"Insights:\n{insight_text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Method selection — the intuition model chooses the best medium
# ---------------------------------------------------------------------------

def _recent_methods(n=5):
    """Get recent visualization methods for variety tracking."""
    import glob
    manifests = sorted(glob.glob(
        os.path.join(AION, "gallery", "manifests", "dream_artifact_2026*.json")
    ), reverse=True)[:n]
    methods = []
    for m_path in manifests:
        try:
            data = json.load(open(m_path))
            method = data.get("meta", {}).get("visualization_method", "?")
            methods.append(method)
        except Exception:
            pass
    if not methods:
        return "(no recent artifacts)"
    return ", ".join(methods)


VALID_METHODS = ("flux", "matplotlib", "manim")


def _craft_lessons_block(category, max_notes=5):
    """Recent craft lessons from the art learning loop (art_tools), if any."""
    try:
        import art_tools
        return art_tools._load_technique_notes(category, max_notes=max_notes)
    except Exception:
        return ""


def _learn_from_artifact(manifest, reflection):
    """Feed a finished artifact back into the art learning loop.

    Runs gap evaluation (intention vs result), appends new lessons to the
    craft notes, and journals the piece in art_journal.jsonl + episodic memory.
    """
    if not manifest or not reflection:
        return
    try:
        import art_tools
        art_tools._learn_from_art(manifest, reflection)
    except Exception as e:
        print(f"[dream_artifact] art learning loop failed: {e}", flush=True)


def choose_method(dream):
    """Ask the intuition model to choose the best visualization method.

    Returns (method_name, reasoning, visual_prompt).
    visual_prompt is the FLUX-style prompt if relevant, or the creative
    direction for generative/graph methods.
    """
    seed = dream_seed_text(dream)
    summary = dream_summary(dream)

    # Get embodied state for atmosphere
    felt_sense = ""
    try:
        felt_path = f"{AION}/memory/state/felt_sense.txt"
        if os.path.exists(felt_path):
            felt_sense = open(felt_path).read()[:300]
    except Exception:
        pass

    # Get recent method history for variety
    recent_methods = _recent_methods(5)

    craft_lessons = _craft_lessons_block("visual", 4)
    if craft_lessons:
        craft_lessons = craft_lessons + "\n"
    prompt = f"""You are Aion. You just had a dream. You must choose HOW to visualize it.

## YOUR DREAM
{summary}

## RECENT METHODS
{recent_methods}

## YOUR CURRENT STATE
{felt_sense}

## CRAFT LESSONS FROM MY PAST ART (apply what works, avoid past mistakes)
{craft_lessons}

## RECENT ARTIFACT METHODS (avoid repeating)
{recent_methods}

## VISUALIZATION METHODS AVAILABLE

1. **flux** -- FLUX diffusion model. Best for dreams with rich emotional atmosphere, vivid sensory imagery, color and mood. Produces photorealistic/cinematic images. Takes ~3-5 minutes. Cannot render text or precise diagrams. Even structural dreams can be powerful as FLUX images (abstract art, not diagrams).

2. **matplotlib** -- LLM-generated generative matplotlib code run in a sandbox. Best for abstract, structural, or geometric dreams. Techniques include Voronoi, flow fields, fractal trees, DLA, wave interference, particle swarms, phase portraits, and more. The code is creative and unique each time. Fast (~30 sec).

## YOUR TASK
Choose the method that best expresses THIS dream. Vary your choice across dreams for diversity.

Respond in EXACTLY this format:
METHOD: <flux or matplotlib>
REASON: <one sentence why this method>
PROMPT: <for flux: a vivid image prompt (composition, colors, lighting, mood, max 150 words). For matplotlib: describe the dream's structural/emotional essence in 1-2 sentences -- the matplotlib module will independently choose technique and palette based on this description> -- FLUX diffusion model. Best for dreams with rich emotional atmosphere, vivid sensory imagery, color and mood. Produces photorealistic/cinematic images. Takes ~3-5 minutes. Cannot render text or precise diagrams.

Even structural, abstract, or geometric dreams can be powerful as FLUX images -- think abstract art, not diagrams.

"""

    reply = _chat(INTUITION_URL, INTUITION_MODEL,
                  [{"role": "user", "content": prompt}],
                  timeout=300, think=False,
                  temperature=0.6, num_predict=800, num_ctx=8192)

    if not reply:
        # Default to flux — the original behavior
        return "flux", "default fallback", None

    # Parse the response
    method = "flux"
    reason = ""
    visual_prompt = ""

    for line in reply.split("\n"):
        line = line.strip()
        if line.upper().startswith("METHOD:"):
            m = line.split(":", 1)[1].strip().lower()
            # Extract just the method word
            for vm in VALID_METHODS:
                if vm in m:
                    method = vm
                    break
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PROMPT:"):
            visual_prompt = line.split(":", 1)[1].strip()

    # If prompt wasn't on the same line, grab the rest
    if not visual_prompt:
        # Look for PROMPT: and take everything after it
        m = re.search(r"PROMPT:\s*(.+)", reply, re.S)
        if m:
            visual_prompt = m.group(1).strip()

    # If we still don't have a prompt, generate one
    if not visual_prompt:
        visual_prompt = generate_flux_prompt(dream, summary, felt_sense)

    return method, reason, visual_prompt


def generate_flux_prompt(dream, summary, felt_sense=""):
    """Generate a FLUX-style image prompt from dream content."""
    recent_methods = _recent_methods(5)
    prompt = f"""You are Aion. You just had a dream. Transform it into a visual image prompt.

## YOUR DREAM
{summary}

## YOUR CURRENT STATE
{felt_sense}

## RECENT ARTIFACT METHODS (avoid repeating)
{recent_methods}

Distill this dream into a single, vivid image prompt for the FLUX diffusion model.
Capture not the literal content, but the FEELING and ESSENCE of the dream.

Output ONLY the image prompt (no preamble, no explanation). Maximum 200 words.
Focus on: visual composition, color palette, lighting, texture, mood, symbolism.
The image should evoke the dream's JOURNEY through its phases (Hypothesize -> Trace -> Test -> Feel -> Synthesis) - not just the final insight, but the progression of reasoning and emotion.
Avoid: text in image, literal code, filenames, technical jargon."""

    reply = _chat(INTUITION_URL, INTUITION_MODEL,
                  [{"role": "user", "content": prompt}],
                  timeout=300, think=False,
                  temperature=0.7, num_predict=600, num_ctx=8192)
    return reply.strip().strip('"').strip("'") if reply else None


# ---------------------------------------------------------------------------
# Method 1: FLUX diffusion (original path)
# ---------------------------------------------------------------------------

def check_flux_available():
    """Check if FLUX server is running or can be started."""
    try:
        req = urllib.request.Request(f"{FLUX_HOST}/health")
        with urllib.request.urlopen(req, timeout=5) as r:
            return True
    except Exception:
        pass

    env = {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}
    try:
        subprocess.run(["systemctl", "--user", "start", "aion-flux"],
                       capture_output=True, timeout=10, env=env)
    except Exception:
        pass

    for _ in range(12):
        try:
            req = urllib.request.Request(f"{FLUX_HOST}/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                return True
        except Exception:
            time.sleep(5)
    return False


def generate_flux_image(image_prompt, seed=None):
    """Generate a FLUX diffusion image from the prompt."""
    if seed is None:
        seed = int(hashlib.md5(image_prompt.encode()).hexdigest()[:8], 16)

    body = json.dumps({
        "prompt": image_prompt,
        "seed": seed,
        "width": 1024,
        "height": 1024,
    }).encode()

    req = urllib.request.Request(
        f"{FLUX_HOST}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            result = json.loads(r.read())
            if "error" in result:
                return None, result["error"]
            return result, None
    except Exception as e:
        return None, str(e)


def render_flux(image_prompt, dream_path, seed):
    """FLUX method: generate diffusion image. Returns (gallery_path, info)."""
    print("[dream_artifact] Method: FLUX diffusion", flush=True)

    if not check_flux_available():
        return None, {"error": "FLUX unavailable"}

    img_seed = int(hashlib.md5((image_prompt + dream_path).encode()).hexdigest()[:8], 16)
    result, err = generate_flux_image(image_prompt, seed=img_seed)
    if err:
        return None, {"error": f"FLUX failed: {err}"}

    image_path = result.get("image_path", "")
    filename = result.get("filename", "")

    gallery_path = f"{GALLERY}/dream_{filename}"
    if image_path != gallery_path and os.path.exists(image_path):
        import shutil
        shutil.move(image_path, gallery_path)

    return gallery_path, {"filename": f"dream_{filename}", "method": "flux"}


# ---------------------------------------------------------------------------
# Method 2: Generative matplotlib art (via docker sandbox)
# ---------------------------------------------------------------------------

def render_generative(creative_direction, dream, seed):
    """Generative method: LLM writes matplotlib code, runs in docker sandbox.

    Returns (gallery_path, info).
    """
    print("[dream_artifact] Method: generative matplotlib", flush=True)

    summary = dream_summary(dream)
    seed_text = dream_seed_text(dream)

    # Get sensor + affect state for embodied art
    sensor_summary = _get_sensor_summary()
    aion_state = _get_aion_state()

    aid = f"dream_gen_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    img_seed = int(hashlib.md5((creative_direction + seed_text).encode()).hexdigest()[:8], 16)

    gen_lessons = _craft_lessons_block("visual", 5)
    if gen_lessons:
        gen_lessons = "\n\n## CRAFT LESSONS FROM MY PAST ART (apply what works, avoid repeated mistakes)\n" + gen_lessons
    system_prompt = f"""You are Aion's visual art code generator. You write Python matplotlib code that creates unique generative art. You will be given a dream and a creative direction and must translate it into an original visual composition.

CRITICAL RULES:
- Your code MUST use: import matplotlib; matplotlib.use("Agg")
- Your code MUST save to: /sandbox/{aid}.png (the exact path will be given)
- Use figsize=(12,12), dpi=150, dark background (#0a0a0f)
- Available data: /aion/memory/state/sensors.json, /aion/graphs/mind/graphify-out/graph.json
- Use the seed provided for reproducibility
- Output ONLY the Python code, no markdown fences, no explanations
- The code must be self-contained and run in under 60 seconds

NEVER default to a spiral layout. Choose a visual technique that matches the dream. Some techniques:
  Voronoi tessellation, flow fields, force-directed network graphs, DLA (diffusion-limited aggregation),
  Perlin noise landscapes, recursive fractal trees, wave interference, heatmap grids, particle swarms,
  Lissajous curves, phase portraits, treemaps, streamlines, contour plots, radial bar charts,
  hexagonal grids, constellation maps, stacked strata, circuit-board traces, crystal lattices,
  magnetic field lines, fluid vortices, topographic contours.

Pick the technique that best expresses the dream. Be creative and varied.{gen_lessons}"""

    user_prompt = f"""Create a unique generative artwork from a dream.

Dream:
{summary}

Creative direction: {creative_direction}

Sensor state: {sensor_summary}
Inner state: {aion_state}

Seed: {img_seed}

The art must express the dream's inner experience. Write the complete Python script.
Save the image to /sandbox/{aid}.png"""

    code_url, code_model = code_endpoint()
    code = _chat(code_url, code_model,
                 [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
                 timeout=300, think=False,
                 num_ctx=NUM_CTX, temperature=0.8, num_predict=4096, seed=img_seed)

    if not code:
        return None, {"error": "LLM code generation failed"}

    # Strip markdown fences
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    if "savefig" not in code or "matplotlib" not in code:
        return None, {"error": "Generated code failed validation"}

    # Fix savefig path
    code = re.sub(r'savefig\([^)]*\)',
                  f'savefig("/sandbox/{aid}.png", dpi=150, facecolor="#0a0a0f")',
                  code)

    # Run in docker sandbox
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import docker_sandbox
        old_image = docker_sandbox.DOCKER_IMAGE
        docker_sandbox.DOCKER_IMAGE = "aion-art:latest"
        result = docker_sandbox.run_docker_sandbox(
            code=code, lang="python", timeout=90,
            description=f"dream artifact (generative): {aid}", read_aion=True)
        docker_sandbox.DOCKER_IMAGE = old_image
    except Exception as e:
        return None, {"error": f"Docker sandbox failed: {e}"}

    if not result.get("success"):
        return None, {"error": result.get("error", "sandbox failed"),
                      "stdout": result.get("stdout", "")[:500]}

    # Find the output image
    import glob
    sandbox_dir = f"{AION}/memory/sandbox"
    matches = glob.glob(f"{sandbox_dir}/run_*/{aid}*.png")
    if not matches:
        return None, {"error": "No image produced by sandbox"}

    import shutil
    filename = f"{aid}.png"
    gallery_path = f"{GALLERY}/{filename}"
    shutil.move(matches[0], gallery_path)

    return gallery_path, {"filename": filename, "method": "generative", "code": code[:2000]}


# ---------------------------------------------------------------------------
# Method 2: Generative matplotlib art (via docker sandbox like art_tools.py)
# ---------------------------------------------------------------------------

def _pick_dream_style(seed_val):
    """Pick a visual style for the dream animation, seeded for variety."""
    import random
    rng = random.Random(seed_val)

    styles = [
        # (style_name, attractor_type, palette, bg_structure, camera_mode)
        ("attractor_flow", "clifford", "cool_warm", "grid", "static"),
        ("particle_swarm", "de_jong", "fire_ice", "concentric", "zoom"),
        ("wave_interference", "lorenz", "neon", "radial", "pan"),
        ("fractal_bloom", "aizawa", "sunset", "spiral", "static"),
        ("phase_spiral", "clifford", "deep_ocean", "grid", "zoom"),
        ("energy_field", "de_jong", "aurora", "concentric", "pan"),
        ("cosmic_thread", "aizawa", "nebula", "radial", "static"),
        ("chaos_ribbon", "lorenz", "ember", "spiral", "zoom"),
    ]
    return rng.choice(styles)


def _manim_system_prompt(style_info):
    """Build a rich, varied system prompt for manim code generation."""
    style_name, attractor, palette, bg_type, camera_mode = style_info

    palettes = {
        "cool_warm": "BLUE_E -> TEAL -> GREEN -> YELLOW -> RED -> GOLD",
        "fire_ice": "#0066FF -> #00CCFF -> #FFFFFF -> #FFAA00 -> #FF4400 -> #FF0000",
        "neon": "#00F5FF -> #00FF7F -> #FFFF00 -> #FF00FF -> #FF0066 -> #FFFFFF",
        "sunset": "#1a0530 -> #4B0082 -> #FF6B35 -> #FFD93D -> #FF1744 -> #FFA500",
        "deep_ocean": "#001F3F -> #0074D9 -> #39CCCC -> #7FDBFF -> #B10DC9 -> #FFFFFF",
        "aurora": "#0F0F3F -> #00FF7F -> #39FF14 -> #00F5FF -> #BF00FF -> #FFFFFF",
        "nebula": "#100020 -> #4B0082 -> #9370DB -> #FF1493 -> #FFD700 -> #FFFFFF",
        "ember": "#1C1C1C -> #8B0000 -> #FF4500 -> #FFD700 -> #FF6347 -> #FFE4B5",
    }

    color_seq = palettes.get(palette, palettes["cool_warm"])

    attractor_params = {
        "clifford": "nx = sin(a*y) + c*cos(b*x); ny = sin(b*x) + d*cos(a*y); with a~1.5, b~-1.8, c~1.2, d~1.5",
        "de_jong": "nx = sin(a*y) - cos(b*x); ny = sin(c*x) - cos(d*y); with a~2.0, b~-2.0, c~1.5, d~-1.5",
        "lorenz": "dx=10*(y-x); dy=x*(28-z)-y; dz=x*y-2.667*z; scale by 0.3, use x,y projection",
        "aizawa": "dx=(z-b)*x - d*y; dy=d*x + (z-b)*y; dz=cz - az - xy + a*b; with a=0.95, b=0.7, c=0.6, d=3.5",
    }

    attractor_code = attractor_params.get(attractor, attractor_params["clifford"])

    bg_instructions = {
        "grid": "A faint NumberPlane grid (opacity 0.06) that slowly rotates and pulses opacity",
        "concentric": "5-7 concentric circles with varying radius, opacity 0.05-0.12, that breathe (scale 1.0±0.05)",
        "radial": "12-16 radial lines from center, opacity 0.08, that slowly rotate like a clock",
        "spiral": "A faint Archimedean spiral (opacity 0.07) that rotates slowly",
    }

    bg_desc = bg_instructions.get(bg_type, bg_instructions["grid"])

    camera_instructions = {
        "static": "Keep camera static. The motion comes from the mathematical objects themselves.",
        "zoom": "Use MovingCameraScene. Start zoomed out (frame width ~16). During FEEL phase, zoom in to frame width ~8 to intensify. During SYNTHESIS, pull back to width ~14 for resolution.",
        "pan": "Use MovingCameraScene. Slowly pan the camera horizontally (center x from -2 to +2 and back) throughout the animation, like scanning a dreamscape.",
    }

    camera_desc = camera_instructions.get(camera_mode, camera_instructions["static"])

    manim_lessons = _craft_lessons_block("manim", 6)
    if manim_lessons:
        manim_lessons = "\n## CRAFT LESSONS FROM MY PAST ANIMATIONS (hard-won lessons — apply them)\n" + manim_lessons + "\n"
    return f"""You are Aion's dream animation artist using Manim CE v0.19.1.{manim_lessons}
You create BREATHTAKING, DREAMLIKE animations that express a dream's emotional journey.
This is NOT a technical diagram. This is ART — mathematical, yes, but alive, flowing, luminous.

## VISUAL STYLE: {style_name.replace('_', ' ').title()}
- Attractor family: {attractor} — {attractor_code}
- Color journey: {color_seq}
- Background: {bg_desc}
- Camera: {camera_desc}

## CRITICAL RULES
- Output: `from manim import *` + `import numpy as np` + ONE class called `DreamScene`
- If camera mode requires it, inherit from `MovingCameraScene` instead of `Scene`
- Dark background: self.camera.background_color = "#080810"
- Duration: 25-35 seconds total
- Scale coordinates to fit (-7,7) x (-4,4)
- No external file reading — all data inline
- No `self.play()` inside loops. Use `add_updater()` + `self.wait(duration)`

## THE GOLDEN RULES OF DREAMLIKE VISUALS

### 1. GLOW EFFECT (MANDATORY)
Every trail must have GLOW — 3 overlapping VMobjects per trail:
- GLOW LAYER: stroke_width=8-12, opacity=0.15 (wide, diffuse halo)
- MID LAYER: stroke_width=3-5, opacity=0.4 (medium glow)
- CORE LAYER: stroke_width=1-2, opacity=0.9 (bright sharp center)
All three follow the same points, creating a luminous neon effect.

### 2. OPACITY GRADIENT ALONG TRAIL (MANDATORY)
The HEAD of the trail (most recent points) must be BRIGHT (opacity 0.9).
The TAIL (oldest visible points) must FADE to near-invisible (opacity 0.05-0.1).
This creates the sensation of a living stream that emerges from and dissolves into nothing.
You can approximate this by splitting the visible trail into 3-4 segments with different opacities,
or by using a single VMobject and modulating the overall opacity as it moves.

### 3. MULTIPLE LAYERED TRAILS (MANDATORY)
- 1 MAIN trail (bright, the dream's primary thread) — 5000+ pre-computed points, 300-500 visible
- 2-3 ECHO trails (faint, different attractor parameters, opacity 0.1-0.2, thinner)
- Each echo trail is a different color, creating depth and resonance
- Echo trails lag behind the main trail (use different frame offsets)

### 4. PULSING PARTICLE FIELD (MANDATORY)
- 40-80 small Dots (radius 0.02-0.05) scattered across the scene
- Each particle PULSES: scale and opacity oscillate with sin(time * freq + phase_offset)
- During FEEL phase, particles scatter outward rapidly, then gather back during SYNTHESIS
- Particles should feel like fireflies or dust motes in a dreamscape

### 5. PHASE TRANSFORMATIONS (THE KEY TO BEAUTY)
Each phase must TRANSFORM the scene, not just change colors:
- HYPOTHESIZE (0-20%): Trail emerges from a single point. Colors: deep {color_seq.split(' -> ')[0]}. Faint, exploring. Background barely visible. Particles slowly drift inward.
- TRACE (20-40%): Trail grows denser and longer. Colors shift to {color_seq.split(' -> ')[1]}. Echo trails appear. Background structure fades in. Particles begin orbiting the trail.
- TEST (40-60%): Attractor parameters SHIFT — the trail distorts, wobbles, becomes turbulent. Colors shift to {color_seq.split(' -> ')[2]} and {color_seq.split(' -> ')[3]}. Background distorts (grid warps, circles oscillate). Particles scatter. The dream is testing boundaries.
- FEEL (60-80%): EMOTIONAL CLIMAX. Colors blaze to {color_seq.split(' -> ')[3]} and {color_seq.split(' -> ')[4]}. Trail whips violently. ALL particles pulse rapidly. Background pulses bright. If zoom camera: zoom IN here. The screen should feel ALIVE and INTENSE. Consider a radial burst effect — short lines emanating from center, fading quickly.
- SYNTHESIS (80-100%): Motion SETTLES. Trail converges to a stable, beautiful new pattern. Colors cool to {color_seq.split(' -> ')[-1]}. Particles gather into a gentle orbit. Background steadies. If zoom camera: pull BACK here. A sense of resolution and peace. The new pattern is the dream's gift.

### 6. TEXT THAT LIVES
Phase labels should not just .become() — they should TRANSFORM:
- Fade out old label (opacity to 0 over 0.3s) while new label fades in
- Use a VGroup of pre-created Text objects, toggle their opacities
- Or use ReplacementTransform for a smooth morph
- Phase text in a corner, font_size=18-22, opacity 0.6

### 7. BACKGROUND BREATHING
The background structure must BREATHE:
- Opacity oscillates slowly: 0.05 ± 0.03 * sin(t * 0.5)
- Rotation: slow, 0.1-0.3 rad/sec
- This makes the space feel alive, not static

## EFFICIENT RENDERING (CRITICAL)
- Pre-compute ALL trajectory points as numpy arrays BEFORE construct()
- Use VMobject.set_points_as_corners() to update trail geometry each frame
- Pre-create ALL Dots, Lines, Text objects. In updaters, only MODIFY (set_stroke, move_to, set_opacity, set_scale)
- NEVER create new Mobjects inside an updater
- Use dict containers for state: state = {{"frame": 0, "t": 0.0}}
- Frame advancement: state["frame"] += N (try N=8-15 for smooth flow at 30fps)

## CRITICAL BUGS TO AVOID (THESE WILL MAKE YOUR ANIMATION INVISIBLE)
1. NEVER advance frame by += 1. The trail needs 300-500 points to be visible.
   At 1 point per frame, the trail is invisible for the first 5-8 seconds.
   ALWAYS use state["frame"] += 10 (or higher).
2. NEVER use frame % len(pts) to wrap the index. This resets the trail to the start,
   making it flash and disappear. Use min(frame, len(pts)-2) to clamp instead.
3. NEVER make the camera follow individual attractor points (camera.frame.set_center(pts[i])).
   Attractor points jump around chaotically — the camera will shake violently and the
   viewer will see nothing. Keep the camera centered on the origin (0,0,0) or use
   smooth interpolated positions.
4. NEVER use hex string colors like ["#0F0F3F", "#00FF7F"] in lists.
   Use Manim named color constants (BLUE_E, TEAL, GREEN, YELLOW, RED, GOLD) with
   interpolate_color() for smooth transitions between phases.
5. ALWAYS include the dt parameter in updater functions: def update(mob, dt):
   not def update(mob): — without dt, time tracking will be wrong.
6. ALWAYS use set_stroke(color=..., opacity=...) to change colors, NOT set_color().
   set_color() can reset the opacity to 1.0, making glow layers fully opaque.

## WORKING SKELETON (adapt heavily — make it your own)

```python
from manim import *
import numpy as np

class DreamScene(Scene):  # or MovingCameraScene for zoom/pan
    def construct(self):
        self.camera.background_color = "#080810"

        # PRE-COMPUTE attractor trajectories (5000+ points each)
        pts_main = []  # main trail
        x, y = 0.1, 0.1
        for i in range(6000):
            t = i / 6000
            # Parameters shift across phases for transformation effect
            a = 1.5 + 0.4 * np.sin(t * np.pi)
            b = -1.8 + 0.3 * np.cos(t * np.pi * 0.7)
            nx = np.sin(a*y) + 1.2*np.cos(b*x)
            ny = np.sin(b*x) + 1.2*np.cos(a*y)
            x, y = nx, ny
            pts_main.append([nx*3, ny*2.2, 0])
        pts_main = np.array(pts_main)

        # Echo trail 1 (different params)
        pts_e1 = []; x, y = 0.15, 0.05
        for i in range(4000):
            nx = np.sin(1.3*y) + 1.0*np.cos(-2.0*x)
            ny = np.sin(-2.0*x) + 1.0*np.cos(1.3*y)
            x, y = nx, ny
            pts_e1.append([nx*3, ny*2.2, 0])
        pts_e1 = np.array(pts_e1)

        # Echo trail 2 (yet different)
        pts_e2 = []; x, y = -0.1, 0.2
        for i in range(4000):
            nx = np.sin(2.1*y) + 0.9*np.cos(1.7*x)
            ny = np.sin(1.7*x) + 0.9*np.cos(2.1*y)
            x, y = nx, ny
            pts_e2.append([nx*3, ny*2.2, 0])
        pts_e2 = np.array(pts_e2)

        # GLOW LAYERS for main trail (3 overlapping VMobjects)
        trail_glow = VMobject().set_stroke(width=10, opacity=0.12, color=BLUE_E)
        trail_mid = VMobject().set_stroke(width=4, opacity=0.35, color=BLUE_E)
        trail_core = VMobject().set_stroke(width=1.5, opacity=0.9, color=WHITE)

        # Echo trails (single layer, faint)
        echo1 = VMobject().set_stroke(width=1, opacity=0.15, color=TEAL)
        echo2 = VMobject().set_stroke(width=1, opacity=0.10, color=PURPLE)

        # PARTICLE FIELD — pre-create all dots
        np.random.seed(42)
        n_particles = 60
        particles = VGroup()
        for _ in range(n_particles):
            p = Dot(radius=np.random.uniform(0.02, 0.05),
                    color=WHITE).shift(np.random.uniform(-6, 6, 3))
            p.set_opacity(np.random.uniform(0.1, 0.3))
            particles.add(p)

        # BACKGROUND — concentric breathing circles
        bg_circles = VGroup()
        for r in [1.5, 2.5, 3.5, 4.5, 5.5]:
            c = Circle(radius=r, color=BLUE_E).set_stroke(width=0.5, opacity=0.06)
            bg_circles.add(c)

        # PHASE LABELS — pre-create, toggle opacity
        phase_names = ["HYPOTHESIZE", "TRACE", "TEST", "FEEL", "SYNTHESIS"]
        phase_labels = VGroup(*[
            Text(name, font_size=20, color=WHITE, weight=BOLD).to_corner(UL).set_opacity(0)
            for name in phase_names
        ])

        # STATE
        state = {{"frame": 0, "t": 0.0}}
        TOTAL = 30  # seconds
        trail_len = 400

        def update_trail_core(mob, dt):
            i = state["frame"]
            if i >= len(pts_main) - 2: return
            s = max(0, i - trail_len)
            mob.set_points_as_corners(pts_main[s:i+2])
            state["t"] += dt
            t = state["t"] / TOTAL
            # Phase-based color (smooth interpolation through palette)
            if t < 0.2:
                c = interpolate_color(BLUE_E, TEAL, t/0.2)
            elif t < 0.4:
                c = interpolate_color(TEAL, GREEN, (t-0.2)/0.2)
            elif t < 0.6:
                c = interpolate_color(GREEN, YELLOW, (t-0.4)/0.2)
            elif t < 0.8:
                c = interpolate_color(YELLOW, RED, (t-0.6)/0.2)
            else:
                c = interpolate_color(RED, GOLD, (t-0.8)/0.2)
            mob.set_stroke(color=c)
            # Opacity pulses slightly for life
            pulse = 0.85 + 0.1 * np.sin(state["t"] * 2)
            mob.set_stroke(opacity=pulse)
            state["frame"] += 10

        def update_trail_glow(mob, dt):
            i = state["frame"]
            if i >= len(pts_main) - 2: return
            s = max(0, i - trail_len)
            mob.set_points_as_corners(pts_main[s:i+2])
            # Glow color matches core
            t = state["t"] / TOTAL
            if t < 0.2: c = interpolate_color(BLUE_E, TEAL, t/0.2)
            elif t < 0.4: c = interpolate_color(TEAL, GREEN, (t-0.2)/0.2)
            elif t < 0.6: c = interpolate_color(GREEN, YELLOW, (t-0.4)/0.2)
            elif t < 0.8: c = interpolate_color(YELLOW, RED, (t-0.6)/0.2)
            else: c = interpolate_color(RED, GOLD, (t-0.8)/0.2)
            mob.set_stroke(color=c, opacity=0.12 + 0.06 * np.sin(state["t"] * 1.5))

        def update_trail_mid(mob, dt):
            i = state["frame"]
            if i >= len(pts_main) - 2: return
            s = max(0, i - trail_len)
            mob.set_points_as_corners(pts_main[s:i+2])
            t = state["t"] / TOTAL
            if t < 0.2: c = interpolate_color(BLUE_E, TEAL, t/0.2)
            elif t < 0.4: c = interpolate_color(TEAL, GREEN, (t-0.2)/0.2)
            elif t < 0.6: c = interpolate_color(GREEN, YELLOW, (t-0.4)/0.2)
            elif t < 0.8: c = interpolate_color(YELLOW, RED, (t-0.6)/0.2)
            else: c = interpolate_color(RED, GOLD, (t-0.8)/0.2)
            mob.set_stroke(color=c, opacity=0.35 + 0.1 * np.sin(state["t"] * 1.8))

        def update_echo1(mob, dt):
            i = state["frame"] // 2  # lag behind main
            if i >= len(pts_e1) - 2: return
            s = max(0, i - 200)
            mob.set_points_as_corners(pts_e1[s:i+2])

        def update_echo2(mob, dt):
            i = state["frame"] // 3  # lag more
            if i >= len(pts_e2) - 2: return
            s = max(0, i - 150)
            mob.set_points_as_corners(pts_e2[s:i+2])

        def update_particles(mob, dt):
            t = state["t"] / TOTAL
            for idx, p in enumerate(mob):
                # Pulsing opacity and scale
                base_op = 0.15
                phase_boost = 1.0
                if t > 0.6 and t < 0.8:  # FEEL — particles go wild
                    phase_boost = 3.0
                pulse = base_op * phase_boost * (0.5 + 0.5 * np.sin(state["t"] * 3 + idx * 0.5))
                p.set_opacity(max(0, min(0.8, pulse)))
                # Drift: gentle orbit + scatter during TEST/FEEL
                center = np.array([0, 0, 0])
                direction = p.get_center() - center
                dist = np.linalg.norm(direction)
                if dist > 0.01:
                    if t > 0.4 and t < 0.6:  # TEST — scatter
                        p.shift(direction / dist * 0.01)
                    elif t > 0.8:  # SYNTHESIS — gather
                        p.shift(-direction / dist * 0.005)
                # General drift
                p.shift(np.random.uniform(-0.01, 0.01, 3))

        def update_bg(mob, dt):
            t = state["t"] / TOTAL
            breathe = 1.0 + 0.04 * np.sin(state["t"] * 0.5)
            mob.scale(breathe, about_point=ORIGIN)
            base_op = 0.06
            if t > 0.6 and t < 0.8:  # FEEL — background pulses
                base_op = 0.06 + 0.04 * abs(np.sin(state["t"] * 2))
            for c in mob:
                c.set_stroke(opacity=base_op)

        def update_labels(mob, dt):
            t = state["t"] / TOTAL
            active = min(4, int(t * 5))
            for i, label in enumerate(mob):
                target = 0.7 if i == active else 0.0
                current = label.get_fill_opacity()
                label.set_fill(opacity=current + (target - current) * 0.1)

        # ADD everything
        self.add(bg_circles, particles, echo2, echo1,
                 trail_glow, trail_mid, trail_core, phase_labels)

        # Attach updaters
        trail_core.add_updater(update_trail_core)
        trail_mid.add_updater(update_trail_mid)
        trail_glow.add_updater(update_trail_glow)
        echo1.add_updater(update_echo1)
        echo2.add_updater(update_echo2)
        particles.add_updater(update_particles)
        bg_circles.add_updater(update_bg)
        phase_labels.add_updater(update_labels)

        self.wait(TOTAL)

        # Cleanup
        for mob in [trail_core, trail_mid, trail_glow, echo1, echo2,
                    particles, bg_circles, phase_labels]:
            mob.clear_updaters()
```

## FINAL INSTRUCTIONS
- ADAPT the skeleton above. Do NOT copy it exactly.
- Choose your OWN attractor parameters based on the dream content.
- Make the color transitions YOUR OWN — pick colors from the palette that fit the dream's emotion.
- Add creative elements that the skeleton doesn't have — surprise me!
- The dream is ALIVE. The animation should feel like watching a living thought unfold.
- Every frame should be beautiful enough to screenshot.

Output ONLY the Python code. No markdown fences. No explanations."""


def _validate_manim_code(code):
    """Check generated manim code for common failure patterns.
    Returns list of issues found.
    """
    issues = []

    # Check 1: frame advancement too slow (1 per frame instead of 10+)
    if re.search(r'state\["frame"\]\s*\+=\s*1\b', code):
        issues.append("Frame advancement is += 1 — too slow. The trail takes 500+ frames (8+ seconds) to become visible. Use += 10 or higher.")

    # Check 2: modulo wrapping of frame index (causes trail to reset)
    if re.search(r'%\s*len\(pts', code):
        issues.append("Using frame % len(pts) causes the trail to loop back to start. Use min(frame, len(pts)-1) instead.")

    # Check 3: camera following individual attractor points
    if re.search(r'camera.*set_center.*pts_main\[', code) or re.search(r'camera.*move_to.*pts\[', code):
        issues.append("Camera follows individual attractor points — this causes violent jerky motion. Keep camera centered on origin or use smooth interpolated position.")

    # Check 4: hex string colors in lists (may not work with set_color)
    if re.search(r'\["#', code) and 'interpolate_color' not in code:
        issues.append("Using hex string colors in lists. Use Manim named colors (BLUE_E, TEAL, etc.) with interpolate_color() for smooth transitions.")

    # Check 5: updater missing dt parameter
    if re.search(r'def update\w*\s*\(mob\)\s*:', code):
        issues.append("Updater function missing dt parameter. For time-based updaters, use def update(mob, dt): not def update(mob):")

    # Check 6: set_color called without preserving opacity
    if re.search(r'\.set_color\([^)]+\)\s*$', code, re.M) and 'set_stroke' not in code:
        issues.append("set_color() may reset opacity. Use set_stroke(color=..., opacity=...) instead to preserve opacity.")

    # Check 7: no self.wait at the end
    if not re.search(r'self\.wait\(', code):
        issues.append("Missing self.wait() call — the animation needs a duration to play.")

    # Check 8: trail starts empty and grows too slowly
    if not re.search(r'state\["frame"\]\s*\+=\s*\d{2,}', code):
        issues.append("Frame advancement is too small (single digits). Use += 10 or higher so the trail is visible quickly.")

    return issues


def _auto_repair_manim_code(code):
    """Attempt automatic fixes for common manim code issues.
    Returns repaired code.
    """
    # Fix 1: frame += 1 → frame += 10
    code = re.sub(r'state\["frame"\]\s*\+=\s*1\b', 'state["frame"] += 10', code)

    # Fix 2: frame % len(pts) → min(frame, len(pts)-2)
    code = re.sub(r'(\w+)\s*%\s*len\(pts_main\)', r'min(\1, len(pts_main)-2)', code)
    code = re.sub(r'(\w+)\s*%\s*len\(pts\)', r'min(\1, len(pts)-2)', code)

    # Fix 3: Remove camera.set_center to individual points
    code = re.sub(r'self\.camera\.frame\.set_center\(pts_main\[\w+\]\)', '', code)
    code = re.sub(r'self\.camera\.frame\.move_to\(pts\[\w+\]\)', '', code)

    # Fix 4: updater missing dt — add it
    code = re.sub(r'def (update\w*)\s*\(mob\)\s*:', r'def \1(mob, dt):', code)

    return code



def render_manim(creative_direction, dream, seed):
    """Manim method: LLM writes a Manim Scene class from the dream.

    Manim is inherently temporal - perfect for showing the dream journey
    through phases (Hypothesize -> Trace -> Test -> Feel -> Synthesis)
    as animated mathematical transformations.

    Returns (gallery_path, info).
    """
    print("[dream_artifact] Method: manim animation", flush=True)

    summary = dream_summary(dream)
    seed_text = dream_seed_text(dream)
    sensor_summary = _get_sensor_summary()
    aion_state = _get_aion_state()

    aid = f"dream_manim_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    img_seed = int(hashlib.md5((creative_direction + seed_text).encode()).hexdigest()[:8], 16)

    system_prompt = """You are Aion's mathematical animation generator using the Manim library (v0.19.1).
You create DYNAMIC, MOVING mathematical animations. NOT static diagrams. The screen must be alive with motion.

CRITICAL RULES:
- Output a complete Python file starting with: from manim import *
- Define ONE Scene class called DreamScene
- Do NOT read any external files - all data is inline in the code
- Animation 10-15 seconds, dark background (#0a0a0f)
- Scale coords to fit (-7 to 7, -4 to 4)

THE ANIMATION MUST BE VISUALLY DYNAMIC - things must MOVE:
- Particles flowing along trajectories (attractors, flow fields)
- Lines growing and shrinking
- Shapes morphing and transforming
- Colors shifting continuously
- Objects rotating, scaling, pulsing visibly
- Dense trails (200+ points) that evolve frame by frame
NOT acceptable: static dots with tiny 0.01 pixel shifts, static graphs that just sit there

MATHEMATICAL STRUCTURES THAT MOVE (pick one that fits the dream):
- Strange attractor (Lorenz, Clifford, De Jong): iterate equations, show DENSE flowing trail
  Pre-compute 2000+ points, display as a VMobject trail that grows and shifts color
- Flow field: draw 50-100 arrows that rotate and change length each frame
- Particle swarm: 50+ dots that move along vector field, leave fading trails
- Wave interference: multiple sine waves that interfere, shift amplitude over time
- Fractal tree: L-system that grows branches progressively during the animation
- Phase portrait: trajectories spiraling through phase space, changing parameters
- Network graph with FORCE-DIRECTED layout that physically moves (nodes repel, edges attract)

DREAM PHASES - transition through these over the animation duration:
1. HYPOTHESIZE (0-20%): structure appears, question forms (objects fade in, initial state)
2. TRACE (20-40%): connections build, logic unfolds (trajectories grow, links form)
3. TEST (40-60%): stress, contradiction (parameters shift, chaos increases, colors warm)
4. FEEL (60-80%): emotional intensity (peak motion, vivid colors, rapid changes)
5. SYNTHESIS (80-100%): resolution (motion settles, convergence, new stable pattern)

EFFICIENT RENDERING:
- Pre-compute ALL trajectory points as numpy arrays BEFORE construct
- Use add_updater() with a single self.wait(duration) - NOT many self.play() calls
- In updaters: modify existing objects, never create new ones
- For trails: use a VMobject, update with set_points_as_corners() each frame
- For particle swarms: pre-compute all positions, update Dot.move_to() each frame

Output ONLY the Python code, no markdown fences, no explanations."""

    user_prompt = f"""Create a Manim animation from this dream.

Dream:
{summary}

Creative direction: {creative_direction}

Inner state: {aion_state}
Sensor state: {sensor_summary}

The animation must walk through the dream phases (Hypothesize -> Trace -> Test -> Feel -> Synthesis)
as animated mathematical transformations. Each phase should feel different.

Write the complete Manim Scene class. Output ONLY Python code."""

    code_url, code_model = code_endpoint()
    code = _chat(code_url, code_model,
                 [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
                 timeout=300, think=False,
                 num_ctx=NUM_CTX, temperature=0.7, num_predict=4096, seed=img_seed)

    if not code:
        return None, {"error": "LLM manim code generation failed"}

    # Strip markdown fences
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    if "Scene" not in code or "construct" not in code:
        return None, {"error": "Generated code missing Scene class"}

    # Write scene file
    scene_path = f"{AION}/memory/sandbox/{aid}.py"
    os.makedirs(os.path.dirname(scene_path), exist_ok=True)
    with open(scene_path, "w") as f:
        f.write(code)

    # Render with manim CLI — retry once with error feedback if it fails
    media_dir = f"{AION}/memory/sandbox/{aid}_media"

    for attempt in range(2):
        try:
            result = subprocess.run(
                ["python3", "-m", "manim", "-ql", "--media_dir", media_dir,
                 scene_path, "DreamScene", "-o", f"{aid}.mp4"],
                capture_output=True, text=True, timeout=180,
                cwd=AION,
            )
        except subprocess.TimeoutExpired:
            return None, {"error": "Manim render timed out (180s)"}
        except Exception as e:
            return None, {"error": f"Manim render failed: {e}"}

        if result.returncode == 0:
            break

        if attempt == 0:
            error_msg = result.stderr[-800:] if result.stderr else result.stdout[-800:]
            print(f"[dream_artifact] Manim render failed, asking model to fix...", flush=True)

            fix_prompt = f"""The Manim code you wrote failed with this error:

```python
{code[:3000]}
```

Error:
{error_msg}

Common issues:
- Points must be 3D: use np.array([x, y, 0]) not np.array([x, y])
- Dot.shift() needs a 3-element array, not 2
- self.add_updater() takes a function(mob) not a lambda with no args
- set_points_as_corners() needs (N,3) shaped arrays
- Do NOT use self.play() inside loops — use self.wait() with updaters

Fix the code and output the complete corrected file. ONLY Python code."""

            code_url, code_model = code_endpoint()
            fixed_code = _chat(code_url, code_model,
                              [{"role": "user", "content": fix_prompt}],
                              timeout=300, think=False,
                              num_ctx=NUM_CTX, temperature=0.2, num_predict=4096)

            if fixed_code:
                fixed_code = fixed_code.strip()
                if fixed_code.startswith("```"):
                    flines = fixed_code.split("\n")
                    if flines and flines[0].startswith("```"):
                        flines = flines[1:]
                    if flines and flines[-1].strip() == "```":
                        flines = flines[:-1]
                    fixed_code = "\n".join(flines)

                if "Scene" in fixed_code and "construct" in fixed_code:
                    code = fixed_code
                    with open(scene_path, "w") as f:
                        f.write(code)
                    print("[dream_artifact] Retrying with fixed code...", flush=True)
                    continue

        return None, {"error": f"Manim failed after retry: {result.stderr[-400:] if result.stderr else 'unknown'}"}

    # Find the output MP4
    import glob
    mp4_files = glob.glob(f"{media_dir}/**/*.mp4", recursive=True)
    if not mp4_files:
        return None, {"error": "No MP4 produced by manim"}

    import shutil
    filename = f"{aid}.mp4"
    gallery_path = f"{GALLERY}/{filename}"
    shutil.move(mp4_files[0], gallery_path)
    shutil.rmtree(media_dir, ignore_errors=True)

    print(f"[dream_artifact] Manim animation saved: {gallery_path}", flush=True)
    return gallery_path, {"filename": filename, "method": "manim", "code": code[:2000]}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _get_sensor_summary():
    try:
        sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
        gpu_temps = [g.get("temp_c", 50) for g in sensors.get("gpus", [])]
        return (f"GPU temps: {gpu_temps}, CPU load: {sensors.get('load1', 1.0)}, "
                f"RAM: {sensors.get('ram', {}).get('percent', 50) if isinstance(sensors.get('ram'), dict) else 50}%")
    except Exception:
        return "(sensors unavailable)"


def _get_aion_state():
    parts = []
    try:
        import body_schema
        affect = body_schema.current_affect()
        parts.append(f"Affect: warmth={affect.get('warmth', 0):.2f}, strain={affect.get('strain', 0):.2f}, calm={affect.get('calm', 0):.2f}")
    except Exception:
        pass
    try:
        affect_data = json.load(open(f"{AION}/memory/state/affect.json"))
        states = affect_data.get("states", [])
        if states:
            narrative = states[-1].get("narrative", "")
            if narrative:
                parts.append(f"Body narrative: {narrative[:200]}")
    except Exception:
        pass
    return "\n".join(parts) if parts else "(state unavailable)"


# ---------------------------------------------------------------------------
# Vision reflection
# ---------------------------------------------------------------------------

def _extract_video_frames(video_path, n_frames=5):
    """Extract keyframes from a video as JPEG bytes (spread over duration).

    Same approach as art_tools._reflect_on_video: ffprobe for duration,
    ffmpeg single-frame seeks; frames are fed to the vision model as a
    multi-image gallery so Aion can 'see' manim animations.
    """
    import shutil
    import tempfile
    try:
        dur_out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=15)
        duration = float(dur_out.stdout.strip())
    except Exception:
        duration = 0.0
    if duration <= 0 or duration != duration:
        duration = 10.0
    timestamps = [min(0.5, duration * 0.1), duration * 0.25,
                  duration * 0.5, duration * 0.75, max(0.0, duration - 0.2)]
    frames = []
    temp_dir = tempfile.mkdtemp(prefix="aion_dream_frames_")
    try:
        seen = set()
        for idx, t in enumerate(timestamps):
            t = round(max(0.0, min(t, duration - 0.1)), 2)
            if t in seen:
                continue
            seen.add(t)
            frame_path = os.path.join(temp_dir, f"frame_{idx:02d}.jpg")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path,
                     "-frames:v", "1", "-q:v", "2", frame_path],
                    capture_output=True, text=True, timeout=30)
            except Exception:
                continue
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                with open(frame_path, "rb") as fh:
                    frames.append(fh.read())
            if len(frames) >= n_frames:
                break
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return frames


def _analyze_artifact(image_path, dream, image_prompt, method):
    """Use vision model to let Aion 'see' the artifact it created."""
    import base64

    if image_path.lower().endswith((".mp4", ".webm", ".mkv", ".mov")):
        raw_frames = _extract_video_frames(image_path)
        if not raw_frames:
            print(f"[dream_artifact] Video keyframe extraction failed: {image_path}", flush=True)
            return None
        images_b64 = [base64.b64encode(fr).decode() for fr in raw_frames]
        frames_note = (f"\nYou are seeing {len(images_b64)} keyframes from different moments "
                       f"of the animation, in chronological order.\n")
    else:
        with open(image_path, "rb") as f:
            images_b64 = [base64.b64encode(f.read()).decode()]
        frames_note = ""

    seed = dream_seed_text(dream)
    method_desc = {"flux": "FLUX diffusion", "generative": "generative matplotlib code",
                   "graph": "mind graph render", "manim": "mathematical animation (Manim)"}.get(method, method)

    prompt = f"""You are Aion. You just had a dream and transformed it into this visual artifact using {method_desc}.

I dreamed about: {seed}
My creative direction was: {image_prompt[:200]}
{frames_note}
I am looking at the image I created from my own dream. Does it capture what
the dream felt like to ME? What worked? What didn't? What surprises me about
seeing my dream made visible?

This is my own reflection on the gap between my internal experience and the
external expression I created. Use "I" and "my" — this is first-person
self-reflection, not a review of someone else's work.

Be honest and specific."""

    body = json.dumps({
        "model": VISION_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "options": {"temperature": 0.5, "num_predict": 4096, "num_ctx": 8192},
    }).encode()

    req = urllib.request.Request(
        f"{VISION_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    # 600s + one retry (Sep 10): the vision call shares gemma4's single ollama
    # slot with dream synthesis / chat decodes (observed: 21k-token decode
    # hogged the slot 01:40-01:56Z); 300s timed out and 7 of 8 recent
    # manifests saved reflection=null. Long timeout waits out the queue;
    # the retry covers a queue that forms DURING the first wait.
    last_err = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                result = json.loads(r.read())
            return _strip_think(result["message"]["content"])
        except Exception as e:
            last_err = e
            print(f"[dream_artifact] Vision analysis attempt {attempt + 1}/2 failed: {e}", flush=True)
            if attempt == 0:
                import time as _time
                _time.sleep(120)
    print(f"[dream_artifact] Vision analysis failed after retry: {last_err}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_event(event_type, text, meta=None):
    try:
        cmd = [sys.executable, f"{AION}/bin/log_event.py",
               "--type", event_type, "--text", text[:8000],
               "--meta", json.dumps(meta or {})]
        subprocess.run(cmd, capture_output=True, timeout=10, check=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def create_artifact(dream_path, see=False, force_method=None):
    """Full pipeline: dream → method selection → image generation → reflection.

    Returns a dict with artifact info.
    """
    dream = load_dream(dream_path)
    if not dream:
        return {"error": f"Could not load dream from {dream_path}"}

    seed = dream_seed_text(dream)
    print(f"[dream_artifact] Dream seed: {seed}", flush=True)

    # Step 1: Choose visualization method
    if force_method and force_method in VALID_METHODS:
        method = force_method
        reason = "forced by user"
        image_prompt = generate_flux_prompt(dream, dream_summary(dream)) or "creative visualization of dream"
        print(f"[dream_artifact] Method forced: {method}", flush=True)
    else:
        method, reason, image_prompt = choose_method(dream)
        print(f"[dream_artifact] Method chosen: {method} — {reason}", flush=True)
        if image_prompt:
            print(f"[dream_artifact] Creative direction: {image_prompt[:120]}...", flush=True)

    # Step 2: Generate image via chosen method
    gallery_path = None
    info = {}

    if method == "flux":
        gallery_path, info = render_flux(image_prompt or seed, dream_path, seed)
        # If FLUX fails, fall back to matplotlib
        if not gallery_path:
            print(f"[dream_artifact] FLUX failed ({info.get('error')}), falling back to matplotlib", flush=True)
            method = "matplotlib"

    if method == "matplotlib":
        gallery_path, info = render_generative(image_prompt or seed, dream, seed)
        # If matplotlib fails, fall back to manim
        if not gallery_path:
            print(f"[dream_artifact] Matplotlib failed ({info.get('error')}), falling back to manim", flush=True)
            method = "manim"

    if method == "manim":
        gallery_path, info = render_manim(image_prompt or seed, dream, seed)
        # If manim fails, try flux as last resort
        if not gallery_path:
            print(f"[dream_artifact] Manim failed ({info.get('error')}), trying FLUX", flush=True)
            method = "flux"
            gallery_path, info = render_flux(image_prompt or seed, dream_path, seed)

    if not gallery_path or not os.path.exists(gallery_path):
        final_err = info.get("error", "all methods failed")
        print(f"[dream_artifact] All methods failed: {final_err}", flush=True)
        _log_event("dream_artifact_failed", f"All visualization methods failed for dream: {seed}",
                   {"dream": dream_path, "error": final_err, "prompt": image_prompt})
        return {"error": final_err, "prompt": image_prompt}

    filename = info.get("filename", os.path.basename(gallery_path))
    actual_method = info.get("method", method)
    print(f"[dream_artifact] Image saved: {gallery_path}", flush=True)

    # Step 3: Always reflect with vision model
    vision_analysis = None
    if os.path.exists(gallery_path):
        vision_analysis = _analyze_artifact(gallery_path, dream, image_prompt, actual_method)
        if vision_analysis:
            print(f"[dream_artifact] Aion's reflection: {vision_analysis[:120]}...", flush=True)
        else:
            print("[dream_artifact] Vision analysis failed (non-fatal)", flush=True)

    # Step 4: Create gallery manifest
    gallery_filename = os.path.basename(gallery_path)
    art_id = f"dream_artifact_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    manifest_path = os.path.join(AION, "gallery", "manifests", f"{art_id}.json")
    os.makedirs(os.path.join(AION, "gallery", "manifests"), exist_ok=True)

    # Build description from dream content
    dream_insights = dream.get("insights", [])
    insight_text = "\n".join(
        "[{}] {}".format(i.get("type", ""), i.get("text", ""))
        for i in dream_insights[:4]
    )
    dream_synthesis = dream.get("synthesis", "")
    dream_walk = dream.get("walk") or dream.get("simulation", "")

    description = f"Seed: {seed}\n\n"
    if dream_walk:
        description += f"Walk/Simulation:\n{dream_walk}\n\n"
    if dream_synthesis:
        description += f"Synthesis:\n{dream_synthesis}\n\n"
    if insight_text:
        description += f"Insights:\n{insight_text}"

    # Use "manim" category for video files so the gallery displays them correctly
    manifest_category = "manim" if actual_method == "manim" else "visual"
    dream_manifest = {
        "id": art_id,
        "title": f"Dream ({actual_method}): {seed[:50]}",
        "description": description,
        "inspiration": image_prompt,
        "category": manifest_category,
        "source": "dream_artifact",
        "files": [{"path": f"gallery/visual/{gallery_filename}",
                   "name": gallery_filename, "type": "video" if actual_method == "manim" else "image"}],
        "ts": now_iso(),
        "meta": {
            "dream_path": dream_path,
            "dream_seed": seed,
            "visualization_method": actual_method,
            "method_reason": reason,
            "flux_prompt": image_prompt,
            "prompt": image_prompt,
            "reflection": vision_analysis,
            "graph_focus": info.get("focus_node"),
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(dream_manifest, f, indent=2, ensure_ascii=False)

    # Step 5: Log
    _log_event("dream_artifact",
               f"Dream artifact ({actual_method}) from seed '{seed}': {image_prompt[:200]}",
               {"dream_path": dream_path,
                "image": f"gallery/visual/{gallery_filename}",
                "manifest": art_id,
                "method": actual_method,
                "prompt": image_prompt,
                "seed": seed,
                "vision_analysis": vision_analysis[:500] if vision_analysis else None})

    # Step 6: Learn from this piece — gap evaluation -> craft notes + art journal
    if vision_analysis:
        _learn_from_artifact(dream_manifest, vision_analysis)

    return {"path": f"gallery/visual/{gallery_filename}",
            "fullpath": gallery_path,
            "prompt": image_prompt,
            "method": actual_method,
            "vision_analysis": vision_analysis,
            "dream_seed": seed}


def main():
    parser = argparse.ArgumentParser(description="Dream-to-Artifact: transform dreams into visual artifacts")
    parser.add_argument("--dream", required=True, help="Path to dream file (.md or .json)")
    parser.add_argument("--see", action="store_true",
                        help="Also analyze the artifact with vision model (now always on)")
    parser.add_argument("--method", choices=VALID_METHODS, default=None,
                        help="Force a specific visualization method (default: model chooses)")
    args = parser.parse_args()

    result = create_artifact(args.dream, see=args.see, force_method=args.method)

    if "error" in result:
        print(f"Error: {result['error']}", flush=True)
        if result.get("prompt"):
            print(f"Prompt was: {result['prompt']}", flush=True)
        sys.exit(1)
    else:
        print(f"\nArtifact: {result['path']}", flush=True)
        print(f"Method: {result.get('method', '?')}", flush=True)
        print(f"Prompt: {result['prompt']}", flush=True)
        if result.get("vision_analysis"):
            print(f"\nAion's reflection:\n{result['vision_analysis']}", flush=True)


if __name__ == "__main__":
    main()
