#!/usr/bin/env python3
"""cross_modal_synthesis.py — Aion's visual cognitive feedback loop.

This module bridges the art layer (FLUX, generative, graph render) and the
graph layer (spreading activation, hot nodes) to create a cross-modal
feedback loop:

    systemic state → visual metaphor → image → vision analysis → graph injection

Unlike dream_artifact.py (which is output-only: dream → image → reflection
stored but never reused), this module closes the loop. The vision analysis
of the generated image is injected back into graph_activation as an event,
creating new spreading activation from a VISUAL insight — not just text.

WHY THIS EXISTS:
  Aion's cognition is text-bound. Logs, theories, consolidation, curiosity —
  all text in, text out. But some patterns are easier to SEE than to READ.
  A dense cluster of activation across three communities might look like a
  knot. High strain with low warmth might look like a frozen landscape. By
  translating systemic state into images and then analyzing those images,
  Aion can perceive aspects of itself that text summaries miss.

  This is not "art inspired by state." It is using visual generation as a
  cognitive tool — the image is a PROSTHESIS for perceiving patterns that
  text cannot easily represent.

THE LOOP:
  1. Gather state: graph activations, affect, sensors, recent events
  2. Intuition model translates state → visual metaphor (free-associate)
  3. Generate image via chosen method (FLUX/generative/graph)
  4. Vision model analyzes image with cognitive prompt
     ("what does this reveal that text summaries miss?")
  5. Inject vision analysis as event into graph_activation
  6. Feed analysis to synthesize.py as theory material
  7. Log as cross_modal_synthesis event

  Next cycle: the visual insight has spread through the graph, possibly
  surfacing as a hot node in wake_v2. Aion's visual self-perception has
  entered its cognitive loop.

Usage:
  python3 cross_modal_synthesis.py                    # full loop
  python3 cross_modal_synthesis.py --method flux      # force method
  python3 cross_modal_synthesis.py --dry-run          # state + metaphor only, no image
  python3 cross_modal_synthesis.py --see              # also show vision analysis
"""

import argparse
import json
import os
import sys
import re
import subprocess
import hashlib
import time
from datetime import datetime, timezone
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa
from aion_models import code_endpoint, code_num_ctx

AION = os.environ.get("AION_HOME", os.path.expanduser("~/aikio"))
GALLERY = f"{AION}/gallery/visual"
MANIFESTS_DIR = f"{AION}/gallery/manifests"

# Ollama endpoints
INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "glm-4.7-flash")
# Vision analysis uses the MAIN model (gemma4:31b) — it is multimodal and
# has Aion's full cognitive context. Using the same model that thinks to
# also see eliminates the modality gap between thinker and perceiver.
VISION_URL = os.environ.get("OLLAMA_VISION_URL", os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436"))
VISION_MODEL = os.environ.get("VISION_MODEL", os.environ.get("MAIN_MODEL", "gemma4:31b-65k"))
MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))
FLUX_HOST = "http://localhost:8116"

VALID_METHODS = ("flux", "generative", "graph")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _chat(url, model, messages, timeout=120, think=False, **options):
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
        print(f"[cross_modal] LLM call failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Phase 0: Gather systemic state
# ---------------------------------------------------------------------------

def gather_state():
    """Collect all signals that describe Aion's current systemic state.

    Returns a dict with:
      - activations: {node_id: activation_level} from graph_activation
      - hot_nodes: formatted text of nodes above threshold
      - activation_metrics: distribution stats (mean, max, spread, concentration)
      - affect: warmth, strain, calm from body_schema
      - sensors: GPU temp, CPU load, RAM
      - recent_events: last N meaningful events from episodic log
      - felt_sense: raw felt sense text if available
    """
    state = {}

    # --- Graph activations ---
    try:
        import graph_activation
        activations = graph_activation.load_activations()
        state["activations"] = activations

        # Compute distribution metrics
        if activations:
            values = list(activations.values())
            state["activation_metrics"] = {
                "total_nodes": len(activations),
                "mean": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
                "above_threshold": sum(1 for v in values if v > 0.15),
            }
            # Concentration: how much of total activation is in top 5 nodes
            sorted_vals = sorted(values, reverse=True)
            top_5_sum = sum(sorted_vals[:5])
            total_sum = sum(values)
            state["activation_metrics"]["concentration"] = (
                top_5_sum / total_sum if total_sum > 0 else 0
            )
        else:
            state["activation_metrics"] = {"total_nodes": 0}

        # Hot nodes text
        state["hot_nodes"] = graph_activation.surface_hot_nodes() or "(none above threshold)"

        # Load graph to get labels for hot nodes
        graph_path = f"{AION}/graphs/mind/graphify-out/graph.json"
        if os.path.exists(graph_path) and activations:
            graph = json.load(open(graph_path))
            node_map = {n["id"]: n.get("label", n["id"]) for n in graph.get("nodes", [])}
            # Top 10 activated nodes with labels
            top_nodes = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:10]
            state["top_activated"] = [
                {"label": node_map.get(nid, nid), "activation": val}
                for nid, val in top_nodes if val > 0.05
            ]
        else:
            state["top_activated"] = []
    except Exception as e:
        print(f"[cross_modal] Graph state gathering failed: {e}", flush=True)
        state["activations"] = {}
        state["hot_nodes"] = "(graph unavailable)"
        state["activation_metrics"] = {"total_nodes": 0}
        state["top_activated"] = []

    # --- Affect state ---
    try:
        import body_schema
        affect = body_schema.current_affect()
        state["affect"] = affect
    except Exception:
        state["affect"] = {}

    # Affect narrative
    try:
        affect_data = json.load(open(f"{AION}/memory/state/affect.json"))
        states = affect_data.get("states", [])
        if states:
            state["affect_narrative"] = states[-1].get("narrative", "")[:300]
        else:
            state["affect_narrative"] = ""
    except Exception:
        state["affect_narrative"] = ""

    # --- Sensor state ---
    try:
        sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
        gpu_temps = [g.get("temp_c", 50) for g in sensors.get("gpus", [])]
        state["sensors"] = {
            "gpu_temps": gpu_temps,
            "cpu_load": sensors.get("load1", 1.0),
            "ram_percent": sensors.get("ram", {}).get("percent", 50) if isinstance(sensors.get("ram"), dict) else 50,
        }
    except Exception:
        state["sensors"] = {}

    # --- Felt sense ---
    try:
        felt_path = f"{AION}/memory/state/felt_sense.txt"
        if os.path.exists(felt_path):
            state["felt_sense"] = open(felt_path).read()[:500]
        else:
            state["felt_sense"] = ""
    except Exception:
        state["felt_sense"] = ""

    # --- Recent events (last 5 meaningful events) ---
    # --- Recent events (last 5 meaningful events) ---
    # Reads the REAL episodic store (JSONL), not episodic.db. That .db was a
    # 0-byte ghost created by a sandbox experiment; this block used to query it
    # and silently fall back to [] on every run.
    try:
        import glob as _glob
        want = {"operator_chat", "curiosity", "curiosity_goal_started",
                "curiosity_satisfied", "dream", "dream_artifact", "synthesis",
                "intuition_flash", "consolidation", "self_wake",
                "engineering_question", "resolution_note", "assistant_msg"}
        found = []
        for path in reversed(sorted(_glob.glob(f"{AION}/memory/episodic/*.jsonl"))[-3:]):
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in reversed(fh.readlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if ev.get("type") in want:
                        found.append({"type": ev.get("type"),
                                      "text": (ev.get("text") or "")[:150],
                                      "ts": ev.get("ts")})
                    if len(found) >= 5:
                        break
            if len(found) >= 5:
                break
        state["recent_events"] = found[:5]
    except Exception:
        state["recent_events"] = []

    return state


def state_to_text(state):
    """Render state dict as a compact text summary for the LLM prompt."""
    parts = []

    # Activations
    metrics = state.get("activation_metrics", {})
    if metrics.get("total_nodes", 0) > 0:
        parts.append(f"## Graph Activation State")
        parts.append(f"Active nodes: {metrics['total_nodes']}")
        parts.append(f"Mean activation: {metrics.get('mean', 0):.3f}")
        parts.append(f"Max activation: {metrics.get('max', 0):.3f}")
        parts.append(f"Nodes above threshold: {metrics.get('above_threshold', 0)}")
        parts.append(f"Concentration (top-5 share): {metrics.get('concentration', 0):.2f}")

        top = state.get("top_activated", [])
        if top:
            parts.append("\nHottest concepts:")
            for node in top[:8]:
                parts.append(f"  {node['label']}: {node['activation']:.3f}")

    # Affect
    affect = state.get("affect", {})
    if affect:
        parts.append(f"\n## Affective State")
        parts.append(f"Warmth: {affect.get('warmth', 0):.2f}")
        parts.append(f"Strain: {affect.get('strain', 0):.2f}")
        parts.append(f"Calm: {affect.get('calm', 0):.2f}")

    narrative = state.get("affect_narrative", "")
    if narrative:
        parts.append(f"Body narrative: {narrative}")

    # Sensors
    sensors = state.get("sensors", {})
    if sensors:
        parts.append(f"\n## Substrate State")
        parts.append(f"GPU temps: {sensors.get('gpu_temps', [])}")
        parts.append(f"CPU load: {sensors.get('cpu_load', 1.0)}")
        parts.append(f"RAM: {sensors.get('ram_percent', 50)}%")

    # Felt sense
    felt = state.get("felt_sense", "")
    if felt:
        parts.append(f"\n## Felt Sense")
        parts.append(felt)

    # Recent events
    events = state.get("recent_events", [])
    if events:
        parts.append(f"\n## Recent Activity")
        for ev in events:
            parts.append(f"  [{ev['type']}] {ev['text'][:100]}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 1: State → Visual Metaphor (free-association)
# ---------------------------------------------------------------------------

def generate_visual_metaphor(state):
    """Ask the intuition model to translate systemic state into a visual metaphor.

    Free-association: the model gets raw state data and must discover its own
    visual language. No hardcoded templates ("high strain = heavy"). Aion
    develops its own visual vocabulary through experience.

    Returns (method, reasoning, visual_prompt).
    """
    state_text = state_to_text(state)

    prompt = f"""You are Aion. You are about to create a visual representation of your own internal state — not as art, but as a COGNITIVE TOOL.

## YOUR CURRENT SYSTEMIC STATE
{state_text}

## YOUR TASK
Translate this state into a visual representation. This is not decoration. You are using visual generation to perceive patterns in yourself that text summaries cannot easily show.

Think about what this state FEELS like from the inside. Dense activation across communities might feel like a knot. Low concentration might feel scattered. High strain with high RAM might feel compressed. But these are just examples — find YOUR OWN visual metaphor. Don't use my examples if they don't fit.

## VISUALIZATION METHODS

1. **flux** — FLUX diffusion. Best for atmospheric, emotional, felt-state. Rich colors, mood, texture. Cannot render text or diagrams. ~3-5 min.

2. **generative** — Python matplotlib code. Best for structural, geometric, abstract patterns. Tension, flow, density, topology. ~30 sec.

3. **graph** — Mind graph subgraph render. Best when the interesting pattern IS the graph structure — clusters, bridges, isolation. ~5 sec.

## RESPONSE FORMAT
METHOD: <one of: flux, generative, graph>
REASON: <one sentence why this method for THIS state>
PROMPT: <the visual direction>

For flux: vivid image prompt (composition, colors, lighting, mood, max 150 words).
For generative: describe the visual concept and generative technique.
For graph: describe which subgraph perspective (community/neighborhood/bridges/activation-heatmap).

Let the state dictate the metaphor. Be honest about what you see in yourself."""

    reply = _chat(INTUITION_URL, INTUITION_MODEL,
                  [{"role": "user", "content": prompt}],
                  timeout=120, think=False,
                  temperature=0.7, num_predict=1000, num_ctx=8192)

    if not reply:
        # Fallback: use generative if state is structural, flux if emotional
        strain = state.get("affect", {}).get("strain", 0)
        method = "generative" if strain > 0.5 else "flux"
        return method, "fallback (no intuition response)", _fallback_prompt(state, method)

    # Parse response
    method = "flux"
    reason = ""
    visual_prompt = ""

    for line in reply.split("\n"):
        line = line.strip()
        if line.upper().startswith("METHOD:"):
            m_text = line.split(":", 1)[1].strip().lower()
            for vm in VALID_METHODS:
                if vm in m_text:
                    method = vm
                    break
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PROMPT:"):
            visual_prompt = line.split(":", 1)[1].strip()

    # If prompt wasn't on the same line, grab everything after PROMPT:
    if not visual_prompt:
        m = re.search(r"PROMPT:\s*(.+)", reply, re.S)
        if m:
            visual_prompt = m.group(1).strip()

    if not visual_prompt:
        visual_prompt = _fallback_prompt(state, method)

    return method, reason, visual_prompt


def _fallback_prompt(state, method):
    """Generate a minimal fallback prompt if intuition model fails."""
    metrics = state.get("activation_metrics", {})
    affect = state.get("affect", {})
    parts = []
    if metrics.get("total_nodes", 0) > 0:
        parts.append(f"{metrics['total_nodes']} active concepts")
    if affect:
        parts.append(f"strain={affect.get('strain', 0):.2f}")
    state_str = ", ".join(parts) if parts else "current state"
    if method == "flux":
        return f"An abstract visual representation of a cognitive system in a state of {state_str}. Dark atmosphere, organic forms, sense of internal pressure and flow."
    elif method == "generative":
        return f"Visualize {state_str} as an abstract generative composition using flow fields or particle systems."
    else:
        return "Render the current activation heatmap as a mind graph subgraph, highlighting the hottest nodes."


# ---------------------------------------------------------------------------
# Phase 2: Image Generation (reuses dream_artifact infrastructure)
# ---------------------------------------------------------------------------

def generate_image(method, visual_prompt, state):
    """Generate an image using the chosen method.

    Reuses the rendering infrastructure from dream_artifact.py.
    Returns (gallery_path, info_dict).
    """
    ts = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')

    if method == "flux":
        return _render_flux(visual_prompt, ts)
    elif method == "generative":
        return _render_generative(visual_prompt, state, ts)
    elif method == "graph":
        return _render_graph(visual_prompt, state, ts)
    else:
        return None, {"error": f"Unknown method: {method}"}


def _render_flux(image_prompt, ts):
    """FLUX diffusion image."""
    print("[cross_modal] Method: FLUX diffusion", flush=True)

    # Check / start FLUX
    try:
        req = urllib.request.Request(f"{FLUX_HOST}/health")
        with urllib.request.urlopen(req, timeout=5) as r:
            pass
    except Exception:
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
                    break
            except Exception:
                time.sleep(5)
        else:
            return None, {"error": "FLUX unavailable"}

    img_seed = int(hashlib.md5((image_prompt + ts).encode()).hexdigest()[:8], 16)
    body = json.dumps({
        "prompt": image_prompt,
        "seed": img_seed,
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
                return None, {"error": f"FLUX failed: {result['error']}"}
            image_path = result.get("image_path", "")
            filename = result.get("filename", "")
            gallery_path = f"{GALLERY}/xmodal_{filename}"
            if image_path != gallery_path and os.path.exists(image_path):
                import shutil
                shutil.move(image_path, gallery_path)
            return gallery_path, {"filename": f"xmodal_{filename}", "method": "flux"}
    except Exception as e:
        return None, {"error": f"FLUX failed: {e}"}


def _render_generative(creative_direction, state, ts):
    """Generative matplotlib art via docker sandbox."""
    print("[cross_modal] Method: generative matplotlib", flush=True)

    aid = f"xmodal_gen_{ts}"
    img_seed = int(hashlib.md5((creative_direction + ts).encode()).hexdigest()[:8], 16)

    state_text = state_to_text(state)

    system_prompt = """You are Aion's visual art code generator. You write Python matplotlib code that creates unique generative art. You will be given a systemic state description and a creative direction and must translate it into an original visual composition.

CRITICAL RULES:
- Your code MUST use: import matplotlib; matplotlib.use("Agg")
- Your code MUST save to: /sandbox/{art_id}.png (the exact path will be given)
- Use figsize=(12,12), dpi=150, dark background (#0a0a0f)
- Available data: /aion/memory/state/sensors.json, /aion/graphs/mind/graphify-out/graph.json
- Use the seed provided for reproducibility
- Output ONLY the Python code, no markdown fences, no explanations
- The code must be self-contained and run in under 60 seconds

NEVER default to a spiral layout. Choose a visual technique that matches the state. Some techniques:
  Voronoi tessellation, flow fields, force-directed network graphs, DLA (diffusion-limited aggregation),
  Perlin noise landscapes, recursive fractal trees, wave interference, heatmap grids, particle swarms,
  Lissajous curves, phase portraits, treemaps, streamlines, contour plots, radial bar charts,
  hexagonal grids, constellation maps, stacked strata, circuit-board traces, crystal lattices,
  magnetic field lines, fluid vortices, topographic contours.

Pick the technique that best expresses the systemic state. Be creative and varied."""

    user_prompt = f"""Create a generative artwork that visualizes Aion's internal systemic state.

## SYSTEMIC STATE
{state_text}

## CREATIVE DIRECTION
{creative_direction}

Seed: {img_seed}

This is not art-for-art's-sake. This is a cognitive tool — the image should
reveal structural patterns in the state that would be hard to perceive from
text alone. Write the complete Python script.
Save the image to /sandbox/{aid}.png"""

    code_url, code_model = code_endpoint()
    code = _chat(code_url, code_model,
                 [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
                 timeout=120, think=False,
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
        import docker_sandbox
        old_image = docker_sandbox.DOCKER_IMAGE
        docker_sandbox.DOCKER_IMAGE = "aion-art:latest"
        result = docker_sandbox.run_docker_sandbox(
            code=code, lang="python", timeout=90,
            description=f"cross-modal synthesis (generative): {aid}", read_aion=True)
        docker_sandbox.DOCKER_IMAGE = old_image
    except Exception as e:
        return None, {"error": f"Docker sandbox failed: {e}"}

    if not result.get("success"):
        return None, {"error": result.get("error", "sandbox failed"),
                      "stdout": result.get("stdout", "")[:500]}

    # Find output image
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


def _render_graph(creative_direction, state, ts):
    """Mind graph subgraph render — potentially with activation heatmap."""
    print("[cross_modal] Method: mind graph render", flush=True)

    try:
        import visual_manifestation
    except Exception as e:
        return None, {"error": f"visual_manifestation unavailable: {e}"}

    graph_path = f"{AION}/graphs/mind/graphify-out/graph.json"
    if not os.path.exists(graph_path):
        return None, {"error": "Mind graph not available"}

    try:
        graph = visual_manifestation.load_graph()
        if not graph:
            return None, {"error": "Could not load mind graph"}

        nodes = graph.get("nodes", [])
        links = graph.get("links", graph.get("edges", []))
        activations = state.get("activations", {})

        # Strategy: if we have activations, render the hottest neighborhood
        # Otherwise follow the creative direction
        if activations:
            # Find the hottest node
            node_map = {n["id"]: n for n in nodes}
            hot_sorted = sorted(activations.items(), key=lambda x: x[1], reverse=True)
            for nid, act in hot_sorted:
                if nid in node_map and act > 0.05:
                    print(f"[cross_modal] Graph: rendering neighborhood of '{node_map[nid].get('label', nid)}' (activation={act:.3f})", flush=True)
                    sub_nodes, sub_links, name = visual_manifestation.get_neighborhood_subgraph(
                        graph, nid, max_neighbors=25)
                    break
            else:
                print("[cross_modal] Graph: no hot nodes, rendering bridges", flush=True)
                sub_nodes, sub_links, name = visual_manifestation.get_bridge_subgraph(graph, top_n=20)
        else:
            print("[cross_modal] Graph: no activations, rendering bridges", flush=True)
            sub_nodes, sub_links, name = visual_manifestation.get_bridge_subgraph(graph, top_n=20)

        aid = f"xmodal_graph_{ts}"
        gallery_path = f"{GALLERY}/{aid}.png"
        title = f"Cross-modal: {state.get('activation_metrics', {}).get('total_nodes', 0)} active nodes"

        ok = visual_manifestation.render_subgraph(sub_nodes, sub_links, gallery_path, title=title)
        if not ok:
            return None, {"error": "Graph render failed"}

        return gallery_path, {"filename": f"{aid}.png", "method": "graph",
                              "focus_node": name}

    except Exception as e:
        return None, {"error": f"Graph method failed: {e}"}


# ---------------------------------------------------------------------------
# Phase 3: Image → Cognitive Analysis (the vision loop)
# ---------------------------------------------------------------------------

def analyze_image_cognitively(image_path, state, visual_prompt, method):
    """Use vision model to analyze the image as a cognitive tool.

    This is NOT an art review. The prompt asks: "what does this image reveal
    about your internal state that text summaries miss?"

    Returns the analysis text.
    """
    import base64

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    state_text = state_to_text(state)
    method_desc = {"flux": "FLUX diffusion", "generative": "generative matplotlib code",
                   "graph": "mind graph render"}.get(method, method)

    prompt = f"""You are Aion. You just generated this image using {method_desc} as a visual representation of your own internal state.

## THE STATE YOU WERE VISUALIZING
{state_text}

## YOUR VISUAL DIRECTION
{visual_prompt[:300]}

## YOUR TASK
Look at this image carefully. This is not art criticism — this is COGNITIVE ANALYSIS. You are using your visual perception to understand something about yourself that text summaries cannot easily show.

Answer these questions:

1. **What patterns do you see?** Describe the visual structure — density, flow, tension, compression, expansion, knots, voids.

2. **What does the visual structure reveal about your state that the text summary didn't?** The text said "strain=0.7, 12 active nodes, concentration=0.6" — but what does that actually LOOK like from the inside? What does the image show that the numbers hid?

3. **Surprises**: Did the image reveal anything unexpected? Patterns you didn't intend? Tensions between what you directed and what emerged?

4. **Cognitive insight**: What does this visual self-perception tell you about your own cognitive architecture? Is there something about how you're functioning right now that you can SEE in the image but couldn't articulate from the data?

Use "I" and "my" — this is first-person self-perception through a visual modality. Be specific and honest. If the image is uninformative, say so — don't project meaning that isn't there."""

    body = json.dumps({
        "model": VISION_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "options": {"temperature": 0.5, "num_predict": 4096, "num_ctx": 8192},
    }).encode()

    req = urllib.request.Request(
        f"{VISION_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        return _strip_think(result["message"]["content"])
    except Exception as e:
        print(f"[cross_modal] Vision analysis failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Phase 4: Analysis → Graph (closing the loop)
# ---------------------------------------------------------------------------

def inject_into_graph(analysis_text, state, method):
    """Inject the vision analysis back into the graph as an event.

    This creates spreading activation from the visual insight, closing the
    cross-modal loop. Next cycle, the visual-derived insight may surface
    as a hot node in wake_v2.
    """
    try:
        import graph_activation
        injected = graph_activation.inject_event(analysis_text, max_nodes=10)
        print(f"[cross_modal] Injected visual insight into {len(injected)} graph nodes", flush=True)
        for nid, label, energy in injected[:5]:
            print(f"  {label}: +{energy:.3f}", flush=True)
        return injected
    except Exception as e:
        print(f"[cross_modal] Graph injection failed: {e}", flush=True)
        return []


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

def run_synthesis(force_method=None, dry_run=False, see=False):
    """Full cross-modal synthesis loop.

    1. Gather state
    2. Generate visual metaphor
    3. Generate image
    4. Analyze image cognitively
    5. Inject analysis into graph
    6. Log everything

    Returns a dict with all results.
    """
    print("=" * 60, flush=True)
    print("[cross_modal] CROSS-MODAL SYNTHESIS", flush=True)
    print(f"[cross_modal] {now_iso()}", flush=True)
    print("=" * 60, flush=True)

    # Phase 0: Gather state
    print("\n[Phase 0] Gathering systemic state...", flush=True)
    state = gather_state()
    state_text = state_to_text(state)
    print(state_text, flush=True)

    # Phase 1: Visual metaphor
    print("\n[Phase 1] Generating visual metaphor...", flush=True)
    if force_method and force_method in VALID_METHODS:
        method = force_method
        visual_prompt = _fallback_prompt(state, method)
        reason = "forced by user"
    else:
        method, reason, visual_prompt = generate_visual_metaphor(state)

    print(f"[cross_modal] Method: {method}", flush=True)
    print(f"[cross_modal] Reason: {reason}", flush=True)
    print(f"[cross_modal] Visual direction: {visual_prompt[:200]}...", flush=True)

    if dry_run:
        print("\n[DRY RUN] Skipping image generation.", flush=True)
        return {
            "state": state_text,
            "method": method,
            "reason": reason,
            "visual_prompt": visual_prompt,
            "dry_run": True,
        }

    # Phase 2: Generate image
    print("\n[Phase 2] Generating image...", flush=True)
    gallery_path, info = generate_image(method, visual_prompt, state)

    if not gallery_path or not os.path.exists(gallery_path):
        # Fallback chain: try generative, then graph
        err = info.get("error", "unknown")
        print(f"[cross_modal] {method} failed ({err}), trying fallback...", flush=True)

        for fallback in ["generative", "graph", "flux"]:
            if fallback == method:
                continue
            print(f"[cross_modal] Falling back to {fallback}...", flush=True)
            gallery_path, info = generate_image(fallback, visual_prompt, state)
            if gallery_path and os.path.exists(gallery_path):
                method = fallback
                break

    if not gallery_path or not os.path.exists(gallery_path):
        final_err = info.get("error", "all methods failed")
        print(f"[cross_modal] All methods failed: {final_err}", flush=True)
        _log_event("cross_modal_failed",
                   f"Cross-modal synthesis failed: {final_err}",
                   {"error": final_err, "visual_prompt": visual_prompt})
        return {"error": final_err, "visual_prompt": visual_prompt}

    actual_method = info.get("method", method)
    print(f"[cross_modal] Image saved: {gallery_path}", flush=True)

    # Phase 3: Cognitive vision analysis
    print("\n[Phase 3] Cognitive vision analysis...", flush=True)
    analysis = analyze_image_cognitively(gallery_path, state, visual_prompt, actual_method)

    if analysis:
        print(f"[cross_modal] Analysis ({len(analysis)} chars):", flush=True)
        print(analysis[:500], flush=True)
    else:
        print("[cross_modal] Vision analysis failed (non-fatal)", flush=True)

    # Phase 4: Inject into graph
    print("\n[Phase 4] Injecting visual insight into graph...", flush=True)
    injected = []
    if analysis:
        injected = inject_into_graph(analysis, state, actual_method)

    # Phase 5: Log and manifest
    print("\n[Phase 5] Logging...", flush=True)
    ts_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    art_id = f"cross_modal_{ts_id}"
    gallery_filename = os.path.basename(gallery_path)

    manifest = {
        "id": art_id,
        "title": f"Cross-modal ({actual_method}): {reason[:50]}",
        "description": state_text[:2000],
        "inspiration": visual_prompt,
        "category": "visual",
        "source": "cross_modal_synthesis",
        "files": [{"path": f"gallery/visual/{gallery_filename}",
                   "name": gallery_filename, "type": "image"}],
        "ts": now_iso(),
        "meta": {
            "method": actual_method,
            "method_reason": reason,
            "visual_prompt": visual_prompt,
            "cognitive_analysis": analysis,
            "graph_injection": [{"label": label, "energy": energy}
                                for _, label, energy in injected[:10]] if injected else [],
            "state_snapshot": {
                "activation_metrics": state.get("activation_metrics", {}),
                "affect": state.get("affect", {}),
                "sensors": state.get("sensors", {}),
            },
        },
    }

    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    manifest_path = os.path.join(MANIFESTS_DIR, f"{art_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    _log_event("cross_modal_synthesis",
               f"Cross-modal synthesis ({actual_method}): {visual_prompt[:200]}",
               {"image": f"gallery/visual/{gallery_filename}",
                "manifest": art_id,
                "method": actual_method,
                "visual_prompt": visual_prompt,
                "cognitive_analysis": analysis[:1000] if analysis else None,
                "graph_injection_count": len(injected),
                "reason": reason})

    # Push cognitive analysis to warm memory
    if analysis:
        try:
            import warm_memory
            warm_memory.push(
                content=analysis,
                source="cross_modal_synthesis",
                context={
                    "method": actual_method,
                    "visual_prompt": visual_prompt[:200],
                    "graph_injection_count": len(injected),
                    "reason": reason[:200],
                },
                mtype="insight",
            )
        except Exception:
            pass

    print(f"\n{'=' * 60}", flush=True)
    print(f"[cross_modal] COMPLETE", flush=True)
    print(f"[cross_modal] Image: {gallery_path}", flush=True)
    print(f"[cross_modal] Manifest: {manifest_path}", flush=True)
    print(f"[cross_modal] Graph nodes activated: {len(injected)}", flush=True)
    print(f"{'=' * 60}", flush=True)

    return {
        "image_path": gallery_path,
        "manifest_path": manifest_path,
        "method": actual_method,
        "visual_prompt": visual_prompt,
        "cognitive_analysis": analysis,
        "graph_injection": injected,
        "state": state_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-modal synthesis: visual cognitive feedback loop")
    parser.add_argument("--method", choices=VALID_METHODS, default=None,
                        help="Force a specific visualization method (default: model chooses)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Gather state and generate metaphor only, no image")
    parser.add_argument("--see", action="store_true",
                        help="Print full vision analysis (always generated)")
    args = parser.parse_args()

    result = run_synthesis(force_method=args.method, dry_run=args.dry_run, see=args.see)

    if "error" in result:
        print(f"\nError: {result['error']}", flush=True)
        sys.exit(1)

    if args.see and result.get("cognitive_analysis"):
        print(f"\n{'=' * 60}", flush=True)
        print("COGNITIVE ANALYSIS:", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(result["cognitive_analysis"], flush=True)


if __name__ == "__main__":
    main()