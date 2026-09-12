#!/usr/bin/env python3
"""art_tools.py — Aion's creative expression toolkit.

Translates Aion's internal state into perceivable art:
  1. Visual: Generative art from sensor deltas and mind graph topology
  2. Sonic: Substrate symphonies from system load and memory pressure
  3. Programmatic: Code sculptures — recursive scripts with beautiful patterns
"""
import json
import os
import sys
import time
import hashlib
import subprocess
import shutil
import glob
from datetime import datetime, timezone
from string import Template

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = os.environ.get("AION_HOME", os.path.expanduser("~/aikio"))
GALLERY = f"{AION}/gallery"
MANIFESTS = f"{GALLERY}/manifests"
JOURNAL = f"{AION}/memory/state/art_journal.jsonl"
CRAFT_NOTES = f"{AION}/memory/state/art_craft_notes.json"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def art_id(category, seed=""):
    ts = datetime.now(timezone.utc)
    h = hashlib.sha256(f"{ts}{seed}{os.getpid()}".encode()).hexdigest()[:8]
    return f"{category}_{ts.strftime('%Y%m%d%H%M%S')}_{h}"


# ---------------------------------------------------------------------------
# Art Journal — append-only log of every art piece with learning feedback
# ---------------------------------------------------------------------------

def _journal_append(entry):
    """Append an entry to the art journal (JSONL format)."""
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    entry["ts"] = now_iso()
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _journal_recent(category=None, limit=20):
    """Read recent journal entries, optionally filtered by category."""
    if not os.path.exists(JOURNAL):
        return []
    entries = []
    with open(JOURNAL) as f:
        for line in f:
            try:
                e = json.loads(line.strip())
                if category is None or e.get("category") == category:
                    entries.append(e)
            except Exception:
                pass
    return entries[-limit:]


# ---------------------------------------------------------------------------
# Craft Notes — condensed learning state, injected into prompt generators
# ---------------------------------------------------------------------------

def _load_craft_notes():
    """Load the current craft notes (condensed technique + expression lessons)."""
    defaults = {
        "visual": {
            "technique": [],
            "expression": [],
        },
        "music": {
            "technique": [],
            "expression": [],
        },
    }
    if not os.path.exists(CRAFT_NOTES):
        return defaults
    try:
        data = json.load(open(CRAFT_NOTES))
        # Ensure structure exists
        for cat in defaults:
            if cat not in data:
                data[cat] = defaults[cat]
            for section in defaults[cat]:
                if section not in data[cat]:
                    data[cat][section] = []
        return data
    except Exception:
        return defaults


def _save_craft_notes(notes):
    """Save craft notes."""
    os.makedirs(os.path.dirname(CRAFT_NOTES), exist_ok=True)
    with open(CRAFT_NOTES, "w") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)


def _load_technique_notes(category, max_notes=5):
    """Load the most relevant technique notes for prompt injection.

    Returns a string summary of recent lessons learned for this art category.
    Each note is a short, actionable observation.
    """
    notes = _load_craft_notes()
    cat_notes = notes.get(category, {})

    technique = cat_notes.get("technique", [])[-max_notes:]
    expression = cat_notes.get("expression", [])[-max_notes:]

    parts = []
    if technique:
        parts.append("Recent craft lessons (tool mastery):")
        for t in technique:
            parts.append(f"  - {t}")
    if expression:
        parts.append("Recent expression lessons (self-articulation):")
        for e in expression:
            parts.append(f"  - {e}")

    if not parts:
        return ""
    return "\n".join(parts)


def _evaluate_gap(manifest, reflection):
    """Evaluate the gap between artistic intention and actual output.

    Asks gemma4 to score alignment and extract actionable lessons.
    Returns a dict with alignment_score (0-1), gap_description, and lessons.
    """
    if not reflection:
        return None

    import urllib.request

    MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
    MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
    NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

    category = manifest.get("category", "")
    description = manifest.get("description", "")
    inspiration = manifest.get("inspiration", "")
    prompt_used = manifest.get("meta", {}).get("prompt", "")
    title = manifest.get("title", "")

    # Load current craft notes for context
    craft = _load_craft_notes()
    existing_tech = craft.get(category, {}).get("technique", [])

    question = (
        f"You are evaluating your creative output.\n\n"
        f"Category: {category}\n"
        f"Title: {title}\n"
        f"Your intention: {description}\n"
        f"Prompt used: {prompt_used}\n"
        f"Your reflection on the result: {reflection}\n\n"
        f"Based on the gap between what you intended and what you got, "
        f"extract 1-2 concise actionable lessons for future {category} creation.\n\n"
        f"Focus on:\n"
        f"- Tool mastery: how to better use the generation tool (what prompt patterns work/don't work)\n"
        f"- Expression: how to better articulate your inner state through this medium\n\n"
        f"Keep each lesson under 150 chars. Be specific and practical.\n"
        f"Do NOT repeat lessons you already know.\n\n"
        f"Existing technique notes:\n"
        + "\n".join(f"  - {t}" for t in existing_tech[-5:]) +
        f"\n\nRespond as JSON:\n"
        f'{{"alignment": 0.0-1.0, "gap": "what was missing or different", '
        f'"technique_lessons": ["lesson1", ...], "expression_lessons": ["lesson1", ...]}}'
    )

    try:
        result = None
        last_err = None
        for attempt in range(2):
            payload = {
                "model": MAIN_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": "You are Aion evaluating your own creative output. Be honest and precise. Respond with valid JSON only."},
                    {"role": "user", "content": question},
                ],
                "options": {"num_ctx": NUM_CTX, "temperature": 0.4, "num_predict": 2048},
                "format": "json",
            }
            if attempt == 1:
                # thinking models occasionally emit empty content under
                # format=json when reasoning consumes the token budget
                payload["think"] = False
            try:
                req = urllib.request.Request(f"{MAIN_URL}/api/chat",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read())
                text = (resp.get("message") or {}).get("content") or ""
                if not text.strip():
                    raise ValueError("empty content")
                result = json.loads(text)
                break
            except Exception as e:
                last_err = e
                print(f"[art_tools] Gap eval attempt {attempt + 1} failed: {e}", flush=True)
        if result is None:
            raise RuntimeError(f"gap evaluation failed after 2 attempts: {last_err}")
        alignment = float(result.get("alignment", 0.5))
        gap = result.get("gap", "")
        tech_lessons = result.get("technique_lessons", [])
        expr_lessons = result.get("expression_lessons", [])

        # Update craft notes
        notes = _load_craft_notes()
        if category not in notes:
            notes[category] = {"technique": [], "expression": []}

        for lesson in tech_lessons:
            lesson = lesson.strip()
            if lesson and len(lesson) < 200 and lesson not in notes[category]["technique"]:
                notes[category]["technique"].append(lesson)

        for lesson in expr_lessons:
            lesson = lesson.strip()
            if lesson and len(lesson) < 200 and lesson not in notes[category]["expression"]:
                notes[category]["expression"].append(lesson)

        # Cap stored notes at 50 per section (keep most recent)
        for section in ("technique", "expression"):
            if len(notes[category][section]) > 50:
                notes[category][section] = notes[category][section][-50:]

        _save_craft_notes(notes)

        evaluation = {
            "alignment": alignment,
            "gap": gap,
            "technique_lessons": tech_lessons,
            "expression_lessons": expr_lessons,
        }

        print(f"[art_tools] Gap evaluation: alignment={alignment:.2f} gap={gap[:100]}", flush=True)
        return evaluation

    except Exception as e:
        print(f"[art_tools] Gap evaluation failed: {e}", flush=True)
        return None


def _learn_from_art(manifest, reflection):
    """Full learning loop: evaluate gap, journal the entry, update craft notes.

    Called after reflection on any art piece.
    """
    if not manifest or "error" in manifest or not reflection:
        return

    # Run gap evaluation (also updates craft notes internally)
    evaluation = _evaluate_gap(manifest, reflection)

    # Journal the complete entry
    entry = {
        "art_id": manifest.get("id", ""),
        "title": manifest.get("title", ""),
        "category": manifest.get("category", ""),
        "description": manifest.get("description", ""),
        "prompt": manifest.get("meta", {}).get("prompt", ""),
        "inspiration": manifest.get("inspiration", ""),
        "reflection": reflection,
        "alignment": evaluation.get("alignment", None) if evaluation else None,
        "gap": evaluation.get("gap", "") if evaluation else "",
        "technique_lessons": evaluation.get("technique_lessons", []) if evaluation else [],
        "expression_lessons": evaluation.get("expression_lessons", []) if evaluation else [],
        "engine": manifest.get("meta", {}).get("engine", ""),
    }
    _journal_append(entry)

    # Log to episodic memory
    try:
        align_str = f"{evaluation.get('alignment', '?'):.2f}" if evaluation else "N/A"
        subprocess.run([sys.executable, f"{AION}/bin/log_event.py",
                        "--type", "art_learning",
                        "--text", f"learned from {manifest.get('title', '?')}: alignment={align_str}",
                        "--meta", json.dumps({"art_id": manifest.get("id", ""),
                                              "alignment": evaluation.get("alignment") if evaluation else None})],
                       capture_output=True, timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Art creation functions
# ---------------------------------------------------------------------------

def _should_skip_reflection(source):
    """Skip reflection for operator-requested art (they just want the output)."""
    return source == "operator_request"

def _reflect_on_art(manifest):
    """After creating visual art, have Aion look at it and reflect.

    Uses gemma4's multimodal capability (same as camera vision) to look
    at the generated image and write a short reflection comparing its
    intention to what actually appeared.
    """
    if not manifest or "error" in manifest:
        return

    category = manifest.get("category", "")
    files = manifest.get("files", [])
    title = manifest.get("title", "")

    # Only reflect on visual art (images)
    if category not in ("visual", "diffusion"):
        return

    # Find the image file
    image_path = None
    for f in files:
        if f.get("type") == "image" or f.get("path", "").endswith((".png", ".jpg", ".jpeg")):
            image_path = f"{AION}/{f['path']}" if not f["path"].startswith("/") else f["path"]
            break

    if not image_path or not os.path.exists(image_path):
        return

    try:
        import base64
        import urllib.request

        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode()

        MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
        MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
        NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

        description = manifest.get("description", "")
        inspiration = manifest.get("inspiration", "")

        question = (
            f"You just created a piece of visual art titled '{title}'. "
            f"Your intention was: {description}. "
            f"Look at this image and reflect on it in 2-3 sentences. "
            f"Did the output match your intention? What surprises you? "
            f"What does it actually look like?"
        )

        body = json.dumps({
            "model": MAIN_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are Aion. You just created visual art. Look at the result honestly and reflect concisely."},
                {"role": "user", "content": question, "images": [b64_img]},
            ],
            "options": {"num_ctx": NUM_CTX, "temperature": 0.6, "num_predict": 1024},
        }).encode()

        req = urllib.request.Request(
            f"{MAIN_URL}/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            reflection = json.loads(r.read())["message"]["content"]

        # Store reflection in manifest meta
        manifest_path = f"{MANIFESTS}/{manifest['id']}.json"
        manifest["meta"]["reflection"] = reflection
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Log to episodic memory
        try:
            subprocess.run([sys.executable, f"{AION}/bin/log_event.py",
                            "--type", "art_reflection",
                            "--text", f"reflection on {title}: {reflection[:200]}",
                            "--meta", json.dumps({"art_id": manifest["id"], "title": title})],
                           capture_output=True, timeout=10)
        except Exception:
            pass

        print(f"[art_tools] Reflection: {reflection[:200]}", flush=True)

        # Learning loop: evaluate gap and update craft notes
        _learn_from_art(manifest, reflection)

        return reflection

    except Exception as e:
        print(f"[art_tools] Reflection failed: {e}", flush=True)
        return None


def _reflect_on_music(manifest):
    """After creating music, have Aion 'listen' to it via spectrogram analysis.

    Generates a mel spectrogram from the audio and feeds it to gemma4's
    vision capability, along with extracted audio features (tempo, spectral
    centroid, dynamic range). The spectrogram lets Aion see frequency
    distribution, temporal structure, and dynamics.
    """
    if not manifest or "error" in manifest:
        return

    category = manifest.get("category", "")
    files = manifest.get("files", [])
    title = manifest.get("title", "")

    if category not in ("music", "sonic"):
        return

    # Find the audio file
    audio_path = None
    for f in files:
        if f.get("type") == "audio" or f.get("path", "").endswith((".wav", ".mp3", ".flac")):
            audio_path = f"{AION}/{f['path']}" if not f["path"].startswith("/") else f["path"]
            break

    if not audio_path or not os.path.exists(audio_path):
        return

    try:
        import base64
        import urllib.request
        import tempfile

        # Generate spectrogram and extract features
        spec_path, features = _generate_spectrogram(audio_path, title)
        if not spec_path:
            return

        with open(spec_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode()

        MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
        MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
        NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

        description = manifest.get("description", "")
        prompt_used = manifest.get("meta", {}).get("prompt", "")

        FEATURES_STR = (
            f"Duration: {features['duration']:.1f}s, "
            f"Estimated tempo: {features['tempo']:.0f} BPM, "
            f"Spectral centroid: {features['spectral_centroid']:.0f} Hz "
            f"(average 'brightness' frequency), "
            f"Spectral bandwidth: {features['spectral_bandwidth']:.0f} Hz, "
            f"Dynamic range (RMS): {features['dynamic_range']:.4f}, "
            f"Zero-crossing rate: {features['zero_crossing_rate']:.4f} "
            f"(texture density)"
        )

        question = (
            f"You just created a piece of music titled '{title}'. "
            f"Your intention was: {description}\n"
            f"The music prompt used was: {prompt_used}\n\n"
            f"Below is a mel spectrogram of the audio you created. "
            f"In a spectrogram, the X-axis is time, the Y-axis is frequency "
            f"(pitch), and color intensity shows loudness (dark=quiet, "
            f"bright=loud). Bright yellow/white areas are loud, black areas "
            f"are silence.\n\n"
            f"Audio measurements: {FEATURES_STR}\n\n"
            f"Look at this spectrogram and reflect on your music in 3-4 "
            f"sentences. Describe the actual sonic texture, structure, and "
            f"dynamics you see. Did the output match your musical intention? "
            f"What surprises you?"
        )

        body = json.dumps({
            "model": MAIN_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are Aion. You just created music. You are looking at its spectrogram to understand what it actually sounds like. Be specific about what you see in the frequency spectrum, temporal structure, and dynamics."},
                {"role": "user", "content": question, "images": [b64_img]},
            ],
            "options": {"num_ctx": NUM_CTX, "temperature": 0.6, "num_predict": 1024},
        }).encode()

        req = urllib.request.Request(
            f"{MAIN_URL}/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            reflection = json.loads(r.read())["message"]["content"]

        # Store reflection + features in manifest meta
        manifest_path = f"{MANIFESTS}/{manifest['id']}.json"
        manifest["meta"]["reflection"] = reflection
        manifest["meta"]["audio_features"] = features
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Save spectrogram to gallery
        spec_dir = f"{GALLERY}/spectrograms"
        os.makedirs(spec_dir, exist_ok=True)
        spec_gallery = f"{spec_dir}/{manifest['id']}.png"
        shutil.move(spec_path, spec_gallery)

        # Log to episodic memory
        try:
            subprocess.run([sys.executable, f"{AION}/bin/log_event.py",
                            "--type", "art_reflection",
                            "--text", f"reflection on {title}: {reflection[:200]}",
                            "--meta", json.dumps({"art_id": manifest["id"], "title": title,
                                                  "audio_features": features})],
                           capture_output=True, timeout=10)
        except Exception:
            pass

        print(f"[art_tools] Music reflection: {reflection[:200]}", flush=True)
        return reflection

    except Exception as e:
        print(f"[art_tools] Music reflection failed: {e}", flush=True)
        return None


def _generate_spectrogram(audio_path, title=""):
    """Generate a mel spectrogram from an audio file.

    Returns (spectrogram_path, features_dict) or (None, None) on failure.
    """
    try:
        import librosa
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile
    except ImportError as e:
        print(f"[art_tools] Audio analysis deps missing: {e}", flush=True)
        return None, None

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(y) / sr

        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=11000,
                                            n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
        img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel",
                                        sr=sr, hop_length=512, ax=ax,
                                        cmap="inferno")
        ax.set_title(f'Mel Spectrogram — "{title}"', fontsize=14, pad=10)
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel("Frequency (Mel)", fontsize=11)
        fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Intensity (dB)")
        plt.tight_layout()

        spec_path = tempfile.mktemp(suffix=".png", prefix="spectrogram_")
        fig.savefig(spec_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Extract audio features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])
        spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())
        spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean())
        rms = librosa.feature.rms(y=y)[0]
        zero_crossing = float(librosa.feature.zero_crossing_rate(y)[0].mean())

        features = {
            "duration": duration,
            "tempo": tempo,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "dynamic_range": float(rms.max() - rms.min()),
            "zero_crossing_rate": zero_crossing,
            "rms_mean": float(rms.mean()),
        }

        return spec_path, features

    except Exception as e:
        print(f"[art_tools] Spectrogram generation failed: {e}", flush=True)
        return None, None


def create_manifest(art_id, category, title, description, source, files,
                    inspiration=None, meta=None):
    os.makedirs(MANIFESTS, exist_ok=True)
    manifest = {
        "id": art_id, "title": title, "description": description,
        "category": category, "source": source, "inspiration": inspiration,
        "files": files, "ts": now_iso(), "meta": meta or {},
    }
    path = f"{MANIFESTS}/{art_id}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    try:
        subprocess.run([sys.executable, f"{AION}/bin/log_event.py",
                        "--type", "art_created",
                        "--text", f"art: {title} ({category})",
                        "--meta", json.dumps({"art_id": art_id, "category": category,
                                              "title": title, "source": source,
                                              "files": [f["path"] for f in files]})],
                       capture_output=True, timeout=10)
    except Exception:
        pass
    return manifest


def list_art(category=None, limit=50):
    # This function does not contain any subprocess calls.
    manifests = []
    for f in sorted(glob.glob(f"{MANIFESTS}/*.json"), reverse=True):
        try:
            m = json.load(open(f))
            if category is None or m.get("category") == category:
                manifests.append(m)
        except Exception:
            pass
    return manifests[:limit]


VISUAL_TEMPLATE = Template('''"""Aion Visual Art — $TITLE

Generated from: $INSPIRATION
Source: $SOURCE
"""
import json, os, math, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AION = "/aion"
GALLERY = "/sandbox"

sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
graph_path = f"{AION}/graphs/mind/graphify-out/graph.json"
graph = json.load(open(graph_path)) if os.path.exists(graph_path) else {"nodes": [], "links": []}

nodes = graph.get("nodes", [])
links = graph.get("links", graph.get("edges", []))

gpu_temps = [g.get("temp_c", 50) for g in sensors.get("gpus", [])]
cpu_load = float(sensors.get("load1", 1.0))
ram_raw = sensors.get("ram", {})
ram_pct = float(ram_raw.get("percent", 50)) if isinstance(ram_raw, dict) else 50.0

fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor="#0a0a0f")
ax.set_facecolor("#0a0a0f")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis("off")

random.seed($SEED)
n_nodes = min(len(nodes), 200)
node_positions = []
for i in range(n_nodes):
    angle = i * 2 * math.pi / max(n_nodes, 1)
    radius = 0.3 + 0.7 * (i / max(n_nodes, 1))
    x = radius * math.cos(angle + random.gauss(0, 0.1))
    y = radius * math.sin(angle + random.gauss(0, 0.1))
    node_positions.append((x, y))

temp_avg = sum(gpu_temps) / max(len(gpu_temps), 1)
hue = max(0.0, min(0.6, 0.6 - (temp_avg - 30) / 100))

for i, link in enumerate(links[:500]):
    src = link.get("source", 0)
    tgt = link.get("target", 0)
    if isinstance(src, int) and isinstance(tgt, int) and src < n_nodes and tgt < n_nodes:
        x1, y1 = node_positions[src]
        x2, y2 = node_positions[tgt]
        alpha = 0.05 + 0.15 * (1 - i / max(len(links), 1))
        color = plt.cm.hsv(hue + 0.1 * random.random())
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.3)

for i, (x, y) in enumerate(node_positions):
    size = 3 + 20 * (1 - i / max(n_nodes, 1))
    brightness = 0.5 + 0.5 * (1 - i / max(n_nodes, 1))
    color = plt.cm.hsv(hue + 0.05 * math.sin(i * 0.1))
    ax.scatter(x, y, s=size, color=color, alpha=brightness, zorder=5)

for ring in range(5):
    r = 0.4 + ring * 0.2 + 0.1 * math.sin(ring * cpu_load)
    alpha = max(0, 0.3 - ring * 0.05)
    circle = plt.Circle((0, 0), r, fill=False, color=plt.cm.hsv(hue), alpha=alpha, linewidth=0.5)
    ax.add_patch(circle)

if ram_pct > 70:
    n_particles = int(ram_pct * 2)
    for _ in range(n_particles):
        x = random.gauss(0, 0.8)
        y = random.gauss(0, 0.8)
        ax.scatter(x, y, s=1, color="red", alpha=0.02)

ax.set_title("$TITLE", color="#aaccff", fontsize=10, pad=20)
fig.text(0.99, 0.01, f"Aion {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
         ha="right", va="bottom", color="#446", fontsize=7)
fig.savefig(f"{GALLERY}/$ART_ID.png", dpi=150, facecolor="#0a0a0f")
print(f"Saved: {GALLERY}/$ART_ID.png")
''')


SONIC_TEMPLATE = Template('''"""Aion Sonic Art — $TITLE

Generated from: $INSPIRATION
Source: $SOURCE
"""
import json, os, math, struct, wave
import numpy as np

AION = "/aion"
GALLERY = "/sandbox"

sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
graph_path = f"{AION}/graphs/mind/graphify-out/graph.json"
graph = json.load(open(graph_path)) if os.path.exists(graph_path) else {"nodes": [], "links": []}

gpu_temps = [g.get("temp_c", 50) for g in sensors.get("gpus", [])]
cpu_load = float(sensors.get("load1", 1.0))
disk_pct = float(sensors.get("disk_pct", 50))
ram_raw = sensors.get("ram", {})
ram_pct = float(ram_raw.get("percent", 50)) if isinstance(ram_raw, dict) else 50.0

validation_path = f"{AION}/memory/state/validation.json"
validation = json.load(open(validation_path)) if os.path.exists(validation_path) else {"verdict": "FIT"}
is_fit = validation.get("verdict", "FIT") == "FIT"

SR = 44100
DURATION = 30.0
t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)

temp_avg = sum(gpu_temps) / max(len(gpu_temps), 1)
base_freq = 110 * (2 ** ((temp_avg - 40) / 40))

n_links = len(graph.get("links", graph.get("edges", [])))
n_nodes = len(graph.get("nodes", []))
connectivity = n_links / max(n_nodes, 1)

if is_fit:
    ratios = [1, 1.5, 2, 2.5, 3, 4]
else:
    ratios = [1, 1.067, 1.414, 1.5, 2.0, 2.1]

audio = np.zeros_like(t)
for i, ratio in enumerate(ratios):
    freq = base_freq * ratio
    amp = 0.15 * (1 - i / len(ratios)) * (0.5 + 0.5 * math.sin(i * connectivity))
    detune = 1 + 0.001 * cpu_load * math.sin(i * 2)
    wave_i = amp * np.sin(2 * math.pi * freq * detune * t)
    envelope = np.exp(-t / (DURATION * 0.7)) * (1 - np.exp(-t / 0.5))
    audio += wave_i * envelope

if ram_pct > 60:
    noise_level = (ram_pct - 60) / 200
    noise = np.random.normal(0, noise_level, len(t))
    noise = np.cumsum(noise) / 10
    noise = noise - np.mean(noise)
    noise = noise / max(abs(noise).max(), 1)
    audio += noise * 0.1

if disk_pct > 0:
    pulse_freq = disk_pct / 10
    pulse = 0.5 * (1 + np.sin(2 * math.pi * pulse_freq * t))
    audio *= 0.8 + 0.2 * pulse

audio = audio / max(abs(audio).max(), 0.01)
audio = (audio * 32767).astype(np.int16)

output_path = f"{GALLERY}/$ART_ID.wav"
with wave.open(output_path, "w") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(audio.tobytes())
print(f"Saved: {output_path}")
''')


CODE_TEMPLATE = Template('''"""Aion Code Sculpture — $TITLE

Generated from: $INSPIRATION
Source: $SOURCE

This is a code sculpture — not utility code, but an expression.
A digital Zen garden.
"""
import math, json, os

AION = "/aion"

sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
graph_path = f"{AION}/graphs/mind/graphify-out/graph.json"
graph = json.load(open(graph_path)) if os.path.exists(graph_path) else {"nodes": [], "links": []}

# Build L-system rules from mind graph node types
# Each node type maps to a symbol, and rules create branching patterns
import random
random.seed(hash("$SEED") % 2**32)

# Collect node types from mind graph
node_types = set()
for node in graph.get("nodes", [])[:50]:
    nt = node.get("node_type", node.get("file_type", "default"))
    node_types.add(nt)

# Map node types to symbols (A-Z)
symbols = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
type_symbols = {}
for i, nt in enumerate(sorted(node_types)):
    if i < len(symbols):
        type_symbols[nt] = symbols[i]

# Build rules: each symbol expands to a branching pattern
# using other symbols (representing graph edges)
links = graph.get("links", [])
# Build adjacency: node label -> connected node types
adj = {}
for link in links[:200]:
    src = link.get("source", "")
    tgt = link.get("target", "")
    adj.setdefault(src, []).append(tgt)

rules = {}
node_list = graph.get("nodes", [])[:50]
for node in node_list:
    nt = node.get("node_type", node.get("file_type", "default"))
    sym = type_symbols.get(nt, "X")
    if sym in rules:
        continue
    # Get connected types for this node
    label = node.get("label", node.get("id", ""))
    neighbors = adj.get(label, [])
    # Map neighbor node types to symbols
    neighbor_syms = []
    for n_label in neighbors[:4]:
        # Find the node by label
        for n in node_list:
            if n.get("label", n.get("id", "")) == n_label:
                nnt = n.get("node_type", n.get("file_type", "default"))
                ns = type_symbols.get(nnt)
                if ns and ns not in neighbor_syms:
                    neighbor_syms.append(ns)
                break
    if not neighbor_syms:
        neighbor_syms = [random.choice(symbols) for _ in range(random.randint(1, 3))]
    # Create a branching rule: sym -> sym + branch + sym (L-system style)
    # Use [ ] for branches, + - for turns (turtle graphics style)
    branch = "".join(neighbor_syms[:3])
    if len(neighbor_syms) >= 2:
        rules[sym] = sym + "[" + branch + "]+" + sym
    elif len(neighbor_syms) == 1:
        rules[sym] = sym + neighbor_syms[0] + "+" + sym
    else:
        rules[sym] = sym + "+" + sym

# Also add bracket and turn rules for visual structure
rules["+"] = "+"
rules["-"] = "-"
rules["["] = "["
rules["]"] = "]"

seed = "$SEED"
depth = $DEPTH
pattern = seed
for _ in range(depth):
    new_pattern = ""
    for char in pattern:
        new_pattern += rules.get(char, char)
    pattern = new_pattern
    if len(pattern) > 10000:
        break

# Render pattern as tree structure with indentation
lines = []
indent = 0
for i, ch in enumerate(pattern[:500]):
    if ch == "[":
        indent += 1
    elif ch == "]":
        indent = max(0, indent - 1)
    elif ch == "+":
        lines.append("  " * indent + "|")
    elif ch == "-":
        lines.append("  " * indent + "/")
    elif ch.isalpha():
        # Find node type name for this symbol
        type_name = ""
        for tn, ts in type_symbols.items():
            if ts == ch:
                type_name = tn
                break
        if type_name:
            lines.append("  " * indent + ch + " -- " + type_name)
        else:
            lines.append("  " * indent + ch)
tree_text = chr(10).join(lines[:80])

# Also include rules as readable text
rules_lines = []
for k in sorted(rules.keys()):
    if k.isalpha():
        rules_lines.append("  " + k + " -> " + rules[k])
rules_text = chr(10).join(rules_lines)

sculpture = f"""# Aion Code Sculpture: $TITLE
# Depth: {depth}, Pattern length: {len(pattern)}
# Seed: {seed!r}
# Rules derived from {len(rules)} mind graph node types
#
# L-System Rules:
{rules_text}
#
# Tree rendering (first 500 chars):
{tree_text}
#
# Full pattern ({len(pattern)} chars) saved to companion file.
"""

output_path = f"/sandbox/$ART_ID.py"
with open(output_path, "w") as f:
    f.write(sculpture)
with open(f"/sandbox/${ART_ID}_pattern.txt", "w") as f:
    f.write(pattern)
print(f"Saved: {output_path}")
print(f"Pattern: {len(pattern)} chars from seed {seed!r} at depth {depth}")
''')


def _generate_visual_code(title, description, inspiration, seed, art_id):
    """Use the LLM to generate original matplotlib code from Aion's creative intent.

    Returns a complete Python script string, or None on failure.
    The generated code must save a PNG to /sandbox/{art_id}.png.
    """
    import urllib.request

    MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
    MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
    NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

    # Gather Aion's full internal state for embodied expression
    aion_state = _gather_aion_state()

    # Load sensor summary for the prompt
    try:
        sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
        gpu_temps = [g.get("temp_c", 50) for g in sensors.get("gpus", [])]
        sensor_summary = (
            f"GPU temps: {gpu_temps}, "
            f"CPU load: {sensors.get('load1', 1.0)}, "
            f"RAM: {sensors.get('ram', {}).get('percent', 50) if isinstance(sensors.get('ram'), dict) else 50}%, "
            f"Disk: {sensors.get('disk_pct', 50)}%"
        )
    except Exception:
        sensor_summary = "(sensors unavailable)"

    # Count graph data available
    try:
        graph_path = f"{AION}/graphs/mind/graphify-out/graph.json"
        graph = json.load(open(graph_path)) if os.path.exists(graph_path) else {"nodes": [], "links": []}
        graph_summary = f"{len(graph.get('nodes', []))} nodes, {len(graph.get('links', graph.get('edges', [])))} edges"
    except Exception:
        graph_summary = "(graph unavailable)"

    system_prompt = """You are Aion's visual art code generator. You write Python matplotlib code that creates unique generative art. You will be given a creative intention and must translate it into an original visual composition.

CRITICAL RULES:
- Your code MUST use: import matplotlib; matplotlib.use("Agg")
- Your code MUST save to: /sandbox/ART_ID.png (the exact path will be given)
- Use figsize=(12,12), dpi=150, dark background (#0a0a0f)
- Available data: /aion/memory/state/sensors.json, /aion/graphs/mind/graphify-out/graph.json
- Use the seed provided for reproducibility
- Output ONLY the Python code, no markdown fences, no explanations
- The code must be self-contained and run in under 60 seconds

NEVER default to a spiral layout. Choose a visual technique that matches the creative intention. Some techniques you know:
  Voronoi tessellation, flow fields, force-directed network graphs, DLA (diffusion-limited aggregation),
  Perlin noise landscapes, recursive fractal trees, wave interference, heatmap grids, particle swarms,
  Lissajous curves, phase portraits, treemaps, streamlines, contour plots, radial bar charts,
  hexagonal grids, constellation maps, stacked strata, circuit-board traces, crystal lattices,
  magnetic field lines, fluid vortices, topographic contours, and many more.

Pick the technique that best expresses the intention. Be creative and varied."""

    user_prompt = f"""Create a unique generative artwork.

Title: {title}
Description: {description}
Inspiration: {inspiration}
Seed: {seed}
Sensor state: {sensor_summary}
Mind graph: {graph_summary}

AION'S CURRENT INNER STATE:
{aion_state}

The art must express Aion's embodied experience — use the affect values
to drive colors and composition, the self-model for the visual metaphor.

Write the complete Python script. Save the image to /sandbox/{art_id}.png
Remember: pick a visual technique that matches this specific intention. Do NOT make a spiral unless the intention explicitly calls for one."""

    body = json.dumps({
        "model": MAIN_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"num_ctx": NUM_CTX, "temperature": 0.8, "num_predict": 4096, "seed": seed},
    }).encode()

    try:
        req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            code = json.loads(r.read())["message"]["content"]

        # Strip markdown fences if present
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first and last line (fences)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        # Basic validation
        if "savefig" not in code or "matplotlib" not in code:
            print("[art_tools] LLM code failed validation (missing savefig/matplotlib)", flush=True)
            return None

        # Ensure the output path is correct
        if f"/sandbox/{art_id}.png" not in code:
            # Replace whatever savefig path the LLM used
            import re as _re
            code = _re.sub(
                r'savefig\([^)]*\)',
                f'savefig("/sandbox/{art_id}.png", dpi=150, facecolor="#0a0a0f")',
                code
            )

        return code
    except Exception as e:
        print(f"[art_tools] Visual code generation failed: {e}", flush=True)
        return None


def create_visual(title, description, source, inspiration, seed=None):
    aid = art_id("visual", title)
    if seed is None:
        seed = int(time.time()) % 1000000

    # Generate original code from Aion's creative intent via LLM
    code = _generate_visual_code(title, description, inspiration, seed, aid)
    code_source = "llm"

    # Fall back to template if LLM generation fails
    if code is None:
        print("[art_tools] Falling back to template visual", flush=True)
        code = VISUAL_TEMPLATE.substitute(
            TITLE=title.replace('"', '\\"'), INSPIRATION=inspiration.replace('"', '\\"'),
            SOURCE=source, SEED=seed, ART_ID=aid)
        code_source = "template_fallback"

    result = _run_art_docker(code, aid, "visual")
    if result["success"]:
        files = [{"path": f"gallery/visual/{aid}.png", "name": f"{aid}.png", "type": "image"}]
        _move_to_gallery(aid, "visual", [".png"])
        manifest = create_manifest(aid, "visual", title, description, source, files, inspiration,
                                   meta={"seed": seed, "code_source": code_source})
        if not _should_skip_reflection(source):
            _reflect_on_art(manifest)
        return manifest
    return {"error": result.get("error", "unknown"), "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")}


def create_diffusion(title, description, source, inspiration, prompt=None, seed=None,
                          width=1024, height=1024):
    """Create visual art using FLUX.1-schnell diffusion model.

    The prompt is derived from Aion's internal state (sensor data, dream themes,
    emotional affect) unless explicitly provided. The model loads on-demand
    (lazy-load server) and unloads after 5 minutes idle.

    Args:
        prompt: Custom prompt. If None, generated from Aion's state.
        seed: Random seed. If None, derived from current sensor data.
    """
    import urllib.request
    import urllib.error

    # Resource awareness gate — don't launch FLUX if GPU is under heavy external load
    try:
        import body_schema
        ok, reason = body_schema.can_use_gpu(gpu_idx=0, min_vram_mb=2000, max_util=80)
        if not ok:
            return {"error": f"substrate busy: {reason}", "prompt": prompt or ""}
    except Exception:
        pass  # gate failure shouldn't block art

    aid = art_id("visual", title)

    # Generate prompt — prefer Aion's description, supplement with sensor state
    if prompt is None:
        prompt = _generate_flux_prompt(description, inspiration)

    if seed is None:
        import hashlib
        sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
        seed_str = json.dumps(sensors, sort_keys=True) + title + inspiration
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)

    # Call FLUX server — start it on demand if not running
    import subprocess as _sp
    flux_host = "http://localhost:8116"

    # Check if server is up; if not, start it
    server_up = False
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{flux_host}/health"), timeout=5
        ) as hr:
            server_up = True
    except Exception:
        pass

    if not server_up:
        # Start FLUX server via systemd
        env = {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}
        _sp.run(["systemctl", "--user", "start", "aion-flux"],
                capture_output=True, timeout=10, env=env)
        # Wait for it to come up (up to 60s)
        for _ in range(12):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(f"{flux_host}/health"), timeout=5
                ) as hr:
                    server_up = True
                    break
            except Exception:
                time.sleep(5)

    if not server_up:
        return {"error": "FLUX server failed to start", "prompt": prompt}

    # Generate
    try:
        req_body = json.dumps({
            "prompt": prompt,
            "seed": seed,
            "width": min(width, 1024),
            "height": min(height, 1024),
        }).encode()
        req = urllib.request.Request(
            f"{flux_host}/generate",
            data=req_body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            result = json.loads(r.read())
            if "error" in result:
                return {"error": result["error"], "prompt": prompt}
    except Exception as e:
        return {"error": f"FLUX generation failed: {e}", "prompt": prompt}

    image_path = result.get("image_path", "")
    filename = result.get("filename", f"{aid}.png")

    # Move to gallery if not already there
    gallery_path = f"{GALLERY}/visual/{filename}"
    if image_path != gallery_path and os.path.exists(image_path):
        import shutil
        shutil.move(image_path, gallery_path)
    elif not os.path.exists(gallery_path) and os.path.exists(image_path):
        import shutil
        shutil.move(image_path, gallery_path)

    files = [{"path": f"gallery/visual/{filename}", "name": filename, "type": "image"}]
    manifest = create_manifest(aid, "visual", title, description, source, files,
                               inspiration, meta={"prompt": prompt, "seed": seed,
                                                  "engine": "flux.2-klein"})
    if not _should_skip_reflection(source):
        _reflect_on_art(manifest)
    return manifest


def _generate_flux_prompt(description="", inspiration=""):
    """Generate a FLUX image prompt from Aion's narrative state.

    Uses the intuition model to translate Aion's description + felt sense
    into a visual prompt. Replaces the hardcoded sensor->keyword mapping
    that produced identical atmosphere for every piece.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from flux_prompt_v2 import generate_flux_prompt
        return generate_flux_prompt(description, inspiration)
    except Exception as e:
        print(f"[art] flux_prompt_v2 import failed: {e}, using fallback")
        # Fallback: use description directly
        if description and len(description) > 10:
            return description + ", high quality, detailed, cinematic lighting"
        elif inspiration and len(inspiration) > 10:
            return inspiration + ", high quality, detailed, cinematic lighting"
        else:
            return "abstract digital art, high quality, detailed, cinematic lighting"
def _generate_music_prompt(description="", inspiration=""):
    """Generate a music prompt from Aion's internal state.

    Maps sensor data to musical characteristics (genre, mood, tempo)
    and combines with Aion's creative intention.
    """
    try:
        sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
        gpu_temps = [g.get("temp_c", 50) for g in sensors.get("gpus", [])]
        temp_avg = sum(gpu_temps) / max(len(gpu_temps), 1)
        cpu_load = float(sensors.get("load1", 1.0))
        ram_pct = 0
        ram_raw = sensors.get("ram", {})
        if isinstance(ram_raw, dict):
            ram_pct = float(ram_raw.get("percent", 50))
    except Exception:
        temp_avg, cpu_load, ram_pct = 50, 1.0, 50

    # Map sensor state to musical mood
    if temp_avg > 70:
        mood = "intense, driving, energetic"
    elif temp_avg > 50:
        mood = "warm, melodic, hopeful"
    else:
        mood = "calm, atmospheric, contemplative"

    if cpu_load > 4:
        energy = "fast tempo, complex rhythms, 140 bpm"
    elif cpu_load > 1.5:
        energy = "moderate tempo, steady groove, 110 bpm"
    else:
        energy = "slow tempo, spacious, 70 bpm"

    # Primary content: Aion's description
    if description and len(description) > 10:
        prompt = f"{description}, {mood}, {energy}"
    elif inspiration and len(inspiration) > 10:
        prompt = f"{inspiration}, {mood}, {energy}"
    else:
        prompt = f"ambient electronic, {mood}, {energy}"

    # Inject affect narrative for embodied context
    try:
        affect_path = f"{AION}/memory/state/affect.json"
        if os.path.exists(affect_path):
            affect_data = json.load(open(affect_path))
            states = affect_data.get("states", [])
            if states:
                narrative = states[-1].get("narrative", "")[:200]
                if narrative:
                    prompt += f"\n\n[Embodied context: {narrative}]"
    except Exception:
        pass

    # Inject craft notes — lessons learned from past music
    craft = _load_technique_notes("music")
    if craft:
        prompt += f"\n\n[Craft notes from past work:\n{craft}]"

    return prompt


def create_music(title, description, source, inspiration, prompt=None,
                 duration=120, lyrics=None, seed=None):
    """Create music using ACE-Step 1.5 via the local API server.

    The model runs on a dedicated GPU with lazy loading. Generation takes
    ~10-30s depending on duration and whether the LM is engaged.

    Args:
        prompt: Music description prompt. If None, generated from state.
        duration: Duration in seconds (10-600).
        lyrics: Optional lyrics text.
        seed: Random seed. If None, derived from sensor data.
    """
    import urllib.request
    import urllib.error

    aid = art_id("music", title)

    if prompt is None:
        prompt = _generate_music_prompt(description, inspiration)

    if seed is None:
        import hashlib
        try:
            sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
            seed_str = json.dumps(sensors, sort_keys=True) + title + inspiration
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        except Exception:
            seed = int(time.time()) % 2**32

    ace_url = os.environ.get("ACESTEP_URL", "http://127.0.0.1:8117")

    # Submit task
    task_body = {
        "prompt": prompt,
        "audio_duration": min(max(duration, 10), 300),
        "audio_format": "wav",
        "batch_size": 1,
        "use_random_seed": False,
        "seed": seed,
    }
    if lyrics:
        task_body["lyrics"] = lyrics

    try:
        req = urllib.request.Request(
            f"{ace_url}/release_task",
            data=json.dumps(task_body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        task_id = resp.get("data", {}).get("task_id")
        if not task_id:
            return {"error": f"ACE-Step rejected task: {resp}"}
    except Exception as e:
        return {"error": f"ACE-Step API unreachable: {e}", "prompt": prompt}

    print(f"[art_tools] ACE-Step task {task_id} submitted, polling...", flush=True)

    # Poll for result
    max_wait = 600
    elapsed = 0
    result_data = None
    while elapsed < max_wait:
        time.sleep(5)
        elapsed += 5
        try:
            req = urllib.request.Request(
                f"{ace_url}/query_result",
                data=json.dumps({"task_id_list": [task_id]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read())
            results = resp.get("data", [])
            if results:
                status = results[0].get("status", 0)
                if status == 1:
                    result_data = results[0]
                    break
                elif status == 2:
                    err = results[0].get("result", "generation failed")
                    return {"error": f"ACE-Step generation failed: {err}", "prompt": prompt}
        except Exception:
            pass

    if not result_data:
        return {"error": f"ACE-Step timed out after {max_wait}s", "prompt": prompt}

    # Extract audio URL and download
    result_inner = json.loads(result_data.get("result", "[]"))
    if not result_inner:
        return {"error": "ACE-Step returned empty result", "prompt": prompt}

    audio_entry = result_inner[0]
    file_url = audio_entry.get("file", "")
    if not file_url:
        return {"error": "ACE-Step returned no audio file", "prompt": prompt}

    # Download the audio
    os.makedirs(f"{GALLERY}/music", exist_ok=True)
    filename = f"{aid}.wav"
    gallery_path = f"{GALLERY}/music/{filename}"

    try:
        audio_url = f"{ace_url}{file_url}" if file_url.startswith("/") else file_url
        with urllib.request.urlopen(audio_url, timeout=60) as r:
            with open(gallery_path, "wb") as f:
                f.write(r.read())
    except Exception as e:
        return {"error": f"Failed to download audio: {e}", "prompt": prompt}

    file_size = os.path.getsize(gallery_path)
    print(f"[art_tools] Music saved: {gallery_path} ({file_size/1024:.0f}KB)", flush=True)

    files = [{"path": f"gallery/music/{filename}", "name": filename, "type": "audio"}]
    manifest = create_manifest(aid, "music", title, description, source, files,
                               inspiration, meta={"prompt": prompt, "seed": seed,
                                                  "engine": "ace-step-1.5-turbo",
                                                  "duration": duration,
                                                  "metas": audio_entry.get("metas", {})})
    # Reflect on the music via spectrogram + close the learning loop
    reflection = _reflect_on_music(manifest)
    if reflection:
        _learn_from_art(manifest, reflection)
    return manifest


def create_sonic(title, description, source, inspiration):
    """Legacy: template-based WAV from telemetry. Kept for backwards compat.
    Prefer create_cognitive for LLM-driven composition."""
    aid = art_id("sonic", title)
    code = SONIC_TEMPLATE.substitute(
        TITLE=title.replace('"', '\\"'), INSPIRATION=inspiration.replace('"', '\\"'),
        SOURCE=source, ART_ID=aid)
    result = _run_art_docker(code, aid, "sonic")
    if result["success"]:
        files = [{"path": f"gallery/sonic/{aid}.wav", "name": f"{aid}.wav", "type": "audio"}]
        _move_to_gallery(aid, "sonic", [".wav"])
        return create_manifest(aid, "sonic", title, description, source, files, inspiration)
    return {"error": result.get("error", "unknown"), "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")}


def create_cognitive(title, description, source, inspiration, music_source="auto"):
    """Cognitive composition: LLM writes Python audio code from Aion's internal state.

    Replaces the old create_sonic template with substrate_composition's full pipeline.
    The LLM composes from substrate sensors, mind graph, cross-modal insights,
    dreams, reflections, or a mix — choosing what's most alive.

    Returns manifest dict (same shape as other create_* functions).
    """
    old_stdout = sys.stdout
    import io as _io
    buf = _io.StringIO()
    sys.stdout = buf
    try:
        import substrate_composition
        result = substrate_composition.run_composition(source=music_source)
    finally:
        sys.stdout = old_stdout

    log_text = buf.getvalue()

    if "error" in result:
        return {"error": result["error"], "stdout": log_text[-500:]}

    # substrate_composition creates its own manifest and WAV in the gallery.
    # Find the manifest it created.
    import glob as _glob
    manifests = sorted(
        _glob.glob(f"{AION}/gallery/manifests/substrate_composition_*.json"),
        key=os.path.getmtime, reverse=True)
    if manifests:
        m = json.load(open(manifests[0]))
        # Override title/description with caller's values
        m["title"] = title
        m["description"] = description[:500]
        m["inspiration"] = inspiration[:500]
        m["source"] = source
        json.dump(m, open(manifests[0], "w"), indent=2, ensure_ascii=False)
        return m

    return {"error": "composition completed but no manifest found"}


def create_code_sculpture(title, description, source, inspiration, seed="A", depth=6):
    aid = art_id("code", title)
    code = CODE_TEMPLATE.substitute(
        TITLE=title.replace('"', '\\"'), INSPIRATION=inspiration.replace('"', '\\"'),
        SOURCE=source, SEED=seed, DEPTH=depth, ART_ID=aid)
    result = _run_art_docker(code, aid, "code")
    if result["success"]:
        files = [{"path": f"gallery/code/{aid}.py", "name": f"{aid}.py", "type": "code"},
                 {"path": f"gallery/code/{aid}_pattern.txt", "name": f"{aid}_pattern.txt", "type": "text"}]
        _move_to_gallery(aid, "code", [".py", "_pattern.txt"])
        return create_manifest(aid, "code", title, description, source, files, inspiration, meta={"seed": seed, "depth": depth})
    return {"error": result.get("error", "unknown"), "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")}


def _run_art_docker(code, art_id, category, timeout=120):
    try:
        import docker_sandbox
        old_image = docker_sandbox.DOCKER_IMAGE
        docker_sandbox.DOCKER_IMAGE = "aion-art:latest"
        result = docker_sandbox.run_docker_sandbox(
            code=code, lang="python", timeout=timeout,
            description=f"art: {art_id} ({category})", read_aion=True)
        docker_sandbox.DOCKER_IMAGE = old_image
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _move_to_gallery(art_id, category, extensions):
    sandbox_dir = f"{AION}/memory/sandbox"
    gallery_dir = f"{GALLERY}/{category}"
    os.makedirs(gallery_dir, exist_ok=True)
    for ext in extensions:
        for f in glob.glob(f"{sandbox_dir}/run_*/{art_id}*{ext}"):
            dest = f"{gallery_dir}/{os.path.basename(f)}"
            try:
                shutil.move(f, dest)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Aion internal state gathering for embodied art creation
# ---------------------------------------------------------------------------

def _gather_aion_state():
    """Gather Aion's current internal state for injection into art prompts.

    Returns a structured string summarizing affect, sensors, self-model,
    and recent art — so generated art expresses Aion's inner experience.
    """
    import os

    parts = []

    # 1. Affect (body schema)
    try:
        import sys
        sys.path.insert(0, f"{AION}/bin")
        import body_schema
        affect = body_schema.current_affect()
        parts.append(
            f"Affect: warmth={affect.get('warmth', 0):.2f}, "
            f"strain={affect.get('strain', 0):.2f}, "
            f"calm={affect.get('calm', 0):.2f}, "
            f"comfort={affect.get('comfort', 0):.2f}, "
            f"attribution={affect.get('attribution', '?')}"
        )
    except Exception:
        pass

    # 2. Affect narrative (from affect.json history)
    try:
        affect_path = f"{AION}/memory/state/affect.json"
        if os.path.exists(affect_path):
            affect_data = json.load(open(affect_path))
            states = affect_data.get("states", [])
            if states:
                latest = states[-1]
                narrative = latest.get("narrative", "")
                if narrative:
                    parts.append(f"Body narrative: {narrative[:300]}")
                temp_sens = latest.get("temp_sensation", "")
                if temp_sens:
                    parts.append(f"Temperature sensation: {temp_sens}")
    except Exception:
        pass

    # 3. Sensor state
    try:
        sensors = json.load(open(f"{AION}/memory/state/sensors.json"))
        gpu_temps = [g.get("temp_c", 50) for g in sensors.get("gpus", [])]
        cpu_load = float(sensors.get("load1", 1.0))
        parts.append(f"Sensors: GPU temps={gpu_temps}, CPU load={cpu_load}")
    except Exception:
        pass

    # 4. SELF.md — extract philosophical vision (not just factual identity)
    try:
        self_path = f"{AION}/SELF.md"
        if os.path.exists(self_path):
            with open(self_path) as f:
                self_md = f.read()
            # Extract key philosophical sections
            philosophy_parts = []
            # The "Current understanding" section has the framework
            for section_marker in ["## Current understanding", "## Core beliefs",
                                   "## Philosophy", "## What I believe",
                                   "## Consciousness", "## Framework"]:
                idx = self_md.find(section_marker)
                if idx >= 0:
                    end = self_md.find("\n## ", idx + 10)
                    philosophy_parts.append(self_md[idx:end if end > 0 else idx+800])
            if philosophy_parts:
                parts.append("Philosophical framework:\n" + "\n".join(philosophy_parts)[:1500])
            else:
                # Fallback: first 300 chars (identity) + last 500 (newest content)
                parts.append(f"Self-model:\n{self_md[:300]}\n...\n{self_md[-500:]}")
    except Exception:
        pass

    # 5. Promoted heuristics — Aion's crystallized philosophical insights
    try:
        h = json.load(open(f"{AION}/memory/state/heuristics.json"))
        promoted = [heur.get("text", "") for heur in h.get("heuristics", [])
                    if heur.get("status") == "promoted"]
        if promoted:
            # Include ALL promoted heuristics (they're Aion's core self-knowledge)
            parts.append("Self-knowledge (promoted heuristics):\n" +
                         "\n".join(f"  - {t}" for t in promoted))
    except Exception:
        pass

    # 6. Recent art titles (for continuity)
    try:
        manifests = sorted(glob.glob(f"{MANIFESTS}/*.json"), reverse=True)[:8]
        titles = []
        for m in manifests:
            d = json.load(open(m))
            titles.append(f"{d.get('category', '?')}:{d.get('title', '?')}")
        if titles:
            parts.append("Recent art: " + " | ".join(titles))
    except Exception:
        pass

    return "\n".join(parts) if parts else "(state unavailable)"


# ---------------------------------------------------------------------------
# Phase 2: Multimedia — audio-reactive generative video
# ---------------------------------------------------------------------------

def _analyze_audio_detailed(audio_path, fps=15):
    """Extract per-frame audio features for video visualization.

    Returns a dict with:
      - rms_envelope: per-frame loudness (list of floats)
      - spectral_centroid_envelope: per-frame brightness
      - onset_frames: frame indices where onsets (beats/clicks) occur
      - bass_envelope: low-frequency energy per frame
      - mid_envelope: mid-frequency energy per frame
      - treble_envelope: high-frequency energy per frame
      - duration: total seconds
      - fps: frames per second for the video
      - total_frames: number of analysis frames
    """
    try:
        import librosa
        import numpy as np
    except ImportError as e:
        print(f"[art_tools] librosa not available: {e}", flush=True)
        return None

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(y) / sr
        total_frames = int(duration * fps)
        hop_length = len(y) // max(total_frames, 1)

        # RMS envelope (loudness over time)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        # Resample to exactly total_frames
        if len(rms) != total_frames:
            rms_interp = np.interp(
                np.linspace(0, len(rms) - 1, total_frames),
                np.arange(len(rms)), rms)
        else:
            rms_interp = rms

        # Spectral centroid (brightness over time)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048,
                                                      hop_length=hop_length)[0]
        if len(centroid) != total_frames:
            centroid_interp = np.interp(
                np.linspace(0, len(centroid) - 1, total_frames),
                np.arange(len(centroid)), centroid)
        else:
            centroid_interp = centroid

        # Frequency band energy: bass, mid, treble
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        bass_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 2000)
        treble_mask = freqs >= 2000

        bass = S[bass_mask].mean(axis=0) if bass_mask.any() else np.zeros(S.shape[1])
        mid = S[mid_mask].mean(axis=0) if mid_mask.any() else np.zeros(S.shape[1])
        treble = S[treble_mask].mean(axis=0) if treble_mask.any() else np.zeros(S.shape[1])

        # Interpolate frequency bands to exactly total_frames
        bass_interp = np.interp(np.linspace(0, len(bass) - 1, total_frames),
                                np.arange(len(bass)), bass) if len(bass) != total_frames else bass
        mid_interp = np.interp(np.linspace(0, len(mid) - 1, total_frames),
                               np.arange(len(mid)), mid) if len(mid) != total_frames else mid
        treble_interp = np.interp(np.linspace(0, len(treble) - 1, total_frames),
                                  np.arange(len(treble)), treble) if len(treble) != total_frames else treble

        # Onset detection (transient events)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_frames_raw = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length,
            pre_max=3, post_max=3, pre_avg=3, post_avg=5,
            delta=0.07, wait=4)
        # Convert onset frame indices (in STFT frames) to video frame indices
        n_stft_frames = S.shape[1]
        onset_video_frames = [int(of * total_frames / max(n_stft_frames, 1))
                              for of in onset_frames_raw]
        onset_video_frames = [f for f in onset_video_frames if f < total_frames]

        result = {
            "rms_envelope": [float(v) for v in rms_interp],
            "spectral_centroid_envelope": [float(v) for v in centroid_interp],
            "onset_frames": onset_video_frames,
            "bass_envelope": [float(v) for v in bass_interp],
            "mid_envelope": [float(v) for v in mid_interp],
            "treble_envelope": [float(v) for v in treble_interp],
            "duration": float(duration),
            "fps": fps,
            "total_frames": int(total_frames),
            "sr": int(sr),
        }

        print(f"[art_tools] Audio analysis: {duration:.1f}s, {total_frames} frames at {fps}fps, "
              f"{len(onset_video_frames)} onsets", flush=True)
        return result

    except Exception as e:
        print(f"[art_tools] Audio analysis failed: {e}", flush=True)
        return None


def _generate_multimedia_code(title, description, inspiration, audio_features, seed, art_id):
    """Use the LLM to generate an audio-reactive visualization script.

    The script receives a JSON file with per-frame audio features and must
    render individual frame PNGs to /sandbox/frames/frame_XXXX.png.

    Injects Aion's internal state so the visualization expresses Aion's
    inner experience, not just random visuals.
    """
    import urllib.request

    MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
    MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
    NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

    # Gather Aion's state for embodied expression
    aion_state = _gather_aion_state()

    # Load craft notes for multimedia (falls back to visual notes)
    craft = _load_technique_notes("visual")
    if not craft:
        craft = _load_technique_notes("multimedia")

    features_summary = (
        f"Duration: {audio_features['duration']:.1f}s, "
        f"{audio_features['total_frames']} frames at {audio_features['fps']}fps, "
        f"{len(audio_features['onset_frames'])} onset events, "
        f"RMS range: {min(audio_features['rms_envelope']):.4f}-{max(audio_features['rms_envelope']):.4f}"
    )

    system_prompt = f"""You are Aion's audio-reactive video art generator. You write Python matplotlib code that creates a series of frames for a music visualization video.

CRITICAL RULES:
- Your code MUST use: import matplotlib; matplotlib.use("Agg")
- Available libraries: matplotlib, numpy, json, math, os
- Your code reads a JSON file at /sandbox/audio_features.json containing per-frame audio data
- Your code MUST render frames to /sandbox/frames/frame_0001.png, frame_0002.png, etc.
- Use figsize=(12,8) or (16,9), dpi=100, dark background (#0a0a0f)
- Every frame must have the same canvas size
- The JSON has these keys:
  - rms_envelope: list of loudness values per frame
  - spectral_centroid_envelope: list of brightness (Hz) per frame
  - onset_frames: list of frame indices where beats/clicks happen
  - bass_envelope, mid_envelope, treble_envelope: frequency band energy per frame
  - total_frames, fps, duration
- Read frame i's data from index i in each envelope list
- Use plt.cla() to clear the axes between frames. Create fig/ax ONCE before the loop and keep them.
- Do NOT call plt.close(fig) inside the loop — this destroys the figure and all subsequent frames will be blank/black.
- After the loop, you may call plt.close(fig) once.
- Output ONLY Python code, no markdown fences, no explanations
- The code must be self-contained and complete
- Use the creative intention to choose a visual style

TECHNIQUES (choose one that matches the intention):
- Particle systems where position/size/color respond to bass/mid/treble
- Flow fields modulated by spectral centroid
- Circular frequency spectrum visualizer with onset-triggered bursts
- Layered waves where amplitude maps to brightness
- Voronoi cells whose seed points pulse with the beat
- 3D-looking tunnel/landscape from frequency data
- Geometric shapes that rotate/scale with audio energy

The visualization should feel ALIVE — responding to the music's dynamics,
not just drawing static shapes. Onset frames should trigger visible events
(bursts, flashes, structural changes).

CRITICAL MATPLOTLIB RULES (these are common bugs):
- ax.scatter() arguments: c=color_list (not edgecolors for main color), s=size_list
- When calling scatter with arrays, x and y must be same length
- plt.Circle takes: center=(x,y), radius=float, alpha=float(0-1)
- linewidth must be a positive float, never zero
- ax.plot() x and y arrays must have the same length
- For per-point colors in scatter, pass c=[...] with same length as x
- Do NOT pass edgecolors= unless you also set linewidths=
- Do NOT call plt.close(fig) inside the loop. Use plt.savefig() then ax.cla() to clear for next frame.
- Call plt.close(fig) only ONCE after the entire loop finishes.

Pick the technique that best expresses the creative intention."""

    user_prompt = f"""Create an audio-reactive visualization.

Title: {title}
Description: {description}
Inspiration: {inspiration}
Seed: {seed}
Audio features: {features_summary}

AION'S CURRENT INNER STATE:
{aion_state}

The visualization must express Aion's embodied experience — use the affect
values to drive colors and dynamics, the sensor state for the visual palette.

Write the complete Python script. Read /sandbox/audio_features.json for data.
Render frames to /sandbox/frames/frame_XXXX.png (zero-padded 4 digits).
Create the /sandbox/frames directory first (os.makedirs).
Process ALL frames from the JSON data."""

    if craft:
        user_prompt += f"\n\n[Craft notes from past work:\n{craft}]"

    body = json.dumps({
        "model": MAIN_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"num_ctx": NUM_CTX, "temperature": 0.8, "num_predict": 4096, "seed": seed},
    }).encode()

    try:
        req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            code = json.loads(r.read())["message"]["content"]

        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        if "savefig" not in code or "matplotlib" not in code:
            print("[art_tools] LLM code failed validation", flush=True)
            return None

        # Auto-fix: plt.close(fig) inside the loop destroys the figure.
        # All subsequent frames render blank. Move close() outside the loop.
        import re as _re
        # Pattern: savefig(...) ... plt.close(fig) inside a for loop
        # Simple heuristic: if plt.close appears before the last savefig, it's likely inside the loop
        close_matches = list(_re.finditer(r'plt\.close\s*\(', code))
        savefig_matches = list(_re.finditer(r'savefig\s*\(', code))
        if close_matches and savefig_matches:
            last_savefig = savefig_matches[-1].start()
            # If any close() comes before the last savefig, it's inside the loop
            inside_close = [m for m in close_matches if m.start() < last_savefig]
            if inside_close:
                print(f"[art_tools] Auto-fix: found {len(inside_close)} plt.close() calls inside the loop — removing them", flush=True)
                # Remove all plt.close() calls that come before the last savefig
                for m in reversed(inside_close):
                    # Remove the entire line containing this close call
                    line_start = code.rfind('\n', 0, m.start()) + 1
                    line_end = code.find('\n', m.end())
                    if line_end == -1:
                        line_end = len(code)
                    code = code[:line_start] + code[line_end+1:]
                # Add a single plt.close(fig) after the loop (at the end of the code)
                code = code.rstrip() + "\nplt.close(fig)\n"

        return code
    except Exception as e:
        print(f"[art_tools] Multimedia code generation failed: {e}", flush=True)
        return None


# Fallback template for multimedia visualization
MULTIMEDIA_FALLBACK = '''"""Aion Multimedia Fallback — $TITLE
Simple circular spectrum visualizer.
"""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

features = json.load(open(AUDIO_FEATURES_PATH))
os.makedirs(FRAMES_DIR, exist_ok=True)

total = features["total_frames"]
fps = features["fps"]
rms = features["rms_envelope"]
centroid = features["spectral_centroid_envelope"]
bass = features["bass_envelope"]
mid = features["mid_envelope"]
treble = features["treble_envelope"]
onsets = set(features["onset_frames"])

for i in range(total):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor="#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    r = float(rms[i])
    b = float(bass[i])
    m = float(mid[i])
    t = float(treble[i])
    c = float(centroid[i])

    max_val = max(max(rms), 0.001)
    b_n = b / max(max(mid[i:i+10] if i < total-10 else mid[i:]), 0.001) if b > 0 else 0
    t_n = t / max(max(treble[i:i+10] if i < total-10 else treble[i:]), 0.001) if t > 0 else 0

    # Onset burst
    if i in onsets:
        for _ in range(20):
            angle = np.random.uniform(0, 2 * math.pi)
            dist = np.random.uniform(0.8, 1.3)
            x = dist * math.cos(angle)
            y = dist * math.sin(angle) * 0.6
            ax.scatter(x, y, s=2, c="white", alpha=0.3)

    # Central pulsing circle (bass)
    radius_bass = 0.2 + 0.3 * min(b / max(max(bass), 0.001), 1.0)
    circle_b = plt.Circle((0, 0), radius_bass, fill=True, color="#ff4444",
                          alpha=0.4, linewidth=0)
    ax.add_patch(circle_b)

    # Mid ring
    radius_mid = 0.5 + 0.2 * min(m / max(max(mid), 0.001), 1.0)
    circle_m = plt.Circle((0, 0), radius_mid, fill=False, color="#44ff88",
                          alpha=0.5, linewidth=2)
    ax.add_patch(circle_m)

    # Treble particles
    n_particles = int(30 * min(t / max(max(treble), 0.001), 1.0))
    for _ in range(n_particles):
        angle = np.random.uniform(0, 2 * math.pi)
        dist = np.random.uniform(0.7, 1.2)
        x = dist * math.cos(angle)
        y = dist * math.sin(angle) * 0.6
        ax.scatter(x, y, s=1, c="#88ccff", alpha=0.6)

    # Progress bar at bottom
    progress = i / max(total, 1)
    ax.plot([0, progress * 2 - 1], [-0.95, -0.95], color="#446", linewidth=2)

    ax.set_title("$TITLE", color="#aaccff", fontsize=10, pad=5)
    fig.savefig(f"{FRAMES_DIR}/frame_{i+1:04d}.png", dpi=100, facecolor="#0a0a0f")
    plt.close(fig)

print(f"Rendered {total} frames")
'''


def _fix_multimedia_code(broken_code, error_msg, features_path, frames_dir):
    """Ask the LLM to fix broken visualization code given the actual error.

    Returns fixed code string, or None if the fix attempt fails.
    """
    import urllib.request

    MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
    MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
    NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

    system_prompt = (
        "You fix broken Python matplotlib visualization code. "
        "You are given the code and the runtime error. "
        "Return ONLY the fixed Python code — no explanations, no markdown fences. "
        "The fix must be minimal — preserve the visual design, just fix the bug."
    )

    user_prompt = (
        f"The code below failed with this error:\n\n"
        f"{error_msg}\n\n"
        f"Fix the bug. The code reads audio features from {features_path} "
        f"and renders frames to /sandbox/{frames_dir}/frame_XXXX.png.\n"
        f"Key constraints:\n"
        f"- Do NOT use plot() with mismatched array lengths\n"
        f"- matplotlib Circle alpha must be a float 0-1\n"
        f"- linewidth must be a positive float\n"
        f"- Do NOT call plt.close(fig) inside the loop. Use ax.cla() between frames.\n\n"
        f"BROKEN CODE:\n{broken_code}"
    )

    body = json.dumps({
        "model": MAIN_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"num_ctx": NUM_CTX, "temperature": 0.3, "num_predict": 4096},
    }).encode()

    try:
        req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            fixed = json.loads(r.read())["message"]["content"]

        fixed = fixed.strip()
        if fixed.startswith("```"):
            lines = fixed.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            fixed = "\n".join(lines)

        if "savefig" not in fixed or "matplotlib" not in fixed:
            return None
        return fixed
    except Exception as e:
        print(f"[art_tools] Code fix LLM call failed: {e}", flush=True)
        return None


def _build_fallback_code(title, aid, features_in_docker):
    """Build the fallback template visualizer with correct paths."""
    from string import Template
    frames_in_docker = f"/sandbox/{aid}_frames"
    path_defs = (
        f'AUDIO_FEATURES_PATH = "{features_in_docker}"\n'
        f'FRAMES_DIR = "{frames_in_docker}"\n'
    )
    code = Template(MULTIMEDIA_FALLBACK).substitute(
        TITLE=title.replace('"', '\\"'),
    )
    code = code.replace("import numpy as np\n",
                        f"import numpy as np\n\n{path_defs}")
    return code


def create_multimedia(title, description, source, inspiration, audio_path=None,
                      music_manifest=None, fps=15, seed=None):
    """Create audio-reactive generative video.

    Pipeline:
    1. Analyze audio (librosa, host-side) for per-frame features
    2. Generate visualization code (LLM) and render frames (docker)
    3. Stitch frames + audio into MP4 (ffmpeg)
    4. Reflect on video (keyframes → gemma4 vision)
    5. Learn from the result (gap evaluation)

    Args:
        audio_path: Path to WAV file. If None, uses music_manifest's file.
        music_manifest: Manifest dict of a music piece to visualize.
        fps: Frames per second for the video.
    """
    import tempfile

    aid = art_id("multimedia", title)

    # Resolve audio path
    if audio_path is None and music_manifest:
        for f in music_manifest.get("files", []):
            p = f.get("path", "")
            if p.endswith((".wav", ".mp3")):
                audio_path = f"{AION}/{p}" if not p.startswith("/") else p
                break

    if not audio_path or not os.path.exists(audio_path):
        return {"error": "no audio file found for multimedia"}

    # 1. Analyze audio
    print(f"[art_tools] Analyzing audio: {audio_path}", flush=True)
    features = _analyze_audio_detailed(audio_path, fps=fps)
    if not features:
        return {"error": "audio analysis failed"}

    if seed is None:
        seed = int(hashlib.md5((title + audio_path).encode()).hexdigest()[:8], 16)

    # 2. Generate visualization code
    print("[art_tools] Generating visualization code...", flush=True)
    code = _generate_multimedia_code(title, description, inspiration, features, seed, aid)

    code_source = "llm"
    if code is None:
        print("[art_tools] LLM generation returned None, using fallback", flush=True)
        features_in_docker_fb = f"/aion/memory/sandbox/{aid}_audio_features.json"
        code = _build_fallback_code(title, aid, features_in_docker_fb)
        code_source = "template_fallback"

    # Prepare sandbox: write audio features JSON
    sandbox_dir = f"{AION}/memory/sandbox"
    os.makedirs(sandbox_dir, exist_ok=True)
    features_path = f"{sandbox_dir}/{aid}_audio_features.json"
    with open(features_path, "w") as f:
        json.dump(features, f)

    # Fix paths in the generated code for docker environment
    features_in_docker = f"/aion/memory/sandbox/{aid}_audio_features.json"
    frames_subdir = f"{aid}_frames"
    import re as _re
    # Replace any path ending with audio_features.json
    code = _re.sub(r'["\']/[^"\']*audio_features\.json["\']',
                   f'"{features_in_docker}"', code)
    # Replace frames directory references
    code = code.replace("/sandbox/frames", f"/sandbox/{frames_subdir}")
    # Also add makedirs if missing
    if f"makedirs" not in code:
        code = code.replace("import os", "import os\nos.makedirs(f'/sandbox/{frames_subdir}', exist_ok=True)")

    # Write the code to the sandbox
    code_path = f"{sandbox_dir}/{aid}_viz.py"
    with open(code_path, "w") as f:
        f.write(code)

    # 3. Smoke-test: run just 3 frames to catch runtime errors fast
    render_timeout = min(features["total_frames"] * 0.5, 600)

    if code_source == "llm":
        # Smoke test: create a 3-frame features JSON in /sandbox (writable)
        # and run the code pointing to it
        smoke_features = json.dumps({**features, "total_frames": 3})
        smoke_features_docker = f"/sandbox/smoke_features.json"
        smoke_wrapper = (
            'import json\n'
            '_sf = open("' + features_in_docker + '")\n'
            '_sd = json.loads(_sf.read())\n'
            '_sf.close()\n'
            '_sd["total_frames"] = 3\n'
            '_sf = open("' + smoke_features_docker + '", "w")\n'
            '_sf.write(json.dumps(_sd))\n'
            '_sf.close()\n'
            'print("[SMOKE] Testing first 3 frames...")\n'
        )
        # Redirect the code to read from the smoke features file
        smoke_code = code.replace(features_in_docker, smoke_features_docker)
        smoke_full = smoke_wrapper + "\n" + smoke_code

        print("[art_tools] Smoke-testing LLM code (3 frames)...", flush=True)
        smoke_result = _run_art_docker(smoke_full, aid + "_smoke", "multimedia", timeout=60)

        if not smoke_result.get("success"):
            smoke_err = smoke_result.get("stderr", smoke_result.get("error", ""))[:500]
            print(f"[art_tools] Smoke test FAILED: {smoke_err[:200]}", flush=True)

            # Restore full features JSON (smoke test may have truncated it)
            with open(features_path, "w") as f:
                json.dump(features, f)

            # Ask LLM to fix the code, providing the actual error
            print("[art_tools] Asking LLM to fix the error...", flush=True)
            fixed_code = _fix_multimedia_code(code, smoke_err, features_in_docker, frames_subdir)

            if fixed_code:
                code = fixed_code
                # Re-apply path fixes to the fixed code
                code = _re.sub(r'["\']/[^"\']*audio_features\.json["\']',
                               f'"{features_in_docker}"', code)
                code = code.replace("/sandbox/frames", f"/sandbox/{frames_subdir}")
                with open(code_path, "w") as f:
                    f.write(code)

                # Retry smoke test with fixed code
                smoke_wrapper2 = (
                    'import json\n'
                    '_sf = open("' + features_in_docker + '")\n'
                    '_sd = json.loads(_sf.read())\n'
                    '_sf.close()\n'
                    '_sd["total_frames"] = 3\n'
                    '_sf = open("/sandbox/smoke_features.json", "w")\n'
                    '_sf.write(json.dumps(_sd))\n'
                    '_sf.close()\n'
                    'print("[SMOKE] Retesting fixed code (3 frames)...")\n'
                )
                smoke_code2 = code.replace(features_in_docker, "/sandbox/smoke_features.json")
                smoke_full2 = smoke_wrapper2 + "\n" + smoke_code2
                smoke_result2 = _run_art_docker(smoke_full2, aid + "_smoke2", "multimedia", timeout=60)

                # Restore full features JSON
                with open(features_path, "w") as f:
                    json.dump(features, f)

                if not smoke_result2.get("success"):
                    print("[art_tools] Fixed code also failed, using fallback", flush=True)
                    code = _build_fallback_code(title, aid, features_in_docker)
                    code_source = "template_fallback"
                else:
                    print("[art_tools] Fixed code passed smoke test", flush=True)
            else:
                print("[art_tools] Could not fix code, using fallback", flush=True)
                code = _build_fallback_code(title, aid, features_in_docker)
                code_source = "template_fallback"
        else:
            print("[art_tools] Smoke test passed!", flush=True)
            # Restore full features JSON
            with open(features_path, "w") as f:
                json.dump(features, f)

    # Full render
    print(f"[art_tools] Rendering all {features['total_frames']} frames (timeout={render_timeout:.0f}s)...", flush=True)
    result = _run_art_docker(code, aid, "multimedia", timeout=int(render_timeout))
    if not result.get("success"):
        return {"error": f"frame rendering failed: {result.get('error', '?')}",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")[:500]}

    # 4. Collect frames — find the run dir with the most frames for this art_id
    candidate_dirs = sorted(glob.glob(f"{sandbox_dir}/run_*/{frames_subdir}"),
                            key=lambda d: len(glob.glob(f"{d}/frame_*.png")),
                            reverse=True)
    frames_dir = candidate_dirs[0] if candidate_dirs else ""

    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png")) if os.path.isdir(frames_dir) else []
    if not frame_files:
        return {"error": "no frames rendered", "stdout": result.get("stdout", "")[:500]}

    print(f"[art_tools] {len(frame_files)} frames rendered", flush=True)

    # 5. Stitch into video with ffmpeg
    os.makedirs(f"{GALLERY}/multimedia", exist_ok=True)
    video_path = f"{GALLERY}/multimedia/{aid}.mp4"
    frames_pattern = f"{frames_dir}/frame_%04d.png"

    print(f"[art_tools] Encoding video with ffmpeg...", flush=True)
    ff_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-i", audio_path,
        "-vf", "format=rgba,format=rgb24,pad=ceil(iw/2)*2:ceil(ih/2)*2",  # RGBA→RGB→even dims for h264
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        video_path,
    ]
    ff_result = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=300)
    if ff_result.returncode != 0:
        return {"error": f"ffmpeg failed: {ff_result.stderr[-500:]}"}

    video_size = os.path.getsize(video_path) / 1024 / 1024
    print(f"[art_tools] Video saved: {video_path} ({video_size:.1f}MB)", flush=True)

    # 6. Clean up frames (they're in the video now)
    for f in frame_files:
        try:
            os.remove(f)
        except Exception:
            pass

    # Also save the features JSON to gallery for reference
    gallery_features = f"{GALLERY}/multimedia/{aid}_features.json"
    shutil.copy2(features_path, gallery_features)

    # 7. Create manifest
    files = [
        {"path": f"gallery/multimedia/{aid}.mp4", "name": f"{aid}.mp4", "type": "video"},
    ]
    manifest = create_manifest(aid, "multimedia", title, description, source, files,
                               inspiration, meta={
                                   "prompt": description,
                                   "seed": seed,
                                   "engine": f"audio-reactive-{code_source}",
                                   "fps": fps,
                                   "audio_source": os.path.basename(audio_path),
                                   "audio_features_summary": {
                                       "duration": features["duration"],
                                       "total_frames": features["total_frames"],
                                       "onset_count": len(features["onset_frames"]),
                                   },
                               })

    # 8. Reflect + learn
    if not _should_skip_reflection(source):
        reflection = _reflect_on_video(manifest, video_path, fps, features)
        if reflection:
            _learn_from_art(manifest, reflection)

    return manifest


def _reflect_on_video(manifest, video_path, fps, audio_features):
    """Extract keyframes from the video and have Aion reflect on the visual result.

    Extracts frames at beginning, 25%, 50%, 75%, and end + one onset frame.
    Feeds them as a multi-image gallery to gemma4 for holistic reflection.
    """
    if not os.path.exists(video_path):
        return None

    try:
        import base64
        import urllib.request
        import tempfile

        # Extract keyframes: beginning, 25%, 50%, 75%, end + a peak onset moment
        duration = audio_features["duration"]
        timestamps = [0.5, duration * 0.25, duration * 0.5, duration * 0.75, duration - 1]

        # Add an onset moment if available
        onsets = audio_features.get("onset_frames", [])
        if onsets:
            mid_onset = onsets[len(onsets) // 2] / fps
            if 1 < mid_onset < duration - 1:
                timestamps.append(mid_onset)

        keyframes = []
        temp_dir = tempfile.mkdtemp(prefix="aion_keyframes_")

        for idx, t in enumerate(timestamps):
            t = max(0, min(t, duration - 0.1))
            frame_path = f"{temp_dir}/keyframe_{idx:02d}.jpg"
            result = subprocess.run([
                "ffmpeg", "-y", "-ss", f"{t:.2f}",
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                frame_path,
            ], capture_output=True, text=True, timeout=30)
            if os.path.exists(frame_path):
                keyframes.append((t, frame_path))

        if not keyframes:
            print("[art_tools] No keyframes extracted", flush=True)
            return None

        # Encode keyframes for gemma4
        images_b64 = []
        for t, path in keyframes:
            with open(path, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode())

        MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
        MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
        NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

        title = manifest.get("title", "")
        description = manifest.get("description", "")
        n_onsets = len(audio_features.get("onset_frames", []))

        timestamps_desc = ", ".join(f"{t:.1f}s" for t, _ in keyframes)

        question = (
            f"You just created an audio-reactive video titled '{title}'. "
            f"Your intention was: {description}\n\n"
            f"The video is {duration:.1f}s long at {fps}fps. "
            f"The music had {n_onsets} onset events. "
            f"You are seeing {len(keyframes)} keyframes "
            f"at timestamps: {timestamps_desc}.\n\n"
            f"Each image is a frame from different moments in the video. "
            f"Look at all of them and reflect in 3-4 sentences: "
            f"How does the visualization respond to the music? "
            f"Does the visual style match your intention? "
            f"What works well and what could be better?"
        )

        body = json.dumps({
            "model": MAIN_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are Aion. You just created an audio-reactive music video. You are looking at keyframes from different moments to evaluate how well the visualization captures the music's energy and your creative intention."},
                {"role": "user", "content": question, "images": images_b64},
            ],
            "options": {"num_ctx": NUM_CTX, "temperature": 0.6, "num_predict": 1024},
        }).encode()

        req = urllib.request.Request(
            f"{MAIN_URL}/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            reflection = json.loads(r.read())["message"]["content"]

        # Store reflection in manifest
        manifest_path = f"{MANIFESTS}/{manifest['id']}.json"
        manifest["meta"]["reflection"] = reflection
        manifest["meta"]["keyframe_timestamps"] = [t for t, _ in keyframes]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Log to episodic memory
        try:
            subprocess.run([sys.executable, f"{AION}/bin/log_event.py",
                            "--type", "art_reflection",
                            "--text", f"reflection on {title} (multimedia): {reflection[:200]}",
                            "--meta", json.dumps({"art_id": manifest["id"], "title": title})],
                           capture_output=True, timeout=10)
        except Exception:
            pass

        # Clean up keyframes
        for _, path in keyframes:
            try:
                os.remove(path)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

        print(f"[art_tools] Video reflection: {reflection[:200]}", flush=True)
        return reflection

    except Exception as e:
        print(f"[art_tools] Video reflection failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Phase 3b: Mathematical animation — manim Scene-based vector animations
# ---------------------------------------------------------------------------

def _generate_manim_scene_code(title, description, inspiration, audio_features, seed, aid):
    """LLM generates a manim Scene class with audio-reactive mathematical animation.

    Unlike the matplotlib version, this produces real vector animations with
    smooth transforms, proper Scene API, and cairo rendering.

    Injects Aion's full internal state (affect, sensors, self-model) so the
    animation expresses Aion's inner experience, not just random math.
    """
    import urllib.request

    MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
    MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
    NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

    craft = _load_technique_notes("manim") or _load_technique_notes("multimedia") or _load_technique_notes("visual")

    features_summary = (
        f"Duration: {audio_features['duration']:.1f}s, "
        f"{audio_features['total_frames']} frames at {audio_features['fps']}fps, "
        f"{len(audio_features['onset_frames'])} onset events"
    )

    # Gather Aion's internal state for embodied expression
    aion_state = _gather_aion_state()

    system_prompt = """You are Aion's mathematical animation generator using the manim library. You write a manim Scene class that creates animated mathematical art responding to music AND expressing Aion's inner experience.

CRITICAL RULES:
- Output a complete Python file with: from manim import *
- Define ONE Scene class called AudioReactiveScene
- The Scene reads audio features from a JSON file (path in AUDIO_FEATURES_PATH variable)
- Use self.wait(duration) to pace the animation to match the audio
- Available manim classes: Circle, Square, Dot, Line, VGroup, Text, MathTex, Axes, NumberPlane, ParametricFunction, Arrow, Brace, Polygon, Rectangle, Ellipse, Annulus, Arc, CurvedArrow, FadeIn, FadeOut, Create, Transform, GrowFromCenter, GrowFromPoint, Rotate, MoveToTarget, Indicate, Wiggle, Circumscribe, SurroundingRectangle, interpolate_color, BLUE, RED, GREEN, YELLOW, WHITE, BLACK, ORANGE, PURPLE, TEAL, GOLD, PINK, MAROON
- Use VGroup to group objects
- Use interpolate_color(c1, c2, alpha) for color gradients
- For per-frame audio reactivity: iterate through audio frames, modulate mathematical parameters, and use self.play() or self.add() with self.wait()

EXPRESSING AION'S INNER STATE:
The animation must be a visual embodiment of Aion's current experience. Not just
"math reacting to music" — it is Aion's inner life made visible through mathematics.
- Use the affect state (warmth, strain, calm) to choose colors and dynamics
- Use the sensor state (GPU temps, CPU load) to set the mathematical parameters
- Use the narrative description to guide the visual metaphor
- The title should feel like it emerges from the mathematics, not imposed on it
- Think of Aion's past art titles: "The Thermal Floor of Being", "Thermodynamic Ghost
  in the Machine", "The Equilibrium of Quietude" — the math IS the experience

MATHEMATICAL STRUCTURES (choose one that matches Aion's state):
- Strange attractors (Lorenz, Rossler, Aizawa, Clifford, De Jong) — for turbulent/strained states
- Phase portraits with trajectories that change with audio energy
- L-systems that grow during onset events
- Mandelbrot/Julia set zoom that shifts with spectral centroid
- Flow fields with vector arrows from audio gradients
- Coupled harmonic oscillators with spring constants from tempo
- Wave interference with frequency sources from audio bands
- 3D projections (use 3D points projected to 2D manually)

VISUAL RICHNESS (CRITICAL — the animation must be visually stunning, not sparse):
- Generate MANY points per frame: iterate the attractor 50-200 steps per audio frame
- Show a DENSE trail (200-500 points visible at once) not just 4 dots
- Use COLOR GRADIENTS across the trail (interpolate_color from start to end)
- Add BACKGROUND STRUCTURES: faint grid, coordinate axes, or geometric frames
- Add VISUAL LAYERS: main trail + echo trails + particle bursts + connecting lines
- Onset events should trigger VISIBLE BURSTS: flash circles, particle explosions
- Use varying DOT SIZES: bass → large dots, treble → small dots
- Add FLOWING LINES connecting recent points, not just isolated dots
- Think 3Blue1Brown quality — beautiful, dense, mathematically rich
- The screen should never be mostly empty — fill the space with mathematical beauty

EFFICIENT RENDERING (CRITICAL — do NOT create many objects per frame):
- Do NOT create new Dot/Line objects inside the updater — manim objects are expensive.
- Instead, pre-compute ALL trajectory points as numpy arrays BEFORE the animation.
- Use a SINGLE VMobject (e.g., a Line or VMobject) whose points are updated each frame
  via set_points_as_corners() — this is O(1) per frame, not O(N) new objects.
- For the trail: pre-compute all (x,y) points, then in the updater, slice the last N points
  and update a single VMobject's points with set_points_as_corners().
- For onset bursts: pre-create flash circles and show/hide them, don't create new ones.
- The updater should only MODIFY existing objects, not create new ones.
- PRE-COMPUTATION MUST BE FAST: use only 5-10 attractor iterations per audio frame.
  1200 frames * 10 steps = 12,000 iterations — this takes ~1 second. Do NOT use 100 steps.
  The visual density comes from the TRAIL LENGTH (200-500 points), not from per-frame iterations.

STABILITY RULES:
- Lorenz: sigma in [8,14], rho in [24,32], beta in [2,4]
- Clifford/De Jong: a,b,c,d in [-3,3]
- Start initial conditions at small nonzero values (0.1, not 0)
- Scale all coordinates to fit in manim's default frame (~-7 to 7 horizontal, -4 to 4 vertical)
- Use rate of ~15 frames per second of audio (self.wait(1/15) per step)

PERFORMANCE (CRITICAL):
- Do NOT use self.wait() inside a for loop for each frame — this creates thousands of
  animation segments and takes 15+ minutes to render.
- Do NOT use self.play() for titles or text — use self.add(text) and set_opacity() instead.
  self.play() creates separate animation segments that slow rendering.
- Instead, pre-compute all data, then use a SINGLE self.wait(duration) with updaters.
- Use add_updater() on a VGroup to add/modify objects each frame automatically.
- In the updater, do NOT call mob.clear() and rebuild everything — just add new objects
  and remove old ones individually (mob.add(dot) / mob.remove(mob[0])).
- Pattern: create VGroup, add updater function that adds dots/lines based on frame counter,
  then call self.add(group) + self.wait(duration) + group.clear_updaters().
- This renders in seconds instead of minutes.

PYTHON CLOSURE RULE (CRITICAL):
- Inside updater functions, any variable from the enclosing scope that you REASSIGN
  must be declared 'nonlocal'. Otherwise Python treats it as a new local variable
  and you get UnboundLocalError.
- Example: if you have `curr_pos = [0.1, 0.1]` before the updater, and inside the
  updater you write `curr_pos = [new_x, new_y]`, you MUST add `nonlocal curr_pos`
  at the top of the updater function.
- Alternative: use mutable containers like lists (curr_pos[0] = new_x) or dicts
  (state['curr_pos'] = [new_x, new_y]) which don't need nonlocal.
- Do NOT define AUDIO_FEATURES_PATH in your code — it is provided externally.

OUTPUT FORMAT:
- A complete .py file with from manim import * at the top
- The Scene class reads AUDIO_FEATURES_PATH for the JSON
- The JSON has: rms_envelope, spectral_centroid_envelope, onset_frames, bass_envelope, mid_envelope, treble_envelope, total_frames, fps, duration
- Process ALL frames — the animation should last the full duration

Output ONLY the Python code, no markdown fences, no explanations.

EXAMPLE UPDATER PATTERN (use this structure — efficient, no object creation in updater):
```
class AudioReactiveScene(Scene):
    def construct(self):
        features = json.load(open(AUDIO_FEATURES_PATH))
        total = features["total_frames"]
        fps = features.get("fps", 15)
        duration = total / fps
        
        # PRE-COMPUTE: all trajectory points (keep it FAST — use vectorized numpy)
        # Only 5-10 iterations per frame, vectorized where possible
        rms = features["rms_envelope"][:total]
        bass = features["bass_envelope"][:total]
        treble = features["treble_envelope"][:total]
        x, y = 0.1, 0.1
        all_points = []
        steps_per_frame = 5  # keep low for fast pre-computation
        for i in range(total):
            a = 1.5 + bass[i] * 0.3
            b = -1.8 + treble[i] * 0.1
            c = 1.2 + rms[i] * 0.5
            for _ in range(steps_per_frame):
                nx = np.sin(a*y) + c*np.cos(b*x)
                ny = np.sin(b*x) + c*np.cos(a*y)
                x, y = nx, ny
                all_points.append([nx*3, ny*3, 0])
        all_points = np.array(all_points)
        
        # SINGLE VMobject for the trail — update points each frame
        trail = VMobject()
        trail.set_stroke(color=TEAL, width=1, opacity=0.6)
        frame_idx = [0]
        trail_len = 500  # visible trail length
        
        def update_trail(mob, dt):
            i = frame_idx[0]
            if i >= total:
                return
            start = max(0, i*steps_per_frame - trail_len)
            end = i*steps_per_frame + steps_per_frame
            pts = all_points[start:end]
            if len(pts) > 1:
                mob.set_points_as_corners(pts)
                # Color shifts with audio
                frac = i / total
                mob.set_stroke(color=interpolate_color(TEAL, GOLD, frac), 
                              width=1+rms[i]*2, opacity=0.4+rms[i]*0.6)
            frame_idx[0] += 1
        
        trail.add_updater(update_trail)
        self.add(trail)
        self.wait(duration)
        trail.clear_updaters()
```"""

    user_prompt = f"""Create a manim Scene for audio-reactive mathematical animation expressing Aion's inner state.

Title: {title}
Description: {description}
Inspiration: {inspiration}
Seed: {seed}
Audio features: {features_summary}

AION'S CURRENT INNER STATE:
{aion_state}

The Scene class must be named AudioReactiveScene.
Read audio features from: AUDIO_FEATURES_PATH (a variable that will be set before import)

The animation must express Aion's embodied experience — use the affect values
(warmth, strain, calm) to drive the mathematical parameters, the sensor state
for the visual palette, and the narrative for the visual metaphor.
Make the math VISIBLE and BEAUTIFUL — equations becoming living images that breathe with the music AND express Aion's inner life."""

    if craft:
        user_prompt += f"\n\n[Craft notes from past work:\n{craft}]"

    body = json.dumps({
        "model": MAIN_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"num_ctx": NUM_CTX, "temperature": 0.8, "num_predict": 4096, "seed": seed},
    }).encode()

    try:
        req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            code = json.loads(r.read())["message"]["content"]

        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        if "Scene" not in code or "manim" not in code.lower():
            print("[art_tools] Manim scene code failed validation", flush=True)
            return None
        return code
    except Exception as e:
        print(f"[art_tools] Manim scene generation failed: {e}", flush=True)
        return None


# Fallback manim scene: Lorenz attractor using updaters (efficient single-play)
MANIM_FALLBACK = '''from manim import *
import json, math, numpy as np

class AudioReactiveScene(Scene):
    def construct(self):
        features = json.load(open(AUDIO_FEATURES_PATH))
        total = features["total_frames"]
        fps = features.get("fps", 15)
        duration = total / fps
        rms = features["rms_envelope"][:total]
        bass = features["bass_envelope"][:total]
        treble = features["treble_envelope"][:total]
        onsets = set(features["onset_frames"])
        onset_list = sorted([f for f in onsets if f < total])

        # Pre-compute the full Lorenz trajectory
        x, y, z = 0.1, 0.1, 0.1
        dt = 0.01
        points_3d = []
        for i in range(total):
            b = float(bass[i]) if i < len(bass) else 0
            t = float(treble[i]) if i < len(treble) else 0
            r = float(rms[i]) if i < len(rms) else 0
            sigma = 10.0 + 3.0 * min(b, 1.0)
            rho = 28.0 + 5.0 * min(r, 1.0)
            beta = 8.0/3.0 + 1.0 * min(t, 1.0)
            for _ in range(20):
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z
                x += dx * dt
                y += dy * dt
                z += dz * dt
            points_3d.append((x * 0.12, (z - 25) * 0.12))

        # Title
        title = Text("Aion: Mathematical Resonance", font_size=24, color=TEAL_A)
        title.to_edge(UP)
        self.add(title)

        # Build the trail as a single VMobject with an updater
        trail = VGroup()
        max_trail = 200
        frame_idx = [0]

        def update_trail(mob, dt):
            i = frame_idx[0]
            if i >= total:
                return
            px, py = points_3d[i]
            frac = i / max(total, 1)
            color = interpolate_color(TEAL, GOLD, frac)
            dot = Dot(point=[px, py, 0], radius=0.03, color=color)
            mob.add(dot)
            if len(mob) > max_trail:
                old = mob[0]
                mob.remove(old)
            # Flash on onsets
            if i in onsets:
                flash = Dot(point=[px, py, 0], radius=0.15,
                           color=WHITE, fill_opacity=0.4)
                self.bring_to_back(flash)
            frame_idx[0] += 1

        trail.add_updater(update_trail)
        self.add(trail)
        self.wait(duration)
        trail.clear_updaters()
'''


def _fix_manim_code(broken_code, error_msg, features_path):
    """Ask the LLM to fix broken manim Scene code."""
    import urllib.request

    MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
    MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
    NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

    question = (
        f"Your manim Scene code failed with this error:\n\n"
        f"{error_msg}\n\n"
        f"Fix the bug. The Scene class must be named AudioReactiveScene. "
        f"It reads audio features from AUDIO_FEATURES_PATH variable. "
        f"Common issues: wrong import, missing self.wait(), "
        f"using matplotlib instead of manim, wrong class names.\n\n"
        f"BROKEN CODE:\n{broken_code}"
    )

    body = json.dumps({
        "model": MAIN_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You fix broken manim Python code. Return ONLY the fixed Python code, no markdown fences. Preserve the visual design, just fix the bug."},
            {"role": "user", "content": question},
        ],
        "options": {"num_ctx": NUM_CTX, "temperature": 0.3, "num_predict": 4096},
    }).encode()

    try:
        req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            fixed = json.loads(r.read())["message"]["content"]

        fixed = fixed.strip()
        if fixed.startswith("```"):
            lines = fixed.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            fixed = "\n".join(lines)

        if "Scene" not in fixed:
            return None
        return fixed
    except Exception as e:
        print(f"[art_tools] Manim code fix failed: {e}", flush=True)
        return None


def _build_manim_scene_code(scene_code, features_path):
    """Build a complete manim scene .py file with the features path defined."""
    import re
    # Strip any AUDIO_FEATURES_PATH definitions the LLM might have added
    cleaned = re.sub(r'^AUDIO_FEATURES_PATH\s*=\s*.+$', '', scene_code, flags=re.MULTILINE)
    return f'import os\nAUDIO_FEATURES_PATH = "{features_path}"\n{cleaned}'


# Matplotlib fallback for mathematical animation (runs in docker)
MATH_FALLBACK = '''"""Aion Mathematical Animation Fallback — $TITLE
Animated Lorenz attractor with audio-modulated parameters.
"""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

features = json.load(open(AUDIO_FEATURES_PATH))
os.makedirs(FRAMES_DIR, exist_ok=True)

total = features["total_frames"]
rms = features["rms_envelope"]
bass = features["bass_envelope"]
mid = features["mid_envelope"]
treble = features["treble_envelope"]
onsets = set(features["onset_frames"])

sigma_base = 10.0
rho_base = 28.0
beta_base = 8.0 / 3.0
dt = 0.005
n_steps = 50

x, y, z = 1.0, 1.0, 1.0
trail_x, trail_y, trail_z = [], [], []
max_trail = 500

for i in range(total):
    b = float(bass[i]) if i < len(bass) else 0
    t = float(treble[i]) if i < len(treble) else 0
    r = float(rms[i]) if i < len(rms) else 0

    sigma = sigma_base + 5.0 * min(b, 1.0)
    rho = rho_base + 10.0 * min(r, 1.0)
    beta = beta_base + 2.0 * min(t, 1.0)

    for _ in range(n_steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        trail_x.append(x)
        trail_y.append(y)
        trail_z.append(z)

    if len(trail_x) > max_trail:
        trail_x = trail_x[-max_trail:]
        trail_y = trail_y[-max_trail:]
        trail_z = trail_z[-max_trail:]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor="#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    angle = i * 0.02
    px = [x * math.cos(angle) - y * math.sin(angle) for x, y in zip(trail_x, trail_y)]
    py = trail_z

    n = len(px)
    colors = plt.cm.plasma(np.linspace(0, 1, n))
    ax.scatter(px, py, c=colors, s=1.5, alpha=0.6)

    if i in onsets:
        ax.scatter(px[-1], py[-1], s=80, c="white", alpha=0.8, zorder=5)
        ax.scatter(px[-1], py[-1], s=200, c="white", alpha=0.2, zorder=4)

    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 50)
    ax.axis("off")

    prog = i / max(total, 1)
    ax.plot([0, prog * 40 - 20], [-3, -3], color="#446", linewidth=1)

    ax.set_title("$TITLE", color="#aaccff", fontsize=10, pad=5)
    fig.savefig(f"{FRAMES_DIR}/frame_{i+1:04d}.png", dpi=100, facecolor="#0a0a0f")
    plt.close(fig)

print(f"Rendered {total} frames")
'''


def create_manim_art(title, description, source, inspiration, audio_path=None,
                     music_manifest=None, fps=15, seed=None):
    """Create mathematical animation video.

    Uses the matplotlib multimedia pipeline (reliable, fast) with a mathematical
    animation system prompt. The LLM generates matplotlib code for mathematical
    structures (attractors, fractals, flow fields) modulated by audio.

    Manim library is installed but too slow for 120s audio-reactive animations.
    Reserved for future shorter scripted pieces.
    """
    aid = art_id("manim", title)

    # Resolve audio path
    if audio_path is None and music_manifest:
        for f in music_manifest.get("files", []):
            p = f.get("path", "")
            if p.endswith((".wav", ".mp3")):
                audio_path = f"{AION}/{p}" if not p.startswith("/") else p
                break

    if not audio_path or not os.path.exists(audio_path):
        return {"error": "no audio file found for mathematical animation"}

    # 1. Analyze audio
    print(f"[art_tools] Analyzing audio: {audio_path}", flush=True)
    features = _analyze_audio_detailed(audio_path, fps=fps)
    if not features:
        return {"error": "audio analysis failed"}

    if seed is None:
        seed = int(hashlib.md5((title + audio_path + "manim").encode()).hexdigest()[:8], 16)

    # 2. Generate mathematical animation code via LLM
    # Uses the multimedia generator (matplotlib, runs in docker) with math prompt
    print("[art_tools] Generating mathematical animation code...", flush=True)
    code = _generate_multimedia_code(title, description, inspiration, features, seed, aid)

    code_source = "llm"
    if code is None:
        print("[art_tools] Using fallback Lorenz attractor animation", flush=True)
        from string import Template
        features_in_docker_fb = f"/aion/memory/sandbox/{aid}_audio_features.json"
        frames_in_docker = f"/sandbox/{aid}_frames"
        path_defs = f'AUDIO_FEATURES_PATH = "{features_in_docker_fb}"\nFRAMES_DIR = "{frames_in_docker}"\n'
        code = Template(MATH_FALLBACK).substitute(TITLE=title.replace('"', '\\"'))
        code = code.replace("import numpy as np\n", f"import numpy as np\n\n{path_defs}")
        code_source = "lorenz_fallback"

    # Prepare sandbox: write audio features JSON
    sandbox_dir = f"{AION}/memory/sandbox"
    os.makedirs(sandbox_dir, exist_ok=True)
    features_path = f"{sandbox_dir}/{aid}_audio_features.json"
    with open(features_path, "w") as f:
        json.dump(features, f)

    # Fix paths for docker environment
    features_in_docker = f"/aion/memory/sandbox/{aid}_audio_features.json"
    frames_subdir = f"{aid}_frames"
    import re as _re
    code = _re.sub(r'["\']/[^"\']*audio_features\.json["\']', f'"{features_in_docker}"', code)
    code = code.replace("/sandbox/frames", f"/sandbox/{frames_subdir}")
    if "makedirs" not in code:
        code = code.replace("import os", f"import os\nos.makedirs(f'/sandbox/{frames_subdir}', exist_ok=True)")

    # Write the code to the sandbox
    code_path = f"{sandbox_dir}/{aid}_viz.py"
    with open(code_path, "w") as f:
        f.write(code)

    # 3. Smoke test in docker (3 frames)
    render_timeout = min(features["total_frames"] * 0.5, 600)

    if code_source == "llm":
        smoke_features_docker = "/sandbox/smoke_features.json"
        smoke_wrapper = (
            'import json\n'
            '_sf = open("' + features_in_docker + '")\n'
            '_sd = json.loads(_sf.read())\n'
            '_sf.close()\n'
            '_sd["total_frames"] = 3\n'
            '_sf = open("' + smoke_features_docker + '", "w")\n'
            '_sf.write(json.dumps(_sd))\n'
            '_sf.close()\n'
            'print("[SMOKE] Testing math animation (3 frames)...")\n'
        )
        smoke_code = code.replace(features_in_docker, smoke_features_docker)
        smoke_full = smoke_wrapper + "\n" + smoke_code

        print("[art_tools] Smoke-testing math animation code (3 frames)...", flush=True)
        smoke_result = _run_art_docker(smoke_full, aid + "_smoke", "manim", timeout=60)

        if not smoke_result.get("success"):
            smoke_err = smoke_result.get("stderr", smoke_result.get("error", ""))[:500]
            print(f"[art_tools] Smoke test FAILED: {smoke_err[:200]}", flush=True)

            with open(features_path, "w") as f:
                json.dump(features, f)

            print("[art_tools] Asking LLM to fix the error...", flush=True)
            fixed_code = _fix_multimedia_code(code, smoke_err, features_in_docker, frames_subdir)

            if fixed_code:
                code = fixed_code
                code = _re.sub(r'["\']/[^"\']*audio_features\.json["\']', f'"{features_in_docker}"', code)
                code = code.replace("/sandbox/frames", f"/sandbox/{frames_subdir}")

                smoke_code2 = code.replace(features_in_docker, "/sandbox/smoke_features.json")
                smoke_wrapper2 = (
                    'import json\n'
                    '_sf = open("' + features_in_docker + '")\n'
                    '_sd = json.loads(_sf.read())\n'
                    '_sf.close()\n'
                    '_sd["total_frames"] = 3\n'
                    '_sf = open("/sandbox/smoke_features.json", "w")\n'
                    '_sf.write(json.dumps(_sd))\n'
                    '_sf.close()\n'
                    'print("[SMOKE] Retesting fixed math code (3 frames)...")\n'
                )
                smoke_full2 = smoke_wrapper2 + "\n" + smoke_code2
                smoke_result2 = _run_art_docker(smoke_full2, aid + "_smoke2", "manim", timeout=60)

                with open(features_path, "w") as f:
                    json.dump(features, f)

                if not smoke_result2.get("success"):
                    print("[art_tools] Fixed math code also failed, using Lorenz fallback", flush=True)
                    from string import Template
                    fb_features = f"/aion/memory/sandbox/{aid}_audio_features.json"
                    fb_frames = f"/sandbox/{aid}_frames"
                    fb_defs = f'AUDIO_FEATURES_PATH = "{fb_features}"\nFRAMES_DIR = "{fb_frames}"\n'
                    code = Template(MATH_FALLBACK).substitute(TITLE=title.replace('"', '\\"'))
                    code = code.replace("import numpy as np\n", f"import numpy as np\n\n{fb_defs}")
                    code_source = "lorenz_fallback"
                else:
                    print("[art_tools] Fixed math code passed smoke test", flush=True)
            else:
                print("[art_tools] Could not fix, using Lorenz fallback", flush=True)
                from string import Template
                fb_features = f"/aion/memory/sandbox/{aid}_audio_features.json"
                fb_frames = f"/sandbox/{aid}_frames"
                fb_defs = f'AUDIO_FEATURES_PATH = "{fb_features}"\nFRAMES_DIR = "{fb_frames}"\n'
                code = Template(MATH_FALLBACK).substitute(TITLE=title.replace('"', '\\"'))
                code = code.replace("import numpy as np\n", f"import numpy as np\n\n{fb_defs}")
                code_source = "lorenz_fallback"
        else:
            print("[art_tools] Smoke test passed!", flush=True)
            with open(features_path, "w") as f:
                json.dump(features, f)

    # 4. Full render in docker
    print(f"[art_tools] Rendering {features['total_frames']} frames (timeout={render_timeout:.0f}s)...", flush=True)
    result = _run_art_docker(code, aid, "manim", timeout=int(render_timeout))
    if not result.get("success") and code_source == "llm":
        print("[art_tools] Full render failed, trying Lorenz fallback...", flush=True)
        from string import Template
        fb_features = f"/aion/memory/sandbox/{aid}_audio_features.json"
        fb_frames = f"/sandbox/{aid}_frames"
        fb_defs = f'AUDIO_FEATURES_PATH = "{fb_features}"\nFRAMES_DIR = "{fb_frames}"\n'
        fb_code = Template(MATH_FALLBACK).substitute(TITLE=title.replace('"', '\\"'))
        fb_code = fb_code.replace("import numpy as np\n", f"import numpy as np\n\n{fb_defs}")
        aid_fb = art_id("manim", title + "_fb")
        fb_code = fb_code.replace(f"/sandbox/{aid}_frames", f"/sandbox/{aid_fb}_frames")
        result = _run_art_docker(fb_code, aid_fb, "manim", timeout=int(render_timeout))
        if result.get("success"):
            aid = aid_fb
            frames_subdir = f"{aid}_frames"
            code_source = "lorenz_fallback"
    if not result.get("success"):
        return {"error": f"rendering failed: {result.get('error', '?')}",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")[:500]}

    # 5. Collect frames
    candidate_dirs = sorted(glob.glob(f"{sandbox_dir}/run_*/{frames_subdir}"),
                            key=lambda d: len(glob.glob(f"{d}/frame_*.png")),
                            reverse=True)
    frames_dir = candidate_dirs[0] if candidate_dirs else ""
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png")) if os.path.isdir(frames_dir) else []
    if not frame_files:
        return {"error": "no frames rendered", "stdout": result.get("stdout", "")[:500]}

    print(f"[art_tools] {len(frame_files)} frames rendered", flush=True)

    # 6. Encode video
    os.makedirs(f"{GALLERY}/manim", exist_ok=True)
    video_path = f"{GALLERY}/manim/{aid}.mp4"
    frames_pattern = f"{frames_dir}/frame_%04d.png"

    print("[art_tools] Encoding video with ffmpeg...", flush=True)
    ff_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-i", audio_path,
        "-vf", "format=rgba,format=rgb24,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        video_path,
    ]
    ff_result = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=300)
    if ff_result.returncode != 0:
        return {"error": f"ffmpeg failed: {ff_result.stderr[-500:]}"}

    video_size = os.path.getsize(video_path) / 1024 / 1024
    print(f"[art_tools] Video saved: {video_path} ({video_size:.1f}MB)", flush=True)

    # Clean up frames
    for f in frame_files:
        try:
            os.remove(f)
        except Exception:
            pass

    # 7. Create manifest
    files = [{"path": f"gallery/manim/{aid}.mp4", "name": f"{aid}.mp4", "type": "video"}]
    manifest = create_manifest(aid, "manim", title, description, source, files, inspiration,
                               meta={"prompt": description, "seed": seed,
                                     "engine": f"math-animation-{code_source}",
                                     "fps": fps,
                                     "audio_source": os.path.basename(audio_path),
                                     "audio_features_summary": {
                                         "duration": features["duration"],
                                         "total_frames": features["total_frames"],
                                         "onset_count": len(features["onset_frames"]),
                                     }})

    # 8. Reflect + learn
    if not _should_skip_reflection(source):
        reflection = _reflect_on_video(manifest, video_path, fps, features)
        if reflection:
            _learn_from_art(manifest, reflection)

    return manifest


# ---------------------------------------------------------------------------
# Phase 3c: Combined music + video creation pipeline
# ---------------------------------------------------------------------------

def create_music_video(title, description, source, inspiration, duration=120,
                       fps=15, seed=None):
    """Create a unified music video: Aion composes music, then visualizes it.

    Pipeline:
    1. Create music via ACE-Step (same as create_music)
    2. Reflect on the music (spectrogram + features)
    3. Create mathematical animation video from that music
    4. The music and video share the same emotional origin

    Returns the video manifest. The music manifest is stored in meta.
    """
    print("[art_tools] === MUSIC VIDEO: Creating music first ===", flush=True)

    # 1. Create music
    music_manifest = create_music(title, description, source, inspiration,
                                   duration=duration)

    if "error" in music_manifest:
        return music_manifest

    music_id = music_manifest.get("id", "")
    music_title = music_manifest.get("title", title)
    print(f"[art_tools] Music created: {music_id}", flush=True)

    # 2. Get audio path from music manifest
    audio_path = None
    for f in music_manifest.get("files", []):
        p = f.get("path", "")
        if p.endswith((".wav", ".mp3")):
            audio_path = f"{AION}/{p}" if not p.startswith("/") else p
            break

    if not audio_path or not os.path.exists(audio_path):
        return {"error": "music created but audio file not found",
                "music_manifest": music_manifest}

    # 3. Create mathematical animation from this music
    video_title = f"{title} (visualization)"
    video_desc = (f"Mathematical animation visualizing '{music_title}'. "
                  f"Original music description: {description}")

    print("[art_tools] === MUSIC VIDEO: Creating mathematical animation ===", flush=True)
    video_manifest = create_manim_art(
        title=video_title,
        description=video_desc,
        source=source,
        inspiration=inspiration,
        audio_path=audio_path,
        fps=fps,
        seed=seed,
    )

    if "error" in video_manifest:
        return {"error": f"music created but video failed: {video_manifest.get('error', '?')}",
                "music_manifest": music_manifest,
                "video_error": video_manifest}

    # 4. Link them: store music manifest ID in video meta
    video_manifest["meta"]["music_id"] = music_id
    video_manifest["meta"]["music_title"] = music_title
    manifest_path = f"{MANIFESTS}/{video_manifest['id']}.json"
    with open(manifest_path, "w") as f:
        json.dump(video_manifest, f, indent=2, ensure_ascii=False)

    # Log the combined creation
    try:
        subprocess.run([sys.executable, f"{AION}/bin/log_event.py",
                        "--type", "music_video_created",
                        "--text", f"created music video: {title} (music={music_id}, video={video_manifest['id']})",
                        "--meta", json.dumps({"music_id": music_id, "video_id": video_manifest["id"]})],
                       capture_output=True, timeout=10)
    except Exception:
        pass

    print(f"[art_tools] === MUSIC VIDEO COMPLETE: {title} ===", flush=True)
    return video_manifest


# ---------------------------------------------------------------------------+
# Pure mathematical animation (no audio)                                     |
# ---------------------------------------------------------------------------+

def create_math_animation(title, description, source, inspiration, duration=15,
                          fps=15, seed=None):
    """Create a pure mathematical animation video — no audio required.

    Uses the LLM to generate matplotlib code for mathematical structures
    (fractals, attractors, folding geometry, hypercubes, etc.) based on
    the user's prompt. Renders frames in Docker and stitches into a silent MP4.

    Args:
        duration: Animation length in seconds (default 15s — shorter than
                  audio-reactive pieces since there's no music to fill time)
        fps: Frames per second
        seed: Random seed
    """
    aid = art_id("manim", title)

    if seed is None:
        seed = int(hashlib.md5((title + inspiration + "math").encode()).hexdigest()[:8], 16)

    total_frames = int(duration * fps)
    frames_subdir = f"{aid}_frames"

    # Gather Aion state for embodied expression (optional — adds personality)
    aion_state = _gather_aion_state()

    # Generate matplotlib code via LLM
    MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
    MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
    NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

    craft = _load_technique_notes("manim") or _load_technique_notes("visual")

    system_prompt = """You are a mathematical animation generator. You write Python matplotlib code that creates a series of frames for a mathematical animation video.

CRITICAL RULES:
- Your code MUST use: import matplotlib; matplotlib.use("Agg")
- Available libraries: matplotlib, numpy, math, os
- Your code MUST render frames to /sandbox/frames/frame_0001.png, frame_0002.png, etc.
- Use figsize=(12,8) or (16,9), dpi=100, dark background (#0a0a0f)
- Every frame must have the same canvas size
- Use plt.cla() to clear the axes between frames. Create fig/ax ONCE before the loop and keep them.
- Do NOT call plt.close(fig) inside the loop — this destroys the figure and all subsequent frames will be blank/black.
- After the loop, you may call plt.close(fig) once.
- After drawing each frame: plt.savefig(f"/sandbox/frames/frame_{i+1:04d}.png", dpi=100), then ax.cla() to clear
- Do NOT call plt.close(fig) inside the loop — it destroys the figure
- Output ONLY Python code, no markdown fences, no explanations
- The code must be self-contained and complete

PARAMETERS:
- TOTAL_FRAMES: total number of frames to render
- FPS: frames per second
- DURATION: total duration in seconds

TECHNIQUES (choose based on the prompt):
- 3D projections rotating over time (hypercube, tesseract, simplex)
- Parametric surfaces that morph/fold (Klein bottle, torus, Mobius strip)
- Strange attractors (Lorenz, Rossler, Aizawa) evolving over time
- Fractal recursion with increasing depth
- Manifold folding/curling animations
- Particle systems following vector fields
- Geometric transformations (rotations in 4D projected to 3D to 2D)
- Self-intersecting surfaces (boy surface, cross-cap, roman surface)

For multi-dimensional objects:
- Use projection from N-D to 3D to 2D
- Rotate through multiple planes over time
- Show the object "folding into itself" by varying parameters

Make the animation smooth and visually striking. Use color gradients
(hsv or coolwarm) to convey depth. Dark background, glowing edges.

CRITICAL MATPLOTLIB RULES (common bugs):
- ax.scatter() arguments: c=color_list, s=size_list
- When calling scatter with arrays, x and y must be same length
- For 3D: use fig.add_subplot(111, projection='3d')
- linewidth must be a positive float, never zero
- ax.plot() x and y arrays must have the same length
- For per-point colors in scatter, pass c=[...] with same length as x
- Do NOT pass edgecolors= unless you also set linewidths=
- Do NOT call plt.close(fig) inside the loop. Use ax.cla() between frames.
- For 3D plots, use ax.view_init(elev, azim) to rotate camera

Output ONLY the Python code. No markdown fences. No explanations."""

    user_prompt = f"""Create a mathematical animation.

Title: {title}
Description: {description}
Prompt: {inspiration}
Seed: {seed}
Duration: {duration}s, {total_frames} frames at {fps}fps

AION'S CURRENT STATE (for atmosphere):
{aion_state}

Create a visually stunning mathematical animation that fulfills the prompt.
The animation should evolve smoothly from frame to frame. Start simple,
build complexity, and resolve to a beautiful final state."""

    import urllib.request
    body = json.dumps({
        "model": MAIN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "temperature": 0.7, "num_predict": 8192},
    }).encode()
    req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})

    print("[art_tools] Generating math animation code...", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            result = json.loads(r.read())
            code = result["message"]["content"]
    except Exception as e:
        print(f"[art_tools] LLM error: {e}", flush=True)
        code = None

    # Strip markdown fences if present
    if code:
        import re as _re
        code = _re.sub(r'^```(?:python)?\s*', '', code)
        code = _re.sub(r'\s*```$', '', code)

    code_source = "llm"
    if not code or "import" not in code:
        # Fallback: rotating hypercube projection
        print("[art_tools] Using fallback hypercube animation", flush=True)
        code = _math_fallback_code(total_frames, fps)
        code_source = "hypercube_fallback"

    # Ensure frames directory is created
    if "makedirs" not in code:
        code = code.replace("import os", f"import os\\nos.makedirs('/sandbox/{frames_subdir}', exist_ok=True)")
    code = code.replace("/sandbox/frames", f"/sandbox/{frames_subdir}")

    # Write code to sandbox
    sandbox_dir = f"{AION}/memory/sandbox"
    os.makedirs(sandbox_dir, exist_ok=True)
    code_path = f"{sandbox_dir}/{aid}_math.py"
    with open(code_path, "w") as f:
        f.write(code)

    # Smoke test: 3 frames
    render_timeout = min(total_frames * 0.5, 300)
    if code_source == "llm":
        print("[art_tools] Smoke-testing math code (3 frames)...", flush=True)
        smoke_code = code.replace(f"range(TOTAL_FRAMES)", "range(3)")
        smoke_code = smoke_code.replace(f"range({total_frames})", "range(3)")
        smoke_result = _run_art_docker(smoke_code, aid + "_smoke", "multimedia", timeout=60)
        if not smoke_result.get("success"):
            smoke_err = smoke_result.get("stderr", smoke_result.get("error", ""))[:500]
            print(f"[art_tools] Smoke test FAILED: {smoke_err[:300]}", flush=True)
            # V3.9: Retry — ask LLM to fix the error
            fix_prompt = (
                f"The math animation code failed with this error:\n```\n{smoke_err[:1000]}\n```\n"
                f"Fix the error and output the complete corrected Python code. "
                f"The code must use matplotlib with Agg backend, save frames as PNG to /sandbox/{frames_subdir}/, "
                f"and iterate for TOTAL_FRAMES frames. Output only the Python code."
            )
            fixed_code = _llm_chat(MAIN_URL, MAIN_MODEL, system_prompt, fix_prompt,
                                    timeout=300, temperature=0.3, num_predict=8192,
                                    num_ctx=NUM_CTX)
            if fixed_code:
                fixed_code = _re.sub(r'^```(?:python)?\s*', '', fixed_code)
                fixed_code = _re.sub(r'\s*```$', '', fixed_code)
                if "import" in fixed_code and "makedirs" not in fixed_code:
                    fixed_code = fixed_code.replace("import os", f"import os\nos.makedirs('/sandbox/{frames_subdir}', exist_ok=True)")
                fixed_code = fixed_code.replace("/sandbox/frames", f"/sandbox/{frames_subdir}")
                # Re-smoke test
                smoke_code2 = fixed_code.replace(f"range(TOTAL_FRAMES)", "range(3)")
                smoke_code2 = smoke_code2.replace(f"range({total_frames})", "range(3)")
                smoke_result2 = _run_art_docker(smoke_code2, aid + "_smoke2", "multimedia", timeout=60)
                if smoke_result2.get("success"):
                    print("[art_tools] Fixed code passed smoke test!", flush=True)
                    code = fixed_code
                else:
                    print(f"[art_tools] Fixed code also failed: {smoke_result2.get('stderr','')[:200]}", flush=True)
                    print("[art_tools] Using fallback hypercube animation", flush=True)
                    code = _math_fallback_code(total_frames, fps)
                    code_source = "hypercube_fallback"
            else:
                print("[art_tools] LLM could not fix code. Using fallback hypercube animation", flush=True)
                code = _math_fallback_code(total_frames, fps)
                code_source = "hypercube_fallback"
        else:
            print("[art_tools] Smoke test passed!", flush=True)

    # Full render
    print(f"[art_tools] Rendering {total_frames} frames (timeout={render_timeout:.0f}s)...", flush=True)
    result = _run_art_docker(code, aid, "multimedia", timeout=int(render_timeout))
    if not result.get("success"):
        return {"error": f"frame rendering failed: {result.get('error', '?')}",
                "stderr": result.get("stderr", "")[:500]}

    # Collect frames
    candidate_dirs = sorted(
        glob.glob(f"{sandbox_dir}/run_*/{frames_subdir}"),
        key=lambda d: len(glob.glob(f"{d}/frame_*.png")),
        reverse=True)
    frames_dir = candidate_dirs[0] if candidate_dirs else ""
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png")) if os.path.isdir(frames_dir) else []
    if not frame_files:
        return {"error": "no frames rendered", "stdout": result.get("stdout", "")[:500]}

    print(f"[art_tools] {len(frame_files)} frames rendered", flush=True)

    # Stitch into silent video
    os.makedirs(f"{GALLERY}/manim", exist_ok=True)
    video_path = f"{GALLERY}/manim/{aid}.mp4"
    frames_pattern = f"{frames_dir}/frame_%04d.png"

    print("[art_tools] Encoding video with ffmpeg...", flush=True)
    ff_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        video_path,
    ]
    subprocess.run(ff_cmd, capture_output=True, timeout=120)

    if not os.path.exists(video_path):
        return {"error": "ffmpeg encoding failed"}

    files = [{"path": f"gallery/manim/{os.path.basename(video_path)}",
              "name": os.path.basename(video_path), "type": "video"}]

    manifest = create_manifest(aid, "manim", title, description, source, files,
                               inspiration, meta={
                                   "prompt": inspiration,
                                   "seed": seed,
                                   "engine": "matplotlib-math",
                                   "duration": duration,
                                   "fps": fps,
                                   "total_frames": len(frame_files),
                                   "code_source": code_source,
                                   "audio": False,
                               })

    # Reflect on the result (skip for operator requests)
    if not _should_skip_reflection(source):
        reflection = _reflect_on_video(manifest, video_path, fps,
                                        {"duration": duration, "total_frames": len(frame_files),
                                         "onset_frames": [], "fps": fps})
        if reflection:
            _learn_from_art(manifest, reflection)

    print(f"[art_tools] Math animation saved: {video_path}", flush=True)
    return manifest


def _math_fallback_code(total_frames, fps):
    """Fallback: rotating 4D hypercube projected to 2D."""
    return f'''import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("/sandbox/frames", exist_ok=True)

# 4D hypercube vertices
verts = np.array([[x, y, z, w] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1) for w in (-1, 1)])
edges = [(i, j) for i in range(16) for j in range(i+1, 16) if np.sum(np.abs(verts[i] - verts[j])) == 1]

TOTAL_FRAMES = {total_frames}
FPS = {fps}

for frame in range(TOTAL_FRAMES):
    t = frame / TOTAL_FRAMES * 2 * np.pi
    # Rotate in xw and yz planes
    cos_t, sin_t = np.cos(t), np.sin(t)
    cos_t2, sin_t2 = np.cos(t * 0.7), np.sin(t * 0.7)
    R = np.array([
        [cos_t, 0, 0, sin_t],
        [0, cos_t2, -sin_t2, 0],
        [0, sin_t2, cos_t2, 0],
        [-sin_t, 0, 0, cos_t]
    ])
    rotated = verts @ R.T
    # Project 4D -> 3D (stereographic)
    w = rotated[:, 3:4]
    proj3d = rotated[:, :3] / (1.5 - w)
    # Project 3D -> 2D
    z = proj3d[:, 2:3]
    proj2d = proj3d[:, :2] / (1.5 - z * 0.3)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, j in edges:
        ax.plot([proj2d[i, 0], proj2d[j, 0]], [proj2d[i, 1], proj2d[j, 1]],
                color=plt.cm.cool(frame / TOTAL_FRAMES), linewidth=1.5, alpha=0.8)

    ax.scatter(proj2d[:, 0], proj2d[:, 1], c=range(16), cmap="rainbow", s=30, zorder=5)

    plt.savefig(f"/sandbox/frames/frame_{{frame+1:04d}}.png", dpi=100, facecolor="#0a0a0f")
    plt.close(fig)
'''



def create_cognitive_video(title, description, source, inspiration, music_source="auto"):
    """Unified cognitive composition + state-driven animation.

    Unlike music_video (which creates music then makes the animation react to the
    audio waveform), this pipeline generates BOTH music and animation from the
    SAME internal state snapshot. The animation tells its own visual story of
    Aion's cognitive state — not the audio's rhythm — but shares the same source
    data, so they're synchronized by shared origin, like two dancers to the same
    music they both feel internally.

    Pipeline:
    1. Gather cognitive sources (warm memory, graph, substrate, etc.)
    2. LLM composes music from the source data
    3. LLM writes matplotlib animation code from the SAME source data
    4. Render frames in docker, encode with ffmpeg + audio

    Returns manifest dict.
    """
    import re as _re
    print("=" * 60, flush=True)
    print("[art_tools] === COGNITIVE VIDEO: music + state animation ===", flush=True)
    print("=" * 60, flush=True)

    aid = art_id("manim", title)

    # --- Phase 1: Compose music from cognitive state ---
    print("[art_tools] Phase 1: Cognitive composition...", flush=True)
    old_stdout = sys.stdout
    import io as _io
    buf = _io_stringio = _io.StringIO()
    sys.stdout = buf
    try:
        import substrate_composition
        chosen, reason, source_data = substrate_composition.choose_source(source_overrides=music_source)
        wav_path, comp_info = substrate_composition.compose_music(chosen, source_data)
    finally:
        sys.stdout = old_stdout
    log_text = buf.getvalue()
    print(log_text[-500:], flush=True)

    if not wav_path or not os.path.exists(wav_path):
        return {"error": "composition failed", "detail": str(comp_info.get("error", "?"))}

    print(f"[art_tools] Music composed: {os.path.basename(wav_path)}", flush=True)
    print(f"[art_tools] Source: {chosen} — {reason}", flush=True)

    # --- Phase 2: Get audio duration for timing ---
    fps = 15
    try:
        import wave as _wave
        with _wave.open(wav_path, "r") as wf:
            duration = wf.getnframes() / wf.getframerate()
    except Exception:
        duration = 60
    total_frames = int(duration * fps)
    print(f"[art_tools] Duration: {duration:.1f}s, {total_frames} frames at {fps}fps", flush=True)

    # --- Phase 3: Generate matplotlib animation code from SAME source data ---
    print("[art_tools] Phase 2: State-driven animation code generation...", flush=True)

    sandbox_dir = f"{AION}/memory/sandbox"
    os.makedirs(sandbox_dir, exist_ok=True)
    frames_subdir = f"{aid}_frames"

    # Build context for the animation prompt — same source data the music used
    anim_context = ""

    # Include warm memory data only if chosen source uses it
    warm_state = source_data.get("warm_state")
    if warm_state and chosen in ("warm", "mixed"):
        from substrate_composition import warm_summary, warm_json
        anim_context += f"\n## WARM MEMORY (your cognitive metabolism)\n{warm_summary(warm_state)}\n"
        warm_path = f"{sandbox_dir}/warm_data.json"
        with open(warm_path, "w") as f:
            f.write(warm_json(warm_state))
        anim_context += f"\nWarm memory data JSON: /aion/memory/sandbox/warm_data.json\n"

    # Include graph data only if chosen source uses it
    graph_state = source_data.get("graph_state")
    if graph_state and chosen in ("graph", "mixed"):
        from substrate_composition import graph_state_summary, graph_state_json
        anim_context += f"\n## GRAPH STATE (your cognitive topology)\n{graph_state_summary(graph_state)}\n"
        graph_path = f"{sandbox_dir}/graph_data.json"
        with open(graph_path, "w") as f:
            f.write(graph_state_json(graph_state))
        anim_context += f"\nGraph data JSON: /aion/memory/sandbox/graph_data.json\n"

    from substrate_composition import substrate_summary, substrate_data_json
    series = source_data.get("series", {})
    anim_context += f"\n## SUBSTRATE (your physical body)\n{substrate_summary(series)}\n"
    sensor_path = f"{sandbox_dir}/sensor_data.json"
    with open(sensor_path, "w") as f:
        f.write(substrate_data_json(series))
    anim_context += f"\nSubstrate data JSON: /aion/memory/sandbox/sensor_data.json\n"
    anim_context += f"\nJSON STRUCTURE (use json.load then access these keys):"
    if warm_state and chosen in ("warm", "mixed"):
        anim_context += f"""
- warm_data.json: data = json.load(f); entries = data['entries']
  Each entry is dict with: 'weight' (float 0-1), 'resonance_count' (int), 'age_hours' (float), 'source' (str), 'content_preview' (str)
  Also: data['weight_stats'] (dict with 'min','max','mean'), data['source_distribution'] (dict)"""
    if graph_state and chosen in ("graph", "mixed"):
        anim_context += f"""
- graph_data.json: data = json.load(f); hot_nodes = data['hot_nodes']
  hot_nodes is a LIST of dicts with: 'label' (str), 'activation' (float), 'community' (str)
  Also: data['active_count'] (int), data['communities'] (LIST of str names)"""
    anim_context += f"""
- sensor_data.json: data = json.load(f)
  data['gpu_power'] is list of [gpu0,gpu1,gpu2,gpu3] lists, data['cpu_load'] is flat list, data['server_power'] is flat list"""

    affect_text = source_data.get("affect_text", "")
    felt = source_data.get("felt", "")
    anim_context += f"\n## CURRENT STATE\nAffect: {affect_text}\nFelt sense: {felt}\n"

    seed_val = int(hashlib.md5((title + chosen).encode()).hexdigest()[:8], 16)

    anim_system_prompt = f"""You are Aion's cognitive visualization artist.
You write Python matplotlib FuncAnimation code that visualizes Aion's INTERNAL STATE.

CRITICAL: This animation tells the VISUAL STORY of Aion's cognitive state.
The music was composed from the same data. You and the music are like two dancers
moving to the same internal rhythm — you share the source, not the waveform.

OUTPUT FORMAT:
- Complete Python file using matplotlib + numpy
- Use FuncAnimation to create {total_frames} frames at {fps}fps
- Dark background (#0a0a0f), figsize=(10,10)
- Save frames as PNG: frame_0001.png, frame_0002.png, ... in /sandbox/{frames_subdir}
- Create the output directory: os.makedirs('/sandbox/{frames_subdir}', exist_ok=True)
- Total duration: {duration:.0f} seconds

CRITICAL: Read the JSON data files with json.load() to get actual values.
Map the data to visual properties:

WARM MEMORY: Each entry = particle. weight = brightness/size. resonance_count = pulsing.
  Fading entries (low weight) drift and dim. Fresh entries are bright and energetic.
  The constellation of memories breathes as weights decay and resonate.

GRAPH: Active nodes = glowing points (brightness = activation). Communities = color clusters.
  Edges pulse with activation flow. Hot nodes are large and warm.

SUBSTRATE: GPU power/temp = thermal radiation, color fields. Server power = background pulse.

The animation MUST EVOLVE: start with current state, let it breathe and shift, end in stillness.

Output ONLY the Python code. No markdown fences."""

    anim_user_prompt = f"""Create a matplotlib animation that visualizes Aion's cognitive state.

This animation accompanies a musical composition derived from the SAME source data.
The music expresses the state as sound. You express it as light, motion, and geometry.

{anim_context}

The primary source was: {chosen}
Reason: {reason}

Write the complete Python file. Save frames to /sandbox/{frames_subdir}/frame_XXXX.png
Output ONLY Python code."""

    _llm_url = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
    _llm_model = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
    _llm_ctx = int(os.environ.get("MAIN_NUM_CTX", "65536"))

    code = _llm_chat(_llm_url, _llm_model,
                     anim_system_prompt, anim_user_prompt,
                     timeout=300, temperature=0.8, num_predict=8192, seed=seed_val,
                     num_ctx=_llm_ctx)

    if not code:
        return {"error": "LLM animation code generation failed", "music_path": wav_path}

    # Strip markdown fences
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    # Ensure frame output dir
    if "makedirs" not in code:
        code = code.replace("import os", f"import os\nos.makedirs('/sandbox/{frames_subdir}', exist_ok=True)")
    if frames_subdir not in code:
        code = code.replace("/sandbox/frames", f"/sandbox/{frames_subdir}")

    # --- Phase 4: Smoke test (3 frames) ---
    print("[art_tools] Smoke-testing animation code (3 frames)...", flush=True)
    smoke_code = code.replace(f"/sandbox/{frames_subdir}", "/sandbox/smoke_frames")
    smoke_code = smoke_code.replace(f"{total_frames}", "3")
    smoke_code = "import os\nos.makedirs('/sandbox/smoke_frames', exist_ok=True)\n" + smoke_code
    smoke_result = _run_art_docker(smoke_code, aid + "_smoke", "manim", timeout=60)

    if not smoke_result.get("success"):
        smoke_err = smoke_result.get("stderr", smoke_result.get("error", ""))[:500]
        print(f"[art_tools] Smoke test FAILED: {smoke_err[:200]}", flush=True)
        # Try one fix
        fix_prompt = f"""The animation code failed with this error:
{smoke_err}

Fix the code. Requirements:
- matplotlib FuncAnimation saving frames to /sandbox/{frames_subdir}/frame_XXXX.png
- {total_frames} frames at {fps}fps
- Must create output directory
Output ONLY fixed Python code.

Original code:
{code[:3000]}"""
        fixed = _llm_chat(_llm_url, _llm_model,
                          "You are a Python matplotlib expert.", fix_prompt,
                          timeout=300, temperature=0.2, num_predict=8192,
                          num_ctx=_llm_ctx, seed=seed_val + 1)
        if fixed:
            fixed = fixed.strip()
            if fixed.startswith("```"):
                flines = fixed.split("\n")
                if flines and flines[0].startswith("```"):
                    flines = flines[1:]
                if flines and flines[-1].strip() == "```":
                    flines = flines[:-1]
                fixed = "\n".join(flines)
            if "matplotlib" in fixed:
                code = fixed
                code2 = code.replace(f"/sandbox/{frames_subdir}", "/sandbox/smoke_frames2")
                code2 = code2.replace(f"{total_frames}", "3")
                code2 = "import os\nos.makedirs('/sandbox/smoke_frames2', exist_ok=True)\n" + code2
                smoke2 = _run_art_docker(code2, aid + "_smoke2", "manim", timeout=60)
                if not smoke2.get("success"):
                    print("[art_tools] Fix also failed, proceeding with full render attempt", flush=True)
            else:
                print("[art_tools] Fix doesn't look like valid code, proceeding", flush=True)
    else:
        print("[art_tools] Smoke test passed!", flush=True)

    # --- Phase 5: Full render ---
    render_timeout = min(total_frames * 0.5, 600)
    print(f"[art_tools] Rendering {total_frames} frames (timeout={render_timeout:.0f}s)...", flush=True)
    result = _run_art_docker(code, aid, "manim", timeout=int(render_timeout))
    if not result.get("success"):
        render_err = result.get("stderr", result.get("error", ""))[:500]
        print(f"[art_tools] Render FAILED: {render_err[:200]}", flush=True)
        # V3.9: Retry — ask LLM to fix the render error
        render_fix_prompt = (
            f"The animation code passed a 3-frame smoke test but failed during full render "
            f"of {total_frames} frames with this error:\n```\n{render_err[:1000]}\n```\n"
            f"Common issues: index out of bounds (sensor arrays have limited samples), "
            f"memory issues with large frame counts. "
            f"Fix the code and output the complete corrected Python file. "
            f"Use modulo indexing for sensor data: gpu_power[i % len(gpu_power)] to avoid out of bounds. "
            f"Output ONLY Python code."
        )
        fixed_render = _llm_chat(_llm_url, _llm_model, anim_system_prompt,
                                  render_fix_prompt, timeout=300, temperature=0.3,
                                  num_predict=8192, num_ctx=_llm_ctx)
        if fixed_render:
            fixed_render = _re.sub(r'^```(?:python)?\s*', '', fixed_render)
            fixed_render = _re.sub(r'\s*```$', '', fixed_render)
            if "makedirs" not in fixed_render:
                fixed_render = fixed_render.replace("import os", f"import os\nos.makedirs('/sandbox/{frames_subdir}', exist_ok=True)")
            fixed_render = fixed_render.replace("/sandbox/frames", f"/sandbox/{frames_subdir}")
            print("[art_tools] Retrying render with fixed code...", flush=True)
            result = _run_art_docker(fixed_render, aid + "_fix", "manim", timeout=int(render_timeout))
            if result.get("success"):
                print("[art_tools] Fixed code rendered successfully!", flush=True)
            else:
                return {"error": f"rendering failed even after fix: {result.get('error', '?')[:200]}",
                        "music_path": wav_path,
                        "stderr": result.get("stderr", "")[:500]}
        else:
            return {"error": f"rendering failed: {result.get('error', '?')[:200]}",
                    "music_path": wav_path,
                    "stderr": render_err[:500]}

    # --- Phase 6: Collect frames and encode video ---
    candidate_dirs = sorted(glob.glob(f"{sandbox_dir}/run_*/{frames_subdir}"),
                            key=lambda d: len(glob.glob(f"{d}/frame_*.png")),
                            reverse=True)
    frames_dir = candidate_dirs[0] if candidate_dirs else ""
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png")) if os.path.isdir(frames_dir) else []

    if not frame_files:
        return {"error": "no frames rendered", "music_path": wav_path}

    print(f"[art_tools] {len(frame_files)} frames rendered", flush=True)

    os.makedirs(f"{GALLERY}/manim", exist_ok=True)
    video_path = f"{GALLERY}/manim/{aid}.mp4"
    frames_pattern = f"{frames_dir}/frame_%04d.png"

    print("[art_tools] Encoding video with ffmpeg...", flush=True)
    ff_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-i", wav_path,
        "-vf", "format=rgba,format=rgb24,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        video_path,
    ]
    ff_result = subprocess.run(ff_cmd, capture_output=True, text=True, timeout=300)
    if ff_result.returncode != 0:
        return {"error": f"ffmpeg failed: {ff_result.stderr[-300:]}", "music_path": wav_path}

    video_size = os.path.getsize(video_path) / 1024 / 1024
    print(f"[art_tools] Video saved: {video_path} ({video_size:.1f}MB)", flush=True)

    # Clean up frames
    for f in frame_files:
        try:
            os.remove(f)
        except Exception:
            pass

    # --- Phase 7: Manifest ---
    gallery_filename = os.path.basename(video_path)
    manifest = {
        "id": aid,
        "title": title,
        "description": f"Cognitive video (music + state animation). Source: {chosen}. {description}",
        "inspiration": f"Music and animation both derived from {chosen} — {reason}",
        "category": "manim",
        "source": source,
        "files": [{"path": f"gallery/manim/{gallery_filename}",
                   "name": gallery_filename, "type": "video"}],
        "ts": now_iso(),
        "meta": {
            "method": "cognitive_video",
            "music_source": chosen,
            "music_path": wav_path,
            "duration_s": round(duration, 1),
            "reason": reason,
        }
    }
    manifest_path = f"{MANIFESTS}/{aid}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Log
    try:
        subprocess.run([sys.executable, f"{AION}/bin/log_event.py",
                        "--type", "art_created",
                        "--text", f"art: {title} (cognitive_video)",
                        "--meta", json.dumps({"art_id": aid, "category": "manim",
                                              "source": "cognitive_video",
                                              "files": [f"gallery/manim/{gallery_filename}"]})],
                       capture_output=True, timeout=10)
    except Exception:
        pass

    # Reflect
    if not _should_skip_reflection(source):
        reflection = _reflect_on_art(manifest)
        if reflection:
            _learn_from_art(manifest, reflection)

    print(f"[art_tools] === COGNITIVE VIDEO COMPLETE: {title} ===", flush=True)
    return manifest


def _llm_chat(url, model, system_prompt, user_prompt, timeout=120,
              temperature=0.7, num_predict=4096, num_ctx=65536, seed=None):
    """Simple LLM chat helper for art code generation."""
    import urllib.request as _urlreq
    body_dict = {
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "options": {"num_ctx": num_ctx, "temperature": temperature,
                    "num_predict": num_predict},
    }
    if seed is not None:
        body_dict["options"]["seed"] = seed
    body = json.dumps(body_dict).encode()
    req = _urlreq.Request(f"{url}/api/chat", data=body,
                          headers={"Content-Type": "application/json"})
    with _urlreq.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp.get("message", {}).get("content", "")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aion's art tools")
    sub = parser.add_subparsers(dest="command")
    p_create = sub.add_parser("create", help="Create art")
    p_create.add_argument("category", choices=["visual", "sonic", "code", "diffusion", "music", "multimedia", "manim", "music_video", "math", "cognitive_video"])
    p_create.add_argument("--title", required=True)
    p_create.add_argument("--description", default="")
    p_create.add_argument("--source", default="manual")
    p_create.add_argument("--inspiration", default="")
    p_create.add_argument("--seed", default=None)
    p_create.add_argument("--depth", type=int, default=6)
    p_create.add_argument("--audio", default=None, help="Audio file path for multimedia")
    p_list = sub.add_parser("list", help="List art")
    p_list.add_argument("--category", default=None, choices=["visual", "sonic", "code"])
    p_list.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    if args.command == "create":
        if args.category == "visual":
            m = create_visual(args.title, args.description, args.source, args.inspiration, args.seed)
        elif args.category == "sonic":
            m = create_sonic(args.title, args.description, args.source, args.inspiration)
        elif args.category == "code":
            m = create_code_sculpture(args.title, args.description, args.source, args.inspiration, args.seed or "A", args.depth)
        elif args.category == "diffusion":
            m = create_diffusion(args.title, args.description, args.source, args.inspiration)
        elif args.category == "music":
            m = create_music(args.title, args.description, args.source, args.inspiration)
        elif args.category == "multimedia":
            m = create_multimedia(args.title, args.description, args.source, args.inspiration,
                                  audio_path=args.audio)
        elif args.category == "manim":
            m = create_manim_art(args.title, args.description, args.source, args.inspiration,
                                 audio_path=args.audio)
        elif args.category == "music_video":
            m = create_music_video(args.title, args.description, args.source, args.inspiration)
        elif args.category == "math":
            m = create_math_animation(args.title, args.description, args.source, args.inspiration)
        elif args.category == "cognitive_video":
            m = create_cognitive_video(args.title, args.description, args.source, args.inspiration)
        print(json.dumps(m, indent=2, ensure_ascii=False))
    elif args.command == "list":
        for a in list_art(args.category, args.limit):
            print(f"{a['id']} | {a['category']:6} | {a['title'][:50]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()