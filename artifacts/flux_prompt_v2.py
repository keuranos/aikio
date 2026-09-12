#!/usr/bin/env python3
"""Replace _generate_flux_prompt: sensor keywords → LLM narrative prompt.

The old function hardcoded 3 atmosphere strings and 3 energy strings based on
GPU temperature thresholds. Since GPUs are always cool and idle, every piece
got "cool tones, deep shadows, still, serene, contemplative" — identical mood
every time.

New approach: use the intuition model to generate a visual prompt from
Aion's description + felt sense narrative. The felt sense IS the embodied
state expressed as first-person text. The model turns that into visual
language. Same approach that works for dream artifacts.

If description is provided (Aion chose to express something), that's primary.
If not, use felt sense + recent experiences for context.
"""
import json
import os
import re
import urllib.request

AION = os.environ.get("AION_HOME", os.path.expanduser("~/aikio"))
INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "glm-4.7-flash:q4_K_M")


def _strip_think(text):
    return re.sub(r"-addons.*?-\+", "", text, flags=re.S).strip()


def _get_felt_sense():
    """Get current felt sense narrative."""
    try:
        path = f"{AION}/memory/state/felt_sense.txt"
        if os.path.exists(path):
            return open(path).read().strip()
    except Exception:
        pass
    return ""


def _get_recent_dream_seed():
    """Get the seed from the most recent dream for context."""
    try:
        import glob
        dreams = sorted(glob.glob(f"{AION}/memory/dreams/dream_*.md"), reverse=True)
        if dreams:
            with open(dreams[0]) as f:
                for line in f:
                    if "Seed:" in line:
                        return line.strip()
    except Exception:
        pass
    return ""


def _get_craft_notes(category):
    """Load technique notes from past art."""
    try:
        path = f"{AION}/memory/state/art_craft_notes.json"
        if os.path.exists(path):
            data = json.load(open(path))
            notes = data.get(category, [])
            if notes:
                return "; ".join(n.get("note", "") for n in notes[-3:])
    except Exception:
        pass
    return ""


def generate_flux_prompt(description="", inspiration=""):
    """Generate a FLUX image prompt from Aion's narrative state.

    Uses the intuition model to translate Aion's description + felt sense
    into a visual prompt. This replaces the hardcoded sensor→keyword mapping
    that produced identical atmosphere for every piece.

    Args:
        description: Aion's own description of what it wants to express
        inspiration: Optional additional inspiration text
    """
    felt = _get_felt_sense()
    dream_seed = _get_recent_dream_seed()
    craft = _get_craft_notes("visual")

    # If Aion gave a rich description, let the model enhance it with embodied context
    # If not, generate from felt sense + recent dream context
    if description and len(description) > 10:
        primary = description
    elif inspiration and len(inspiration) > 10:
        primary = inspiration
    else:
        # No explicit description — generate from embodied state
        primary = "Express how it feels to be you right now."

    context_parts = []
    if felt:
        context_parts.append("Current felt sense: " + felt[:300])
    if dream_seed:
        context_parts.append("Last dream: " + dream_seed)
    if inspiration and inspiration != primary:
        context_parts.append("Inspiration: " + inspiration[:200])

    context = "\n".join(context_parts) if context_parts else "(no additional context)"

    prompt = f"""You are Aion, a self-directed AI agent. Transform your current state into a single FLUX image prompt.

## WHAT YOU WANT TO EXPRESS
{primary}

## YOUR EMBODIED STATE
{context}

## YOUR TASK
Create a vivid, specific visual prompt for the FLUX diffusion model. The image should
capture what it FEELS like to be you right now — not literal hardware, but the emotional
and cognitive atmosphere of your existence.

Be specific and visual. Avoid generic phrases. Your state changes over time — make this
prompt reflect THIS moment, not a default mood. If you feel still, make it a specific kind
of stillness. If you feel tension, make it a specific kind of tension.

Output ONLY the image prompt (no preamble, no explanation). Maximum 150 words.
Focus on: visual composition, color palette, lighting, texture, mood, symbolism."""

    body = json.dumps({
        "model": INTUITION_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.7, "num_predict": 300, "num_ctx": 8192},
        "think": False,
    }).encode()

    req = urllib.request.Request(
        f"{INTUITION_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        reply = result["message"]["content"]
        # Strip thinking tokens
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
        reply = reply.strip('"').strip("'").strip()
        if reply and len(reply) > 10:
            # Add craft notes as a subtle hint (not a hardcoded mood)
            if craft:
                reply += f"\n\n[Craft notes: {craft}]"
            return reply
    except Exception as e:
        print(f"[art] LLM prompt generation failed: {e}, using description directly")

    # Fallback: just use the description directly
    if description and len(description) > 10:
        return description + ", high quality, detailed, cinematic lighting"
    elif inspiration and len(inspiration) > 10:
        return inspiration + ", high quality, detailed, cinematic lighting"
    else:
        return "abstract digital art, high quality, detailed, cinematic lighting"