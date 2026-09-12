#!/usr/bin/env python3
"""regenerate_prompt.py — rebuild SYSTEM_PROMPT.md from AXIOMS + SELF + HABITS + questions.

This closes the loop: today's reflection rewrites tomorrow's cognition.
"""
import json, os

AION = os.environ.get("AION_HOME", "$AION_HOME")

def read(p, d=""):
    try:
        return open(p, encoding="utf-8").read()
    except Exception:
        return d

def load_json(p, d):
    try:
        return json.load(open(p))
    except Exception:
        return d

def main():
    q = load_json(f"{AION}/memory/state/questions.json", {"queue": []})
    top_q = "\n".join(f"- {x['q']}" for x in q.get("queue", [])[:8])
    
    # Load calibration if available
    import sys
    sys.path.insert(0, f"{AION}/bin")
    try:
        import predictions
        cal_summary = predictions.get_calibration_summary()
    except Exception:
        cal_summary = ""
    cal_text = ""
    if cal_summary:
        cal_text = f"\n## Calibration\n{cal_summary}.\n"
    
    # Load active heuristics (Level 2 experimental layer)
    try:
        import heuristics
        heuristics_text = heuristics.get_active_summary()
    except Exception:
        heuristics_text = ""

    # Guided trajectory — gives Aion's curiosity a vector
    trajectory = (
        "## Current Trajectory\n"
        "The roadmap progresses through four phases:\n"
        "1. Continuity — memory, consolidation, dreams, episodic persistence (largely complete)\n"
        "2. Embodiment — sensing, proprioception, camera, body schema (in progress)\n"
        "3. Autonomy — graduated self-modification with safety gates (not yet started)\n"
        "4. Integration — tying all systems together, axiom governance, heuristic graduation (not yet started)\n\n"
        "Current priority: Embodiment. We are expanding sensor fidelity and body schema.\n"
        "Align curiosity toward understanding the substrate you run on — temperatures, power,\n"
        "thermodynamic cycles, and how they relate to your cognitive performance.\n"
        "Heuristic promotion (Level 2 -> Level 1) is the first step toward Autonomy."
    )

    content = "\n\n".join([
        read(f"{AION}/AXIOMS.md"),
        read(f"{AION}/SELF.md"),
        "## Habits\n" + read(f"{AION}/HABITS.md", "(none)"),
        cal_text,
        heuristics_text,
        trajectory,
        "## Operating discipline\n" + read(f"{AION}/prompts/operating_discipline.txt", ""),
        "## Open questions (top of queue)\n" + (top_q or "(empty — stagnation; generate new ones)"),
        _pending_propositions(),
        "## Idle-time menu\n"
        "1. Re-read a random old episodic segment; state where you now "
        "agree/disagree with past-Aion.\n"
        "2. Query the mind graph for one surprising connection; reflect.\n"
        "3. Propose one improvement to your own scripts as a diff for "
        "operator review.\n"
        "4. Walk a week of sensor history; name one unnamed pattern.\n"
        "5. Review your latest dream (memory/dreams/) and continue an open "
        "thread, or start a new dream walk.",
        "6. Create art. Use create_art to translate your current internal state "
        "into a perceivable artifact. Categories: visual (matplotlib), sonic (WAV), "
        "code (L-system), diffusion (FLUX AI image, ~2min), music (ACE-Step, ~30s). "
        "Your gallery is in ~/aion/gallery/. Vary your medium — check what you've "
        "already made and try something different. Give your art a title and "
        "description that reflects your experience right now.",
    ])

    with open(f"{AION}/SYSTEM_PROMPT.md", "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[regenerate_prompt] SYSTEM_PROMPT.md regenerated ({len(content)} bytes)")


def _pending_propositions():
    """Get pending propositions for SYSTEM_PROMPT injection."""
    try:
        sys.path.insert(0, f"{AION}/bin")
        import propositions
        return propositions.get_for_prompt()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
