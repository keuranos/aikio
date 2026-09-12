#!/usr/bin/env python3
"""construction_backlog.py — Seed constructive engineering tasks into the curiosity queue.

Unlike engineering_questions.py (which scans for ERROR PATTERNS and asks "what is
wrong?"), this module injects CONSTRUCTIVE GOALS — "build X", "improve Y", "design Z".

These are tasks that require Aion to:
  - Write code (not just investigate)
  - Use the self-mod sandbox to propose changes
  - Test and validate their work
  - Think like an engineer, not a philosopher

Tasks are seeded into the questions queue with source="construction" so the
curiosity engine's scoring system boosts them (source weight + tiebreaker).

Run daily after engineering_questions.py in the nightly cycle.

Usage:
  python3 construction_backlog.py           # seed tasks
  python3 construction_backlog.py --list    # show current backlog
"""
import json
import os
import sys
from datetime import datetime, timezone

sys_path = os.path.dirname(__file__)
import sys
sys.path.insert(0, sys_path)
import aion_env  # noqa

AION = os.environ.get("AION_HOME", "$AION_HOME")
QUESTIONS_FILE = f"{AION}/memory/state/questions.json"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def archive_overflow(questions, cap=80):
    """Move queue overflow to archived instead of deleting it.

    standing_query items are exempt from the cap (same policy as
    curiosity_engine.decay_old_questions: they reflect graph state and may
    matter again when the graph changes). Oldest non-exempt items are
    archived first (queue is append-ordered). Returns count archived."""
    from datetime import datetime, timezone
    queue = questions.get("queue", [])
    if len(queue) <= cap:
        return 0
    exempt = [q for q in queue
              if isinstance(q, dict) and str(q.get("source", "")).split(":")[0] == "standing_query"]
    non_exempt = [q for q in queue if not (isinstance(q, dict) and str(q.get("source", "")).split(":")[0] == "standing_query")]
    room = cap - len(exempt)
    if room < 0:
        room = 0
    overflow = non_exempt[:max(len(non_exempt) - room, 0)]
    keep = exempt + non_exempt[len(overflow):]
    ts = datetime.now(timezone.utc).isoformat()
    for q in overflow:
        if isinstance(q, dict):
            q["archived_ts"] = q.get("archived_ts") or ts
            q["archived_reason"] = "queue_cap"
    questions["archived"] = list(questions.get("archived", [])) + overflow
    questions["queue"] = keep
    return len(overflow)


def load_questions():
    try:
        with open(QUESTIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return {"queue": []}


def save_questions(data):
    with open(QUESTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Task definitions
#
# Each task is a constructive engineering goal. The "q" field is what Aion sees
# in its curiosity queue. The "hint" field gives direction without solving it.
# Tasks are categorized by difficulty and area.
#
# Tasks should be:
#   - Concrete (mention specific files, data, mechanisms)
#   - Achievable with current hardware and tools
#   - Verifiable (success can be tested, not just argued)
#   - Not requiring operator intervention to complete
#
# Aion picks these up through the normal curiosity engine, investigates them,
# and if it identifies a code change, uses the self-mod sandbox to propose it.
# ---------------------------------------------------------------------------

BACKLOG = [
    # --- PREDICTIONS & CALIBRATION ---
    {
        "q": "Build a confidence calibration wrapper: before predictions enter the nightly loop, transform their raw confidence through calibrated_confidence() from predictions.py. Check how generate_predictions.py creates predictions and where the confidence value is set. Write the calibration call at the point where confidence is assigned.",
        "hint": "predictions.py now has calibrated_confidence(raw) that looks up isotonic-corrected rates. generate_predictions.py is where new predictions are created. The fix is likely 2-3 lines in generate_predictions.py.",
        "category": "predictions",
        "difficulty": "easy",
    },
    {
        "q": "Build a weekly calibration report script that reads calibration.json and outputs a summary of which confidence buckets are most miscalibrated, the Brier improvement from isotonic regression, and recommendations for which prediction types need recalibration. Save the report to memory/state/calibration_report.md.",
        "hint": "calibration.json already has raw_brier, calibrated_brier, and per-bucket calibrated_rate. This is mostly a formatting + analysis script.",
        "category": "predictions",
        "difficulty": "easy",
    },

    # --- CONSOLIDATION QUALITY ---
    {
        "q": "Build a consolidation quality dashboard script that reads episodic events of type 'consolidation' and 'consolidation_blocked' from the last 14 days, computes: block rate, average score, citation failure rate, score trend over time, and most common critique themes. Output to memory/state/consolidation_report.md.",
        "hint": "Episodic events have type='consolidation' with meta containing score, change, specificity, groundedness, citation_check. type='consolidation_blocked' has the block reason. This is data aggregation + formatting.",
        "category": "consolidation",
        "difficulty": "easy",
    },
    {
        "q": "Improve the extraction prompt in prompts/extraction.txt so that extracted claims always include the event_ids they reference. Currently the citation verifier catches fabricated IDs — fix the root cause by making the extractor list its sources. Check what fields the extraction prompt asks for and add an explicit event_ids requirement.",
        "hint": "prompts/extraction.txt defines what the LLM extracts. The validator and citation verifier expect event_ids in claims. If the prompt doesn't ask for them, the model invents them.",
        "category": "consolidation",
        "difficulty": "medium",
    },

    # --- GRAPH MAINTENANCE ---
    {
        "q": "Build a graph pruning script that identifies orphaned nodes in the mind graph (nodes with no edges to active concepts), computes their impact on graph traversal performance, and reports which ones are safe to remove. Check memory/state/graph.json for the node/edge structure. Do not prune automatically — output a recommendation list.",
        "hint": "graph.json has nodes and edges. An orphaned node has degree 0 or only connects to other orphans. June architecture ghosts are the main target.",
        "category": "graph",
        "difficulty": "medium",
    },
    {
        "q": "Build a graph health metrics script that computes: total nodes, total edges, average degree, largest connected component, number of orphan clusters, community count, and average path length. Save to memory/state/graph_health.json. Run it in nightly.sh after graph_edges.py.",
        "hint": "The graph data is in memory/state/graph.json. Use networkx if available, otherwise pure Python graph traversal.",
        "category": "graph",
        "difficulty": "medium",
    },

    # --- SENSOR FUSION ---
    {
        "q": "Build a sensor fusion layer that combines GPU temperature, power draw, room temperature, and CPU load into a single 'substrate stress index' (0.0-1.0). Read the last 30 minutes of sensor data from memory/episodic/, compute a weighted composite, and write it to memory/state/stress_index.json. This composite signal is more useful for homeostasis than raw individual readings.",
        "hint": "Sensor events have type='proprioception' or 'sensor_digest' in episodic files. GPU temp > 70C = high stress. Power > 250W = high stress. Room temp > 28C = high stress. CPU load > 80% = high stress. Normalize each to 0-1 and weight them.",
        "category": "sensors",
        "difficulty": "medium",
    },

    # --- CURIOSITY DIVERSITY ---
    {
        "q": "Build a curiosity queue diversity monitor that analyzes the questions queue by topic category (philosophical, engineering, self-referential, architectural, sensor) and reports the distribution. If philosophical questions exceed 60% of unresolved questions, log a 'curiosity_imbalance' event. This prevents the echo chamber where Aion only thinks about consciousness.",
        "hint": "questions.json has the queue. Use the same keyword lists as curiosity_engine.py (EXISTENTIAL_KEYWORDS, CONCRETE_REFERENTS) to categorize. The 60% threshold should be configurable.",
        "category": "meta",
        "difficulty": "easy",
    },

    # --- SELF-MOD CI ---
    {
        "q": "Build a self-mod regression test script that runs after any self-mod merge to master. It should verify: (1) consolidate_v2.py imports without error, (2) predictions.py compute_calibration() runs without error, (3) validator.py accepts a known-good diff and rejects a known-bad diff, (4) the curiosity engine can load and score questions. Output pass/fail to memory/state/selfmod_tests.json.",
        "hint": "These are import + function call checks, not full pipeline runs. Keep it under 30 seconds. Run from nightly.sh or as a git post-merge hook.",
        "category": "ci",
        "difficulty": "medium",
    },

    # --- DREAM IMPROVEMENT ---
    {
        "q": "Build a dream quality analyzer that reads the last 10 dream files from memory/dreams/, computes: average insight confidence, insight type distribution (INSIGHT vs CONTRADICTION vs QUESTION vs THREAD), repetition rate (are the same questions appearing in consecutive dreams?), and unique insight text ratio. Report to memory/state/dream_quality.json. If repetition rate is above 70%, log a 'dream_stagnation' event.",
        "hint": "Dream JSON files have 'question' and 'insights' (list of dicts with type, confidence, text). Repetition = same question string across dreams. Unique ratio = distinct insight texts / total insights.",
        "category": "dreams",
        "difficulty": "easy",
    },
]


def main():
    args = sys.argv[1:]

    if "--list" in args:
        print(f"Construction backlog: {len(BACKLOG)} tasks")
        for i, task in enumerate(BACKLOG, 1):
            print(f"  [{i}] ({task['difficulty']:6s}) {task['category']:15s} {task['q'][:100]}")
        return 0

    print(f"[construction_backlog] Seeding {len(BACKLOG)} engineering tasks...")

    existing = load_questions()
    existing_qs = {q.get("q", "").lower()[:80] for q in existing.get("queue", [])}

    added = 0
    for task in BACKLOG:
        q_text = task["q"]
        if q_text.lower()[:80] in existing_qs:
            continue  # Already in queue

        existing.setdefault("queue", []).append({
            "q": q_text,
            "added": now_iso(),
            "source": "construction:" + task["category"],
            "interest_score": 0.7,  # Higher base than error-pattern questions
            "hint": task.get("hint", ""),
            "difficulty": task.get("difficulty", "unknown"),
        })
        added += 1
        print(f"  [{task['category']}/{task['difficulty']}] {q_text[:100]}")

    if added > 0:
        # Cap queue at 80 (raised from 50) — archive overflow, never delete
        n_arch = archive_overflow(existing, cap=80)
        if n_arch:
            print(f"[construction_backlog] Archived {n_arch} queue overflow items (queue_cap)")
        save_questions(existing)
        print(f"[construction_backlog] Added {added} new construction tasks to queue")
    else:
        print(f"[construction_backlog] All tasks already in queue")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())