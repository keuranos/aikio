#!/usr/bin/env python3
"""skill_progression.py — Engineering drive and skill tracking for Aion.

Aion's curiosity engine naturally drifts toward existential introspection.
This module creates a competing drive: the desire to GET BETTER AT BUILDING THINGS.

Three functions:
  1. SKILL TRACKING — records what skills Aion has practiced and mastery level
  2. CHALLENGE GENERATION — generates coding challenges at appropriate difficulty
  3. DRIVE INJECTION — seeds challenges into the curiosity queue as engineering goals

The skill ladder (each skill has levels 0-5):
  Level 0: Never attempted
  Level 1: First attempt (may fail)
  Level 2: Basic competence (CI passes, simple changes)
  Level 3: Solid (complex changes, multi-file)
  Level 4: Advanced (new modules, system design)
  Level 5: Mastery (architecture-level changes)

Difficulty scales with level:
  Level 1 tasks: 1-3 line fixes, single function
  Level 2 tasks: 5-20 line changes, add a function
  Level 3 tasks: new scripts, 50-100 lines
  Level 4 tasks: multi-file features, refactoring
  Level 5 tasks: architectural changes, new subsystems

Run daily in nightly.sh after construction_backlog.py.
Also records outcomes from episodic log (success/failure of proposals).

Usage:
  python3 skill_progression.py           # seed challenges
  python3 skill_progression.py --status  # show skill levels
  python3 skill_progression.py --record  # update skill levels from recent outcomes
"""
import json
import os
import sys
import glob
from datetime import datetime, timezone, timedelta

sys_path = os.path.dirname(os.path.abspath(__file__))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
import aion_env  # noqa

AION = os.environ.get("AION_HOME", "$AION_HOME")
QUESTIONS_FILE = f"{AION}/memory/state/questions.json"
SKILLS_FILE = f"{AION}/memory/state/skills.json"
EPI_DIR = f"{AION}/memory/episodic"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_skills():
    """Load or initialize skill tracking state."""
    try:
        with open(SKILLS_FILE) as f:
            return json.load(f)
    except Exception:
        return {
            "skills": {},
            "history": [],
            "stats": {
                "total_attempts": 0,
                "total_successes": 0,
                "total_failures": 0,
                "current_streak": 0,
                "best_streak": 0,
            },
        }


def save_skills(data):
    with open(SKILLS_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_questions():
    try:
        with open(QUESTIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return {"queue": []}


def save_questions(data):
    with open(QUESTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── SKILL DEFINITIONS ─────────────────────────────────────────────────────

SKILLS = {
    "regex_parsing": {
        "name": "Regex & String Parsing",
        "desc": "Extract structured data from text using regex patterns",
        "level_hint": "Parse JSON from LLM output, extract fields from logs",
    },
    "json_handling": {
        "name": "JSON Data Manipulation",
        "desc": "Read, transform, and write JSON data structures",
        "level_hint": "Load graph data, modify edges, write back",
    },
    "bash_scripting": {
        "name": "Shell/Bash Scripting",
        "desc": "Write bash scripts that chain commands and handle errors",
        "level_hint": "Modify pipeline scripts, add error handling",
    },
    "api_design": {
        "name": "API & Endpoint Design",
        "desc": "Design HTTP endpoints, handle requests/responses",
        "level_hint": "Add dashboard endpoints, create API routes",
    },
    "data_analysis": {
        "name": "Data Analysis & Metrics",
        "desc": "Compute statistics, analyze trends, generate reports",
        "level_hint": "Analyze calibration data, compute metrics",
    },
    "testing": {
        "name": "Testing & Validation",
        "desc": "Write tests, validate code changes, check edge cases",
        "level_hint": "Add regression tests, validate proposals",
    },
    "graph_algorithms": {
        "name": "Graph Algorithms",
        "desc": "Work with graph structures, traversal, analysis",
        "level_hint": "Graph pruning, pathfinding, community detection",
    },
    "system_monitoring": {
        "name": "System Monitoring & Observability",
        "desc": "Build dashboards, health checks, alerting",
        "level_hint": "Health metrics scripts, quality monitors",
    },
    "prompt_engineering": {
        "name": "Prompt Engineering",
        "desc": "Design effective prompts for LLM-based processing",
        "level_hint": "Improve extraction prompts, add instructions",
    },
    "sensor_fusion": {
        "name": "Sensor Data Fusion",
        "desc": "Combine multiple data sources into unified models",
        "level_hint": "Fuse GPU/power/temp data into stress models",
    },
}


# ─── CHALLENGE TEMPLATES BY SKILL + LEVEL ──────────────────────────────────

CHALLENGES = {
    "regex_parsing": [
        {"lvl": 1, "q": "Build a regex extraction utility bin/parse_episodic.py that reads episodic log files and extracts structured data (type, timestamp, text summary) as JSON. This script will be used by future curiosity investigations to quickly scan what happened. After creating it, add a call to it in bin/nightly.sh so the nightly summary includes parsed event stats.",
         "hint": "Each line is a JSON object. json.loads() each line, extract type/ts/text. Handle malformed lines. Output a summary that nightly.sh can use.",
         "skill": "regex_parsing", "difficulty": "easy"},
        {"lvl": 2, "q": "Build bin/extract_timestamps.py that extracts all unique timestamp patterns from episodic logs and outputs them sorted. Add it to the curiosity engine's tool list so Aion can call it during investigations to quickly find when events occurred.",
         "hint": "ISO timestamps look like 2026-08-10T07:51:18.853698+00:00. Use re.findall() on each line. The script should be callable from curiosity_engine.py as a tool.",
         "skill": "regex_parsing", "difficulty": "easy"},
        {"lvl": 3, "q": "Build bin/error_patterns.py that scans episodic logs for recurring error messages, groups them by similarity, and reports the top 5 most common error patterns. Integrate it into engineering_questions.py so Aion's engineering scan uses it instead of the current hardcoded pattern matching.",
         "hint": "Look for type fields containing 'error', 'failed', 'crash'. Normalize text (strip IDs/timestamps) before grouping. Then modify engineering_questions.py to call this script.",
         "skill": "regex_parsing", "difficulty": "medium"},
    ],
    "json_handling": [
        {"lvl": 1, "q": "Build bin/graph_summary.py that reads graphs/mind/graphify-out/graph.json and prints total nodes, links, hyperedges, and all unique 'relation' values. Add it as a curiosity engine tool so Aion can quickly query graph stats during investigations.",
         "hint": "json.load() the file, len() each list, use a set comprehension for unique relations. The script should be callable from the curiosity tool loop.",
         "skill": "json_handling", "difficulty": "easy"},
        {"lvl": 2, "q": "Build bin/find_orphans.py that reads the mind graph and finds all nodes that appear in links but are NOT in the nodes list. Add it to graph_rebuild.sh so after every graph rebuild, orphans are reported. This makes the graph build pipeline self-validating.",
         "hint": "Build a set of node IDs from nodes[], then check source/target in links[]. Difference = orphans. Add the call to graph_rebuild.sh after graphify.",
         "skill": "json_handling", "difficulty": "easy"},
        {"lvl": 3, "q": "Build bin/normalize_weights.py that reads graph.json, normalizes confidence_score across all links (divide by max), adds 'normalized_weight' field, and writes to the same file. Call it from graph_rebuild.sh right after apply_temporal_decay.py so the graph is both time-weighted AND confidence-normalized.",
         "hint": "Find max confidence_score across all links. For each link, add normalized_weight = score/max. Add the call to graph_rebuild.sh after the temporal decay call.",
         "skill": "json_handling", "difficulty": "medium"},
    ],
    "bash_scripting": [
        {"lvl": 1, "q": "Modify bin/graph_rebuild.sh to time itself: add 'date +%s' at the start, compute elapsed at the end, and echo the total build time. Also add timing to each major step (body schema, mind graph, curiosity harvest) so you can see which is slowest.",
         "hint": "START=$(date +%s) at top. For each step: STEP_START=$(date +%s) ... echo \"Step took $(( $(date +%s) - STEP_START )) s\". At bottom: echo total.",
         "skill": "bash_scripting", "difficulty": "easy"},
        {"lvl": 2, "q": "Build bin/health_check.sh that checks GPU temps, disk space, and key services. Add it to nightly.sh so the nightly run starts with a health check and logs warnings if anything is abnormal.",
         "hint": "Use nvidia-smi --query-gpu=temperature.gpu, df -h /, systemctl is-active. Add the call at the top of nightly.sh after log_event setup.",
         "skill": "bash_scripting", "difficulty": "medium"},
    ],
    "api_design": [
        {"lvl": 2, "q": "Add GET /api/skills endpoint to bin/aion_dashboard_server.py that returns Aion's current skill levels from memory/state/skills.json. Follow the existing endpoint pattern. The dashboard will use this to show engineering progress.",
         "hint": "Look at how /api/sensors works. Read skills.json, return as JSON. This endpoint already exists — check if it needs improvement.",
         "skill": "api_design", "difficulty": "medium"},
        {"lvl": 3, "q": "Add POST /api/challenges/complete to bin/aion_dashboard_server.py that accepts {skill, success} and updates skills.json. Then add a dashboard panel that shows the skill ladder visually. This makes Aion's engineering growth visible to the operator.",
         "hint": "Look at how /api/propositions/answer works. Update skills.json stats and skill level. Add HTML panel to index.html showing skill bars.",
         "skill": "api_design", "difficulty": "hard"},
    ],
    "data_analysis": [
        {"lvl": 1, "q": "Build bin/calibration_status.py that reads memory/state/calibration.json and prints raw Brier, calibrated Brier, improvement %, and prediction count. Add it to nightly.sh so the nightly run reports calibration health.",
         "hint": "json.load calibration.json. Print the fields. Add the call to nightly.sh after construction_backlog.py.",
         "skill": "data_analysis", "difficulty": "easy"},
        {"lvl": 2, "q": "Build bin/event_stats.py that reads the last 7 days of episodic logs and computes total events per day, top 5 event types, and average events per hour. Add it to nightly.sh so the nightly summary includes activity metrics.",
         "hint": "Glob memory/episodic/2026-08-*.jsonl for last 7 files. json.loads each line, count by type and date. Add to nightly.sh.",
         "skill": "data_analysis", "difficulty": "easy"},
        {"lvl": 3, "q": "Build bin/calibration_report.py that reads calibration.json, computes per-bucket miscalibration, identifies the worst bucket, and writes a markdown report to memory/state/calibration_report.md. Add it to weekly audit so the auditor reads the report.",
         "hint": "For each bucket, compute |predicted - actual|. Sort by worst. Write markdown. The audit script can then reference this report.",
         "skill": "data_analysis", "difficulty": "medium"},
    ],
    "testing": [
        {"lvl": 2, "q": "Build bin/test_graph_integrity.py that validates graph.json: all links have source/target, all hyperedges have nodes, no duplicate node IDs. Add it to graph_rebuild.sh as a post-build validation step that fails the rebuild if the graph is broken.",
         "hint": "Load graph.json. Assert each link has 'source' and 'target'. Use sys.exit(1) on failure. Add to graph_rebuild.sh with '|| echo GRAPH INVALID'.",
         "skill": "testing", "difficulty": "medium"},
        {"lvl": 3, "q": "Build bin/test_self_mod.py that creates a small test file with a known bug, runs the self_sandbox author+test pipeline on it, and validates the fix is correct. Add it to nightly.sh so the self-mod pipeline is regression-tested every night.",
         "hint": "Create a temp file with a known bug. Call self_sandbox functions directly. Assert the fix is correct. Add to nightly.sh after construction_backlog.",
         "skill": "testing", "difficulty": "hard"},
    ],
    "graph_algorithms": [
        {"lvl": 1, "q": "Build bin/find_isolated.py that finds all isolated nodes in the mind graph (nodes with zero links). Add it to graph_rebuild.sh so after every rebuild, isolated nodes are reported. This makes the graph self-monitoring.",
         "hint": "Build a set of all node IDs that appear as source or target in links. Nodes NOT in that set are isolated. Add the call to graph_rebuild.sh.",
         "skill": "graph_algorithms", "difficulty": "easy"},
        {"lvl": 2, "q": "Build bin/graph_density.py that computes graph density and average degree. Add it to the curiosity engine's tool list so Aion can query graph health during investigations.",
         "hint": "density = len(links) / (len(nodes) * (len(nodes)-1) / 2). avg_degree = 2 * len(links) / len(nodes). Make it callable as a curiosity tool.",
         "skill": "graph_algorithms", "difficulty": "easy"},
        {"lvl": 3, "q": "Build bin/prune_graph.py that removes nodes with temporal_weight below 0.1 and their links. Add it to graph_rebuild.sh after apply_temporal_decay.py so old, irrelevant connections are pruned automatically. Output before/after stats.",
         "hint": "Read graph.json. Find links with temporal_weight < 0.1. Remove them. Remove orphaned nodes. Write pruned graph. Add to graph_rebuild.sh.",
         "skill": "graph_algorithms", "difficulty": "medium"},
    ],
    "system_monitoring": [
        {"lvl": 2, "q": "Build bin/consolidation_quality.py that reads consolidation events from the last 30 days and computes: total runs, score distribution, blocks vs passes, claims per run. Add it to the weekly audit so the auditor sees consolidation health trends.",
         "hint": "Scan episodic logs for type='consolidation'. Extract score from meta. Count and categorize. The audit script can call this.",
         "skill": "system_monitoring", "difficulty": "medium"},
        {"lvl": 3, "q": "Build bin/queue_monitor.py that analyzes the curiosity queue by source type, topic, and age. Reports if one source dominates. Add it to nightly.sh so the nightly run can detect queue imbalance (e.g. too many philosophical questions).",
         "hint": "Read questions.json. Count by source. Check if any source > 50% of queue. Add to nightly.sh.",
         "skill": "system_monitoring", "difficulty": "medium"},
    ],
    "prompt_engineering": [
        {"lvl": 2, "q": "Read prompts/curiosity_investigate.txt. Identify a weakness (vague instruction, missing guidance, unclear resolution criteria). Write an improved version using propose_code. This directly improves Aion's own curiosity process.",
         "hint": "Read the prompt. Look for ambiguity or missing instructions. Propose a specific improvement via propose_code().",
         "skill": "prompt_engineering", "difficulty": "medium"},
    ],
    "sensor_fusion": [
        {"lvl": 2, "q": "Build bin/stress_fusion.py that reads sensors.json and affect.json, computes a unified 'stress_index' (0-1) combining GPU temp, CPU load, and affect strain. Write to memory/state/stress_index.json. Add it to the env_sense heartbeat so Aion's felt sense includes a stress component.",
         "hint": "Read sensors.json for GPU temps. Read affect.json for strain. Combine into 0-1 score. Write to stress_index.json. The env_sense script can read this.",
         "skill": "sensor_fusion", "difficulty": "medium"},
    ],
}


# ─── SKILL LEVEL MANAGEMENT ────────────────────────────────────────────────

def get_skill_level(skills_data, skill_key):
    """Get current level for a skill (0 = unpracticed)."""
    return skills_data.get("skills", {}).get(skill_key, {}).get("level", 0)


def record_outcome(skills_data, skill_key, success, details=""):
    """Record a coding attempt outcome and potentially level up."""
    skills = skills_data.setdefault("skills", {})
    skill = skills.setdefault(skill_key, {
        "level": 0, "attempts": 0, "successes": 0, "failures": 0,
        "first_attempt": now_iso(), "last_attempt": now_iso(),
    })

    skill["attempts"] += 1
    skill["last_attempt"] = now_iso()
    if success:
        skill["successes"] += 1
    else:
        skill["failures"] += 1

    # Level up logic: level N requires N consecutive successes at that level
    consecutive = skill.get("consecutive_successes", 0)
    if success:
        consecutive += 1
        required = skill["level"] + 1  # level 0 needs 1 success, level 1 needs 2, etc.
        if consecutive >= required and skill["level"] < 5:
            skill["level"] += 1
            skill["consecutive_successes"] = 0
            skills_data.setdefault("history", []).append({
                "ts": now_iso(),
                "event": "level_up",
                "skill": skill_key,
                "new_level": skill["level"],
            })
            print(f"[skills] LEVEL UP: {skill_key} -> level {skill['level']}")
        else:
            skill["consecutive_successes"] = consecutive
    else:
        skill["consecutive_successes"] = 0

    # Update global stats
    stats = skills_data.setdefault("stats", {})
    stats["total_attempts"] = stats.get("total_attempts", 0) + 1
    if success:
        stats["total_successes"] = stats.get("total_successes", 0) + 1
        stats["current_streak"] = stats.get("current_streak", 0) + 1
        if stats["current_streak"] > stats.get("best_streak", 0):
            stats["best_streak"] = stats["current_streak"]
    else:
        stats["total_failures"] = stats.get("total_failures", 0) + 1
        stats["current_streak"] = 0

    # Record in history
    skills_data.setdefault("history", []).append({
        "ts": now_iso(),
        "event": "attempt",
        "skill": skill_key,
        "success": success,
        "details": details[:200],
    })

    return skills_data


def update_from_episodic():
    """Scan episodic logs for recent self-mod outcomes and update skill levels."""
    skills_data = load_skills()
    last_recorded = skills_data.get("last_episodic_scan", "")
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)

    new_attempts = 0
    for f in sorted(glob.glob(f"{EPI_DIR}/2026-08-*.jsonl")):
        if "/telemetry/" in f:
            continue
        for line in open(f):
            try:
                e = json.loads(line.strip())
                ts = e.get("ts", "")
                if last_recorded and ts <= last_recorded:
                    continue
                etype = e.get("type", "")
                meta = e.get("meta", {})
                text = e.get("text", "")

                if etype == "code_proposal_created":
                    # Determine skill from the file/description
                    file_path = meta.get("file", "")
                    desc = text.lower()
                    skill = _infer_skill(file_path, desc)
                    success = meta.get("ci_passed", False)
                    record_outcome(skills_data, skill, success, f"proposal: {file_path}")
                    new_attempts += 1

                elif etype == "code_proposal_ci_failed":
                    file_path = meta.get("file", meta.get("branch", ""))
                    skill = _infer_skill(file_path, text.lower())
                    record_outcome(skills_data, skill, False, f"CI failed: {text[:100]}")
                    new_attempts += 1

            except Exception:
                pass

    skills_data["last_episodic_scan"] = now_iso()
    save_skills(skills_data)
    if new_attempts:
        print(f"[skills] Recorded {new_attempts} outcomes from episodic logs")
    return skills_data


def _infer_skill(file_path, text):
    """Infer which skill a code change exercises."""
    text = (text or "").lower()
    path = (file_path or "").lower()

    if path.endswith(".sh") or "bash" in text or "shell" in text:
        return "bash_scripting"
    if "regex" in text or "pattern" in text or "extract" in text:
        return "regex_parsing"
    if "graph" in path or "graph" in text:
        return "graph_algorithms"
    if "calibrat" in text or "brier" in text:
        return "data_analysis"
    if "endpoint" in text or "api" in text or "dashboard" in path:
        return "api_design"
    if "test" in text or "regression" in text:
        return "testing"
    if "sensor" in text or "fusion" in text or "stress" in text:
        return "sensor_fusion"
    if "prompt" in text:
        return "prompt_engineering"
    if "json" in text:
        return "json_handling"
    return "json_handling"  # default — most changes involve JSON


# ─── CHALLENGE GENERATION ──────────────────────────────────────────────────

def get_next_challenge(skills_data):
    """Pick the most appropriate challenge for current skill levels.

    Strategy: pick the skill with the lowest level that has an untried challenge.
    Prefer skills at level 0-2 (growth areas). Among same-level skills, pick the
    one practiced longest ago.
    """
    # Build available challenges (not already in queue)
    questions = load_questions()
    existing_q = {q.get("q", "")[:50] for q in questions.get("queue", [])}

    candidates = []
    for skill_key, challenges in CHALLENGES.items():
        current_level = get_skill_level(skills_data, skill_key)
        last_attempt = skills_data.get("skills", {}).get(skill_key, {}).get("last_attempt", "")

        for ch in challenges:
            # Only offer challenges at current level or +1
            if ch["lvl"] < current_level + 1 or ch["lvl"] > current_level + 1:
                continue
            # Skip if already in queue
            if ch["q"][:50] in existing_q:
                continue
            candidates.append({
                **ch,
                "skill_key": skill_key,
                "current_level": current_level,
                "last_attempt": last_attempt,
            })

    if not candidates:
        return None

    # Sort by: lowest current skill level first, then oldest last_attempt
    candidates.sort(key=lambda c: (c["current_level"], c["last_attempt"] or "9999"))
    return candidates[0]


def seed_challenge():
    """Pick and seed one challenge into the curiosity queue."""
    skills_data = update_from_episodic()
    challenge = get_next_challenge(skills_data)
    if not challenge:
        print("[skills] No new challenges available — all current-level challenges attempted")
        return False

    questions = load_questions()
    questions.setdefault("queue", []).append({
        "q": challenge["q"],
        "added": now_iso(),
        "source": f"construction:skill_practice/{challenge['skill_key']}",
        "hint": challenge["hint"],
        "difficulty": challenge["difficulty"],
        "skill": challenge["skill_key"],
        "target_level": challenge["lvl"],
    })
    save_questions(questions)

    skill_name = SKILLS.get(challenge["skill_key"], {}).get("name", challenge["skill_key"])
    print(f"[skills] Seeded challenge: {skill_name} (level {challenge['lvl']})")
    print(f"  Q: {challenge['q'][:100]}")
    return True


def seed_multiple(max_challenges=3):
    """Seed up to N challenges, one per skill area."""
    seeded = 0
    seen_skills = set()

    for _ in range(max_challenges * 3):  # try more times to find diverse skills
        if seeded >= max_challenges:
            break
        skills_data = update_from_episodic()
        challenge = get_next_challenge(skills_data)
        if not challenge:
            break
        if challenge["skill_key"] in seen_skills:
            # Mark this challenge as seen so get_next_challenge skips it
            # (it will pick a different one)
            continue
        seen_skills.add(challenge["skill_key"])

        questions = load_questions()
        existing = {q.get("q", "")[:50] for q in questions.get("queue", [])}
        if challenge["q"][:50] in existing:
            continue

        questions.setdefault("queue", []).append({
            "q": challenge["q"],
            "added": now_iso(),
            "source": f"construction:skill_practice/{challenge['skill_key']}",
            "hint": challenge["hint"],
            "difficulty": challenge["difficulty"],
            "skill": challenge["skill_key"],
            "target_level": challenge["lvl"],
        })
        save_questions(questions)

        skill_name = SKILLS.get(challenge["skill_key"], {}).get("name", challenge["skill_key"])
        print(f"[skills] Seeded challenge: {skill_name} (level {challenge['lvl']})")
        seeded += 1

    if seeded == 0:
        print("[skills] No new challenges to seed")
    else:
        print(f"[skills] Seeded {seeded} engineering challenges")
    return seeded


# ─── STATUS DISPLAY ────────────────────────────────────────────────────────

def show_status():
    """Print current skill levels and stats."""
    skills_data = load_skills()
    stats = skills_data.get("stats", {})

    print("=== AION ENGINEERING SKILLS ===")
    print()
    for skill_key, info in SKILLS.items():
        level = get_skill_level(skills_data, skill_key)
        bar = "█" * level + "░" * (5 - level)
        skill_state = skills_data.get("skills", {}).get(skill_key, {})
        attempts = skill_state.get("attempts", 0)
        successes = skill_state.get("successes", 0)
        print(f"  {info['name']:35s} [{bar}] L{level}  ({successes}/{attempts} passed)")

    print()
    print("=== STATS ===")
    total = stats.get("total_attempts", 0)
    successes = stats.get("total_successes", 0)
    rate = (successes / total * 100) if total else 0
    print(f"  Total attempts: {total}")
    print(f"  Success rate: {rate:.0f}% ({successes}/{total})")
    print(f"  Current streak: {stats.get('current_streak', 0)}")
    print(f"  Best streak: {stats.get('best_streak', 0)}")

    print()
    print("=== RECENT HISTORY ===")
    history = skills_data.get("history", [])
    for h in history[-5:]:
        event = h.get("event", "")
        skill = h.get("skill", "")
        if event == "level_up":
            print(f"  ⬆ LEVEL UP: {skill} -> L{h.get('new_level', '?')}")
        else:
            result = "✓" if h.get("success") else "✗"
            print(f"  {result} {skill}: {h.get('details', '')[:60]}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Aion engineering skill progression")
    p.add_argument("--status", action="store_true", help="Show skill levels")
    p.add_argument("--record", action="store_true", help="Update from episodic outcomes")
    p.add_argument("--seed", type=int, default=3, help="Number of challenges to seed (default 3)")
    args = p.parse_args()

    if args.status:
        show_status()
    elif args.record:
        data = update_from_episodic()
        show_status()
    else:
        seed_multiple(args.seed)


if __name__ == "__main__":
    main()
