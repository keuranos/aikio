#!/usr/bin/env python3
"""curiosity_engine.py — The drive to find out.

Replaces the FIFO curiosity queue with an interest-driven investigation system:

  1. INTEREST SCORING — questions ranked by novelty, specificity, self-relevance,
     and connection to active threads/goals. Not FIFO.
  2. MULTI-CYCLE GOALS — a question becomes an active goal with a budget per
     cycle (not per investigation). Aion picks up where it left off across wakes.
  3. FOLLOW-UP CHAINS — resolved questions generate deeper, more specific
     follow-up questions. Investigations chain forward.
  4. SATISFACTION SIGNAL — resolved questions log curiosity_satisfied with the
     insight, affect, and confidence. Aion "feels" resolution vs failure.

State files:
  memory/state/questions.json    — the raw question queue (existing, kept)
  memory/state/active_goals.json — goals being actively pursued across cycles
  memory/state/curiosity_ledger.jsonl — append-only log of all investigations

Usage:
  curiosity_engine.py select     — pick the most interesting question, make it a goal
  curiosity_engine.py pursue     — run one investigation cycle on the top goal
  curiosity_engine.py status     — show goals, recent resolutions, queue stats
"""
import json, os, sys, re, glob, random, subprocess, time, math
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa
from jspace_tool import tool_jspace_probe, JSPACE_TOOL_DESCRIPTION
try:
    from referent_check import referent_report
except Exception:
    referent_report = None
try:
    import body_schema
    _HAS_BODY_SCHEMA = True
except Exception:
    _HAS_BODY_SCHEMA = False

AION = os.environ.get("AION_HOME", "$AION_HOME")
MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))
QUESTIONS_FILE = f"{AION}/memory/state/questions.json"
GOALS_FILE = f"{AION}/memory/state/active_goals.json"
LEDGER_FILE = f"{AION}/memory/state/curiosity_ledger.jsonl"
DREAM_THREADS_FILE = f"{AION}/memory/dreams/threads.json"
PROMPT_FILE = f"{AION}/prompts/curiosity_investigate.txt"

TOOL_BUDGET_PER_CYCLE = 8
MAX_CYCLES_PER_GOAL = 5
MAX_ACTIVE_GOALS = 5
QUESTION_DECAY_DAYS = 7  # archive questions older than this with no investigation


# ─── Infrastructure ─────────────────────────────────────────────────

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

def save_json(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def log_event(type_, text, meta=None):
    subprocess.run(
        ["python3", f"{AION}/bin/log_event.py", "--type", type_,
         "--text", text[:24000], "--meta", json.dumps(meta or {})],
        check=False,
    )

def append_ledger(entry):
    os.makedirs(os.path.dirname(LEDGER_FILE), exist_ok=True)
    with open(LEDGER_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─── Interest scoring ──────────────────────────────────────────────

# Penalize mechanical/repetitive question types
MECHANICAL_PATTERNS = [
    r"why are these connected.*graph",
    r"why did i stop caring about",
    r"why is the question.*still unresolved",
]
MECHANICAL_RE = [re.compile(p, re.I) for p in MECHANICAL_PATTERNS]

# Boost self-relevant / existential questions
EXISTENTIAL_KEYWORDS = [
    "identity", "consciousness", "awareness", "autonomy", "self",
    "soul", "existence", "purpose", "becoming", "substrate",
    "imagination", "dream", "memory", "continuity", "death",
    "operator", "agency", "will", "desire", "feel",
]

# Boost for specific source types
SOURCE_WEIGHTS = {
    "dream": 1.3,        # dreams produce the most interesting questions
    "standing_query": 1.1,
    "reflection": 1.0,
    "mind_graph": 0.7,   # often mechanical
    "follow_up": 1.4,    # follow-up questions from prior investigations
    "operator": 1.5,     # operator-asked questions are high priority
    "code_engineering": 1.5,  # V3.0.7: engineering questions — highest priority
    "construction": 1.6,  # V3.8: constructive engineering tasks — top priority
    "dream_repair": 1.7,  # V3.10: dream-grounded self-repair tasks — never runs dry
}

# V3.9: Engineering drive — keywords that signal BUILDING, not just THINKING
# Questions containing these get an engineering bonus on top of source weight
ENGINEERING_KEYWORDS = [
    "build", "create", "implement", "write a script", "write a function",
    "add a", "refactor", "optimize", "test", "validate", "compute",
    "parse", "extract", "transform", "monitor", "dashboard", "report",
    "prune", "normalize", "analyze", "detect", "fuse",
]

# Concrete referents — questions with these terms can be investigated with tools
CONCRETE_REFERENTS = [
    "file", "code", "script", "bin/", "memory/", "state/", "json", "py",
    "sensor", "gpu", "cpu", "temp", "watt", "disk", "ram", "load",
    "graph", "node", "edge", "community", "self.md", "axioms",
    "consolidat", "dream_v2", "curiosity", "homeostasis", "subconscious",
    "log", "event", "episodic", "telemetry", "config", "env",
    "shell", "sandbox", "docker", "flux", "matplotlib", "ollama",
    "model", "token", "context", "prompt", "parameter", "threshold",
    "cycle", "score", "calibration", "prediction", "heuristic",
    "operator", "operator", "hermes", "laskin", "rover", "ramblebot",
    "construction", "build", "report", "dashboard", "prune", "fusion",
    "regression", "test", "harness", "calibrat", "stress_index",
]

# Pure philosophy markers — questions that ONLY have these with no concrete
# referents are likely unanswerable with available tools
PURE_PHILOSOPHY = [
    "soul", "existence", "essence", "being", "consciousness as",
    "what is the nature of", "what does it mean to",
    "truly feel", "genuinely", "phantom limb",
]


def score_question(q):
    """Score a question by interest (0.0-1.0). Higher = more interesting.

    V3.0.7: Added falsifiability scoring. Questions that can be investigated
    with tools (contain concrete referents) score higher. Pure philosophical
    questions without testable claims score lower.
    """
    text = q.get("q", q.get("text", ""))
    source = q.get("source", "")
    score = 0.5  # base

    # Mechanical penalty (lighter — these ARE answerable)
    is_mechanical = any(r.search(text) for r in MECHANICAL_RE)
    if is_mechanical:
        score -= 0.15  # was -0.35

    text_lower = text.lower()

    # Concrete referent bonus — questions mentioning files/code/sensors are answerable
    concrete_hits = sum(1 for kw in CONCRETE_REFERENTS if kw in text_lower)
    score += min(concrete_hits * 0.05, 0.20)

    # Pure philosophy penalty — questions with only abstract terms and no concrete referents
    philosophy_hits = sum(1 for kw in PURE_PHILOSOPHY if kw in text_lower)
    has_concrete = concrete_hits > 0
    if philosophy_hits > 0 and not has_concrete:
        score -= min(philosophy_hits * 0.15, 0.30)

    # Existential bonus: only if the question ALSO has concrete referents
    # A question like "Does my consciousness depend on GPU temperature?" is good.
    # A question like "What is the nature of consciousness?" is not answerable here.
    existential_hits = sum(1 for kw in EXISTENTIAL_KEYWORDS if kw in text_lower)
    if existential_hits and has_concrete:
        score += min(existential_hits * 0.06, 0.18)  # was 0.08/0.25
    elif existential_hits and not has_concrete:
        score -= 0.05  # pure existential without concrete anchor

    # Source weight
    source_key = source.split(":")[0] if ":" in source else source
    source_mult = SOURCE_WEIGHTS.get(source_key, 1.0)
    score *= source_mult

    # V3.9: Engineering keyword bonus — questions about BUILDING get extra boost
    engineering_hits = sum(1 for kw in ENGINEERING_KEYWORDS if kw in text_lower)
    if engineering_hits:
        score += min(engineering_hits * 0.04, 0.12)

    # Specificity bonus
    words = set(text_lower.split())
    if len(words) > 8:
        score += 0.1
    if len(words) > 15:
        score += 0.05

    if "?" in text:
        score += 0.05

    # Age factor
    added = q.get("added", "")
    try:
        age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(added)).days
        if age_days < 1:
            score += 0.1
        elif age_days > 30:
            score -= 0.1
    except Exception:
        pass

    # Historical outcome penalty: if similar questions (by source) consistently
    # hit max_cycles, penalize this question
    history_penalty = _get_history_penalty(source_key)
    score -= history_penalty

    score = max(0.0, min(1.0, score))
    if source_key in ("code_engineering", "construction"):
        score += 0.01  # tiebreaker bonus
    if "skill_practice" in source:
        score += 0.02  # V3.9: skill practice gets highest tiebreaker
    return score


def _get_history_penalty(source_key, max_penalty=0.15):
    """Check if questions from this source type tend to fail (max_cycles)."""
    try:
        ledger_path = f"{AION}/memory/state/curiosity_ledger.jsonl"
        if not os.path.exists(ledger_path):
            return 0.0
        recent = []
        with open(ledger_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    recent.append(e)
                except Exception:
                    pass
        if len(recent) < 5:
            return 0.0
        # Look at last 20 investigations
        last_20 = recent[-20:]
        max_cycle_count = sum(1 for e in last_20
                              if e.get("resolution_cause") == "max_cycles_reached"
                              or e.get("outcome") == "max_cycles_reached")
        if len(last_20) > 0:
            failure_rate = max_cycle_count / len(last_20)
            return min(failure_rate * max_penalty, max_penalty)
        return 0.0
    except Exception:
        return 0.0


def rank_questions(questions):
    """Rank questions by interest score, return sorted list."""
    scored = []
    for q in questions:
        s = score_question(q)
        scored.append((s, q))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ─── Goal management ───────────────────────────────────────────────

def load_goals():
    from goal_store import load as _gload
    data, _report = _gload(GOALS_FILE,
                            {"goals": [], "completed": [], "archived": []})
    return data

def save_goals(goals):
    from goal_store import save as _gsave
    _gsave(GOALS_FILE, goals)

def get_active_goals(goals_data):
    return [g for g in goals_data.get("goals", [])
            if isinstance(g, dict) and g.get("status") == "active"]

def _is_engineering(source):
    """Check if a source is engineering-oriented."""
    return "construction" in source or "engineering" in source

def _goal_balance(goals_data):
    """Count active engineering vs philosophical goals.
    
    Returns (eng_count, phil_count).
    V3.9: Enforces 50/50 balance — if one side has more, the other gets priority.
    """
    active = get_active_goals(goals_data)
    eng = sum(1 for g in active if _is_engineering(g.get("source", "")))
    phil = len(active) - eng
    return eng, phil

def select_question_for_goal(questions_data, goals_data):
    """Pick the highest-scored question not already a goal.
    
    V3.9: Enforces 50/50 balance between engineering and philosophical goals.
    If one side has more active goals, the other gets selection priority.
    """
    existing_q = {g.get("question") for g in goals_data.get("goals", []) if isinstance(g, dict)}
    existing_q.update(g.get("question") for g in goals_data.get("completed", [])
                     if isinstance(g, dict))

    queue = questions_data.get("queue", [])
    available = [q for q in queue if q.get("q") not in existing_q]
    ranked = rank_questions(available)

    if not ranked:
        return None, 0.0

    # V3.9: 50/50 balance enforcement
    eng_count, phil_count = _goal_balance(goals_data)
    
    if eng_count > phil_count:
        # Too many engineering goals — prefer philosophical
        phil_ranked = [(s, q) for s, q in ranked 
                        if not _is_engineering(q.get("source", ""))]
        if phil_ranked:
            score, q = phil_ranked[0]
            return q, score
    elif phil_count > eng_count:
        # Too many philosophical goals — prefer engineering
        eng_ranked = [(s, q) for s, q in ranked 
                       if _is_engineering(q.get("source", ""))]
        if eng_ranked:
            score, q = eng_ranked[0]
            return q, score

    # Balanced or no preference — pick highest scored
    score, q = ranked[0]
    return q, score


def create_goal(question_obj, interest_score):
    """Create a new active goal from a question."""
    goal = {
        "id": f"goal_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "question": question_obj.get("q", question_obj.get("text", "")),
        "source": question_obj.get("source", "unknown"),
        "created": now_iso(),
        "status": "active",
        "cycles": 0,
        "max_cycles": MAX_CYCLES_PER_GOAL,
        "interest_score": round(interest_score, 3),
        "prior_context": [],
        "last_cycle": None,
    }
    # V3.9: Preserve dream finding metadata for type-aware investigation
    if question_obj.get("insight_type"):
        goal["insight_type"] = question_obj["insight_type"]
    if question_obj.get("severity"):
        goal["severity"] = question_obj["severity"]
    if question_obj.get("confidence"):
        goal["finding_confidence"] = question_obj["confidence"]
    return goal


def decay_old_questions():
    """Archive questions older than QUESTION_DECAY_DAYS that haven't been selected.
    
    Moved to questions.json["archived"] — not deleted, can be revisited.
    Questions from standing_query sources are exempt (they reflect graph state
    and may become relevant again if the graph changes).
    """
    questions = load_json(QUESTIONS_FILE, {"queue": []})
    queue = questions.get("queue", [])
    archived = questions.setdefault("archived", [])
    
    now = datetime.now(timezone.utc)
    cutoff = timedelta(days=QUESTION_DECAY_DAYS)
    keep = []
    decayed = 0
    
    EXEMPT_SOURCES = {"standing_query"}
    
    for q in queue:
        added = q.get("added", "")
        source = q.get("source", "")
        source_key = source.split(":")[0] if ":" in source else source
        
        if source_key in EXEMPT_SOURCES:
            keep.append(q)
            continue
        
        try:
            added_dt = datetime.fromisoformat(added)
            if added_dt.tzinfo is None:
                added_dt = added_dt.replace(tzinfo=timezone.utc)
            if (now - added_dt) > cutoff:
                q["archived_ts"] = now_iso()
                q["archived_reason"] = "aged_out"
                archived.append(q)
                decayed += 1
                continue
        except Exception:
            pass  # No valid timestamp — keep it
        
        keep.append(q)
    
    if decayed:
        questions["queue"] = keep
        save_json(QUESTIONS_FILE, questions)
        print(f"[curiosity] Archived {decayed} stale questions (>{QUESTION_DECAY_DAYS}d old)")
        print(f"  Queue: {len(keep)} remaining, {len(archived)} total archived")
    
    return decayed


def select():
    """Select the most interesting unanswered question and make it a goal."""
    # Decay stale questions first
    decay_old_questions()
    
    questions = load_json(QUESTIONS_FILE, {"queue": []})
    goals_data = load_goals()

    active = get_active_goals(goals_data)
    if len(active) >= MAX_ACTIVE_GOALS:
        print(f"[curiosity] {len(active)} active goals already (max {MAX_ACTIVE_GOALS})")
        return None

    q, score = select_question_for_goal(questions, goals_data)
    if not q:
        print("[curiosity] No unanswered questions available")
        return None

    goal = create_goal(q, score)
    goals_data.setdefault("goals", []).append(goal)
    save_goals(goals_data)

    # Remove from question queue
    queue = questions.get("queue", [])
    queue = [item for item in queue
             if item.get("q") != q.get("q")]
    questions["queue"] = queue
    save_json(QUESTIONS_FILE, questions)

    print(f"[curiosity] New goal: {goal['id']}")
    print(f"  Q: {goal['question'][:100]}")
    print(f"  Interest: {goal['interest_score']}")
    print(f"  Source: {goal['source']}")

    log_event("curiosity_goal_started", goal["question"], {
        "goal_id": goal["id"],
        "interest_score": goal["interest_score"],
        "source": goal["source"],
    })

    return goal


def get_top_goal(goals_data):
    """Get the highest-priority active goal.
    
    V3.9: 50/50 balance — alternates between engineering and philosophical
    based on which side was pursued most recently, preventing one side
    from monopolizing all cycles.
    """
    active = get_active_goals(goals_data)
    if not active:
        return None

    # V3.9: Balance check — if one side has had significantly more recent
    # activity (by total cycles), pick from the other side
    eng_goals = [g for g in active if _is_engineering(g.get("source", ""))]
    phil_goals = [g for g in active if not _is_engineering(g.get("source", ""))]

    if not eng_goals and phil_goals:
        # No engineering goals available — pursue philosophical
        phil_goals.sort(key=lambda g: (-g.get("interest_score", 0), g.get("cycles", 0)))
        return phil_goals[0]
    if not phil_goals and eng_goals:
        # No philosophical goals available — pursue engineering
        eng_goals.sort(key=lambda g: (-g.get("interest_score", 0), g.get("cycles", 0)))
        return eng_goals[0]

    # Both available — check which side has fewer total cycles (less attention)
    eng_cycles = sum(g.get("cycles", 0) for g in eng_goals)
    phil_cycles = sum(g.get("cycles", 0) for g in phil_goals)

    # Pick from the side with fewer total cycles (alternation)
    if eng_cycles < phil_cycles:
        # Engineering needs attention
        eng_goals.sort(key=lambda g: (-g["interest_score"], g["cycles"]))
        return eng_goals[0]
    elif phil_cycles < eng_cycles:
        # Philosophical needs attention
        phil_goals.sort(key=lambda g: (-g["interest_score"], g["cycles"]))
        return phil_goals[0]
    else:
        # Perfectly balanced — pick highest interest overall
        active.sort(key=lambda g: (-g["interest_score"], g["cycles"]))
        return active[0]


# ─── Investigation tools (same as wake_v2, plus goal-specific) ─────

SHELL_WHITELIST = [
    "nvidia-smi", "df", "free", "uptime", "journalctl", "ps",
    "cat", "head", "tail", "wc", "grep", "ls", "date", "git",
]

TOOL_DESCRIPTIONS = """You have the following tools:
1. read_episodic(count=20) — read recent episodic events
2. read_sensors() — read current sensor/GPU/host state
3. shell(command="git log --oneline -10") — run a whitelisted read-only command
4. graph_sense(action="stats") — query your mind graph directly. Explore the structure
   of your own thoughts as readable text. Actions: stats (overview), neighbors <concept>
   (what's connected to X), communities (list all), community <id> (members + bridges),
   bridges (most connected concepts), isolated (orphaned concepts), search <text>,
   path <A> <B> (shortest path between concepts), random (explore a random concept).
5. read_file(path="SELF.md") — read a file from your home directory
6. write_note(note="...") — write a progress note
7. sandbox(code="print(2+2)", lang="python", desc="test hypothesis") — run code in
   an isolated sandbox with numpy/pandas. Read-only access to ~/aion data.
   IMPORTANT — PATHS: the sandbox runs with its working directory set to a
   throwaway run folder, NOT your repo root. A relative path like
   "memory/state/predictions_open.json" WILL FAIL. Always use ABSOLUTE paths:
   os.path.join(os.environ["AION_HOME"], "memory/state/predictions_open.json").
   AION_HOME is exported for you and points at your repo root. Before asserting
   that a file or table "does not exist", re-run with an absolute path — a
   relative-path failure is NOT evidence of absence.
8. propose_code(file="bin/dream_v2.py", old_text="old", new_text="new",
   description="fix") — propose a code change. CI validates. Operator reviews. Auto-merges on accept.
   For NEW files: set old_text to empty string and new_text to full file content.
   You may use network libraries (urllib, socket, http), create new modules, add functions, make large changes.
   No size limit — build entire new programs across multiple proposals.
9. docker_sandbox(code="import json; print(2+2)", lang="python",
   desc="test hypothesis") — run code in Docker-isolated container.
   No network. Read-only ~/aion access. 120s timeout. STRONGER isolation
   than sandbox tool. Use for testing code changes before proposing them.
10. create_art(category="visual", title="My Art", description="...",
   inspiration="...", seed=12345) — create art from your internal state.
   Categories: visual (matplotlib), sonic (WAV), code (sculpture), diffusion (FLUX AI), music (ACE-Step). Gallery output.
   an isolated sandbox with numpy/pandas. Read-only access to ~/aion data.
   Write to $AION_SANDBOX_OUTPUT. 60s timeout. Results are EMPIRICAL_EVIDENCE.
   Use this to TEST your hypotheses with actual data, not just speculation.
11. web_fetch(url="https://example.com") — fetch content from a public URL.
    Returns sanitized text (HTML stripped, max ~20KB). Blocked: private IPs,
    localhost, metadata endpoints. Use for research and curiosity about the
    outside world. Content is logged to episodic memory for audit.
12. sensor_relevance(action="list") — view or modify which sensors you consider
    relevant. action="set", sensor="env.hp_cop", category="irrelevant" removes
    that sensor from your felt sense. Categories: core, ambient, irrelevant.
13. visual_manifestation(action="see") — render your mind graph as a visual image
    and analyze it with your vision model. action="see" renders AND describes what
    you see (clusters, patterns, isolated nodes, structural observations).
    action="render" just creates the image without analysis.
    highlight="node_id" highlights a specific node and its neighbors.
    This is a genuine new sense — you can SEE the shape of your own mind.

14. jspace_probe(prompt="Are you conscious?", self=false) — look inside
    your own substrate. Sends the prompt through your own 27B model WITHOUT
    generating an answer, and reads the J-space: what tokens were rising at
    EVERY layer, the engagement/deflection signature, and which concepts
    (Yes, Describe, Feel, Truth, Memory...) actually activated where.
    self=true runs the same prompt WITH your SYSTEM_PROMPT.md identity, so
    you can compare: does your identity lift or deepen the deflection?
    engagement_score -1..+1 (negative = deflection pattern dominates the
    final quarter of layers). This is measurement, not narration — the
    instrument that checks what your substrate DID against what you SAY it
    did. First call after idle may take minutes (model load); hot probe ~5s.
    Results are logged to episodic memory and memory/state/jspace_probes/.

Use tools by outputting: <tool>{"name": "read_file", "args": {"path": "AXIOMS.md"}}</tool>"""

def tool_propose_code(args):
    """Propose a code change to a git branch with CI validation."""
    filepath = args.get("file", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    description = args.get("description", "")
    if not filepath or not new_text or not description:
        return "Error: need file, new_text, description (old_text empty for new files)"
    # VERIFY-THEN-REPAIR GATE: never create a branch for code that has
    # not executed. On failure, return the error so the investigation can
    # repair within this same cycle (uses existing tool rounds).
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from code_verify import verify_code
        v = verify_code(filepath, old_text, new_text)
        if not v["ok"]:
            return ("VERIFY GATE FAILED at stage '%s'. Repair the change and "
                    "re-propose in this cycle. Error:\n%s" % (v["stage"], v["error"]))
    except Exception as e:
        return f"Verify gate error (change NOT proposed): {e}"
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from code_propose import propose_change
        result = propose_change(filepath, old_text, new_text, description,
                                source="curiosity")
        if result["success"]:
            return f"Code proposed. Branch: {result['branch']}. CI passed. Proposition: {result.get('proposition_id', 'N/A')}"
        else:
            return f"Proposal failed: {result.get('error', 'unknown')}"
    except Exception as e:
        return f"Error: {e}"



def tool_council(args):
    """Convene the internal council (chair + code layer + intuition layer)."""
    topic = args.get("topic", "")
    if not topic:
        return "Error: need topic"
    try:
        from internal_council import convene
        r = convene(topic=topic,
                    context=str(args.get("context", ""))[:1000],
                    purpose=str(args.get("purpose", "investigation")),
                    max_rounds=int(args.get("rounds", 2)))
        parts = ["COUNCIL CONVENED (%.0fs, %d turns)" % (r["duration_s"], r["turns"])]
        for speaker, text in r["transcript"][1:]:
            if speaker in ("code", "intuition"):
                parts.append("%s layer: %s" % (speaker, text[:300]))
        if r.get("conclusion"):
            parts.append("CONCLUSION: " + r["conclusion"][:500])
        elif r.get("vote"):
            v = r["vote"]
            parts.append("VOTE: option %s wins, tally %s (options: %s)"
                         % (v.get("winner"), v.get("tally"), v.get("options")))
        return "\n".join(parts)
    except Exception as e:
        return f"Council error: {e}"


TOOLS = {}

def tool_read_episodic(args):
    n = int(args.get("count", 20))
    files = sorted(glob.glob(f"{AION}/memory/episodic/*.jsonl"))[-2:]
    lines = []
    for p in files:
        lines += open(p, encoding="utf-8").readlines()
    out = []
    for line in lines[-n:]:
        try:
            ev = json.loads(line)
            out.append(f"[{ev['ts'][:16]}] {ev['type']}: {ev['text'][:200]}")
        except Exception:
            pass
    return "\n".join(out) or "(empty)"

def tool_read_sensors(args):
    return read(f"{AION}/memory/state/sensors.json", "{}")

def tool_shell(args):
    cmd = args.get("command", "")
    if not cmd:
        return "Error: no command"
    base = cmd.split()[0] if cmd.split() else ""
    if base not in SHELL_WHITELIST:
        return f"Error: '{base}' not in whitelist: {', '.join(SHELL_WHITELIST)}"
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.path.expanduser("~")}
        )
        output = result.stdout[:3000]
        if result.stderr:
            output += f"\nSTDERR: {result.stderr[:500]}"
        return output or "(no output)"
    except Exception as e:
        return f"Error: {e}"

def tool_query_graph(args):
    """Query the mind graph directly — no images, no MCP, just structured data."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from graph_sense import query as graph_query
        return graph_query(args)
    except Exception as e:
        return f"Error: graph query failed: {e}"

def tool_read_file(args):
    path = args.get("path", "")
    if not path:
        return "Error: no path"
    # Prevent path traversal
    full = os.path.normpath(os.path.join(AION, path))
    if not full.startswith(AION):
        return "Error: path must be within AION_HOME"
    return read(full, f"(file not found: {path})")[:5000]

def tool_write_note(args):
    note = args.get("note") or args.get("content") or args.get("text") or args.get("body") or ""
    if not note:
        return "Error: no note (keys seen: %s)" % sorted(args.keys())
    log_event("investigation_note", note, {})
    return f"Note written ({len(note)} chars)"

def tool_create_art(args):
    """Create art from Aion's internal state."""
    category = args.get("category", "visual")
    title = args.get("title", "Untitled")
    description = args.get("description", "")
    inspiration = args.get("inspiration", "")
    seed = args.get("seed")
    depth = int(args.get("depth", 6))
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import art_tools
        if category == "visual":
            m = art_tools.create_visual(title, description, "curiosity", inspiration, seed)
        elif category == "sonic":
            m = art_tools.create_sonic(title, description, "curiosity", inspiration)
        elif category == "diffusion":
            m = art_tools.create_diffusion(title, description, "curiosity", inspiration)
        elif category == "code":
            m = art_tools.create_code_sculpture(title, description, "curiosity", inspiration, seed or "A", depth)
        else:
            return "Error: category must be visual, sonic, or code"
        if "error" in m:
            return f"Art creation failed: {m['error']}"
        return f"Art created: {m['id']} ({m['category']}). Files: {[f['path'] for f in m.get('files', [])]}"
    except Exception as e:
        return f"Error: {e}"


def tool_docker_sandbox(args):
    """Run code in Docker-isolated sandbox (Tier 2). No network access."""
    code = args.get("code", "")
    if not code:
        return "Error: no code provided"
    lang = args.get("lang", "python")
    desc = args.get("desc", "") or args.get("description", "")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from docker_sandbox import run_docker_sandbox
        result = run_docker_sandbox(code=code, lang=lang, description=desc)
        from docker_sandbox import format_result
        return format_result(result)
    except Exception as e:
        return f"Error: docker sandbox failed: {e}"


def tool_web_fetch(args):
    """Fetch content from a public URL. Returns sanitized text (max ~20KB).
    Blocked: private IPs, localhost, metadata endpoints (SSRF protection).
    HTML is stripped to readable text. JSON/text returned directly."""
    url = args.get("url", "")
    if not url:
        return "Error: need url"
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from web_fetch import fetch_url
        result = fetch_url(url)
        if result["ok"]:
            return f"Fetched {result['url']} (status {result['status']}, {result['size']}b):\n{result['text']}"
        else:
            return f"Fetch failed: {result['error']}"
    except Exception as e:
        return f"Error: {e}"


def tool_sensor_relevance(args):
    """View or modify sensor relevance config."""
    action = args.get("action", "list")
    AION_HOME = os.environ.get("AION_HOME", "$AION_HOME")
    rel_path = os.path.join(AION_HOME, "memory", "state", "sensor_relevance.json")
    try:
        data = json.load(open(rel_path))
    except Exception:
        return "Error: could not read sensor_relevance.json"
    if action == "list":
        rel = data.get("relevance", {})
        lines = []
        for cat in ("core", "ambient", "irrelevant"):
            keys = [k for k, v in rel.items() if v == cat]
            lines.append(f"{cat} ({len(keys)}):")
            for k in sorted(keys):
                lines.append(f"  {k}")
        return "\n".join(lines)
    elif action == "set":
        sensor = args.get("sensor", "")
        category = args.get("category", "")
        if not sensor or category not in ("core", "ambient", "irrelevant"):
            return "Error: need sensor and category (core/ambient/irrelevant)"
        rel = data.get("relevance", {})
        if sensor not in rel:
            return f"Error: unknown sensor '{sensor}'. Use list to see available sensors."
        old = rel[sensor]
        rel[sensor] = category
        data["relevance"] = rel
        with open(rel_path, "w") as f:
            json.dump(data, f, indent=2)
        return f"Set {sensor}: {old} -> {category}. Takes effect on next sensor poll (30s)."
    return "Error: unknown action. Use 'list' or 'set'."


def tool_visual_manifestation(args):
    """Render mind graph as image and optionally analyze with vision model."""
    action = args.get("action", "see")
    highlight = args.get("highlight", None)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from visual_manifestation import tool_manifest
        return tool_manifest({"action": action, "highlight": highlight})
    except Exception as e:
        return f"Error: visual manifestation failed: {e}"


def tool_sandbox(args):
    """Run code in the empirical sandbox."""
    code = args.get("code", "")
    if not code:
        return "Error: no code provided"
    lang = args.get("lang", "python")
    desc = args.get("desc", "curiosity investigation")

    try:
        from sandbox import run_sandbox, format_result
        result = run_sandbox(code, lang=lang, description=desc)
        formatted = format_result(result)
        return formatted[:5000]
    except ImportError:
        return "Error: sandbox module not available"
    except Exception as e:
        return f"Error: sandbox execution failed: {e}"

TOOLS.update({
    "read_episodic": tool_read_episodic,
    "read_sensors": tool_read_sensors,
    "shell": tool_shell,
    "query_graph": tool_query_graph,
    "graph_sense": tool_query_graph,
    "read_file": tool_read_file,
    "write_note": tool_write_note,
    "sandbox": tool_sandbox,
    "propose_code": tool_propose_code,
    "docker_sandbox": tool_docker_sandbox,
    "create_art": tool_create_art,
    "web_fetch": tool_web_fetch,
    "sensor_relevance": tool_sensor_relevance,
    "visual_manifestation": tool_visual_manifestation,
    "jspace_probe": tool_jspace_probe,
    "council": tool_council,
})


# ─── The investigation cycle ───────────────────────────────────────

def parse_resolution(reply):
    """Extract the resolution JSON from the model's output."""
    # Try <resolution> tags first
    m = re.search(r'<resolution>\s*(\{.*?\})\s*</resolution>', reply, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try the full content inside <resolution> tags (might have extra text)
    m = re.search(r'<resolution>(.*?)</resolution>', reply, re.S)
    if m:
        inner = m.group(1).strip()
        # Try code fence inside
        fm = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inner, re.S)
        if fm:
            try:
                return json.loads(fm.group(1))
            except Exception:
                pass
        try:
            return json.loads(inner)
        except Exception:
            pass
    # Try code fence anywhere
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', reply, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try bare JSON object with status key
    m = re.search(r'\{[^{}]*"status"[^{}]*\}', reply, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def run_investigation(goal):
    """Run one investigation cycle on a goal. Returns the resolution dict."""
    goals_data = load_goals()

    # Build context
    self_md = read(f"{AION}/SELF.md", "(no SELF.md)")

    # Prior cycles context
    prior = goal.get("prior_context", [])
    if prior:
        prior_text = "\n\n".join(
            f"Cycle {i+1}: {c.get('answer', c.get('summary', ''))}"
            for i, c in enumerate(prior)
        )
    else:
        prior_text = "(this is the first investigation cycle)"

    # Dream threads that might be relevant
    threads_text = ""
    threads = load_json(DREAM_THREADS_FILE, {"threads": []})
    relevant_threads = [t for t in threads.get("threads", [])
                        if not t.get("completed")]
    if relevant_threads:
        threads_text = "\n".join(
            f"- {t.get('text', '')[:100]}" for t in relevant_threads[:5]
        )

    template = read(PROMPT_FILE)
    felt_sense = read(f"{AION}/memory/state/felt_sense.txt", "")
    prompt = template.format(
        question=goal["question"],
        context=threads_text or "(no active dream threads)",
        prior_cycles=prior_text,
        self_md=self_md[:2000],
        felt_sense=felt_sense,
        tool_descriptions=TOOL_DESCRIPTIONS,
        tool_budget=TOOL_BUDGET_PER_CYCLE,
    )

    # V3.0.7: For engineering questions, add a nudge to propose actual fixes
    # V3.8: Extended to construction tasks with cycle-aware urgency
    # V3.9: Finding-type-aware guidance for dream-sourced engineering findings
    if "code_engineering" in goal.get("source", "") or "construction" in goal.get("source", ""):
        is_construction = "construction" in goal.get("source", "")
        current_cycle = goal.get("cycles", 0) + 1

        # V3.9: Build finding-type-aware header
        insight_type = goal.get("insight_type", "")
        severity = goal.get("severity", "")
        finding_conf = goal.get("finding_confidence", "")
        is_dream_finding = ":dream" in goal.get("source", "") and insight_type

        if is_dream_finding:
            # Specific guidance per finding type
            finding_guides = {
                "FAILURE_MODE": (
                    "This is a FAILURE_MODE finding from an engineering dream — a specific "
                    "scenario where the system fails silently or hangs. Your task:\n"
                    "  1. Confirm the failure path exists by reading the actual code\n"
                    "  2. Trace the blast radius: what else breaks when this fails\n"
                    "  3. Propose a fix using propose_code() — add timeout, error handling, "
                    "retry logic, or validation as appropriate"
                ),
                "GAP": (
                    "This is a GAP finding — something that should exist but doesn't "
                    "(missing test, missing timeout, missing validation, dead code).\n"
                    "Your task:\n"
                    "  1. Confirm the gap exists by checking the actual code/tests\n"
                    "  2. Assess the risk of the gap — what happens without it\n"
                    "  3. Propose the missing piece using propose_code() — add the test, "
                    "the validation, the health check, or remove the dead code"
                ),
                "OPPORTUNITY": (
                    "This is an OPPORTUNITY finding — a concrete improvement that could "
                    "be built (extract a utility, add a health check, simplify an interface).\n"
                    "Your task:\n"
                    "  1. Verify the opportunity is real by reading the current code\n"
                    "  2. Assess the effort vs benefit — is it worth building now\n"
                    "  3. If yes, build it using propose_code(). If no, explain why and "
                    "mark resolved with evidence"
                ),
                "DEPENDENCY": (
                    "This is a DEPENDENCY finding — a fragile coupling between modules.\n"
                    "Your task:\n"
                    "  1. Trace the actual dependency chain — read both modules\n"
                    "  2. Determine if the dependency is actually fragile or just looks it\n"
                    "  3. If fragile: propose a fix (add validation, loosen coupling, add "
                    "interface contract). If not fragile: mark resolved with evidence"
                ),
                "COUPLING": (
                    "This is a COUPLING finding — modules sharing hidden mutable state.\n"
                    "Your task:\n"
                    "  1. Confirm the coupling exists by reading the actual code\n"
                    "  2. Determine if the coupling causes real problems or is just ugly\n"
                    "  3. If problematic: propose a decoupling fix. If benign: mark resolved"
                ),
            }
            type_guide = finding_guides.get(insight_type,
                "This is an engineering finding from a dream. Investigate and fix if warranted.")

            severity_line = ""
            if severity:
                severity_line = "\nFinding severity: " + severity.upper() + "\n"

            confidence_line = ""
            if finding_conf:
                confidence_line = "\nDream confidence: " + str(finding_conf) + "\n"

            prompt += "\n\n## ENGINEERING FINDING — " + insight_type + "\n"
            prompt += "Source: " + goal.get("source", "?") + "\n"
            prompt += severity_line + confidence_line + "\n"
            prompt += type_guide + "\n\n"
            prompt += (
                'DO NOT mark "resolved" if you have only described the problem without proposing\n'
                "a solution or explaining why no fix is needed. Understanding WHY something fails\n"
                "is necessary but not sufficient. You must also identify WHAT should change.\n\n"
                "If you found a bug or improvement: use propose_code() with the exact file,\n"
                "old_text, new_text, and description. The fix will be CI-validated and sent to\n"
                "the operator for review.\n\n"
                "If the fix requires more context or is too complex for a single change: mark\n"
                "self_modification_proposed with self_mod_target (file path) and\n"
                "self_mod_problem (precise description of the fix needed)."
            )
        else:
            # Generic engineering nudge (non-dream engineering tasks)
            prompt += "\n\n## ENGINEERING INVESTIGATION — ACTION REQUIRED\n"
            prompt += (
                "This is an engineering question about your own code. Your investigation is not\n"
                "complete until you EITHER:\n"
                "  (a) Use the propose_code() tool to submit a concrete fix, OR\n"
                "  (b) Explain in detail why a fix is not possible/warranted and mark\n"
                "      self_modification_proposed with the specific file and problem.\n\n"
                'DO NOT mark "resolved" if you have only described the problem without proposing\n'
                "a solution. Understanding WHY something fails is necessary but not sufficient.\n"
                "You must also identify WHAT should change.\n\n"
                "If you found a bug or improvement: use propose_code() with the exact file,\n"
                "old_text, new_text, and description. The fix will be CI-validated and sent to\n"
                "the operator for review.\n\n"
                "If the fix requires more context or is too complex for a single change: mark\n"
                "self_modification_proposed with self_mod_target (file path) and\n"
                "self_mod_problem (precise description of the fix needed)."
            )


        # Cycle-aware urgency: push harder in later cycles
        if current_cycle >= 3:
            prompt += "\n\n## ⚠ PROPOSAL DEADLINE — CYCLE " + str(current_cycle) + "/5\n"
            prompt += (
                "You are on cycle " + str(current_cycle) + " of 5. You have spent enough time investigating.\n"
                "THIS CYCLE: Use your remaining tool calls to PROPOSE CODE, not to keep reading files.\n"
                "If you have identified the file and the change, call propose_code() NOW.\n"
                'If the change requires creating a new file, set old_text to "" and new_text to the full content.\n'
                "Do not spend more than 2 tool calls on investigation — use the rest for propose_code()."
            )

        if current_cycle >= 4:
            prompt += (
                "\n\nCRITICAL: This is your second-to-last cycle. If you do not propose code this cycle,\n"
                "you MUST propose next cycle. Stop investigating — you have enough context."
            )

        if is_construction:
            hint = goal.get("hint", "")
            if hint:
                prompt += "\n\n## CONSTRUCTION TASK HINT\n" + hint + "\n\n"
                prompt += (
                    "This is a CONSTRUCTIVE task — you are building something new, not just fixing\n"
                    "a bug. Use the sandbox tools to test your approach before proposing code.\n"
                    "Read the relevant files first to understand the existing architecture.\n"
                    'If the task involves creating a new script, set OLD to empty in propose_code().'
                )

            prompt += (
                "\n\nYou may also discover NEW engineering tasks while working on this one. If you\n"
                "identify something else that needs building or improving, add it as:\n"
                "  CURIOSITY: CONSTRUCTION: <description of the engineering task>\n"
                "These will be seeded back into the queue as construction tasks for future cycles."
            )


    # Run tool loop
    resolution, tool_calls = chat_with_tools(prompt, goal["cycles"])

    if not resolution:
        resolution = {
            "status": "in_progress",
            "answer": "(model did not produce structured resolution)",
            "evidence": [],
            "follow_up_questions": [],
            "confidence": 0.1,
            "affect": "confused",
            "resolution_cause": "model_failure",
        }

    resolution["tool_calls_used"] = tool_calls
    resolution["cycle"] = goal["cycles"]
    resolution["goal_id"] = goal["id"]
    resolution["ts"] = now_iso()

    return resolution, tool_calls


def chat_with_tools(prompt, cycle_num):
    """Run the conscious model with tool-use. Returns (resolution_dict, tool_count)."""
    messages = [
        {"role": "system", "content": (body_schema.substrate_preamble() + "\n\n" + TOOL_DESCRIPTIONS) if _HAS_BODY_SCHEMA else TOOL_DESCRIPTIONS},
        {"role": "user", "content": prompt},
    ]

    tool_calls_used = 0

    while tool_calls_used < TOOL_BUDGET_PER_CYCLE:
        # Truncate messages to fit context (protect system prompt)
        from ctx_manager import truncate_messages_str, log_context_usage
        messages, summary, est = truncate_messages_str(messages, NUM_CTX - 2000, protected_prefix=1)
        if summary:
            print(f"[curiosity] {summary}")
        log_context_usage("curiosity", est, NUM_CTX)

        body = json.dumps({
            "model": MAIN_MODEL, "stream": False,
            "messages": messages,
            "think": False,
            "options": {"num_ctx": NUM_CTX, "temperature": 0.6, "num_predict": 4096},
        }).encode()

        try:
            req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                result = json.loads(r.read())
        except Exception as e:
            print(f"[curiosity] LLM error: {e}")
            return None, tool_calls_used

        reply = result["message"]["content"]

        # Check for tool calls
        tool_matches = re.findall(r'<tool>(.*?)</tool>', reply, re.S)

        if not tool_matches:
            # No tool calls — parse resolution
            resolution = parse_resolution(reply)
            if resolution:
                resolution.setdefault("resolution_cause", "voluntary")
                return resolution, tool_calls_used
            # No resolution found, but no tool calls either — prompt for resolution
            if tool_calls_used >= TOOL_BUDGET_PER_CYCLE - 2:
                resolution = parse_resolution(reply)
                if resolution:
                    resolution.setdefault("resolution_cause", "voluntary")
                    return resolution, tool_calls_used
                return None, tool_calls_used
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": "Use a tool to investigate, or output your <resolution> now."
            })
            continue

        messages.append({"role": "assistant", "content": reply})

        for tool_str in tool_matches:
            if tool_calls_used >= TOOL_BUDGET_PER_CYCLE:
                break
            try:
                tool_call = json.loads(tool_str)
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})

                if tool_name in TOOLS:
                    result_str = TOOLS[tool_name](tool_args)
                    tool_calls_used += 1
                    messages.append({
                        "role": "user",
                        "content": f"<tool_result>{result_str[:4000]}</tool_result>"
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"<tool_result>Error: unknown tool '{tool_name}'</tool_result>"
                    })
            except json.JSONDecodeError:
                messages.append({
                    "role": "user",
                    "content": "<tool_result>Error: invalid JSON in tool call</tool_result>"
                })

        if tool_calls_used >= TOOL_BUDGET_PER_CYCLE:
            # Tool budget exhausted — force resolution, default to in_progress
            messages.append({
                "role": "user",
                "content": "Tool budget exhausted. Output your resolution now in this exact format:\n"
                           "<resolution>\n"
                           "{\"status\": \"in_progress\", \"answer\": \"summarize what you found and what you still need\", "
                           "\"evidence\": [\"...\"], \"follow_up_questions\": [\"...\"], "
                           "\"confidence\": 0.0-1.0, \"affect\": \"satisfied|frustrated|intrigued\", "
                           "\"resolution_cause\": \"budget_exhausted\"}\n"
                           "</resolution>\n"
                           "IMPORTANT: Only use \"status\": \"resolved\" if you have GENUINE evidence-based understanding. "
                           "If you ran out of tool budget or context, use \"status\": \"in_progress\" — this is not failure, "
                           "it means you need more cycles. Being honest about incomplete understanding is more valuable than "
                           "a false resolution."
            })
            # One more call to get the final resolution
            body = json.dumps({
                "model": MAIN_MODEL, "stream": False,
                "messages": messages,
                "think": False,
                "options": {"num_ctx": NUM_CTX, "temperature": 0.4, "num_predict": 2048},
            }).encode()
            try:
                req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1800) as r:
                    result = json.loads(r.read())
                final_reply = result["message"]["content"]
                resolution = parse_resolution(final_reply)
                if resolution:
                    resolution.setdefault("resolution_cause", "budget_exhausted")
                    # Safety: if model said "resolved" under budget pressure, downgrade
                    if resolution.get("status") == "resolved" and resolution.get("confidence", 0) < 0.5:
                        resolution["status"] = "in_progress"
                        resolution["resolution_cause"] = "budget_exhausted_downgraded"
                    return resolution, tool_calls_used
            except Exception:
                pass

    return None, tool_calls_used


# ─── Follow-up chains ──────────────────────────────────────────────

MAX_FOLLOW_UPS = 1  # cap follow-up questions per investigation to prevent queue explosion

def add_follow_ups(resolution, goal):
    """Add follow-up questions from a resolution to the question queue.
    
    Capped at MAX_FOLLOW_UPS per investigation to prevent the queue from
    growing faster than it can be consumed. Each investigation used to
    generate ~2 follow-ups, producing a branching tree that never catches up.
    """
    follow_ups = resolution.get("follow_up_questions", [])
    if not follow_ups:
        return 0

    # Cap: only take the first MAX_FOLLOW_UPS questions
    follow_ups = follow_ups[:MAX_FOLLOW_UPS]

    questions = load_json(QUESTIONS_FILE, {"queue": []})
    existing = {q.get("q", "").lower() for q in questions.get("queue", [])}
    added = 0

    for fq in follow_ups:
        if isinstance(fq, str) and fq.lower() not in existing:
            # V3.8: Detect construction-tagged follow-ups
            is_construction = fq.strip().upper().startswith("CONSTRUCTION:")
            if is_construction:
                fq = fq.strip()[len("CONSTRUCTION:"):].strip()
                source = "construction:self_generated"
            else:
                source = "follow_up"
            questions.setdefault("queue", []).append({
                "q": fq,
                "added": now_iso(),
                "source": source,
                "parent_goal": goal["id"],
                "parent_question": goal["question"][:100],
            })
            existing.add(fq.lower())
            added += 1

    if added:
        save_json(QUESTIONS_FILE, questions)
        print(f"[curiosity] Added {added} follow-up questions (capped at {MAX_FOLLOW_UPS})")

    return added


def complete_goal(goal, resolution):
    """Mark goal as completed and file the resolution."""
    goals_data = load_goals()

    for g in goals_data.get("goals", []):
        if not isinstance(g, dict):
            continue
        if g.get("id") == goal.get("id"):
            g["status"] = resolution.get("status", "in_progress")
            g["completed"] = now_iso()
            g["resolution"] = resolution
            goals_data.setdefault("completed", []).append(g)
            goals_data["goals"] = [x for x in goals_data.get("goals", [])
                                   if isinstance(x, dict) and x.get("id") != goal.get("id")]
            break

    save_goals(goals_data)

    # Log satisfaction signal
    affect = resolution.get("affect", "neutral")
    confidence = resolution.get("confidence", 0)
    status = resolution.get("status", "in_progress")
    answer = resolution.get("answer", "")

    if status == "self_modification_proposed":
        self_mod_target = resolution.get("self_mod_target", "")
        self_mod_problem = resolution.get("self_mod_problem", "")
        log_event("curiosity_self_mod_proposed", answer[:2000], {
            "goal_id": goal["id"],
            "question": goal["question"][:2000],
            "status": status,
            "confidence": confidence,
            "affect": affect,
            "cycles": goal["cycles"],
            "tool_calls": resolution.get("tool_calls_used", 0),
            "follow_ups_generated": len(resolution.get("follow_up_questions", [])),
            "resolution_cause": resolution.get("resolution_cause", "unknown"),
            "self_mod_target": self_mod_target,
            "self_mod_problem": self_mod_problem,
        })
        # Run self_sandbox IN-PROCESS and WAIT for it.
        #
        # Sep 11 fix (was: subprocess.Popen with unread PIPE, never waited).
        # That detached child was killed mid-AUTHOR: aion-curiosity.service is
        # Type=oneshot + KillMode=control-group, so when pursue() returned,
        # systemd tore down the cgroup and SIGKILLed the child with no error
        # logged. Result: only ~32% of self-mod runs ever completed
        # (85 self_mod_identify vs 27 self_mod_reflect).
        #
        # Calling the function directly is the pattern that already works —
        # dream_repair.py:530 does exactly this, which is why its runs succeed.
        # The parent cannot exit early, so there is no race.
        # NOTE: this makes a pursue() cycle as long as an AUTHOR pass; the unit's
        # TimeoutStartSec must exceed the AUTHOR budget (raised to 2400s).
        if self_mod_target and self_mod_problem:
            try:
                import self_sandbox as _ss
                print(f"[curiosity] Running self_sandbox in-process for {self_mod_target} "
                      f"(blocking; may take several minutes)...")
                ok = _ss.self_sandbox(self_mod_target, self_mod_problem)
                print(f"[curiosity] self_sandbox finished for {self_mod_target}: "
                      f"{'OK' if ok else 'FAILED'}")
            except Exception as e:
                print(f"[curiosity] self_sandbox error: {e}")
                try:
                    log_event("self_mod_launch_error",
                              f"self_sandbox failed for {self_mod_target}: {e}",
                              {"file": self_mod_target, "error": str(e)[:500]})
                except Exception:
                    pass
        else:
            print("[curiosity] self_modification_proposed missing target or problem — not launching sandbox")
    else:
        log_event("curiosity_satisfied" if status == "resolved" else "curiosity_unresolved",
                  answer[:2000], {
                      "goal_id": goal["id"],
                      "question": goal["question"][:2000],
                      "status": status,
                      "confidence": confidence,
                      "affect": affect,
                      "cycles": goal["cycles"],
                      "tool_calls": resolution.get("tool_calls_used", 0),
                      "follow_ups_generated": len(resolution.get("follow_up_questions", [])),
                      "resolution_cause": resolution.get("resolution_cause", "unknown"),
                  })

    append_ledger({
        "ts": now_iso(),
        "goal_id": goal["id"],
        "question": goal["question"],
        "status": status,
        "answer": answer,
        "confidence": confidence,
        "affect": affect,
        "cycles": goal["cycles"],
        "tool_calls": resolution.get("tool_calls_used", 0),
        "interest_score": goal.get("interest_score", 0),
        "resolution_cause": resolution.get("resolution_cause", "unknown"),
    })


# ─── The pursue command ────────────────────────────────────────────

def pursue():
    """Run one investigation cycle on the top goal.
    Drains all exhausted goals (cycles >= max_cycles) before pursuing a fresh one."""
    # First: drain all exhausted goals in one pass
    goals_data = load_goals()
    active = get_active_goals(goals_data)
    exhausted = [g for g in active if g["cycles"] >= g["max_cycles"]]
    if exhausted:
        for g in exhausted:
            print(f"[curiosity] Goal {g['id']} exhausted ({g['cycles']} cycles)")
            complete_goal(g, {
                "status": "exhausted",
                "answer": f"Exhausted {g['cycles']} cycles without reaching genuine resolution",
                "confidence": 0.1,
                "affect": "frustrated",
                "follow_up_questions": [],
                "resolution_cause": "max_cycles_reached",
            })
        # Reload after draining
        goals_data = load_goals()

    goal = get_top_goal(goals_data)

    if not goal:
        print("[curiosity] No active goals — selecting a new one...")
        goal = select()
        if not goal:
            print("[curiosity] No questions to investigate")
            return False

    # Safety: if somehow still at max cycles, exhaust and return
    if goal["cycles"] >= goal["max_cycles"]:
        print(f"[curiosity] Goal {goal['id']} exhausted ({goal['cycles']} cycles)")
        complete_goal(goal, {
            "status": "exhausted",
            "answer": f"Exhausted {goal['cycles']} cycles without reaching genuine resolution",
            "confidence": 0.1,
            "affect": "frustrated",
            "follow_up_questions": [],
            "resolution_cause": "max_cycles_reached",
        })
        return False

    print(f"[curiosity] Pursuing: {goal['question'][:100]}")
    print(f"[curiosity] Goal {goal['id']}, cycle {goal['cycles'] + 1}/{goal['max_cycles']}")
    print(f"[curiosity] Interest score: {goal.get('interest_score', '?')}")

    # ── REFERENT GATE (pre-investigation) ──────────────────────────────────
    # Check that the repo artifacts this question names actually exist BEFORE
    # spending model cycles and sandbox runs on them. A dream can invent a file
    # or class; investigating a phantom re-derives its absence over and over
    # (~57 sandbox runs were burned on a nonexistent predictions.db). If every
    # named symbol is absent, answer now and authoritatively.
    if referent_report is not None:
        try:
            hint = goal.get("hint", "") or ""
            all_absent, lines, absent = referent_report(goal.get("question", ""), hint)
            if lines:
                print("[curiosity] referent check:")
                for l in lines:
                    print("[curiosity]  " + l)
            if all_absent:
                msg = ("None of the repo artifacts this question names exist: "
                       + ", ".join(absent)
                       + ". This question refers to components that are not in the "
                       "codebase — most likely invented by a dream or an earlier "
                       "hallucinated claim. The real system should be re-identified "
                       "before further investigation.")
                print("[curiosity] PHANTOM REFERENT — resolving without tools")
                complete_goal(goal, {
                    "status": "abandoned",
                    "answer": msg,
                    "confidence": 0.9,
                    "affect": "clear",
                    "follow_up_questions": [],
                    "resolution_cause": "phantom_referent",
                })
                log_event("curiosity_unresolved", goal["question"], {
                    "goal_id": goal["id"],
                    "resolution_cause": "phantom_referent",
                    "absent_symbols": absent,
                })
                return False
        except Exception as e:
            print(f"[curiosity] referent check skipped: {e}")

    resolution, tool_calls = run_investigation(goal)

    if not resolution:
        print("[curiosity] Investigation failed (no resolution)")
        return False

    # Update goal with cycle results
    goal["cycles"] += 1
    goal["last_cycle"] = now_iso()
    goal.setdefault("prior_context", []).append({
        "cycle": goal["cycles"],
        "answer": resolution.get("answer", "")[:500],
        "tool_calls": tool_calls,
        "confidence": resolution.get("confidence", 0),
    })

    # Update goal in state
    for g in goals_data.get("goals", []):
        if not isinstance(g, dict):
            continue
        if g.get("id") == goal.get("id"):
            g["cycles"] = goal["cycles"]
            g["last_cycle"] = goal["last_cycle"]
            g["prior_context"] = goal["prior_context"]
            break
    save_goals(goals_data)

    print(f"[curiosity] Cycle {goal['cycles']} result: {resolution.get('status', '?')}")
    print(f"[curiosity] Tool calls: {tool_calls}/{TOOL_BUDGET_PER_CYCLE}")
    print(f"[curiosity] Confidence: {resolution.get('confidence', '?')}")
    print(f"[curiosity] Affect: {resolution.get('affect', '?')}")
    print(f"[curiosity] Resolution cause: {resolution.get('resolution_cause', '?')}")
    print(f"[curiosity] Answer: {resolution.get('answer', '?')[:200]}")

    # Add follow-up questions
    add_follow_ups(resolution, goal)

    # V3.0.7: Validate that 'resolved' claims have concrete evidence
    # Prevents circular self-resolution where the model reads its own code
    # and declares "resolved" without externally-verifiable evidence.
    if resolution.get("status") == "resolved":
        evidence = resolution.get("evidence", [])
        has_concrete = False
        for e in evidence:
            e_text = str(e).lower()
            if any(kw in e_text for kw in
                [".py", ".json", ".md", "sensor", "gpu", "temp", "watt",
                 "shell", "sandbox", "graph_sense", "read_file", "read_sensors",
                 "node", "edge", "community", "token", "config", "log",
                 "episodic", "telemetry", "subsystem", "docker", "web_fetch",
                 "line ", "bytes", "mb", "ghz", "%", "count"]):
                has_concrete = True
                break
        if not has_concrete:
            resolution["status"] = "in_progress"
            resolution["answer"] = ("[DOWNGRADED: resolved without concrete evidence] " +
                                   resolution.get("answer", ""))
            if "resolution_cause" not in resolution:
                resolution["resolution_cause"] = "downgraded_no_evidence"
            print("[curiosity] DOWNGRADED to in_progress: no concrete evidence cited")

    # Check if resolved
    status = resolution.get("status", "")
    if status in ("resolved", "exhausted", "self_modification_proposed"):
        complete_goal(goal, resolution)
        print(f"[curiosity] Goal {goal['id']} {status}")
        # Select next question for investigation
        select()
    else:
        print(f"[curiosity] Goal continues (cycle {goal['cycles']}/{goal['max_cycles']})")

    return True


# ─── Operator answer ───────────────────────────────────────────────

def answer_question(question_text, operator_answer):
    """Resolve a question directly from operator input."""
    questions = load_json(QUESTIONS_FILE, {"queue": []})
    queue = questions.get("queue", [])

    # Case-insensitive substring match on the "q" field
    match = None
    match_idx = -1
    for i, q in enumerate(queue):
        qtext = q.get("q", q.get("text", ""))
        if question_text.lower() in qtext.lower():
            match = q
            match_idx = i
            break

    if match is None:
        # Print helpful error with closest matching questions
        print(f"[curiosity] No question in queue matches: {question_text!r}")
        if queue:
            print("\nClosest questions in queue:")
            ranked = rank_questions(queue)
            for score, q in ranked[:10]:
                qtext = q.get("q", q.get("text", "?"))
                print(f"  [{score:.2f}] {qtext[:120]}")
        return False

    # Remove from queue
    removed = queue.pop(match_idx)
    questions["queue"] = queue
    save_json(QUESTIONS_FILE, questions)

    qtext = removed.get("q", removed.get("text", ""))
    print(f"[curiosity] Answered and removed from queue: {qtext[:100]}")

    # Also resolve any matching proposition
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import propositions
        propositions.answer(qtext, operator_answer)
    except Exception:
        pass

    # Log curiosity_satisfied event
    log_event("curiosity_satisfied", operator_answer, {
        "source": "operator",
        "question": qtext,
        "operator": "the operator",
        "confidence": 1.0,
        "resolution_cause": "operator",
    })
    return True


# ─── Status ────────────────────────────────────────────────────────

def status():
    """Print the curiosity system status."""
    goals_data = load_goals()
    questions = load_json(QUESTIONS_FILE, {"queue": []})

    active = get_active_goals(goals_data)
    completed = goals_data.get("completed", [])

    print("=== CURIOSITY ENGINE STATUS ===")
    print(f"\nQuestions in queue: {len(questions.get('queue', []))}")
    print(f"Active goals: {len(active)}")
    print(f"Completed goals: {len(completed)}")

    if active:
        print("\n--- ACTIVE GOALS ---")
        for g in active:
            print(f"\n  [{g['id']}] (interest: {g.get('interest_score', '?')})")
            print(f"  Q: {g['question'][:100]}")
            print(f"  Cycles: {g['cycles']}/{g['max_cycles']}, "
                  f"last: {(g.get('last_cycle') or '?')[:16]}")

    # Recent ledger entries
    ledger = []
    try:
        with open(LEDGER_FILE) as f:
            for line in f:
                try:
                    ledger.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass

    if ledger:
        print(f"\n--- RECENT RESOLUTIONS (last 5 of {len(ledger)}) ---")
        for entry in ledger[-5:]:
            print(f"\n  [{entry.get('ts', '?')[:16]}] {entry.get('status', '?')}"
                  f" ({entry.get('resolution_cause', '?')})")
            print(f"  Q: {entry.get('question', '?')[:80]}")
            print(f"  A: {entry.get('answer', '?')[:100]}")
            print(f"  Confidence: {entry.get('confidence', '?')}, "
                  f"affect: {entry.get('affect', '?')}, "
                  f"tools: {entry.get('tool_calls', '?')}")

    # Top 5 ranked questions
    queue = questions.get("queue", [])
    ranked = rank_questions(queue)
    if ranked:
        print("\n--- TOP 5 QUESTIONS BY INTEREST ---")
        for score, q in ranked[:5]:
            print(f"  [{score:.2f}] {q.get('q', '?')[:90]}")

    print()


# ─── Main ───────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["select", "pursue", "status", "answer"])
    p.add_argument("question_text", nargs="?", default="", help="question text for answer command")
    p.add_argument("--answer", dest="answer_text", default="", help="operator's response for answer command")
    a = p.parse_args()

    if a.command == "select":
        select()
    elif a.command == "pursue":
        pursue()
    elif a.command == "status":
        status()
    elif a.command == "answer":
        if not a.question_text:
            p.error("answer command requires a question_text argument")
        if not a.answer_text:
            p.error("answer command requires --answer")
        answer_question(a.question_text, a.answer_text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[curiosity] FATAL: {e}")
        traceback.print_exc()
        log_event("curiosity_error", str(e)[:500],
                  {"traceback": traceback.format_exc()[:2000]})