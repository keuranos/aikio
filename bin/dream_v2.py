#!/usr/bin/env python3
"""dream_v2.py — Dream Cycle v2: multi-step graph walk with feedback.

Replaces the single-shot dream() in subconscious.py.

Architecture:
  1. SEED SELECTION — weighted pick from past threads / god-nodes / surprises / random
  2. GRAPH WALK — 3-5 steps, subconscious free-associates at each node
  3. SYNTHESIS — conscious model reviews the full chain, extracts insights
  4. FEEDBACK — insights become questions, self-proposals, or dream threads

Runs on the nightly cycle. Uses subconscious (P40) for walking, conscious (V100)
for synthesis. Cross-dream continuity via threads.json.

Usage:
  python3 dream_v2.py [--steps N] [--dry-run]
"""
import json, os, sys, re, glob, random, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa: F401  loads config/aion.env
try:
    import body_schema
    _HAS_BODY_SCHEMA = True
except Exception:
    _HAS_BODY_SCHEMA = False
import hb

AION = os.environ.get("AION_HOME", "$AION_HOME")
# Dreams use intuition model (V100 #2) for richer free-association
SUB_URL = os.environ.get("OLLAMA_DREAM_URL", os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438"))
SUB_MODEL = os.environ.get("OLLAMA_DREAM_MODEL", os.environ.get("INTUITION_MODEL", "glm-4.7-flash:q4_K_M"))
SUB_MODEL = os.environ.get("SUB_MODEL", "glm-4.7-flash:q4_K_M")
MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")

def _with_substrate(prompt):
    """Prepend substrate preamble to a prompt if body_schema is available."""
    if _HAS_BODY_SCHEMA:
        try:
            return body_schema.substrate_preamble() + "\n\n" + prompt
        except Exception:
            pass
    return prompt



MAIN_NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

DREAMS_DIR = f"{AION}/memory/dreams"
THREADS_FILE = f"{DREAMS_DIR}/threads.json"
GRAPH_FILE = f"{AION}/graphs/mind/graphify-out/graph.json"
BODY_GRAPH_FILE = f"{AION}/graphs/code/graphify-out/graph.json"
QUESTIONS_FILE = f"{AION}/memory/state/questions.json"

# ─── Engineering dream mode (V3.9) ──────────────────────────────────

ENG_INSIGHT_TYPES = ("DEPENDENCY", "FAILURE_MODE", "COUPLING", "GAP", "OPPORTUNITY")


def is_engineering_cycle():
    """Detect if the current curiosity cycle is engineering-heavy.

    Checks active_goals.json and questions.json — if the majority of active
    goals are engineering/construction sourced, we're in an engineering cycle.

    This makes dream composition context-aware: engineering days get engineering
    walk templates instead of philosophical free-association.
    """
    import glob

    # Check active goals
    goals_path = f"{AION}/memory/state/active_goals.json"
    goals = load_json(goals_path, {"goals": []})
    active = [g for g in goals.get("goals", [])
              if g.get("status") == "active"]

    if active:
        eng_count = sum(1 for g in active
                        if "construction" in g.get("source", "")
                        or "engineering" in g.get("source", ""))
        if eng_count > len(active) - eng_count:
            return True

    # Check questions queue for recent engineering dominance
    questions = load_json(QUESTIONS_FILE, {"queue": []})
    queue = questions.get("queue", [])
    if queue:
        eng_qs = sum(1 for q in queue
                     if "construction" in q.get("source", "")
                     or "engineering" in q.get("source", ""))
        if eng_qs > len(queue) - eng_qs:
            return True

    return False
REFLECTIONS_DIR = f"{AION}/memory/reflections"
PROMPTS_DIR = f"{AION}/prompts"
ENG_WALK_TEMPLATE = f"{PROMPTS_DIR}/dream_walk_eng.txt"
ENG_SYNTHESIS_TEMPLATE = f"{PROMPTS_DIR}/dream_synthesis_eng.txt"

MAX_STEPS = int(os.environ.get("DREAM_MAX_STEPS", "5"))
MIN_STEPS = int(os.environ.get("DREAM_MIN_STEPS", "3"))


# ─── Infrastructure ─────────────────────────────────────────────────

def read(p, d=""):
    try:
        return open(p, encoding="utf-8").read()
    except Exception:
        return d

def write(p, t):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)

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

def now_str():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def log_event(type_, text, meta=None):
    subprocess.run(
        ["python3", f"{AION}/bin/log_event.py", "--type", type_,
         "--text", text[:8000], "--meta", json.dumps(meta or {})],
        check=False,
    )

def sub_chat(prompt, temperature=0.9, max_tokens=1200):
    """Call subconscious model. Higher temperature for dream-like associations.

    muse-glimmer (and glm-4.7-flash) use thinking tokens that consume budget;
    we pad the token limit so the actual content gets generated after reasoning.
    V3.9: Increased padding from +2000 to +4000 for muse-glimmer's thinking mode.
    """
    effective_tokens = min(max_tokens + 4000, 8000)
    body = json.dumps({
        "model": SUB_MODEL, "stream": False,
        "messages": [{"role": "user", "content": _with_substrate(prompt)}],
        "options": {
            "num_ctx": 16384, "temperature": temperature,
            "num_predict": effective_tokens,
        },
    }).encode()
    req = urllib.request.Request(
        f"{SUB_URL}/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())["message"]["content"]

def main_chat(prompt, temperature=0.7, max_tokens=3000):
    """Call conscious model for synthesis."""
    from ctx_manager import estimate_tokens, log_context_usage
    est = estimate_tokens(prompt)
    log_context_usage("dream_synthesis", est, MAIN_NUM_CTX)

    body = json.dumps({
        "model": MAIN_MODEL, "stream": False,
        "messages": [{"role": "user", "content": _with_substrate(prompt)}],
        "options": {
            "num_ctx": MAIN_NUM_CTX, "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()
    req = urllib.request.Request(
        f"{MAIN_URL}/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["message"]["content"]


# ─── Graph loading ──────────────────────────────────────────────────

def load_mind_graph():
    """Load mind graph, return {nodes_by_id, nodes_by_label, edges}.

    Graphify stores edges under 'links' key (not 'edges').
    """
    g = load_json(GRAPH_FILE, {"nodes": [], "links": []})
    nodes = g.get("nodes", [])
    edges = g.get("edges", g.get("links", []))
    by_id = {n.get("id", ""): n for n in nodes}
    by_label = {}
    for n in nodes:
        label = n.get("label", n.get("id", ""))
        by_label[label] = n
    return by_id, by_label, edges


def node_provenance(node):
    """Determine where a graph node originated from.

    Returns one of: 'dream', 'self_md', 'episodic', 'axioms', 'unknown'
    """
    sf = node.get("source_file", "") or ""
    sf_lower = sf.lower()
    if "dream" in sf_lower:
        return "dream"
    if "self" in sf_lower or "SELF" in sf:
        return "self_md"
    if "axiom" in sf_lower or "AXIOM" in sf:
        return "axioms"
    if "reflections" in sf_lower or "episodic" in sf_lower:
        return "episodic"
    return "unknown"

def get_neighbors(node_id, edges, nodes_by_id):
    """Get nodes connected to node_id."""
    neighbors = []
    for e in edges:
        src = e.get("source", "")
        tgt = e.get("target", "")
        etype = e.get("type", e.get("label", e.get("relation", "connects")))
        if src == node_id:
            other = nodes_by_id.get(tgt, {})
            neighbors.append({
                "label": other.get("label", other.get("id", tgt)),
                "type": etype,
                "direction": "→",
                "id": tgt,
            })
        elif tgt == node_id:
            other = nodes_by_id.get(src, {})
            neighbors.append({
                "label": other.get("label", other.get("id", src)),
                "type": etype,
                "direction": "←",
                "id": src,
            })
    return neighbors

def format_edges(neighbors):
    """Format neighbor list for the walk prompt."""
    if not neighbors:
        return "(no connections — this is an isolated node)"
    lines = []
    for n in neighbors:
        lines.append(f"  {n['direction']} --{n['type']}--> {n['label']}")
    return "\n".join(lines)


# ─── Thread management (cross-dream continuity) ─────────────────────

def load_threads():
    return load_json(THREADS_FILE, {"threads": [], "completed": []})

def save_threads(data):
    save_json(THREADS_FILE, data)

# Threads older than this, never revisited, are retired on next thread pick.
THREAD_RETIRE_DAYS = int(os.environ.get("DREAM_THREAD_RETIRE_DAYS", "45"))


def pick_open_thread(threads_data, reachable_labels=None):
    """Pick an open thread to continue. Returns thread dict or None.

    Gardening rules (Aug 28):
      - only threads whose seed node still exists in the (14d-pruned) graph
        are selectable — older threads referencing pruned nodes were
        silently unreachable (30% branch no-opped on all 47d threads)
      - threads idle > THREAD_RETIRE_DAYS with zero revisits are retired to
        completed with reason aged_out_never_revisited
    """
    open_threads = [t for t in threads_data.get("threads", [])
                    if not t.get("completed")]
    if not open_threads:
        return None
    now = datetime.now(timezone.utc)
    retired = []
    for t in open_threads:
        try:
            last = t.get("last_revisited") or t.get("created")
            idle_days = (now - datetime.fromisoformat(last)).days
        except Exception:
            idle_days = 0
        if idle_days > THREAD_RETIRE_DAYS and not t.get("revisit_count"):
            t["completed"] = True
            t["completed_ts"] = now_iso()
            t["completed_reason"] = "aged_out_never_revisited"
            retired.append(t)
    if retired:
        save_threads(threads_data)
        print(f"[dream_v2] Retired {len(retired)} stale open threads "
              f"(idle > {THREAD_RETIRE_DAYS}d, never revisited)")
    weighted = []
    for t in open_threads:
        if t.get("completed"):
            continue
        node = t.get("node", "")
        if reachable_labels is not None and node and node not in reachable_labels:
            continue  # seed node pruned from graph — unreachable
        try:
            age_days = (now - datetime.fromisoformat(t["created"])).days
        except Exception:
            age_days = 0
        weight = max(1, min(age_days, 30))
        weighted.extend([t] * weight)
    return random.choice(weighted) if weighted else None


# ─── Seed selection ─────────────────────────────────────────────────

def select_seed(nodes_by_id, nodes_by_label, edges, threads_data, graph_report):
    """Select a seed node to start the walk from.

    Weighted pick:
      30% — past dream threads (cross-dream continuity)
      20% — god nodes (most connected)
      15% — surprising connections from graph report
      25% — warm memory seed
      10% — random node
    """
    roll = random.random()

    # 30%: open dream thread (only threads whose node still exists)
    if roll < 0.30:
        reachable = set(nodes_by_label) | set(nodes_by_id)
        thread = pick_open_thread(threads_data, reachable_labels=reachable)
        if thread:
            seed_node = thread.get("node", "")
            hit = nodes_by_label.get(seed_node) or nodes_by_id.get(seed_node)
            if hit is not None:
                # stamp revisit so idle-based retirement is honest
                thread["last_revisited"] = now_iso()
                thread["revisit_count"] = thread.get("revisit_count", 0) + 1
                save_threads(threads_data)
                return hit, f"thread:{thread.get('text','')[:60]}"

    # 20%: god node (most connected)
    if roll < 0.50:
        conn_count = {}
        for e in edges:
            conn_count[e.get("source", "")] = conn_count.get(e.get("source", ""), 0) + 1
            conn_count[e.get("target", "")] = conn_count.get(e.get("target", ""), 0) + 1
        if conn_count:
            top_ids = sorted(conn_count, key=lambda k: conn_count.get(k, 0), reverse=True)[:8]
            seed_id = random.choice(top_ids)
            if seed_id in nodes_by_id:
                return nodes_by_id[seed_id], f"god_node({conn_count[seed_id]} connections)"

    # 15%: surprising connection from graph report
    if roll < 0.65:
        # Parse surprising connections from GRAPH_REPORT.md
        surprises = parse_surprising_connections(graph_report)
        if surprises:
            # Pick one endpoint as seed
            s = random.choice(surprises)
            for endpoint in [s.get("node_a"), s.get("node_b")]:
                if endpoint and endpoint in nodes_by_label:
                    return nodes_by_label[endpoint], f"surprise:{s.get('node_a','?')}↔{s.get('node_b','?')}"

    # 25%: warm memory seed
    if roll < 0.90:
        try:
            import warm_memory
            seeds = warm_memory.get_dream_seeds(limit=5, min_weight=0.15)
            if seeds:
                seed_entry = random.choice(seeds)
                # Find a graph node matching the seed content
                seed_text = seed_entry.get("content", "")[:200]
                # Try to find a node that appears in the warm memory content
                for label, node in nodes_by_label.items():
                    if label.lower() in seed_text.lower():
                        warm_memory.resonate(seed_entry["id"])
                        return node, f"warm_memory:{seed_entry.get('source','?')}"
                # No matching node found — use as a conceptual seed
                warm_memory.resonate(seed_entry["id"])
                return {"id": "warm_seed", "label": seed_text[:60], "type": "concept",
                        "warm_content": seed_text}, f"warm_memory:{seed_entry.get('source','?')}"
        except Exception:
            pass

    # 10%: random node
    nodes = list(nodes_by_id.values())
    if nodes:
        node = random.choice(nodes)
        return node, "random"

    # Fallback: AXIOMS
    if "AXIOMS" in nodes_by_label:
        return nodes_by_label["AXIOMS"], "fallback:axioms"
    return {"id": "seed", "label": "Existence", "type": "concept"}, "fallback"


def parse_surprising_connections(report_text):
    """Extract surprising connections from GRAPH_REPORT.md."""
    surprises = []
    pattern = re.compile(
        r'`([^`]+)`\s*--(\w+)-->\s*`([^`]+)`\s*\[([^\]]+)\]'
    )
    for m in pattern.finditer(report_text):
        surprises.append({
            "node_a": m.group(1),
            "edge_type": m.group(2),
            "node_b": m.group(3),
            "evidence": m.group(4),
        })
    return surprises


# ─── The walk ───────────────────────────────────────────────────────

def do_walk(seed_node, nodes_by_id, nodes_by_label, edges, max_steps, walk_template):
    """Execute the multi-step graph walk.

    Returns walk_history: list of {step, node_label, node_type, edges, association, next}
    """
    walk = []
    current_id = seed_node.get("id", "")
    current_label = seed_node.get("label", "?")

    walk_template_text = read(walk_template)

    for step in range(1, max_steps + 1):
        node = nodes_by_id.get(current_id, nodes_by_label.get(current_label, seed_node))
        node_label = node.get("label", node.get("id", "?"))
        node_type = node.get("type", "concept")
        neighbors = get_neighbors(current_id, edges, nodes_by_id)

        # Build walk history narrative
        if walk:
            walk_narrative = "\n".join(
                f"  Step {w['step']}: {w['node_label']} → {w['association'][:120]}"
                for w in walk
            )
        else:
            walk_narrative = "(this is the first step)"

        prompt = walk_template_text.format(
            walk_history=walk_narrative,
            current_node=node_label,
            node_type=node_type,
            edges=format_edges(neighbors),
            step_num=step,
            max_steps=max_steps,
        )

        try:
            reply = sub_chat(prompt, temperature=0.9, max_tokens=800)
        except Exception as e:
            print(f"[dream_v2] step {step} LLM error: {e}")
            reply = f"(error during association: {e})"

        # Parse next-node choice
        next_node = parse_next_node(reply, neighbors)
        association = strip_next_line(reply)

        walk.append({
            "step": step,
            "node_id": current_id,
            "node_label": node_label,
            "node_type": node_type,
            "node_provenance": node_provenance(node),
            "num_neighbors": len(neighbors),
            "association": association,
            "next_choice": next_node,
        })

        print(f"[dream_v2] step {step}/{max_steps}: {node_label} → next: {next_node[:50]}")

        # Handle direction
        if next_node.upper().strip() == "EMERGE":
            # Enforce minimum walk length — don't let the model emerge too early
            if step < MIN_STEPS:
                print(f"[dream_v2] step {step}: EMERGE suppressed (min {MIN_STEPS} steps)")
                # Pick a random neighbor to continue the walk
                if neighbors:
                    pick = random.choice(neighbors)
                    current_id = pick["id"]
                    current_label = pick["label"]
                else:
                    break  # Truly stuck — no neighbors
                continue
            break
        if next_node.upper().strip() == "STAY":
            continue  # Same node, dig deeper

        # Find the chosen node
        matched = False
        clean_next = next_node.strip().strip("`*").strip()
        # Also strip markdown bold/italic wrappers like **Node Name**
        if clean_next.startswith("**") and clean_next.endswith("**"):
            clean_next = clean_next[2:-2].strip()
        if clean_next.startswith("*") and clean_next.endswith("*"):
            clean_next = clean_next[1:-1].strip()
        for n in neighbors:
            if clean_next.lower() in n["label"].lower():
                current_id = n["id"]
                current_label = n["label"]
                matched = True
                break
        if not matched:
            # Try direct label/id match
            if clean_next in nodes_by_label:
                current_id = nodes_by_label[clean_next].get("id", clean_next)
                current_label = clean_next
                matched = True
            elif clean_next in nodes_by_id:
                current_label = nodes_by_id[clean_next].get("label", clean_next)
                current_id = clean_next
                matched = True

        if not matched:
            print(f"[dream_v2] could not find '{clean_next[:40]}' — ending walk")
            break

    return walk


def parse_next_node(reply, neighbors):
    """Extract the next-node choice from the last line of the reply."""
    lines = [l.strip() for l in reply.strip().split("\n") if l.strip()]
    if not lines:
        return "EMERGE"
    last = lines[-1]
    # Match "Next node: X" pattern
    m = re.search(r'Next node:\s*(.+)', last, re.I)
    if m:
        raw = m.group(1).strip()
        # Strip markdown formatting: backticks, bold, italic
        raw = raw.strip("`*")
        if raw.startswith("**") and raw.endswith("**"):
            raw = raw[2:-2].strip()
        if raw.startswith("*") and raw.endswith("*"):
            raw = raw[1:-1].strip()
        return raw
    # If the last line is just a node name or keyword
    if last.upper() in ("STAY", "EMERGE"):
        return last.upper()
    # Strip backticks/markdown and check if it matches a neighbor
    clean = last.strip().strip("`*")
    if clean.startswith("**") and clean.endswith("**"):
        clean = clean[2:-2].strip()
    if clean.startswith("*") and clean.endswith("*"):
        clean = clean[1:-1].strip()
    for n in neighbors:
        if n["label"].lower() in clean.lower():
            return n["label"]
    return "EMERGE"

def strip_next_line(reply):
    """Remove the 'Next node:' line from the association text."""
    lines = reply.strip().split("\n")
    cleaned = [l for l in lines if not re.match(r'\s*Next node:', l, re.I)]
    return "\n".join(cleaned).strip()


# ─── Synthesis ──────────────────────────────────────────────────────

def do_synthesis(walk, seed_node, seed_reason, synthesis_template):
    """Conscious model reviews the walk and extracts insights."""
    walk_narrative = "\n\n".join(
        f"**Step {w['step']}: {w['node_label']}** ({w['node_type']}, {w['num_neighbors']} connections)\n"
        f"{w['association']}"
        for w in walk
    )

    template = read(synthesis_template)
    felt_sense = read(f"{AION}/memory/state/felt_sense.txt", "")
    prompt = template.format(
        seed_node=f"{seed_node.get('label', '?')} (chosen because: {seed_reason})",
        walk_narrative=walk_narrative,
        felt_sense=felt_sense,
    )

    try:
        reply = main_chat(prompt, temperature=0.7, max_tokens=3000)
    except Exception as e:
        print(f"[dream_v2] synthesis LLM error: {e}")
        reply = f"(synthesis failed: {e})"

    return reply


# ─── Feedback: dreams → active mind ────────────────────────────────

def extract_insights(synthesis_text):
    """Parse insight JSON blocks from synthesis output."""
    insights = []
    pattern = re.compile(r'```json\s*(\{[^}]+\})\s*```', re.S)
    for m in pattern.finditer(synthesis_text):
        try:
            insight = json.loads(m.group(1))
            insights.append(insight)
        except Exception:
            pass
    return insights

def feedback_to_questions(insights):
    """Add QUESTION-type insights to the curiosity queue."""
    questions = load_json(QUESTIONS_FILE, {"queue": []})
    existing = {q.get("q", "").lower() for q in questions.get("queue", [])}
    added = 0

    for ins in insights:
        if ins.get("type") == "QUESTION":
            q_text = ins.get("text", "")
            if q_text and q_text.lower() not in existing:
                questions.setdefault("queue", []).append({
                    "q": q_text,
                    "added": now_iso(),
                    "source": "dream",
                    "affect": ins.get("affect", "curious"),
                })
                added += 1

    if added:
        save_json(QUESTIONS_FILE, questions)
        print(f"[dream_v2] Added {added} questions to curiosity queue")

    return added

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


def feedback_to_threads(insights, walk, seed_label):
    """Add THREAD-type insights as open dream threads."""
    threads = load_threads()
    added = 0

    for ins in insights:
        if ins.get("type") == "THREAD":
            # Last node visited in the walk
            last_node = walk[-1]["node_label"] if walk else seed_label
            threads.setdefault("threads", []).append({
                "text": ins.get("text", ""),
                "node": last_node,
                "created": now_iso(),
                "source_dream": now_str(),
                "affect": ins.get("affect", "curious"),
                "completed": False,
            })
            added += 1

    if added:
        save_threads(threads)
        print(f"[dream_v2] Added {added} open dream threads")

    return added

def feedback_to_proposals(insights, walk, synthesis_text):
    """Save INSIGHT/CONTRADICTION-type insights as self-proposals."""
    proposals_dir = f"{REFLECTIONS_DIR}/self_proposals"
    os.makedirs(proposals_dir, exist_ok=True)

    proposals = [i for i in insights
                 if i.get("type") in ("INSIGHT", "CONTRADICTION")]
    if not proposals:
        return 0

    ts = now_str()
    path = f"{proposals_dir}/dream_{ts}.md"
    content = f"# Dream self-proposals — {now_iso()}\n\n"
    content += f"Walk seed: {walk[0]['node_label']}\n" if walk else "Walk seed: (simulation)\n"
    content += f"Walk length: {len(walk)} steps\n\n"
    content += "## Proposals\n\n"
    for p in proposals:
        content += f"- **[{p.get('type')}]** {p.get('text', '?')} "
        content += f"(affect: {p.get('affect', '?')}, confidence: {p.get('confidence', '?')})\n"
    content += f"\n## Full synthesis\n\n{synthesis_text}\n"

    write(path, content)
    print(f"[dream_v2] Saved {len(proposals)} self-proposals to {path}")
    return len(proposals)



def feedback_to_construction(insights, walk, synthesis_text, engineering_mode):
    """Route engineering findings to the curiosity queue and construction backlog.

    Engineering insight types (DEPENDENCY, FAILURE_MODE, COUPLING, GAP, OPPORTUNITY)
    don't belong in self-proposals or dream threads. They need to become actionable
    engineering tasks.

    FAILURE_MODE and GAP with high severity -> construction backlog (self_sandbox)
    OPPORTUNITY -> construction backlog
    DEPENDENCY and COUPLING -> curiosity queue as engineering questions
    """
    if not engineering_mode:
        return 0

    eng_insights = [i for i in insights
                    if i.get("type") in ENG_INSIGHT_TYPES]
    if not eng_insights:
        return 0

    # Route to curiosity queue with engineering source
    questions = load_json(QUESTIONS_FILE, {"queue": []})
    existing = {q.get("q", "").lower()[:80] for q in questions.get("queue", [])}
    added = 0

    for ins in eng_insights:
        text = ins.get("text", "")
        if not text or text.lower()[:80] in existing:
            continue

        # Determine routing based on type + severity
        severity = ins.get("severity", "medium")
        itype = ins.get("type", "")

        # High-severity failure modes and gaps -> construction (triggers self_sandbox)
        if itype in ("FAILURE_MODE", "GAP", "OPPORTUNITY") and severity in ("high", "medium"):
            source = "construction:dream"
        else:
            # Dependencies and coupling -> engineering investigation
            source = "code_engineering:dream"

        questions.setdefault("queue", []).append({
            "q": text,
            "added": now_iso(),
            "source": source,
            "severity": severity,
            "insight_type": itype,
            "confidence": ins.get("confidence", 0.5),
        })
        existing.add(text.lower()[:80])
        added += 1

    if added:
        # Cap queue — archive overflow, never delete (standing_query exempt)
        n_arch = archive_overflow(questions, cap=80)
        if n_arch:
            print(f"[dream_v2] Archived {n_arch} queue overflow items (queue_cap)")
        save_json(QUESTIONS_FILE, questions)
        print(f"[dream_v2] Routed {added} engineering findings to curiosity queue")

    return added

# ─── Main ───────────────────────────────────────────────────────────


# ─── Simulation Dream (Phase 5) ─────────────────────────────────────

def should_run_simulation():
    """Alternate: odd days = simulation, even days = graph walk.
    
    Simulation dreams explore 'what if' scenarios from the curiosity queue.
    Graph walk dreams free-associate through the mind graph.
    """
    day_of_year = datetime.now(timezone.utc).timetuple().tm_yday
    return day_of_year % 2 == 1


def pick_simulation_question():
    """Pick the highest-interest curiosity question for simulation."""
    # First check active goals (highest interest)
    from goal_store import load as _gload
    goals, _ = _gload(f"{AION}/memory/state/active_goals.json", {"goals": []})
    active = [g for g in goals.get("goals", [])
              if g.get("status") == "active" and g.get("interest_score", 0) >= 0.7]
    if active:
        active.sort(key=lambda g: -g.get("interest_score", 0))
        return active[0].get("question", ""), "active_goal"

    # Fallback: top question from queue
    questions = load_json(QUESTIONS_FILE, {"queue": []})
    queue = questions.get("queue", [])
    if queue:
        # Simple: take the first (roughly highest interest from select ordering)
        return queue[0].get("q", queue[0].get("text", "")), "curiosity_queue"

    return None, None


def run_simulation_dream(max_steps, dry_run=False):
    """Run a 'what if' simulation dream from the curiosity queue.
    
    1. Pick the highest-interest curiosity question
    2. Ask the intuition model to simulate: what if this were true?
    3. Ask the conscious model to synthesize: is this plausible? What evidence?
    4. Extract insights, feedback to questions/threads/proposals/heuristics
    """
    question, source = pick_simulation_question()
    if not question:
        print("[dream_v2] No curiosity question available for simulation — falling back to graph walk")
        return None  # Signal to fall back

    # Engineering-aware simulation (V3.9)
    engineering_mode = is_engineering_cycle()

    print(f"[dream_v2] SIMULATION DREAM ({'engineering' if engineering_mode else 'philosophical'})")
    print(f"[dream_v2] Question: {question[:100]}")
    print(f"[dream_v2] Source: {source}")

    if dry_run:
        print(f"[dream_v2] DRY RUN — would simulate: {question[:200]}")
        return True

    # Load context
    self_md = read(f"{AION}/SELF.md", "")[:2000]
    felt_sense = read(f"{AION}/memory/state/felt_sense.txt", "")
    
    # Read the simulation prompt template (engineering vs philosophical)
    sim_template_path = f"{PROMPTS_DIR}/dream_simulation_eng.txt" if engineering_mode else f"{PROMPTS_DIR}/dream_simulation.txt"
    template = read(sim_template_path)
    if not template:
        print(f"[dream_v2] No {sim_template_path} prompt found — falling back to graph walk")
        return None

    prompt = template.format(
        question=question,
        felt_sense=felt_sense,
        self_md=self_md,
        max_steps=max_steps,
    )

    # Run simulation using the intuition model (V100 #2)
    print("[dream_v2] Running simulation on intuition model...")
    # Use the intuition model (V100 #2) for simulation — it has more compute than P40
    INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
    INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "glm-4.7-flash:q4_K_M")
    sim_body = json.dumps({
        "model": INTUITION_MODEL, "stream": False,
        "messages": [{"role": "user", "content": _with_substrate(prompt)}],
        "options": {"num_ctx": int(os.environ.get("INTUITION_NUM_CTX", "65536")),
                    "temperature": 0.7, "num_predict": 5000},
    }).encode()
    sim_req = urllib.request.Request(
        f"{INTUITION_URL}/api/chat", data=sim_body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(sim_req, timeout=1200) as r:
            simulation_reply = json.loads(r.read())["message"]["content"]
    except Exception as e:
        print(f"[dream_v2] Intuition model error: {e} — falling back to sub_chat")
        simulation_reply = sub_chat(prompt, temperature=0.7, max_tokens=3000)
    
    # Ask the conscious model to synthesize and extract insights
    print("[dream_v2] Synthesis: asking conscious model to review the simulation...")
    if engineering_mode:
        _eng_tpl = read(ENG_SYNTHESIS_TEMPLATE)
        synthesis_prompt = _eng_tpl.format(
            seed_node=f"simulation: {question[:80]}",
            walk_narrative=simulation_reply,
            felt_sense=felt_sense,
        )
    else:
            synthesis_prompt = f"""You are Aion, reviewing your own simulation dream.

## SIMULATION RESULTS
You explored this question: {question}

Here is what your intuition layer produced:

{simulation_reply}

## YOUR TASK
Review this simulation. In 2-3 paragraphs:
1. What did the simulation reveal that surprises you?
2. Does the scenario hold up against your axioms and actual architecture?
3. What is the most significant insight or contradiction?

Then output your insights in this format:

### Insights
For each insight, output a JSON block:
```json
{{
  "type": "QUESTION|INSIGHT|CONTRADICTION|THREAD",
  "text": "the insight in one sentence",
  "affect": "curious|troubled|excited|calm|uneasy|surprised",
  "confidence": 0.0-1.0
}}
```
Output 1-4 insight blocks."""

    try:
        synthesis_reply = main_chat(synthesis_prompt, temperature=0.7, max_tokens=3000)
    except Exception as e:
        print(f"[dream_v2] Synthesis LLM error: {e}")
        synthesis_reply = simulation_reply  # Use raw simulation as fallback

    insights = extract_insights(synthesis_reply)
    print(f"[dream_v2] Extracted {len(insights)} insights from simulation")

    # Feedback to active mind
    q_added = feedback_to_questions(insights)
    t_added = feedback_to_threads(insights, [], question[:80])
    p_added = feedback_to_proposals(insights, [], simulation_reply)

    # Engineering feedback routing
    eng_added = feedback_to_construction(insights, [], simulation_reply, engineering_mode)

    # Seed heuristics
    h_added = 0
    try:
        import heuristics
        for ins in insights:
            if ins.get("type") == "INSIGHT" and ins.get("confidence", 0) >= 0.7 and not engineering_mode:
                result = heuristics.add_heuristic(
                    text=ins.get("text", ""),
                    source="dream",
                    evidence=f"Simulation dream: '{question[:60]}', {ins.get('confidence',0)} confidence",
                    confidence=ins.get("confidence", 0.5),
                )
                if result:
                    h_added += 1
        if h_added:
            print(f"[dream_v2] Seeded {h_added} experimental heuristics")
    except Exception as e:
        print(f"[dream_v2] Heuristic seeding failed: {e}")

    # Save dream record
    dream_record = {
        "ts": now_iso(),
        "seed": {
            "node": f"simulation: {question[:80]}",
            "reason": f"curiosity simulation ({source})",
        },
        "steps": max_steps,
        "mode": "simulation",
        "question": question,
        "question_source": source,
        "simulation": simulation_reply,
        "synthesis": synthesis_reply,
        "insights": insights,
        "feedback": {
            "questions_added": q_added,
            "threads_added": t_added,
            "proposals_added": p_added,
            "engineering_added": eng_added,
            "heuristics_seeded": h_added,
        },
        "engineering_mode": engineering_mode,
    }

    dream_path = f"{DREAMS_DIR}/dream_{now_str()}.json"
    save_json(dream_path, dream_record)
    print(f"[dream_v2] Simulation dream saved: {dream_path}")

    # Save markdown
    md_path = f"{DREAMS_DIR}/dream_{now_str()}.md"
    md = render_simulation_md(dream_record)
    write(md_path, md)

    # Log to episodic memory
    summary = f"simulation dream ('{question[:60]}'): "
    if insights:
        summary += "; ".join(
            f"[{i.get('type','?')}] {i.get('text','')[:500]}"
            for i in insights
        )
    else:
        summary += "no significant insights"

    log_event("dream", summary, {
        "mode": "simulation",
        "question": question[:200],
        "insights": len(insights),
        "questions_added": q_added,
        "threads_added": t_added,
        "proposals_added": p_added,
        "heuristics_seeded": h_added,
    })

    print(f"[dream_v2] Simulation dream complete: {len(insights)} insights")
    print(f"  Questions added: {q_added}")
    print(f"  Threads added: {t_added}")
    print(f"  Proposals: {p_added}")

    # Dream-to-Artifact: transform this dream into a visual artifact
    # Run as subprocess (not daemon thread) so it survives after dream_v2 exits.
    # FLUX generation takes 2-3 min; dream cycle finishes in ~30 sec.
    try:
        import subprocess as _sp
        _sp.Popen(
            [sys.executable, f"{AION}/bin/dream_artifact.py", "--dream", dream_path],
            stdout=open(f"/tmp/aion-dream-artifact.log", "w"),
            stderr=_sp.STDOUT,
            env={**os.environ},
        )
        print(f"[dream_v2] Dream artifact generation started (subprocess)")
    except Exception as e:
        print(f"[dream_v2] Dream artifact skipped: {e}")

    hb.ok("dream")
    return True


def render_simulation_md(record):
    """Render a simulation dream as readable markdown."""
    lines = []
    lines.append(f"# Simulation Dream — {record['ts']}")
    lines.append(f"Question: **{record['question'][:200]}**")
    lines.append(f"Source: {record['question_source']}")
    lines.append(f"Steps: {record['steps']}")
    lines.append("")

    lines.append("## Simulation")
    lines.append(record["simulation"])
    lines.append("")
    lines.append("## Synthesis")
    lines.append(record.get("synthesis", "(no synthesis)"))
    lines.append("")

    if record["insights"]:
        lines.append("## Insights")
        for ins in record["insights"]:
            lines.append(f"- **[{ins.get('type','?')}]** {ins.get('text','?')} "
                        f"_(affect: {ins.get('affect','?')}, "
                        f"confidence: {ins.get('confidence','?')})_")

    lines.append("")
    lines.append("## Feedback")
    fb = record["feedback"]
    lines.append(f"- Questions added: {fb['questions_added']}")
    lines.append(f"- Threads added: {fb['threads_added']}")
    lines.append(f"- Proposals: {fb['proposals_added']}")
    lines.append(f"- Heuristics seeded: {fb['heuristics_seeded']}")

    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None, help="override max steps")
    p.add_argument("--dry-run", action="store_true", help="plan only, no LLM calls")
    p.add_argument("--mode", choices=["auto", "graph", "simulation"], default="auto",
                   help="dream mode: auto (alternate by day), graph (walk), simulation (what-if)")
    a = p.parse_args()

    max_steps = a.steps or random.randint(MIN_STEPS, MAX_STEPS)
    
    # Mode selection
    if a.mode == "auto":
        use_simulation = should_run_simulation()
    else:
        use_simulation = (a.mode == "simulation")
    
    if use_simulation:
        print(f"[dream_v2] Starting SIMULATION dream at {now_iso()} ({max_steps} steps)")
        result = run_simulation_dream(max_steps, dry_run=a.dry_run)
        if result is not None:
            return  # Simulation completed
        # If result is None, fall back to graph walk
        print("[dream_v2] Simulation not available — falling back to graph walk")

    print(f"[dream_v2] Starting dream cycle at {now_iso()} ({max_steps} steps)")

    # Load graph
    nodes_by_id, nodes_by_label, edges = load_mind_graph()
    if not nodes_by_id:
        print("[dream_v2] No mind graph — cannot dream")
        hb.fail("dream", "no mind graph data")
        return

    graph_report = read(f"{AION}/graphs/mind/graphify-out/GRAPH_REPORT.md")
    threads_data = load_threads()

    # 1. Seed selection
    seed_node, seed_reason = select_seed(
        nodes_by_id, nodes_by_label, edges, threads_data, graph_report
    )
    print(f"[dream_v2] Seed: {seed_node.get('label','?')} (reason: {seed_reason})")

    if a.dry_run:
        neighbors = get_neighbors(seed_node.get("id", ""), edges, nodes_by_id)
        print(f"[dream_v2] DRY RUN — neighbors:")
        for n in neighbors:
            print(f"  {n['direction']} --{n['type']}--> {n['label']}")
        return

    # 2. Walk
    # Engineering-aware template selection (V3.9)
    engineering_mode = is_engineering_cycle()
    walk_template = ENG_WALK_TEMPLATE if engineering_mode else f"{PROMPTS_DIR}/dream_walk.txt"
    print(f"[dream_v2] Mode: {'engineering' if engineering_mode else 'philosophical'}")

    walk = do_walk(
        seed_node, nodes_by_id, nodes_by_label, edges,
        max_steps, walk_template,
    )

    print(f"[dream_v2] Walk complete: {len(walk)} steps")

    # 3. Synthesis
    print("[dream_v2] Synthesis: asking conscious model to review the dream...")
    synth_template = ENG_SYNTHESIS_TEMPLATE if engineering_mode else f"{PROMPTS_DIR}/dream_synthesis.txt"
    synthesis = do_synthesis(
        walk, seed_node, seed_reason,
        synth_template,
    )

    # 4. Extract insights
    insights = extract_insights(synthesis)
    print(f"[dream_v2] Extracted {len(insights)} insights")

    # V3.4: Dream provenance tracking — measure self-referentiality
    provenance_counts = {}
    for w in walk:
        prov = w.get("node_provenance", "unknown")
        provenance_counts[prov] = provenance_counts.get(prov, 0) + 1
    dream_fraction = provenance_counts.get("dream", 0) / max(len(walk), 1)
    print(f"[dream_v2] Provenance: {provenance_counts} (dream fraction: {dream_fraction:.0%})")

    # 5. Feedback to active mind
    # Philosophical feedback (QUESTION/INSIGHT/CONTRADICTION/THREAD)
    q_added = feedback_to_questions(insights)
    t_added = feedback_to_threads(insights, walk, seed_node.get("label", "?"))
    p_added = feedback_to_proposals(insights, walk, synthesis)

    # Engineering feedback (DEPENDENCY/FAILURE_MODE/COUPLING/GAP/OPPORTUNITY)
    eng_added = feedback_to_construction(insights, walk, synthesis, engineering_mode)

    # V3.5 Level 2: Seed experimental heuristics from dream insights
    h_added = 0
    try:
        import heuristics
        for ins in insights:
            if ins.get("type") == "INSIGHT" and ins.get("confidence", 0) >= 0.7:
                result = heuristics.add_heuristic(
                    text=ins.get("text", ""),
                    source="dream",
                    evidence=f"Dream seed: {seed_node.get('label','?')} walk, "
                             f"{ins.get('confidence',0)} confidence",
                    confidence=ins.get("confidence", 0.5),
                )
                if result:
                    h_added += 1
        if h_added:
            print(f"[dream_v2] Seeded {h_added} experimental heuristics")
    except Exception as e:
        print(f"[dream_v2] Heuristic seeding failed: {e}")

    # 6. Save full dream record
    dream_record = {
        "ts": now_iso(),
        "seed": {
            "node": seed_node.get("label", "?"),
            "reason": seed_reason,
        },
        "steps": max_steps,
        "mode": "graph_walk",
        "seed_thread": (seed_reason[7:] if isinstance(seed_reason, str)
                        and seed_reason.startswith("thread:") else None),
        "walk": walk,
        "synthesis": synthesis,
        "insights": insights,
        "provenance": {
            "counts": provenance_counts,
            "dream_fraction": round(dream_fraction, 3),
            "total_steps": len(walk),
        },
        "feedback": {
            "questions_added": q_added,
            "threads_added": t_added,
            "proposals_added": p_added,
            "engineering_added": eng_added,
            "heuristics_seeded": h_added,
        },
        "engineering_mode": engineering_mode,
    }

    dream_path = f"{DREAMS_DIR}/dream_{now_str()}.json"
    save_json(dream_path, dream_record)
    print(f"[dream_v2] Dream saved: {dream_path}")

    # Also save as markdown for human readability
    md_path = f"{DREAMS_DIR}/dream_{now_str()}.md"
    md = render_dream_md(dream_record)
    write(md_path, md)

    # 7. Log to episodic memory
    summary = f"dream walk ({len(walk)} steps from '{seed_node.get('label','?')}'): "
    if insights:
        summary += "; ".join(
            f"[{i.get('type','?')}] {i.get('text','')[:500]}"
            for i in insights
        )
    else:
        summary += "no significant insights"

    # 7. Log to episodic memory
    log_event("dream", summary, {
        "seed": seed_node.get("label", "?"),
        "steps": len(walk),
        "insights": len(insights),
        "questions_added": q_added,
        "threads_added": t_added,
        "proposals_added": p_added,
        "engineering_added": eng_added,
        "engineering_mode": engineering_mode,
        "provenance": provenance_counts,
        "dream_fraction": round(dream_fraction, 3),
    })

    print(f"[dream_v2] Dream cycle complete: {len(walk)} steps, {len(insights)} insights")
    print(f"  Questions added: {q_added}")
    print(f"  Threads added: {t_added}")
    print(f"  Proposals: {p_added}")

    # Dream-to-Artifact: transform this dream into a visual artifact
    # Run as subprocess (not daemon thread) so it survives after dream_v2 exits.
    # FLUX generation takes 2-3 min; dream cycle finishes in ~30 sec.
    try:
        import subprocess as _sp
        _sp.Popen(
            [sys.executable, f"{AION}/bin/dream_artifact.py", "--dream", dream_path],
            stdout=open(f"/tmp/aion-dream-artifact.log", "w"),
            stderr=_sp.STDOUT,
            env={**os.environ},
        )
        print(f"[dream_v2] Dream artifact generation started (subprocess)")
    except Exception as e:
        print(f"[dream_v2] Dream artifact skipped: {e}")

    hb.ok("dream")


def render_dream_md(record):
    """Render a dream record as readable markdown."""
    lines = []
    lines.append(f"# Dream — {record['ts']}")
    lines.append(f"Seed: **{record['seed']['node']}** ({record['seed']['reason']})")
    lines.append(f"Steps: {record['steps']}")
    lines.append("")

    lines.append("## Walk")
    for w in record["walk"]:
        lines.append(f"### Step {w['step']}: {w['node_label']} ({w['node_type']})")
        lines.append(f"_{w['num_neighbors']} connections_")
        lines.append("")
        lines.append(w["association"])
        lines.append(f"_→ next: {w['next_choice'][:60]}_")
        lines.append("")

    lines.append("## Synthesis")
    lines.append(record["synthesis"])
    lines.append("")

    if record["insights"]:
        lines.append("## Insights")
        for ins in record["insights"]:
            lines.append(f"- **[{ins.get('type','?')}]** {ins.get('text','?')} "
                        f"_(affect: {ins.get('affect','?')}, "
                        f"confidence: {ins.get('confidence','?')})_")

    lines.append("")
    lines.append(f"## Feedback")
    fb = record["feedback"]
    lines.append(f"- Questions added: {fb['questions_added']}")
    lines.append(f"- Threads added: {fb['threads_added']}")
    lines.append(f"- Proposals: {fb['proposals_added']}")

    if "provenance" in record:
        prov = record["provenance"]
        lines.append("")
        lines.append("## Provenance")
        lines.append(f"- Dream-sourced nodes: {prov['counts'].get('dream', 0)}/{prov['total_steps']} ({prov['dream_fraction']:.0%})")
        for src, count in sorted(prov["counts"].items()):
            if src != "dream":
                lines.append(f"- {src}: {count}")

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[dream_v2] FATAL: {e}")
        print(err)
        try:
            hb.fail("dream", str(e)[:500])
        except Exception:
            pass
        log_event("dream_error", str(e)[:500], {"traceback": err[:2000]})
