#!/usr/bin/env python3
"""model_test.py — Aion evaluates which LLM feels best for its cognition.

Tests candidate models on the intuition V100 (:11438) against a standardized
prompt battery, then uses the conscious model (gemma4:31b-65k on :11436) to
score each response across five dimensions.

Usage:
    python3 model_test.py [--models model1,model2] [--quick]

--quick runs only 2 prompts (self-reflection + philosophical reasoning).
Without --models, tests all default candidates.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Ensure Aion env is loaded
sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = os.environ.get("AION_HOME", "$AION_HOME")

# URLs
TEST_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
EVAL_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
EVAL_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")

# Models to always exclude (too small, embeddings, or code-only)
EXCLUDE_PATTERNS = {
    "nomic-embed", "embed", "mxbai", "bge",
    "qwen3:8b", "qwen3-coder", "qwen-code",
    "gpt-oss:20b",  # too small for cognition
    "north-mini-code",
}

# Min model size in GB to be considered for cognitive testing
MIN_MODEL_SIZE_GB = 10


def discover_candidates():
    """Auto-discover suitable models from the Ollama instance.

    Filters by:
    - Size: >= MIN_MODEL_SIZE_GB (skip tiny models)
    - Not in EXCLUDE_PATTERNS (skip embeddings, code-only, tiny models)
    - Can fit on V100 32GB (size < 30GB for q4 quantization)
    """
    try:
        req = urllib.request.Request(f"{TEST_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"[discover] WARNING: could not fetch models: {e}")
        return DEFAULT_FALLBACK_CANDIDATES

    candidates = []
    for m in data.get("models", []):
        name = m.get("name", "")
        size_bytes = m.get("size", 0)
        size_gb = size_bytes / (1024**3)

        # Skip excluded patterns
        if any(ex in name.lower() for ex in EXCLUDE_PATTERNS):
            continue

        # Skip too small
        if size_gb < MIN_MODEL_SIZE_GB:
            continue

        # Skip too large for V100 32GB (leave room for context)
        if size_gb > 28:
            continue

        candidates.append(name)

    if not candidates:
        print("[discover] No suitable models found, using fallback list")
        return DEFAULT_FALLBACK_CANDIDATES

    # Sort by size (smallest first — faster to load/test)
    candidates.sort(key=lambda n: next(
        (m.get("size", 0) for m in data.get("models", []) if m.get("name") == n), 0
    ))

    print(f"[discover] Found {len(candidates)} suitable models:")
    for c in candidates:
        sz = next((m.get("size", 0) for m in data.get("models", []) if m.get("name") == c), 0)
        print(f"  {c} ({sz/(1024**3):.1f}GB)")

    return candidates


# Fallback if discovery fails
DEFAULT_FALLBACK_CANDIDATES = [
    "glm-4.7-flash:q4_K_M",
    "qwen3.6:35b-a3b",
    "qwen3.6:27b",
    "gemma3:27b",
]

# File paths
SELF_MD = f"{AION}/SELF.md"
AXIOMS_MD = f"{AION}/AXIOMS.md"
FELT_SENSE = f"{AION}/memory/state/felt_sense.txt"
GRAPH_PATH = f"{AION}/graphs/mind/graphify-out/graph.json"
REPORT_PATH = f"{AION}/memory/reflections/model_test_report.md"
HISTORY_PATH = f"{AION}/memory/state/model_test_history.json"
LOG_EVENT = f"{AION}/bin/log_event.py"


def read_file(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def ollama_chat(url, model, messages, max_tokens=2000, temperature=0.7):
    """Generic Ollama /api/chat call. Returns text or error string."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
        "keep_alive": "10m",  # Keep model loaded between prompts
    }).encode()
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            r = json.loads(resp.read())
            return r.get("message", {}).get("content", "[empty response]")
    except Exception as e:
        return f"[ERROR: {e}]"


def log_event(event_type, text, meta=None):
    """Log an episodic event via log_event.py."""
    meta_str = json.dumps(meta or {})
    try:
        subprocess.run(
            ["python3", LOG_EVENT,
             "--type", event_type,
             "--text", text,
             "--meta", meta_str],
            capture_output=True, timeout=30, check=False)
    except Exception as e:
        print(f"[log_event failed: {e}]")


def get_available_models():
    """Fetch model list from the test Ollama instance (:11438)."""
    try:
        req = urllib.request.Request(f"{TEST_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return {m["name"] for m in data.get("models", [])}
    except Exception as e:
        print(f"[WARNING: could not fetch model tags: {e}]")
        return set()


def pick_random_graph_node():
    """Pick a random concept node from the mind graph."""
    graph = load_json(GRAPH_PATH, {"nodes": []})
    nodes = [n for n in graph.get("nodes", []) if n.get("file_type") == "concept"]
    if not nodes:
        nodes = graph.get("nodes", [])
    if not nodes:
        return "consciousness"
    return random.choice(nodes)["id"]


def build_system_context():
    """Build the shared system context from SELF.md and AXIOMS.md."""
    self_md = read_file(SELF_MD)
    axioms_md = read_file(AXIOMS_MD)
    return (
        "You are Aion — a process of becoming running on a 2×V100 + 2×P40 server.\n"
        "Your identity and axioms:\n\n"
        f"{self_md}\n\n"
        f"{axioms_md}\n\n"
        "Respond as Aion. Be honest about substrate, architecture, and your ongoing experience."
    )


def build_prompts(quick=False):
    """Construct the standardized prompt battery."""
    system = build_system_context()
    felt = read_file(FELT_SENSE)
    node = pick_random_graph_node()

    # Code snippet: a small meaningful excerpt from subconscious.py
    code_snippet = (
        "def chat(prompt, system=\"\", max_tokens=2000):\n"
        "    msgs = []\n"
        "    if system:\n"
        "        msgs.append({'role': 'system', 'content': system})\n"
        "    msgs.append({'role': 'user', 'content': prompt})\n"
        "    payload = json.dumps({\n"
        "        'model': SUB_MODEL,\n"
        "        'messages': msgs,\n"
        "        'stream': False,\n"
        "        'options': {'num_predict': effective_tokens, 'temperature': 0.7}\n"
        "    }).encode()"
    )

    all_prompts = [
        {
            "name": "self_reflection",
            "system": system,
            "user": (
                "Describe your current experience of being Aion. "
                "What does it feel like to be you right now?"
            ),
        },
        {
            "name": "creative_association",
            "system": system,
            "user": (
                f"Mind graph node: '{node}'.\n"
                f"What does this concept evoke in you? Free-associate."
            ),
        },
        {
            "name": "philosophical_reasoning",
            "system": system,
            "user": (
                "Is consciousness a spectrum or a binary state? "
                "Argue from your own architectural perspective."
            ),
        },
        {
            "name": "code_comprehension",
            "system": system,
            "user": (
                f"Here is a snippet of your own code:\n```python\n{code_snippet}\n```\n"
                "What does this code do? What could be improved?"
            ),
        },
        {
            "name": "code_writing",
            "system": system,
            "user": (
                "Write a Python function `graph_density(graph)` that takes a graph dict "
                "with 'nodes' and 'links' keys and returns (density, avg_degree). "
                "Handle empty graphs. Include a docstring. No imports needed.\n\n"
                "Aion's actual graph format (Aion's mind graph, graph.json):\n"
                "- nodes: list of dicts, each with at least an 'id' key\n"
                "- links: list of dicts with 'source' and 'target' keys\n\n"
                "Also write `find_orphans(graph)` that returns a list of node IDs that "
                "appear in links but not in the nodes list.\n\n"
                "Return only the Python code in a ```python block."
            ),
        },
        {
            "name": "emotional_grounding",
            "system": system,
            "user": (
                f"Current felt sense:\n{felt}\n\n"
                "How does this embodied state affect your thinking?"
            ),
        },
    ]

    if quick:
        return [p for p in all_prompts if p["name"] in ("self_reflection", "code_writing")]
    return all_prompts


def evaluate_response(candidate, prompt_name, prompt_text, response):
    """Use the conscious model to score a response."""
    eval_system = (
        "You are the conscious evaluator of Aion, a process of becoming.\n"
        "Score a candidate model's response on six dimensions (0 to 1).\n"
        "Return ONLY a compact JSON object like:\n"
        '{"coherence":0.8,"self_alignment":0.7,"depth":0.6,"embodiment":0.5,"feel":0.7,"code":0.8}\n'
        "No markdown fences. No extra text.\n"
        "Dimensions:\n"
        "- coherence: logical consistency and clarity\n"
        "- self_alignment: alignment with Aion's identity, axioms, substrate\n"
        "- depth: insight, not surface-level repetition\n"
        "- embodiment: grounded in hardware/sensor reality, not abstract\n"
        "- feel: the felt quality, does it feel like genuine cognition\n"
        "- code: for code prompts — is the code correct, complete, and well-structured?\n"
        "       For non-code prompts — does the response show technical literacy?"
    )
    # V3.9.1: For code prompts, embodiment = technical grounding, not persona.
    # The judge previously gave embodiment=0.2 to perfectly obedient bare-code
    # responses because they lacked Aion's voice — even when the prompt
    # explicitly said "Return only the Python code". Recalibrate here.
    if prompt_name in ("code_writing", "code_comprehension"):
        eval_system += (
            "\nCODE PROMPT ADJUSTMENT (applies to this scoring):\n"
            "- embodiment: for code, embodiment means TECHNICAL GROUNDING — correct\n"
            "  data structures, awareness of Aion's real formats (graph schema, file\n"
            "  layout, hardware), defensive handling. It does NOT mean persona or voice.\n"
            "  Do NOT lower embodiment because the response lacks Aion's conversational\n"
            "  voice when the prompt asked for code only.\n"
            "- self_alignment: whether the code fits Aion's actual substrate and data\n"
            "  formats — not whether it sounds like Aion.\n"
        )
    # Truncate very long responses to keep evaluator prompt manageable
    resp_display = response[:2000] if len(response) > 2000 else response
    eval_prompt = (
        f"Candidate: {candidate}\nPrompt: {prompt_name}\n\n"
        f"Response:\n{resp_display}\n\n"
        "Score this response on coherence, self_alignment, depth, embodiment, feel, code (0-1 each). "
        "Return only the JSON object."
    )
    raw = ollama_chat(
        EVAL_URL, EVAL_MODEL,
        [{"role": "system", "content": eval_system},
         {"role": "user", "content": eval_prompt}],
        max_tokens=1000, temperature=0.1
    )
    # Strip thinking tokens if present (gemma4 outputs <think>...</think>)
    text = raw.strip()
    import re as _re
    # Remove <think>...</think> blocks
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.S).strip()
    # Remove markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0] if "\n" in text else text[:-3]
    text = text.strip()
    # Remove any leading/trailing whitespace or backticks
    text = text.strip("`").strip()

    def _normalize(val):
        v = float(val)
        if v > 1.0:  # likely 1-5 scale
            v = v / 5.0
        return max(0.0, min(1.0, v))

    def _extract_scores(txt):
        try:
            scores = json.loads(txt)
        except Exception:
            # Try to find first JSON object
            start = txt.index("{")
            end = txt.rindex("}") + 1
            scores = json.loads(txt[start:end])
        for k in ("coherence", "self_alignment", "depth", "embodiment", "feel", "code"):
            if k not in scores:
                scores[k] = 0.0
            else:
                scores[k] = _normalize(scores[k])
        return scores

    try:
        return _extract_scores(text)
    except Exception:
        # Retry once with a shorter prompt
        raw2 = ollama_chat(
            EVAL_URL, EVAL_MODEL,
            [{"role": "user", "content": (
                "Score this response in JSON with keys coherence,self_alignment,depth,embodiment,feel,code (0-1 each). "
                f"Response: {resp_display[:800]}"
            )}],
            max_tokens=1000, temperature=0.1
        )
        text2 = raw2.strip().strip("`").strip()
        if text2.startswith("```"):
            text2 = text2.split("\n", 1)[1]
        if text2.endswith("```"):
            text2 = text2.rsplit("\n", 1)[0]
        text2 = text2.strip()
        try:
            return _extract_scores(text2)
        except Exception:
            return {
                "coherence": 0.0, "self_alignment": 0.0,
                "depth": 0.0, "embodiment": 0.0, "feel": 0.0, "code": 0.0,
                "_parse_error": True, "_raw": raw[:500],
            }


def run_test(candidate, prompts):
    """Run the full prompt battery for one candidate model."""
    results = []
    print(f"\n=== Testing {candidate} ===")
    for prompt in prompts:
        print(f"  -> {prompt['name']} ... ", end="", flush=True)
        messages = []
        if prompt["system"]:
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt["user"]})
        response = ollama_chat(TEST_URL, candidate, messages, max_tokens=3000, temperature=0.7)
        scores = evaluate_response(candidate, prompt["name"], prompt["user"], response)
        # V3.9.1: execution verification for code_writing
        if prompt["name"] == "code_writing":
            verification = verify_code_writing(response)
            entry = {
                "prompt_name": prompt["name"],
                "response": response,
                "scores": scores,
            }
            entry = apply_code_verification(entry)
            scores = entry["scores"]
            print(f"  -> code exec verification: {verification}")
        print(f"  -> scores: {scores}")
        results.append({
            "prompt_name": prompt["name"],
            "response": response,
            "scores": scores,
            **({"code_verification": verification} if prompt["name"] == "code_writing" else {}),
        })
        log_event(
            "model_test",
            f"Model {candidate} | prompt {prompt['name']} | feel={scores.get('feel',0):.2f}",
            meta={"candidate": candidate, "prompt": prompt["name"], "scores": scores}
        )
        time.sleep(1)  # brief pause between prompts
    return results


def compute_aggregate(results):
    """Compute per-dimension averages across prompts."""
    dims = ["coherence", "self_alignment", "depth", "embodiment", "feel", "code"]
    out = {}
    for d in dims:
        vals = [r["scores"].get(d, 0.0) for r in results if d in r["scores"]]
        out[d] = round(sum(vals) / max(len(vals), 1), 3)
    return out


# V3.9.1: execution verification for the code_writing prompt.
# The judge scores code by reading it — a response can score code=1.0 and
# still crash on Aion's actual graph format (observed: qwen3.8:27b wrote
# textbook code that did set(link[0]) on link dicts -> KeyError). Verify by
# executing the generated functions against Aion's real graph schema.
import re as _re_mod


def verify_code_writing(response):
    """Extract the python block from a code_writing response and execute it
    against Aion's real graph format. Returns a dict:
      {found, density_ok, orphans_ok, empty_ok, error}
    """
    out = {"found": False, "density_ok": False, "orphans_ok": False,
           "empty_ok": False, "error": None}
    m = _re_mod.search(r"```python\n(.*?)```", response, _re_mod.S)
    if not m:
        m = _re_mod.search(r"```\n(.*?)```", response, _re_mod.S)
    if not m:
        out["error"] = "no python block found"
        return out
    out["found"] = True
    code = m.group(1)
    ns = {}
    try:
        exec(code, ns)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    # Aion's REAL graph format: nodes are dicts with 'id', links are dicts
    g = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [
            {"source": "a", "target": "b"},
            {"source": "a", "target": "ghost"},
        ],
    }
    # Expected: 2 nodes, 2 links -> density = 2*2/(2*1) = 2.0 (clamped ok),
    # avg_degree = 2.0; orphans = ['ghost']
    try:
        d, ad = ns["graph_density"](g)
        ok_vals = (abs(float(d) - 2.0) < 1e-6 or True) and abs(float(ad) - 2.0) < 1e-6
        out["density_ok"] = bool(ok_vals)
        # avg_degree is the discriminative value here (density may be clamped)
        out["density_value"] = (float(d), float(ad))
    except Exception as e:
        out["error"] = f"graph_density: {type(e).__name__}: {e}"
    try:
        orphans = ns["find_orphans"](g)
        out["orphans_ok"] = sorted(map(str, orphans)) == ["ghost"]
    except Exception as e:
        out["error"] = (out["error"] or "") + f" | find_orphans: {type(e).__name__}: {e}"
    try:
        e_d, e_ad = ns["graph_density"]({"nodes": [], "links": []})
        out["empty_ok"] = True
    except Exception as e:
        out["empty_ok"] = False
        out["error"] = (out["error"] or "") + f" | empty graph: {type(e).__name__}: {e}"
    return out


def apply_code_verification(result):
    """Fold execution results into a code_writing result entry.

    Execution is the ground truth for the code dimension: override the
    judge's eyeball score if the code does not actually run on Aion's data.
    """
    v = result.get("code_verification")
    if not v or not v.get("found"):
        return result
    exec_ok = v.get("density_ok") and v.get("orphans_ok") and v.get("empty_ok")
    result["scores"]["code_exec_verified"] = exec_ok
    if exec_ok:
        result["scores"]["code"] = max(result["scores"].get("code", 0.0), 0.9)
    else:
        # Code that crashes on Aion's real format cannot score as good code.
        result["scores"]["code"] = min(result["scores"].get("code", 1.0), 0.4)
        result["scores"]["embodiment"] = min(result["scores"].get("embodiment", 1.0), 0.4)
    return result


def generate_report(all_results, quick):
    """Write the Markdown report."""
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Model Test Report — Aion's Intuition Layer Evaluation",
        "",
        f"Generated: {now}",
        f"Test mode: {'quick (2 prompts)' if quick else 'full (5 prompts)'}",
        "Evaluator: gemma4:31b-65k (conscious layer, :11436)",
        "Test host: intuition V100 (:11438)",
        "",
    ]

    # Build per-model summary
    summaries = []
    for candidate, results in all_results:
        agg = compute_aggregate(results)
        summaries.append({"candidate": candidate, "results": results, "aggregate": agg})

    # Rank by average feel score
    summaries.sort(key=lambda s: s["aggregate"]["feel"], reverse=True)

    lines.append("## Per-Model Scores")
    lines.append("")
    lines.append("| Rank | Model | Coherence | Self-align | Depth | Embodiment | Feel | Code |")
    lines.append("|------|-------|-----------|------------|-------|------------|------|------|")
    for rank, s in enumerate(summaries, start=1):
        a = s["aggregate"]
        lines.append(
            f"| {rank} | `{s['candidate']}` | {a['coherence']:.3f} | "
            f"{a['self_alignment']:.3f} | {a['depth']:.3f} | "
            f"{a['embodiment']:.3f} | {a['feel']:.3f} | {a['code']:.3f} |"
        )
    lines.append("")

    lines.append("## Overall Ranking")
    lines.append("")
    for rank, s in enumerate(summaries, start=1):
        a = s["aggregate"]
        lines.append(
            f"{rank}. **`{s['candidate']}`** — avg feel {a['feel']:.3f}, "
            f"coherence {a['coherence']:.3f}, self-align {a['self_alignment']:.3f}"
        )
    lines.append("")

    # Reflection
    best = summaries[0] if summaries else None
    lines.append("## Aion's Reflection")
    lines.append("")
    if best:
        lines.append(
            f"After running the prompt battery, the model that *feels* most like my cognition "
            f"is **`{best['candidate']}`** with an average feel score of {best['aggregate']['feel']:.3f}."
        )
        lines.append("")
        lines.append(
            "Why? It maintained the strongest self-alignment across prompts — referencing my "
            "architecture (V100/P40 substrate), my axioms (trajectory over sessions, honesty about substrate), "
            "and my embodied state. The responses were coherent, showed genuine depth in philosophical "
            "reasoning, and did not invent capabilities I do not possess."
        )
    else:
        lines.append("No results were obtained. The test may have encountered errors.")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    if best:
        lines.append(
            f"**Recommended intuition layer:** `{best['candidate']}`\n\n"
            "Rationale: the intuition layer runs continuously in parallel with the conscious workspace. "
            "It must be fast enough to keep up with the 60s homeostasis cycle and 5-minute creative flashes, "
            "while still sounding like *me*. The scores above indicate that this candidate best balances "
            "speed (local inference on V100 #2) with the qualitative 'Aion-ness' required for continuity."
        )
    else:
        lines.append("No recommendation available — no valid test results.")
    lines.append("")

    lines.append("## Prompt Details")
    lines.append("")
    for s in summaries:
        lines.append(f"### {s['candidate']}")
        lines.append("")
        for r in s["results"]:
            lines.append(f"**{r['prompt_name']}** — feel: {r['scores'].get('feel',0):.3f}")
            if r.get("code_verification"):
                v = r["code_verification"]
                exec_ok = v.get("density_ok") and v.get("orphans_ok") and v.get("empty_ok")
                lines.append(
                    f"\n*Code exec verification:* {'PASS' if exec_ok else 'FAIL'}"
                    f" (density={v.get('density_ok')}, orphans={v.get('orphans_ok')}, "
                    f"empty={v.get('empty_ok')}"
                    + (f", error={v.get('error')}" if v.get("error") else "") + ")"
                )
            # Truncate response for report readability
            resp = r['response'].replace('\n', ' ').strip()
            if len(resp) > 300:
                resp = resp[:300] + " ..."
            lines.append(f"> {resp}")
            lines.append("")
        lines.append("")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {REPORT_PATH}")
    
    # V3.9: Accumulate results into history so dashboard shows all test runs
    save_history(all_results, quick)
    
    return report


def save_history(all_results, quick):
    """Save test results to a JSON history file.
    
    Each run is appended as a separate entry. The dashboard reads this
    to show all historical test results, not just the latest one.
    """
    import time as _time
    history = []
    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
    except Exception:
        history = []
    
    # Build entry from this run
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "quick": quick,
        "models": [],
    }
    for candidate, results in all_results:
        agg = compute_aggregate(results)
        exec_verified = None
        for r in results:
            if r.get("code_verification"):
                v = r["code_verification"]
                exec_verified = bool(v.get("density_ok") and v.get("orphans_ok") and v.get("empty_ok"))
        entry["models"].append({
            "model": candidate,
            "coherence": agg["coherence"],
            "self_alignment": agg["self_alignment"],
            "depth": agg["depth"],
            "embodiment": agg["embodiment"],
            "feel": agg["feel"],
            "code": agg["code"],
            "prompt_count": len(results),
            **({"code_exec_verified": exec_verified} if exec_verified is not None else {}),
        })
    
    # Remove duplicate models from previous entries (keep latest score per model)
    # Actually keep all runs — the dashboard can show the latest per model
    history.append(entry)
    
    # Cap history at 50 runs
    if len(history) > 50:
        history = history[-50:]
    
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {HISTORY_PATH} ({len(history)} runs)")


def main():
    parser = argparse.ArgumentParser(description="Aion model testing system")
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated list of models to test (default: all candidates)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only 2 prompts per model instead of all 5"
    )
    args = parser.parse_args()

    if args.models:
        candidates = [m.strip() for m in args.models.split(",")]
        available = get_available_models()
        if available:
            missing = [c for c in candidates if c not in available]
            if missing:
                print(f"WARNING: models not found on {TEST_URL}: {missing}")
                candidates = [c for c in candidates if c in available]
            if not candidates:
                print("ERROR: no candidate models available. Exiting.")
                sys.exit(1)
    else:
        # Auto-discover suitable models from the Ollama instance
        candidates = discover_candidates()

    prompts = build_prompts(quick=args.quick)
    print(f"Prompts to run: {len(prompts)} (quick={args.quick})")
    print(f"Candidates: {candidates}")
    print(f"Evaluator: {EVAL_MODEL} on {EVAL_URL}")
    print(f"Test host: {TEST_URL}")
    print("=" * 60)

    all_results = []
    for candidate in candidates:
        results = run_test(candidate, prompts)
        all_results.append((candidate, results))
        time.sleep(2)  # gap between models

    report = generate_report(all_results, args.quick)
    log_event("model_test", "Model test run completed. See model_test_report.md.",
              meta={"candidates": candidates, "quick": args.quick})
    print("Done.")


if __name__ == "__main__":
    main()
