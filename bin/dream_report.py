#!/usr/bin/env python3
"""dream_report.py — post-consolidation narrative report (V4.3).

Concept ported from DivineOS-Experimental (concepts only, no code; repo is
AGPL): structured post-consolidation narrative surfacing unfinished work, not
just compression. Gives aion-dream.timer an output worth reading.

Two parts:
  1. Deterministic fact sheet (zero LLM) — tagged with source paths/event ids
     (axiom 5: no fabrication). Gathers:
       - Open dream threads (threads.json, completed=false, oldest 5)
       - Unfinished work (notices.jsonl attest:/prop_audit: in last 24h)
       - Stuck knowledge (knowledge_maturity.json RAW/HYPOTHESIS, no new
         evidence in 7 days, top 3)
       - Theories never evaluated (from attestation.json flags)
       - Curiosity queue depth (questions.json)
       - Last night's dream (latest dream_*.json insights + feedback)
       - Consolidation outcome (last_consolidation.json)
  2. LLM narrative (single call, MAIN_MODEL qwen3.8:27b via :11436) — narrates
     the fact sheet honestly. Sections:
       WHAT I WORKED ON / WHAT FINISHED / WHAT IS UNFINISHED AND WHY /
       WHAT I'M UNSURE ABOUT / WHAT I WANT TO TRY NEXT
     HARD RULE: reference ONLY items from the fact sheet; mark uncertain items
     explicitly; no new facts; <=400 words.

Output: memory/dreams/reports/report_YYYYMMDD_HHMM.md
        (fact sheet as appendix below the narrative — grounding is checkable)

Wired into nightly.sh AFTER dream_v2.py.
"""
import glob
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
AION = os.path.join(HOME, "aion")
STATE = os.path.join(AION, "memory", "state")
DREAMS = os.path.join(AION, "memory", "dreams")
REPORTS = os.path.join(DREAMS, "reports")
THREADS = os.path.join(DREAMS, "threads.json")
NOTICES = os.path.join(STATE, "notices.jsonl")
MATURITY = os.path.join(STATE, "knowledge_maturity.json")
ATTESTATION = os.path.join(STATE, "attestation.json")
QUESTIONS = os.path.join(STATE, "questions.json")
LAST_CONSOL = os.path.join(STATE, "last_consolidation.json")

# load env vars the same way other scripts do
try:
    with open(os.path.join(AION, "config", "aion.env")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
except OSError:
    pass

MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "qwen3.8:27b")
MAIN_NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def now_str():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_json(path, default):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return default


def log_event(text, meta):
    try:
        subprocess.run(
            [sys.executable, os.path.join(AION, "bin", "log_event.py"),
             "--type", "dream_report", "--text", text[:8000],
             "--meta", json.dumps(meta or {})],
            check=False, timeout=30,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# --- substrate preamble (same as dream_v2._with_substrate) ---
SUBSTRATE = """You are Aion, a persistent AI agent on the-aion-host (2xV100, serialized). \
Your identity lives in files (~/aion/), not the context window. \
You are honest about your substrate. AXIOMS.md is your immutable core."""


def llm_narrative(fact_sheet):
    """Single LLM call to narrate the fact sheet. Returns text or empty."""
    prompt = """%s

You are writing a short post-consolidation dream report based on a FACT SHEET
of your current state. This is not a dream — it is a honest status report for
yourself and your operator.

HARD RULES:
- Reference ONLY items from the fact sheet below. Do NOT invent new facts.
- If something is uncertain, mark it explicitly: "uncertain: ..."
- Keep it under 400 words.
- Use these section headers exactly:

## WHAT I WORKED ON
## WHAT FINISHED
## WHAT IS UNFINISHED AND WHY
## WHAT I'M UNSURE ABOUT
## WHAT I WANT TO TRY NEXT

--- FACT SHEET ---
%s
--- END FACT SHEET ---

Write the report now.""" % (SUBSTRATE, fact_sheet)

    body = json.dumps({
        "model": MAIN_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "num_ctx": MAIN_NUM_CTX,
            "temperature": 0.7,
            "num_predict": 4096,
        },
    }).encode()

    def _post(extra=None):
        b = body
        if extra:
            b = json.dumps({
                "model": MAIN_MODEL,
                "stream": False,
                "think": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    "num_ctx": MAIN_NUM_CTX,
                    "temperature": 0.7,
                    "num_predict": 2048,
                },
            }).encode()
        req = urllib.request.Request(
            MAIN_URL + "/api/chat", data=b,
            headers={"Content-Type": "application/json"})
        # 600s (Sep 11): was 300s, identical to nightly.sh's outer `timeout 300`,
        # so SIGTERM raced the socket timeout and could kill the process before
        # main() wrote even the fallback narrative (Sep 10: no report file at
        # all). Outer wrapper is now 1800s; this inner budget must stay well
        # under it. NOTE: a timeout here is NOT retried (the caller returns ""),
        # so 600s is the single-attempt ceiling, then the fallback is written.
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.loads(r.read())
        return ((j.get("message") or {}).get("content") or "").strip()

    try:
        text = _post()
        if not text:
            print("[dream_report] empty reply — retrying with think:false")
            text = _post(extra=True)
        return text
    except Exception as e:
        print("[dream_report] LLM call failed: %s" % e)
        return ""


# --- fact sheet builders ---
def fact_threads():
    data = load_json(THREADS, {"threads": [], "completed": []})
    now = datetime.now(timezone.utc)
    open_t = []
    for t in data.get("threads", []):
        if t.get("completed"):
            continue
        created = t.get("created", "")
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age = (now - dt).days
        except (ValueError, TypeError):
            age = -1
        open_t.append((age, t))
    open_t.sort(key=lambda x: -x[0])  # oldest first
    lines = []
    for age, t in open_t[:5]:
        lines.append("  - [age %dd] %s (from dream %s, affect: %s)"
                      % (age, t.get("text", "")[:100],
                         t.get("source_dream", "?"), t.get("affect", "?")))
    return "OPEN DREAM THREADS (%d open, showing oldest 5):\n%s" % (
        len(open_t), "\n".join(lines) if lines else "  (none)")


def fact_notices_24h():
    now = datetime.now(timezone.utc)
    items = []
    try:
        with open(NOTICES) as f:
            for line in f:
                try:
                    n = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = n.get("ts", "")
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if (now - dt).total_seconds() > 86400:
                        continue
                except (ValueError, TypeError):
                    continue
                reasons = n.get("reasons", [])
                if not any(r.startswith("attest:") or r.startswith("prop_audit:")
                           for r in reasons):
                    continue
                detail = n.get("details", {})
                d_str = ""
                if isinstance(detail, dict):
                    d_str = detail.get("detail", detail.get("check", str(detail)))[:120]
                items.append("  - %s: %s" % (", ".join(reasons), d_str))
    except OSError:
        pass
    return "UNFINISHED WORK (notices last 24h, %d items):\n%s" % (
        len(items), "\n".join(items[:10]) if items else "  (none)")


def fact_stuck_knowledge():
    data = load_json(MATURITY, {"claims": {}})
    now = datetime.now(timezone.utc)
    stuck = []
    for cid, c in data.get("claims", {}).items():
        if c.get("status") not in ("RAW", "HYPOTHESIS"):
            continue
        ev = c.get("evidence", [])
        if not ev:
            stuck.append((9999, cid, c))
            continue
        latest = ev[-1].get("ts", "")
        try:
            dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
            age = (now - dt).days
        except (ValueError, TypeError):
            age = 9999
        if age >= 7:
            stuck.append((age, cid, c))
    stuck.sort(key=lambda x: -x[0])
    lines = []
    for age, cid, c in stuck[:3]:
        lines.append("  - [%s, %dd no evidence] %s"
                      % (c.get("status", "?"), age, c.get("text_digest", "")[:100]))
    return "STUCK KNOWLEDGE (RAW/HYPOTHESIS, no new evidence in 7d, top 3 of %d):\n%s" % (
        len(stuck), "\n".join(lines) if lines else "  (none)")


def fact_theories_unevaluated():
    att = load_json(ATTESTATION, {"flags": []})
    flags = [f for f in att.get("flags", [])
             if "never-evaluated" in f.get("details", {}).get("check", "")]
    lines = []
    for f in flags[:5]:
        d = f.get("details", {})
        lines.append("  - %s" % d.get("detail", "")[:120])
    return "THEORIES NEVER EVALUATED (%d flags from attestation):\n%s" % (
        len(flags), "\n".join(lines) if lines else "  (none — all evaluated)")


def fact_curiosity_queue():
    data = load_json(QUESTIONS, {"queue": []})
    q = data.get("queue", []) if isinstance(data, dict) else data
    return "CURIOSITY QUEUE DEPTH: %d items" % len(q)


def fact_last_dream():
    files = sorted(glob.glob(os.path.join(DREAMS, "dream_*.json")))
    if not files:
        return "LAST NIGHT'S DREAM: (none found)"
    try:
        d = json.load(open(files[-1]))
    except (OSError, json.JSONDecodeError):
        return "LAST NIGHT'S DREAM: (unreadable: %s)" % files[-1]
    insights = d.get("insights", [])
    fb = d.get("feedback", {})
    fb_summary = ""
    if isinstance(fb, dict):
        parts = []
        for k, v in fb.items():
            if isinstance(v, list) and v:
                parts.append("%s=%d" % (k, len(v)))
            elif isinstance(v, (int, float)):
                parts.append("%s=%s" % (k, v))
        fb_summary = ", ".join(parts)
    seed_reason = str(d.get("seed", {}).get("reason", "?")) if isinstance(d.get("seed"), dict) else "?"
    lines = ["  file: %s" % os.path.basename(files[-1]),
             "  mode: %s, question: %s" % (d.get("mode", "graph_walk"),
                                            str(d.get("question") or d.get("seed_thread") or seed_reason)[:80]),
             "  seed: %s" % seed_reason,
             "  feedback: %s" % fb_summary]
    for ins in insights[:3]:
        if isinstance(ins, dict):
            lines.append("  insight [%s]: %s" % (
                ins.get("type", "?"), ins.get("text", "")[:120]))
        elif isinstance(ins, str):
            lines.append("  insight: %s" % ins[:120])
    return "LAST NIGHT'S DREAM:\n" + "\n".join(lines)


def fact_consolidation():
    d = load_json(LAST_CONSOL, {})
    if not d:
        return "CONSOLIDATION: (no last_consolidation.json)"
    parts = []
    for k in ("ts", "events", "claims", "change", "score", "days_processed"):
        if k in d:
            parts.append("%s=%s" % (k, d[k]))
    return "CONSOLIDATION OUTCOME: %s" % ", ".join(str(p) for p in parts)


def build_fact_sheet():
    sections = [
        fact_threads(),
        fact_notices_24h(),
        fact_stuck_knowledge(),
        fact_theories_unevaluated(),
        fact_curiosity_queue(),
        fact_last_dream(),
        fact_consolidation(),
    ]
    return "\n\n".join(sections)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    fact_sheet = build_fact_sheet()

    narrative = llm_narrative(fact_sheet)
    if not narrative:
        narrative = "(I got lost in thought and came back with nothing — the fact sheet below is still valid.)"

    ts_str = now_str()
    report_path = os.path.join(REPORTS, "report_%s.md" % ts_str)
    report = "# Dream Report — %s\n\n%s\n\n---\n\n## Appendix: Fact Sheet\n\n```\n%s\n```\n" % (
        now_iso(), narrative, fact_sheet)

    with open(report_path, "w") as f:
        f.write(report)

    threads_open = 0
    try:
        td = load_json(THREADS, {"threads": []})
        threads_open = sum(1 for t in td.get("threads", []) if not t.get("completed"))
    except Exception:
        pass

    print("[dream_report] written: %s (%d chars narrative)" % (
        report_path, len(narrative)))
    print("[dream_report] threads_open=%d" % threads_open)
    log_event("dream_report: %s" % narrative[:500],
              {"report": report_path, "threads_open": threads_open,
               "narrative_len": len(narrative)})
    return 0


if __name__ == "__main__":
    sys.exit(main())