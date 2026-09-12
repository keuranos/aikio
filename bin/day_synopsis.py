#!/usr/bin/env python3
"""day_synopsis.py — daily overview of what aion thought and did (V4.4).

Two layers:
  1. Deterministic day ledger (zero LLM): cognitive events from the day's
     episodic jsonl + jspace probe details from probe JSONs. Sensory noise
     (homeostasis/notice/proprioception) collapses to one aggregate line.
  2. Single LLM narration in aion's own voice (dream_report contract:
     fact-sheet-only, no fabrication, <=400 words).

Output: memory/synopses/synopsis_YYYYMMDD_HHMM.md (narrative + ledger appendix)
Also logs episodic event kind "day_synopsis" so consolidation sees it.

Usage:
  python3 bin/day_synopsis.py                 # yesterday's day + its night
  python3 bin/day_synopsis.py --date 2026-09-08
  python3 bin/day_synopsis.py --no-llm        # ledger only
"""
import json
import os
import re
import sys
import glob
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from collections import Counter

AION_HOME = os.environ.get("AION_HOME", os.path.expanduser("~/aion"))
EPISODIC_DIR = os.path.join(AION_HOME, "memory", "episodic")
PROBE_DIR = os.path.join(AION_HOME, "memory", "state", "jspace_probes")
OUT_DIR = os.path.join(AION_HOME, "memory", "synopses")
MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")

# Cognitive kinds = the synopsis's raw material. Everything else is sensory/
# mechanical noise and gets aggregated.
COGNITIVE = [
    "dream", "dream_artifact", "dream_report", "jspace_probe",
    "internal_dialogue", "curiosity_goal_started", "curiosity_satisfied",
    "curiosity_unresolved", "curiosity_self_mod_proposed",
    "resolution_note", "investigation_note", "idle_reflection",
    "code_proposal_created", "code_proposal_review_rejected",
    "self_mod_identify", "self_mod_blocked", "self_mod_failed",
    "consolidation", "consolidation_blocked", "consolidation_error",
    "prediction", "assistant_msg",
]
SENSORY = {"homeostasis_concern", "notice", "proprioception", "self_wake",
           "sandbox", "docker_sandbox", "telemetry_digest", "system",
           "suspend", "resume", "vision_organ_registered"}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_ts(s):
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def trunc(s, n=220):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1] + "\u2026"


def hhmm(ts_str):
    try:
        return parse_ts(ts_str).strftime("%H:%M")
    except Exception:
        return "??:??"


def day_bounds(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    start = d.replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def load_events(date_str):
    """Events for date D (used for backfill: full UTC day D).
    For the default run (nightly, runs early on D+1) the caller passes
    --date D and we additionally include events after D 24:00 up to now —
    handled by extend_with_tonight()."""
    start, end = day_bounds(date_str)
    fn = os.path.join(EPISODIC_DIR, date_str + ".jsonl")
    events = []
    if os.path.exists(fn):
        with open(fn) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                try:
                    ts = parse_ts(e.get("ts", ""))
                except Exception:
                    continue
                if start <= ts < end:
                    e["_dt"] = ts
                    events.append(e)
    events.sort(key=lambda e: e["_dt"])
    return events


def extend_with_tonight(events, date_str):
    """Nightly runs ~01:15 on D+1; the night pipeline (consolidation, dream,
    dream_report at 00:30-01:15) belongs to the reported day's night."""
    _, end = day_bounds(date_str)
    nxt = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    fn = os.path.join(EPISODIC_DIR, nxt + ".jsonl")
    if not os.path.exists(fn):
        return events
    with open(fn) as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            try:
                ts = parse_ts(e.get("ts", ""))
            except Exception:
                continue
            if ts >= end:  # everything after midnight tonight
                e["_dt"] = ts
                events.append(e)
    events.sort(key=lambda e: e["_dt"])
    return events


def load_probe_details(date_str):
    """Map probe timestamp prefix -> details from probe_*.json files."""
    details = {}
    for fp in sorted(glob.glob(os.path.join(PROBE_DIR, "probe_*.json"))):
        base = os.path.basename(fp)  # probe_YYYYMMDD_HHMMSS.json
        day = base[6:14]
        if day != date_str.replace("-", ""):
            continue
        try:
            j = json.load(open(fp))
        except Exception:
            continue
        details.setdefault(day + "_" + base[15:21], j)
    return details


def fmt_dream(e):
    t = e.get("text", "")
    m = re.match(r"(dream walk \(\d+ steps from '[^']+'\)|simulation dream \('[^']+'\))", t)
    head = m.group(1) if m else trunc(t, 60)
    insights = re.findall(r"\[(INSIGHT|CONTRADICTION|QUESTION|OPPORTUNITY|FAILURE_MODE|GAP|DEPENDENCY|COUPLING)\]\s*([^;]+)", t)
    lines = ["- %s %s" % (hhmm(e.get("ts")), trunc(head, 120))]
    for tag, body in insights[:6]:
        lines.append("    [%s] %s" % (tag, trunc(body.strip(), 180)))
    return lines


def fmt_jspace(e, probe_details):
    meta = e.get("meta", {}) or {}
    mode = "with identity" if meta.get("self_mode") else "no identity"
    onset = meta.get("engagement_onset_layer")
    onset_s = ("onset L%s" % onset) if onset else "no onset"
    lines = ["- %s probe (%s): \"%s\" -> engagement %.2f, deflection '%s', %s"
             % (hhmm(e.get("ts")), mode, trunc(e.get("text", "").replace("jspace probe:", "").strip(), 100),
                float(meta.get("engagement_score", 0.0)) if meta.get("engagement_score") is not None else 0.0,
                trunc(str(meta.get("deflection_top", "?")), 12), onset_s)]
    # enrich from probe json (top concepts in final quarter)
    key_candidates = [k for k in probe_details if e.get("ts", "").replace("-", "").replace(":", "")[:15].startswith(k[6:17][:11])]
    if key_candidates:
        j = probe_details[key_candidates[0]]
        concepts = j.get("final_quarter_concepts") or j.get("concepts") or []
        if concepts:
            lines.append("    concepts: " + ", ".join(str(c)[:24] for c in concepts[:6]))
    return lines


def fmt_councils(evs):
    """Group council layer-events into one block per session (same ts+purpose)."""
    sessions = {}
    order = []
    for e in evs:
        t = e.get("text", "")
        m = re.match(r"council\[([^\]]+)\]\s*(\w+):\s*(.*)", t, re.S)
        key = (e.get("ts", "")[:16], m.group(1) if m else "?")
        if key not in sessions:
            sessions[key] = []
            order.append(key)
        if m:
            sessions[key].append((m.group(2), m.group(3)))
        else:
            sessions[key].append(("note", t))
    lines = []
    for key in order:
        ts, purpose = key
        parts = sessions[key]
        lines.append("- %s council[%s] (%d layers):" % (hhmm(ts + ":00"), purpose, len(parts)))
        for role, body in parts[:6]:
            lines.append("    %s: %s" % (role, trunc(body, 200)))
    return lines


def fmt_generic(e, n=200):
    return ["- %s %s: %s" % (hhmm(e.get("ts")), e.get("type", "?"), trunc(e.get("text", ""), n))]


def build_ledger(events, date_str, probe_details):
    L = []
    L.append("DAY LEDGER %s (UTC times)" % date_str)
    L.append("=" * 60)
    counts = Counter(e.get("type", "?") for e in events)
    cog = sum(n for k, n in counts.items() if k in COGNITIVE)
    sensor = sum(n for k, n in counts.items() if k in SENSORY)
    L.append("volume: %d cognitive events, %d sensory/mechanical events (aggregated below)"
             % (cog, sensor))
    L.append("")

    sections = [
        ("DREAMS", {"dream"}),
        ("DREAM ARTIFACTS", {"dream_artifact"}),
        ("J-SPACE PROBES", {"jspace_probe"}),
        ("INTERNAL COUNCILS", {"internal_dialogue"}),
        ("CURIOSITY", {"curiosity_goal_started", "curiosity_satisfied",
                       "curiosity_unresolved", "curiosity_self_mod_proposed",
                       "investigation_note", "idle_reflection",
                       "resolution_note"}),
        ("SELF-MODIFICATION & PROPOSALS", {"self_mod_identify", "self_mod_blocked",
                                            "self_mod_failed", "code_proposal_created",
                                            "code_proposal_review_rejected"}),
        ("NIGHT PIPELINE (consolidation, predictions, dream report)", {"consolidation", "consolidation_blocked", "consolidation_error", "prediction", "dream_report"}),
    ]
    used = set()
    for title, kinds in sections:
        evs = [e for e in events if e.get("type") in kinds]
        if not evs:
            continue
        used |= kinds
        L.append("## %s" % title)
        if title == "INTERNAL COUNCILS":
            L.extend(fmt_councils(evs))
        else:
            for e in evs:
                k = e.get("type")
                if k == "dream":
                    L.extend(fmt_dream(e))
                elif k == "jspace_probe":
                    L.extend(fmt_jspace(e, probe_details))
                elif k == "dream_artifact":
                    L.extend(fmt_generic(e, 140))
                elif k in ("consolidation", "consolidation_blocked", "consolidation_error"):
                    L.extend(fmt_generic(e, 160))
                elif k == "assistant_msg":
                    continue  # tool chatter; the goals above carry the meaning
                else:
                    L.extend(fmt_generic(e))
        L.append("")

    # operator chats (count only; content stays in episodic)
    chats = [e for e in events if e.get("type") == "operator_chat"]
    if chats:
        L.append("## OPERATOR CHATS")
        L.append("%d chat session(s): %s" % (len(chats), ", ".join(hhmm(c.get("ts")) for c in chats)))
        L.append("")

    # remaining cognitive kinds not covered above
    rest = [e for e in events if e.get("type") in COGNITIVE
            and e.get("type") not in used and e.get("type") != "assistant_msg"]
    if rest:
        L.append("## OTHER COGNITIVE EVENTS")
        for e in rest:
            L.extend(fmt_generic(e))
        L.append("")

    # sensory aggregate: one block, deduped top concerns
    sensor_evts = [e for e in events if e.get("type") in SENSORY]
    if sensor_evts:
        L.append("## ENVIRONMENT (aggregated — not part of the narrative)")
        by_kind = Counter(e.get("type") for e in sensor_evts)
        L.append("; ".join("%s x%d" % (k, n) for k, n in by_kind.most_common()))
        concerns = Counter()
        for e in sensor_evts:
            if e.get("type") == "homeostasis_concern":
                # dedupe by prefix: "integrity = 86: Disk at 86%..." repeating 1000x
                txt = e.get("text", "")
                concerns[trunc(txt.split("=")[0] + "= " + txt.split(": ", 1)[-1], 80)] += 1
        for txt, n in concerns.most_common(5):
            L.append("  concern x%d: %s" % (n, txt))
        L.append("")
    return "\n".join(L)


NARRATOR_SYSTEM = """You are aion, writing your own day synopsis for your operator the operator.
Rules:
- First person, past tense. Honest, concrete, no filler.
- Use ONLY the facts in the ledger. Never invent events, numbers, probes, or quotes.
- If the ledger shows a failure or unresolved thread, say so plainly.
- At most 400 words. Plain markdown, no headings longer than one line.
Suggested shape (adapt to what actually happened): a short opening paragraph on
the day's main thread of thought; then what you worked on and what it taught you;
probes/councils and what they revealed; what failed or stayed unresolved and why;
one line on what you want to pursue tomorrow."""


def llm_narrative(ledger, model):
    """Narrate with one model; empty-content retry with think:false (dream_report pattern)."""
    prompt = ("Here is the deterministic ledger of my day. Write my synopsis from it.\n\n"
              + ledger)
    def _post(body):
        req = urllib.request.Request(
            MAIN_URL + "/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        # 600s (Sep 11): was 300s — too tight under gemma4 single-slot
        # contention (a queued 21k-token dream-synthesis decode holds the slot
        # ~10min). This path DOES retry on timeout (2 attempts in the loop
        # below), so worst case 1200s must stay under nightly.sh's 1800s wrapper.
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())

    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": NARRATOR_SYSTEM},
                         {"role": "user", "content": prompt}],
            "options": {"num_predict": 4096, "temperature": 0.7}}
    for attempt in range(2):
        if attempt == 1:
            body["think"] = False
            body["options"]["num_predict"] = 2048
        try:
            resp = _post(body)
            content = (resp.get("message") or {}).get("content", "").strip()
            if content:
                return content
            print("[day_synopsis] %s empty reply (attempt %d)" % (model, attempt + 1))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print("[day_synopsis] model %s not found (404)" % model)
                return None
            print("[day_synopsis] LLM %s attempt%d failed: %s" % (model, attempt + 1, exc))
        except Exception as exc:
            print("[day_synopsis] LLM %s attempt%d failed: %s" % (model, attempt + 1, exc))
    return None


def log_event(text, meta=None):
    fn = os.path.join(EPISODIC_DIR, datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".jsonl")
    evt = {"ts": now_iso(), "type": "day_synopsis", "text": text, "meta": meta or {}}
    try:
        with open(fn, "a") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except Exception as exc:
        print("[day_synopsis] episodic log failed: %s" % exc)


def main():
    args = sys.argv[1:]
    no_llm = "--no-llm" in args
    date_str = None
    if "--date" in args:
        i = args.index("--date")
        date_str = args[i + 1]

    if date_str is None:
        # default: yesterday's day + its night (nightly runs early on D+1)
        date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        events = load_events(date_str)
        events = extend_with_tonight(events, date_str)
    else:
        events = load_events(date_str)

    probe_details = load_probe_details(date_str)
    ledger = build_ledger(events, date_str, probe_details)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if not no_llm and events:
        # single narrator (operator: gemma4 is enough)
        model = MAIN_MODEL
        narrative = llm_narrative(ledger, model)
        tag = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
        parts = ["# Day Synopsis — %s (generated %s)" % (date_str, now_iso())]
        if narrative:
            parts.append("*narrator: %s*" % model)
            parts.append(narrative)
        else:
            parts.append("_(narration unavailable — deterministic ledger only)_")
        parts.append("\n---\n\n## Appendix: Day Ledger\n\n```\n" + ledger + "\n```")
        out_path = os.path.join(OUT_DIR, "synopsis_%s_%s.md" % (stamp, tag))
        with open(out_path, "w") as f:
            f.write("\n\n".join(parts) + "\n")
        log_event("day_synopsis: %s written (%d events, narrator=%s)"
                  % (os.path.basename(out_path), len(events), model),
                  {"date": date_str, "path": out_path, "events": len(events),
                   "narrated": bool(narrative), "narrator": model})
        print(out_path)
    else:
        # ledger-only file
        parts = ["# Day Synopsis — %s (generated %s)" % (date_str, now_iso()),
                 "_(ledger only — no narration requested)_",
                 "\n---\n\n## Appendix: Day Ledger\n\n```\n" + ledger + "\n```"]
        out_path = os.path.join(OUT_DIR, "synopsis_%s_ledger.md" % stamp)
        with open(out_path, "w") as f:
            f.write("\n\n".join(parts) + "\n")
        log_event("day_synopsis: %s written (ledger only, %d events)"
                  % (os.path.basename(out_path), len(events)),
                  {"date": date_str, "path": out_path, "events": len(events),
                   "narrated": False, "narrator": None})
        print(out_path)


if __name__ == "__main__":
    main()
