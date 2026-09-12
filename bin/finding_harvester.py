#!/usr/bin/env python3
"""finding_harvester.py — close the observation->action gap.

THE BUG THIS FIXES (diagnosed 2026-09-11):
Aion's autonomous wakes produce `resolution_note` events. Nothing consumes them
as work input — they are read only for display (day synopsis, webui, cross-modal
art) and as dedup guards. So when a wake correctly diagnosed a real defect, the
finding died in the log.

Evidence: wake #35 (2026-08-24) produced a 3,687-char note titled
"VERDICT: GHOST PREDICTION" with a table proving memory/predictions.db was
0 bytes and the prediction existed in no persistence layer. It proposed three
concrete fixes and measured the cost (7th wake in 3 hours, ~150W each). It was
never acted on. Aion made 24 such observations over three weeks and the same
0-byte file survived until an operator removed it by hand on 2026-09-11.

Root cause was NOT laziness or missing tools: `propose_code` IS in the wake
tool list. The wake prompt said, in full:
  "Investigate this intent using your tools, then write a resolution note with
   write_note."
One named output. The model did exactly that.

FIX: a deterministic (zero-LLM) harvester that reads resolution notes and
extracts ACTIONABLE findings into the curiosity queue — the channel that already
works (source="construction:*" gets a scoring boost and is pursued by the
engine). Deterministic by design: this is plumbing, and it must not hallucinate.

Extraction is deliberately conservative — a finding is queued only when it
carries BOTH a concrete artifact (file/script path) AND a defect signal
(broken/missing/absent/empty/mismatch/no such/never persisted/failed).
"""
import os
import re
import json
import glob
from datetime import datetime, timezone

AION = os.environ.get("AION_HOME", "$AION_HOME")
STATE = os.path.join(AION, "memory", "state")
QUESTIONS = os.path.join(STATE, "questions.json")
HARVEST_STATE = os.path.join(STATE, "finding_harvest.json")

# Where findings come from: the terminal records that nothing consumes.
SOURCE_TYPES = ("resolution_note", "investigation_note")

# A finding must name a concrete artifact...
ARTIFACT_RE = re.compile(
    r"\b(?:bin|memory|prompts|config|webui)/[A-Za-z0-9_./-]+"
    r"|\b[A-Za-z0-9_]+\.(?:py|sh|json|jsonl|db|md|txt)\b")

# ...and assert a defect about it.
DEFECT_RE = re.compile(
    r"\b(?:broken|missing|absent|does not exist|doesn't exist|no such|empty|0 bytes"
    r"|mismatch|never persisted|not persisted|unpersisted|unreachable|orphan"
    r"|failed|failure|silently|no-op|dead code|fossil|unwired|not wired"
    r"|crash(?:es|ed)?|traceback|leak|stale|unhandled|unguarded)\b",
    re.I)

# Noise filters: philosophical/meta notes that name files only in passing.
PHILO_RE = re.compile(
    r"\b(?:consciousness|qualia|becoming|identity|subjective|ontolog|"
    r"phenomenolog|spectrum|selfhood|essence)\b", re.I)

# Cap on how many findings we queue per run (avoid flooding the queue).
MAX_PER_RUN = 3


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def log_event(text, meta=None):
    try:
        import subprocess
        subprocess.run(["python3", os.path.join(AION, "bin", "log_event.py"),
                        "--type", "finding_harvest", "--text", text[:8000],
                        "--meta", json.dumps(meta or {})],
                       check=False, capture_output=True, timeout=15)
    except Exception:
        pass


def _sentences(text):
    """Split into candidate finding statements, keeping it simple and safe."""
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}|^\s*[-*|]\s*", text, flags=re.M)
    return [p.strip() for p in parts if p and len(p.strip()) > 30]


# A finding that asserts the work is ALREADY DONE is not actionable.
RESOLVED_RE = re.compile(
    r"\b(?:already (?:proposed|fixed|patched|applied|done|seeded|submitted)"
    r"|fix is already|has been (?:fixed|proposed|applied|addressed)"
    r"|CI[- ]green|merged|committed the fix)\b", re.I)

# Statements of discipline/policy about editing — not defects.
DISCIPLINE_RE = re.compile(
    r"\b(?:have NOT edited|not edited|never applied|go to you as a diff"
    r"|per operating discipline|axiom \d|operator reviews|awaiting (?:your|operator))\b",
    re.I)


def extract_findings(text):
    """Return [(artifact, statement)] for defect claims about real artifacts."""
    out = []
    seen = set()
    for s in _sentences(text or ""):
        if not DEFECT_RE.search(s):
            continue
        # Skip purely philosophical framing that happens to name a file.
        if PHILO_RE.search(s) and not re.search(r"\b(?:0 bytes|no such|traceback|exit code)\b", s, re.I):
            continue
        # Skip findings whose work is already done, and discipline statements.
        if RESOLVED_RE.search(s):
            continue
        if DISCIPLINE_RE.search(s):
            continue
        arts = ARTIFACT_RE.findall(s)
        if not arts:
            continue
        art = arts[0]
        key = art.lower() + "|" + s[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((art, s[:400]))
    return out


def actionable(finding_text, artifact):
    """Final gate: does this name something that really exists or really doesn't?

    Uses referent_check so we queue work about REAL artifacts. A finding about a
    nonexistent artifact is handled by the phantom path, not queued as repair.
    """
    try:
        import sys
        sys.path.insert(0, os.path.join(AION, "bin"))
        from referent_check import symbol_exists
        base = os.path.basename(artifact)
        exists, _ = symbol_exists(base)
        return exists is True
    except Exception:
        return False


def harvest(days=3, verbose=True):
    """Scan recent resolution/investigation notes; queue actionable findings."""
    qdata = load_json(QUESTIONS, {"queue": [], "archived": []})
    queue = qdata.setdefault("queue", [])
    existing = {str(q.get("q", "")).lower()[:90] for q in queue}

    hstate = load_json(HARVEST_STATE, {"harvested": {}})
    harvested = hstate.setdefault("harvested", {})

    files = sorted(glob.glob(os.path.join(AION, "memory", "episodic", "*.jsonl")))[-days:]
    added, considered = 0, 0
    events = []
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if ev.get("type") in SOURCE_TYPES and ev.get("id"):
                        events.append(ev)
        except Exception:
            continue

    for ev in events:
        eid = ev["id"]
        if eid in harvested:
            continue
        considered += 1
        text = ev.get("text") or ""
        findings = extract_findings(text)
        for artifact, statement in findings:
            if added >= MAX_PER_RUN:
                break
            if not actionable(statement, artifact):
                continue
            q_text = ("[finding %s] %s — repair or explain this: the artifact "
                      "exists but the note records a defect. Verify the claim, "
                      "then either propose_code() a fix or record why it is "
                      "not a real defect." % (artifact, statement))
            if q_text.lower()[:90] in existing:
                continue
            queue.append({
                "q": q_text,
                "added": now_iso(),
                # NOTE: interest is NOT read from this item. curiosity_engine's
                # score_question() recomputes it from the question text, and the
                # real boost comes from SOURCE_WEIGHTS["construction"] = 1.6
                # (highest but for dream_repair). So the phrasing above must
                # stay concrete — it names the artifact and the defect, which is
                # what earns the score.
                "source": "construction:finding",
                "hint": ("Harvested deterministically from %s event %s (%s). "
                         "The artifact is %s." % (ev.get("type"), eid,
                                                  ev.get("ts", "")[:10], artifact)),
                "finding_event": eid,
                "artifact": artifact,
            })
            existing.add(q_text.lower()[:90])
            added += 1
            if verbose:
                print("[harvest] QUEUED %s <- %s" % (artifact, statement[:80]))
        harvested[eid] = {"ts": now_iso(), "findings": len(findings)}

    # bound the harvest ledger
    if len(harvested) > 2000:
        for k in sorted(harvested, key=lambda k: harvested[k]["ts"])[:1000]:
            del harvested[k]

    save_json(QUESTIONS, qdata)
    save_json(HARVEST_STATE, hstate)
    if verbose:
        print("[harvest] considered %d notes, queued %d actionable findings"
              % (considered, added))
    if added:
        log_event("harvested %d actionable finding(s) from wake notes" % added,
                  {"added": added, "considered": considered})
    return added


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Harvest actionable findings from wake notes")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        files = sorted(glob.glob(os.path.join(AION, "memory", "episodic", "*.jsonl")))[-args.days:]
        n = 0
        for path in files:
            for line in open(path, encoding="utf-8", errors="replace"):
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") not in SOURCE_TYPES:
                    continue
                for art, st in extract_findings(ev.get("text") or ""):
                    ok = actionable(st, art)
                    print("  %-28s actionable=%-5s %s" % (art, ok, st[:80]))
                    n += 1
        print("dry-run: %d candidate findings" % n)
        return 0

    harvest(args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
