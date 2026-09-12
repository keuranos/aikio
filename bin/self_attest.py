#!/usr/bin/env python3
"""self_attest.py — wired-vs-declared capability attestation (V4.3).

Concept ported from DivineOS-Experimental (concepts only, no code; repo is
AGPL): when Aion CLAIMS a capability, verify it is actually wired into the
flow that uses it. Rule-based, zero LLM calls.

Claim sources & checks:
  1. memory/state/skills.json    level>=2: successes >= level, last_attempt
                                  within 14 days              -> attest:stale-skill
  2. memory/state/competence.md  dated bullets naming files that must exist
                                  -> attest:unverifiable-competence
  3. SELF.md tier:stable         "I <verb>" capability sentences: referenced
                                  scripts must exist AND be wired into
                                  nightly.sh / config/*.sh / systemd units
                                  -> attest:declared-not-wired
  4. memory/state/theories.jsonl active theories with last_evaluated null,
                                  empty verdicts, older than 7 days
                                  -> attest:declared-never-evaluated

Outputs:
  memory/state/attestation.json   full report
  memory/state/notices.jsonl      attest:* flags (prop_audit dedup style:
                                   stable attest_id + reasons over last 300)
  episodic "system" event via bin/log_event.py

Exit 0 even when flags are found (flags are findings, not crashes).
"""
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
AION = os.path.join(HOME, "aion")
STATE = os.path.join(AION, "memory", "state")
SKILLS = os.path.join(STATE, "skills.json")
COMPETENCE = os.path.join(STATE, "competence.md")
SELF_MD = os.path.join(AION, "SELF.md")
THEORIES = os.path.join(STATE, "theories.jsonl")
ATTESTATION = os.path.join(STATE, "attestation.json")
NOTICES = os.path.join(STATE, "notices.jsonl")

STALE_DAYS = 14
THEORY_GRACE_DAYS = 7
CAP_VERBS = ("have", "can", "use", "run", "monitor", "generate")
# subsystem keyword -> systemd unit name fragment that proves wiring
SUBSYSTEM_UNITS = {
    "sensor stream": "aion-sensors",
    "homeostasis": "aion-homeostasis",
    "curiosity": "aion-curiosity",
    "dream": "aion-dream",
    "graph": "aion-graph",
    "intuition daemon": "intuition",
    "heartbeat": "aion-heartbeat",
    "cross-modal": "aion-cross-modal",
}
LINE_RE = re.compile(r"^- \((\d{4}-\d{2}-\d{2})\) (.+)$")
FILE_RE = re.compile(r"\b([a-z0-9_]+\.(?:py|sh|js))\b")
TIER_RE = re.compile(r"<!--\s*tier:stable\s*-->(.*?)<!--\s*/tier:stable\s*-->", re.S)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def flag(check, source, detail, attest_id):
    return {
        "ts": now_iso(),
        "score": 0.4,
        "reasons": ["attest:" + check],
        "details": {"attest_id": attest_id, "source": source,
                    "check": check, "detail": detail},
    }


def wiring_text_and_refs():
    """Concatenate pipeline wiring files; return (text, script basenames)."""
    paths = [os.path.join(AION, "bin", "nightly.sh")]
    paths += sorted(glob.glob(os.path.join(AION, "config", "*.sh")))
    paths += sorted(glob.glob(os.path.join(HOME, ".config", "systemd", "user", "*")))
    chunks = []
    refs = set()
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                t = f.read()
        except OSError:
            continue
        chunks.append(os.path.basename(p) + "\n" + t)
        for m in FILE_RE.finditer(t):
            refs.add(m.group(1))
    return "\n".join(chunks), refs


def unit_names():
    return {os.path.basename(p) for p in glob.glob(
        os.path.join(HOME, ".config", "systemd", "user", "*"))}


def script_exists(name):
    for d in ("bin", ".", "webui"):
        if os.path.isfile(os.path.join(AION, d, name)):
            return True
    return False


def check_skills(flags, stats):
    try:
        with open(SKILLS) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        stats["errors"] += 1
        return
    now = datetime.now(timezone.utc)
    for name, s in (data.get("skills", {}) or {}).items():
        lvl = s.get("level", 0)
        if lvl < 2:
            continue
        stats["checks"] += 1
        succ = s.get("successes", 0)
        la = parse_ts(s.get("last_attempt"))
        age = (now - la).days if la else None
        if succ < lvl:
            flags.append(flag("stale-skill", "skills.json",
                              "skill '%s' level %d but only %d successes"
                              % (name, lvl, succ), attest_id="skill:" + name))
        elif age is None or age > STALE_DAYS:
            flags.append(flag("stale-skill", "skills.json",
                              "skill '%s' level %d not exercised in %s days"
                              % (name, lvl, age), attest_id="skill:" + name))
        else:
            stats["wired"] += 1


def check_competence(flags, stats):
    try:
        with open(COMPETENCE, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        stats["errors"] += 1
        return
    for i, raw in enumerate(lines):
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        body = m.group(2)
        names = FILE_RE.findall(body)
        if not names:
            continue
        stats["checks"] += 1
        missing = [n for n in names if not script_exists(n)]
        if missing:
            flags.append(flag("unverifiable-competence", "competence.md",
                              "line %d references missing file(s): %s"
                              % (i + 1, ", ".join(missing)),
                              attest_id="competence:%d:%s" % (i + 1, missing[0])))
        else:
            stats["wired"] += 1


def check_self(flags, stats, wiring_refs, wiring_text, units):
    try:
        with open(SELF_MD, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        stats["errors"] += 1
        return
    m = TIER_RE.search(text)
    if not m:
        return
    for raw in m.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("<!--"):
            continue
        low = line.lower()
        if not any(("i " + v) in low for v in CAP_VERBS):
            continue
        scripts = FILE_RE.findall(line)
        subs = [k for k in SUBSYSTEM_UNITS if k in low]
        if not scripts and not subs:
            continue
        stats["checks"] += 1
        ok = True
        for s in scripts:
            if not script_exists(s):
                flags.append(flag("declared-not-wired", "SELF.md:tier:stable",
                                  "capability references missing script: " + s,
                                  attest_id="self:" + s))
                ok = False
            elif s not in wiring_refs:
                flags.append(flag("declared-not-wired", "SELF.md:tier:stable",
                                  "script exists but not wired into any "
                                  "pipeline/timer: " + s, attest_id="self:" + s))
                ok = False
        for k in subs:
            core = SUBSYSTEM_UNITS[k]
            wired = any(core in u for u in units) or (core in wiring_text)
            if not wired:
                flags.append(flag("declared-not-wired", "SELF.md:tier:stable",
                                  "claimed subsystem not wired: " + k,
                                  attest_id="self:sub:" + k))
                ok = False
        if ok:
            stats["wired"] += 1


def check_theories(flags, stats):
    now = datetime.now(timezone.utc)
    try:
        with open(THEORIES, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        stats["errors"] += 1
        return
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            t = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if t.get("status") != "active":
            continue
        stats["checks"] += 1
        created = parse_ts(t.get("created"))
        age = (now - created).days if created else 9999
        if (t.get("last_evaluated") is None and not t.get("verdicts")
                and age > THEORY_GRACE_DAYS):
            flags.append(flag("declared-never-evaluated", "theories.jsonl",
                              "theory '%s' active for %d days, never evaluated"
                              % (t.get("id", "?"), age),
                              attest_id="theory:" + str(t.get("id", "?"))))
        else:
            stats["wired"] += 1


def append_notices(entries):
    """Dedup on (attest_id, reasons) over the last 300 lines (prop_audit style)."""
    recent = []
    try:
        with open(NOTICES) as f:
            recent = f.readlines()[-300:]
    except OSError:
        pass
    seen = set()
    for ln in recent:
        try:
            r = json.loads(ln)
            d = r.get("details", {})
            if isinstance(d, dict) and d.get("attest_id"):
                seen.add((d.get("attest_id"), tuple(sorted(r.get("reasons", [])))))
        except (json.JSONDecodeError, AttributeError):
            continue
    fresh = [e for e in entries
             if (e["details"].get("attest_id"),
                 tuple(sorted(e["reasons"]))) not in seen]
    if fresh:
        with open(NOTICES, "a") as f:
            for e in fresh:
                f.write(json.dumps(e) + "\n")
    return len(fresh)


def log_event(text, meta):
    try:
        subprocess.run(
            [sys.executable, os.path.join(AION, "bin", "log_event.py"),
             "--type", "system", "--text", text, "--meta", json.dumps(meta)],
            check=False, timeout=30,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main():
    flags = []
    stats = {"checks": 0, "wired": 0, "errors": 0}
    wiring_text, wiring_refs = wiring_text_and_refs()
    units = unit_names()
    check_skills(flags, stats)
    check_competence(flags, stats)
    check_self(flags, stats, wiring_refs, wiring_text, units)
    check_theories(flags, stats)
    n_new = append_notices(flags)
    report = {"ts": now_iso(), "checks_run": stats["checks"],
              "wired": stats["wired"], "flags": flags,
              "notices_appended": n_new, "errors": stats["errors"]}
    with open(ATTESTATION, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[self_attest] checks=%d wired=%d flags=%d new_notices=%d errors=%d"
          % (stats["checks"], stats["wired"], len(flags), n_new, stats["errors"]))
    for fl in flags[:10]:
        print("  FLAG %s | %s" % (fl["reasons"][0], fl["details"]["detail"]))
    log_event("self_attest: %d checks, %d wired, %d flags"
              % (stats["checks"], stats["wired"], len(flags)),
              {"flags": len(flags), "new_notices": n_new})
    return 0


if __name__ == "__main__":
    sys.exit(main())
