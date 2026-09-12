#!/usr/bin/env python3
"""prop_audit.py — post-merge proposition execution audit (V1.1).

Detects the silent-failure class that CI cannot see:
  - ZERO_EFFECT:   merged prop runs but processes 0 items against REAL data
                    (the apply_temporal_decay "0 edges" class)
  - ORPHANED_HOOK:  prop adds a function claimed as an integration hook but
                    nothing in the repo ever calls it (dead code on arrival)
  - SCHEMA_GUESS:   prop reads fields that exist in NO known state file
                    (confabulated-schema class; per-file union whitelisting
                    keeps cross-file functions from false-positiving)
  - LEAK:           sandboxed script wrote to production state — attributed
                    only when the script itself opens the leaked file for
                    writing AND its production mtime falls inside the run
                    window (concurrent timers excluded)
  - UNSAFE_SKIP:    script touches LLMs/services/git — flagged for manual verify

Sandbox: temp AION_HOME with COPIES of real state + bin/, 60s timeout,
OLLAMA_*/GH_TOKEN scrubbed. Leak check = production state fingerprints.

Loop closure: flags are appended to memory/state/notices.jsonl (same channel
dream_stagnation uses) so consolidation/dreams can pick them up as
FAILURE_MODE findings, and written to memory/state/prop_audit.json for the
operator/nightly summary.

Usage:
  python3 bin/prop_audit.py [--days 7] [--episodic-note]
No LLM calls. No GPU. Safe to run anytime.

V1.1 changes:
  - census covers ALL memory/state files (depth-2, incl. heartbeats/)
  - leak attribution requires script write-path + mtime window match
  - union whitelist restricted to files referenced in the same code block
  - notices.jsonl integration (opt-in via flag in V1, see LOOP note)
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROPS_FILE = os.path.join(REPO, "memory", "state", "propositions.json")
OUT_FILE = os.path.join(REPO, "memory", "state", "prop_audit.json")
STATE_DIR = os.path.join(REPO, "memory", "state")
NOTICES = os.path.join(STATE_DIR, "notices.jsonl")

# Patterns that mean "ran fine but did nothing" (zero-effect). Tight on purpose.
ZERO_EFFECT_PATTERNS = [
    (re.compile(r"insufficient[_ ]data", re.I), "insufficient_data status"),
    (re.compile(r"\b(resolved|applied|processed|scored|extracted|found|updated)\D{0,12}\b0\b", re.I),
     "verb + 0 items"),
    (re.compile(r"\bn[_ =:]+0\b"), "n=0"),
    (re.compile(r"\bno\s+(resolved|matching|valid|usable)\b", re.I), "no usable records"),
]

# Scripts matching these are NOT auto-run (LLM/service side effects).
UNSAFE_PATTERNS = [
    (re.compile(r"ollama|1143[0-9]|localhost:\d{4}", re.I), "touches Ollama instances"),
    (re.compile(r"systemctl|\.service\b", re.I), "touches systemd services"),
    (re.compile(r"\bgit\s+(push|commit|merge)\b"), "mutates git history"),
    (re.compile(r"\brm\s+-rf?\b"), "bulk deletes"),
    (re.compile(r"requests\.(post|put|delete)|urllib\.request", re.I), "outbound HTTP writes"),
]

ENV_NOISE = {"AION_HOME"}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def run_git(*args):
    r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, timeout=30)
    return r.stdout if r.returncode == 0 else ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def census_state_files():
    """(relative_name, abs_path, depth) for every file under memory/state (depth 2)."""
    out = []
    for root, dirs, files in os.walk(STATE_DIR):
        rel_root = os.path.relpath(root, STATE_DIR)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        if depth > 1:
            dirs[:] = []
            continue
        for name in files:
            out.append((os.path.join(rel_root, name) if rel_root != "." else name,
                        os.path.join(root, name), depth))
    return out


def file_field_keys(path):
    """Top-level fields actually present in a state file. dict/JSONL-aware."""
    try:
        if path.endswith(".jsonl"):
            seen, n = {}, 0
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    n += 1
                    if isinstance(rec, dict):
                        for k in rec:
                            seen[k] = seen.get(k, 0) + 1
            return seen, n
        data = json.load(open(path))
        if isinstance(data, dict):
            items = None
            for k in ("propositions", "goals", "questions", "open", "buckets"):
                if isinstance(data.get(k), list) and data[k]:
                    items = data[k]
                    break
            if isinstance(items, list) and items and isinstance(items[0], dict):
                seen, n = {}, 0
                for it in items:
                    n += 1
                    for k in it:
                        seen[k] = seen.get(k, 0) + 1
                return seen, n
            return {k: 1 for k in data}, 1
        if isinstance(data, list) and data and isinstance(data[0], dict):
            seen, n = {}, 0
            for it in data:
                n += 1
                for k in it:
                    seen[k] = seen.get(k, 0) + 1
            return seen, len(data)
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    return None


def real_file_keys():
    keys = {}
    for rel, path, depth in census_state_files():
        if not (rel.endswith(".json") or rel.endswith(".jsonl")):
            continue
        parsed = file_field_keys(path)
        if parsed:
            keys[rel] = parsed
    return keys


def state_fingerprint():
    fp = {}
    for rel, path, _ in census_state_files():
        if rel == "prop_audit.json":
            continue
        try:
            fp[rel] = sha256(path)
        except OSError:
            pass
    return fp


def schema_guess_check(source, real_keys):
    """Fields accessed on state files that exist in NO known state file.

    Cross-file conflation guard: the union whitelist only includes fields
    from OTHER state files whose names/constants are referenced inside the
    same top-level code block as the access.
    """
    findings = []
    const_vals = dict(re.findall(r'(?m)^([A-Z_][A-Z0-9_]*)\s*=\s*[fF]?["\']([^"\']+)["\']', source))
    blocks = re.split(r'(?m)^(?=def |class )', source)
    for fname_state, (seen, n) in real_keys.items():
        if n == 0:
            continue
        base = fname_state.replace(".jsonl", "").replace(".json", "")
        file_consts = [c for c, v in const_vals.items() if fname_state in v]
        accessing_blocks = []
        for i, b in enumerate(blocks):
            if fname_state in b or ('"%s"' % base) in b or ("'%s'" % base) in b:
                accessing_blocks.append((i, b))
                continue
            if any(re.search(r'\b%s\b' % re.escape(c), b) for c in file_consts):
                accessing_blocks.append((i, b))
        if not accessing_blocks:
            continue
        # union whitelist: fields from other state files referenced in the SAME blocks
        whitelist = set()
        for other, (other_seen, _) in real_keys.items():
            if other == fname_state:
                continue
            other_consts = [c for c, v in const_vals.items() if other in v]
            obase = other.replace(".jsonl", "").replace(".json", "")
            for i, b in accessing_blocks:
                if other in b or ('"%s"' % obase) in b or ("'%s'" % obase) in b or \
                   any(re.search(r'\b%s\b' % re.escape(c), b) for c in other_consts):
                    whitelist |= set(other_seen)
        scoped_src = "\n".join(b for _, b in accessing_blocks)
        accessed = set(re.findall(r'\.get\(["\'](\w+)["\']', scoped_src)) | \
                   set(re.findall(r'\[\s*["\'](\w+)["\']\s*\]', scoped_src))
        accessed -= ENV_NOISE
        # fields the script itself WRITES (dict literals / item assignment):
        # decorate-then-filter on an in-memory field is legitimate; a true
        # confabulation exists neither on disk nor among the script's writes
        written = set(re.findall(r'\[\s*["\'](\w+)["\']\s*\]\s*=[^=]', scoped_src)) | \
                  set(re.findall(r'["\'](\w+)["\']\s*:', scoped_src))
        ghosts_all = sorted(a for a in accessed
                            if a not in seen and a not in whitelist and a not in written)
        if not ghosts_all:
            continue
        # Tiering: the silent-failure signature is FILTERING on a ghost field
        # (if/while/comprehension condition -> 0 matches -> no-op). Accesses
        # outside filter context may be legacy fallbacks or transient dicts —
        # informational only, not a FLAG.
        strong, weak = [], []
        for g in ghosts_all:
            lines = [ln for ln in scoped_src.splitlines() if ('"%s"' % g) in ln or ("'%s'" % g) in ln]
            in_filter = any(re.search(r'\b(if|while)\b', ln) for ln in lines)
            (strong if in_filter else weak).append(g)
        if strong:
            findings.append({
                "file": fname_state, "ghost_fields": strong,
                "weak_accesses": weak,
                "real_fields": sorted(seen.keys()), "n_records": n,
                "note": "fields used as record filters but absent from all real state files",
            })
    return findings


def sandbox_run(script_rel, timeout_s=60):
    """Run a NEW prop script against a COPY of real state in a temp AION_HOME."""
    tmp = tempfile.mkdtemp(prefix="prop_audit_")
    result = {"sandbox": True, "tmp": tmp}
    try:
        os.makedirs(os.path.join(tmp, "memory", "state"), exist_ok=True)
        for rel, path, depth in census_state_files():
            if depth == 0:
                shutil.copy2(path, os.path.join(tmp, "memory", "state", rel))
        bin_src = os.path.join(REPO, "bin")
        if os.path.isdir(bin_src):
            shutil.copytree(bin_src, os.path.join(tmp, "bin"), dirs_exist_ok=True)
        before = state_fingerprint()
        t_start = datetime.now(timezone.utc).timestamp()

        env = dict(os.environ)
        env["AION_HOME"] = tmp
        env.pop("GH_TOKEN", None)
        for k in list(env):
            if k.startswith("OLLAMA_"):
                env.pop(k)

        script_path = os.path.join(tmp, script_rel)
        if not os.path.exists(script_path):
            return {"sandbox": False, "error": "script missing: %s" % script_rel}
        try:
            r = subprocess.run([sys.executable, script_path], capture_output=True,
                               text=True, timeout=timeout_s, cwd=os.path.join(tmp, "bin"), env=env)
            result["exit"] = r.returncode
            result["stdout_tail"] = (r.stdout or "")[-1500:]
            result["stderr_tail"] = (r.stderr or "")[-500:]
        except subprocess.TimeoutExpired:
            result["exit"] = "timeout"
            result["stdout_tail"] = ""
            result["stderr_tail"] = "timed out after %ss" % timeout_s
        t_end = datetime.now(timezone.utc).timestamp()

        # leak: production file changed AND script contains a write-path to it
        source = open(os.path.join(REPO, script_rel), errors="replace").read()
        changed = [rel for rel, h in state_fingerprint().items() if before.get(rel) != h]
        leaks = []
        for rel in changed:
            base = os.path.basename(rel)
            writes_it = re.search(r'open\([^)]*["\']([a-zA-Z0-9_./]*%s["\'])' % re.escape(base),
                                  source) or re.search(r'["\'][^"\']*%s["\']' % re.escape(base), source)
            mtime = os.path.getmtime(os.path.join(STATE_DIR, rel)) \
                if os.path.exists(os.path.join(STATE_DIR, rel)) else 0
            in_window = t_start - 2 <= mtime <= t_end + 2
            if writes_it and in_window:
                leaks.append(rel)
        result["production_leak"] = leaks
        result["changed_but_unattributed"] = [r for r in changed if r not in leaks]
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scan_zero_effect(stdout):
    hits = []
    for rx, label in ZERO_EFFECT_PATTERNS:
        m = rx.search(stdout or "")
        if m:
            hits.append({"pattern": label, "match": m.group(0)[:60]})
    return hits


def orphaned_hook_check(prop):
    mc, pmc, f = prop.get("merge_commit"), prop.get("pre_merge_commit"), prop.get("file")
    if not (mc and pmc and f):
        return None
    diff = run_git("diff", pmc, mc, "--", f)
    added_defs = re.findall(r'^\+def\s+(\w+)\(', diff, re.M)
    if not added_defs:
        return None
    findings = []
    for fn in added_defs:
        callers = []
        for name in os.listdir(os.path.join(REPO, "bin")):
            if not name.endswith(".py"):
                continue
            src = open(os.path.join(REPO, "bin", name), errors="replace").read()
            calls = re.findall(r'^.*%s\s*\(' % re.escape(fn), src, re.M)
            real = [c for c in calls if not c.strip().startswith("def ") and not c.strip().startswith("#")]
            if real:
                callers.append(name)
        findings.append({"function": fn, "called_in": callers,
                         "orphaned": len(callers) == 0})
    return findings


def append_notices(entries):
    """Loop closure: surface flags in notices.jsonl for dreams/consolidation.

    Dedup: skip a notice whose prop id + flag set already appears in the
    last 300 lines (prevents 7 nights of identical re-flags per window).
    """
    try:
        recent = []
        try:
            with open(NOTICES) as f:
                recent = f.readlines()[-300:]
        except OSError:
            pass
        seen_sigs = set()
        for ln in recent:
            try:
                r = json.loads(ln)
                d = r.get("details", {})
                if isinstance(d, dict) and d.get("prop"):
                    seen_sigs.add((d.get("prop"), tuple(sorted(r.get("reasons", [])))))
            except (json.JSONDecodeError, AttributeError):
                continue
        fresh = [e for e in entries
                 if (e["details"].get("prop"), tuple(sorted(e["reasons"]))) not in seen_sigs]
        if fresh:
            with open(NOTICES, "a") as f:
                for e in fresh:
                    f.write(json.dumps(e) + "\n")
        return len(fresh)
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--notices", action="store_true",
                    help="append FLAG entries to notices.jsonl (loop closure)")
    args = ap.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    data = json.load(open(PROPS_FILE))
    items = data.get("propositions", data) if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())

    real_keys = real_file_keys()
    report = {"timestamp": now_iso(), "window_days": args.days,
              "audited": [], "summary": {}}
    flags = {"zero_effect": 0, "orphaned_hook": 0, "schema_guess": 0,
             "leak": 0, "unsafe_skip": 0, "ok": 0, "errors": 0}
    notice_entries = []

    for p in items:
        if p.get("status") != "accepted" or not p.get("merged_to"):
            continue
        ts = parse_ts(p.get("answer_ts") or p.get("ts"))
        if not ts or ts < cutoff:
            continue
        f = p.get("file", "")
        entry = {"id": p.get("id"), "file": f, "merge_commit": p.get("merge_commit", "")[:8],
                 "checks": []}

        src_path = os.path.join(REPO, f)
        if not f.endswith(".py") or not os.path.exists(src_path):
            entry["checks"].append({"check": "scope", "result": "SKIP",
                                    "note": "not a python file or missing"})
            report["audited"].append(entry)
            continue
        source = open(src_path, errors="replace").read()

        ghosts = schema_guess_check(source, real_keys)
        if ghosts:
            flags["schema_guess"] += 1
            entry["checks"].append({"check": "schema_guess", "result": "FLAG", "detail": ghosts})

        pmc = p.get("pre_merge_commit")
        is_new = bool(pmc) and subprocess.run(
            ["git", "-C", REPO, "cat-file", "-e", "%s:%s" % (pmc, f)],
            capture_output=True).returncode != 0

        if is_new:
            unsafe = [why for rx, why in UNSAFE_PATTERNS if rx.search(source)]
            if unsafe:
                flags["unsafe_skip"] += 1
                entry["checks"].append({"check": "sandbox_run", "result": "SKIP",
                                        "note": "unsafe patterns: %s" % ", ".join(unsafe)})
            else:
                rr = sandbox_run(f)
                if rr.get("production_leak"):
                    flags["leak"] += 1
                    entry["checks"].append({"check": "leak", "result": "FLAG",
                                            "files": rr["production_leak"]})
                ze = scan_zero_effect(rr.get("stdout_tail", ""))
                if ze:
                    flags["zero_effect"] += 1
                    entry["checks"].append({"check": "zero_effect", "result": "FLAG",
                                            "hits": ze, "exit": rr.get("exit"),
                                            "stdout_tail": rr.get("stdout_tail", "")[-400:]})
                elif not rr.get("production_leak"):
                    entry["checks"].append({"check": "processeffect", "result": "OK",
                                            "exit": rr.get("exit")})
        else:
            hooks = orphaned_hook_check(p)
            if hooks:
                orph = [h for h in hooks if h["orphaned"]]
                if orph:
                    flags["orphaned_hook"] += len(orph)
                    entry["checks"].append({"check": "run_scope", "result": "FLAG",
                                            "type": "orphaned_hook", "detail": hooks})

        if entry["checks"] and all(c.get("result") != "FLAG" for c in entry["checks"]):
            flags["ok"] += 1
            entry["result"] = "OK"
        elif entry["checks"]:
            entry["result"] = "FLAG"
            if args.notices:
                notice_entries.append({
                    "ts": now_iso(), "score": 0.5,
                    "reasons": ["prop_audit:" + c["check"] for c in entry["checks"]
                                if c.get("result") == "FLAG"],
                    "details": {"prop": p.get("id"), "file": f,
                                "flags": [c for c in entry["checks"] if c.get("result") == "FLAG"]},
                })
        report["audited"].append(entry)

    report["summary"] = flags
    with open(OUT_FILE, "w") as fo:
        json.dump(report, fo, indent=2)

    print("=== Proposition Execution Audit (%d-day window) ===" % args.days)
    for e in report["audited"]:
        print("  [%s] %s (%s)" % (e.get("result", "-"), e["id"], e["file"]))
        for c in e["checks"]:
            if c.get("result") == "FLAG":
                print("      FLAG %-14s %s" % (c["check"], json.dumps(
                    {k: v for k, v in c.items() if k not in ("check", "result")})[:300]))
            elif c.get("result") == "SKIP":
                print("      SKIP %-14s %s" % (c.get("check", "?"), c.get("note", "")))
    print("\nSummary:", json.dumps(flags))
    print("Written:", OUT_FILE)
    if args.notices:
        n = append_notices(notice_entries)
        print("Notices appended:", n if n is not None else "FAILED")


if __name__ == "__main__":
    main()
