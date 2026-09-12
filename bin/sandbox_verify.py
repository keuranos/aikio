#!/usr/bin/env python3
"""sandbox_verify.py — pre-proposal verification for self-modification candidates.

Runs INSIDE the self_sandbox pipeline between AUTHOR and TEST:

    IDENTIFY -> SNAPSHOT -> AUTHOR -> SANDBOX-VERIFY -> TEST -> PROPOSE
                                          |  repair hint
                                          +--> AUTHOR retry (max 2)

Purpose: kill the silent-failure proposition class BEFORE it reaches the
operator. CI proves code is valid; this proves it DOES something against
copies of real state. Detector logic is imported from prop_audit.py so the
pre-merge and post-merge layers share one source of truth — improvements
to either propagate to both.

Checks:
  - schema_guess (static): filters on fields that exist in NO real state
    file (the CalFram 'resolved' class). Strong = filter context.
  - sandbox_run (dynamic): candidate applied to a temp AION_HOME with
    COPIES of real state; captures zero-effect and key-shrink.
  - zero_effect (dynamic): ran fine but processed 0 items on real data.
  - key_shrink (dynamic): script rewrites a state file and top-level or
    record keys DISAPPEAR (the calibration.json clobber class).
  - orphaned_def (static): functions added by the change are called by
    nothing in bin/ after the change is applied (dead-on-arrival hooks).
    Strong only when the description/problem claims wiring/integration —
    claim-vs-effect.

FAIL-OPEN: if the verifier itself errors, it returns pass=True with the
error recorded. A broken verifier must never become the new silent
blocker of Aion's self-modification loop.

No LLM calls. No GPU. Reads real state (copies), writes only temp dirs.

Usage (library):
    import sandbox_verify
    verdict = sandbox_verify.verify_change("bin/foo.py", old, new, desc)

Usage (CLI, for testing):
    python3 sandbox_verify.py --file bin/foo.py --new-file-content "$(cat x.py)"
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prop_audit  # noqa: E402  (detectors + sandbox conventions)

AION = os.environ.get("AION_HOME", "$AION_HOME")

# Claim words: when the prop claims wiring/integration, orphaned defs are
# a claim-vs-effect violation, not just future library code.
CLAIM_WORDS = re.compile(
    r"\b(hook|wire|wired|wiring|integrate|integrates|integrated|"
    r"invokes?|calls?|connect|connects)\b", re.I)

DEF_RX = re.compile(r"^def\s+(\w+)\(", re.M)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _write_path_in(source, rel):
    """Candidate source references this state file by name (write heuristic)."""
    base = os.path.basename(rel)
    stem = base.replace(".jsonl", "").replace(".json", "")
    return bool(re.search(r'[\'"][^\'"]*%s[^\'"]*[\'"]' % re.escape(base), source)
                or re.search(r'[\'"][^\'"]*%s[^\'"]*[\'"]' % re.escape(stem), source))


def _keys_of(path):
    """(field set, n) via prop_audit's census logic; None if unreadable."""
    parsed = prop_audit.file_field_keys(path)
    if not parsed:
        return None
    seen, n = parsed
    return set(seen.keys()), n


def _orphaned_defs(candidate, original, bin_dir, claim_match):
    """Defs added by the change that nothing in bin/ calls after applying it."""
    added = set(DEF_RX.findall(candidate)) - set(DEF_RX.findall(original))
    if not added:
        return None
    findings = []
    for fn in sorted(added):
        callers = []
        for name in os.listdir(bin_dir):
            if not name.endswith(".py"):
                continue
            try:
                src = open(os.path.join(bin_dir, name), errors="replace").read()
            except OSError:
                continue
            for ln in src.splitlines():
                s = ln.strip()
                if s.startswith("def ") or s.startswith("#"):
                    continue
                if re.search(r'\b%s\s*\(' % re.escape(fn), ln):
                    callers.append(name)
                    break
        if not callers:
            findings.append({
                "function": fn,
                "claim_match": claim_match,
                "note": ("added function is called nowhere in bin/ — "
                         "prop claims wiring" if claim_match else
                         "added function is called nowhere in bin/ (informational)"),
            })
    return findings or None


def _sandbox_run_checks(candidate, filepath, is_new, timeout_s, verdict):
    """Apply candidate in a temp AION_HOME (copies of real state) and run it.

    Returns (tmp_root, bin_dir). Caller owns cleanup of tmp_root — the
    orphaned-def scan must run against bin_dir while it still exists.
    """
    tmp = tempfile.mkdtemp(prefix="sverify_")
    try:
        os.makedirs(os.path.join(tmp, "memory", "state"), exist_ok=True)
        for rel, path, depth in prop_audit.census_state_files():
            if depth == 0:
                shutil.copy2(path, os.path.join(tmp, "memory", "state", rel))

        bin_src = os.path.join(AION, "bin")
        if not os.path.isdir(bin_src):
            # AION_HOME may be an isolated/test home without bin/ — the real
            # repo bin is the reference for execution and orphan scanning
            # (in production AION_HOME == REPO anyway).
            bin_src = os.path.join(prop_audit.REPO, "bin")
        shutil.copytree(bin_src, os.path.join(tmp, "bin"), dirs_exist_ok=True)

        # apply candidate
        cand_path = os.path.join(tmp, filepath)
        os.makedirs(os.path.dirname(cand_path), exist_ok=True)
        with open(cand_path, "w") as f:
            f.write(candidate)

        before_keys = {}
        for rel, path, depth in prop_audit.census_state_files():
            if depth == 0 and rel.endswith((".json", ".jsonl")):
                k = _keys_of(path)
                if k:
                    before_keys[rel] = k

        env = dict(os.environ)
        env["AION_HOME"] = tmp
        env.pop("GH_TOKEN", None)
        for k in list(env):
            if k.startswith("OLLAMA_"):
                env.pop(k)

        stdout = ""
        try:
            r = subprocess.run([sys.executable, cand_path],
                               capture_output=True, text=True,
                               timeout=timeout_s, cwd=os.path.join(tmp, "bin"),
                               env=env)
            stdout = r.stdout or ""
            verdict["run_exit"] = r.returncode
        except subprocess.TimeoutExpired:
            verdict["run_exit"] = "timeout"
        except Exception as e:
            verdict["run_exit"] = "error: %s" % e

        if verdict.get("run_exit") == 0:
            ze = prop_audit.scan_zero_effect(stdout)
            if ze:
                verdict["findings"].append({
                    "check": "zero_effect", "strength": "strong",
                    "hits": ze,
                    "stdout_tail": stdout[-300:]})
        else:
            verdict["skipped"].append({
                "check": "zero_effect",
                "reason": "run exit %s — behavioral TEST will judge" % verdict.get("run_exit")})

        # key-shrink: state files the candidate writes that LOST keys
        for rel, path, depth in prop_audit.census_state_files():
            if depth != 0 or rel not in before_keys:
                continue
            after_path = os.path.join(tmp, "memory", "state", rel)
            if not os.path.exists(after_path):
                continue  # deletion of state file: out of scope here
            after = _keys_of(after_path)
            if not after:
                continue
            (b_keys, b_n), (a_keys, a_n) = before_keys[rel], after
            if a_n == 0:
                continue
            if not _write_path_in(candidate, rel):
                continue
            lost = sorted(b_keys - a_keys)
            if lost:
                verdict["findings"].append({
                    "check": "key_shrink", "strength": "strong",
                    "file": rel, "lost_keys": lost,
                    "note": "script rewrites %s and these fields vanish" % rel})
        return tmp, os.path.join(tmp, "bin")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def verify_change(filepath, old_text, new_text, description="", problem="",
                  is_new=None, timeout_s=60):
    """Verify a candidate change. Returns verdict dict.

    verdict = {
        pass: bool        — no STRONG findings (weak ones don't block)
        findings: [...]   — {check, strength: strong|weak, ...}
        skipped: [...]    — {check, reason}
        repair_hint: str  — compact feedback for the AUTHOR retry prompt
        error: str|None   — verifier internal error (pass stays True)
    }
    """
    verdict = {"pass": True, "findings": [], "skipped": [], "repair_hint": "",
               "error": None, "ts": _now_iso()}
    tmp = None
    try:
        fullpath = os.path.join(AION, filepath)
        original = ""
        if os.path.exists(fullpath):
            with open(fullpath, errors="replace") as f:
                original = f.read()

        if is_new is None:
            is_new = (not old_text) and (not original)

        if old_text:
            if old_text not in original:
                verdict["pass"] = False
                verdict["findings"].append({
                    "check": "internal", "strength": "strong",
                    "note": "old_text not found in current file — cannot verify"})
                return verdict
            candidate = original.replace(old_text, new_text, 1)
        else:
            candidate = new_text

        real_keys = prop_audit.real_file_keys()
        claim_match = bool(CLAIM_WORDS.search(description or " ")
                           or CLAIM_WORDS.search(problem or " "))

        # ── Static: ghost fields on real state files ────────────────────
        try:
            ghosts = prop_audit.schema_guess_check(candidate, real_keys)
            if ghosts:
                verdict["findings"].append({
                    "check": "schema_guess", "strength": "strong",
                    "detail": ghosts})
        except Exception as e:
            verdict["skipped"].append({"check": "schema_guess",
                                       "reason": "verifier error: %s" % e})

        # ── Sandbox run (candidate applied to copy of real state) ───────
        runnable = is_new or ("__main__" in candidate)
        unsafe = [why for rx, why in prop_audit.UNSAFE_PATTERNS
                  if rx.search(candidate)]
        bin_dir = os.path.join(AION, "bin")
        if not os.path.isdir(bin_dir):
            bin_dir = os.path.join(prop_audit.REPO, "bin")

        if unsafe:
            verdict["skipped"].append({"check": "sandbox_run",
                                       "reason": "unsafe patterns: %s" % ", ".join(unsafe)})
        elif not runnable:
            verdict["skipped"].append({"check": "sandbox_run",
                                       "reason": "no __main__ entry (library edit)"})
        else:
            ran = _sandbox_run_checks(candidate, filepath, is_new,
                                      timeout_s, verdict)
            tmp, bin_dir = ran  # caller owns cleanup of tmp

        # ── Static: orphaned defs (against bin/ with candidate applied) ──
        try:
            orph = _orphaned_defs(candidate, original, bin_dir, claim_match)
            if orph:
                verdict["findings"].append({
                    "check": "orphaned_def",
                    "strength": "strong" if claim_match else "weak",
                    "detail": orph})
        except Exception as e:
            verdict["skipped"].append({"check": "orphaned_def",
                                       "reason": "verifier error: %s" % e})

        strong = [f for f in verdict["findings"] if f.get("strength") == "strong"]
        verdict["pass"] = not strong
        if strong:
            verdict["repair_hint"] = _build_repair_hint(strong, real_keys)
        return verdict

    except Exception as e:  # FAIL-OPEN
        verdict["error"] = "%s: %s" % (type(e).__name__, e)
        verdict["pass"] = True
        return verdict
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _build_repair_hint(strong, real_keys):
    """Compact, factual feedback for the AUTHOR retry prompt."""
    lines = ["Your previous attempt FAILED verification against REAL state data:"]
    for f in strong:
        if f["check"] == "schema_guess":
            for g in f["detail"]:
                lines.append(
                    "- You filter records in %s on field(s) %s that exist in NO "
                    "real record (real fields: %s; %d records on disk)." % (
                        g["file"], ", ".join(g["ghost_fields"]),
                        ", ".join(g["real_fields"][:15]), g["n_records"]))
        elif f["check"] == "zero_effect":
            hits = ", ".join(h.get("pattern", "?") for h in f.get("hits", []))
            lines.append("- Your script ran but processed 0 items on real data (%s). "
                         "Match the real record schema; do not invent fields." % hits)
            if f.get("stdout_tail"):
                lines.append("  Output was: %s" % f["stdout_tail"][-200:])
        elif f["check"] == "key_shrink":
            lines.append("- Your script rewrites %s and DESTROYS existing keys %s. "
                         "Merge your keys into the existing file; never replace it." % (
                             f["file"], ", ".join(f["lost_keys"])))
        elif f["check"] == "orphaned_def":
            for o in f["detail"]:
                lines.append(
                    "- Function %s is defined but called NOWHERE. If you claim it "
                    "is wired in, add the real call site." % o["function"])
        else:
            lines.append("- %s: %s" % (f["check"], f.get("note", "failed")))
    lines.append("Rewrite the fix using EXACTLY these real schemas. Do not repeat "
                 "the same mistakes.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Verify a self-mod candidate")
    ap.add_argument("--file", required=True)
    ap.add_argument("--old-text")
    ap.add_argument("--new-text")
    ap.add_argument("--new-file-content", help="content for new-file mode")
    ap.add_argument("--description", default="")
    ap.add_argument("--problem", default="")
    args = ap.parse_args()

    old = args.old_text or ""
    new = args.new_text or args.new_file_content or ""
    is_new = None
    if args.new_file_content is not None:
        is_new = True
    v = verify_change(args.file, old, new, args.description, args.problem,
                      is_new=is_new)
    print(json.dumps(v, indent=2))
    sys.exit(0 if v["pass"] else 1)


if __name__ == "__main__":
    main()
