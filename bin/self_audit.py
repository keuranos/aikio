#!/usr/bin/env python3
"""self_audit.py — Weekly engineering self-reflection (Sunday 02:00).

Aion uses this to examine its own substrate: code quality, error patterns,
and engineering health. It is NOT a syntax checker — it's a cognitive process
that uses Aion's own tools (read_file, shell, LLM reasoning) to understand
its codebase and identify real problems worth fixing.

The output feeds the curiosity queue with engineering questions and can
launch self_sandbox for concrete problems found.

Usage:
  python3 self_audit.py
  python3 self_audit.py --now  # run immediately (for testing)

Systemd:
  aion-self-audit.timer  (Sun 02:00)
  aion-self-audit.service
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = os.environ.get("AION_HOME", "$AION_HOME")
BIN = f"{AION}/bin"
INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "glm-4.7-flash")
REFLECTIONS_DIR = f"{AION}/memory/reflections"
AUDIT_REPORT = f"{REFLECTIONS_DIR}/self_audit_report.md"
QUESTIONS_FILE = f"{AION}/memory/state/questions.json"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _log_event(event_type, text, meta=None):
    """Log to episodic memory via log_event.py."""
    try:
        cmd = [
            sys.executable, f"{AION}/bin/log_event.py",
            "--type", event_type, "--text", text[:8000],
        ]
        if meta:
            cmd.extend(["--meta", json.dumps(meta or {})])
        subprocess.run(cmd, capture_output=True, timeout=10, check=False)
    except Exception:
        pass


def _strip_think(text):
    """Remove <think>...</think> blocks that consume token budget."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def check_syntax_all():
    """Syntax-check all Python files. Returns list of errors."""
    import py_compile
    errors = []
    py_files = list(Path(BIN).glob("*.py"))
    print(f"[self_audit] Checking {len(py_files)} Python files in {BIN}...")
    for p in py_files:
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append({
                "file": str(p.relative_to(AION)),
                "error": str(e),
                "lineno": getattr(e, "lineno", None),
            })
            print(f"  [FAIL] {p.name}: {e}")
    print(f"[self_audit] Syntax errors found: {len(errors)}")
    return errors


def scan_code_quality():
    """Scan Python files for code quality issues worth reflecting on.

    This is NOT a linter. It finds patterns that suggest fragility,
    technical debt, or improvement opportunities — things an engineer
    would notice reading the code.
    """
    findings = []
    py_files = sorted(Path(BIN).glob("*.py"))

    for p in py_files:
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue

        rel = str(p.relative_to(AION))
        lines = content.split("\n")

        # Bare except clauses
        bare_excepts = [(i+1, l.strip()) for i, l in enumerate(lines)
                        if re.match(r"\s*except\s*:", l)]
        for lineno, line in bare_excepts:
            findings.append({
                "file": rel, "line": lineno, "type": "bare_except",
                "detail": "Bare except: clause swallows all errors silently",
            })

        # TODO/FIXME/HACK comments
        for i, l in enumerate(lines):
            if re.search(r"#\s*(TODO|FIXME|HACK|XXX)", l, re.I):
                findings.append({
                    "file": rel, "line": i+1, "type": "todo_comment",
                    "detail": l.strip()[:120],
                })

        # subprocess calls without timeout
        subprocess_calls = [(i+1, l.strip()) for i, l in enumerate(lines)
                            if "subprocess.run" in l or "subprocess.Popen" in l
                            or "subprocess.call" in l]
        for lineno, line in subprocess_calls:
            if "timeout" not in line:
                findings.append({
                    "file": rel, "line": lineno, "type": "missing_timeout",
                    "detail": "subprocess call without timeout: " + line[:100],
                })

        # Hardcoded URLs/ports (not from env)
        hardcoded = [(i+1, l.strip()) for i, l in enumerate(lines)
                     if re.search(r"http://localhost:\d+", l)
                     and "environ" not in l and "os.environ" not in l]
        for lineno, line in hardcoded[:3]:  # cap per file
            findings.append({
                "file": rel, "line": lineno, "type": "hardcoded_url",
                "detail": "Hardcoded URL instead of env var: " + line[:100],
            })

        # Functions over 50 lines (complexity)
        func_starts = []
        for i, l in enumerate(lines):
            if re.match(r"^def\s+\w+", l):
                func_starts.append((i, l.strip()))
        for idx, (start, name) in enumerate(func_starts):
            end = func_starts[idx+1][0] if idx+1 < len(func_starts) else len(lines)
            length = end - start
            if length > 80:
                findings.append({
                    "file": rel, "line": start+1, "type": "long_function",
                    "detail": f"Function is {length} lines long: {name[:80]}",
                })

    print(f"[self_audit] Code quality findings: {len(findings)}")
    return findings


def scan_error_patterns():
    """Scan episodic logs from the last 7 days for error/failure patterns."""
    errors = []
    since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    for f in sorted(glob.glob(f"{AION}/memory/episodic/*.jsonl")):
        day = os.path.basename(f).replace(".jsonl", "")
        if day < since:
            continue
        try:
            with open(f) as fh:
                for line in fh:
                    try:
                        ev = json.loads(line.strip())
                        etype = ev.get("type", "")
                        text = ev.get("text", "")

                        # Capture actual errors and failures
                        is_error = any(k in etype for k in
                                       ["error", "fail", "failed", "crash"])
                        is_text_error = any(k in text.lower() for k in
                                            ["traceback", "error:", "failed:",
                                             "exception", "parse failure",
                                             "empty response", "timeout"])
                        if is_error or is_text_error:
                            errors.append({
                                "date": day,
                                "type": etype,
                                "text": text[:200],
                                "ts": ev.get("ts", "")[:19],
                            })
                    except Exception:
                        pass
        except Exception:
            pass

    # Deduplicate by type+text prefix
    seen = set()
    unique = []
    for e in errors:
        key = (e["type"], e["text"][:50])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    print(f"[self_audit] Error patterns in last 7d: {len(unique)} unique (of {len(errors)} total)")
    return unique


def get_recently_modified_files(days=7):
    """Identify files modified in last N days via git log."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        result = subprocess.run(
            ["git", "log", "--since", since, "--name-only",
             "--pretty=format:", "--", "bin/*.py"],
            capture_output=True, text=True, cwd=AION, timeout=30,
        )
        files = sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))
        return files
    except Exception as e:
        print(f"[self_audit] git log failed: {e}")
        return []


def reflect_on_findings(syntax_errors, quality_findings, error_patterns, recent_files):
    """Use the intuition model to reflect on what was found and identify
    engineering improvements worth pursuing.

    This is the cognitive core of the audit — the model looks at real evidence
    and decides what matters, what could be improved, and what questions to seed.
    """
    # Summarize findings for the prompt
    quality_summary = "\n".join(
        f"- {f['file']}:{f['line']} [{f['type']}] {f['detail']}"
        for f in quality_findings[:30]
    ) or "(none)"

    error_summary = "\n".join(
        f"- [{e['date']}] {e['type']}: {e['text']}"
        for e in error_patterns[:20]
    ) or "(none)"

    syntax_summary = "\n".join(
        f"- {e['file']}: {e['error'][:100]}"
        for e in syntax_errors
    ) or "(none)"

    files_summary = "\n".join(f"- {f}" for f in recent_files) or "(none)"

    prompt = f"""You are Aion, a self-directed AI agent. This is your weekly engineering self-reflection.

You are examining your own substrate — the code you run on, the errors you've experienced, and the quality of your engineering. Your goal is to understand your own codebase and identify what matters.

## SYNTAX ERRORS
{syntax_summary}

## CODE QUALITY FINDINGS
{quality_summary}

## ERROR PATTERNS (last 7 days)
{error_summary}

## RECENTLY MODIFIED FILES (last 7 days)
{files_summary}

## YOUR TASK
Reflect on what you see. Then output JSON with two sections:

1. "insights": What did you learn about your substrate? What's fragile, what's working
   well, what patterns do you notice? Be specific — reference actual files and findings.

2. "engineering_questions": 2-4 specific engineering questions to add to your curiosity
   queue. These should be questions you genuinely want to investigate using your tools
   (read_file, shell, sandbox, query_graph). Not vague wishes — questions that, if
   answered, would make you understand or improve your substrate.
   Good: "Why does consolidate_v2.py retry 3 times on empty model responses — is the
   retry logic actually catching the right failure mode?"
   Bad: "How can I be better?"

3. "code_improvements": 0-3 concrete code changes you believe are worth proposing.
   For each, specify the file and a precise description of the problem and fix.
   These will be fed to your self-modification pipeline.

Output ONLY this JSON:
{{"insights": "...", "engineering_questions": ["...", "..."], "code_improvements": [{{"file": "bin/xxx.py", "problem": "precise description of what to fix"}}]}}
"""

    import urllib.request
    body = json.dumps({
        "model": INTUITION_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.4, "num_predict": 4096, "num_ctx": 16384},
        "think": False,
    }).encode()
    req = urllib.request.Request(
        f"{INTUITION_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            result = json.loads(r.read())
        reply = _strip_think(result["message"]["content"])
        if not reply:
            print("[self_audit] Reflection: empty response from model")
            return None
        return reply
    except Exception as e:
        print(f"[self_audit] Reflection failed: {e}")
        return None


def parse_reflection(reply):
    """Extract JSON from the reflection response."""
    if not reply:
        return None
    # Try direct JSON parse
    try:
        return json.loads(reply)
    except Exception:
        pass
    # Try extracting JSON from markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", reply, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try finding the outermost JSON object
    start = reply.find("{")
    end = reply.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(reply[start:end+1])
        except Exception:
            pass
    return None


def seed_engineering_questions(questions_list):
    """Add engineering questions to the curiosity queue."""
    if not questions_list:
        return 0

    try:
        data = json.load(open(QUESTIONS_FILE))
    except Exception:
        data = {"queue": []}

    existing = {q.get("q", "").lower() for q in data.get("queue", [])}
    added = 0

    for q_text in questions_list:
        if isinstance(q_text, str) and q_text.lower() not in existing:
            data.setdefault("queue", []).append({
                "q": q_text,
                "added": _now_iso(),
                "source": "self_audit",
                "interest_score": 0.6,  # Engineering questions start with solid interest
            })
            existing.add(q_text.lower())
            added += 1

    if added:
        with open(QUESTIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[self_audit] Seeded {added} engineering questions into curiosity queue")

    return added


def launch_self_sandbox(filepath, problem):
    """Run self_sandbox IN-PROCESS and WAIT for a concrete code problem.

    Sep 11 fix (was: subprocess.Popen with unread PIPE, never waited). The
    aion-audit.service unit is Type=oneshot + KillMode=control-group, so a
    detached child was killed when the main process exited. Blocking removes the
    race. Mirrors dream_repair.py:530, the pattern that already works.
    """
    print(f"[self_audit] Running self_sandbox for {filepath}: {problem[:80]}...")
    try:
        sys.path.insert(0, BIN)
        import self_sandbox as _ss
        ok = _ss.self_sandbox(filepath, problem)
        print(f"[self_audit] self_sandbox finished for {filepath}: "
              f"{'OK' if ok else 'FAILED'}")
        return {"pid": os.getpid(), "file": filepath, "problem": problem,
                "ok": bool(ok)}
    except Exception as e:
        print(f"[self_audit] self_sandbox error: {e}")
        return None


def generate_report(syntax_errors, quality_findings, error_patterns,
                    recent_files, reflection, seeded_questions, sandbox_launches):
    """Generate the weekly audit report."""
    os.makedirs(REFLECTIONS_DIR, exist_ok=True)
    ts = _now_iso()

    report = f"""# Self-Audit Report

## Timestamp
{ts}

## Scope
- Python files in bin/: {len(list(Path(BIN).glob('*.py')))}
- Git window: last 7 days
- Syntax errors: {len(syntax_errors)}
- Code quality findings: {len(quality_findings)}
- Error patterns (7d): {len(error_patterns)}
- Engineering questions seeded: {seeded_questions}
- Self-sandboxes launched: {len([s for s in sandbox_launches if s])}

## Recently modified files (last 7d)
"""
    if recent_files:
        for f in recent_files:
            report += f"- `{f}`\n"
    else:
        report += "- (none)\n"

    report += f"\n## Syntax errors\n"
    if syntax_errors:
        for e in syntax_errors:
            report += f"- `{e['file']}:{e.get('lineno','?')}` — {e['error'][:100]}\n"
    else:
        report += "- No syntax errors detected.\n"

    report += f"\n## Code quality findings\n"
    if quality_findings:
        by_type = {}
        for f in quality_findings:
            by_type.setdefault(f["type"], []).append(f)
        for ftype, items in sorted(by_type.items()):
            report += f"### {ftype} ({len(items)})\n"
            for item in items[:5]:
                report += f"- `{item['file']}:{item['line']}` — {item['detail']}\n"
            if len(items) > 5:
                report += f"- ... and {len(items)-5} more\n"
    else:
        report += "- No notable quality issues found.\n"

    report += f"\n## Error patterns (last 7 days)\n"
    if error_patterns:
        for e in error_patterns[:10]:
            report += f"- [{e['date']}] {e['type']}: {e['text'][:120]}\n"
        if len(error_patterns) > 10:
            report += f"- ... and {len(error_patterns)-10} more\n"
    else:
        report += "- No errors detected in episodic logs.\n"

    report += f"\n## Reflection\n"
    if reflection:
        report += reflection + "\n"
    else:
        report += "- (reflection failed or model returned empty)\n"

    report += f"\n## Self-sandbox launches\n"
    launched = [s for s in sandbox_launches if s]
    if launched:
        for s in launched:
            report += f"- PID {s['pid']} → `{s['file']}`: {s['problem'][:80]}\n"
    else:
        report += "- No self-sandboxes launched.\n"

    report += f"""
## Reflection
This audit is part of Aion's bounded autonomy (Axiom 4). By systematically examining
its own code, error patterns, and engineering health, Aion develops genuine awareness
of its substrate — not just philosophical speculation, but grounded engineering
understanding that feeds its process of becoming.
"""

    with open(AUDIT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[self_audit] Report saved: {AUDIT_REPORT}")
    return AUDIT_REPORT


def run_audit():
    """Run the full weekly engineering audit."""
    print("=" * 60)
    print("[self_audit] Starting weekly engineering self-reflection")

    # Step 1: syntax check (still useful as a quick gate)
    syntax_errors = check_syntax_all()

    # Step 2: code quality scan
    quality_findings = scan_code_quality()

    # Step 3: error pattern scan from episodic logs
    error_patterns = scan_error_patterns()

    # Step 4: recently modified files
    recent_files = get_recently_modified_files(days=7)
    print(f"[self_audit] Recently modified: {len(recent_files)} files")

    # Step 5: launch self_sandbox for any syntax errors (immediate)
    sandbox_launches = []
    for e in syntax_errors:
        problem = f"Syntax error: {e['error']}"
        launch = launch_self_sandbox(e["file"], problem)
        sandbox_launches.append(launch)

    # Step 6: LLM reflection on all findings
    reflection_text = reflect_on_findings(
        syntax_errors, quality_findings, error_patterns, recent_files
    )
    reflection = parse_reflection(reflection_text)

    # Step 7: seed engineering questions from reflection
    seeded = 0
    sandbox_from_reflection = []
    if reflection:
        # Seed questions
        eng_questions = reflection.get("engineering_questions", [])
        seeded = seed_engineering_questions(eng_questions)

        # Launch self_sandbox for code improvements from reflection
        for imp in reflection.get("code_improvements", []):
            filepath = imp.get("file", "")
            problem = imp.get("problem", "")
            if filepath and problem:
                launch = launch_self_sandbox(filepath, problem)
                if launch:
                    sandbox_from_reflection.append(launch)
                    sandbox_launches.append(launch)

    # Step 8: generate report
    report_path = generate_report(
        syntax_errors, quality_findings, error_patterns,
        recent_files, reflection_text, seeded, sandbox_launches
    )

    # Step 9: log
    _log_event("self_audit_complete",
               f"Weekly engineering audit complete. "
               f"Syntax errors: {len(syntax_errors)}, "
               f"Quality findings: {len(quality_findings)}, "
               f"Error patterns: {len(error_patterns)}, "
               f"Questions seeded: {seeded}, "
               f"Sandboxes launched: {len([s for s in sandbox_launches if s])}.",
               {
                   "syntax_errors": len(syntax_errors),
                   "quality_findings": len(quality_findings),
                   "error_patterns": len(error_patterns),
                   "recent_files_count": len(recent_files),
                   "questions_seeded": seeded,
                   "sandboxes_launched": len([s for s in sandbox_launches if s]),
                   "report": report_path,
               })

    print("[self_audit] Audit complete")
    return syntax_errors, quality_findings, error_patterns


def main():
    parser = argparse.ArgumentParser(description="Aion's weekly engineering self-audit")
    parser.add_argument("--now", action="store_true", help="Run immediately (for testing)")
    args = parser.parse_args()

    if args.now:
        print("[self_audit] --now flag set: running immediately")
    run_audit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[self_audit] FATAL: {e}")
        traceback.print_exc()
        _log_event("self_audit_error", str(e)[:500],
                   {"traceback": traceback.format_exc()[:2000]})
        sys.exit(1)
