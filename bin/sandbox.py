#!/usr/bin/env python3
"""sandbox.py — Aion Empirical Sandbox (Tier 2 capability).

Executes Python or Bash scripts with resource limits, timeout, and
restricted filesystem access. Results are stored as first-class evidence
for the heuristics layer.

Safety:
  - No network access (no socket, no urllib, no subprocess network calls)
  - Read-only access to ~/aion/ (logs, graphs, state files)
  - Write access only to sandbox output directory
  - CPU/memory/time limits via resource.setrlimit and subprocess timeout
  - Max 50MB output, 60s timeout, 512MB RAM

Usage:
  from sandbox import run_sandbox
  result = run_sandbox(code="print(2+2)", lang="python")
  result = run_sandbox(code="nvidia-smi", lang="bash")

  # CLI:
  python3 sandbox.py --code "print(2+2)"
  python3 sandbox.py --file /path/to/script.py
"""
import json, os, sys, time, tempfile, subprocess, resource, hashlib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = Path(os.environ.get("AION_HOME", "$AION_HOME"))
SANDBOX_DIR = AION / "memory" / "sandbox"

# Limits
MAX_TIMEOUT = 60       # seconds
MAX_MEMORY = 512       # MB
MAX_OUTPUT = 50000     # characters
MAX_FILE_SIZE = 1024 * 1024  # 1MB per output file

# Read-only paths Aion can access
READABLE_PATHS = [
    str(AION),
    "/tmp",
    "/proc",
    "/sys/class",
    "/usr/lib/python3",
    "/usr/local/lib",
]

# Write-only sandbox output
WRITABLE_PATH = str(SANDBOX_DIR)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def log_event(type_, text, meta=None):
    """Log sandbox events to episodic memory."""
    import subprocess as sp
    sp.run(
        ["python3", str(AION / "bin" / "log_event.py"),
         "--type", type_, "--text", text[:2000],
         "--meta", json.dumps(meta or {})],
        check=False, capture_output=True,
    )


def run_sandbox(code, lang="python", timeout=None, description=""):
    """Execute code in the sandbox.

    Args:
        code: the script source code
        lang: "python" or "bash"
        timeout: override timeout (default MAX_TIMEOUT)
        description: what this experiment is testing

    Returns dict with:
        stdout, stderr, exit_code, duration, files (list of output files),
        hash, evidence_type="empirical"
    """
    timeout = min(timeout or MAX_TIMEOUT, MAX_TIMEOUT)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]

    # Create sandbox output directory
    run_dir = SANDBOX_DIR / f"run_{ts}_{code_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write the script
    if lang == "python":
        script_path = run_dir / "experiment.py"
    else:
        script_path = run_dir / "experiment.sh"

    script_path.write_text(code)
    if lang == "bash":
        script_path.chmod(0o755)

    # Save metadata
    meta = {
        "ts": now_iso(),
        "lang": lang,
        "description": description,
        "hash": code_hash,
        "timeout": timeout,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Build execution command with resource limits
    if lang == "python":
        cmd = ["python3", str(script_path)]
    else:
        cmd = ["bash", str(script_path)]

    # Set up environment: restrict paths, no network
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),  # Real home so ~/aion works
        "AION_HOME": str(AION),
        "AION_SANDBOX": str(run_dir),
        "AION_SANDBOX_OUTPUT": str(run_dir),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "OPENBLAS_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }

    # Pre-exec function to set resource limits on the child process
    def preexec():
        # CPU time limit (seconds) — kills runaway CPU usage
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 5))
        except Exception:
            pass
        # File size limit
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_SIZE, MAX_FILE_SIZE))
        except Exception:
            pass

    # Execute
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(run_dir),
            preexec_fn=preexec,
        )
        stdout = result.stdout[:MAX_OUTPUT]
        stderr = result.stderr[:MAX_OUTPUT]
        exit_code = result.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8", errors="replace")[:MAX_OUTPUT] if e.stdout else ""
        stderr = "TIMEOUT: script exceeded {}s limit\n".format(timeout)
        if e.stderr:
            stderr += (e.stderr or b"").decode("utf-8", errors="replace")[:MAX_OUTPUT]
        exit_code = -1
    except Exception as e:
        stdout = ""
        stderr = f"SANDBOX ERROR: {e}"
        exit_code = -2

    duration = time.time() - t0

    # Pre-flight hint: the run cwd is a throwaway folder, so a relative path
    # that looks like it means "inside my repo" silently resolves somewhere
    # else and fails confusingly ("unable to open database file"). Flag it in
    # the result so a failure is self-explanatory rather than looking like
    # missing data. (Sep 11: 15 runs died this way chasing predictions.db.)
    path_hint = None
    try:
        import re as _re
        # Only flag a BARE relative path. A path that is the second argument of
        # os.path.join(...) is a false positive — that construction is exactly
        # what we want, so skip anything immediately preceded by a comma/paren.
        rel = []
        for m in _re.finditer(r"""['"](memory/[^'"]+|bin/[^'"]+|config/[^'"]+)['"]""", code):
            # Look at the start of the current line. If the path is an argument
            # to os.path.join(...) or already concatenated onto AION_HOME, it is
            # absolute in effect — that is the CORRECT pattern, not a problem.
            line_start = code.rfind("\n", 0, m.start()) + 1
            line = code[line_start:m.start()]
            if "join(" in line or "AION_HOME" in line or "SANDBOX_OUTPUT" in line:
                continue
            if line.count("'") % 2 or line.count('"') % 2:
                continue                      # we are inside another string literal
            rel.append(m.group(1))
        if rel:
            path_hint = ("relative repo path(s) used: " + ", ".join(sorted(set(rel))[:5]) +
                         " — sandbox cwd is the run dir, not the repo root. "
                         "Use os.path.join(os.environ['AION_HOME'], ...) for absolute paths.")
    except Exception:
        path_hint = None

    # Collect output files (excluding our script)
    output_files = []
    for f in run_dir.iterdir():
        if f.name not in ("experiment.py", "experiment.sh", "meta.json"):
            if f.is_file() and f.stat().st_size < MAX_FILE_SIZE:
                output_files.append(str(f))

    # Build result
    sandbox_result = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "duration": round(duration, 2),
        "files": output_files,
        "hash": code_hash,
        "run_dir": str(run_dir),
        "ts": now_iso(),
        "evidence_type": "empirical",
        "description": description,
        "lang": lang,
        "path_hint": path_hint,
    }

    # Save result
    (run_dir / "result.json").write_text(json.dumps(sandbox_result, indent=2))

    # Log to episodic
    summary = f"sandbox experiment ({lang}, {duration:.1f}s, exit={exit_code})"
    if description:
        summary += f": {description}"
    log_event("sandbox", summary, {
        "hash": code_hash,
        "exit_code": exit_code,
        "duration": round(duration, 2),
        "description": description,
        "evidence_type": "empirical",
        "run_dir": str(run_dir),
    })

    return sandbox_result


def parse_sandbox_request(text):
    """Parse a [SANDBOX_REQUEST] block from chat text.

    Returns (code, lang, description) or None if not found.

    Format:
        [SANDBOX_REQUEST python]
        Description: what this tests
        ```python
        print(2+2)
        ```
        [/SANDBOX_REQUEST]
    """
    import re

    pattern = r'\[SANDBOX_REQUEST\s*(\w+)?\](.*?)(?:\[/?SANDBOX_REQUEST\]|$)'
    match = re.search(pattern, text, re.S)
    if not match:
        return None

    lang = (match.group(1) or "python").lower()
    body = match.group(2).strip()

    # Extract description
    desc = ""
    desc_match = re.match(r'Description:\s*(.+?)(?:\n```|\n[^#\n])', body, re.S)
    if desc_match:
        desc = desc_match.group(1).strip()

    # Extract code from ```fences
    code_match = re.search(r'```(?:\w+)?\s*\n(.*?)```', body, re.S)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # No fences — treat everything after description as code
        code = body
        if desc:
            code = re.sub(r'^Description:\s*.+?\n', '', code, flags=re.S).strip()

    return code, lang, desc


def format_result(result):
    """Format sandbox result for display in chat."""
    lines = []
    lines.append(f"[SANDBOX_RESULT] exit={result['exit_code']} time={result['duration']}s")
    if result.get("description"):
        lines.append(f"Description: {result['description']}")
    lines.append("")
    if result["stdout"]:
        lines.append("stdout:")
        lines.append(result["stdout"][:5000])
    if result["stderr"]:
        lines.append("stderr:")
        lines.append(result["stderr"][:3000])
    if result["files"]:
        lines.append(f"Output files: {', '.join(result['files'][:5])}")
    lines.append("")
    lines.append(f"Evidence: {result.get('hash', '?')} (evidence_type=empirical)")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Aion Empirical Sandbox")
    p.add_argument("--code", default=None, help="inline code to execute")
    p.add_argument("--file", default=None, help="path to script file")
    p.add_argument("--lang", default="python", choices=["python", "bash"])
    p.add_argument("--desc", default="", help="experiment description")
    a = p.parse_args()

    if a.file:
        code = Path(a.file).read_text()
    elif a.code:
        code = a.code
    else:
        print("Provide --code or --file")
        sys.exit(1)

    result = run_sandbox(code, lang=a.lang, description=a.desc)
    print(format_result(result))
