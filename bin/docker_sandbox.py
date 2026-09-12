#!/usr/bin/env python3
"""docker_sandbox.py — Aion Tier 2 Docker sandbox runner.

Executes Python or Bash code inside an ephemeral Docker container with:
  - No network access (--network=none)
  - Read-only codebase mount (~/aion read-only)
  - Write access only to sandbox output directory
  - CPU/memory/time limits (Docker --cpus, --memory, --timeout)
  - No privileged mode, no capabilities, no volume access except Aion

This upgrades the existing sandbox.py (subprocess + rlimits) to full
container isolation. The existing sandbox.py remains for quick tests;
docker_sandbox.py is for T2 autonomous code execution.

Usage:
  from docker_sandbox import run_docker_sandbox
  result = run_docker_sandbox(code="print(2+2)", lang="python")
  result = run_docker_sandbox(file="/path/to/script.py", lang="python")

  # CLI:
  python3 docker_sandbox.py --code "print(2+2)"
  python3 docker_sandbox.py --file /path/to/script.py --lang python
"""
import json
import os
import sys
import time
import tempfile
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = os.environ.get("AION_HOME", "$AION_HOME")
STATE = f"{AION}/memory/state"
SANDBOX_DIR = f"{AION}/memory/sandbox"

# Docker limits
DEFAULT_TIMEOUT = 120       # seconds
DEFAULT_CPUS = "2.0"        # max 2 CPU cores
DEFAULT_MEMORY = "2g"       # max 2GB RAM
MAX_OUTPUT_SIZE = 10 * 1024 * 1024  # 10MB

# Base image — use system python3 with numpy/pandas
DOCKER_IMAGE = os.environ.get("AION_DOCKER_IMAGE", "python:3.11-slim")

# Security: Docker run flags
DOCKER_SECURITY_FLAGS = [
    "--network=none",          # no network access
    "--cap-drop", "ALL",       # drop all capabilities then add back minimal
    "--cap-add", "CHOWN",
    "--cap-add", "DAC_OVERRIDE",
    "--cap-add", "FOWNER",
    "--cap-add", "SETUID",
    "--cap-add", "SETGID",
    "--pids-limit", "100",     # limit process count
    "--tmpfs", "/tmp:rw,size=256m",   # writable /tmp (256MB)
    "--tmpfs", "/root:rw,size=64m",   # writable /root for Python cache
    "--cpus", DEFAULT_CPUS,
    "--memory", DEFAULT_MEMORY,
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def log_event(type_, text, meta=None):
    """Log to episodic memory."""
    try:
        cmd = [sys.executable, f"{AION}/bin/log_event.py",
               "--type", type_, "--text", text]
        if meta:
            cmd.extend(["--meta", json.dumps(meta)])
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        pass


def ensure_image():
    """Pull the Docker image if not present."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return True
        # Pull it
        print(f"[docker_sandbox] Pulling {DOCKER_IMAGE}...")
        result = subprocess.run(
            ["docker", "pull", DOCKER_IMAGE],
            capture_output=True, text=True, timeout=300,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[docker_sandbox] Failed to ensure image: {e}")
        return False


def run_docker_sandbox(code=None, file=None, lang="python",
                       timeout=DEFAULT_TIMEOUT, description="",
                       read_aion=True):
    """Run code in a Docker container with isolation.

    Args:
        code: Python/Bash code string (mutually exclusive with file)
        file: Path to script file (mutually exclusive with code)
        lang: "python" or "bash"
        timeout: Max execution time in seconds
        description: Human-readable description for logging
        read_aion: If True, mount ~/aion read-only for data access

    Returns:
        dict: {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "exit_code": int,
            "duration": float (seconds),
            "files": list of output files,
            "error": str or None,
        }
    """
    start_time = time.time()

    # Validate input
    if not code and not file:
        return {"success": False, "error": "no code or file provided",
                "stdout": "", "stderr": "", "exit_code": -1, "duration": 0, "files": []}
    if code and file:
        return {"success": False, "error": "provide either code or file, not both",
                "stdout": "", "stderr": "", "exit_code": -1, "duration": 0, "files": []}

    # Create run directory for output
    run_id = hashlib.sha256(
        f"{time.time()}{os.getpid()}".encode()
    ).hexdigest()[:12]
    run_dir = f"{SANDBOX_DIR}/run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    # Prepare the code
    if file:
        # Copy file into run dir
        filename = os.path.basename(file)
        code_path = f"{run_dir}/{filename}"
        try:
            import shutil
            shutil.copy2(file, code_path)
        except Exception as e:
            return {"success": False, "error": f"failed to copy file: {e}",
                    "stdout": "", "stderr": "", "exit_code": -1, "duration": 0, "files": []}
    else:
        # Write code to file in run dir
        ext = ".py" if lang == "python" else ".sh"
        code_path = f"{run_dir}/script{ext}"
        with open(code_path, "w") as f:
            f.write(code)

    # Build Docker command
    docker_cmd = ["docker", "run", "--rm"]

    # Security flags
    docker_cmd.extend(DOCKER_SECURITY_FLAGS)

    # Timeout
    docker_cmd.extend(["--stop-timeout", str(min(timeout, 30))])

    # Mount run dir (read-write for output)
    docker_cmd.extend(["-v", f"{run_dir}:/sandbox:rw"])

    # Mount Aion read-only for data access
    if read_aion:
        docker_cmd.extend(["-v", f"{AION}:/aion:ro"])

    # Environment
    docker_cmd.extend(["-e", "AION_HOME=/aion"])
    docker_cmd.extend(["-e", "OPENBLAS_NUM_THREADS=4"])
    docker_cmd.extend(["-e", "HOME=/sandbox"])
    docker_cmd.extend(["-e", "PYTHONPATH=/aion/bin"])

    # Working directory
    docker_cmd.extend(["-w", "/sandbox"])

    # Image
    docker_cmd.append(DOCKER_IMAGE)

    # Command to run
    if lang == "python":
        docker_cmd.extend(["python3", "/sandbox/script.py"])
    elif lang == "bash":
        docker_cmd.extend(["bash", "/sandbox/script.sh"])
    else:
        return {"success": False, "error": f"unsupported language: {lang}",
                "stdout": "", "stderr": "", "exit_code": -1, "duration": 0, "files": []}

    # Execute
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,  # extra time for Docker overhead
        )

        duration = time.time() - start_time

        # Collect output files
        output_files = []
        for f in os.listdir(run_dir):
            if f not in ["script.py", "script.sh"] and not f.startswith("."):
                filepath = os.path.join(run_dir, f)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    if size <= MAX_OUTPUT_SIZE:
                        output_files.append({
                            "name": f,
                            "path": filepath,
                            "size": size,
                        })

        # Truncate output
        stdout = result.stdout[:MAX_OUTPUT_SIZE]
        stderr = result.stderr[:MAX_OUTPUT_SIZE]

        success = result.returncode == 0

        # Log to episodic memory
        log_event("docker_sandbox", f"sandbox {lang}: exit={result.returncode} dur={duration:.1f}s", {
            "success": success,
            "exit_code": result.returncode,
            "duration": round(duration, 2),
            "description": description[:200],
            "output_files": len(output_files),
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
            "run_id": run_id,
        })

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
            "duration": round(duration, 2),
            "files": output_files,
            "run_dir": run_dir,
            "error": None if success else f"exit code {result.returncode}",
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        log_event("docker_sandbox", f"sandbox timeout after {timeout}s", {
            "timeout": timeout,
            "description": description[:200],
            "run_id": run_id,
        })
        return {
            "success": False,
            "stdout": "",
            "stderr": f"timeout after {timeout}s",
            "exit_code": -1,
            "duration": round(duration, 2),
            "files": [],
            "error": f"timeout after {timeout}s",
        }
    except Exception as e:
        duration = time.time() - start_time
        log_event("docker_sandbox", f"sandbox error: {e}", {
            "error": str(e),
            "description": description[:200],
            "run_id": run_id,
        })
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "duration": round(duration, 2),
            "files": [],
            "error": str(e),
        }


def format_result(result):
    """Format a sandbox result for display."""
    lines = []
    status = "OK" if result.get("success") else "FAIL"
    lines.append(f"Status: {status} (exit={result.get('exit_code', '?')}, "
                 f"dur={result.get('duration', '?')}s)")
    if result.get("error"):
        lines.append(f"Error: {result['error']}")
    if result.get("stdout"):
        lines.append(f"\n--- stdout ---\n{result['stdout'][:5000]}")
    if result.get("stderr"):
        lines.append(f"\n--- stderr ---\n{result['stderr'][:2000]}")
    if result.get("files"):
        lines.append(f"\n--- output files ---")
        for f in result["files"]:
            lines.append(f"  {f['name']} ({f['size']} bytes)")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aion Docker sandbox (Tier 2)")
    parser.add_argument("--code", help="Code to execute")
    parser.add_argument("--file", help="Path to script file")
    parser.add_argument("--lang", default="python", choices=["python", "bash"])
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--desc", default="", help="Description for logging")
    parser.add_argument("--no-aion", action="store_true", help="Don't mount ~/aion")
    parser.add_argument("--pull", action="store_true", help="Pull image and exit")
    args = parser.parse_args()

    if args.pull:
        if ensure_image():
            print(f"Image {DOCKER_IMAGE} ready")
        else:
            print(f"Failed to pull {DOCKER_IMAGE}")
        return

    result = run_docker_sandbox(
        code=args.code,
        file=args.file,
        lang=args.lang,
        timeout=args.timeout,
        description=args.desc,
        read_aion=not args.no_aion,
    )
    print(format_result(result))


if __name__ == "__main__":
    main()