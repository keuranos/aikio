#!/usr/bin/env python3
"""ci_check.py — Aion Tier 1 CI validation script.

Runs when Aion proposes code changes to a git branch.
Checks:
  1. Python syntax (py_compile on all .py files)
  2. Bash syntax (bash -n on all .sh files)
  3. Axiom compliance (no edits to AXIOMS.md or protected config)
  4. No secrets/API keys exposed
  5. No destructive commands (rm -rf /, drop, mkfs, etc.)

Network access is ALLOWED — Aion may import urllib, requests, socket, etc.
This restriction was lifted 2026-07-29 to allow Aion to build networked features.

Usage:
  python3 ci_check.py [--branch <branch_name>] [--base <base_branch>]
  python3 ci_check.py  # checks working tree against HEAD

Exit codes:
  0 = all checks pass
  1 = check(s) failed
"""
import sys
import os
import subprocess
import re
import argparse
from pathlib import Path

AION = os.environ.get("AION_HOME", "$AION_HOME")
BASE_BRANCH = "master"

# Files that must NEVER be modified by Aion
PROTECTED_FILES = [
    "AXIOMS.md",
    "config/secrets.env",
    "config/aion.env",
]

# Patterns that indicate destructive commands
# (kept — these are safety rails, not capability restrictions)
DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bdrop\s+(table|database)\b",
    r"\bmkfs\b",
    r"\bdd\s+if=.*of=/dev/",
    r"\bshutdown\s+-[hr]",
    r"\breboot\s*$",
    r"\bkill\s+-9\s+1\b",
    r"\biptables\s+-F\b",
    r"\bchmod\s+-R\s+777\s+/",
]

# Patterns that indicate secrets
SECRET_PATTERNS = [
    r"(?i)(api[_-]?key|token|password|secret)\s*=\s*['\"][^'\"]{10,}",
    r"(?i)(BEGIN\s+(RSA|OPENSSH|PGP)\s+(PRIVATE|PUBLIC)\s+KEY)",
    r"[a-f0-9]{32,}",  # long hex strings (potential keys)
]


def get_changed_files(base="HEAD"):
    """Get list of files changed vs base."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base],
            capture_output=True, text=True, cwd=AION, timeout=30,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
    except Exception:
        pass
    # Fallback: check working tree
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=AION, timeout=30,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                f = line[3:].strip()
                if f:
                    files.append(f)
        return files
    except Exception:
        return []


def check_python_syntax(filepath):
    """Check Python syntax using py_compile."""
    fullpath = os.path.join(AION, filepath)
    if not os.path.exists(fullpath):
        return True, "file deleted"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", fullpath],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, "OK"
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


def check_bash_syntax(filepath):
    """Check Bash syntax using bash -n."""
    fullpath = os.path.join(AION, filepath)
    if not os.path.exists(fullpath):
        return True, "file deleted"
    try:
        result = subprocess.run(
            ["bash", "-n", fullpath],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, "OK"
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


def check_protected_files(changed_files):
    """Check that protected files are not modified."""
    violations = []
    for f in changed_files:
        for pf in PROTECTED_FILES:
            if f == pf or f.endswith("/" + pf) or pf in f:
                violations.append(f)
    if violations:
        return False, f"Protected files modified: {', '.join(violations)}"
    return True, "OK"


def check_content_patterns(filepath):
    """Check file content for forbidden patterns."""
    fullpath = os.path.join(AION, filepath)
    if not os.path.exists(fullpath):
        return True, "OK"
    try:
        content = open(fullpath).read()
    except Exception:
        return True, "OK"  # skip if can't read

    issues = []

    # Check destructive patterns
    for pattern in DESTRUCTIVE_PATTERNS:
        for line_num, line in enumerate(content.split("\n"), 1):
            if re.search(pattern, line, re.IGNORECASE):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                issues.append(f"destructive pattern '{pattern}' at line {line_num}")

    # Check secret patterns (only in .py files)
    if filepath.endswith(".py"):
        for pattern in SECRET_PATTERNS:
            matches = re.findall(pattern, content)
            if matches and len(matches) > 2:
                issues.append(f"potential secrets ({len(matches)} matches)")

    if issues:
        return False, "; ".join(issues[:5])
    return True, "OK"


def run_ci(base=BASE_BRANCH):
    """Run all CI checks. Returns (passed:bool, results:list)."""
    changed = get_changed_files(base)
    if not changed:
        print("[ci_check] No changes detected — PASS (vacuous)")
        return True, [{"check": "no_changes", "passed": True, "msg": "no changes"}]

    print(f"[ci_check] {len(changed)} file(s) changed vs {base}:")
    for f in changed:
        print(f"  - {f}")
    print()

    results = []
    all_passed = True

    # Check 1: Protected files
    passed, msg = check_protected_files(changed)
    results.append({"check": "protected_files", "passed": passed, "msg": msg})
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] protected_files: {msg}")
    if not passed:
        all_passed = False

    # Check 2-5: Per-file checks (only for code files, not memory/state)
    SKIP_DIRS = ("memory/", "graphs/", "webui/data/", ".git")

    for filepath in changed:
        if not filepath:
            continue
        # Skip non-code files (memory state, graph data, etc.)
        if any(filepath.startswith(d) for d in SKIP_DIRS):
            continue

        # Python syntax
        if filepath.endswith(".py"):
            passed, msg = check_python_syntax(filepath)
            results.append({"check": f"python_syntax:{filepath}", "passed": passed, "msg": msg})
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] python_syntax:{filepath}: {msg[:80]}")
            if not passed:
                all_passed = False

        # Bash syntax
        if filepath.endswith(".sh"):
            passed, msg = check_bash_syntax(filepath)
            results.append({"check": f"bash_syntax:{filepath}", "passed": passed, "msg": msg})
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] bash_syntax:{filepath}: {msg[:80]}")
            if not passed:
                all_passed = False

        # Content patterns (only for .py and .sh files)
        if filepath.endswith(".py") or filepath.endswith(".sh"):
            passed, msg = check_content_patterns(filepath)
        else:
            passed, msg = True, "OK (non-code file)"
        results.append({"check": f"content_patterns:{filepath}", "passed": passed, "msg": msg})
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] content_patterns:{filepath}: {msg[:80]}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("[ci_check] ALL CHECKS PASSED")
    else:
        print("[ci_check] CHECKS FAILED — see above")

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(description="Aion Tier 1 CI validation")
    parser.add_argument("--branch", default=None, help="Branch to check (checkout first)")
    parser.add_argument("--base", default=BASE_BRANCH, help="Base branch to diff against")
    args = parser.parse_args()

    if args.branch:
        # Checkout the branch first
        subprocess.run(["git", "checkout", args.branch], cwd=AION, capture_output=True)

    passed, results = run_ci(base=args.base)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()