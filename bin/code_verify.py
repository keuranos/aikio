#!/usr/bin/env python3
"""code_verify.py — verify-then-repair gate for Aion-proposed code.

Never trust a diff that hasn't executed. Called by tool_propose_code BEFORE
any branch is created. Three stages on the RESULTING file content:

  1. syntax   — ast.parse of the full post-patch content
  2. old_text — early check that the anchor exists (saves a wasted cycle)
  3. import   — real interpreter import of the module in a subprocess
                (catches NameError at module level, missing deps, mangled
                function bodies that still parse)
  4. smoke    — for files with an argparse main: run --help

Known limits (stated honestly): cannot catch runtime logic errors like a
wrong model name used inside a function — those need the tool to actually
run, which is the council review's job. This gate catches everything that
dies before or during import.

On failure the caller returns stage + traceback to the investigating model
so it can repair within the same cycle. Repair loop = existing tool rounds.
"""
import ast
import os
import subprocess
import sys
import tempfile

AION = os.environ.get("AION_HOME", "$AION_HOME")
IMPORT_TIMEOUT = 30
SMOKE_TIMEOUT = 15


def _resulting_content(filepath, old_text, new_text):
    """Full content of the file as it would exist after the patch."""
    full = os.path.join(AION, filepath)
    is_new = not old_text and not os.path.exists(full)
    if is_new:
        return new_text, True
    if not os.path.exists(full):
        return None, False
    content = open(full, encoding="utf-8").read()
    if old_text == new_text:
        return None, False
    if old_text not in content:
        return None, False
    return content.replace(old_text, new_text, 1), False


def verify_code(filepath, old_text, new_text):
    """Verify a proposed change. Returns {ok, stage, error}."""
    if not filepath.endswith(".py"):
        return {"ok": True, "stage": "skip", "error": "",
                "note": "non-python file, execution gate not applicable"}

    content, is_new = _resulting_content(filepath, old_text, new_text)
    if content is None:
        if os.path.exists(os.path.join(AION, filepath)) and old_text:
            return {"ok": False, "stage": "old_text",
                    "error": "old_text not found in %s (or identical to "
                             "new_text) — re-read the file and re-propose" % filepath}
        return {"ok": False, "stage": "content",
                "error": "cannot produce resulting content for %s" % filepath}

    # Stage 1: syntax
    try:
        ast.parse(content)
    except SyntaxError as e:
        return {"ok": False, "stage": "syntax",
                "error": "SyntaxError: %s (line %s)" % (e.msg, e.lineno)}

    # Stage 2: import in isolated subprocess
    modname = os.path.basename(filepath)[:-3]
    tmpdir = tempfile.mkdtemp(prefix="aion_verify_")
    tmppath = os.path.join(tmpdir, os.path.basename(filepath))
    open(tmppath, "w", encoding="utf-8").write(content)
    # Aion modules import siblings (aion_env etc.) — put both bin/ and the
    # temp dir on sys.path, run with AION_HOME set like production.
    cmd = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); "
           "import %s" % (os.path.join(AION, "bin"), tmpdir, modname))
    env = dict(os.environ, AION_HOME=AION)
    try:
        p = subprocess.run(
            [sys.executable, "-c", cmd], cwd=AION, env=env,
            capture_output=True, text=True, timeout=IMPORT_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "import",
                "error": "import timed out after %ds — module-level code "
                         "blocks (loops, network calls at import time?)" % IMPORT_TIMEOUT}
    if p.returncode != 0:
        tail = "\n".join((p.stderr or p.stdout).strip().splitlines()[-6:])
        return {"ok": False, "stage": "import",
                "error": "import failed:\n%s" % tail}

    # Stage 3: smoke --help for argparse mains.
    # The temp file lives outside bin/, so put bin/ + tmpdir on PYTHONPATH
    # to mirror production layout (Aion modules import their siblings).
    if "__main__" in content and "argparse" in content:
        env["PYTHONPATH"] = os.path.join(AION, "bin") + os.pathsep + tmpdir
        try:
            p = subprocess.run(
                [sys.executable, tmppath, "--help"], cwd=AION, env=env,
                capture_output=True, text=True, timeout=SMOKE_TIMEOUT)
            if p.returncode != 0:
                tail = "\n".join((p.stderr or "").strip().splitlines()[-6:])
                return {"ok": False, "stage": "smoke",
                        "error": "--help exited %d:\n%s" % (p.returncode, tail)}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "smoke",
                    "error": "--help timed out after %ds" % SMOKE_TIMEOUT}

    return {"ok": True, "stage": "all",
            "error": "", "note": "syntax+import%s passed" %
                                 ("+smoke" if "__main__" in content and "argparse" in content else "")}


if __name__ == "__main__":
    import json
    if len(sys.argv) < 3:
        print("usage: code_verify.py <filepath> <old_text|-> <new_text|->")
        sys.exit(2)
    fp = sys.argv[1]
    old = "" if sys.argv[2] == "-" else open(sys.argv[2]).read() if os.path.exists(sys.argv[2]) else sys.argv[2]
    new = "" if sys.argv[3] == "-" else open(sys.argv[3]).read() if os.path.exists(sys.argv[3]) else sys.argv[3]
    print(json.dumps(verify_code(fp, old, new), indent=1))
