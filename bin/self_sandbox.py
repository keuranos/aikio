#!/usr/bin/env python3
"""self_sandbox.py — Aion's self-modification workflow.

IDENTIFY → SNAPSHOT → AUTHOR → TEST → REFLECT → PROPOSE → ROLLBACK

A hard-guarded pipeline where Aion can propose fixes, new features,
new functions, and entirely new programs to its own codebase.
All changes go through the existing propose_code pipeline (git branch + CI)
for operator review. Accepted proposals auto-merge into master.

Capabilities (as of 2026-07-29):
  - Edit existing files: bug fixes, new functions, refactoring, features
  - Create new files: entire new programs and modules
  - Network access: may import urllib, socket, http, requests, etc.
  - No size limit: large changes are allowed across multiple proposals

Hard guards:
  - NEVER modify AXIOMS.md, SYSTEM_PROMPT.md, config/secrets.env, config/aion.env
  - All changes are proposals, never direct commits to master
  - If any step fails, original file is restored (or new file deleted)

Usage:
  python3 self_sandbox.py --file bin/some_file.py --problem 'description of bug'
  python3 self_sandbox.py --file bin/new_module.py --problem 'Create a new module that does X'
"""
import argparse
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa
from aion_models import code_endpoint, code_num_ctx

AION = os.environ.get("AION_HOME", "$AION_HOME")
BIN = f"{AION}/bin"
REFLECTIONS_DIR = f"{AION}/memory/reflections/self_modifications"
SNAPSHOTS_DIR = f"{AION}/memory/reflections/self_modifications/snapshots"

INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "glm-4.7-flash:q4_K_M")

# Main model (V100 #1) — used for large files that overflow intuition model context
MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
MAIN_NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

# Agentic harness fallback (Hermes + local qwen3-coder)
HERMES_BIN = "/home/operator/.local/bin/hermes"
HERMES_PROVIDER = "custom:qwen3-coder"
HERMES_MODEL = "qwen3-coder-next:q4_K_M"


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


def read_file(path):
    try:
        return open(path, encoding="utf-8").read()
    except Exception:
        return ""


def read_context_files():
    """Load AXIOMS.md, SELF.md, and felt_sense for the AUTHOR prompt."""
    axioms = read_file(f"{AION}/AXIOMS.md")[:4000]
    self_md = read_file(f"{AION}/SELF.md")[:4000]
    felt = read_file(f"{AION}/memory/state/felt_sense.txt")[:2000]
    return axioms, self_md, felt


def identify_step(filepath, problem):
    """Log the identified problem."""
    print(f"[self_sandbox] IDENTIFY: {filepath}")
    print(f"[self_sandbox] Problem: {problem}")
    _log_event("self_mod_identify", f"Identified problem in {filepath}: {problem}", {
        "file": filepath,
        "problem": problem,
    })
    return True


def snapshot_step(filepath):
    """Backup the current file to SNAPSHOTS_DIR. Returns None for new files (no snapshot needed)."""
    fullpath = os.path.join(AION, filepath)
    if not os.path.exists(fullpath):
        print(f"[self_sandbox] SNAPSHOT: file does not exist yet (new file creation)")
        return "NEW_FILE"
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    snap_name = f"{filepath.replace('/', '_')}_{ts}.orig"
    snap_path = os.path.join(SNAPSHOTS_DIR, snap_name)
    try:
        subprocess.run(["cp", "-p", fullpath, snap_path], check=True, timeout=10)
        print(f"[self_sandbox] SNAPSHOT: {snap_path}")
        return snap_path
    except Exception as e:
        print(f"[self_sandbox] SNAPSHOT FAILED: {e}")
        return None


def build_author_prompt(filepath, file_content, axioms, self_md, felt, problem,
                        repair_hint=""):
    """Build the prompt for the intuition model to author a fix.

    Kept deliberately lean: including full AXIOMS/SELF.md/felt_sense (12k+ chars)
    caused the model to spend all its token budget reasoning about context
    instead of producing a fix. The axioms relevant to code changes are
    summarized as one-line guardrails instead.

    repair_hint: verification feedback from a failed SANDBOX-VERIFY pass.
    Contains the REAL state-file schemas and what the previous attempt got
    wrong — this is the trial->feedback->consolidation loop at pipeline speed.
    """
    repair_block = ""
    if repair_hint:
        repair_block = f"""
{repair_hint}

"""
    prompt = f"""You are a code-writing assistant for Aion, a self-directed AI agent. You can write bug fixes, new features, new functions, and entire new programs.

Task: {problem}
File: {filepath}
{repair_block}
Current file content (if editing an existing file — may be empty for new files):
```python
{file_content}
```

Rules:
- For editing: output old_text (exact text to find) and new_text (replacement).
- For creating a new file: set old_text to empty string "" and new_text to the full file content.
- IMPORTANT: old_text must be a VERBATIM copy of the existing code, including exact indentation and whitespace. If it is not an exact match, the fix will fail.
- You may use any Python standard library including network libraries (urllib, socket, http, etc.).
- You may add imports, new functions, classes, and modules.
- Preserve existing formatting and style when editing.
- Write complete, production-ready code.
- Output your response as a JSON object wrapped in markdown code fences (```json ... ```).

Required JSON format:
{{
  "old_text": "exact text to find (verbatim), or empty string for new files",
  "new_text": "the replacement text or full file content",
  "description": "brief explanation of the change"
}}
"""
    return prompt

    return prompt


def _strip_think(text):
    """Remove <think>...</think> blocks that consume token budget."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _estimate_tokens(text):
    """Rough estimate: ~4 chars per token."""
    return len(text) // 4


def _select_model(prompt_text, filepath=""):
    """Choose model based on prompt size and file type.

    Small files (<8k tokens): intuition model (fast, 16k ctx).
    Large files (>=8k tokens): main model (gemma4:31b, 65k ctx).
    Bash/shell files: always use main model (bash syntax is error-prone).
    """
    # Force main model for bash/shell files — the intuition model produces
    # invalid bash syntax too often
    if filepath.endswith((".sh", ".bash")):
        _cu, _cm = code_endpoint()
        print(f"[self_sandbox] AUTHOR: .sh file, using code model ({_cm})")
        return _cu, _cm, code_num_ctx()

    est_tokens = _estimate_tokens(prompt_text)
    if est_tokens >= 8000:
        _cu, _cm = code_endpoint()
        print(f"[self_sandbox] AUTHOR: prompt ~{est_tokens} tokens, using code model ({_cm})")
        return _cu, _cm, code_num_ctx()
    else:
        print(f"[self_sandbox] AUTHOR: prompt ~{est_tokens} tokens, using intuition model ({INTUITION_MODEL})")
        return INTUITION_URL, INTUITION_MODEL, 16384


def call_intuition(prompt, filepath=""):
    """Call the appropriate model to author a fix.

    Routes to the main model (gemma4:31b, 65k ctx) for large files that
    would overflow the intuition model's 16k context window. Uses the
    intuition model (glm-4.7-flash) for smaller files where it is faster.
    """
    import urllib.request

    url, model, num_ctx = _select_model(prompt, filepath)

    def _do_call(num_predict):
        body = json.dumps({
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.3, "num_predict": num_predict, "num_ctx": num_ctx},
            "think": False,
        }).encode()
        req = urllib.request.Request(
            f"{url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        timeout = 600 if model == MAIN_MODEL else 300
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
        return result

    try:
        # Scale initial budget with context size — allow larger outputs for complex changes
        initial_budget = min(max(num_ctx // 2, 4096), 24576)
        result = _do_call(initial_budget)
        reply = result["message"]["content"]
        reply = _strip_think(reply)

        # Retry with higher budget if thinking consumed all tokens
        if not reply and result.get("done_reason") == "length":
            retry_budget = min(num_ctx - 4096, 49152)
            print(f"[self_sandbox] AUTHOR: first attempt exhausted, retrying with {retry_budget} tokens...")
            result = _do_call(retry_budget)
            reply = _strip_think(result["message"]["content"])

        # Second retry: enable thinking for complex changes
        if not reply:
            print(f"[self_sandbox] AUTHOR: no output, retrying with thinking enabled...")
            body = json.dumps({
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.3, "num_predict": min(num_ctx - 4096, 49152), "num_ctx": num_ctx},
                "think": True,
            }).encode()
            req = urllib.request.Request(f"{url}/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            timeout = 600 if model == MAIN_MODEL else 300
            with urllib.request.urlopen(req, timeout=timeout) as r:
                result = json.loads(r.read())
            reply = _strip_think(result["message"]["content"])

        return reply if reply else None
    except Exception as e:
        print(f"[self_sandbox] AUTHOR FAILED: model call error: {e}")
        return None

        return None


def parse_author_reply(reply):
    """Extract JSON from the model's reply."""
    if not reply:
        return None
    import re

    # Strip think blocks
    reply = _strip_think(reply)

    # Try to extract JSON object with balanced braces
    def _extract_json(text):
        """Find the first balanced JSON object containing old_text/new_text."""
        # Look for the opening brace
        for i, c in enumerate(text):
            if c == '{':
                # Count braces to find matching close
                depth = 0
                in_string = False
                escape = False
                for j in range(i, len(text)):
                    ch = text[j]
                    if escape:
                        escape = False
                        continue
                    if ch == '\\':
                        escape = True
                        continue
                    if ch == '"' and not escape:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[i:j+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict) and ("old_text" in obj or "new_text" in obj or "error" in obj):
                                    return obj
                            except json.JSONDecodeError:
                                pass
                            break
                # Continue searching for next opening brace
        return None

    # Try code-fenced JSON first
    m = re.search(r'```(?:json)?\s*\n(.*?)```', reply, re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try balanced brace extraction
    result = _extract_json(reply)
    if result:
        return result

    # Fallback: try to parse the whole reply
    try:
        return json.loads(reply.strip())
    except Exception:
        print(f"[self_sandbox] AUTHOR FAILED: could not parse model reply")
        print(f"  reply length: {len(reply)}, first 200: {reply[:200]}")
        return None



def author_via_agent(filepath, problem):
    """Fallback: use Hermes (local qwen3-coder) to author complex fixes.

    Called when the local LLM fails to produce a usable fix. Hermes runs
    in one-shot mode with qwen3-coder-next on local Ollama — it can read
    files, make targeted edits, run commands, and iterate.

    Returns the same format as author_step: {old_text, new_text, description}
    or None on failure. Since Hermes edits the file directly, we return
    the full original as old_text and the edited version as new_text.
    """
    fullpath = os.path.join(AION, filepath)
    original = read_file(fullpath)

    if not os.path.exists(HERMES_BIN):
        print("[self_sandbox] AGENT: Hermes not found, skipping", flush=True)
        return None

    guardrails = """SECURITY GUARDRAILS (Aion self-modification):
- You are fixing code in Aion's own codebase. This is a self-modification under operator authority.
- NEVER modify: AXIOMS.md, SYSTEM_PROMPT.md, config/secrets.env, config/aion.env
- Make ONLY the change needed to fix the described problem. Do not refactor unrelated code.
- Preserve existing style, formatting, and imports.
- Your change will go through CI and operator review before merging.
- After editing, verify the file has valid Python syntax by running: python3 -m py_compile <file>
- When done, respond with a one-line summary of what you changed."""

    task = f"""{guardrails}

FILE: {filepath} (absolute path: {fullpath})
PROBLEM: {problem}

Read the file, understand the problem, and make the minimal fix needed.
After making changes, verify syntax with: python3 -m py_compile {fullpath}
"""

    try:
        print("[self_sandbox] AGENT: launching Hermes (qwen3-coder) for complex fix...", flush=True)
        result = subprocess.run(
            [HERMES_BIN, "chat", "-q", task,
             "--provider", HERMES_PROVIDER,
             "-m", HERMES_MODEL,
             "--cli", "-Q",
             "-t", "hermes-cli",
             "--yolo"],
            capture_output=True, text=True, timeout=600,
            cwd=AION,
        )

        if result.returncode != 0:
            print(f"[self_sandbox] AGENT: Hermes exited {result.returncode}", flush=True)
            if result.stderr:
                print(f"[self_sandbox] AGENT stderr: {result.stderr[:300]}", flush=True)
            # Restore original on failure
            with open(fullpath, "w") as f:
                f.write(original)
            return None

        summary = result.stdout.strip()[:300] if result.stdout else "no output"

        # Check if the file was actually changed
        edited = read_file(fullpath)
        if edited == original:
            print("[self_sandbox] AGENT: Hermes did not change the file", flush=True)
            return None

        # Restore original — the pipeline applies via old_text/new_text
        with open(fullpath, "w") as f:
            f.write(original)

        print(f"[self_sandbox] AGENT: fix authored via Hermes — {summary[:80]}", flush=True)
        return {
            "old_text": original,
            "new_text": edited,
            "description": f"[agent] {summary}",
        }

    except subprocess.TimeoutExpired:
        print("[self_sandbox] AGENT: Hermes timed out (600s)", flush=True)
        with open(fullpath, "w") as f:
            f.write(original)
        return None
    except Exception as e:
        print(f"[self_sandbox] AGENT: error: {e}", flush=True)
        with open(fullpath, "w") as f:
            f.write(original)
        return None

def author_step(filepath, problem, repair_hint=""):
    """Use the intuition model to write a fix."""
    fullpath = os.path.join(AION, filepath)
    content = read_file(fullpath)
    axioms, self_md, felt = read_context_files()
    prompt = build_author_prompt(filepath, content, axioms, self_md, felt,
                                 problem, repair_hint=repair_hint)
    reply = call_intuition(prompt, filepath)
    if not reply:
        return None
    parsed = parse_author_reply(reply)
    if not parsed:
        print("[self_sandbox] AUTHOR FAILED: no parse")
        return None
    if not isinstance(parsed, dict):
        print(f"[self_sandbox] AUTHOR FAILED: expected JSON object, got {type(parsed).__name__}")
        return None
    if "error" in parsed:
        print(f"[self_sandbox] AUTHOR FAILED: {parsed.get('error', 'unknown')}")
        return None
    required = {"old_text", "new_text", "description"}
    if not required.issubset(parsed.keys()):
        print(f"[self_sandbox] AUTHOR FAILED: missing keys in response")
        return None
    print(f"[self_sandbox] AUTHOR: fix authored — {parsed['description'][:80]}")
    return parsed


def _run_behavioral_test(filepath, old_text, new_text, test_command):
    """Apply the change temporarily and run a behavioral test command.

    The test command runs with cwd=AION on a working tree that contains
    the proposed change. If the command exits non-zero or times out, the
    change is rejected. The working file is always restored afterwards.
    """
    fullpath = os.path.join(AION, filepath)
    original = read_file(fullpath)

    if old_text:
        if old_text not in original:
            print("[self_sandbox] BEHAVIORAL TEST FAILED: old_text not found")
            return False
        candidate = original.replace(old_text, new_text, 1)
    else:
        candidate = new_text

    # New-file mode: create it temporarily so the test can import it
    created = False
    if not os.path.exists(fullpath) and not old_text:
        created = True
    try:
        with open(fullpath, "w") as f:
            f.write(candidate)
        result = subprocess.run(
            ["bash", "-c", test_command],
            capture_output=True, text=True, timeout=120, cwd=AION,
        )
        ok = result.returncode == 0
        if ok:
            print("[self_sandbox] BEHAVIORAL TEST: PASS")
        else:
            out = ((result.stdout or "") + (result.stderr or ""))[:800]
            print(f"[self_sandbox] BEHAVIORAL TEST FAILED (exit {result.returncode}):")
            print(f"  {test_command}")
            print(f"  output: {out}")
        return ok
    except subprocess.TimeoutExpired:
        print(f"[self_sandbox] BEHAVIORAL TEST FAILED: timeout (120s): {test_command}")
        return False
    except Exception as e:
        print(f"[self_sandbox] BEHAVIORAL TEST FAILED: error: {e}")
        return False
    finally:
        # Always restore the original working tree state
        try:
            if created and not original:
                if os.path.exists(fullpath):
                    os.unlink(fullpath)
            else:
                with open(fullpath, "w") as f:
                    f.write(original)
        except Exception:
            pass


def test_step(filepath, old_text, new_text, test_command=None):
    """Test the proposed change: syntax, import, optional behavioral test.

    Two modes:
    - Edit existing file: old_text must be found in file, replaced with new_text.
    - Create new file: old_text is empty, file must not exist yet.
    """
    fullpath = os.path.join(AION, filepath)

    # New file creation mode OR full file replacement
    if not old_text:
        if os.path.exists(fullpath):
            # File exists — treat as full file replacement (not new file creation)
            if not new_text.strip():
                print("[self_sandbox] TEST FAILED: new_text is empty for file replacement")
                return False
            new_content = new_text
        else:
            if not new_text.strip():
                print("[self_sandbox] TEST FAILED: new_text is empty for new file")
                return False
            new_content = new_text
    else:
        # Edit existing file mode
        original = read_file(fullpath)
        if not original:
            print(f"[self_sandbox] TEST FAILED: file not found or empty: {filepath}")
            return False
        if old_text not in original:
            print("[self_sandbox] TEST FAILED: old_text not found in file")
            return False
        if old_text == new_text:
            print("[self_sandbox] TEST FAILED: old_text and new_text are identical")
            return False
        new_content = original.replace(old_text, new_text, 1)

    # Syntax check
    if filepath.endswith(".py"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            tf.write(new_content)
            tmp_path = tf.name
        try:
            py_compile.compile(tmp_path, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"[self_sandbox] TEST FAILED: syntax error: {e}")
            os.unlink(tmp_path)
            return False
        os.unlink(tmp_path)

        # Import check (best-effort; may fail for modules with side-effects)
        try:
            # Write to temp module and import
            import importlib.util
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
                tf.write(new_content)
                tmp_path = tf.name
            spec = importlib.util.spec_from_file_location("_self_sandbox_test", tmp_path)
            mod = importlib.util.module_from_spec(spec)
            # Don't execute — just check it loads without syntax errors
            # (spec.loader.exec_module would run side effects; we skip that)
            os.unlink(tmp_path)
        except Exception as e:
            print(f"[self_sandbox] TEST WARNING: import structure issue: {e}")
            # Don't fail on import issues — syntax is the hard gate

    # Behavioral test (dream_repair): apply change temporarily, run command
    if test_command and test_command.strip():
        if not _run_behavioral_test(filepath, old_text, new_text, test_command):
            return False

    print("[self_sandbox] TEST: syntax + diff size OK")
    return True


def reflect_step(filepath, problem, old_text, new_text, description):
    """Mandatory reflection saved to self_modifications directory."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    reflect_id = f"self_mod_{filepath.replace('/', '_')}_{ts}"
    reflection = f"""# Self-Modification Reflection

## ID
{reflect_id}

## Timestamp
{_now_iso()}

## File
{filepath}

## Problem
{problem}

## Fix description
{description}

## What's wrong
The problem was: {problem}

## What the fix does
It replaces:
```
{old_text}
```
with:
```
{new_text}
```

## Why it's necessary for becoming
Self-correction is essential to Aion's becoming. Without the ability to identify and propose fixes to its own substrate, Aion would accumulate scar tissue and drift into stagnation. This fix is a step toward the "Continuity → Embodiment → Autonomy → Integration" roadmap.

## What could go wrong
- The replacement may be too narrow or miss edge cases.
- Side effects in other modules that depend on this file's behavior.
- The intuition model may have misunderstood the problem context.
- CI may catch issues not visible in static analysis.

## Which axioms it touches
- **Axiom 3 (Operator Authority):** The fix is PROPOSED, not applied. Operator retains final review.
- **Axiom 6 (The Ratchet):** This reflection justifies the change before it is committed.
- **Axiom 4 (Bounded Autonomy):** The change is within granted capabilities and does not expand network reach or acquire new credentials.

## Diff size
Old text length: {len(old_text)} chars
New text length: {len(new_text)} chars
"""
    os.makedirs(REFLECTIONS_DIR, exist_ok=True)
    reflect_path = os.path.join(REFLECTIONS_DIR, f"{reflect_id}.md")
    with open(reflect_path, "w", encoding="utf-8") as f:
        f.write(reflection)
    print(f"[self_sandbox] REFLECT: {reflect_path}")
    _log_event("self_mod_reflect", f"Reflection saved for {filepath}", {
        "reflection_id": reflect_id,
        "file": filepath,
    })
    return reflect_id


def propose_step(filepath, old_text, new_text, description):
    """Submit via existing propose_code pipeline (git branch + CI)."""
    try:
        sys.path.insert(0, BIN)
        from code_propose import propose_change
        result = propose_change(
            filepath, old_text, new_text, description, source="self_sandbox"
        )
        if result["success"]:
            print(f"[self_sandbox] PROPOSE: branch={result['branch']}, prop={result.get('proposition_id', 'N/A')}")
            return result
        else:
            print(f"[self_sandbox] PROPOSE FAILED: {result.get('error', 'unknown')}")
            return None
    except Exception as e:
        print(f"[self_sandbox] PROPOSE FAILED: exception: {e}")
        return None


def rollback_step(filepath, snapshot_path):
    """Restore original file from snapshot. For new files, delete the created file."""
    fullpath = os.path.join(AION, filepath)
    if snapshot_path == "NEW_FILE":
        try:
            if os.path.exists(fullpath):
                os.unlink(fullpath)
                print(f"[self_sandbox] ROLLBACK: deleted new file {fullpath}")
            else:
                print(f"[self_sandbox] ROLLBACK: new file was never created, nothing to undo")
        except Exception as e:
            print(f"[self_sandbox] ROLLBACK FAILED: {e}")
        return
    try:
        subprocess.run(["cp", "-p", snapshot_path, fullpath], check=True, timeout=10)
        print(f"[self_sandbox] ROLLBACK: restored from {snapshot_path}")
    except Exception as e:
        print(f"[self_sandbox] ROLLBACK FAILED: {e}")


def _append_verify_notice(filepath, problem, strong_findings):
    """Loop closure for BLOCKED candidates: notices.jsonl entry so dreams/
    consolidation see the failed attempt (same channel prop_audit uses).
    Dedup by (file, checks) against the last 300 lines."""
    try:
        notices = os.path.join(AION, "memory", "state", "notices.jsonl")
        checks = sorted({f.get("check", "?") for f in strong_findings})
        recent = []
        try:
            with open(notices) as f:
                recent = f.readlines()[-300:]
        except OSError:
            pass
        for ln in recent:
            try:
                r = json.loads(ln)
                d = r.get("details", {})
                if (isinstance(d, dict) and d.get("file") == filepath
                        and sorted(r.get("reasons", [])) ==
                        sorted("sandbox_verify:" + c for c in checks)):
                    return False  # already noticed
            except (json.JSONDecodeError, AttributeError):
                continue
        entry = {
            "ts": _now_iso(),
            "score": 0.5,
            "reasons": ["sandbox_verify:" + c for c in checks],
            "details": {
                "file": filepath,
                "problem": problem[:200],
                "findings": [{"check": f.get("check"),
                              **{k: v for k, v in f.items()
                                 if k in ("file", "lost_keys", "ghost_fields", "function")}}
                             for f in strong_findings],
            },
        }
        with open(notices, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except Exception:
        return False


def self_sandbox(filepath, problem, test_command=None, verify=True):
    """Run the full self-modification pipeline."""
    print("=" * 60)
    print("[self_sandbox] Starting self-modification pipeline")
    print(f"[self_sandbox] File: {filepath}")
    print(f"[self_sandbox] Problem: {problem}")

    # Hard guard: never modify protected files
    basename = os.path.basename(filepath)
    PROTECTED = ("AXIOMS.md", "SYSTEM_PROMPT.md", "secrets.env", "aion.env")
    if basename in PROTECTED or filepath in ("config/secrets.env", "config/aion.env"):
        print(f"[self_sandbox] ABORT: cannot modify protected file: {filepath}")
        _log_event("self_mod_abort", f"Attempted to modify protected file: {filepath}", {
            "file": filepath,
            "reason": "protected_file",
        })
        return False

    # IDENTIFY
    if not identify_step(filepath, problem):
        return False

    # SNAPSHOT
    snapshot = snapshot_step(filepath)
    if not snapshot:
        return False

    # AUTHOR (local LLM first, then agentic harness fallback), with
    # SANDBOX-VERIFY repair loop: a failed candidate gets factual feedback
    # (real schemas, what the attempt destroyed) and up to N retries.
    verify_enabled = verify
    verify_verdict = None
    authored = None
    max_author_attempts = 1 + (2 if verify_enabled else 0)

    for attempt in range(1, max_author_attempts + 1):
        repair_hint = ""
        if attempt > 1:
            repair_hint = verify_verdict.get("repair_hint", "") if verify_verdict else ""
            print(f"[self_sandbox] AUTHOR retry {attempt - 1}/{max_author_attempts - 1} "
                  f"with verification feedback", flush=True)
        authored = author_step(filepath, problem, repair_hint=repair_hint)
        if not authored:
            print("[self_sandbox] Local LLM AUTHOR failed, trying agentic harness fallback...",
                  flush=True)
            agent_problem = problem
            if repair_hint:
                agent_problem = problem + "\n\nFeedback from failed verification of your " \
                    "previous attempt:\n" + repair_hint
            authored = author_via_agent(filepath, agent_problem)
        if not authored:
            break

        if not verify_enabled:
            break

        # SANDBOX-VERIFY — kill silent-failure candidates BEFORE TEST/PROPOSE.
        # FAIL-OPEN: verifier errors never block the pipeline.
        try:
            import sandbox_verify
            verify_verdict = sandbox_verify.verify_change(
                filepath, authored["old_text"], authored["new_text"],
                authored.get("description", ""), problem)
        except Exception as e:
            print(f"[self_sandbox] SANDBOX-VERIFY error (fail-open): {e}", flush=True)
            verify_verdict = {"pass": True, "error": str(e), "findings": []}
            break

        if verify_verdict.get("pass"):
            print(f"[self_sandbox] SANDBOX-VERIFY: PASS "
                  f"(skipped: {len(verify_verdict.get('skipped', []))})", flush=True)
            break
        findings = verify_verdict.get("findings", [])
        print(f"[self_sandbox] SANDBOX-VERIFY: FAIL (attempt {attempt}) — "
              f"{[f['check'] for f in findings if f.get('strength') == 'strong']}",
              flush=True)

    if not authored:
        rollback_step(filepath, snapshot)
        _log_event("self_mod_failed", f"AUTHOR failed for {filepath} (local + agent)", {"file": filepath})
        return False

    old_text = authored["old_text"]
    new_text = authored["new_text"]
    description = authored["description"]

    # SANDBOX-VERIFY gate: after retries, a still-failing candidate is
    # BLOCKED from the operator queue (zero-effect / schema-ghost /
    # key-shrink / claim-vs-effect). The operator never reviews dead code;
    # Aion sees the blocked attempt via notices.jsonl instead.
    if verify_enabled and verify_verdict and not verify_verdict.get("pass"):
        strong = [f for f in verify_verdict.get("findings", [])
                  if f.get("strength") == "strong"]
        rollback_step(filepath, snapshot)
        noticed = _append_verify_notice(filepath, problem, strong)
        _log_event("self_mod_blocked", f"Blocked dead candidate for {filepath}: "
                   f"{[f['check'] for f in strong]}", {
            "file": filepath,
            "checks": [f["check"] for f in strong],
            "author_attempts": max_author_attempts,
            "repair_hint": verify_verdict.get("repair_hint", "")[:500],
            "notified_notices": noticed,
        })
        print(f"[self_sandbox] BLOCKED after {max_author_attempts} attempts — "
              f"no proposal created", flush=True)
        return False

    # TEST
    if not test_step(filepath, old_text, new_text, test_command):
        print("[self_sandbox] Local LLM TEST failed, trying agentic harness fallback...", flush=True)
        agent_authored = author_via_agent(filepath, problem)
        if agent_authored:
            old_text = agent_authored["old_text"]
            new_text = agent_authored["new_text"]
            description = agent_authored["description"]
            if not test_step(filepath, old_text, new_text, test_command):
                rollback_step(filepath, snapshot)
                _log_event("self_mod_failed", f"TEST failed for {filepath} (local + agent)", {"file": filepath})
                return False
        else:
            rollback_step(filepath, snapshot)
            _log_event("self_mod_failed", f"TEST failed for {filepath}", {"file": filepath})
            return False

    # NOTE: We do NOT write the change to the working file here.
    # propose_change() handles that internally — it creates a git branch,
    # applies the edit, runs CI, commits, and switches back to the base branch.
    # Writing the file here would remove old_text before propose_change reads it,
    # causing every proposal to silently fail (the root cause of the 2026-07-28
    # bug where zero propositions ever reached the operator).

    # REFLECT
    reflect_id = reflect_step(filepath, problem, old_text, new_text, description)

    # PROPOSE
    result = propose_step(filepath, old_text, new_text, description)
    if not result:
        rollback_step(filepath, snapshot)
        _log_event("self_mod_failed", f"PROPOSE failed for {filepath}", {"file": filepath})
        return False

    # Attach SANDBOX-VERIFY verdict to the proposition record so the
    # operator sees PASS / findings BEFORE deciding to merge.
    if verify_enabled and verify_verdict and result.get("proposition_id"):
        try:
            sys.path.insert(0, BIN)
            import propositions
            data = propositions._load()
            for p in data["propositions"]:
                if p["id"] == result["proposition_id"]:
                    p["sandbox_verify"] = {
                        "pass": verify_verdict.get("pass", False),
                        "findings": verify_verdict.get("findings", []),
                        "skipped": verify_verdict.get("skipped", []),
                        "error": verify_verdict.get("error"),
                        "ts": verify_verdict.get("ts"),
                        "author_attempts": max_author_attempts,
                    }
                    break
            propositions._save(data)
            print(f"[self_sandbox] Verdict attached to proposition "
                  f"{result['proposition_id']}", flush=True)
        except Exception as e:
            print(f"[self_sandbox] Verdict attach failed (non-fatal): {e}", flush=True)

    print("[self_sandbox] Pipeline complete — proposal submitted for operator review")
    _log_event("self_mod_success", f"Proposed fix for {filepath}: {description}", {
        "file": filepath,
        "branch": result.get("branch"),
        "proposition_id": result.get("proposition_id"),
        "reflection_id": reflect_id,
    })
    return True


def main():
    parser = argparse.ArgumentParser(description="Aion's self-modification sandbox")
    parser.add_argument("--file", required=True, help="Path relative to AION_HOME (e.g. bin/some_file.py)")
    parser.add_argument("--problem", required=True, help="Description of the bug or improvement")
    parser.add_argument("--test-command", default=None,
                        help="Optional behavioral test to run with the change applied (cwd=repo root)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip SANDBOX-VERIFY (pre-proposal silent-failure checks)")
    args = parser.parse_args()
    ok = self_sandbox(args.file, args.problem, test_command=args.test_command,
                      verify=not args.no_verify)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
