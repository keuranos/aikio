#!/usr/bin/env python3
"""code_propose.py — Aion's Tier 1 code-writing workflow.

Lets Aion write code changes to a git branch, run CI validation,
and create a proposition for the operator to review/merge.

Flow:
  1. Aion (via wake/idle/curiosity) calls propose_change() with a file path,
     old text, new text, and description
  2. This creates a git branch (aion/code-{ts})
  3. Applies the change
  4. Runs ci_check.py
  5. If CI passes: commits the change, creates a proposition
  6. If CI fails: reverts, logs the failure, no proposition

State: propositions.json (proposition with source="code_proposal")
"""
import json
import os
import sys
import subprocess
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import aion_env

AION = os.environ.get("AION_HOME", "$AION_HOME")
BIN = f"{AION}/bin"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ts_short():
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _log_event(event_type, text, meta=None):
    """Log to episodic memory."""
    try:
        cmd = [sys.executable, f"{AION}/bin/log_event.py",
               "--type", event_type, "--text", text]
        if meta:
            cmd.extend(["--meta", json.dumps(meta)])
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        pass


def propose_change(filepath, old_text, new_text, description,
                    source="self_wake", commit_message=None):
    """Propose a code change to a git branch with CI validation.

    Args:
        filepath: Path relative to AION_HOME (e.g. "bin/dream_v2.py")
        old_text: The text to find in the file
        new_text: The replacement text
        description: Human-readable description of the change
        source: Where this proposal came from (self_wake, idle, curiosity)
        commit_message: Optional custom commit message

    Returns:
        dict: Result with keys:
            - success: bool
            - branch: str or None
            - ci_passed: bool or None
            - ci_results: list
            - proposition_id: str or None
            - error: str or None
    """
    fullpath = os.path.join(AION, filepath)
    result = {
        "success": False,
        "branch": None,
        "ci_passed": None,
        "ci_results": [],
        "proposition_id": None,
        "error": None,
    }

    # Two modes: edit existing file, or create new file
    is_new_file = not old_text and not os.path.exists(fullpath)

    if not is_new_file:
        # Edit existing file: check file exists and old_text is present
        if not os.path.exists(fullpath):
            result["error"] = f"file not found: {filepath}"
            return result
        content = open(fullpath).read()
        if old_text not in content:
            result["error"] = f"old_text not found in {filepath}"
            return result
        if old_text == new_text:
            result["error"] = "old_text and new_text are identical"
            return result
    else:
        content = ""

    # Create a git branch
    branch_name = f"aion/code-{_ts_short()}"
    base_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, cwd=AION, timeout=10,
    ).stdout.strip()

    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            capture_output=True, text=True, cwd=AION, timeout=30,
        )
        result["branch"] = branch_name
    except Exception as e:
        result["error"] = f"failed to create branch: {e}"
        return result

    # Apply the change
    try:
        if is_new_file:
            os.makedirs(os.path.dirname(fullpath), exist_ok=True)
            new_content = new_text
        else:
            new_content = content.replace(old_text, new_text, 1)
        with open(fullpath, "w") as f:
            f.write(new_content)
    except Exception as e:
        result["error"] = f"failed to write file: {e}"
        _revert_to_base(branch_name, base_branch, filepath if is_new_file else None)
        return result

    # Run CI check
    try:
        ci_result = subprocess.run(
            [sys.executable, f"{BIN}/ci_check.py", "--base", base_branch],
            capture_output=True, text=True, cwd=AION, timeout=120,
        )
        result["ci_passed"] = (ci_result.returncode == 0)
        result["ci_output"] = ci_result.stdout + ci_result.stderr
        print(ci_result.stdout)
        if ci_result.stderr:
            print(ci_result.stderr)
    except Exception as e:
        result["ci_passed"] = False
        result["ci_output"] = str(e)

    if not result["ci_passed"]:
        # CI failed — revert and log
        _log_event("code_proposal_ci_failed", f"CI failed for {filepath}: {description[:100]}", {
            "branch": branch_name,
            "file": filepath,
            "description": description[:200],
            "ci_output": result.get("ci_output", "")[:500],
        })
        _revert_to_base(branch_name, base_branch, filepath if is_new_file else None)
        result["error"] = "CI check failed"
        return result

    # CI passed — commit the change
    msg = commit_message or f"aion: {description[:100]}"
    try:
        subprocess.run(["git", "add", filepath], cwd=AION, capture_output=True, timeout=10)
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=AION, capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        result["error"] = f"failed to commit: {e}"
        _revert_to_base(branch_name, base_branch, filepath if is_new_file else None)
        return result

    # ── SEMANTIC REVIEW (peer layer checks the diff against ground truth) ──
    # The authoring model wrote it; a different-family model verifies it
    # against a facts sheet of production reality. REJECT = the branch is
    # reverted and the reasons return as repair feedback — the proposal
    # never reaches the operator's dashboard broken.
    review_data = None
    try:
        from code_review import review_change
        review_data = review_change(filepath, old_text, new_text, description)
        result["review_verdict"] = review_data["verdict"]
        result["review"] = review_data.get("review", "")[:2000]
        result["coherence"] = review_data.get("coherence", "")
        if review_data["verdict"] == "REJECT":
            _revert_to_base(branch_name, base_branch,
                            filepath if is_new_file else None)
            _log_event("code_proposal_review_rejected",
                       f"peer review rejected {filepath}: "
                       f"{review_data.get('review', '')[:300]}", {
                           "branch": branch_name,
                           "file": filepath,
                           "review": review_data.get("review", "")[:800],
                       })
            result["error"] = (
                "PEER REVIEW REJECTED this change. Repair and re-propose in "
                "this cycle. Reviewer reasons:\n%s"
                % review_data.get("review", "")[:1200])
            return result
    except Exception as e:
        # fail-open: review infrastructure broken, don't block proposals
        review_data = {"verdict": "review_failed", "review": str(e)}
        result["review_verdict"] = "review_failed"

    # Create a proposition for the operator
    try:
        sys.path.insert(0, BIN)
        import propositions
        prop = propositions.create(
            text=f"Code change: {description}",
            prop_type="change_proposal",
            source=source,
            source_id=branch_name,
            priority="medium",
            notify=False,  # code proposals don't need WhatsApp
        )
        result["proposition_id"] = prop["id"]
        # Mark the proposition with the branch info + review outcome
        prop["git_branch"] = branch_name
        prop["file"] = filepath
        prop["description"] = description
        prop["review_verdict"] = review_data.get("verdict") if review_data else None
        if review_data and review_data.get("review"):
            prop["peer_review"] = str(review_data.get("review", ""))[:1500]
        if review_data and review_data.get("coherence"):
            prop["coherence_check"] = str(review_data.get("coherence", ""))[:400]
        _save_prop_update(prop)
    except Exception as e:
        # Proposition creation failed but code is committed
        result["error"] = f"code committed but proposition failed: {e}"

    # Switch back to base branch
    subprocess.run(
        ["git", "checkout", base_branch],
        cwd=AION, capture_output=True, timeout=30,
    )

    result["success"] = True
    _log_event("code_proposal_created", f"proposed: {description[:100]}", {
        "branch": branch_name,
        "file": filepath,
        "proposition_id": result.get("proposition_id"),
        "ci_passed": True,
    })

    return result


def _save_prop_update(prop):
    """Update a proposition in the state file."""
    try:
        sys.path.insert(0, BIN)
        import propositions
        data = propositions._load()
        for i, p in enumerate(data["propositions"]):
            if p["id"] == prop["id"]:
                data["propositions"][i] = prop
                break
        propositions._save(data)
    except Exception:
        pass


def _revert_to_base(branch_name, base_branch, new_file_path=None):
    """Revert to base branch and delete the proposed branch.
    If new_file_path is set, also remove the untracked new file."""
    try:
        subprocess.run(
            ["git", "checkout", base_branch],
            cwd=AION, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=AION, capture_output=True, timeout=10,
        )
        if new_file_path:
            fullpath = os.path.join(AION, new_file_path)
            if os.path.exists(fullpath):
                os.unlink(fullpath)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Aion Tier 1 code proposal system")
    sub = parser.add_subparsers(dest="command")

    # propose — write a code change to a branch
    p_propose = sub.add_parser("propose", help="Propose a code change")
    p_propose.add_argument("filepath", help="Path relative to AION_HOME (e.g. bin/dream_v2.py)")
    p_propose.add_argument("--old-file", help="File containing the old text to find")
    p_propose.add_argument("--new-file", help="File containing the new text to replace with")
    p_propose.add_argument("--description", required=True, help="Description of the change")
    p_propose.add_argument("--source", default="manual")
    p_propose.add_argument("--commit-message", default=None)

    # ci — run CI on current branch
    p_ci = sub.add_parser("ci", help="Run CI checks on current branch")
    p_ci.add_argument("--base", default="master")

    # list — list code proposals
    sub.add_parser("list", help="List code proposals")

    args = parser.parse_args()

    if args.command == "propose":
        old_text = open(args.old_file).read() if args.old_file else ""
        new_text = open(args.new_file).read() if args.new_file else ""
        if not new_text:
            print("Error: --new-file required")
            sys.exit(1)
        # old_text may be empty for new file creation
        result = propose_change(args.filepath, old_text, new_text,
                                args.description, args.source, args.commit_message)
        if result["success"]:
            print(f"SUCCESS: branch={result['branch']}")
            print(f"  CI: PASSED")
            print(f"  Proposition: {result.get('proposition_id', 'N/A')}")
        else:
            print(f"FAILED: {result.get('error', 'unknown')}")
            if result.get("ci_passed") is False:
                print(f"  CI output:\n{result.get('ci_output', '')}")
        sys.exit(0 if result["success"] else 1)

    elif args.command == "ci":
        passed, results = run_ci(base=args.base)
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['check']}: {r.get('msg','')}")
        sys.exit(0 if passed else 1)

    elif args.command == "list":
        sys.path.insert(0, BIN)
        import propositions
        props = propositions.list_props(prop_type="change_proposal", limit=20)
        for p in props:
            branch = p.get("git_branch", "N/A")
            print(f"{p['id']} | {p['status']:10} | {branch:30} | {p['text'][:60]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()