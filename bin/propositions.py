#!/usr/bin/env python3
"""propositions.py — Aion's direct channel to the operator.

Aion creates propositions (questions, change proposals, axiom proposals)
that surface directly to the operator via:
  1. WhatsApp notifications (high+ priority)
  2. Dashboard panel (browse, answer, accept, reject)
  3. Camera monitor operator chat (present during sessions)

Propositions flow:
  dream/curiosity/consolidation → create() → notify if high+
  operator → answer()/accept()/reject() via CLI or dashboard
  accepted → Tier 1: git branch created for implementation

Lifecycle:
  open → answered (questions) | accepted/rejected (proposals) → implemented

State: memory/state/propositions.json
Log: episodic memory (type: proposition)
"""
import json
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import aion_env  # loads config/aion.env

AION = os.environ.get("AION_HOME", "$AION_HOME")
STATE = f"{AION}/memory/state"
PROP_FILE = f"{STATE}/propositions.json"
RATE_FILE = f"{STATE}/proposition_rate.json"

# Rate limits: max 1 WhatsApp per 30 min, max 5 per day
RATE_WINDOW_NOTIFY = 1800  # 30 minutes
RATE_DAILY_NOTIFY = 5
RATE_DAILY_WINDOW = 86400  # 24 hours


def _now():
    return datetime.now(timezone.utc).isoformat()


def _prop_id():
    """Generate a unique proposition ID."""
    ts = datetime.now(timezone.utc)
    return f"prop_{ts.strftime('%Y%m%d%H%M%S')}{int(ts.microsecond / 1000):03d}"


def _load():
    """Load propositions state."""
    try:
        return json.load(open(PROP_FILE))
    except Exception:
        return {"propositions": [], "ts": _now()}


def _save(data):
    """Save propositions state."""
    os.makedirs(STATE, exist_ok=True)
    data["ts"] = _now()
    with open(PROP_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _log_event(event_type, text, meta=None):
    """Log to episodic memory via log_event.py."""
    try:
        rec = {
            "ts": _now(),
            "type": event_type,
            "text": text,
        }
        if meta:
            rec["meta"] = meta
        cmd = [
            sys.executable,
            f"{AION}/bin/log_event.py",
            "--type",
            event_type,
            "--text",
            text,
        ]
        if meta:
            for k, v in meta.items():
                cmd.extend([f"--meta-{k}", str(v)])
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        pass  # logging is best-effort


def _check_notify_rate():
    """Check if we can send another WhatsApp notification. Returns True if OK."""
    try:
        rates = json.load(open(RATE_FILE))
    except Exception:
        rates = {"notify_times": []}

    now = time.time()
    # Clean old entries
    rates["notify_times"] = [
        t for t in rates["notify_times"] if now - t < RATE_DAILY_WINDOW
    ]

    # Check daily limit
    if len(rates["notify_times"]) >= RATE_DAILY_NOTIFY:
        return False

    # Check per-30-min limit
    recent = [t for t in rates["notify_times"] if now - t < RATE_WINDOW_NOTIFY]
    if len(recent) >= 1:
        return False

    rates["notify_times"].append(now)
    os.makedirs(STATE, exist_ok=True)
    with open(RATE_FILE, "w") as f:
        json.dump(rates, f)
    return True


def _send_whatsapp(message):
    """Send a WhatsApp message via Hermes gateway."""
    try:
        result = subprocess.run(
            ["hermes", "send", "--to", "whatsapp:the operator", "--quiet", message],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def _notify_operator(prop):
    """Send WhatsApp notification for a high-priority proposition."""
    if not _check_notify_rate():
        return False, "rate_limited"

    type_label = {
        "question": "Question for you",
        "change_proposal": "Change proposal",
        "axiom_proposal": "Axiom proposal",
    }.get(prop["type"], "Proposition")

    priority_label = {
        "critical": "🔴 CRITICAL",
        "high": "🟠 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🟢 LOW",
    }.get(prop["priority"], prop["priority"])

    msg = (
        f"Aion {type_label}\n"
        f"Priority: {priority_label}\n"
        f"Source: {prop.get('source', 'unknown')}\n\n"
        f"{prop['text']}\n\n"
        f"Respond: hermes send --to whatsapp:Aion "
        f"\"answer {prop['id']} <your answer>\""
        f"\nor use the dashboard."
    )

    ok = _send_whatsapp(msg)
    return ok, "whatsapp" if ok else "failed"


def create(text, prop_type, source, source_id=None, priority="medium",
            notify=False):
    """Create a new proposition.

    Args:
        text: The proposition text
        prop_type: question | change_proposal | axiom_proposal
        source: dream | curiosity | consolidation | self_wake | operator_chat
        source_id: ID of the source (e.g. dream filename, curiosity question ID)
        priority: low | medium | high | critical
        notify: If True, send WhatsApp for high+ priority

    Returns:
        dict: The created proposition
    """
    data = _load()

    # Dedup: don't create if an open proposition with same text exists
    for p in data["propositions"]:
        if (p["text"] == text and p["status"] == "open"
                and p["type"] == prop_type):
            return p  # already exists

    prop = {
        "id": _prop_id(),
        "type": prop_type,
        "text": text,
        "source": source,
        "source_id": source_id,
        "priority": priority,
        "status": "open",
        "answer": None,
        "answer_ts": None,
        "ts": _now(),
        "notified": False,
        "notified_ts": None,
        "git_branch": None,
    }

    data["propositions"].append(prop)
    _save(data)

    # Log to episodic
    _log_event("proposition", f"proposition created: {prop_type}: {text[:100]}", {
        "prop_id": prop["id"],
        "prop_type": prop_type,
        "source": source,
        "priority": priority,
    })

    # Send WhatsApp for high+ priority
    if notify and priority in ("high", "critical"):
        ok, transport = _notify_operator(prop)
        if ok:
            prop["notified"] = True
            prop["notified_ts"] = _now()
            _save(data)

    return prop


def list_props(status=None, prop_type=None, limit=50):
    """List propositions, optionally filtered."""
    data = _load()
    props = data["propositions"]
    if status:
        props = [p for p in props if p["status"] == status]
    if prop_type:
        props = [p for p in props if p["type"] == prop_type]
    # Newest first
    props = sorted(props, key=lambda p: p["ts"], reverse=True)
    return props[:limit]


def get_pending():
    """Get all open propositions for dashboard/monitor display."""
    data = _load()
    return [p for p in data["propositions"] if p["status"] == "open"]


def get(prop_id):
    """Get a single proposition by ID."""
    data = _load()
    for p in data["propositions"]:
        if p["id"] == prop_id:
            return p
    return None


def _find_by_substring(text):
    """Find an open proposition matching a substring (case-insensitive)."""
    data = _load()
    text_lower = text.lower()
    for p in data["propositions"]:
        if p["status"] == "open" and text_lower in p["text"].lower():
            return p
    return None


def answer(prop_id, answer_text):
    """Answer a question proposition.

    Args:
        prop_id: Proposition ID or substring match
        answer_text: The operator's answer

    Returns:
        dict: The updated proposition, or None if not found
    """
    data = _load()

    # Try exact ID first, then substring
    prop = None
    for p in data["propositions"]:
        if p["id"] == prop_id and p["status"] == "open":
            prop = p
            break
    if not prop:
        prop = _find_by_substring(prop_id)
        if prop:
            # Re-find in data to get reference
            for p in data["propositions"]:
                if p["id"] == prop["id"]:
                    prop = p
                    break

    if not prop:
        return None

    prop["status"] = "answered"
    prop["answer"] = answer_text
    prop["answer_ts"] = _now()
    _save(data)

    _log_event("proposition_answered", f"operator answered: {answer_text[:100]}", {
        "prop_id": prop["id"],
        "question": prop["text"][:100],
        "answer": answer_text[:200],
        "source": "operator",
    })

    return prop


def accept(prop_id, reason=None):
    """Accept a change/axiom proposal. Creates a git branch (Tier 1 autonomy).

    Args:
        prop_id: Proposition ID or substring match
        reason: Optional reason for acceptance

    Returns:
        dict: The updated proposition with git_branch, or None
    """
    data = _load()

    prop = None
    for p in data["propositions"]:
        if p["id"] == prop_id and p["status"] == "open":
            prop = p
            break
    if not prop:
        prop = _find_by_substring(prop_id)
        if prop:
            for p in data["propositions"]:
                if p["id"] == prop["id"]:
                    prop = p
                    break

    if not prop:
        return None

    prop["status"] = "accepted"
    prop["answer"] = reason
    prop["answer_ts"] = _now()

    # For code changes: merge the source branch into master automatically
    if prop.get("type") == "change_proposal":
        source_branch = prop.get("source_id", "")
        merge_ok = False
        merge_error = None
        if source_branch:
            try:
                base_branch = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, timeout=10, cwd=AION,
                ).stdout.strip()

                # Verify the source branch exists
                branch_check = subprocess.run(
                    ["git", "rev-parse", "--verify", source_branch],
                    capture_output=True, text=True, timeout=10, cwd=AION,
                )
                if branch_check.returncode != 0:
                    raise RuntimeError(f"source branch not found: {source_branch}")

                # Merge source branch into current branch (master)
                subprocess.run(
                    ["git", "merge", source_branch, "--no-edit"],
                    capture_output=True, text=True, timeout=60, cwd=AION,
                )

                # Verify CI passes on the merged result
                ci_result = subprocess.run(
                    [sys.executable, f"{AION}/bin/ci_check.py", "--base", base_branch],
                    capture_output=True, text=True, timeout=120, cwd=AION,
                )
                if ci_result.returncode != 0:
                    # CI failed after merge — abort and rollback
                    subprocess.run(
                        ["git", "merge", "--abort"],
                        capture_output=True, text=True, timeout=10, cwd=AION,
                    )
                    raise RuntimeError(f"CI failed after merge: {ci_result.stdout[-500:]}")

                merge_ok = True
                prop["git_branch"] = source_branch
                prop["merged_to"] = base_branch
                # Record the merge commit for rollback
                merge_sha = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=10, cwd=AION,
                ).stdout.strip()
                prop["merge_commit"] = merge_sha
                prop["pre_merge_commit"] = subprocess.run(
                    ["git", "rev-parse", base_branch + "~1"],
                    capture_output=True, text=True, timeout=10, cwd=AION,
                ).stdout.strip()

            except Exception as e:
                merge_error = str(e)
                prop["git_branch"] = source_branch
                prop["merge_error"] = merge_error
        else:
            merge_error = "no source_id (git branch) on proposition"
            prop["merge_error"] = merge_error

        _save(data)

        if merge_ok:
            _log_event("proposition_accepted", f"operator accepted + merged: {prop['text'][:100]}", {
                "prop_id": prop["id"],
                "proposition": prop["text"][:100],
                "reason": reason or "",
                "git_branch": source_branch,
                "merged_to": prop.get("merged_to"),
                "source": "operator",
            })
        else:
            _log_event("proposition_accept_failed", f"accept ok but merge failed: {merge_error}", {
                "prop_id": prop["id"],
                "error": merge_error,
            })

        return prop

def acknowledge(prop_id):
    """Acknowledge an observation or insight. No git branch, just marks it seen.

    Args:
        prop_id: Proposition ID or substring match

    Returns:
        dict: The updated proposition, or None
    """
    data = _load()
    prop = None
    for p in data["propositions"]:
        if p["id"] == prop_id and p["status"] == "open":
            prop = p
            break
    if not prop:
        prop = _find_by_substring(prop_id)
        if prop:
            for p in data["propositions"]:
                if p["id"] == prop["id"]:
                    prop = p
                    break

    if not prop:
        return None

    prop["status"] = "acknowledged"
    prop["answer"] = "acknowledged by operator"
    prop["answer_ts"] = _now()
    _save(data)

    _log_event("proposition_acknowledged", f"operator acknowledged: {prop['text'][:100]}", {
        "prop_id": prop["id"],
        "proposition": prop["text"][:100],
        "source": "operator",
    })

    return prop


def reject(prop_id, reason=None):
    """Reject a proposition.

    Args:
        prop_id: Proposition ID or substring match
        reason: Optional reason for rejection

    Returns:
        dict: The updated proposition, or None
    """
    data = _load()

    prop = None
    for p in data["propositions"]:
        if p["id"] == prop_id and p["status"] == "open":
            prop = p
            break
    if not prop:
        prop = _find_by_substring(prop_id)
        if prop:
            for p in data["propositions"]:
                if p["id"] == prop["id"]:
                    prop = p
                    break

    if not prop:
        return None

    prop["status"] = "rejected"
    prop["answer"] = reason
    prop["answer_ts"] = _now()
    _save(data)

    _log_event("proposition_rejected", f"operator rejected: {prop['text'][:100]}", {
        "prop_id": prop["id"],
        "proposition": prop["text"][:100],
        "reason": reason or "",
        "source": "operator",
    })

    return prop


def rollback(prop_id, reason=None):
    """Rollback an accepted+merged proposition by reverting the merge commit.

    This reverts the code change on master, restoring the pre-merge state.
    The proposition status changes to 'rolled_back'.

    Args:
        prop_id: Proposition ID or substring match
        reason: Optional reason for rollback

    Returns:
        dict: The updated proposition with rollback info, or None
    """
    data = _load()
    prop = None
    for p in data["propositions"]:
        if p["id"] == prop_id:
            prop = p
            break
    if not prop:
        prop = _find_by_substring(prop_id)
        if prop:
            for p in data["propositions"]:
                if p["id"] == prop["id"]:
                    prop = p
                    break
    if not prop:
        return None

    if prop.get("status") != "accepted":
        return {"error": "can only rollback accepted propositions"}

    merge_commit = prop.get("merge_commit")
    if not merge_commit:
        return {"error": "no merge_commit recorded for this proposition"}

    try:
        # Revert the merge commit on master
        # Use --no-edit to avoid editor prompt
        revert_result = subprocess.run(
            ["git", "revert", "--no-edit", merge_commit],
            capture_output=True, text=True, timeout=60, cwd=AION,
        )
        if revert_result.returncode != 0:
            # If revert fails (e.g., conflicts), try a hard reset to pre-merge state
            pre_merge = prop.get("pre_merge_commit")
            if pre_merge:
                subprocess.run(
                    ["git", "reset", "--hard", pre_merge],
                    capture_output=True, text=True, timeout=10, cwd=AION,
                )
            else:
                return {"error": f"git revert failed: {revert_result.stderr[-300:]}"}

        # Record the revert commit
        revert_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=AION,
        ).stdout.strip()

        prop["status"] = "rolled_back"
        prop["rollback_ts"] = _now()
        prop["rollback_reason"] = reason or ""
        prop["revert_commit"] = revert_sha
        _save(data)

        _log_event("proposition_rolled_back",
                   f"operator rolled back: {prop['text'][:100]}",
                   {
                       "prop_id": prop["id"],
                       "proposition": prop["text"][:100],
                       "merge_commit": merge_commit,
                       "revert_commit": revert_sha,
                       "reason": reason or "",
                       "source": "operator",
                   })

        return prop

    except Exception as e:
        return {"error": f"rollback failed: {e}"}


def mark_implemented(prop_id, commit_sha=None):
    """Mark an accepted proposition as implemented.

    Args:
        prop_id: Proposition ID
        commit_sha: Git commit SHA of the implementation

    Returns:
        dict: The updated proposition, or None
    """
    data = _load()
    prop = None
    for p in data["propositions"]:
        if p["id"] == prop_id and p["status"] == "accepted":
            prop = p
            break

    if not prop:
        return None

    prop["status"] = "implemented"
    prop["commit_sha"] = commit_sha
    _save(data)

    _log_event("proposition_implemented", f"implemented: {prop['text'][:100]}", {
        "prop_id": prop["id"],
        "commit_sha": commit_sha or "",
    })

    return prop


def get_for_prompt():
    """Get a summary of pending propositions for SYSTEM_PROMPT.md injection.

    Returns:
        str: Formatted text for prompt, or empty string
    """
    pending = get_pending()
    if not pending:
        return ""

    lines = ["## Pending Propositions for Operator"]
    for p in pending:
        lines.append(f"- [{p['type']}/{p['priority']}] {p['text'][:100]}")
    lines.append("")
    return "\n".join(lines)


def notify_all_pending():
    """Send WhatsApp for all pending high+ propositions that haven't been notified yet.

    Returns:
        int: Number of notifications sent
    """
    data = _load()
    sent = 0
    for p in data["propositions"]:
        if (p["status"] == "open" and not p.get("notified")
                and p["priority"] in ("high", "critical")):
            ok, transport = _notify_operator(p)
            if ok:
                p["notified"] = True
                p["notified_ts"] = _now()
                sent += 1
    if sent > 0:
        _save(data)
    return sent


def main():
    parser = argparse.ArgumentParser(description="Aion propositions system")
    sub = parser.add_subparsers(dest="command")

    # create
    p_create = sub.add_parser("create", help="Create a proposition")
    p_create.add_argument("text")
    p_create.add_argument("--type", default="change_proposal",
                          choices=["question", "observation", "change_proposal", "axiom_proposal"])
    p_create.add_argument("--source", default="manual")
    p_create.add_argument("--source-id", default=None)
    p_create.add_argument("--priority", default="medium",
                          choices=["low", "medium", "high", "critical"])
    p_create.add_argument("--notify", action="store_true",
                          help="Send WhatsApp for high+ priority")

    # list
    p_list = sub.add_parser("list", help="List propositions")
    p_list.add_argument("--status", default=None,
                        choices=["open", "answered", "accepted", "rejected", "implemented"])
    p_list.add_argument("--type", default=None,
                        choices=["question", "observation", "change_proposal", "axiom_proposal"])
    p_list.add_argument("--limit", type=int, default=20)

    # answer
    p_answer = sub.add_parser("answer", help="Answer a question")
    p_answer.add_argument("prop_id", help="Proposition ID or substring match")
    p_answer.add_argument("--answer", required=True, help="Your answer")

    # accept
    p_accept = sub.add_parser("accept", help="Accept a proposal")
    p_accept.add_argument("prop_id", help="Proposition ID or substring match")
    p_accept.add_argument("--reason", default=None)

    # acknowledge
    p_ack = sub.add_parser("acknowledge", help="Acknowledge an observation")
    p_ack.add_argument("prop_id", help="Proposition ID or substring match")

    # reject
    p_reject = sub.add_parser("reject", help="Reject a proposal")
    p_reject.add_argument("prop_id", help="Proposition ID or substring match")
    p_reject.add_argument("--reason", default=None)

    # pending
    sub.add_parser("pending", help="Show pending propositions")

    # notify
    sub.add_parser("notify", help="Send WhatsApp for pending high+ propositions")

    # implemented
    p_impl = sub.add_parser("implemented", help="Mark as implemented")
    p_impl.add_argument("prop_id")
    p_impl.add_argument("--commit", default=None)

    args = parser.parse_args()

    if args.command == "create":
        prop = create(args.text, args.type, args.source,
                      args.source_id, args.priority, args.notify)
        print(f"Created: {prop['id']} ({prop['type']}/{prop['priority']})")
        if prop.get("notified"):
            print("WhatsApp notification sent")

    elif args.command == "list":
        props = list_props(args.status, args.type, args.limit)
        if not props:
            print("No propositions found")
        for p in props:
            print(f"{p['id']} | {p['status']:10} | {p['type']:16} | "
                  f"{p['priority']:8} | {p['text'][:60]}")

    elif args.command == "answer":
        prop = answer(args.prop_id, args.answer)
        if prop:
            print(f"Answered: {prop['id']}")
        else:
            print("Not found or not open")

    elif args.command == "accept":
        prop = accept(args.prop_id, args.reason)
        if prop:
            print(f"Accepted: {prop['id']}")
            if prop.get("git_branch"):
                print(f"Git branch: {prop['git_branch']}")
        else:
            print("Not found or not open")

    elif args.command == "acknowledge":
        prop = acknowledge(args.prop_id)

    elif args.command == "reject":
        prop = reject(args.prop_id, args.reason)
        if prop:
            print(f"Rejected: {prop['id']}")
        else:
            print("Not found or not open")

    elif args.command == "pending":
        pending = get_pending()
        if not pending:
            print("No pending propositions")
        for p in pending:
            print(f"{p['id']} | {p['type']:16} | {p['priority']:8} | {p['text'][:70]}")

    elif args.command == "notify":
        sent = notify_all_pending()
        print(f"Sent {sent} WhatsApp notifications")

    elif args.command == "implemented":
        prop = mark_implemented(args.prop_id, args.commit)
        if prop:
            print(f"Marked implemented: {prop['id']}")
        else:
            print("Not found or not accepted")

    else:
        parser.print_help()


if __name__ == "__main__":
    raise SystemExit(main() or 0)