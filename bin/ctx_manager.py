#!/usr/bin/env python3
"""ctx_manager.py — shared context window management for Aion.

Prevents silent context overflow by estimating token counts and truncating
message history before sending to ollama.

Usage:
  from ctx_manager import estimate_tokens, truncate_messages, fit_events

  # Truncate a messages list to fit within a token budget
  messages = truncate_messages(messages, max_tokens=60000)

  # Fit episodic events into a token budget (keeps most recent)
  events_text = fit_events(events, max_tokens=20000)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

# Rough token estimation: ~4 chars per token for English/code
# (gemma tokenizer averages 3.5-4.5; this is conservative)
CHARS_PER_TOKEN = 4


def estimate_tokens(text):
    """Estimate token count for a string."""
    if not text:
        return 0
    return max(1, len(str(text)) // CHARS_PER_TOKEN)


def estimate_messages_tokens(messages):
    """Estimate total tokens for a list of {role, content} messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4  # role overhead per message
    return total


def truncate_messages(messages, max_tokens, protected_prefix=1):
    """Truncate a messages list to fit within max_tokens.

    Always preserves:
      - First `protected_prefix` messages (system prompt)
      - Last message (most recent context)

    Removes from the middle (oldest non-protected messages first).

    Args:
        messages: list of {role, content} dicts
        max_tokens: target token budget for the message list
        protected_prefix: number of messages from the start to always keep

    Returns:
        (truncated_messages, was_truncated, est_tokens)
    """
    est = estimate_messages_tokens(messages)
    if est <= max_tokens:
        return messages, False, est

    if len(messages) <= protected_prefix + 1:
        return messages, False, est

    protected = messages[:protected_prefix]
    middle = messages[protected_prefix:-1]
    last = messages[-1:]

    # Remove oldest middle messages until we fit
    removed = 0
    while middle and estimate_messages_tokens(protected + middle + last) > max_tokens:
        middle.pop(0)
        removed += 1

    result = protected + middle + last
    final_est = estimate_messages_tokens(result)
    return result, removed > 0, final_est


def truncate_messages_str(messages, max_tokens, protected_prefix=1):
    """Like truncate_messages but returns a summary string of what was removed.

    Returns:
        (truncated_messages, summary_string, est_tokens)
    """
    original_len = len(messages)
    result, was_truncated, est = truncate_messages(messages, max_tokens, protected_prefix)

    if not was_truncated:
        return result, "", est

    removed_count = original_len - len(result)
    summary = f"[context manager: {removed_count} older messages truncated to fit {max_tokens} token budget]"
    return result, summary, est


def fit_events(events, max_tokens, _format_fn=None):
    """Fit a list of episodic events into a token budget.

    Keeps the most recent events, dropping oldest first.

    Args:
        events: list of event dicts
        max_tokens: token budget
        _format_fn: optional function(event) -> string

    Returns:
        (events_text, count_included, count_dropped)
    """
    if _format_fn is None:
        def default_format(e):
            return f"[{e.get('ts','?')[:19]}] ({e.get('type','?')}) {str(e.get('text',''))[:300]}"
        fmt = default_format
    else:
        fmt = _format_fn

    # Format all events
    formatted = [fmt(e) for e in events]

    # Estimate and fit
    total_text = "\n".join(formatted)
    total_est = estimate_tokens(total_text)

    if total_est <= max_tokens:
        return total_text, len(events), 0

    # Drop oldest events until we fit
    included = len(formatted)
    while included > 1:
        included -= 1
        subset = "\n".join(formed for formed in formatted[-included:])
        if estimate_tokens(subset) <= max_tokens:
            dropped = len(events) - included
            subset = f"[context manager: {dropped} older events omitted to fit budget]\n" + subset
            return subset, included, dropped

    # Only 1 event fits
    return formatted[-1], 1, len(events) - 1


def log_context_usage(script_name, est_tokens, max_tokens, truncated=False):
    """Log context usage for monitoring. Called after every LLM call."""
    pct = round(est_tokens / max_tokens * 100, 1) if max_tokens > 0 else 0
    status = "OK"
    if pct > 90:
        status = "CRITICAL"
    elif pct > 75:
        status = "HIGH"
    elif pct > 50:
        status = "MODERATE"

    print(f"[ctx] {script_name}: {est_tokens}/{max_tokens} tokens ({pct}%) [{status}]{' TRUNCATED' if truncated else ''}")
    return pct
