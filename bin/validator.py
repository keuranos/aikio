#!/usr/bin/env python3
"""validator.py — R4.2: Pure-Python diff validator.

Rejects a reflection diff if:
(a) any hunk lacks event-ID citations
(b) cited IDs don't exist in episodic memory
(c) the reflection text contains no numbers/filenames/timestamps
(d) >20 changed lines in one night
(e) changes to tier:core or tier:stable sections

Empty diff is VALID — returns (True, "empty_diff").
"""
import re, os, glob, json, sys
from datetime import datetime, timezone

AION = os.environ.get("AION_HOME", "$AION_HOME")

def get_all_event_ids():
    """Collect all event IDs from episodic memory."""
    ids = set()
    for path in glob.glob(f"{AION}/memory/episodic/*.jsonl"):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                        if ev.get("id"):
                            ids.add(ev["id"])
                    except Exception:
                        pass
        except Exception:
            pass
    return ids

def count_changed_lines(diff_text):
    """Count actual content changes (added/removed, excluding diff headers)."""
    count = 0
    for line in diff_text.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count

def has_specific_content(diff_text):
    """Check if the diff contains at least some specific details."""
    added_lines = [l for l in diff_text.split("\n") if l.startswith("+") and not l.startswith("+++")]
    text = " ".join(added_lines)
    # Look for: numbers, dates, filenames, timestamps, temperatures
    has_number = bool(re.search(r'\d+', text))
    has_date = bool(re.search(r'\d{4}-\d{2}-\d{2}', text))
    has_filename = bool(re.search(r'[\w]+\.\w{2,4}', text))
    has_temp = bool(re.search(r'\d+\s*[°°]C', text))
    return has_number  # At minimum, some digits

def check_tier_safety(diff_text, self_md):
    """Check that tier:core and tier:stable sections are not modified."""
    # Find the lines being changed
    changed_line_nums = set()
    for line in diff_text.split("\n"):
        m = re.match(r'^@@.*\+(\d+)', line)
        if m:
            changed_line_nums.add(int(m.group(1)))
    
    # Check if any changed lines fall in protected tiers
    lines = self_md.split("\n")
    current_tier = None
    protected_changes = []
    
    for i, line in enumerate(lines, 1):
        if "<!-- tier:core -->" in line:
            current_tier = "core"
        elif "<!-- tier:stable -->" in line:
            current_tier = "stable"
        elif "<!-- tier:recent -->" in line:
            current_tier = "recent"
        elif "<!-- /tier:" in line:
            current_tier = None
        
        if i in changed_line_nums and current_tier in ("core", "stable"):
            protected_changes.append((i, current_tier))
    
    return len(protected_changes) == 0, protected_changes

def extract_cited_ids(diff_text):
    """Extract all cited event IDs from # refs: comments.
    
    Event IDs are 12-character hex strings. The regex must NOT match across
    newlines — otherwise the next diff line (starting with ' - ') gets
    captured as part of the ID.
    """
    cited = set()
    # [^\S\n] matches whitespace except newlines, so we stay on one line
    for m in re.finditer(r'#\s*refs?:\s*([\w_,]+(?:[^\S\n][\w_-]+)*)', diff_text):
        for id_str in m.group(1).split(","):
            id_str = id_str.strip()
            if id_str:
                cited.add(id_str)
    return cited

def validate(diff_text, self_md=""):
    """Validate a reflection diff. Returns (is_valid, reason)."""
    
    # R5.4: Check identity freeze
    freeze_file = f"{AION}/memory/state/identity_freeze.json"
    try:
        with open(freeze_file) as f:
            freeze = json.load(f)
        if freeze.get("frozen"):
            return False, f"identity_frozen: {freeze.get('reason', 'unknown')} — operator must clear"
    except Exception:
        pass
    
    # Empty diff is valid
    if diff_text.strip() == "EMPTY" or not diff_text.strip():
        return True, "empty_diff"
    
    # Strip diff fences if present
    diff_text = re.sub(r'^```diff\s*', '', diff_text.strip(), flags=re.M)
    diff_text = re.sub(r'^```\s*$', '', diff_text, flags=re.M)
    
    # (f) Check for truncated diffs — last line should not be a dangling remove
    diff_lines = diff_text.strip().split("\n")
    if diff_lines:
        last_line = diff_lines[-1].strip()
        # A diff ending mid-remove is truncated by the LLM
        if last_line.startswith("-") and not last_line.startswith("---"):
            # Check if the remove line is suspiciously short (truncated mid-word)
            content = last_line[1:].strip()
            if len(content) < 80 and not content.endswith((".", "!", "?", ")", "\"", "'")):
                return False, f"truncated_diff: last remove line appears cut off: '{content[:40]}...'"
    
    # (e) Check tier safety
    if self_md:
        safe, violations = check_tier_safety(diff_text, self_md)
        if not safe:
            return False, f"tier_safety: changes to protected tier(s): {violations}"
    
    # (a) Check for event-ID citations
    cited_ids = extract_cited_ids(diff_text)
    if not cited_ids:
        return False, "no_citations: every hunk must cite event IDs"
    
    # (b) Check cited IDs exist
    all_ids = get_all_event_ids()
    if all_ids:  # Only check if we have events to compare against
        missing = cited_ids - all_ids
        if missing:
            return False, f"invalid_citations: IDs not in episodic memory: {missing}"
    
    # (c) Check for specific content
    if not has_specific_content(diff_text):
        return False, "no_specifics: diff contains no numbers/filenames/timestamps"
    
    # (d) Rate limit: max 20 changed lines
    changed = count_changed_lines(diff_text)
    if changed > 20:
        return False, f"rate_limit: {changed} changed lines (max 20)"
    
    return True, f"valid ({changed} lines, {len(cited_ids)} citations)"


if __name__ == "__main__":
    # CLI: read diff from stdin, validate against SELF.md
    diff_text = sys.stdin.read()
    self_md = ""
    try:
        with open(f"{AION}/SELF.md") as f:
            self_md = f.read()
    except Exception:
        pass
    
    valid, reason = validate(diff_text, self_md)
    print(json.dumps({"valid": valid, "reason": reason}))
    sys.exit(0 if valid else 1)
