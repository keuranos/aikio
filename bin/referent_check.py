#!/usr/bin/env python3
"""FIX C: pre-investigation referent check.

Before a curiosity goal spends model cycles and sandbox runs investigating a
named symbol (file, class, module, DB), check ONCE whether it actually exists
in the repo. If not, answer the goal immediately and authoritatively instead of
re-deriving absence over and over.

Why this is needed, concretely: Aion spent ~57 sandbox runs (33 in one day)
probing `predictions.db`, which does not exist. It HAD correctly determined this
several times ("predictions.db does not exist — the actual prediction system
uses JSON files") and even proposed deleting the ghost file. The correct
conclusion never stopped the behaviour, because nothing checked the referent
before opening a new investigation.

This is the same principle as the existing verify-then-repair gate for code
proposals: check the referent exists before acting on it.
"""
import os
import re
import glob
import subprocess

AION = "$AION_HOME"

# Symbols that look like repo artifacts worth checking. Deliberately
# conservative: only file-like, class-like and module-like tokens.
SYMBOL_PATTERNS = [
    r"\b[a-z0-9_]+\.(?:py|sh|db|json|jsonl|md|txt)\b",   # files
    r"\b[A-Z][A-Za-z0-9]+(?:Wrapper|Manager|Engine|Handler|Builder|Model|Predictor|Registry)\b",  # class-ish
]

# Words that are real but live outside the repo, so absence is not meaningful.
IGNORE = {
    "README.md", "LICENSE", "config.yaml", "requirements.txt",
    "predictions.jsonl", "predictions_open.json",  # real, but confirm anyway
}


def extract_symbols(text, limit=8):
    """Pull candidate repo-artifact symbols out of a question/hint."""
    found = []
    for pat in SYMBOL_PATTERNS:
        for m in re.finditer(pat, text or ""):
            s = m.group(0)
            if s in IGNORE:
                continue
            if s not in found:
                found.append(s)
            if len(found) >= limit:
                return found
    return found


SELF_FILE = "referent_check.py"


def symbol_exists(sym):
    """True if the symbol exists as a real artifact in the live repo.

    A checker that counts its own text as evidence is worthless: the first run
    reported the phantom `predictions.db` as PRESENT because this file's own
    docstring mentions it. So exclude self, exclude backups and docs, and treat
    a match as evidence only if it is a real file or a code/config reference.
    """
    try:
        # 1. as a real filename anywhere in the tree
        hits = glob.glob(f"{AION}/**/{sym}", recursive=True)
        hits = [h for h in hits
                if "/.git/" not in h and "/.venv/" not in h
                and not h.endswith(".bak") and "/backups/" not in h
                and os.path.basename(h) != SELF_FILE]
        if hits:
            return True, hits[:3]

        # 2. A FILE-like symbol (has an extension) exists only if the file is
        #    really on disk. A mention in a comment or docstring proves nothing
        #    — `predictions.db` appears as an example string in bin/sandbox.py
        #    but no such file exists. This distinction is the whole point.
        if "." in sym:
            return False, []

        # 3. A CLASS-like symbol exists by being defined/mentioned in code.
        r = subprocess.run(
            ["grep", "-rl", "--binary-files=without-match",
             "--include=*.py", "--include=*.sh", "--include=*.json",
             "--include=*.yaml", "--include=*.yml", "--include=*.html",
             "--include=*.js",
             "--exclude-dir=.git", "--exclude-dir=.venv",
             "--exclude-dir=node_modules", "--exclude-dir=gallery",
             "--exclude-dir=memory", "--exclude-dir=backups",
             "--exclude=*.bak", "--exclude=" + SELF_FILE,
             sym, f"{AION}/bin", f"{AION}/prompts", f"{AION}/config", f"{AION}/webui"],
            capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return True, r.stdout.strip().splitlines()[:3]
        return False, []
    except Exception as e:
        return None, [f"check failed: {e}"]     # unknown, do not block


def referent_report(question, hint=""):
    """Return (all_absent, report_lines, absent_symbols).

    all_absent=True means EVERY named repo symbol is absent — in which case the
    goal is almost certainly chasing a dream phantom and can be answered now.
    """
    text = (question or "") + " " + (hint or "")
    syms = extract_symbols(text)
    if not syms:
        return False, [], []

    lines, absent, present = [], [], []
    for s in syms:
        exists, where = symbol_exists(s)
        if exists is True:
            present.append(s)
            lines.append(f"  PRESENT  {s}  ({where[0].replace(AION + '/', '') if where else '?'})")
        elif exists is False:
            absent.append(s)
            lines.append(f"  ABSENT   {s}")
        else:
            lines.append(f"  UNKNOWN  {s}  ({where[0] if where else '?'})")

    all_absent = bool(absent) and not present
    return all_absent, lines, absent


if __name__ == "__main__":
    # Self-test against the real phantom and a real file.
    for q in [
        "The fallback path in UncertaintyWrapper leaves predictions.db schema mismatch causing incomplete rows",
        "Why does bin/predictions.py not track retry counts?",
        "Is the calibration_report.py isotonic fallback correct?",
    ]:
        aa, lines, absent = referent_report(q)
        print("Q:", q[:70])
        for l in lines:
            print(l)
        print("  -> ALL ABSENT (phantom):", aa)
        print()
