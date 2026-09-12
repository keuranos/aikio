#!/usr/bin/env python3
"""code_review.py — semantic review of Aion-proposed code by the other layers.

Sits between the verify gate (syntax/import) and the operator dashboard.
The authoring model writes the diff; gemma4 (the conscious model, different
family from qwen3.8) reviews it against a FACTS SHEET of ground truth from
the production system — real model names, real registry locations, real
files. This is the layer that would have caught all three resonance_probe
defects. muse-glimmer adds a light coherence check: does this change fit
what Aion has been trying to do lately?

Verdicts:
  APPROVE       — diff semantically sound; review text attached to the prop
  REJECT        — diff never reaches the dashboard; reasons returned to the
                  caller (tool_propose_code) as repair feedback for the author
  review_failed — fail-open (proposition proceeds unreviewed, flagged)

Never trust a diff that hasn't executed (code_verify's job);
never trust a diff that only its author understands (this file's job).
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aion_env  # noqa: F401
try:
    from aion_models import code_endpoint
except Exception:
    def code_endpoint(**kw):
        return (os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436"),
                os.environ.get("MAIN_MODEL", "gemma4:31b-65k"))

AION = os.environ.get("AION_HOME", "$AION_HOME")
MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "muse-glimmer:latest")
REVIEW_TIMEOUT = 240
REVIEW_NUM_PREDICT = 2048


def _chat(url, model, messages, num_ctx, think=True):
    def _call(t):
        body = json.dumps({
            "model": model, "stream": False, "messages": messages,
            "options": {"num_predict": REVIEW_NUM_PREDICT, "num_ctx": num_ctx,
                        "temperature": 0.3},
            **({} if t else {"think": False}),
        }).encode()
        req = urllib.request.Request(url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=REVIEW_TIMEOUT) as r:
            return json.loads(r.read()).get("message", {}).get("content", "")
    out = _call(think)
    if not out.strip():
        out = _call(False)
    return out.strip()


def build_facts_sheet():
    """Ground truth about the production system, from source — not memory."""
    facts = []
    # models actually configured
    for var in ("MAIN_MODEL", "INTUITION_MODEL", "SUB_MODEL", "CODE_MODEL",
                "CRITIC_MODEL"):
        v = os.environ.get(var)
        if v:
            facts.append("%s=%s" % (var, v))
    # tool registries: files + tool names actually registered
    bin_dir = os.path.join(AION, "bin")
    for reg_file in ("curiosity_engine.py", "wake_v2.py", "jspace_tool.py"):
        p = os.path.join(bin_dir, reg_file)
        if not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8").read()
        tools = sorted(set(re.findall(
            r'"([a-z_]+)":\s*(?:tool_|def\s+tool_)?[a-z_]+', src)))[:40]
        facts.append("%s tool registrations: %s" % (reg_file,
                     ", ".join(t for t in tools if len(t) > 3) or "none found"))
    # bin/ inventory (top-level names only)
    try:
        files = sorted(f for f in os.listdir(bin_dir) if f.endswith(".py"))
        facts.append("bin/ has %d python files including: %s" %
                     (len(files), ", ".join(files[:25])))
    except Exception:
        pass
    return "\n".join("  - " + f for f in facts)


def review_change(filepath, old_text, new_text, description,
                  review_intuition=True):
    """Semantic review. Returns {verdict, review, reasons, reviewer}."""
    full = os.path.join(AION, filepath)
    is_new = not old_text and not os.path.exists(full)
    resulting = new_text if is_new else None
    if resulting is None:
        if not os.path.exists(full):
            return {"verdict": "review_failed", "review": "",
                    "reasons": "file missing", "reviewer": MAIN_MODEL}
        content = open(full, encoding="utf-8").read()
        resulting = content.replace(old_text, new_text, 1) \
            if (old_text and old_text in content) else new_text

    diff_preview = resulting[:12000]
    facts = build_facts_sheet()

    review = _chat(MAIN_URL, MAIN_MODEL, [
        {"role": "system", "content":
            "You are reviewing a code change proposed by Aion's code layer "
            "before it reaches the operator. You are given a FACTS SHEET of "
            "ground truth about the production system. Check the change "
            "against it: are model names real? Do referenced files, tools "
            "and functions exist? Is the registry being edited the right "
            "one? Does the wiring actually activate (no commented-out "
            "registration, no dead code paths)? Is anything hallucinated? "
            "Answer in this exact format:\n"
            "VERDICT: APPROVE or REJECT\n"
            "REASONS: specific, cite the exact defect or the exact fact "
            "checked"},
        {"role": "user", "content":
            "FACTS SHEET (ground truth):\n%s\n\n"
            "FILE: %s (%s)\nDESCRIPTION: %s\n\n"
            "RESULTING CONTENT (first 12k chars):\n%s"
            % (facts, filepath, "new file" if is_new else "edit",
               description, diff_preview)},
    ], 65536)

    verdict = "review_failed"
    m = re.search(r"VERDICT:\s*(APPROVE|REJECT)", review, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()

    coherence = ""
    if review_intuition:
        try:
            coherence = _chat(INTUITION_URL, INTUITION_MODEL, [
                {"role": "system", "content":
                    "You are Aion's intuition layer. A code change has been "
                    "proposed. In under 80 words: does this change cohere "
                    "with Aion's recent direction (self-measurement, "
                    "engineering drive, consciousness research), or does it "
                    "duplicate something it already built? One line."},
                {"role": "user", "content":
                    "FILE: %s\nDESCRIPTION: %s\nCHANGE HEAD:\n%s"
                    % (filepath, description, diff_preview[:3000])},
            ], 16384, think=False)
        except Exception as e:
            coherence = "(coherence check unavailable: %s)" % e

    return {"verdict": verdict, "review": review,
            "coherence": coherence, "reasons": review[:1500],
            "reviewer": MAIN_MODEL}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("filepath")
    p.add_argument("--description", default="")
    p.add_argument("--old", default="")
    p.add_argument("--new", default="")
    a = p.parse_args()
    old = a.old
    if old and os.path.exists(old):
        old = open(old).read()
    new = a.new
    if new and os.path.exists(new):
        new = open(new).read()
    r = review_change(a.filepath, old, new, a.description)
    print(json.dumps(r, indent=1))
