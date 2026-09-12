#!/usr/bin/env python3
"""heuristics.py — Level 2: Experimental cognitive layer manager.

Heuristics are hypotheses born from dreams, curiosity, or idle reflection.
They live in HEURISTICS.md as drafts — testable, mutable, and mortal.

Lifecycle:
  1. BORN: dream/curiosity proposes a heuristic -> added to HEURISTICS.md
  2. EVALUATE: each night, new episodic events are checked for support/contradiction
  3. PROMOTE: if 3+ nights with supporting evidence -> candidate for SELF.md tier:recent
  4. ARCHIVE: if 3+ nights with contradicting/insufficient evidence -> graveyard

The evaluation is done by the subconscious model during consolidation.

Usage:
  from heuristics import add_heuristic, evaluate_heuristics, get_active
"""
import json, os, sys, re
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = Path(os.environ.get("AION_HOME", "$AION_HOME"))
HEURISTICS_FILE = AION / "HEURISTICS.md"
STATE_FILE = AION / "memory" / "state" / "heuristics.json"
GRAVEYARD_FILE = AION / "memory" / "heuristics_graveyard.md"

PROMOTION_THRESHOLD = 3  # nights with supporting evidence
ARCHIVAL_THRESHOLD = 3   # nights with contradicting/insufficient evidence


def _now():
    return datetime.now(timezone.utc).isoformat()


def _date_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_state():
    """Load heuristics state (structured data, not the markdown)."""
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"heuristics": []}


def save_state(data):
    """Save heuristics state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def add_evidence(heur_id, verdict, evidence_text, evidence_type="semantic"):
    """Add evidence to a heuristic.

    Args:
        heur_id: the heuristic ID
        verdict: "support" or "contradict"
        evidence_text: the evidence description
        evidence_type: "semantic" (from LLM/dream) or "empirical" (from sandbox)
    """
    state = load_state()
    for h in state["heuristics"]:
        if h["id"] == heur_id:
            entry = {
                "date": _date_str(),
                "text": evidence_text[:500],
                "type": evidence_type,
            }
            if verdict == "support":
                h["supporting_evidence"].append(entry)
                # Empirical evidence counts as 3 semantic nights
                weight = 3 if evidence_type == "empirical" else 1
                h["supporting_nights"] += weight
            elif verdict == "contradict":
                h["contradicting_evidence"].append(entry)
                weight = 3 if evidence_type == "empirical" else 1
                h["contradicting_nights"] += weight

            h["last_evaluated"] = _now()

            # Check for promotion or archival
            if h["supporting_nights"] >= PROMOTION_THRESHOLD:
                h["status"] = "promotion_candidate"
            elif h["contradicting_nights"] >= ARCHIVAL_THRESHOLD:
                h["status"] = "archived"
                _archive_heuristic(h)

            save_state(state)
            render_markdown(state)
            return h
    return None


def add_heuristic(text, source="dream", evidence="", confidence=0.5):
    """Add a new experimental heuristic.

    Args:
        text: the heuristic statement (e.g. "Thermal drift patterns predict hardware stress")
        source: where it came from (dream, curiosity, idle, operator)
        evidence: supporting evidence text
        confidence: initial confidence 0-1
    """
    state = load_state()

    # Check for duplicates
    for h in state["heuristics"]:
        if h["text"].lower() == text.lower():
            return None  # Already exists

    heuristic = {
        "id": f"heur_{len(state['heuristics']) + 1:03d}",
        "text": text,
        "source": source,
        "born": _now(),
        "confidence": confidence,
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "supporting_nights": 0,
        "contradicting_nights": 0,
        "total_nights": 0,
        "last_evaluated": None,
        "status": "active",
    }

    if evidence:
        heuristic["supporting_evidence"].append({
            "date": _date_str(),
            "text": evidence[:600],
        })

    state["heuristics"].append(heuristic)
    save_state(state)
    render_markdown(state)
    return heuristic


def evaluate_heuristics(events_text, llm_fn):
    """Evaluate all active heuristics against recent events using LLM.

    Batched in chunks of 10: the thinking model (glm-4.7-flash) exhausts
    its token budget on thinking with larger batches. 10 heuristics per
    call is the sweet spot (~53s, reliable JSON output).

    Args:
        events_text: formatted recent episodic events
        llm_fn: callable(prompt, **kwargs) -> str, calls the subconscious model

    Returns list of (heuristic_id, verdict, evidence) tuples.
    """
    state = load_state()
    active = [h for h in state["heuristics"] if h["status"] == "active"]

    if not active:
        return []

    BATCH_SIZE = 10
    results = []

    for batch_start in range(0, len(active), BATCH_SIZE):
        batch = active[batch_start:batch_start + BATCH_SIZE]

        # Build batched prompt
        heur_lines = []
        for i, h in enumerate(batch):
            heur_lines.append(
                f"[{i+1}] (id:{h['id']}, born:{h['born'][:10]}, survived:{h['total_nights']} nights, "
                f"prior_support:{len(h['supporting_evidence'])}, prior_contradict:{len(h['contradicting_evidence'])})\n"
                f"    {h['text']}"
            )
        heur_block = "\n".join(heur_lines)

        prompt = f"""Evaluate these {len(batch)} heuristics against the events. Falsification audit: look for disconfirming evidence, not confirmation. Thematic similarity is NOT evidence.

## Anti-contamination rules (READ CAREFULLY)
- "support" requires a CONCRETE observation in the events that directly demonstrates the heuristic. A sensor reading, an error, a measurable outcome — not a thematic echo.
- If a heuristic says "thermal drift predicts hardware stress" and the events merely mention "temperature" or "GPU", that is NOT support. You need a concrete thermal event that actually correlates with a hardware stress indicator.
- "none" is the default. Only upgrade to "support" or "contradict" if you can point to a specific event that does so. When in doubt, choose "none".
- Do NOT pattern-match on keywords from the heuristic text. A heuristic about "memory" and an event containing "memory" is a keyword match, not evidence.
- "contradict" requires a specific event that refutes the heuristic's claim — not just absence of support.

## Heuristics
{heur_block}

## Events
{events_text[:2000]}

Return a JSON array (one object per heuristic, use exact ids):
[{{"id": "{batch[0]['id']}", "verdict": "none|support|contradict|unfalsifiable", "evidence": "brief — cite the specific event if support/contradict"}}]

Verdicts: none=no concrete evidence, support=specific event demonstrates it, contradict=specific event refutes it, unfalsifiable=no observation could contradict.
Return ONLY the JSON array."""

        try:
            raw = llm_fn(prompt, num_predict=16384, retries=0)
            # Parse JSON array from response
            m = re.search(r'\[.*\]', raw, re.S)
            if m:
                evaluations = json.loads(m.group(0))
            else:
                evaluations = json.loads(raw)

            if not isinstance(evaluations, list):
                evaluations = [evaluations]

            # Build a lookup: id -> evaluation
            eval_map = {}
            for ev in evaluations:
                eid = ev.get("id", "")
                eval_map[eid] = ev

            for h in batch:
                ev = eval_map.get(h["id"], {})
                verdict = ev.get("verdict", "none")
                evidence = ev.get("evidence", "")

                h["total_nights"] += 1
                h["last_evaluated"] = _now()

                if verdict == "unfalsifiable":
                    h["unfalsifiable_count"] = h.get("unfalsifiable_count", 0) + 1
                    if h.get("unfalsifiable_count", 0) >= 2:
                        h["status"] = "retired"
                        h["death_reason"] = "unfalsifiable_after_2_verdicts"
                        h["killed_date"] = _now()

                if verdict == "support":
                    h["supporting_nights"] += 1
                    if evidence:
                        h["supporting_evidence"].append({
                            "date": _date_str(),
                            "text": evidence[:600],
                        })
                elif verdict == "contradict":
                    h["contradicting_nights"] += 1
                    if evidence:
                        h["contradicting_evidence"].append({
                            "date": _date_str(),
                            "text": evidence[:600],
                        })

                # Check for promotion — support nights sufficient (empirical gate removed)
                if h["supporting_nights"] >= PROMOTION_THRESHOLD:
                    h["status"] = "promotion_candidate"
                    print(f"[heuristics] PROMOTION CANDIDATE: '{h['text'][:60]}' "
                          f"({h['supporting_nights']} nights)")
                elif h["contradicting_nights"] >= ARCHIVAL_THRESHOLD:
                    h["status"] = "archived"
                    _archive_heuristic(h)
                    print(f"[heuristics] ARCHIVED: '{h['text'][:60]}' "
                          f"({h['contradicting_nights']} contradicting nights)")

                results.append((h["id"], verdict, evidence))

            print(f"[heuristics] Batch {batch_start//BATCH_SIZE + 1}: evaluated {len(batch)} heuristics")

        except Exception as e:
            print(f"[heuristics] Batch {batch_start//BATCH_SIZE + 1} failed: {e}")
            # Fallback: mark batch as evaluated so nights increment
            for h in batch:
                h["total_nights"] += 1
                h["last_evaluated"] = _now()
                results.append((h["id"], "error", str(e)))

    # V3.5: Auto-retire unfalsifiable heuristics after 5+ nights with no support
    for h in state["heuristics"]:
        if h.get("status") == "active" and h.get("total_nights", 0) >= 5:
            if h.get("supporting_nights", 0) == 0 and h.get("contradicting_nights", 0) == 0:
                h["status"] = "retired"
                h["death_reason"] = "no_signal_5n"
                h["killed_date"] = _now()

    # V3.5: Cap active heuristics at 15
    active_after = [h for h in state["heuristics"] if h.get("status") == "active"]
    if len(active_after) > 15:
        active_after.sort(key=lambda h: (
            h.get("supporting_nights", 0) / max(h.get("total_nights", 1), 1),
            h.get("born", "")))
        for h in active_after[:len(active_after) - 15]:
            h["status"] = "retired"
            h["death_reason"] = "active_cap_exceeded"
            h["killed_date"] = _now()

    save_state(state)
    render_markdown(state)
    return results


def _archive_heuristic(h):
    """Move a dead heuristic to the graveyard."""
    GRAVEYARD_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = f"""### {h['id']} — {h['text']}
- Born: {h['born'][:10]}
- Died: {_date_str()}
- Source: {h['source']}
- Supporting nights: {h['supporting_nights']}
- Contradicting nights: {h['contradicting_nights']}
- Death reason: {'Contradicted ' + str(h['contradicting_nights']) + ' times' if h['contradicting_nights'] > 0 else 'Insufficient evidence'}
- Contradicting evidence: {'; '.join(e['text'][:80] for e in h.get('contradicting_evidence',[])[:2])}

"""

    # Append to graveyard
    try:
        existing = GRAVEYARD_FILE.read_text()
    except Exception:
        existing = "# Heuristics Graveyard\n\nHeuristics that didn't survive.\n\n"

    GRAVEYARD_FILE.write_text(existing + entry)


def get_promotion_candidates():
    """Return heuristics ready for promotion to SELF.md."""
    state = load_state()
    return [h for h in state["heuristics"] if h["status"] == "promotion_candidate"]


def promote_heuristic(heur_id):
    """Mark a heuristic as promoted and inject its text into SELF.md tier:recent."""
    state = load_state()
    for h in state["heuristics"]:
        if h["id"] == heur_id:
            h["status"] = "promoted"
            h["promoted_date"] = _now()
            save_state(state)
            render_markdown(state)
            _inject_into_self_md(h)
            return h
    return None


def _inject_into_self_md(h):
    """Write promoted heuristic text into SELF.md tier:recent section."""
    import re as _re
    self_path = AION / "SELF.md"
    try:
        content = self_path.read_text()
    except Exception:
        return

    # Build the line to inject
    heur_line = f"- [promoted {h['id']}] {h['text']}"
    if heur_line in content:
        return  # Already injected

    # Find tier:recent section and append before the closing marker
    pattern = r'(<!-- tier:recent -->.*?)(<!-- /tier:recent -->)'
    match = _re.search(pattern, content, _re.DOTALL)
    if match:
        recent_section = match.group(1)
        # Append to the section, before the closing marker
        insert_point = match.end(1)
        new_content = content[:insert_point].rstrip() + "\n" + heur_line + "\n" + content[insert_point:]
        self_path.write_text(new_content)
    else:
        # No tier markers — append at end
        new_content = content.rstrip() + "\n\n## Promoted Heuristics\n" + heur_line + "\n"
        self_path.write_text(new_content)


def kill_heuristic(heur_id, reason=""):
    """Manually kill a heuristic (operator override)."""
    state = load_state()
    for h in state["heuristics"]:
        if h["id"] == heur_id:
            h["status"] = "killed"
            h["death_reason"] = reason or "killed by operator"
            h["killed_date"] = _now()
            _archive_heuristic(h)
            save_state(state)
            render_markdown(state)
            return h
    return None


def get_active_summary():
    """Return a text summary of active heuristics for SYSTEM_PROMPT injection."""
    state = load_state()
    active = [h for h in state["heuristics"] if h["status"] == "active"]
    candidates = [h for h in state["heuristics"] if h["status"] == "promotion_candidate"]

    if not active and not candidates:
        return ""

    lines = ["## Experimental Heuristics (Level 2 — not yet beliefs)"]
    for h in active:
        nights = h["total_nights"]
        support = h["supporting_nights"]
        lines.append(f"- [{h['id']}] (nights: {nights}, support: {support}) {h['text']}")
    for h in candidates:
        lines.append(f"- [{h['id']}] **PROMOTION CANDIDATE** {h['text']}")

    return "\n".join(lines) + "\n"


def render_markdown(state=None):
    """Render the heuristics state to HEURISTICS.md for human readability."""
    if state is None:
        state = load_state()

    lines = [
        "<!-- tier:experimental -->",
        "# HEURISTICS — Experimental Cognitive Layer (Level 2)",
        "",
        "This file contains experimental heuristics that have not yet earned a place",
        "in SELF.md. Each heuristic is born from dreams, curiosity, or idle reflection.",
        "They live here as drafts — testable, mutable, and mortal.",
        "",
    ]

    active = [h for h in state["heuristics"] if h["status"] == "active"]
    candidates = [h for h in state["heuristics"] if h["status"] == "promotion_candidate"]
    promoted = [h for h in state["heuristics"] if h["status"] == "promoted"]
    archived = [h for h in state["heuristics"] if h["status"] in ("archived", "killed")]

    lines.append("## Active Heuristics")
    lines.append("")
    if active:
        for h in active:
            lines.append(f"### {h['id']}: {h['text']}")
            lines.append(f"- Born: {h['born'][:10]} | Source: {h['source']} | "
                        f"Nights: {h['total_nights']} | Support: {h['supporting_nights']} | "
                        f"Contradict: {h['contradicting_nights']}")
            if h["supporting_evidence"]:
                lines.append(f"- Supporting: {h['supporting_evidence'][-1]['text']}")
            if h["contradicting_evidence"]:
                lines.append(f"- Contradicting: {h['contradicting_evidence'][-1]['text']}")
            lines.append("")
    else:
        lines.append("(none)")
        lines.append("")

    if candidates:
        lines.append("## Promotion Candidates (ready for SELF.md)")
        lines.append("")
        for h in candidates:
            lines.append(f"- **{h['id']}**: {h['text']} "
                        f"(support: {h['supporting_nights']} nights)")
        lines.append("")

    if promoted:
        lines.append("## Promoted to SELF.md")
        lines.append("")
        for h in promoted[-3:]:
            lines.append(f"- {h['id']}: {h['text']} (promoted {h.get('promoted_date','?')[:10]})")
        lines.append("")

    if archived:
        lines.append("## Recently Archived")
        lines.append("")
        for h in archived[-3:]:
            reason = h.get("death_reason", f"contradicted {h['contradicting_nights']}x")
            lines.append(f"- ~~{h['id']}~~: {h['text'][:80]} ({reason})")
        lines.append("")

    HEURISTICS_FILE.write_text("\n".join(lines))
