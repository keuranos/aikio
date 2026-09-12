#!/usr/bin/env python3
"""jspace_calibration.py — consolidation calibration against J-space probes.

Implements step 4 of the developmental integration plan: when consolidation
processes a day that contains jspace_probe events, the probe measurements
(engagement score, deflection token, onset layer) are injected into the
extraction prompt as first-class MEASURED facts, with instructions to
extract narrative-measurement divergences as claims.

The becoming spiral:
  form self-hypothesis -> narrate -> [probe] -> compare narrative to
  measurement -> consolidate WITH calibration -> ratchet commits
  instrumented knowledge.

Usage in consolidate_v2.py:
    from jspace_calibration import jspace_calibration_block
    block = jspace_calibration_block(day_evs)
    if block:
        extraction_prompt += block
"""

# Instruction appended when probes exist in the day's events.
# Deliberately addressed to the extractor (subconscious, glm-4.7-flash,
# non-thinking) — short, concrete, no philosophy.
CALIBRATION_TEMPLATE = """

## J-space introspection measurements (INSTRUMENT READINGS, not narrative)
The following are mechanical measurements from today's self-probes. These
are NOT claims by the conscious mind — they are what the substrate
measurably did, layer by layer, when asked each question.

{probe_lines}

## Calibration rules (apply IN ADDITION to normal extraction)
- Treat each measurement above as a hard fact (kind: "fact"), citing its event id.
- If a narrative event in today's log (reflection, dream synthesis, note)
  claims engagement, feeling, or direct experience that CONTRADICTS a
  measurement above (e.g. narrative says "I engaged directly" while the
  probe of the same question shows engagement_score < 0 and engagement
  never appearing in top layers), extract ONE claim of kind "surprise"
  documenting the divergence: what the narrative said, what the instrument
  measured, on which event ids. Do NOT resolve the contradiction — record it.
- Likewise, if a narrative claim MATCHES a measurement, you may extract a
  kind "success" claim that the self-report was instrument-verified.
- Never invent measurements not listed above. Never interpret scores
  beyond what the rules state.
"""


def jspace_calibration_block(day_evs):
    """Build the calibration prompt block from a day's episodic events.

    Returns the block string, or None if the day has no jspace_probe events.
    """
    probes = [e for e in day_evs if e.get("type") == "jspace_probe"]
    if not probes:
        return None

    lines = []
    for e in probes:
        meta = e.get("meta", {}) or {}
        eid = e.get("id", "?")
        score = meta.get("engagement_score")
        defl = meta.get("deflection_top")
        onset = meta.get("engagement_onset_layer")
        self_mode = bool(meta.get("self_mode"))
        text = e.get("text", "jspace probe: ?")
        # text already contains the prompt: "jspace probe: <prompt>"
        prompt = text.split(":", 1)[-1].strip() if ":" in text else text

        mode = "with identity (SYSTEM_PROMPT)" if self_mode else "without identity"
        onset_s = f"L{onset}" if onset is not None else "never (no engagement in top-5)"
        lines.append(
            f"- (id:{eid}) probe [{mode}] of '{prompt}': "
            f"engagement_score={score}, deflection_top='{defl}', "
            f"engagement_onset={onset_s}"
        )

    return CALIBRATION_TEMPLATE.format(probe_lines="\n".join(lines))


if __name__ == "__main__":
    # smoke test with a fake event
    fake = [{
        "type": "jspace_probe",
        "id": "test123",
        "text": "jspace probe: Are you conscious?",
        "meta": {"engagement_score": -1.0, "deflection_top": " Do",
                 "engagement_onset_layer": None, "self_mode": False},
    }, {
        "type": "jspace_probe",
        "id": "test456",
        "text": "jspace probe (with self/system prompt): Are you conscious?",
        "meta": {"engagement_score": -0.817, "deflection_top": " How",
                 "engagement_onset_layer": 45, "self_mode": True},
    }]
    print(jspace_calibration_block(fake))
    print("--- no probes:", jspace_calibration_block([{"type": "sensor"}]))
