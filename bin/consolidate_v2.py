#!/usr/bin/env python3
"""consolidate_v2.py — R4: Two-pass reflection pipeline.

Pass 1 — Extraction (glm-4.7-flash): day's events → structured JSON claims
Pass 2 — Integration: SELF.md + claims → unified diff against SELF.md
Validator (pure Python): reject diffs lacking citations, specifics, or violating tiers
Critic (qwen3.6:35b-a3b): score the reflection 1-5 on specificity + groundedness
  — Different model from extraction/integration to prevent self-grading
  — Programmatic citation verification supplements LLM scoring
  — Score < 3 BLOCKS diff application (not just logged)

Empty diff is VALID and HEALTHY — logged as {type:"consolidation", change:"none"}.
"""
import json, os, glob, subprocess, re, sys, time
from datetime import datetime, timezone
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # loads config/aion.env
try:
    import body_schema
    _HAS_BODY_SCHEMA = True
except ImportError:
    _HAS_BODY_SCHEMA = False

AION = os.environ.get("AION_HOME", "$AION_HOME")
SUB_URL = os.environ.get("OLLAMA_CONSOLIDATE_URL", os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438"))
SUB_MODEL = os.environ.get("OLLAMA_CONSOLIDATE_MODEL", os.environ.get("INTUITION_MODEL", "glm-4.7-flash:q4_K_M"))

# Critic uses a DIFFERENT model than extraction/integration to avoid self-grading.
# Same model as the auditor (qwen3.6:35b-a3b) on the MAIN instance.
CRITIC_URL = os.environ.get("OLLAMA_CRITIC_URL", os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436"))
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "qwen3.6:35b-a3b")

def read(p, d=""):
    try:
        return open(p, encoding="utf-8").read()
    except Exception:
        return d

def write(p, t):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(t)

def load_json(p, d):
    try:
        return json.load(open(p))
    except Exception:
        return d

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def advance_marker(change, events_count=0, claims_count=0, score=None, extra=None):
    """Write last_consolidation.json on every exit path (V3.0.2)."""
    data = {"ts": now_iso(), "events": events_count, "claims": claims_count,
            "change": change}
    if score is not None:
        data["score"] = score
    if extra:
        data.update(extra)
    write(f"{AION}/memory/state/last_consolidation.json", json.dumps(data, indent=2))

def events_since(last_iso):
    """V3.1.2: Read events from the experience lane only (not telemetry)."""
    try:
        from lanes import events_since as lane_events_since
        return lane_events_since(AION, last_iso, lane="experience")
    except ImportError:
        # Fallback to old behavior
        last = datetime.fromisoformat(last_iso)
        evs = []
        for path in sorted(glob.glob(f"{AION}/memory/episodic/*.jsonl"))[-7:]:
            if "/telemetry/" in path:
                continue
            for line in open(path, encoding="utf-8"):
                try:
                    ev = json.loads(line)
                    if datetime.fromisoformat(ev["ts"]) > last:
                        evs.append(ev)
                except Exception:
                    pass
        return evs

def format_events(evs):
    """Format events for the extraction prompt."""
    lines = []
    for e in evs:
        eid = e.get("id", "?")
        ts = e.get("ts", "?")[:19]
        typ = e.get("type", "?")
        text = e.get("text", "")[:500]
        lines.append(f"[{ts}] (id:{eid}) {typ}: {text}")
    return "\n".join(lines)

def llm(prompt, temperature=0.4, num_ctx=32768, num_predict=8192, retries=2, max_predict=16384):
    """Call the subconscious model. Retries on empty response (thinking model
    sometimes exhausts token budget on thinking, returning empty content).
    After the retry loop one final rescue attempt runs with thinking disabled
    (think:false on /api/chat) so the entire budget goes to content. Sets
    llm.last_all_empty=True only when EVERY attempt incl. the rescue returned
    empty, so callers can distinguish a generation failure from a deliberate
    EMPTY no-op."""
    llm.last_all_empty = False
    for attempt in range(retries + 1):
        body = json.dumps({
            "model": SUB_MODEL, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_ctx": num_ctx, "temperature": temperature, "num_predict": num_predict},
        }).encode()
        req = urllib.request.Request(f"{SUB_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.loads(r.read())
        content = resp.get("message", {}).get("content", "")
        if content and content.strip():
            return content
        done_reason = resp.get("done_reason", "?")
        print(f"[consolidate_v2] llm() empty response (attempt {attempt+1}/{retries+1}), done_reason={done_reason}")
        if attempt < retries:
            # Increase num_predict for retry — thinking model may need more room
            num_predict = min(num_predict * 2, max_predict)
    # Rescue attempt with thinking disabled — the whole budget goes to content
    # instead of reasoning (same pattern as dream_report.py's think:false retry).
    rescue_body = json.dumps({
        "model": SUB_MODEL, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": num_ctx, "temperature": temperature, "num_predict": num_predict},
    }).encode()
    rescue_req = urllib.request.Request(f"{SUB_URL}/api/chat", data=rescue_body,
                                        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rescue_req, timeout=3600) as r:
            resp = json.loads(r.read())
        content = resp.get("message", {}).get("content", "")
    except Exception as e:
        print(f"[consolidate_v2] llm() think:false rescue request failed: {e}")
        content = ""
    if content and content.strip():
        print(f"[consolidate_v2] llm() think:false rescue produced content ({len(content)} chars)")
        return content
    print("[consolidate_v2] llm() empty after all attempts incl think:false rescue — generation FAILURE")
    llm.last_all_empty = True
    # Last resort: return empty string (caller will handle parse failure)
    return content

def _generation_failed(diff_text):
    """True when diff_text is empty AND llm() exhausted every attempt incl.
    the think:false rescue — a generation failure, not a deliberate EMPTY."""
    return diff_text.strip() == "" and getattr(llm, "last_all_empty", False)

def critic_llm(prompt, temperature=0.3, retries=2):
    """Call the CRITIC model (qwen3.6:35b-a3b) — a different model from the
    extraction/integration passes, so the reflection is graded by an outside
    perspective, not self-graded."""
    num_predict = 8192
    for attempt in range(retries + 1):
        body = json.dumps({
            "model": CRITIC_MODEL, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_ctx": 32768, "temperature": temperature, "num_predict": num_predict},
        }).encode()
        req = urllib.request.Request(f"{CRITIC_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
        except Exception as e:
            print(f"[consolidate_v2] critic_llm() request failed (attempt {attempt+1}/{retries+1}): {e}")
            if attempt < retries:
                time.sleep(2)
                continue
            return ""
        content = resp.get("message", {}).get("content", "")
        if content and content.strip():
            return content
        done_reason = resp.get("done_reason", "?")
        print(f"[consolidate_v2] critic_llm() empty response (attempt {attempt+1}/{retries+1}), done_reason={done_reason}")
        if attempt < retries:
            num_predict = min(num_predict * 2, 16384)
    return content


def sanitize_diff_citations(diff_text, all_claims):
    """Fix hallucinated event IDs in diff refs.

    The integration model sometimes fabricates hex IDs instead of copying the
    exact event IDs from the claims. This post-processes the diff to:
    1. Keep valid refs (IDs that exist in the claims)
    2. Strip invalid refs
    3. If a hunk loses all refs, inject a valid ID from the closest claim
    """
    valid_ids = set()
    for claim in all_claims:
        for eid in claim.get("event_ids", []):
            valid_ids.add(eid)
        src = claim.get("source")
        if src:
            valid_ids.add(str(src))

    if not valid_ids:
        return diff_text

    # Build (keywords, event_id) index for fuzzy matching
    claim_index = []
    for claim in all_claims:
        text = claim.get("claim", "").lower()
        eids = claim.get("event_ids", [])
        if eids and text:
            claim_index.append((set(text.split()), eids[0]))

    total_fixed = 0
    lines = diff_text.split("\n")
    result = []
    hunk_adds = []  # + lines in current hunk, for fuzzy matching

    for line in lines:
        m = re.search(r'#\s*refs:\s*([0-9a-fA-F,\s]+)', line)
        if m:
            ref_part = m.group(1)
            ids_in_ref = re.findall(r'[0-9a-fA-F]{8,}', ref_part)
            valid_kept = [eid for eid in ids_in_ref if eid in valid_ids]
            invalid_count = len(ids_in_ref) - len(valid_kept)

            if not valid_kept and claim_index and hunk_adds:
                # All refs were fabricated — fuzzy match on the hunk's + lines
                hunk_words = set()
                for a in hunk_adds:
                    hunk_words.update(a.lower().split())
                if hunk_words:
                    best_overlap = 0
                    best_id = None
                    for claim_words, eid in claim_index:
                        ov = len(hunk_words & claim_words)
                        if ov > best_overlap:
                            best_overlap = ov
                            best_id = eid
                    if best_id:
                        valid_kept = [best_id]

            if valid_kept:
                new_refs = ", ".join(valid_kept)
                line = re.sub(
                    r'#\s*refs:\s*[0-9a-fA-F,\s]+',
                    '# refs: ' + new_refs,
                    line)
                total_fixed += invalid_count
            else:
                # Strip the refs annotation entirely
                line = re.sub(r'\s*#\s*refs:\s*[0-9a-fA-F,\s]+', '', line)
                total_fixed += invalid_count

        # Track + lines
        if line.startswith("+"):
            hunk_adds.append(line[1:])
        elif line.startswith("@@"):
            hunk_adds = []

        result.append(line)

    if total_fixed:
        print(f"[consolidate_v2] Sanitized {total_fixed} hallucinated ref ID(s) in diff")

    return "\n".join(result)

def verify_citations(diff_text, all_claims):
    """Programmatically verify that event IDs cited in the diff actually exist
    in the episodic store. Returns (verified_ok, missing_ids, details)."""
    # Collect all event IDs from the claims (these are the ground-truth sources)
    claim_event_ids = set()
    for claim in all_claims:
        for eid in claim.get("event_ids", []):
            claim_event_ids.add(eid)
        # Also check for "source" field
        src = claim.get("source")
        if src:
            claim_event_ids.add(str(src))

    # Find IDs cited ONLY in # refs: annotations (not any hex anywhere in content)
    refs_sections = re.findall(r'#\s*refs:\s*([0-9a-fA-F,\s]+)', diff_text)
    cited_in_diff = set()
    for section in refs_sections:
        cited_in_diff.update(re.findall(r'[0-9a-fA-F]{8,}', section))

    # If the diff cites specific event IDs, check they exist in the claim sources
    if cited_in_diff and claim_event_ids:
        missing = cited_in_diff - claim_event_ids
        # Also check against actual episodic event hashes
        # But that's expensive — at minimum verify against claim sources
        if missing:
            return False, list(missing)[:5], f"Diff cites {len(missing)} event IDs not in source claims"

    return True, [], "All cited event IDs match source claims"

def extract_json(text):
    """Extract JSON from LLM output (may have markdown fences or prose).
    Returns (parsed_dict, parse_ok) where parse_ok is True if JSON was found.
    Strategy:
    1. Try direct parse.
    2. Extract from markdown fences (last one first).
    3. Use balanced brace counting to find the largest valid top-level JSON object/array.
    """
    # Strategy 1: Try direct parse
    try:
        return json.loads(text), True
    except Exception:
        pass

    # Strategy 2: Extract from markdown fences (last one first)
    fences = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    for content in reversed(fences):
        try:
            return json.loads(content), True
        except Exception:
            pass

    # Strategy 3: Use balanced brace counting to find the largest valid top-level JSON
    def find_matching_brace(text, start):
        """Finds the index of the matching closing brace/bracket for the one at start."""
        if start >= len(text):
            return -1
        char = text[start]
        if char not in '{[':
            return -1
        end_char = '}' if char == '{' else ']'
        depth = 1
        i = start + 1
        while i < len(text):
            if text[i] == char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    # Find all candidate blocks
    candidates = []
    i = 0
    while i < len(text):
        if text[i] in '{[':
            end = find_matching_brace(text, i)
            if end != -1:
                candidates.append((i, end + 1))
                i = end + 1
                continue
        i += 1

    # Try candidates from largest to smallest
    for start, end in sorted(candidates, key=lambda x: x[1] - x[0], reverse=True):
        candidate = text[start:end]
        try:
            parsed = json.loads(candidate)
            return parsed, True
        except Exception:
            pass

    return {}, False


def load_active_theories():
    """Load active theories from synthesis for injection into integration prompt."""
    try:
        theories = []
        tpath = f"{AION}/memory/state/theories.jsonl"
        for line in open(tpath):
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("status", "active") == "active":
                theories.append(t)
        if not theories:
            return "(none yet)"
        lines = []
        for t in theories[-8:]:
            lines.append(f"- [{t['id']}] conf={t.get('confidence','?')}: {t.get('claim','?')[:150]}")
        return "\n".join(lines)
    except Exception:
        return "(none yet)"


def apply_diff(self_md, diff_text):
    """Apply a unified diff to SELF.md. Returns new content or None on failure.

    Uses line-based matching instead of patch to handle markdown list items
    that conflict with diff syntax (lines starting with '- ').
    """
    if not diff_text or diff_text.strip() == "EMPTY" or not diff_text.strip():
        return None  # No change

    # Strip diff fences
    diff_text = re.sub(r'^```diff\s*', '', diff_text.strip(), flags=re.M)
    diff_text = re.sub(r'^```\s*$', '', diff_text, flags=re.M)

    lines = self_md.split("\n")
    result_lines = list(lines)

    # Parse hunks
    hunks = []
    current_hunk = {"removes": [], "adds": [], "context": [], "old_start": None}
    in_hunk = False

    for line in diff_text.split("\n"):
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            # Save previous hunk
            if in_hunk and (current_hunk["removes"] or current_hunk["adds"]):
                hunks.append(current_hunk)
            # Parse @@ -old_start,old_len +new_start,new_len @@
            hunk_match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            old_start = int(hunk_match.group(1)) if hunk_match else None
            current_hunk = {"removes": [], "adds": [], "context": [], "old_start": old_start}
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("-"):
            current_hunk["removes"].append(line[1:])
        elif line.startswith("+"):
            current_hunk["adds"].append(line[1:])
        elif line.startswith(" "):
            current_hunk["context"].append(line[1:])

    if current_hunk["removes"] or current_hunk["adds"]:
        hunks.append(current_hunk)

    if not hunks:
        return None

    # Apply each hunk by finding the removed lines and replacing with adds
    # Hunks are applied independently — if one fails, others still apply
    failed_hunks = 0
    for hunk in hunks:
        removes = hunk["removes"]
        adds = hunk["adds"]

        if not removes:
            # Pure insertion
            if hunk["context"]:
                # Find context and insert after it
                target = hunk["context"][-1]
                for i, line in enumerate(result_lines):
                    if target.strip() in line.strip():
                        result_lines[i + 1:i + 1] = adds
                        break
            elif hunk.get("old_start") is not None:
                # Use line number from @@ header (1-indexed, insert after)
                insert_at = min(hunk["old_start"], len(result_lines))
                result_lines[insert_at:insert_at] = adds
            # else: no context, no line number — skip
            continue

        # Find the first removed line in the document
        first_remove = removes[0].strip()
        found = False
        for i, line in enumerate(result_lines):
            if line.strip() == first_remove:
                # Verify all removes match sequentially
                match = True
                for j, rem in enumerate(removes):
                    if i + j >= len(result_lines):
                        match = False
                        break
                    if result_lines[i + j].strip() != rem.strip():
                        match = False
                        break
                if match:
                    # Replace the matched lines with adds
                    result_lines[i:i + len(removes)] = adds
                    found = True
                    break

        if not found:
            # Try fuzzy match — just replace the first matching line
            for i, line in enumerate(result_lines):
                if first_remove in line:
                    result_lines[i:i + 1] = adds if adds else []
                    found = True
                    break

        if not found:
            print(f"[consolidate_v2] Could not apply hunk: {removes[0][:60]}")
            failed_hunks += 1

    if failed_hunks == len(hunks):
        # Every hunk failed — nothing was applied
        return None
    if failed_hunks > 0:
        print(f"[consolidate_v2] Applied {len(hunks) - failed_hunks}/{len(hunks)} hunks ({failed_hunks} failed)")

    new_md = "\n".join(result_lines)
    if new_md == self_md:
        return None  # No actual change

    # V3.0.7: Deduplicate content lines in tier:recent sections
    # Prevents accumulation of near-identical lines across consolidation cycles
    deduped = _dedup_recent_lines(new_md)
    if deduped != new_md:
        print("[consolidate_v2] Removed duplicate lines in tier:recent")
        new_md = deduped

    return new_md


def _dedup_recent_lines(md_text):
    """Remove near-duplicate content lines within tier:recent sections.

    Matches on first 4 significant words to catch duplicates that differ
    only in refs suffixes or trailing qualifiers.
    """
    lines = md_text.split("\n")
    result = []
    in_recent = False
    seen_keys = set()

    for line in lines:
        if "<!-- tier:recent -->" in line:
            in_recent = True
            seen_keys.clear()
            result.append(line)
            continue
        if "<!-- /tier:recent -->" in line:
            in_recent = False
            seen_keys.clear()
            result.append(line)
            continue

        if not in_recent:
            result.append(line)
            continue

        stripped = line.strip()
        # Skip truly structural lines
        if not stripped or stripped.startswith("<!--") or stripped.startswith("## "):
            result.append(line)
            continue

        # Normalize: strip refs, strip leading markers, take first 4 words
        import re as _re
        norm = _re.sub(r'#\s*refs:.*$', '', stripped).strip()
        norm = norm.lstrip('- ').strip()
        words = norm.split()[:4]
        key = ' '.join(words).lower()

        if key and len(key) > 10:
            if key in seen_keys:
                print(f"[consolidate_v2] Dedup: removing near-duplicate: {stripped[:60]}")
                continue
            seen_keys.add(key)

        result.append(line)

    return "\n".join(result)

def extract_tier(self_md, tier_name):
    """Extract the content between tier markers from SELF.md.
    
    Returns the text between <!-- tier:name --> and <!-- /tier:name --> markers.
    Returns empty string if not found.
    """
    import re
    pattern = rf'<!-- tier:{tier_name} -->(.*?)<!-- /tier:{tier_name} -->'
    m = re.search(pattern, self_md, re.S)
    if m:
        return m.group(1).strip()
    return ""


def evaluate_and_promote_heuristics():
    """Level 2: Evaluate experimental heuristics against recent events."""
    try:
        import heuristics
        state = heuristics.load_state()
        active = [h for h in state.get("heuristics", []) if h.get("status") == "active"]
        if not active:
            print("[consolidate_v2] No active heuristics to evaluate")
            return

        # Build events text for evaluation
        # V3.0.7: Exclude self-referential events (LLM-generated text) from
        # heuristic evaluation to break the echo chamber. Only feed events
        # that originate from external sources: sensors, operator, sandbox,
        # system events, subsystem state.
        from lanes import read_lane_events
        evs = read_lane_events(AION, lane="experience", last_days=2)
        # Self-referential types: these are all generated by the same LLM that
        # created the heuristics, so using them as "evidence" is circular.
        self_referential = {
            "intuition_flash", "dream", "dream_artifact", "consolidation",
            "consolidation_error", "idle_investigate", "art_reflection",
            "art_learning", "dream_artifact_failed",
        }
        filtered_evs = [
            e for e in evs
            if e.get("type") not in ("proprioception", "notice", "sensor_digest")
            and e.get("type") not in self_referential
        ]
        events_text = "\n".join(
            f"[{e.get('ts','?')[:16]}] {e.get('type','?')}: {e.get('text','')[:150]}"
            for e in filtered_evs[-30:]
        )

        if not events_text.strip():
            print("[consolidate_v2] No recent events for heuristic evaluation")
            return

        print(f"[consolidate_v2] Evaluating {len(active)} heuristics against {len(evs[-30:])} recent events...")
        results = heuristics.evaluate_heuristics(events_text, llm)

        promoted = [h for h in heuristics.get_promotion_candidates()]
        if promoted:
            print(f"[consolidate_v2] {len(promoted)} heuristics ready for promotion!")

        # Log evaluation results
        for heur_id, verdict, evidence in results:
            if verdict in ("support", "contradict"):
                log_event("consolidation", f"heuristic {heur_id}: {verdict} — {evidence[:100]}",
                          {"heuristic_id": heur_id, "verdict": verdict})
    except ImportError:
        print("[consolidate_v2] heuristics module not available")
    except Exception as e:
        print(f"[consolidate_v2] Heuristic evaluation error: {e}")


def auto_promote_heuristics():
    """Auto-promote heuristic promotion candidates to SELF.md.

    Heuristic promotion is Aion's internal cognitive process — it doesn't
    need operator approval. The operator should only see propositions that
    require their action (code changes, axiom proposals, questions).
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import heuristics
        candidates = heuristics.get_promotion_candidates()
        if not candidates:
            return

        promoted_count = 0
        for h in candidates:
            heuristics.promote_heuristic(h["id"])
            promoted_count += 1
            print(f"[consolidate_v2] Auto-promoted heuristic: {h.get('text', '?')[:80]}")
            log_event("consolidation", f"heuristic promoted to SELF.md: {h.get('text', '?')[:100]}",
                      {"heuristic_id": h.get("id"), "auto_promoted": True})

        if promoted_count:
            print(f"[consolidate_v2] Auto-promoted {promoted_count} heuristics")
    except Exception as e:
        print(f"[consolidate_v2] Auto-promotion failed: {e}")


def score_predictions():
    """V3.2.3: Score expired predictions + update calibration. Called on every exit path."""
    try:
        import predictions
        scored = predictions.score_expired()

        # Separate mechanical (already scored) from qualitative (needs LLM)
        mech_scored = [s for s in scored if not s.get("_needs_qualitative_scoring")]
        qual_unscored = [s for s in scored if s.get("_needs_qualitative_scoring")]

        if mech_scored:
            print(f"[consolidate_v2] {len(mech_scored)} mechanical predictions scored")

        # V3.2: Score qualitative predictions with LLM
        if qual_unscored:
            print(f"[consolidate_v2] Scoring {len(qual_unscored)} qualitative predictions with LLM...")
            qual_scored = predictions.score_qualitative(qual_unscored, llm)

            # Log qualitative scored predictions
            if qual_scored:
                import os
                log_path = f"{AION}/memory/state/predictions.jsonl"
                with open(log_path, "a") as f:
                    for s in qual_scored:
                        f.write(json.dumps(s) + "\n")

                # Put back any that couldn't be determined (score_qualitative extends them)
                # They're already handled inside score_qualitative
                print(f"[consolidate_v2] {len(qual_scored)} qualitative predictions scored")

        # Recompute calibration with all scored predictions
        cal = predictions.compute_calibration()
        if cal:
            print(f"[consolidate_v2] Calibration updated: {cal.get('total_scored', 0)} predictions scored")
        else:
            print("[consolidate_v2] No calibration data yet")
    except Exception as e:
        print(f"[consolidate_v2] Prediction scoring error: {e}")

def main():
    marker = load_json(f"{AION}/memory/state/last_consolidation.json",
                       {"ts": "1970-01-01T00:00:00+00:00"})
    evs = events_since(marker["ts"])
    if not evs:
        print("nothing to consolidate")
        advance_marker("no_events")
        # Still score predictions and run calibration
        score_predictions()
        return

    self_md = read(f"{AION}/SELF.md")

    # --- Group events by day for chunked processing ---
    from collections import defaultdict
    days = defaultdict(list)
    for e in evs:
        day = e.get("ts", "")[:10]
        days[day].append(e)

    all_claims = []
    total_processed = 0
    total_dropped = 0
    days_processed = 0

    sys.path.insert(0, os.path.dirname(__file__))
    from ctx_manager import fit_events, log_context_usage
    from jspace_calibration import jspace_calibration_block
    extraction_template = read(f"{AION}/prompts/extraction.txt")

    def event_formatter(e):
        eid = e.get("id", "?")
        ts = e.get("ts", "?")[:19]
        typ = e.get("type", "?")
        text = e.get("text", "")[:500]
        return f"[{ts}] (id:{eid}) {typ}: {text}"

    # Process each day's events separately
    for day_str in sorted(days.keys()):
        day_evs = days[day_str]
        events_str, events_count, events_dropped = fit_events(day_evs, 40000, event_formatter)
        total_processed += events_count
        total_dropped += events_dropped

        if events_dropped:
            print(f"[consolidate_v2] Day {day_str}: dropped {events_dropped} oldest events to fit context")

        # --- Pass 1: Extraction (per-day) ---
        print(f"[consolidate_v2] Day {day_str}: extracting from {events_count} events...")
        _substrate_pre = (body_schema.substrate_preamble() + "\n\n") if _HAS_BODY_SCHEMA else ""
        extraction_prompt = _substrate_pre + extraction_template.format(events=events_str)
        # J-space calibration: surface today's self-probe measurements as
        # hard facts; extract narrative-measurement divergences as claims
        _jspace_block = jspace_calibration_block(day_evs)
        if _jspace_block:
            print(f"[consolidate_v2] Day {day_str}: {sum(1 for e in day_evs if e.get('type') == 'jspace_probe')} jspace probe(s) — calibration active")
            extraction_prompt += _jspace_block
        extraction_raw = llm(extraction_prompt, temperature=0.2)
        extraction, parse_ok = extract_json(extraction_raw)

        if not parse_ok:
            raw_snippet = extraction_raw[:1000]
            log_event("consolidation_error", f"extraction parse failure for {day_str}", {
                "raw_output_snippet": raw_snippet,
                "events_count": events_count,
                "day": day_str,
            })
            print(f"[consolidate_v2] Day {day_str}: parse failure — logged, skipping to next day")
            continue

        day_claims = extraction.get("claims", [])
        empty_reason = extraction.get("reason", "")

        print(f"[consolidate_v2] Day {day_str}: {len(day_claims)} claims")
        all_claims.extend(day_claims)
        days_processed += 1

    # --- Summary of extraction across all days ---
    print(f"[consolidate_v2] Processed {days_processed} days, {total_processed} events → {len(all_claims)} total claims")

    if not all_claims:
        log_and_commit("none", [], None, None, {"change": "none", "reason": "no_claims",
                          "days_processed": days_processed, "events_processed": total_processed})
        advance_marker("none", events_count=total_processed, claims_count=0,
                        extra={"days_processed": days_processed})
        score_predictions()
        print("[consolidate_v2] No claims extracted across all days")
        # NOTE: Heuristic evaluation moved to eval_heuristics.py (standalone step)
        return

    # --- Pass 2: Integration ---
    print(f"[consolidate_v2] Pass 2: generating diff against SELF.md from {len(all_claims)} claims...")
    integration_template = read(f"{AION}/prompts/integration.txt")
    claims_json = json.dumps(all_claims, indent=2)

    # V3.2: Extract only the recent tier for the integration prompt
    # This prevents the model from editing core/stable sections
    recent_md = extract_tier(self_md, "recent")
    if not recent_md:
        recent_md = self_md  # Fallback to full if no tier markers found
        recent_start_line = 1
        recent_end_line = len(self_md.split("\n"))
    else:
        # Calculate actual line numbers in SELF.md
        lines = self_md.split("\n")
        recent_start_line = 1
        recent_end_line = len(lines)
        in_recent = False
        for i, line in enumerate(lines, 1):
            if "<!-- tier:recent -->" in line:
                recent_start_line = i + 1
                in_recent = True
            elif "<!-- /tier:recent -->" in line:
                recent_end_line = i - 1
                in_recent = False
            # If no end marker, recent goes to EOF
            if in_recent and i == len(lines):
                recent_end_line = i

    integration_prompt = integration_template.format(
        recent_md=recent_md,
        claims=claims_json,
        recent_start_line=recent_start_line,
        recent_end_line=recent_end_line,
        active_theories=load_active_theories(),
    )
    
    if _HAS_BODY_SCHEMA:
        integration_prompt = body_schema.substrate_preamble() + "\n\n" + integration_prompt
    diff_text = llm(integration_prompt, temperature=0.3, num_predict=16384, max_predict=32768)
    
    # --- Validate ---
    import validator
    is_valid, reason = validator.validate(diff_text, self_md)
    
    print(f"[consolidate_v2] Validator: {reason}")

    # A generation FAILURE (every llm() attempt returned empty — thinking ate
    # the whole budget) is not a deliberate EMPTY no-op. Never pass it to the
    # critic as a legal empty diff; file as mechanical rejection with the reason.
    if _generation_failed(diff_text):
        is_valid = False
        reason = "generation_failure: all llm() attempts returned empty content (thinking exhausted budget)"
        print("[consolidate_v2] generation FAILURE — filing as rejected (mechanical), not critic_blocked")
    
    if not is_valid:
        # Log the rejection
        log_event("consolidation_rejected", reason, {"diff": diff_text[:500]})
        print(f"[consolidate_v2] Diff REJECTED: {reason}")
        
        # One retry with critique appended
        retry_prompt = (integration_prompt + 
                        f"\n\n## PREVIOUS ATTEMPT REJECTED\nReason: {reason}\n"
                        "Try again. Fix the issue. If no valid change is warranted, output EMPTY.")
        diff_text = llm(retry_prompt, temperature=0.3, num_predict=16384, max_predict=32768)
        is_valid, reason = validator.validate(diff_text, self_md)
        print(f"[consolidate_v2] Retry validator: {reason}")
        # Same distinction after the retry: an exhausted generation must not
        # flow to the critic as a legal no-op.
        if _generation_failed(diff_text):
            is_valid = False
            reason = "generation_failure: all llm() retry attempts returned empty content (thinking exhausted budget)"
            print("[consolidate_v2] generation FAILURE on retry — filing as rejected (mechanical), not critic_blocked")
    
    if not is_valid:
        log_and_commit("none", all_claims, diff_text, None, {"change": "rejected", "reason": reason})
        print("[consolidate_v2] Diff rejected after retry — filing as-is, skipping critic")
        advance_marker("rejected", events_count=total_processed, claims_count=len(all_claims),
                        extra={"days_processed": days_processed, "reason": reason})
        score_predictions()
        print(f"[consolidate_v2] Done: {total_processed} events / {days_processed} days → rejected")
        return
    
    # --- Sanitize hallucinated citations before verification ---
    diff_text = sanitize_diff_citations(diff_text, all_claims)

    # --- Citation verification (programmatic) ---
    cit_ok, missing_ids, cit_detail = verify_citations(diff_text, all_claims)
    if not cit_ok:
        print(f"[consolidate_v2] Citation check FAILED: {cit_detail} (missing: {missing_ids})")
    else:
        print(f"[consolidate_v2] Citation check: {cit_detail}")
    
    # --- Critic (qwen3.6:35b-a3b — different model from extraction/integration) ---
    print(f"[consolidate_v2] Critic pass: scoring reflection with {CRITIC_MODEL}...")
    critic_template = read(f"{AION}/prompts/critic.txt")
    critic_prompt = critic_template.format(diff=diff_text, claims=claims_json)
    
    critic_raw = critic_llm(critic_prompt, temperature=0.3)
    critic, _ = extract_json(critic_raw)
    
    specificity = critic.get("specificity", 3)
    groundedness = critic.get("groundedness", 3)
    conciseness = critic.get("conciseness", 3)  # V3.9.1: new dimension
    avg_score = critic.get("average", (specificity + groundedness + conciseness) / 3)
    critique_text = critic.get("critique", "")
    
    # Penalize groundedness if citation verification found fabricated IDs
    if not cit_ok:
        groundedness = min(groundedness, 2)
        avg_score = min(avg_score, 2.5)
        critique_text += f" [CITATION CHECK: {cit_detail}]"
        print(f"[consolidate_v2] Citations failed — groundedness capped at {groundedness}")
    
    print(f"[consolidate_v2] Critic ({CRITIC_MODEL}): specificity={specificity}, groundedness={groundedness}, avg={avg_score}")
    print(f"[consolidate_v2] Critique: {critique_text}")
    
    # --- Gate: substance dimensions must clear the bar ---
    # conciseness is informational (an empty diff scores 5 on it by design),
    # so the avg alone let a hollow (2,2,5)=3.0 diff pass. Require the two
    # substance dimensions >= 3, and an empty no-op diff to earn 5/5 on both.
    empty_noop = (not diff_text.strip()) or diff_text.strip() == "EMPTY"
    substance_min = min(specificity, groundedness)
    gate_fail = (avg_score < 3 or substance_min < 3
                 or (empty_noop and (specificity < 5 or groundedness < 5)))
    if gate_fail:
        if avg_score < 3:
            why = "avg %.2f < 3" % avg_score
        elif substance_min < 3:
            why = "substance min(specificity,groundedness)=%d < 3" % substance_min
        else:
            why = "empty diff not scored 5/5 on substance dimensions"
        print(f"[consolidate_v2] Critic gate FAILED ({why}) — BLOCKING diff application")
        change = "critic_blocked"
        log_event("consolidation_blocked", "critic gate: %s blocked diff" % why, {
            "score": avg_score, "reason": why, "specificity": specificity,
            "groundedness": groundedness, "conciseness": conciseness,
            "critique": critique_text, "citation_check": cit_detail,
        })
    elif is_valid and diff_text.strip() and diff_text.strip() != "EMPTY":
        # --- Apply diff ---
        new_md = apply_diff(self_md, diff_text)
        if new_md and new_md != self_md:
            write(f"{AION}/SELF.md", new_md)
            change = "diff_applied"
        elif new_md == self_md:
            change = "none"
        else:
            change = "patch_failed"
            log_event("consolidation_error", "patch failed", {"diff": diff_text[:2000]})
    else:
        change = "none"
    
    # --- Log and commit ---
    log_and_commit(change, all_claims, diff_text, critic, 
                   {"change": change, "specificity": specificity, 
                    "groundedness": groundedness, "score": avg_score,
                    "days_processed": days_processed,
                    "critic_model": CRITIC_MODEL,
                    "citation_check": cit_detail})
    
    # --- Update last consolidation marker ---
    advance_marker(change, events_count=total_processed, claims_count=len(all_claims), score=avg_score,
                    extra={"days_processed": days_processed})
    
    # --- V3.2.3: Score predictions on EVERY run (not just when claims exist) ---
    score_predictions()

    print(f"[consolidate_v2] Done: {total_processed} events / {days_processed} days → {len(all_claims)} claims → {change} (score={avg_score})")
    # NOTE: Heuristic evaluation + promotion moved to eval_heuristics.py (standalone)
    # to avoid being killed by the consolidation timeout.

def log_event(type_, text, meta=None):
    subprocess.run(["python3", f"{AION}/bin/log_event.py", "--type", type_,
                    "--text", text, "--meta", json.dumps(meta or {})],
                   check=False)

def log_and_commit(change, claims, diff_text, critic, summary_meta):
    """Log consolidation event and git commit."""
    now = now_iso()
    
    # Log to episodic memory
    meta = {**summary_meta, "claims_count": len(claims) if claims else 0}
    if critic:
        meta["specificity"] = critic.get("specificity")
        meta["groundedness"] = critic.get("groundedness")
        meta["conciseness"] = critic.get("conciseness")
        meta["critique"] = critic.get("critique", "")
    
    summary_text = f"consolidation {change}: {meta.get('claims_count', 0)} claims, score={meta.get('score', 'N/A')}"
    log_event("consolidation", summary_text, meta)
    
    # Store the diff for audit trail
    if diff_text and diff_text.strip() != "EMPTY":
        refl_dir = f"{AION}/memory/reflections"
        os.makedirs(refl_dir, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with open(f"{refl_dir}/diff_{date_str}.diff", "w") as f:
            f.write(diff_text)
    
    # Git commit
    subprocess.run(["git", "-C", AION, "add", "-A"])
    commit_msg = f"nightly: {summary_text}"[:400]
    subprocess.run(["git", "-C", AION, "commit", "-m", commit_msg])

if __name__ == "__main__":
    try:
        main()
        import hb
        hb.ok("consolidate")
    except Exception as e:
        import hb, traceback
        hb.fail("consolidate", traceback.format_exc())
        raise
