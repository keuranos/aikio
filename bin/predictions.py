#!/usr/bin/env python3
"""predictions.py — R5.1/R5.2: Prediction lifecycle.

Schema for mechanical predictions (~70%):
  {id, made_ts, resolve_ts, variable, operator, threshold, confidence, raw_confidence, calibrated_confidence, type: "mechanical"}

Schema for qualitative predictions (~30%):
  {id, made_ts, resolve_ts, statement, confidence, raw_confidence, calibrated_confidence, type: "qualitative", event_ids: [...]}

Scorer runs inside nightly consolidation for anything past resolve_ts.
Brier score per prediction. Monthly calibration table in calibration.json.
"""
import json, os, time
from datetime import datetime, timezone

AION = os.environ.get("AION_HOME", "$AION_HOME")
PREDICTIONS_OPEN = f"{AION}/memory/state/predictions_open.json"
PREDICTIONS_LOG = f"{AION}/memory/state/predictions.jsonl"
CALIBRATION_FILE = f"{AION}/memory/state/calibration.json"

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_open():
    try:
        with open(PREDICTIONS_OPEN) as f:
            return json.load(f)
    except Exception:
        return {"open": []}

def save_open(data):
    with open(PREDICTIONS_OPEN, "w") as f:
        json.dump(data, f, indent=2)

def add_mechanical(variable, operator, threshold, resolve_ts, confidence):
    """Add a mechanical prediction.
    
    operator: "gt", "lt", "eq", "gte", "lte"
    Example: add_mechanical("server_temp", "gt", 50.0, "2026-07-12T00:00:00+00:00", 0.8)
    """
    data = load_open()
    pred = {
        "id": f"pred_{int(time.time()*1000)}_{len(data.get('open',[]))}",
        "type": "mechanical",
        "made_ts": now_iso(),
        "resolve_ts": resolve_ts,
        "variable": variable,
        "operator": operator,
        "threshold": threshold,
        "confidence": confidence,
        "raw_confidence": confidence,
        "calibrated_confidence": calibrated_confidence(confidence),
    }
    data.setdefault("open", []).append(pred)
    save_open(data)
    return pred

def add_qualitative(statement, resolve_ts, confidence, event_ids=None):
    """Add a qualitative prediction."""
    data = load_open()
    pred = {
        "id": f"pred_{int(time.time()*1000)}_{len(data.get('open',[]))}",
        "type": "qualitative",
        "made_ts": now_iso(),
        "resolve_ts": resolve_ts,
        "statement": statement,
        "confidence": confidence,
        "raw_confidence": confidence,
        "calibrated_confidence": calibrated_confidence(confidence),
        "event_ids": event_ids or [],
    }
    data.setdefault("open", []).append(pred)
    save_open(data)
    return pred

def read_variable(variable):
    """Read the current value of a mechanical prediction variable from env_history."""
    history_path = f"{AION}/memory/state/env_history.jsonl"
    try:
        with open(history_path) as f:
            lines = f.readlines()
        if not lines:
            return None
        last = json.loads(lines[-1])
        # V3.2.2: env_history stores flat keys (not wrapped in raw/derived)
        return last.get(variable) or last.get("raw", {}).get(variable) or last.get("derived", {}).get(variable)
    except Exception:
        return None

def evaluate_mechanical(pred):
    """Evaluate a mechanical prediction. Returns True/False/None (unresolvable)."""
    val = read_variable(pred["variable"])
    if val is None:
        return None
    
    op = pred["operator"]
    thresh = pred["threshold"]
    
    if op == "gt":
        return val > thresh
    elif op == "lt":
        return val < thresh
    elif op == "eq":
        return val == thresh
    elif op == "gte":
        return val >= thresh
    elif op == "lte":
        return val <= thresh
    return None

def brier_score(confidence, outcome):
    """Brier score: lower is better. 0 = perfect, 1 = worst.
    confidence: 0-1 probability assigned
    outcome: True/False
    """
    actual = 1.0 if outcome else 0.0
    return (confidence - actual) ** 2


def run_eci_calibration():
    """Run CalFram ECI decomposition on resolved predictions.
    
    Called after score_expired() to augment calibration.json with:
      - ECIb (Balance): SIGNED over/under-confidence measure
      - ECIL (Local): per-category calibration breakdown
      - Adaptive bin count via monotonic sweep
    
    This fills the gap where Brier only gives magnitude (HOW MUCH wrong)
    without direction (over vs under) or location (per-category).
    
    Returns the ECI result dict, or None if insufficient data.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from calibration_eci import run_eci_decomposition
        return run_eci_decomposition()
    except ImportError:
        # calibration_eci.py not yet available — fall back to basic stats
        return {"status": "eci_module_missing", "note": "bin/calibration_eci.py not found"}
    except Exception as e:
        return {"status": "eci_error", "error": str(e)}

def score_expired():
    """Score all predictions past their resolve_ts. Returns scored list."""
    data = load_open()
    now = now_iso()
    still_open = []
    scored = []
    abandoned = []
    
    for pred in data.get("open", []):
        # V3.0.4: Type guard — skip non-dict entries (pre-v2 legacy format)
        if not isinstance(pred, dict):
            abandoned.append(str(pred)[:200])
            continue
        
        # V3.0.4: Guard against missing resolve_ts
        resolve_ts = pred.get("resolve_ts")
        if not resolve_ts:
            abandoned.append(f"missing resolve_ts: {pred.get('id', '?')}")
            continue
        
        if resolve_ts > now:
            still_open.append(pred)
            continue
        
        # Past resolve_ts — score it
        if pred["type"] == "mechanical":
            outcome = evaluate_mechanical(pred)
            if outcome is None:
                # Can't resolve — extend by 24h, capped at MAX_RETRIES
                from datetime import timedelta
                retry_count = pred.get("retry_count", 0)
                MAX_RETRIES = 7  # 7 days of retries before abandoning
                if retry_count >= MAX_RETRIES:
                    # Abandon: variable has been unresolvable for too long
                    abandoned.append(
                        f"unresolvable after {retry_count} retries: "
                        f"{pred.get('id','?')} variable={pred.get('variable','?')}"
                    )
                    continue
                pred["retry_count"] = retry_count + 1
                pred["last_retry_ts"] = now
                resolve_dt = datetime.fromisoformat(pred["resolve_ts"])
                pred["resolve_ts"] = (resolve_dt + timedelta(days=1)).isoformat()
                still_open.append(pred)
                continue
            
            bs = brier_score(pred["confidence"], outcome)
            scored.append({
                **pred,
                "outcome": outcome,
                "brier": round(bs, 4),
                "scored_ts": now,
            })
        else:
            # Qualitative — needs LLM scoring
            scored.append({**pred, "_needs_qualitative_scoring": True})

    # Save remaining open (including unresolved qualitative that got extended)
    # Note: qualitative preds that need LLM scoring are in 'scored' list,
    # not in still_open. They'll be scored by score_qualitative() or
    # put back if undeterminable.
    save_open({"open": still_open})

    # V3.0.4: Log abandoned legacy entries
    if abandoned:
        import subprocess
        subprocess.run(
            ["python3", f"{AION}/bin/log_event.py", "--type", "predictions_abandoned",
             "--text", f"Abandoned {len(abandoned)} legacy prediction(s)",
             "--meta", json.dumps({"abandoned": abandoned})],
            check=False,
        )
        print(f"[predictions] Abandoned {len(abandoned)} legacy entry(ies): {abandoned}")

    # Return both mechanical scored and qualitative unscored
    # Mechanical are already logged to PREDICTIONS_LOG
    # Qualitative need LLM scoring via score_qualitative()
    mech_scored = [s for s in scored if not s.get("_needs_qualitative_scoring")]
    qual_unscored = [s for s in scored if s.get("_needs_qualitative_scoring")]

    # Log mechanical scored predictions
    with open(PREDICTIONS_LOG, "a") as f:
        for s in mech_scored:
            f.write(json.dumps(s) + "\n")

    return mech_scored + qual_unscored


def score_qualitative(unscored, llm_fn):
    """Score qualitative predictions using an LLM.

    Args:
        unscored: list of prediction dicts with _needs_qualitative_scoring=True
        llm_fn: callable(prompt) -> str, calls the subconscious model

    Returns list of scored predictions with outcome and brier.
    """
    if not unscored:
        return []

    import glob, os

    # Gather recent episodic events for context (last 3 days, experience lane)
    events_text = []
    for path in sorted(glob.glob(f"{AION}/memory/episodic/*.jsonl"))[-3:]:
        if "/telemetry/" in path:
            continue
        for line in open(path, encoding="utf-8"):
            try:
                ev = json.loads(line)
                # Skip telemetry types
                if ev.get("type") in ("proprioception", "notice", "sensor_digest",
                                      "subsystem_dead", "heartbeat"):
                    continue
                events_text.append(
                    f"[{ev.get('ts','?')[:16]}] {ev.get('type','?')}: {ev.get('text','')[:150]}"
                )
            except Exception:
                pass

    events_summary = "\n".join(events_text[-50:])  # Last 50 experience events

    scored = []
    for pred in unscored:
        statement = pred.get("statement", "")
        made_ts = pred.get("made_ts", "")[:16]
        resolve_ts = pred.get("resolve_ts", "")[:16]

        prompt = f"""You are evaluating whether a prediction came true.

## Prediction (made {made_ts}, resolved {resolve_ts})
"{statement}"

## Recent events from the prediction's time window
{events_summary}

## Your task
Determine whether this prediction came true based on the events above.
Answer with ONLY a JSON object:

{{"outcome": true}}  if the prediction came true
{{"outcome": false}}  if the prediction did not come true
{{"outcome": null}}  if you cannot determine it from the available evidence

Return ONLY the JSON, nothing else."""

        try:
            raw = llm_fn(prompt)
            # Parse JSON
            import re
            m = re.search(r'\{[^}]*\}', raw, re.S)
            if m:
                result = json.loads(m.group(0))
            else:
                result = json.loads(raw)

            outcome = result.get("outcome")
            if outcome is None:
                # Can't determine — extend by 24h
                from datetime import timedelta
                resolve_dt = datetime.fromisoformat(pred["resolve_ts"])
                pred["resolve_ts"] = (resolve_dt + timedelta(days=1)).isoformat()
                # Put back in open queue
                continue

            bs = brier_score(pred["confidence"], outcome)
            scored_pred = {**pred}
            del scored_pred["_needs_qualitative_scoring"]
            scored_pred["outcome"] = outcome
            scored_pred["brier"] = round(bs, 4)
            scored_pred["scored_ts"] = now_iso()
            scored.append(scored_pred)
            print(f"[predictions] Qualitative scored: '{statement[:60]}' -> {outcome} (brier={bs:.3f})")

        except Exception as e:
            print(f"[predictions] Qualitative scoring failed for '{statement[:60]}': {e}")
            # Put back with extended deadline
            from datetime import timedelta
            resolve_dt = datetime.fromisoformat(pred["resolve_ts"])
            pred["resolve_ts"] = (resolve_dt + timedelta(days=1)).isoformat()
            continue

    return scored

def _isotonic_fit(points):
    """Fit isotonic regression (monotonic non-decreasing) using PAV algorithm.
    
    Args:
        points: list of (x, w) tuples — (confidence, outcome_bool) sorted by x.
    Returns:
        dict mapping original bucket -> calibrated rate (0-1).
    
    V3.9.1: Bucket-level PAV with MIN_BUCKET_N guard. Buckets below the
    minimum are excluded from the fit entirely — their calibrated_rate
    falls back to their raw true_rate (not a pooled neighbor's rate).
    """
    MIN_BUCKET_N = 5
    
    from collections import defaultdict
    bucket_outcomes = defaultdict(list)
    for conf, outcome in points:
        bucket = round(conf * 20) / 20
        bucket_outcomes[bucket].append(outcome)
    
    # Build bucket-level points: only buckets with enough data
    bucket_points = []
    for bucket in sorted(bucket_outcomes.keys()):
        outcomes = bucket_outcomes[bucket]
        n = len(outcomes)
        if n < MIN_BUCKET_N:
            continue
        rate = sum(1 for o in outcomes if o) / n
        bucket_points.append((bucket, rate, n))
    
    if len(bucket_points) < 2:
        return {}
    
    # PAV on bucket-level rates, weighted by bucket size
    blocks = []
    for bp in bucket_points:
        n = bp[2]
        true_count = sum(1 for o in bucket_outcomes[bp[0]] if o)
        rate = true_count / n
        blocks.append([true_count, n, rate])
    
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i+1][2]:
            merged_true = blocks[i][0] + blocks[i+1][0]
            merged_n = blocks[i][1] + blocks[i+1][1]
            merged = [merged_true, merged_n, merged_true / merged_n if merged_n > 0 else 0]
            blocks[i] = merged
            del blocks[i+1]
            if i > 0:
                i -= 1
        else:
            i += 1
    
    fitted = {}
    idx = 0
    for _, cnt, val in blocks:
        for _ in range(int(cnt)):
            if idx < len(bucket_points):
                fitted[bucket_points[idx][0]] = val
                idx += 1
    
    return fitted


def compute_calibration():
    """Compute calibration table from scored predictions.
    
    Now includes isotonic-regression-corrected confidence for each bucket.
    The 'calibrated_rate' field shows the empirically corrected true probability
    for that confidence level, accounting for miscalibration via PAV.
    """
    try:
        with open(PREDICTIONS_LOG) as f:
            scored = [json.loads(l) for l in f if l.strip()]
    except Exception:
        scored = []
    
    if not scored:
        return None
    
    # Build (confidence, outcome) pairs for isotonic regression
    iso_points = []
    for s in scored:
        conf = s.get("confidence", 0.5)
        outcome = bool(s.get("outcome", False))
        iso_points.append((conf, outcome))
    
    # Sort by confidence for PAV
    iso_points.sort(key=lambda p: p[0])
    
    # Fit isotonic regression
    iso_fitted = _isotonic_fit(iso_points) if len(iso_points) >= 10 else {}
    
    # Group by confidence bucket
    buckets = {}
    for s in scored:
        conf = s.get("confidence", 0.5)
        bucket = round(conf * 20) / 20  # Round to nearest 0.05
        buckets.setdefault(bucket, []).append(s.get("outcome", False))
    
    calibration = {}
    for bucket, outcomes in sorted(buckets.items()):
        n = len(outcomes)
        true_rate = sum(1 for o in outcomes if o) / n
        # Get isotonic-corrected rate for this bucket
        # Use median confidence within bucket as lookup key
        bucket_conf = bucket
        calibrated_rate = iso_fitted.get(bucket_conf, true_rate)
        calibration[f"{bucket:.2f}"] = {
            "n": n,
            "true_rate": round(true_rate * 100, 1),
            "calibrated_rate": round(calibrated_rate * 100, 1),
            "mean_brier": round(sum((bucket - (1 if o else 0))**2 for o in outcomes) / n, 4),
            "calibrated_brier": round(sum((calibrated_rate - (1 if o else 0))**2 for o in outcomes) / n, 4),
            **({"low_n_warning": True} if n < 5 else {}),  # V3.9.1: flag sparse buckets
        }
    
    # Compute overall Brier improvement from calibration
    total_raw_brier = sum((p[0] - (1 if p[1] else 0))**2 for p in iso_points) / len(iso_points)
    total_cal_brier = None
    if iso_fitted:
        total_cal_brier = sum((iso_fitted.get(p[0], p[0]) - (1 if p[1] else 0))**2 for p in iso_points) / len(iso_points)
    
    # V3.9.1: Only use isotonic calibration if it actually improves Brier score.
    # If the data is too noisy/non-monotonic, PAV over-flattens and makes
    # calibration WORSE. In that case, fall back to raw true_rates per bucket.
    if total_cal_brier is not None and total_cal_brier >= total_raw_brier:
        print(f"[predictions] Isotonic calibration made Brier WORSE "
              f"({total_raw_brier:.4f} → {total_cal_brier:.4f}), falling back to raw true_rates")
        iso_fitted = {}
        total_cal_brier = None
        # Re-compute calibration entries without isotonic correction
        for bk, bv in calibration.items():
            bv["calibrated_rate"] = bv["true_rate"]
            bv["calibrated_brier"] = bv["mean_brier"]
    
    result = {
        "ts": now_iso(),
        "total_scored": len(scored),
        "buckets": calibration,
        "isotonic_fit": bool(iso_fitted),
        "raw_brier": round(total_raw_brier, 4),
        "calibrated_brier": round(total_cal_brier, 4) if total_cal_brier is not None else None,
    }
    
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(result, f, indent=2)
    
    if total_cal_brier is not None and total_cal_brier < total_raw_brier:
        improvement = (1 - total_cal_brier / total_raw_brier) * 100
        print(f"[predictions] Isotonic calibration: Brier {total_raw_brier:.4f} → {total_cal_brier:.4f} ({improvement:.1f}% improvement)")
    
    return result


def calibrated_confidence(raw_confidence):
    """Look up the isotonic-regression-corrected confidence for a raw value.
    
    Falls back to the raw value if no calibration data exists.
    Used by heuristic promotion to avoid miscalibrated confidence inflating scores.
    """
    try:
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
    except Exception:
        return raw_confidence
    
    buckets = cal.get("buckets", {})
    if not buckets:
        return raw_confidence
    
    bucket_key = f"{round(raw_confidence * 20) / 20:.2f}"
    entry = buckets.get(bucket_key)
    if entry and "calibrated_rate" in entry:
        return entry["calibrated_rate"] / 100.0
    return raw_confidence


def get_calibration_summary():
    """Get calibration summary for SYSTEM_PROMPT injection.
    
    Reports both raw and isotonic-calibrated accuracy for key buckets,
    so Aion's self-model is aware of its own miscalibration.
    """
    try:
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
    except Exception:
        return ""
    
    buckets = cal.get("buckets", {})
    parts = []
    
    # Report the most miscalibrated buckets
    for bk in ["0.6", "0.7", "0.9"]:
        entry = buckets.get(bk)
        if not entry:
            continue
        raw = entry.get("true_rate", "?")
        cal_rate = entry.get("calibrated_rate", raw)
        if raw != cal_rate:
            parts.append(f"{bk}-conf→{raw}% actual ({cal_rate}% calibrated)")
        else:
            parts.append(f"{bk}-conf→{raw}% actual")
    
    if parts:
        return "Prediction calibration: " + ", ".join(parts)
    
    # Fallback
    b9 = buckets.get("0.9", {})
    if b9:
        return f"Your 0.9-confidence predictions resolve true {b9.get('true_rate', 'N/A')}% of the time"
    
    return ""
