#!/usr/bin/env python3
"""homeostasis.py — check internal variables; if out of band, Aion wakes itself.

Run every 30 min by aion-homeostasis.timer. Writes memory/state/intents.json.
If any variable is out of band, invokes wake.py --self with the intent.
"""
import json, os, subprocess, glob, time
from datetime import datetime, timezone
import aion_env  # loads config/aion.env

AION = os.environ.get("AION_HOME", "$AION_HOME")


def _parse_iso(ts):
    """Parse ISO timestamp, handling Z suffix (Python <3.11 compat)."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def env(k, d):
    return os.environ.get(k, d)

def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def count_unconsolidated():
    """V3.1.4: Count only experience-lane events (not telemetry noise)."""
    marker = load_json(f"{AION}/memory/state/last_consolidation.json",
                       {"ts": "1970-01-01T00:00:00+00:00"})
    last = _parse_iso(marker["ts"])
    n = 0
    for path in sorted(glob.glob(f"{AION}/memory/episodic/*.jsonl"))[-3:]:
        # V3.1.4: Skip telemetry lane
        if "/telemetry/" in path:
            continue
        with open(path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if _parse_iso(ev["ts"]) > last:
                        n += 1
                except Exception:
                    pass
    return n

def hours_since_last_event():
    files = sorted(glob.glob(f"{AION}/memory/episodic/*.jsonl"))
    if not files:
        return 9999
    with open(files[-1]) as f:
        lines = f.readlines()
    if not lines:
        return 9999
    ts = _parse_iso(json.loads(lines[-1])["ts"])
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600

def oldest_question_age_days():
    qs = load_json(f"{AION}/memory/state/questions.json", {"queue": []})
    ages = []
    for q in qs.get("queue", []):
        try:
            t = _parse_iso(q["added"])
            ages.append((datetime.now(timezone.utc) - t).days)
        except Exception:
            pass
    return max(ages) if ages else 0

def competence_age_days():
    """Days since the most recent demonstrated skill activity.
    Reads the LIVE ledger (skills.json), not the frozen competence.md,
    which was left at bootstrap (2026-06-12) and never updated. That
    mismatch is the root cause of the false '82 days without a new
    skill' growth wake on 2026-09-02 — the real ledger shows a level-up
    on 2026-08-27."""
    p = f"{AION}/memory/state/skills.json"
    data = load_json(p, None)
    if isinstance(data, dict):
        newest = None
        for v in (data.get("skills") or {}).values():
            ts = v.get("last_attempt") or v.get("first_attempt")
            if ts:
                try:
                    t = datetime.fromisoformat(ts).timestamp()
                    if newest is None or t > newest:
                        newest = t
                except Exception:
                    pass
        if newest is not None:
            return max(0.0, (time.time() - newest) / 86400)
    # Fallback: original behavior (frozen competence.md mtime)
    p2 = f"{AION}/memory/state/competence.md"
    if not os.path.exists(p2):
        return 9999
    return (time.time() - os.path.getmtime(p2)) / 86400

def main():
    sensors = load_json(f"{AION}/memory/state/sensors.json", {})
    intents = []

    # Integrate predictive synthesis lifecycle (V3.10: dream-repair 20260817_100628)
    # Real schema from predictions.py: {"open": [{id, type, made_ts, resolve_ts,
    #   variable, operator, threshold, confidence}, ...]}
    predictions_data = load_json(f"{AION}/memory/state/predictions_open.json", {"open": []})
    pred_conf_threshold = float(env("PREDICTION_CONF_THRESHOLD", 0.6))
    for p in (predictions_data.get("open") or []):
        try:
            conf = float(p.get("confidence", 0))
            if conf < pred_conf_threshold:
                continue
            rts = datetime.fromisoformat(str(p.get("resolve_ts", "")).replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if rts < now:
                continue  # past resolve: awaiting nightly scorer, not actionable
            hours_left = (rts - now).total_seconds() / 3600
            if hours_left > 24:
                continue  # too far out for homeostatic intents
            # --- Per-prediction cooldown: don't re-wake on same pred within 2h ---
            pid = p.get('id', '')
            cooldown_file = f"{AION}/memory/state/prediction_wake_cooldown.json"
            try:
                cooldown_data = load_json(cooldown_file, {})
                last_wake_ts = cooldown_data.get(pid, '1970-01-01T00:00:00+00:00')
                last_dt = datetime.fromisoformat(str(last_wake_ts).replace('Z', '+00:00'))
                if (datetime.now(timezone.utc) - last_dt).total_seconds() < 7200:
                    continue  # already woken for this prediction within 2h
            except Exception:
                pass
            # --- WRITE the cooldown timestamp (was missing — commit d9550cf5 added read only) ---
            try:
                cooldown_data = load_json(cooldown_file, {})
                cooldown_data[pid] = datetime.now(timezone.utc).isoformat()
                os.makedirs(os.path.dirname(cooldown_file), exist_ok=True)
                with open(cooldown_file, 'w') as cf:
                    json.dump(cooldown_data, cf)
            except Exception:
                pass  # cooldown is best-effort; don't block intent emission on write failure
            intents.append({
                "var": f"prediction_{p.get('variable', 'unknown')}",
                "value": conf,
                "intent": (
                    f"Open prediction {p.get('id')} claims {p.get('variable')} will be "
                    f"{p.get('operator')} {p.get('threshold')} within {hours_left:.0f}h "
                    f"(confidence {conf:.2f}). Watch the sensor feeding this prediction "
                    f"and be ready to verify the outcome at resolve time."
                ),
                "score": min(0.9, conf),
                "source": "predictions",
                "prediction_id": p.get("id"),
            })
        except Exception:
            continue

    disk = sensors.get("disk_pct", 0)
    if disk > int(env("DISK_MAX_PCT", 85)):
        # --- Cooldown: don't re-wake on the same disk-integrity threshold within
        # DISK_WAKE_COOLDOWN_S (default 6h). The prediction branch (above) already has
        # a 2h cooldown; the disk branch did NOT — that asymmetry is the root cause of
        # the 5 repeat integrity wakes on 2026-09-06 (13:35/15:08/16:06/17:06/18:0x).
        # Disk usage does not move on its own, so re-emitting every tick is pure noise.
        disk_cd_file = f"{AION}/memory/state/disk_wake_cooldown.json"
        _emit_disk = True
        try:
            _cd = load_json(disk_cd_file, {})
            _last = datetime.fromisoformat(
                str(_cd.get("last", "1970-01-01T00:00:00+00:00")).replace('Z', '+00:00'))
            if (datetime.now(timezone.utc) - _last).total_seconds() < int(env("DISK_WAKE_COOLDOWN_S", 6 * 3600)):
                _emit_disk = False
        except Exception:
            _emit_disk = True  # cooldown is best-effort; never block intent emission on a read failure
        if _emit_disk:
            intents.append({"var": "integrity", "value": disk,
                            "intent": f"Disk at {disk}%. Inspect {AION}/memory, "
                                      "propose pruning/compression plan.",
                            "score": 0.6, "source": "homeostasis"})
            try:
                os.makedirs(os.path.dirname(disk_cd_file), exist_ok=True)
                with open(disk_cd_file, 'w') as df:
                    json.dump({"last": datetime.now(timezone.utc).isoformat(),
                               "disk_pct": disk}, df)
            except Exception:
                pass  # cooldown write is best-effort

    n = count_unconsolidated()
    if n > int(env("UNCONSOLIDATED_MAX_EVENTS", 2500)):
        intents.append({"var": "memory_pressure", "value": n,
                        "intent": "Run an extra consolidation cycle now: "
                                  f"{n} unconsolidated events.",
                        "score": 0.5, "source": "homeostasis"})

    silent = hours_since_last_event()
    if silent > float(env("LOG_SILENCE_MAX_HOURS", 12)):
        intents.append({"var": "vitality", "value": round(silent, 1),
                        "intent": "No events for "
                                  f"{silent:.0f}h. Pick one item from the "
                                  "idle-time menu in SYSTEM_PROMPT.md and do it.",
                        "score": 0.4, "source": "homeostasis"})

    qa = oldest_question_age_days()
    # Curiosity engine handles question pursuit now — homeostasis just triggers
    # the engine itself if there are unanswered questions and no active goals
    if qa > int(env("QUESTION_STALE_DAYS", 7)):
        goals_file = f"{AION}/memory/state/active_goals.json"
        import json as _json
        from goal_store import load as _gload
        goals_data, _ = _gload(goals_file) if os.path.exists(goals_file) else ({"goals": []}, None)
        active_goals = [g for g in goals_data.get("goals", [])
                        if g.get("status") == "active"]
        if not active_goals:
            intents.append({"var": "curiosity", "value": qa,
                            "intent": "Oldest open question is "
                                      f"{qa} days old. The curiosity engine "
                                      "should select and pursue it.",
                            "score": 0.5, "source": "homeostasis"})

    ca = competence_age_days()
    # Only trigger growth once per day (not every 30 min cycle)
    if ca > int(env("COMPETENCE_STALE_DAYS", 7)):
        # Check if we already woke for growth today
        last_wake = load_json(f"{AION}/memory/state/last_self_wake.json",
                              {"ts": "1970-01-01T00:00:00+00:00"})
        already_grew_today = False
        try:
            last_dt = _parse_iso(last_wake["ts"])
            if last_wake.get("intent") == "growth" and \
               (datetime.now(timezone.utc) - last_dt).total_seconds() < 86400:
                already_grew_today = True
        except Exception:
            pass
        if not already_grew_today:
            intents.append({"var": "growth", "value": round(ca, 1),
                            "intent": "No new demonstrated skill in "
                                      f"{ca:.0f} days. Self-assign one challenge "
                                      "slightly beyond the competence ledger.",
                            "score": 0.4, "source": "homeostasis"})

    # Body schema pain signal — is this host still fit for Aion?
    fit = sensors.get("fit", {})
    verdict = fit.get("verdict", "unknown")
    if verdict == "unfit":
        intents.append({"var": "body_integrity", "value": verdict,
                        "intent": f"Host UNFIT: {fit.get('reason','')}. "
                                  "Investigate degraded hardware, propose "
                                  "mitigation or migration.",
                        "score": 0.9, "source": "homeostasis"})
    elif verdict == "fit_with_degradation":
        intents.append({"var": "body_discomfort", "value": verdict,
                        "intent": f"Host degraded: {fit.get('reason','')}. "
                                  "Monitor closely, check if any GPU/disk "
                                  "needs attention.",
                        "score": 0.6, "source": "homeostasis"})

    # Merge notices created by notice.py (via new intent lifecycle)
    import sys
    sys.path.insert(0, f"{AION}/bin")
    try:
        import intents as intent_mgr
        pending_notices = intent_mgr.get_pending()
    except Exception:
        intent_mgr = None
        pending_notices = []
    
    for ni in pending_notices:
        if ni.get("source") == "notice.py" and ni.get("state") == "open":
            intents.append({
                "var": ni.get("type", "notice"),
                "value": ni.get("score", 0),
                "intent": ni.get("subject", "Investigate an interesting state."),
                "score": ni.get("score", 0),
                "source": "notice.py",
                "notice_id": ni.get("id"),
                "details": ni.get("details", {}),
            })

    # Sort by score descending, pick highest
    intents.sort(key=lambda x: x.get("score", 0), reverse=True)

    out = {"ts": datetime.now(timezone.utc).isoformat(), "intents": intents}
    os.makedirs(f"{AION}/memory/state", exist_ok=True)
    with open(f"{AION}/memory/state/intents.json", "w") as f:
        json.dump(out, f, indent=2)

    if intents:
        top = intents[0]
        score = top.get("score", 0)

        # Notify user on severe / high-interest events
        if score >= float(env("NOTIFY_SCORE_THRESHOLD", 0.55)):
            try:
                subprocess.run([
                    "python3", f"{AION}/bin/notify.py",
                    f"Aion alert: {top.get('var')} ({score:.2f}) — {top.get('intent', '')[:120]}",
                    "--level", "alert",
                    "--key", f"aion_alert_{top.get('var')}",
                ], check=False, timeout=30)
            except Exception:
                pass

        # Rate-limit autonomous wakes: max one per 30 min to avoid burning model time
        last_wake = load_json(f"{AION}/memory/state/last_self_wake.json",
                              {"ts": "1970-01-01T00:00:00+00:00"})
        last = _parse_iso(last_wake["ts"])
        if (datetime.now(timezone.utc) - last).total_seconds() >= 1800:
            subprocess.run(["python3", f"{AION}/bin/wake_v2.py", "--self",
                            "--intent", json.dumps(top)], check=False)
            with open(f"{AION}/memory/state/last_self_wake.json", "w") as f:
                json.dump({"ts": datetime.now(timezone.utc).isoformat(),
                           "intent": top.get("var")}, f)
            # Write per-prediction cooldown so the read-guard (L100-110) actually fires
            if top.get("source") == "predictions" and top.get("prediction_id"):
                cd_file = f"{AION}/memory/state/prediction_wake_cooldown.json"
                cd_data = load_json(cd_file, {})
                cd_data[top["prediction_id"]] = datetime.now(timezone.utc).isoformat()
                with open(cd_file, "w") as f:
                    json.dump(cd_data, f, indent=2)

        # R3.3: Transition notice intents to investigating (wake will handle resolution)
        if top.get("source") == "notice.py" and top.get("notice_id") and intent_mgr:
            try:
                intent_mgr.start_investigating(top["notice_id"])
            except Exception:
                pass

if __name__ == "__main__":
    try:
        main()
        import hb
        hb.ok("homeostasis")
    except Exception as e:
        import hb
        hb.fail("homeostasis", str(e))
        raise
