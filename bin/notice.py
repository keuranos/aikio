#!/usr/bin/env python3
"""notice.py — Aion's attention system.

Scans recent sensor/environment history for interesting events. When the
combined interestingness score crosses a threshold, appends an intent to
memory/state/intents.json for homeostasis.py to consider.

Interestingness dimensions:
  - thermal: room temp rate of change, server room hot/cold, inter-room spread
  - energy: total system power jumps, heat pump COP drop, HP fault
  - load: load avg, GPU temp spikes, RAM pressure
  - novelty: first time a state has been seen today (e.g. first DHW cycle)

Outputs:
  - memory/state/notices.jsonl (rolling 24h)
  - appends to memory/state/intents.json when score >= threshold
  - logs a notice event to memory/episodic/<today>.jsonl
"""
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import aion_env  # loads config/aion.env

AION = os.environ.get("AION_HOME", "$AION_HOME")
HISTORY_PATH = f"{AION}/memory/state/env_history.jsonl"
SENSORS_PATH = f"{AION}/memory/state/sensors.json"
NOTICES_PATH = f"{AION}/memory/state/notices.jsonl"
INTENTS_PATH = f"{AION}/memory/state/intents.json"
THRESHOLD = 0.35  # cumulative interestingness to create an intent
DEDUP_STATE = f"{AION}/memory/state/notice_last_logged.json"


def should_log_episodic(score, reasons):
    """Check if this notice is different enough from the last logged one.
    Prevents the same notice from being logged to episodic memory every 5 min.
    """
    try:
        state = json.loads(open(DEDUP_STATE).read())
    except Exception:
        state = {}
    sig = "+".join(sorted(reasons))
    last = state.get(sig)
    if last is None:
        state[sig] = {"score": score, "ts": datetime.now(timezone.utc).isoformat()}
        _save_dedup_state(state)
        return True
    if abs(score - last.get("score", 0)) > 0.1:
        state[sig] = {"score": score, "ts": datetime.now(timezone.utc).isoformat()}
        _save_dedup_state(state)
        return True
    state[sig] = {"score": score, "ts": datetime.now(timezone.utc).isoformat()}
    _save_dedup_state(state)
    return False


def _save_dedup_state(state):
    try:
        os.makedirs(os.path.dirname(DEDUP_STATE), exist_ok=True)
        trimmed = dict(list(state.items())[-20:])
        open(DEDUP_STATE, "w").write(json.dumps(trimmed))
    except Exception:
        pass


def read_json(path, default=None):
    try:
        return json.loads(open(path).read())
    except Exception:
        return default if default is not None else {}


def read_env_history(n=12):
    """Read last n env_sense records (each 10 min apart = up to 2h)."""
    try:
        with open(HISTORY_PATH) as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:] if l.strip()]
    except Exception:
        return []


def parse_ts(ts):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def compute_noteworthiness(history, sensors, env):
    """Return (score, reasons, details)."""
    score = 0.0
    reasons = []
    details = {}

    if len(history) < 2:
        return score, reasons, details

    now_rec = history[-1]
    prev_rec = history[-2]

    # --- Thermal changes ---
    room_keys = [
        "room_alakerta_makuuhuone",
        "room_ylakerta_olohuone",
        "room_ylakerta_makuuhuone",
        "room_workshop_paapan",
        "room_workshop_mummin",
        "room_heatpump_air",
    ]
    max_delta = 0.0
    max_room = None
    for k in room_keys:
        c = now_rec.get(k)
        p = prev_rec.get(k)
        if c is not None and p is not None:
            d = abs(c - p)
            if d > max_delta:
                max_delta = d
                max_room = k
    if max_delta >= 0.5:
        s = min(max_delta / 2.0, 0.4)
        score += s
        reasons.append(f"room_temp_delta:{max_room} {max_delta:+.1f}C")
        details["room_temp_delta_C"] = round(max_delta, 2)
        details["room_delta_key"] = max_room

    # Server room thermal stress
    srv = env.get("server_room_c") or now_rec.get("room_heatpump_air")
    if srv is not None:
        if srv >= 28:
            score += 0.5
            reasons.append(f"server_room_hot:{srv:.1f}C")
            details["server_room_hot"] = round(srv, 1)
        elif srv <= 15:
            score += 0.3
            reasons.append(f"server_room_cold:{srv:.1f}C")
            details["server_room_cold"] = round(srv, 1)

    # Outdoor rate of change
    out_now = now_rec.get("outdoor")
    out_prev = prev_rec.get("outdoor")
    if out_now is not None and out_prev is not None:
        out_delta = abs(out_now - out_prev)
        if out_delta >= 2.0:
            s = min(out_delta / 10.0, 0.25)
            score += s
            reasons.append(f"outdoor_delta:{out_delta:+.1f}C")
            details["outdoor_delta_C"] = round(out_delta, 2)

    # --- Energy / heat pump ---
    hp_in = env.get("hp_electric_power_kw") or now_rec.get("hp_electric_power_kw")
    hp_out = env.get("hp_heat_output_kw") or now_rec.get("hp_heat_output_kw")
    cop = env.get("hp_cop") or now_rec.get("hp_cop", {}).get("hp_cop")
    if isinstance(cop, dict):
        cop = None
    if cop is not None and cop < 2.0 and hp_in is not None and hp_in > 0.3:
        score += 0.3
        reasons.append(f"low_cop:{cop:.1f}")
        details["low_cop"] = cop

    # Total system power jump
    power_now = sensors.get("total_power_w", 0)
    if power_now >= 250:
        score += min((power_now - 250) / 500, 0.3)
        reasons.append(f"high_power:{power_now:.0f}W")
        details["high_power_W"] = round(power_now, 0)

    # Heat pump fault
    fault = env.get("hp_fault") or now_rec.get("hp_fault")
    if fault not in (None, 0, "0", "OK"):
        score += 0.6
        reasons.append(f"hp_fault:{fault}")
        details["hp_fault"] = fault

    # --- Load / compute stress ---
    load1 = sensors.get("load1", 0)
    if load1 >= 4.0:
        score += min((load1 - 4.0) / 8.0, 0.25)
        reasons.append(f"high_load:{load1:.2f}")
        details["high_load1"] = round(load1, 2)

    ram_pct = 0
    ram = sensors.get("ram", {})
    if ram.get("total_mb", 0) > 0:
        ram_pct = ram.get("used_mb", 0) / ram["total_mb"] * 100
    if ram_pct >= 85:
        score += min((ram_pct - 85) / 15, 0.25)
        reasons.append(f"ram_pressure:{ram_pct:.0f}%")
        details["ram_pct"] = round(ram_pct, 1)

    # --- Novelty: active state changed ---
    state_now = now_rec.get("hp_active_state")
    state_prev = prev_rec.get("hp_active_state")
    if state_now and state_prev and state_now != state_prev:
        score += 0.25
        reasons.append(f"hp_state_change:{state_prev}->{state_now}")
        details["hp_state_change"] = f"{state_prev}->{state_now}"

    # --- Duplicate suppression: same reason repeatedly should not respam intents ---
    # only the first occurrence of a given state change in a window gets novelty credit
    try:
        recent_notices = []
        with open(NOTICES_PATH) as f:
            for line in f:
                if line.strip():
                    recent_notices.append(json.loads(line))
        # look back 1h for same hp_state_change
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        seen = {l.get("details", {}).get("hp_state_change")
                for l in recent_notices
                if l.get("details", {}).get("hp_state_change")
                and l.get("ts", "") > cutoff}
        if details.get("hp_state_change") in seen:
            score -= 0.25
            reasons = [r for r in reasons if not r.startswith("hp_state_change:")]
            details.pop("hp_state_change", None)
    except Exception:
        pass

    return round(min(score, 1.0), 3), reasons, details


def load_intents():
    try:
        data = json.loads(open(INTENTS_PATH).read())
        if isinstance(data, list):
            return data
        return data.get("intents", [])
    except Exception:
        return []


def save_intent(score, reasons, details):
    """R3.4: Edge-triggered intent creation with N=3 confirmation + exponential backoff.
    
    - Require N=3 consecutive out-of-band readings before flagging
    - Exponential backoff: 1h → 4h → 24h on repeats of the same notice type
    - Use intent lifecycle (add_or_increment) instead of creating duplicates
    """
    import intents as intent_mgr
    import sys
    sys.path.insert(0, f"{AION}/bin")
    
    # Build a subject key from reasons (normalize so similar notices group)
    subject = "+".join(sorted(reasons[:3])) if reasons else "unknown"
    intent_type = "self_investigate"
    
    # Check consecutive confirmations
    confirm_file = f"{AION}/memory/state/notice_confirm_{intent_type}.json"
    try:
        with open(confirm_file) as f:
            confirm = json.load(f)
    except Exception:
        confirm = {}
    
    key = subject
    now = datetime.now(timezone.utc)
    
    # Reset if last reading was too long ago (state returned to normal between)
    last_ts = confirm.get(key, {}).get("last_ts")
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts)
            if (now - last_dt).total_seconds() > 1800:  # 30 min gap = reset
                confirm[key] = {"count": 0}
        except Exception:
            pass
    
    # Increment confirmation count
    confirm.setdefault(key, {"count": 0})
    confirm[key]["count"] = confirm[key].get("count", 0) + 1
    confirm[key]["last_ts"] = now.isoformat()
    confirm[key]["last_score"] = score
    
    with open(confirm_file, "w") as f:
        json.dump(confirm, f, indent=2)
    
    # N=3 consecutive confirmations required
    N_CONFIRM = 3
    if confirm[key]["count"] < N_CONFIRM:
        return None  # Not enough confirmations yet
    
    # Check exponential backoff: has this subject been flagged recently?
    intent = intent_mgr.get_open(
        {"intents": intent_mgr.load().get("intents", [])},
        intent_type, subject
    )
    if intent:
        # Already open — check backoff
        last_seen = intent.get("last_seen", "")
        if last_seen:
            try:
                last_dt = datetime.fromisoformat(last_seen)
                age_hours = (now - last_dt).total_seconds() / 3600
                count = intent.get("count", 1)
                # Backoff schedule: count 1→1h, 2→4h, 3→24h, 4+→24h
                backoff = min(24, [1, 4, 24][min(count - 1, 2)])
                if age_hours < backoff:
                    return None  # Too soon to re-flag
            except Exception:
                pass
    
    # Create or increment the intent
    result = intent_mgr.add_or_increment(
        intent_type=intent_type,
        subject=subject,
        score=score,
        reasons=reasons,
        details=details,
        source="notice.py",
    )
    return result


def log_notice(score, reasons, details):
    # V3.0.8: Don't log zero-score notices — they're noise that inflates memory pressure
    if score == 0:
        return
    # Dedup: skip episodic logging if same reasons + similar score recently logged
    if not should_log_episodic(score, reasons):
        return
    now = datetime.now(timezone.utc)
    record = {
        "ts": now.isoformat(),
        "score": score,
        "reasons": reasons,
        "details": details,
    }
    with open(NOTICES_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    # trim to 24h
    cutoff = (now - timedelta(hours=24)).isoformat()
    try:
        lines = open(NOTICES_PATH).readlines()
        keep = [l for l in lines if l.strip() and json.loads(l)["ts"] > cutoff]
        if len(keep) < len(lines):
            with open(NOTICES_PATH, "w") as f:
                f.writelines(keep)
    except Exception:
        pass

    # episodic event
    day = now.strftime("%Y-%m-%d")
    ev = {
        "ts": now.isoformat(),
        "type": "notice",
        "text": f"noticed {', '.join(reasons)} (score {score})",
        "meta": {"score": score, "details": details},
        "id": f"notice_{int(time.time())}",
    }
    path = f"{AION}/memory/episodic/{day}.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(ev) + "\n")


def main():
    sensors = read_json(SENSORS_PATH, {})
    env = sensors.get("environment", {})
    history = read_env_history()
    score, reasons, details = compute_noteworthiness(history, sensors, env)

    log_notice(score, reasons, details)

    if score >= THRESHOLD:
        intent = save_intent(score, reasons, details)
        if intent:
            print(f"NOTICE intent created (score={score}): {intent.get('id')} reasons={reasons}")
        else:
            print(f"notice: score={score} (accumulating confirmations or in backoff)")
        return 0
    else:
        print(f"notice: score={score}, reasons={reasons}")
        return 0


if __name__ == "__main__":
    try:
        rc = main()
        import hb
        hb.ok("notice")
        raise SystemExit(rc)
    except Exception as e:
        import hb
        hb.fail("notice", str(e))
        raise
