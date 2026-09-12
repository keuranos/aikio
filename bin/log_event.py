#!/usr/bin/env python3
"""log_event.py — the ONLY writer to the episodic log.

V3.1.2: Routes events to experience or telemetry lane based on type.
  - experience: memory/episodic/YYYY-MM-DD.jsonl (dreams, chats, curiosity, vision, etc.)
  - telemetry:  memory/episodic/telemetry/YYYY-MM-DD.jsonl (notices, proprioception, sensor_digest, subsystem_dead)

Usage:
  log_event.py --type user_msg --text "..." [--meta '{"k":"v"}']
  echo "long text" | log_event.py --type reflection --stdin
Types: user_msg, assistant_msg, tool_call, tool_result, sensor_digest,
       self_wake, reflection, audit, prediction, system
"""
import argparse, json, os, sys, time, hashlib
from datetime import datetime, timezone
from pathlib import Path

AION_HOME = os.environ.get("AION_HOME", "$AION_HOME")
EPISODIC = os.path.join(AION_HOME, "memory", "episodic")

# Import lane routing (V3.1.2)
sys.path.insert(0, os.path.join(AION_HOME, "bin"))
try:
    from lanes import lane_path, lane_for
except ImportError:
    # Fallback: all events go to the main episodic dir (pre-v3.1 behavior)
    def lane_path(aion_home, event_type, day_str):
        return os.path.join(aion_home, "memory", "episodic", f"{day_str}.jsonl")
    def lane_for(event_type):
        return "experience"

def _validate_sensor_digest(text, meta):
    """Error boundary: validate sensor_digest content integrity.
    
    Returns (ok: bool, reason: str).
    A valid sensor_digest must have:
      - non-empty text that is valid JSON (structured sensor payload)
      - OR meta containing a 'sensors' dict with at least one key
    """
    # Check text: must be non-empty and valid JSON
    if text and text.strip():
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return True, ""
            if isinstance(parsed, list) and len(parsed) > 0:
                return True, ""
        except (json.JSONDecodeError, TypeError):
            pass
    # Check meta: must contain structured sensor data
    if isinstance(meta, dict):
        sensors = meta.get("sensors")
        if isinstance(sensors, dict) and len(sensors) > 0:
            return True, ""
        # meta with explicit readings is also valid
        if any(k in meta for k in ("readings", "values", "data")):
            return True, ""
    # Both empty → corrupt
    if not text or not text.strip():
        return False, "text is empty or whitespace-only"
    return False, "text is not valid JSON and meta lacks structured sensor data"


def _validate_proprioception(text, meta):
    """Error boundary: validate proprioception content integrity.
    
    Returns (ok: bool, reason: str).
    A valid proprioception must have:
      - non-empty text that is valid JSON (structured pose/joint payload)
      - OR meta containing a 'proprioception' dict with at least one key
    """
    # Check text: must be non-empty and valid JSON
    if text and text.strip():
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return True, ""
            if isinstance(parsed, list) and len(parsed) > 0:
                return True, ""
        except (json.JSONDecodeError, TypeError):
            pass
    # Check meta: must contain structured proprioception data
    if isinstance(meta, dict):
        proprio = meta.get("proprioception")
        if isinstance(proprio, dict) and len(proprio) > 0:
            return True, ""
        # meta with explicit readings is also valid
        if any(k in meta for k in ("readings", "values", "data", "pose", "joint")):
            return True, ""
    # Both empty → corrupt
    if not text or not text.strip():
        return False, "text is empty or whitespace-only"
    return False, "text is not valid JSON and meta lacks structured proprioception data"


def _write_event(event, event_type):
    """Write a single event to the correct lane. Returns event id."""
    event["id"] = hashlib.sha1(
        f"{event['ts']}{event['type']}{event['text'][:80]}".encode()).hexdigest()[:12]
    day_str = event["ts"][:10]  # ISO date from ts
    path = lane_path(AION_HOME, event["type"], day_str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event["id"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--type", required=True)
    p.add_argument("--text", default=None)
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--meta", default="{}")
    a = p.parse_args()

    text = sys.stdin.read() if a.stdin else (a.text or "")
    try:
        meta = json.loads(a.meta)
    except json.JSONDecodeError:
        meta = None  # corrupt meta — treat as None

    now = datetime.now(timezone.utc)

    # ── ERROR BOUNDARY: validate sensor_digest integrity ──
    if a.type == "sensor_digest":
        ok, reason = _validate_sensor_digest(text, meta)
        if not ok:
            # Write a subsystem_dead event (consolidate_v2.py processes this)
            # so the corruption is NOT silently dropped.
            err_event = {
                "ts": now.isoformat(),
                "type": "subsystem_dead",
                "text": f"sensor_digest corruption: {reason}",
                "meta": {"source": "log_event_validation", "original_type": "sensor_digest", "reason": reason},
            }
            eid = _write_event(err_event, err_event["type"])
            print(f"CORRUPT: {reason} → wrote subsystem_dead {eid}", file=sys.stderr)
            sys.exit(1)  # non-zero so upstream (sensors.sh) can react

    # ── ERROR BOUNDARY: validate proprioception integrity ──
    if a.type == "proprioception":
        ok, reason = _validate_proprioception(text, meta)
        if not ok:
            # Write a subsystem_dead event so corruption triggers consolidation
            err_event = {
                "ts": now.isoformat(),
                "type": "subsystem_dead",
                "text": f"proprioception corruption: {reason}",
                "meta": {"source": "log_event_validation", "original_type": "proprioception", "reason": reason},
            }
            eid = _write_event(err_event, a.type)
            print(f"CORRUPT: {reason} → wrote subsystem_dead {eid}", file=sys.stderr)
            sys.exit(1)

    event = {
        "ts": now.isoformat(),
        "type": a.type,
        "text": text,
        "meta": meta,
    }

    # V3.1.2: Route to the correct lane
    eid = _write_event(event, a.type)
    print(eid)

if __name__ == "__main__":
    main()
