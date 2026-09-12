#!/usr/bin/env python3
"""intents.py — R3.3: Intent lifecycle management.

States: open → investigating → resolved | abandoned
Each intent: {id, state, type, subject, count, first_seen, last_seen, resolution, score, ...}

Key rule: a new notice matching an open intent (same type+subject) increments count
instead of creating a duplicate.
"""
import json, os, time
from datetime import datetime, timezone

AION = os.environ.get("AION_HOME", "$AION_HOME")
INTENTS_FILE = f"{AION}/memory/state/intents.json"

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def load():
    try:
        with open(INTENTS_FILE) as f:
            return json.load(f)
    except Exception:
        return {"ts": now_iso(), "intents": []}

def save(data):
    data["ts"] = now_iso()
    with open(INTENTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_open(data, intent_type, subject):
    """Find an open intent matching type+subject."""
    for i in data.get("intents", []):
        if (i.get("type") == intent_type and 
            i.get("subject") == subject and
            i.get("state") in ("open", "investigating")):
            return i
    return None

def add_or_increment(intent_type, subject, score=0.0, reasons=None, details=None, source="notice.py"):
    """Add a new intent or increment existing. Returns the intent."""
    data = load()
    
    existing = get_open(data, intent_type, subject)
    if existing:
        existing["count"] = existing.get("count", 1) + 1
        existing["last_seen"] = now_iso()
        existing["score"] = max(existing.get("score", 0), score)
        if reasons:
            for r in reasons:
                if r not in existing.get("reasons", []):
                    existing.setdefault("reasons", []).append(r)
        if details:
            existing.setdefault("details", {}).update(details)
        save(data)
        return existing
    
    intent = {
        "id": f"{intent_type}_{int(time.time())}",
        "state": "open",
        "type": intent_type,
        "subject": subject,
        "count": 1,
        "first_seen": now_iso(),
        "last_seen": now_iso(),
        "score": score,
        "source": source,
    }
    if reasons:
        intent["reasons"] = reasons
    if details:
        intent["details"] = details
    
    data.setdefault("intents", []).append(intent)
    save(data)
    return intent

def transition(intent_id, new_state, resolution=""):
    """Transition an intent to a new state. Returns the intent or None."""
    data = load()
    for i in data.get("intents", []):
        if i.get("id") == intent_id:
            i["state"] = new_state
            i["resolution"] = resolution
            i["resolved_ts"] = now_iso()
            save(data)
            return i
    return None

def transition_by_type(intent_type, new_state, resolution=""):
    """Transition all open intents of a type. Returns count."""
    data = load()
    count = 0
    for i in data.get("intents", []):
        if i.get("type") == intent_type and i.get("state") in ("open", "investigating"):
            i["state"] = new_state
            i["resolution"] = resolution
            i["resolved_ts"] = now_iso()
            count += 1
    if count:
        save(data)
    return count

def get_pending():
    """Get all open/investigating intents sorted by score."""
    data = load()
    pending = [i for i in data.get("intents", []) if i.get("state") in ("open", "investigating")]
    return sorted(pending, key=lambda x: x.get("score", 0), reverse=True)

def start_investigating(intent_id):
    """Mark an intent as being investigated."""
    data = load()
    for i in data.get("intents", []):
        if i.get("id") == intent_id:
            i["state"] = "investigating"
            save(data)
            return i
    return None
