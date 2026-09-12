#!/usr/bin/env python3
"""telemetry_digest.py — V3.1.3: Nightly telemetry summarizer.

Reads the day's telemetry-lane events and produces one synthetic experience
event summarizing counts, anomalies, and top repeated notices. Pure Python,
no LLM call. Runs as the first step of nightly.sh before consolidation.

The digest event type is "telemetry_digest" (classified as experience in lanes.py)
so consolidation extraction sees it alongside dreams, chats, and curiosity.
"""
import json, os, sys, glob
from datetime import datetime, timezone, timedelta
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa
from lanes import telemetry_dir, experience_dir, lane_path, is_telemetry

AION = os.environ.get("AION_HOME", "$AION_HOME")


def read_telemetry_for_day(day_str):
    """Read all telemetry events for a given day."""
    path = os.path.join(telemetry_dir(AION), f"{day_str}.jsonl")
    events = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events


def summarize(events):
    """Turn raw telemetry events into a human-readable digest string."""
    if not events:
        return "No telemetry events today."

    counts = Counter(e.get("type", "?") for e in events)
    parts = []

    # Counts per type
    parts.append("Telemetry summary for the day:")
    for t, c in counts.most_common():
        parts.append(f"  {t}: {c}")

    # Top repeated notices (if any)
    notices = [e for e in events if e.get("type") == "notice"]
    if notices:
        notice_reasons = Counter()
        for n in notices:
            for r in n.get("meta", {}).get("details", {}).values():
                if isinstance(r, str) and len(r) < 100:
                    notice_reasons[r] += 1
                break  # just first detail
            # Also try the text
            text = n.get("text", "")
            if text:
                notice_reasons[text[:80]] += 1

        top_notices = notice_reasons.most_common(3)
        if top_notices:
            parts.append("\nTop notice triggers:")
            for reason, count in top_notices:
                parts.append(f"  ({count}x) {reason}")

    # Subsystem deaths (if any)
    deaths = [e for e in events if e.get("type") == "subsystem_dead"]
    if deaths:
        dead_subs = Counter()
        for d in deaths:
            text = d.get("text", "")
            try:
                info = json.loads(text)
                dead_subs[info.get("subsystem", "?")] += 1
            except Exception:
                dead_subs[text[:50]] += 1
        parts.append("\nSubsystem deaths:")
        for sub, count in dead_subs.most_common():
            parts.append(f"  {sub}: {count}x")

    # Proprioception highlights (temp extremes)
    proprio = [e for e in events if e.get("type") == "proprioception"]
    if proprio:
        parts.append(f"\nProprioception readings: {len(proprio)}")
        # Sample first and last
        first = proprio[0].get("text", "")[:120]
        last = proprio[-1].get("text", "")[:120]
        parts.append(f"  First: {first}")
        parts.append(f"  Last:  {last}")

    return "\n".join(parts)


def main():
    # Summarize yesterday's telemetry (we run at 03:30, so yesterday is done)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    events = read_telemetry_for_day(yesterday)

    if not events:
        print(f"[telemetry_digest] No telemetry events for {yesterday}")
        # Still log an empty digest so the marker advances
        summary_text = f"No telemetry events for {yesterday}."
    else:
        summary_text = summarize(events)

    # Log as a synthetic experience event
    meta = {
        "day": yesterday,
        "telemetry_count": len(events),
        "synthetic": True,
    }

    # Write to the experience lane using log_event.py
    import subprocess
    result = subprocess.run(
        ["python3", f"{AION}/bin/log_event.py",
         "--type", "telemetry_digest",
         "--text", summary_text,
         "--meta", json.dumps(meta)],
        capture_output=True, text=True
    )

    event_id = result.stdout.strip()
    print(f"[telemetry_digest] {yesterday}: {len(events)} telemetry events -> 1 digest event (id: {event_id})")
    print(f"[telemetry_digest] Summary preview: {summary_text[:200]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())