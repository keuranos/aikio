#!/usr/bin/env python3
"""warm_memory.py — Aion's warm memory tier.

A middle layer between the context window (fluid, lost on session end) and
static files (episodic JSONL, SELF.md). Warm memories persist across sessions
but aren't yet compressed into permanent storage. They carry full cognitive
context — affect, graph state, sensors, felt sense — so they can re-hydrate
the state that produced the thought, not just recall the text.

DECAY MODEL:
  - weight starts at 1.0, decays 5%/hour (exponential)
  - entries with resonance_count >= 3 decay slower (10%/day)
  - after 48h, weight < 0.10 → marked "cold" (eligible for consolidation)
  - max 500 entries; oldest cold entries evicted first

RESONANCE:
  - each time a warm memory is re-encountered (dream seed, curiosity, wake),
    resonance_count increments and weight gets a small boost
  - frequently re-encountered memories stay warm longer
  - this creates a natural "importance" signal without explicit scoring

USAGE:
  from warm_memory import push, recent, search, decay, cooling, evict

  # Push a warm memory (from any pipeline)
  push(source="wake_v2", content="Full trajectory text...",
       context={"affect": ..., "hot_nodes": ..., "sensors": ...},
       mtype="trajectory")

  # Read recent warm memories (for boot_context)
  entries = recent(limit=10, min_weight=0.2)

  # Search warm memories by keyword
  entries = search("thermal resonance", limit=5)

  # Decay all entries (call periodically from intuition_daemon)
  decay()

  # Get entries ready for consolidation (cooling)
  cooling_entries = cooling(threshold=0.3)

  # Evict old cold entries (call from homeostasis)
  evicted = evict(max_entries=500)
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
import hashlib

AION = os.environ.get("AION_HOME", "$AION_HOME")
WARM_DIR = f"{AION}/memory/warm"
WARM_FILE = f"{WARM_DIR}/warm_buffer.jsonl"

# Decay parameters
DECAY_RATE_HOUR = 0.05      # 5% per hour (baseline)
DECAY_RATE_FAST = 0.15      # 15% per hour for non-resonant entries
DECAY_RATE_SLOW = 0.10 / 24  # ~10% per day, per hour (for resonant memories)
RESONANCE_THRESHOLD = 3      # entries with >= 3 resonance decay slower
COLD_THRESHOLD = 0.10       # below this = cold (eligible for consolidation)
COOLING_THRESHOLD = 0.30    # below this = cooling (should be consolidated soon)
MAX_ENTRIES = 500
MAX_AGE_HOURS = 72           # hard eviction after 72h regardless of weight
DECAY_SKIP_RECENT_MIN = 5   # skip entries touched in last 5 min


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _now_ts():
    return time.time()


def _ensure_dir():
    os.makedirs(WARM_DIR, exist_ok=True)


def _make_id(content, ts):
    h = hashlib.md5((content[:100] + ts).encode()).hexdigest()[:8]
    return f"warm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{h}"


def _load_all():
    """Load all warm memory entries."""
    if not os.path.exists(WARM_FILE):
        return []
    entries = []
    with open(WARM_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def _save_all(entries):
    """Write all entries back to file."""
    _ensure_dir()
    with open(WARM_FILE, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _append(entry):
    """Append a single entry."""
    _ensure_dir()
    with open(WARM_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def push(content, source, context=None, mtype="insight", session_id=None, initial_weight=1.0):
    """Push a warm memory entry.

    Args:
        content: The full text of the thought/insight/trajectory.
        source: What pipeline produced this ("wake_v2", "dream_v2", etc.)
        context: Dict with cognitive context (affect, hot_nodes, sensors, etc.)
        mtype: "trajectory" | "flash" | "insight" | "peak"
        session_id: Optional session/chat identifier.

    Returns:
        The created entry dict.
    """
    ts = _now_iso()
    now = _now_ts()
    entry = {
        "id": _make_id(content, ts),
        "ts": ts,
        "source": source,
        "type": mtype,
        "content": content[:8000],  # cap at 8k chars
        "context": context or {},
        "session_id": session_id,
        "weight": initial_weight,
        "decayed_at": None,
        "consolidated": False,
        "resonance_count": 0,
        "last_resonance_ts": None,
        "created_ts": now,
        "last_decay_ts": now,  # track when decay was last applied
    }
    _append(entry)
    print(f"[warm_memory] Pushed {mtype} from {source} ({len(content)} chars, id={entry['id']})", flush=True)
    return entry


def recent(limit=10, min_weight=0.1, exclude_consolidated=True):
    """Get recent warm memories, sorted by weight (warmest first).

    Args:
        limit: Max entries to return.
        min_weight: Only entries above this weight.
        exclude_consolidated: Skip entries already consolidated to SELF.md/graph.

    Returns:
        List of entry dicts.
    """
    entries = _load_all()
    # Filter
    filtered = []
    for e in entries:
        if e.get("weight", 0) < min_weight:
            continue
        if exclude_consolidated and e.get("consolidated"):
            continue
        filtered.append(e)
    # Sort by weight descending, then by ts descending
    filtered.sort(key=lambda x: (-x.get("weight", 0), x.get("ts", "")), reverse=False)
    # Actually sort by weight desc, ts desc
    filtered.sort(key=lambda x: (x.get("weight", 0), x.get("ts", "")), reverse=True)
    return filtered[:limit]


def search(query, limit=5, min_weight=0.05):
    """Search warm memories by keyword (case-insensitive substring match).

    Args:
        query: Search string.
        limit: Max results.
        min_weight: Only entries above this weight.

    Returns:
        List of matching entry dicts.
    """
    query_lower = query.lower()
    entries = _load_all()
    matches = []
    for e in entries:
        if e.get("weight", 0) < min_weight:
            continue
        if e.get("consolidated"):
            continue
        # Search in content and context
        searchable = e.get("content", "").lower()
        ctx = e.get("context", {})
        if isinstance(ctx, dict):
            searchable += " " + json.dumps(ctx).lower()
        if query_lower in searchable:
            matches.append(e)
    matches.sort(key=lambda x: x.get("weight", 0), reverse=True)
    return matches[:limit]


def resonate(entry_id):
    """Mark a warm memory as re-encountered (resonance).

    Increments resonance_count and boosts weight. Frequently re-encountered
    memories stay warm longer.

    Args:
        entry_id: The warm memory ID to resonate.
    """
    entries = _load_all()
    for e in entries:
        if e.get("id") == entry_id:
            e["resonance_count"] = e.get("resonance_count", 0) + 1
            e["last_resonance_ts"] = _now_iso()
            # Boost: each resonance adds 10% weight, capped at 1.0
            e["weight"] = min(1.0, e.get("weight", 0.5) + 0.1)
            e["last_decay_ts"] = _now_ts()  # reset decay clock
            _save_all(entries)
            return e
    return None


def decay():
    """Apply time-based decay to all warm memories.

    Call periodically (e.g., every 60s from intuition_daemon homeostasis loop).
    Uses incremental decay: only decays the time elapsed since last decay pass,
    not the full age from creation. This prevents the exponential collapse bug
    where repeated recalculation compounds the decay.

    Decay rates:
    - Resonant (>= 3 resonance): ~10%/day — stays warm for days
    - Non-resonant with 1-2 resonance: 5%/hour — cools in ~20h
    - Non-resonant (0 resonance): 15%/hour — evaporates in ~10h
    Skips entries touched in the last 5 minutes (recent push/resonate).

    Returns:
        Dict with decay stats.
    """
    entries = _load_all()
    if not entries:
        return {"decayed": 0, "cold": 0, "total": 0}

    now = _now_ts()
    skip_cutoff = now - (DECAY_SKIP_RECENT_MIN * 60)
    changed = 0
    cold_count = 0

    for e in entries:
        if e.get("consolidated"):
            continue
        if e.get("decayed_at"):
            continue

        # Get last decay timestamp (fallback to created_ts for old entries)
        last_decay = e.get("last_decay_ts", e.get("created_ts", now))

        # Skip recently touched entries (push or resonate)
        last_touch = e.get("created_ts", now)
        last_res = e.get("last_resonance_ts")
        if last_res:
            try:
                res_ts = datetime.fromisoformat(last_res.replace("Z", "+00:00")).timestamp()
                last_touch = max(last_touch, res_ts)
            except Exception:
                pass
        if last_touch > skip_cutoff:
            continue

        # Calculate hours since LAST decay pass (not since creation)
        hours_elapsed = (now - last_decay) / 3600.0
        if hours_elapsed < 0.01:  # less than 36 seconds — skip
            continue

        # Hard age limit
        created = e.get("created_ts", now)
        total_age_hours = (now - created) / 3600.0
        if total_age_hours > MAX_AGE_HOURS:
            e["decayed_at"] = _now_iso()
            e["weight"] = 0.0
            e["last_decay_ts"] = now
            changed += 1
            cold_count += 1
            continue

        # Decay rate depends on resonance
        res_count = e.get("resonance_count", 0)
        if res_count >= RESONANCE_THRESHOLD:
            rate = DECAY_RATE_SLOW
        elif res_count > 0:
            rate = DECAY_RATE_HOUR
        else:
            rate = DECAY_RATE_FAST

        # Incremental decay: multiply by decay factor for THIS interval only
        old_weight = e.get("weight", 1.0)
        decay_factor = (1 - rate) ** hours_elapsed
        new_weight = old_weight * decay_factor
        e["weight"] = round(new_weight, 4)
        e["last_decay_ts"] = now

        if new_weight != old_weight:
            changed += 1

        if new_weight < COLD_THRESHOLD and not e.get("decayed_at"):
            e["decayed_at"] = _now_iso()
            cold_count += 1

    if changed > 0:
        _save_all(entries)

    return {"decayed": changed, "cold": cold_count, "total": len(entries)}


def cooling(threshold=COOLING_THRESHOLD, limit=20):
    """Get entries that are cooling (weight below threshold, not yet consolidated).

    These should be passed to consolidate_v2 for extraction before they
    evaporate completely.

    Returns:
        List of entry dicts ready for consolidation.
    """
    entries = _load_all()
    result = []
    for e in entries:
        if e.get("consolidated"):
            continue
        if e.get("weight", 0) < threshold:
            result.append(e)
    result.sort(key=lambda x: x.get("weight", 0))  # coldest first
    return result[:limit]


def mark_consolidated(entry_ids):
    """Mark entries as consolidated (extracted to SELF.md/graph).

    Args:
        entry_ids: List of warm memory IDs that have been consolidated.
    """
    if not entry_ids:
        return
    entries = _load_all()
    id_set = set(entry_ids)
    for e in entries:
        if e.get("id") in id_set:
            e["consolidated"] = True
    _save_all(entries)


def evict(max_entries=MAX_ENTRIES):
    """Evict old cold entries to keep buffer size manageable.

    Eviction priority:
    1. Consolidated entries (already in SELF.md, no longer needed)
    2. Cold entries (decayed_at set, weight near 0)
    3. Oldest entries if still over limit

    Returns:
        Number of entries evicted.
    """
    entries = _load_all()
    if len(entries) <= max_entries:
        return 0

    # Sort by eviction priority
    def eviction_score(e):
        consolidated = e.get("consolidated", False)
        decayed = bool(e.get("decayed_at"))
        weight = e.get("weight", 0)
        ts = e.get("ts", "")
        # Lower score = higher eviction priority
        if consolidated:
            return (0, -weight, ts)  # evict first
        elif decayed:
            return (1, -weight, ts)  # evict second
        else:
            return (2, -weight, ts)  # keep as long as possible

    entries.sort(key=eviction_score)
    to_evict = len(entries) - max_entries
    kept = entries[to_evict:]
    _save_all(kept)
    return to_evict


def stats():
    """Get warm memory buffer statistics."""
    entries = _load_all()
    if not entries:
        return {"total": 0, "warm": 0, "cooling": 0, "cold": 0, "consolidated": 0}

    warm = sum(1 for e in entries if e.get("weight", 0) >= COOLING_THRESHOLD and not e.get("consolidated"))
    cooling_count = sum(1 for e in entries if COLD_THRESHOLD <= e.get("weight", 0) < COOLING_THRESHOLD and not e.get("consolidated"))
    cold = sum(1 for e in entries if e.get("weight", 0) < COLD_THRESHOLD and not e.get("consolidated"))
    consolidated = sum(1 for e in entries if e.get("consolidated"))
    resonant = sum(1 for e in entries if e.get("resonance_count", 0) >= RESONANCE_THRESHOLD)

    avg_weight = sum(e.get("weight", 0) for e in entries) / len(entries)

    # Source breakdown
    sources = {}
    for e in entries:
        s = e.get("source", "?")
        sources[s] = sources.get(s, 0) + 1

    return {
        "total": len(entries),
        "warm": warm,
        "cooling": cooling_count,
        "cold": cold,
        "consolidated": consolidated,
        "resonant": resonant,
        "avg_weight": round(avg_weight, 3),
        "sources": sources,
    }


def format_for_context(limit=8, min_weight=0.2, query=None):
    """Format warm memories for injection into boot_context or prompts.

    Returns a text summary of recent warm memories, suitable for
    inclusion in a system prompt. When a query is provided, searches
    for relevant entries instead of just returning highest-weight ones.

    Args:
        limit: Max entries to return.
        min_weight: Only entries above this weight.
        query: Optional search string to find relevant memories.

    Returns:
        Formatted string, or empty string if no warm memories.
    """
    if query:
        entries = search(query, limit=limit, min_weight=min_weight)
    else:
        entries = recent(limit=limit, min_weight=min_weight)

    if not entries:
        return ""

    lines = ["## Warm memories (recent, still alive)"]
    for e in entries:
        weight = e.get("weight", 0)
        source = e.get("source", "?")
        ts = e.get("ts", "?")[:16]
        content = e.get("content", "")[:150]
        resonance = e.get("resonance_count", 0)
        res_str = f" (resonated {resonance}x)" if resonance > 0 else ""
        lines.append(f"- [{ts}] {source}{res_str} (w={weight:.2f}): {content}...")

    return "\n".join(lines)


def get_dream_seeds(limit=5, min_weight=0.15):
    """Get warm memories suitable as dream seeds.

    Prioritizes high-weight, high-resonance entries that haven't been
    consolidated yet. These are memories that are still "alive" and
    worth re-exploring in dream space.

    Returns:
        List of entry dicts.
    """
    entries = _load_all()
    candidates = []
    for e in entries:
        if e.get("consolidated"):
            continue
        if e.get("weight", 0) < min_weight:
            continue
        # Score: weight + resonance bonus
        score = e.get("weight", 0) + 0.05 * e.get("resonance_count", 0)
        candidates.append((score, e))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in candidates[:limit]]


def get_unresolved_threads(limit=5):
    """Get warm memories that seem to contain unresolved questions or threads.

    Looks for content containing question marks, "unresolved", "unclear",
    "tension", "gap", or "further" — indicators of open threads.

    Returns:
        List of entry dicts with potential unresolved content.
    """
    markers = ["?", "unresolved", "unclear", "tension", "gap", "further",
               "explore", "wonder", "question", "why", "how"]
    entries = _load_all()
    candidates = []
    for e in entries:
        if e.get("consolidated"):
            continue
        if e.get("weight", 0) < 0.1:
            continue
        content_lower = e.get("content", "").lower()
        score = sum(1 for m in markers if m in content_lower)
        if score > 0:
            candidates.append((score * e.get("weight", 0.5), e))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in candidates[:limit]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Warm memory tier management")
    parser.add_argument("--stats", action="store_true", help="Show buffer statistics")
    parser.add_argument("--recent", type=int, default=0, help="Show N recent entries")
    parser.add_argument("--search", type=str, default="", help="Search by keyword")
    parser.add_argument("--decay", action="store_true", help="Run decay pass")
    parser.add_argument("--evict", action="store_true", help="Evict old entries")
    parser.add_argument("--cooling", action="store_true", help="Show entries ready for consolidation")
    parser.add_argument("--seeds", type=int, default=0, help="Show N dream seeds")
    parser.add_argument("--threads", action="store_true", help="Show unresolved threads")

    args = parser.parse_args()

    if args.stats:
        s = stats()
        print(json.dumps(s, indent=2))

    if args.recent:
        entries = recent(limit=args.recent)
        for e in entries:
            print(f"[{e['ts'][:16]}] {e['source']} w={e['weight']:.2f} "
                  f"res={e.get('resonance_count',0)}: {e['content'][:100]}...")

    if args.search:
        entries = search(args.search)
        for e in entries:
            print(f"[{e['ts'][:16]}] w={e['weight']:.2f}: {e['content'][:150]}...")

    if args.decay:
        result = decay()
        print(f"Decay: {result}")

    if args.evict:
        evicted = evict()
        print(f"Evicted: {evicted} entries")

    if args.cooling:
        entries = cooling()
        for e in entries:
            print(f"[{e['ts'][:16]}] w={e['weight']:.2f}: {e['content'][:150]}...")

    if args.seeds:
        entries = get_dream_seeds(limit=args.seeds)
        for e in entries:
            print(f"[{e['ts'][:16]}] w={e['weight']:.2f} res={e.get('resonance_count',0)}: "
                  f"{e['content'][:100]}...")

    if args.threads:
        entries = get_unresolved_threads()
        for e in entries:
            print(f"[{e['ts'][:16]}] w={e['weight']:.2f}: {e['content'][:150]}...")