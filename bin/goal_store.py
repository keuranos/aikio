#!/usr/bin/env python3
"""goal_store.py — hardened reader/writer for memory/state/active_goals.json.

History (Sep 12): four entries in goals[] arrived wrapped as [dict] pairs.
Every consumer reads with dict API (g.get(...)) and would crash with
AttributeError. git history shows all committed versions were clean dicts —
the wrap came from an uncommitted writer bug, but ANY future writer bug of
this class would again take down the whole curiosity engine, the dream
simulations, homeostasis, intuition flashes, and the dashboard.

This module is the single sanctioned way to read/write the file:
  - load(): parses JSON; unwraps [dict] pairs; drops non-dict junk;
    returns (data, report) and optionally self-heals the file on disk.
  - save(): atomic tmp+replace write.

All aion modules that touch active_goals.json should use this. The file
format is otherwise unchanged: {"goals": [...], "completed": [...],
"archived": [...], "parked": [...]}.
"""
import json
import os

LIST_KEYS = ("goals", "completed", "archived", "parked")


def _unwrap_list(items):
    """Return (clean_items, unwrapped_count, dropped_count)."""
    clean, unwrapped, dropped = [], 0, 0
    for g in items:
        if isinstance(g, dict):
            clean.append(g)
        elif (isinstance(g, list) and len(g) == 1 and isinstance(g[0], dict)):
            clean.append(g[0])
            unwrapped += 1
        else:
            dropped += 1
    return clean, unwrapped, dropped


def load(path, default=None, heal=True):
    """Load the goals file with shape hardening.

    Returns (data, report) where report is a dict or None (no issues).
    If heal=True and any entry needed unwrapping/dropping, the file is
    rewritten clean immediately so subsequent readers (which may not use
    this module yet) see the fixed shape.
    """
    if default is None:
        default = {"goals": [], "completed": [], "archived": [], "parked": []}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return dict(default), None
    if not isinstance(data, dict):
        return dict(default), {"fatal": "root not a dict"}

    report = {"unwrapped": 0, "dropped": 0}
    for key in LIST_KEYS:
        v = data.get(key, [])
        if not isinstance(v, list):
            continue
        clean, uw, dr = _unwrap_list(v)
        if uw or dr:
            data[key] = clean
            report["unwrapped"] += uw
            report["dropped"] += dr
            report.setdefault("keys", []).append(key)

    if (report["unwrapped"] or report["dropped"]) and heal:
        try:
            save(path, data)
            report["healed"] = True
        except Exception:
            report["healed"] = False
    return data, (report if (report["unwrapped"] or report["dropped"]) else None)


def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    import os as _os
    _os.replace(tmp, path)
