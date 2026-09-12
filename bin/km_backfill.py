#!/usr/bin/env python3
"""km_backfill.py — one-shot merge of near-duplicate claims.

The digest-vs-digest bug (fixed Aug 28) created a new claim per nightly
re-ingest for any observation longer than 200 chars: 705 motor claims for
~50 unique physical observations. This script re-clusters claims within
each domain using the SAME digest-vs-digest Jaccard rule the fixed
_find_or_create_claim now uses, pools their evidence (deduped by
kind+session+detail), records the merge in history, and drops the phantom.

Run:  python3 bin/km_backfill.py [--apply]
      (default is dry-run: prints what would merge)
"""
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

AION = os.environ.get("AION_HOME", os.path.expanduser("~/aion"))
MATURITY_FILE = f"{AION}/memory/state/knowledge_maturity.json"
THRESHOLD = 0.6


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _digest(text, limit=200):
    return re.sub(r"\s+", " ", text).strip()[:limit]


def main():
    apply = "--apply" in sys.argv
    data = json.load(open(MATURITY_FILE))
    claims = data["claims"]
    before = len(claims)

    # canonical = earliest claim id per duplicate group; walk sorted ids
    ids = sorted(claims.keys())
    canon_of = {}          # duplicate id -> canonical id
    canon_list = []        # canonical ids in insertion order
    merges = 0

    for cid in ids:
        c = claims[cid]
        dig = _digest(c.get("text_digest", ""))
        target = None
        best = 0.0
        for kid in canon_list:
            k = claims[kid]
            if k.get("domain") != c.get("domain"):
                continue
            s = _overlap(_digest(k.get("text_digest", "")), dig)
            if s > best:
                best, target = s, kid
        if target and best > THRESHOLD:
            canon_of[cid] = target
            merges += 1
        else:
            canon_list.append(cid)

    if not apply:
        print("DRY RUN: %d claims -> %d canonical (%d merges)" % (before, len(canon_list), merges))
        # show biggest merge groups
        groups = {}
        for dup, kid in canon_of.items():
            groups.setdefault(kid, []).append(dup)
        for kid, dups in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:10]:
            print("  %s absorbs %d: %s" % (kid, len(dups), ", ".join(dups[:4]) + ("..." if len(dups) > 4 else "")))
        return 0

    ts = datetime.now(timezone.utc).isoformat()
    bak = MATURITY_FILE + ".bak-backfill-" + ts[:10]
    shutil.copy2(MATURITY_FILE, bak)

    for dup, kid in canon_of.items():
        d, k = claims[dup], claims[kid]
        seen = {(e.get("kind"), e.get("session"), e.get("detail")) for e in k["evidence"]}
        for e in d["evidence"]:
            key = (e.get("kind"), e.get("session"), e.get("detail"))
            if key not in seen:
                k["evidence"].append(e)
                seen.add(key)
        k["history"].append({
            "ts": ts, "from": k["status"], "to": k["status"],
            "why": "backfill: merged duplicate claim %s (%d evidence items pooled)"
                   % (dup, len(d["evidence"]))})
        del claims[dup]

    data["claims"] = claims
    data["last_backfill"] = ts
    json.dump(data, open(MATURITY_FILE, "w"), indent=1)
    print("APPLIED: %d -> %d claims (%d merged). Backup: %s"
          % (before, len(claims), merges, bak))
    print("Now run: python3 bin/knowledge_maturity.py --promote")
    return 0


if __name__ == "__main__":
    sys.exit(main())