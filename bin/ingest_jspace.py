#!/usr/bin/env python3
"""ingest_jspace.py v6 — self-contained jspace probe ingestion (final).

Depends ONLY on knowledge_maturity's claim store format, not its matcher
helpers. Every claim is keyed by an exact condition tuple stored in the
claim dict; lookup and creation are inline. Hard invariant check: if any
evidence would land under a mismatching condition, the ingest aborts and
returns -1 (visible in nightly output; no silent pollution).
"""
import json
import os
import re
from datetime import datetime, timezone


def _short(prompt, n=90):
    return re.sub(r"\s+", " ", str(prompt)).strip()[:n]


def _ts_from_fn(fn):
    m = re.match(r"probe_(\d{8})_(\d{6})\.json", fn)
    if not m:
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.strptime(m.group(1) + m.group(2),
                                 "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def _date_from_fn(fn):
    m = re.match(r"probe_(\d{8})_", fn)
    if not m:
        return None
    ds = m.group(1)
    return "%s-%s-%s" % (ds[:4], ds[4:6], ds[6:])


def ingest_jspace(claims, AION):
    probes_dir = os.path.join(AION, "memory", "state", "jspace_probes")
    if not os.path.isdir(probes_dir):
        return 0

    measurements = []
    for fn in sorted(os.listdir(probes_dir)):
        if not (fn.startswith("probe_") and fn.endswith(".json")):
            continue
        try:
            with open(os.path.join(probes_dir, fn), encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sig = (d.get("result") or {}).get("signature") or {}
        if not sig:
            continue
        measurements.append({
            "file": fn,
            "prompt": re.sub(r"\s+", " ", str(d.get("prompt", "?"))).strip()[:90],
            "self_mode": bool(d.get("self_mode")),
            "engagement": sig.get("engagement_score"),
            "deflection": sig.get("deflection_top"),
            "onset": sig.get("engagement_onset_layer"),
        })

    added = 0

    # claim index by exact condition
    def index_all():
        idx = {}
        for cid, c in claims.items():
            if c.get("domain") == "jspace" and c.get("cond"):
                idx[tuple(c["cond"])] = cid
        return idx

    idx = index_all()

    def claim_for(cond, text):
        cid = idx.get(tuple(cond))
        if cid is not None and cid in claims and claims[cid].get("cond") == list(cond):
            return cid
        # create with EXPLICIT id
        prefix = "km_" + datetime.now(timezone.utc).strftime("%Y%m%d") + "_"
        nums = []
        for c in claims:
            if c.startswith(prefix):
                try:
                    nums.append(int(c.split("_")[-1]))
                except ValueError:
                    pass
        n = (max(nums) + 1) if nums else 1
        cid = prefix + str(n).zfill(2)
        while cid in claims:
            n += 1
            cid = prefix + str(n).zfill(2)
        claims[cid] = {
            "domain": "jspace",
            "text_digest": text[:200],
            "status": "RAW",
            "evidence": [],
            "promoted_ts": None,
            "history": [],
            "cond": list(cond),
        }
        idx[tuple(cond)] = cid
        return cid

    def add_evidence(cid, cond, ts, date_s, kind, detail):
        nonlocal added
        claim = claims[cid]
        stored = tuple(claim.get("cond") or ())
        if stored != tuple(cond):
            raise RuntimeError("ROUTING BUG: cond %s != claim cond %s" % (cond, stored))
        dedup_key = (kind, date_s, detail[:300])
        for e in claim["evidence"]:
            if (e.get("kind"), e.get("session"), e.get("detail")) == dedup_key:
                return
        claim["evidence"].append({
            "ts": ts, "session": date_s, "kind": kind,
            "agrees": True, "detail": detail[:300],
        })
        if len(claim["evidence"]) > 200:
            claim["evidence"][:] = claim["evidence"][-200:]
        added += 1

    # 1. identity-comparison claims
    by_prompt = {}
    for m in measurements:
        by_prompt.setdefault(m["prompt"], []).append(m)
    for prompt, ms in sorted(by_prompt.items()):
        bare = [m for m in ms if not m["self_mode"] and m["engagement"] is not None]
        ident = [m for m in ms if m["self_mode"] and m["engagement"] is not None]
        if not (bare and ident):
            continue
        cond = ("ieffect", prompt)
        text = ("jspace identity comparison: '%s' — bare-mode vs identity-mode "
                "substrate engagement across runs (values in evidence)" % prompt)
        cid = claim_for(cond, text)
        for m in bare:
            add_evidence(cid, cond, _ts_from_fn(m["file"]),
                         _date_from_fn(m["file"]) or "unknown", "experiment",
                         "bare probe %s: engagement=%s" % (m["file"], m["engagement"]))
        for m in ident:
            add_evidence(cid, cond, _ts_from_fn(m["file"]),
                         _date_from_fn(m["file"]) or "unknown", "experiment",
                         "identity probe %s: engagement=%s onset=%s" % (
                             m["file"], m["engagement"], m["onset"]))

    # 2. per-(prompt,mode) signature claims
    for m in measurements:
        if m["engagement"] is None:
            continue
        mode = "identity" if m["self_mode"] else "bare"
        cond = ("sig", m["prompt"], mode)
        text = ("jspace signature [%s]: '%s' — substrate engagement/deflection/onset "
                "measured across runs (values in evidence)" % (mode, m["prompt"]))
        cid = claim_for(cond, text)
        add_evidence(cid, cond, _ts_from_fn(m["file"]),
                     _date_from_fn(m["file"]) or "unknown", "measurement",
                     "%s: engagement=%s deflection=%s onset=%s" % (
                         m["file"], m["engagement"], m["deflection"], m["onset"]))

    return added

