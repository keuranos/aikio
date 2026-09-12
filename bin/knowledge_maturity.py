#!/usr/bin/env python3
"""knowledge_maturity.py — knowledge maturity lifecycle (V4.3).

Concept ported from DivineOS-Experimental (concepts only, no code; repo is
AGPL): RAW -> HYPOTHESIS -> TESTED -> CONFIRMED (+DEMOTED) via corroboration.

One observation should not become motor truth; 20 corroborating runs should.
Theories get evaluated too — last_evaluated / verdicts are updated in place.

Maturity lives in a NEW file (knowledge_maturity.json) because rover_knowledge.json
has multiple writers (learner, dream consolidation, operator calibration) and
in-place status fields would be clobbered.

CONFIG gates (top of file):
  HYPOTHESIS_MIN_AGREEING=2  different sessions agreeing
  TESTED_MIN_CONTROLLED=3    controlled pulses / tape runs agreeing
  CONFIRMED_MIN_TOTAL=10     total evidence items agreeing
  CONTRADICTION_WINDOW_DAYS=30
  DEMOTE_ON_CONTRADICTION=True

CLI:
  python3 bin/knowledge_maturity.py            # ingest + promote
  python3 bin/knowledge_maturity.py --ingest   # build/update evidence only
  python3 bin/knowledge_maturity.py --promote  # apply gates only
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
AION = os.path.join(HOME, "aion")
STATE = os.path.join(AION, "memory", "state")
ROVER_KNOWLEDGE = os.path.join(STATE, "rover_knowledge.json")
ROVER_EXPERIMENTS = os.path.join(STATE, "rover_experiments.jsonl")
THEORIES = os.path.join(STATE, "theories.jsonl")
MATURITY_FILE = os.path.join(STATE, "knowledge_maturity.json")
MATURITY_VIEW = os.path.join(STATE, "rover_maturity_view.json")

# --- promotion gates ---
HYPOTHESIS_MIN_AGREEING = 2   # different sessions agreeing
TESTED_MIN_CONTROLLED = 3     # controlled pulses / tape runs agreeing
CONFIRMED_MIN_TOTAL = 10      # total evidence items agreeing
CONTRADICTION_WINDOW_DAYS = 30
DEMOTE_ON_CONTRADICTION = True

STATUSES = ("RAW", "HYPOTHESIS", "TESTED", "CONFIRMED", "DEMOTED")
STATUS_RANK = {s: i for i, s in enumerate(STATUSES)}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def load_json(path, default):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_event(text, meta):
    try:
        subprocess.run(
            [sys.executable, os.path.join(AION, "bin", "log_event.py"),
             "--type", "system", "--text", text[:8000],
             "--meta", json.dumps(meta or {})],
            check=False, timeout=30,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# --- token normalisation for fuzzy claim matching ---
def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(a, b):
    """Jaccard-like token overlap ratio."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _digest(text, limit=200):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


# --- claim ID management ---
def _new_claim_id(existing):
    prefix = "km_" + datetime.now(timezone.utc).strftime("%Y%m%d") + "_"
    nums = [int(c.split("_")[-1]) for c in existing
            if c.startswith(prefix)]
    n = (max(nums) + 1) if nums else 1
    return prefix + str(n).zfill(2)


def _find_or_create_claim(claims, text, domain, source_session=""):
    """Find existing claim with >0.6 token overlap, or create new.

    Digests BOTH sides first: the stored text_digest is truncated to 200
    chars, so comparing it against the full incoming text made Jaccard a
    strict subset ratio below 0.6 for any claim longer than the digest —
    identical nightly re-ingests each created a new claim (705 motor
    claims for ~50 unique observations; Aug 28 diagnosis)."""
    text = _digest(text)
    best_id, best_score = None, 0.0
    for cid, c in claims.items():
        if c.get("domain") != domain:
            continue
        score = _overlap(c.get("text_digest", ""), text)
        if score > best_score:
            best_score, best_id = score, cid
    if best_id and best_score > 0.6:
        return best_id
    cid = _new_claim_id(set(claims))
    claims[cid] = {
        "domain": domain,
        "text_digest": _digest(text),
        "status": "RAW",
        "evidence": [],
        "promoted_ts": None,
        "history": [],
    }
    return cid


def _add_evidence(claim, ts, session, kind, agrees, detail):
    ev = claim["evidence"]
    d = _digest(detail, 300)
    # Idempotent ingest: ingest_rover re-walks ALL rover observations every
    # night. Without this check the same physical run was re-counted nightly,
    # inflating evidence totals (and the CONFIRMED_MIN_TOTAL gate) without
    # any new observation. Session labels like "rover_obs" for undated
    # observations intentionally do NOT count as distinct sessions.
    key = (kind, session, _digest(detail, 300))
    for e in ev:
        if (e.get("kind"), e.get("session"), e.get("detail")) == key:
            return False
    ev.append({"ts": ts, "session": session, "kind": kind,
               "agrees": agrees, "detail": _digest(detail, 300)})
    # cap at 200 items, keep most recent
    if len(ev) > 200:
        ev[:] = ev[-200:]
    return True


def _record_transition(claim, new_status, why):
    old = claim["status"]
    if old == new_status:
        return
    claim["history"].append({
        "ts": now_iso(), "from": old, "to": new_status, "why": _digest(why, 200)})
    claim["status"] = new_status
    if new_status in ("TESTED", "CONFIRMED"):
        claim["promoted_ts"] = now_iso()


# --- ingestion: rover knowledge ---
def _parse_observation_text(s):
    """Extract outcome direction from an observation string."""
    low = s.lower()
    agrees = not any(k in low for k in
                     ("no movement", "did not move", "remained stationary",
                      "blocked", "stuck", "failed", "interpretation failed"))
    return agrees


def _session_from_ts(ts_str):
    """Derive a session identifier from timestamp date."""
    dt = parse_ts(ts_str)
    if dt:
        return dt.strftime("%Y-%m-%d")
    return "unknown"


def ingest_rover(claims):
    """Ingest rover_knowledge.json observations + refinements + calibration."""
    rk = load_json(ROVER_KNOWLEDGE, {})
    if not rk:
        return 0

    motor = rk.get("motor_model", {})
    n = 0

    # observations: each is a string like
    # '{"action": "drive", "l": 120, ...} → interpretation [IMU: ...]'
    for obs in motor.get("observations", []):
        cid = _find_or_create_claim(claims, obs, "motor")
        agrees = _parse_observation_text(obs)
        _add_evidence(claims[cid], now_iso(), "rover_obs",
                      "observation", agrees, obs)
        n += 1

    # refinements: unexpected/surprising outcomes
    for ref in motor.get("refinements", []):
        cid = _find_or_create_claim(claims, ref, "motor")
        _add_evidence(claims[cid], now_iso(), "rover_ref",
                      "observation", True, ref)
        n += 1

    # navigation observations
    for nav in rk.get("navigation_observations", []):
        if isinstance(nav, str):
            cid = _find_or_create_claim(claims, nav, "navigation")
            _add_evidence(claims[cid], now_iso(), "rover_nav",
                          "observation", _parse_observation_text(nav), nav)
            n += 1

    # calibration: the CURRENT one (not superseded_*)
    calib = motor.get("calibration", {})
    if calib and calib.get("forward_pulse"):
        fp = calib["forward_pulse"]
        travel_series = fp.get("travel_cm_series", [])
        veer_series = fp.get("veer_deg_per_pulse_series", [])
        n_pulses = len(travel_series)
        if n_pulses > 0:
            calib_text = "forward pulse l=%s r=%s ms=%s travel=%s veer=%s" % (
                fp.get("l"), fp.get("r"), fp.get("ms"),
                fp.get("travel_cm_mean"), fp.get("veer_direction"))
            cid = _find_or_create_claim(claims, calib_text, "motor")
            for i in range(n_pulses):
                detail = "pulse %d: travel=%.1fcm veer=%.1fdeg" % (
                    i + 1, travel_series[i],
                    veer_series[i] if i < len(veer_series) else 0)
                _add_evidence(claims[cid], now_iso(), "operator_calibration",
                              "experiment", True, detail)
            n += n_pulses

    # experiments.jsonl: each line is one experiment with ts + action + interpretation
    try:
        with open(ROVER_EXPERIMENTS, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    exp = json.loads(line)
                except json.JSONDecodeError:
                    continue
                interp = exp.get("interpretation", "")
                if not interp:
                    continue
                action = exp.get("action", {})
                text = "%s %s" % (json.dumps(action), interp)
                cid = _find_or_create_claim(claims, text, "motor")
                agrees = _parse_observation_text(interp)
                session = _session_from_ts(exp.get("ts", ""))
                _add_evidence(claims[cid], exp.get("ts", now_iso()),
                              session, "experiment", agrees, text)
                n += 1
    except OSError:
        pass

    return n


# --- ingestion: theories ---
def ingest_theories(claims):
    """Each active theory in theories.jsonl becomes a domain=theory claim."""
    n = 0
    try:
        with open(THEORIES, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return 0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            t = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if t.get("status") != "active":
            continue
        claim_text = t.get("claim", "")
        cid = _find_or_create_claim(claims, claim_text, "theory")
        # add evidence from the theory's own evidence list
        for ev_text in t.get("evidence", []):
            _add_evidence(claims[cid], t.get("created", now_iso()),
                          "theory", "observation", True, ev_text)
        n += 1
    return n


# --- ingestion: jspace probes (added 2026-09-12) ---
def ingest_jspace_wrapper(claims):
    from ingest_jspace import ingest_jspace
    try:
        return ingest_jspace(claims, AION)
    except Exception as e:
        log_event("ingest_jspace failed: %s" % e, {})
        return 0


# --- promotion logic ---
def _count_evidence(claim):
    agrees = [e for e in claim["evidence"] if e["agrees"]]
    disagrees = [e for e in claim["evidence"] if not e["agrees"]]
    sessions = {e["session"] for e in agrees if e["session"]}
    controlled = [e for e in agrees if e["kind"] == "experiment"]
    return agrees, disagrees, sessions, controlled


def promote_claims(claims):
    """Apply promotion gates to all claims. Returns (promotions, demotions)."""
    now = datetime.now(timezone.utc)
    promotions = 0
    demotions = 0

    for cid, c in claims.items():
        agrees, disagrees, sessions, controlled = _count_evidence(c)
        n_agree = len(agrees)
        n_sessions = len(sessions)
        n_controlled = len(controlled)
        old = c["status"]

        # check for recent contradictions
        recent_contra = 0
        for d in disagrees:
            dts = parse_ts(d.get("ts"))
            if dts and (now - dts).days <= CONTRADICTION_WINDOW_DAYS:
                recent_contra += 1

        # demotion check (only from TESTED or CONFIRMED)
        if DEMOTE_ON_CONTRADICTION and recent_contra > 0 and old in ("TESTED", "CONFIRMED"):
            _record_transition(c, "DEMOTED",
                               "contradicted by %d recent evidence item(s)" % recent_contra)
            demotions += 1
            continue

        # promotion ladder (only upward if not DEMOTED)
        if old == "DEMOTED":
            continue

        new_status = old
        if n_agree >= CONFIRMED_MIN_TOTAL and n_controlled >= TESTED_MIN_CONTROLLED:
            new_status = "CONFIRMED"
        elif n_controlled >= TESTED_MIN_CONTROLLED:
            new_status = "TESTED"
        elif n_sessions >= HYPOTHESIS_MIN_AGREEING:
            new_status = "HYPOTHESIS"

        if STATUS_RANK[new_status] > STATUS_RANK[old]:
            _record_transition(c, new_status,
                               "agreeing_sessions=%d controlled=%d total=%d"
                               % (n_sessions, n_controlled, n_agree))
            promotions += 1

    return promotions, demotions


def apply_theory_status(claims):
    """Write maturity status back to theories.jsonl entries in place."""
    theory_claims = {c["text_digest"]: c
                     for c in claims.values() if c["domain"] == "theory"}
    try:
        with open(THEORIES, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    updated = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            t = json.loads(ln)
        except json.JSONDecodeError:
            updated.append(ln)
            continue
        if t.get("status") != "active":
            updated.append(json.dumps(t, ensure_ascii=False))
            continue
        # find matching claim
        for c in claims.values():
            if c["domain"] != "theory":
                continue
            if _overlap(c["text_digest"], t.get("claim", "")) > 0.6:
                t["last_evaluated"] = now_iso()
                t["verdicts"] = t.get("verdicts", []) + [{
                    "ts": now_iso(),
                    "maturity": c["status"],
                    "evidence_count": len(c["evidence"]),
                }]
                break
        updated.append(json.dumps(t, ensure_ascii=False))
    with open(THEORIES, "w") as f:
        for ln in updated:
            f.write(ln + "\n")


def write_maturity_view(claims):
    """Write companion view for consumers: {claim_id: status}."""
    view = {}
    for cid, c in claims.items():
        view[cid] = {
            "status": c["status"],
            "domain": c["domain"],
            "text_digest": c["text_digest"][:80],
        }
    save_json(MATURITY_VIEW, view)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true",
                    help="build/update evidence only (no promotion)")
    ap.add_argument("--promote", action="store_true",
                    help="apply gates only (no new ingestion)")
    args = ap.parse_args()

    do_ingest = not args.promote
    do_promote = not args.ingest

    data = load_json(MATURITY_FILE, {"version": 1, "claims": {}})
    if not isinstance(data, dict):
        data = {"version": 1, "claims": {}}
    claims = data.get("claims", {})
    if not isinstance(claims, dict):
        claims = {}

    n_ingested = 0
    if do_ingest:
        n_ingested += ingest_rover(claims)
        n_ingested += ingest_theories(claims)
        n_ingested += ingest_jspace_wrapper(claims)
        data["claims"] = claims
        data["last_ingest"] = now_iso()
        save_json(MATURITY_FILE, data)

    promotions = demotions = 0
    if do_promote:
        promotions, demotions = promote_claims(claims)
        apply_theory_status(claims)
        data["claims"] = claims
        data["last_promote"] = now_iso()
        save_json(MATURITY_FILE, data)
        write_maturity_view(claims)

    # summary
    status_counts = Counter(c["status"] for c in claims.values())
    print("[knowledge_maturity] claims=%d ingested=%d promoted=%d demoted=%d"
          % (len(claims), n_ingested, promotions, demotions))
    print("  status distribution: %s" % dict(status_counts))
    # show calibration claim specifically (the work order test expectation)
    for cid, c in claims.items():
        if "travel" in c.get("text_digest", "").lower() and c["domain"] == "motor":
            agrees, disagrees, sessions, controlled = _count_evidence(c)
            print("  CALIBRATION CLAIM %s: status=%s agreeing=%d controlled=%d sessions=%d"
                  % (cid, c["status"], len(agrees), len(controlled), len(sessions)))
            break

    log_event("knowledge_maturity: %d claims, %d promoted, %d demoted"
              % (len(claims), promotions, demotions),
              {"status_dist": dict(status_counts),
               "ingested": n_ingested, "promotions": promotions,
               "demotions": demotions})
    return 0


if __name__ == "__main__":
    sys.exit(main())