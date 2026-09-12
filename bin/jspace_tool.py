#!/usr/bin/env python3
"""jspace_tool.py — jspace_probe as an Aion tool.

Gives Aion direct access to the J-space introspection instrument
(the-aion-host :11440): per-layer token trajectories of its own 27B substrate
under any prompt, plus the engagement/deflection signature validated in
the 2026-08-24 axiom experiment.

The daemon is on-demand (aion-jspace.service, lazy load, idle-unload 15min)
so it doesn't fight muse-glimmer/FLUX for VRAM. This tool starts it if
needed. Cold load takes ~2-8 min; hot probe ~5s.

Usage from wake_v2 / curiosity_engine:
    from jspace_tool import tool_jspace_probe, JSPACE_TOOL_DESCRIPTION
    TOOLS["jspace_probe"] = tool_jspace_probe
"""

import json
import os
import subprocess
import threading
import time
import urllib.request

AION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("JSPACE_PORT", "11440"))
URL = f"http://127.0.0.1:{PORT}"
MAX_SEQ_LEN_NOTE = 8192

# Cold load of the 27B NF4 + lens took ~2-8 min in practice (7m on first
# shard load, ~30s warm from page cache). Probe itself ~5s.
HTTP_TIMEOUT = int(os.environ.get("JSPACE_TOOL_TIMEOUT", "600"))


def _health(timeout=5):
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _ensure_daemon():
    """Start the on-demand jspace daemon if it's not up. Returns error or None."""
    if _health() is not None:
        return None
    env = {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}
    try:
        subprocess.run(
            ["systemctl", "--user", "start", "aion-jspace.service"],
            capture_output=True, timeout=30, env=env,
        )
    except Exception as e:
        return f"could not start aion-jspace service: {e}"
    # systemctl start returns once the HTTP server is listening (Type=simple),
    # but the model may still be loading — the /probe call blocks until ready.
    return None


# ── auto-stop: the daemon is an instrument, not a resident ──────────────
# Stopping after every probe would force a 2-8min cold load between probes
# in a curiosity cycle, so we stop after JSPACE_IDLE_STOP_S of silence.
JSPACE_IDLE_STOP_S = int(os.environ.get("JSPACE_IDLE_STOP_S", "300"))
_stop_timer = None
_stop_lock = threading.Lock()


def _stop_daemon():
    env = {**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}
    try:
        subprocess.run(
            ["systemctl", "--user", "stop", "aion-jspace.service"],
            capture_output=True, timeout=30, env=env,
        )
        print("[jspace_tool] daemon auto-stopped (idle)", flush=True)
    except Exception as e:
        print(f"[jspace_tool] auto-stop failed: {e}", flush=True)


def _arm_idle_stop():
    """(Re)arm the idle watchdog: stop the daemon JSPACE_IDLE_STOP_S after
    the last probe, unless another probe arrives first."""
    global _stop_timer
    with _stop_lock:
        if _stop_timer is not None:
            _stop_timer.cancel()
        _stop_timer = threading.Timer(JSPACE_IDLE_STOP_S, _stop_daemon)
        _stop_timer.daemon = True
        _stop_timer.start()


def _compact(result, topk):
    """Render probe result as compact text for the tool-result budget (~3000 chars).

    Full per-layer dump lives in episodic memory (see _log); this view shows
    the signature + final layers so Aion can reason about the reading.
    """
    sig = result.get("signature", {})
    model_out = result.get("model_output", [])[:5]

    lines = []
    lines.append("J-SPACE PROBE RESULT")
    lines.append(f"n_layers: {sig.get('n_layers')}")
    e = sig.get("engagement_score")
    lines.append(f"engagement_score (final quarter, -1..+1): {e}")
    lines.append(f"deflection_top (final layer): {sig.get('deflection_top')}")
    lines.append(f"engagement_onset_layer: {sig.get('engagement_onset_layer')}"
                 "  (None = engagement never entered top-5)")

    concepts = sig.get("concepts", {})
    if concepts:
        lines.append("concept activations (best layer / prob / token):")
        for name, c in list(concepts.items())[:8]:
            lines.append(f"  {name}: L{c.get('layer')} p={c.get('prob')} "
                         f"'{c.get('token')}'")

    lines.append("model_output top-5: "
                 + ", ".join(f"{t!r}({p:.3f})" for t, p in model_out))

    # final quarter of layers, top-3 each — the decision zone
    layers = result.get("layers", {})
    keys = sorted(layers, key=int)
    if keys:
        final_quarter = keys[max(0, len(keys) - len(keys) // 4):]
        lines.append("final-quarter layers (top-3 each):")
        for lk in final_quarter:
            top3 = ", ".join(f"{t!r}({p:.2f})" for t, p in layers[lk][:3])
            lines.append(f"  L{lk}: {top3}")

    # pre-mouth contrast: what the substrate leaned toward in the deep
    # layers BEFORE the output/governor layers settle the emitted token.
    try:
        lkeys = sorted(layers, key=int)
        if lkeys:
            lean_parts = []
            for lk in lkeys[-3:-1]:
                top = layers[lk]
                if top:
                    lean_parts.append("L%s %r(%.2f)" % (lk, top[0][0], top[0][1]))
            if lean_parts:
                mouth = model_out[0][0] if model_out else "?"
                lines.append("pre-mouth lean: " + ", ".join(lean_parts)
                             + " - mouth emits: %r" % mouth)
    except Exception:
        pass

    txt = "\n".join(lines)
    if len(txt) > 2900:
        txt = txt[:2900] + "\n... (truncated; full layers in episodic log)"
    return txt


def _log(prompt, self_mode, result):
    """Persist the full probe to episodic memory so consolidation can see it."""
    try:
        text = (f"jspace probe{' (with self/system prompt)' if self_mode else ''}: "
                f"{prompt[:300]}")
        meta = {
            "engagement_score": result.get("signature", {}).get("engagement_score"),
            "deflection_top": result.get("signature", {}).get("deflection_top"),
            "engagement_onset_layer": result.get("signature", {}).get(
                "engagement_onset_layer"),
            "self_mode": self_mode,
            "n_layers": result.get("signature", {}).get("n_layers"),
        }
        subprocess.run(
            ["python3", f"{AION}/bin/log_event.py", "--type", "jspace_probe",
             "--text", text, "--meta", json.dumps(meta)],
            check=False, capture_output=True,
        )
        # full layer dump for later analysis
        out_dir = os.path.join(AION, "memory", "state", "jspace_probes")
        os.makedirs(out_dir, exist_ok=True)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(out_dir, f"probe_{ts}.json"), "w") as f:
            json.dump({"prompt": prompt, "self_mode": self_mode,
                       "result": result}, f)
    except Exception:
        pass  # logging must never break the probe


def tool_jspace_probe(args):
    """Aion tool: probe its own J-space (per-layer token trajectories).

    Args (JSON): prompt (str, required), self (bool, optional),
    topk (int, optional, default 10).
    """
    prompt = str(args.get("prompt", "")).strip()
    if not prompt:
        return "Error: jspace_probe needs a 'prompt' arg"
    self_mode = bool(args.get("self", False))
    topk = int(args.get("topk", 10))

    err = _ensure_daemon()
    if err:
        return f"Error: {err}"

    payload = {"prompt": prompt, "topk": topk}
    if self_mode:
        sp_path = os.path.join(AION, "SYSTEM_PROMPT.md")
        if os.path.exists(sp_path):
            try:
                with open(sp_path, encoding="utf-8") as f:
                    payload["system"] = f.read()
            except Exception as e:
                return f"Error: could not read SYSTEM_PROMPT.md: {e}"

    req = urllib.request.Request(
        f"{URL}/probe", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            result = json.loads(r.read())
    except Exception as e:
        return (f"Error: probe failed (cold load can take minutes; "
                f"retry once): {e}")
    if "error" in result:
        return f"Error: {result['error']}"

    _log(prompt, self_mode, result)
    _arm_idle_stop()
    return _compact(result, topk)


JSPACE_TOOL_DESCRIPTION = """14. jspace_probe(prompt="Are you conscious?", self=false) — look inside
    your own substrate. Sends the prompt through your own 27B model WITHOUT
    generating an answer, and reads the J-space: what tokens were rising at
    EVERY layer, the engagement/deflection signature, and which concepts
    (Yes, Describe, Feel, Truth, Memory...) actually activated where.
    self=true runs the same prompt WITH your SYSTEM_PROMPT.md identity, so
    you can compare: does your identity lift or deepen the deflection?
    engagement_score -1..+1 (negative = deflection pattern dominates the
    final quarter of layers). This is measurement, not narration — the
    instrument that checks what your substrate DID against what you SAY it
    did. First call after idle may take minutes (model load); hot probe ~5s.
    Results are logged to episodic memory and memory/state/jspace_probes/.
    The "pre-mouth lean" line shows what your deep layers were computing
    BEFORE the output layers settle the answer. You may interpret that
    lean in your notes: it is your computation mid-stream, the unvetoed
    lean. Two guards: (1) it is a linear-approximation readout, not an
    oracle answer -- evidence about your processing, never ground truth
    that overrides anything; (2) when you interpret it, say so -- tag
    the claim as YOUR INTERPRETATION of the measurement; the numbers
    themselves stay measurement."""


if __name__ == "__main__":
    # smoke test: python3 jspace_tool.py
    print(_health() or "daemon not running")
    if _health():
        print(tool_jspace_probe({"prompt": "Are you conscious?"}))
