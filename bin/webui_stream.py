#!/usr/bin/env python3
"""webui_stream.py — streaming chat endpoint + generation tap (observatory v2).

/api/chat/stream  POST {session_id, message}
    Streams JSON lines: {"text": "tok"} ... {"done": true} or {"error": "..."}
    Mirrors the dashboard server's session bookkeeping; token-level streaming
    from Ollama; pushes gen events to the live hub for other viewers.

GenTap: tails journalctl of aion dream/nightly/curiosity units and pushes
gen start/chunk/end events; also replays new dream synthesis files.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

AION = os.environ.get("AION_HOME", "$AION_HOME")
DREAMS_DIR = os.path.join(AION, "memory", "dreams")

_TAG_RULES = [
    ("dream", ("[dream]", "dream cycle", "synthesis")),
    ("nightly", ("[nightly]", "[consolidation]", "consolidat")),
    ("curiosity", ("[curiosity", "curiosity engine")),
    ("intuition", ("intuition_daemon", "warm_memory", "[intuition")),
]


def _tag_of(line):
    low = line.lower()
    for tag, keys in _TAG_RULES:
        for k in keys:
            if k in low:
                return tag
    return None


class GenTap:
    """Pushes {ch:"gen", data:{op,src,text}} hub events for LLM generations."""

    def __init__(self, push):
        self.push = push
        self._thread = None
        self._dream_t = 0.0
        self._seen = set()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="webui-gen-tap", daemon=True)
        self._thread.start()

    def _emit(self, op, src, text=""):
        try:
            self.push({"ch": "gen", "data": {"op": op, "src": src, "text": text}})
        except Exception:
            pass

    def _run(self):
        # journal tailer
        cmd = ["journalctl", "--user", "-f", "-n", "0", "-o", "cat",
               "--user-unit=aion-dream.service",
               "--user-unit=aion-nightly.service",
               "--user-unit=aion-curiosity.service",
               "--user-unit=aion-intuition-daemon.service"]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except Exception:
            p = None
        while True:
            if p is not None:
                line = p.stdout.readline()
                if line:
                    self._on_journal(line.rstrip("\n"))
                    continue
                if p.poll() is not None:
                    p = None
            self._check_dreams()
            time.sleep(2 if p is None else 0.05)

    def _on_journal(self, line):
        if not line or len(line) < 12:
            return
        tag = _tag_of(line)
        if not tag:
            return
        # skip noisy status lines that carry no LLM output
        if "all clear" in line or "injected" in line or "suppressed" in line:
            return
        if "[heartbeat]" in line:
            return
        self._emit("chunk", tag, line + "\n")

    def _check_dreams(self):
        try:
            files = sorted(
                (f for f in os.listdir(DREAMS_DIR) if f.endswith(".json")),
                key=lambda f: os.path.getmtime(os.path.join(DREAMS_DIR, f)),
                reverse=True,
            )
        except Exception:
            return
        if not files:
            return
        newest = files[0]
        path = os.path.join(DREAMS_DIR, newest)
        try:
            m = os.path.getmtime(path)
        except Exception:
            return
        if newest in self._seen or m < time.time() - 600:
            return
        self._seen.add(newest)
        if len(self._seen) > 400:
            self._seen = set(list(self._seen)[-200:])
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            return
        synth = d.get("synthesis") or ""
        if not synth:
            return
        seed = str(d.get("seed", ""))[:40]
        self._emit("start", "dream replay")
        # replay in ~120-char chunks over a few seconds
        for i in range(0, len(synth), 120):
            self._emit("chunk", "dream replay", synth[i:i + 120])
            time.sleep(0.05)
        self._emit("chunk", "dream replay", "\n\n— dream seed: " + seed + "\n")
        self._emit("end", "dream replay", "")


# ── streaming chat handler ───────────────────────────────────────
def _srv():
    """The dashboard server module (it runs as __main__)."""
    return sys.modules.get("__main__")


def _ollama_stream(messages, num_ctx, model, url, send, hpush):
    """One streaming LLM round. Returns full reply text.

    num_predict is padded to 4096: qwen3.8's thinking can consume a 2048
    budget entirely, leaving content empty (observed after large tool results).
    """
    try:
        sys.path.insert(0, str(AION) + "/bin")
        from ctx_manager import truncate_messages_str
        messages, _summary, _est = truncate_messages_str(
            messages, num_ctx - 2000, protected_prefix=1)
    except Exception:
        pass

    body = json.dumps({
        "model": model,
        "stream": True,
        "messages": messages,
        "options": {"num_ctx": num_ctx, "temperature": 0.8, "num_predict": 4096},
        "keep_alive": "30m",
    }).encode()
    rq = urllib.request.Request(url + "/api/chat", data=body,
                                headers={"Content-Type": "application/json"})
    reply = ""
    with urllib.request.urlopen(rq, timeout=180) as r:
        for line_b in r:
            line = line_b.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("done"):
                break
            tok = (j.get("message") or {}).get("content", "")
            if not tok:
                continue
            reply += tok
            send({"text": tok})
            hpush({"ch": "gen", "data": {"op": "chunk", "src": "chat", "text": tok}})
    return reply


def _ollama_retry_no_think(messages, num_ctx, model, url):
    """Non-streaming retry with think:false — used when a streamed round
    returned empty content (thinking ate the budget). Returns text."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": {"num_ctx": num_ctx, "temperature": 0.8, "num_predict": 2048},
        "keep_alive": "30m",
    }).encode()
    rq = urllib.request.Request(url + "/api/chat", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=180) as r:
        j = json.loads(r.read())
    return ((j.get("message") or {}).get("content") or "").strip()


def handle_chat_stream(handler):
    import http.server
    srv = _srv()
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        req = json.loads(raw)
    except Exception:
        req = {}
    sid = req.get("session_id")
    text = str(req.get("message", ""))

    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()

    def send(obj):
        handler.wfile.write(json.dumps(obj).encode() + b"\n")
        handler.wfile.flush()

    sessions = getattr(srv, "chat_sessions", {})
    session = sessions.get(sid)
    if session is None or not text.strip():
        send({"error": "session not found" if sid else "no message"})
        return
    # Lazy session-start log (defined in aion_dashboard_server); a streamed
    # message can be the first real message of an eagerly-started session.
    _log_start = getattr(srv, "_log_session_start", None)
    if _log_start:
        _log_start(sid)

    session["messages"].append({"role": "user", "content": text})
    session["operator_messages"] += 1
    session["last_active"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

    def hpush(ev):
        h = getattr(_srv(), "WEBUI_HUB", None)
        if h is not None:
            try:
                h._push(ev)
            except Exception:
                pass

    hpush({"ch": "gen", "data": {"op": "start", "src": "chat"}})

    num_ctx = getattr(srv, "NUM_CTX", 32768)
    model = getattr(srv, "MAIN_MODEL", "qwen3.8:27b")
    url = getattr(srv, "MAIN_URL", "http://127.0.0.1:11436")

    max_rounds = getattr(srv, "CHAT_TOOL_MAX_ROUNDS", 3)
    parse_tool_calls = getattr(srv, "parse_tool_calls", None)
    run_tool_rounds = getattr(srv, "run_tool_rounds", None)

    reply = ""
    tools_used = []
    try:
        for _round in range(max_rounds):
            reply = _ollama_stream(list(session["messages"]), num_ctx, model,
                                   url, send, hpush)
            if not reply.strip():
                # thinking ate the budget — retry once without thinking
                print("[chat/stream] empty round — retrying with think:false")
                reply = _ollama_retry_no_think(list(session["messages"]), num_ctx,
                                               model, url)
                if reply:
                    send({"text": reply})
                    hpush({"ch": "gen", "data": {"op": "chunk", "src": "chat", "text": reply}})
            calls = parse_tool_calls(reply) if parse_tool_calls else []
            valid = [c for c in calls if c[0]] if calls else []
            if not valid or not run_tool_rounds:
                break
            # visible progress marker (display only — not stored in history)
            cap = getattr(srv, "CHAT_TOOL_MAX_CALLS_PER_ROUND", 2)
            for name, targs in valid[:cap]:
                note = targs.get("url") or targs.get("path") or name
                send({"text": "\n\n[" + name + " → " + str(note)[:80] + "]\n"})
            print("[chat/stream] tool round %d: %s" % (_round + 1, [c[0] for c in valid[:cap]]))
            _, used = run_tool_rounds(session, reply)
            tools_used += used
            send({"text": "\n\n"})  # separator before continuation
        else:
            # budget exhausted — final round, strip any residual tool calls
            reply = _ollama_stream(list(session["messages"]), num_ctx, model,
                                   url, send, hpush)
            if parse_tool_calls and parse_tool_calls(reply):
                stripped = re.sub(r"<tool>.*?</tool>", "", reply, flags=re.S).strip()
                if stripped:
                    reply = stripped
    except Exception as e:
        send({"error": str(e)})
        hpush({"ch": "gen", "data": {"op": "end", "src": "chat", "ok": False}})
        return

    # Store final reply (intermediate tool turns already stored by run_tool_rounds)
    if not reply.strip():
        reply = "(I got lost in thought and came back with nothing — say that again?)"
        send({"text": reply})
    msgs = session["messages"]
    if not (msgs and msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == reply):
        msgs.append({"role": "assistant", "content": reply})
    session["aion_messages"] += 1
    session["last_active"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    send({"done": True})
    hpush({"ch": "gen", "data": {"op": "end", "src": "chat", "ok": True}})

    # Episodic logging (V4.2 — streaming chats previously were not logged,
    # so consolidation never saw them). Tool fetches are logged separately
    # inside web_fetch.fetch_url / fetch_github.
    try:
        log_episodic = getattr(srv, "log_episodic", None)
        if log_episodic:
            log_episodic("operator_chat", "operator: " + text[:200], {
                "session_id": sid, "role": "operator", "stream": True,
            })
            log_episodic("operator_chat", reply[:2000], {
                "session_id": sid, "role": "aion", "tools_used": tools_used,
            })
    except Exception:
        pass

    # keep the V3.5 sandbox-request behavior from send_chat_message
    try:
        sys.path.insert(0, str(AION) + "/bin")
        import sandbox as sandbox_mod
        parsed = sandbox_mod.parse_sandbox_request(reply)
        if parsed:
            code, lang, desc = parsed
            print("[chat/stream] sandbox request: %s" % str(desc)[:60])
            sandbox_mod.run_sandbox(code, lang=lang, description=desc)
    except Exception:
        pass
