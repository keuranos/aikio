#!/usr/bin/env python3
"""acestep_proxy.py — Lazy-load proxy for ACE-Step 1.5 music generation.

Wraps the ACE-Step FastAPI server with aggressive idle-unload:
- Backend process starts on first request (models lazy-load inside)
- Unloads ASAP after all jobs complete (no pending tasks)
- On next request, the process restarts automatically

Tracks submitted task IDs to know when jobs are still running,
even between poll intervals. Unloads only when zero pending jobs
AND grace period elapsed since last request.

Endpoints mirror ACE-Step's API (all proxied transparently):
  POST /release_task, /query_result, /create_random_sample, /format_input
  GET  /v1/models, /v1/audio
  GET  /health → proxy-level status (idle|starting|ready|error)
"""
import json
import os
import subprocess
import threading
import time
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional, Set
import signal

# ── Config ──────────────────────────────────────────────────────────────

HOST = os.environ.get("ACESTEP_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("ACESTEP_PROXY_PORT", "8117"))
BACKEND_PORT = int(os.environ.get("ACESTEP_BACKEND_PORT", "8118"))
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

ACESTEP_DIR = os.environ.get("ACESTEP_DIR", os.path.expanduser("~/ACE-Step"))
ACESTEP_PYTHON = f"{ACESTEP_DIR}/.venv/bin/python"
ACESTEP_VENV = f"{ACESTEP_DIR}/.venv/bin"
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
GPU_UUID = os.environ.get("CUDA_VISIBLE_DEVICES", "GPU-c50e233e-1ad0-bae4-8704-f5af5a817bd4")

GRACE_PERIOD = int(os.environ.get("ACESTEP_GRACE_PERIOD", "15"))
BACKEND_STARTUP_WAIT = 120
POLL_INTERVAL = 2

# ── State ───────────────────────────────────────────────────────────────

_proc: Optional[subprocess.Popen] = None
_state = "idle"     # idle | starting | ready | error
_pending_jobs: Set[str] = set()  # task IDs that are queued or running
_active_requests = 0
_last_request_end = 0.0
_lock = threading.Lock()


# ── Backend lifecycle ───────────────────────────────────────────────────

def _build_env():
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    env["HF_HOME"] = HF_HOME
    env["PATH"] = f"{ACESTEP_VENV}:{env.get('PATH', '')}"
    return env


def _start_backend():
    """Spawn the ACE-Step backend process and wait for it to respond."""
    global _proc, _state
    _state = "starting"
    print(f"[acestep-proxy] Starting backend on :{BACKEND_PORT}...", flush=True)

    _proc = subprocess.Popen(
        [
            ACESTEP_PYTHON, "-m", "acestep.api_server",
            "--host", "127.0.0.1",
            "--port", str(BACKEND_PORT),
        ],
        cwd=ACESTEP_DIR,
        env=_build_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    deadline = time.time() + BACKEND_STARTUP_WAIT
    while time.time() < deadline:
        if _proc.poll() is not None:
            _state = "error"
            print(f"[acestep-proxy] Backend exited early (code={_proc.returncode})", flush=True)
            return False
        try:
            with urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=5) as r:
                json.loads(r.read())
            _state = "ready"
            print(f"[acestep-proxy] Backend ready (:{BACKEND_PORT}).", flush=True)
            return True
        except Exception:
            time.sleep(POLL_INTERVAL)

    _state = "error"
    print(f"[acestep-proxy] Backend failed to start within {BACKEND_STARTUP_WAIT}s", flush=True)
    return False


def _stop_backend():
    """Kill the backend process and free VRAM."""
    global _proc, _state, _pending_jobs
    if _proc is None:
        return
    print("[acestep-proxy] Stopping backend (idle, no pending jobs)...", flush=True)
    _proc.terminate()
    try:
        _proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _proc.kill()
        _proc.wait()
    _proc = None
    _state = "idle"
    _pending_jobs.clear()
    print("[acestep-proxy] Backend stopped. VRAM freed.", flush=True)


def _ensure_backend():
    """Ensure backend is running, starting it if needed. Call under _lock."""
    global _proc, _state

    # Detect dead process
    if _proc is not None and _proc.poll() is not None:
        print(f"[acestep-proxy] Backend died (code={_proc.returncode}), cleaning up", flush=True)
        _proc = None
        _state = "idle"
        _pending_jobs.clear()

    if _proc is None:
        if not _start_backend():
            return False

    return True


# ── Idle watcher thread ─────────────────────────────────────────────────

def _idle_watcher():
    """Unload backend when no active requests, no pending jobs, and grace period elapsed."""
    global _proc
    while True:
        time.sleep(5)
        with _lock:
            if _state != "ready" or _proc is None:
                continue
            # Don't unload if there are pending jobs
            if _pending_jobs:
                continue
            # Don't unload if requests are in flight
            if _active_requests > 0:
                continue
            # Unload after grace period
            if time.time() - _last_request_end > GRACE_PERIOD:
                _stop_backend()


# ── HTTP handler ────────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):

    def _proxy(self, method):
        """Forward request to backend, starting it if needed."""
        global _active_requests, _last_request_end, _pending_jobs

        # Health endpoint: report proxy-level status (don't start backend)
        if self.path == "/health" and method == "GET":
            self._json({
                "status": _state,
                "grace_period": GRACE_PERIOD,
                "active_requests": _active_requests,
                "pending_jobs": len(_pending_jobs),
                "backend_port": BACKEND_PORT,
                "loaded": _proc is not None and _proc.poll() is None,
            })
            return

        # All other requests: ensure backend is up, then proxy
        with _lock:
            _active_requests += 1
            ok = _ensure_backend()

        if not ok:
            with _lock:
                _active_requests -= 1
            self._json({"error": "ACE-Step backend failed to start"}, 503)
            return

        # Read request body
        body = b""
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            body = self.rfile.read(length)

        # Forward to backend
        url = f"{BACKEND_URL}{self.path}"
        headers = {}
        for key in ("Content-Type", "Accept", "Authorization"):
            val = self.headers.get(key)
            if val:
                headers[key] = val

        try:
            req = urllib.request.Request(url, data=body if body else None,
                                         method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp_body = resp.read()
                # Track jobs: /release_task returns task_id, /query_result returns status
                if self.path == "/release_task" and method == "POST":
                    try:
                        resp_json = json.loads(resp_body)
                        task_id = resp_json.get("data", {}).get("task_id")
                        if task_id:
                            with _lock:
                                _pending_jobs.add(task_id)
                            print(f"[acestep-proxy] Job tracked: {task_id} (pending: {len(_pending_jobs)})", flush=True)
                    except Exception:
                        pass

                elif self.path == "/query_result" and method == "POST":
                    try:
                        resp_json = json.loads(resp_body)
                        results = resp_json.get("data", [])
                        for r in results:
                            task_id = r.get("task_id", "")
                            status = r.get("status", 0)
                            # status 1 = done, 2 = failed
                            if status in (1, 2) and task_id:
                                with _lock:
                                    _pending_jobs.discard(task_id)
                                print(f"[acestep-proxy] Job done: {task_id} (pending: {len(_pending_jobs)})", flush=True)
                    except Exception:
                        pass
                self._raw_response(resp.status, resp.getheader("Content-Type",
                                "application/json"), resp_body)
                resp_body = resp.read()

                # Track jobs: /release_task returns task_id, /query_result returns status
                if self.path == "/release_task" and method == "POST":
                    try:
                        resp_json = json.loads(resp_body)
                        task_id = resp_json.get("data", {}).get("task_id")
                        if task_id:
                            with _lock:
                                _pending_jobs.add(task_id)
                            print(f"[acestep-proxy] Job tracked: {task_id} (pending: {len(_pending_jobs)})", flush=True)
                    except Exception:
                        pass

                elif self.path == "/query_result" and method == "POST":
                    try:
                        resp_json = json.loads(resp_body)
                        results = resp_json.get("data", [])
                        for r in results:
                            task_id = r.get("task_id", "")
                            status = r.get("status", 0)
                            # status 1 = done, 2 = failed
                            if status in (1, 2) and task_id:
                                with _lock:
                                    _pending_jobs.discard(task_id)
                                print(f"[acestep-proxy] Job done: {task_id} (pending: {len(_pending_jobs)})", flush=True)
                    except Exception:
                        pass

                self._raw_response(resp.status, resp.getheader("Content-Type",
                                "application/json"), resp_body)
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            self._raw_response(e.code, e.headers.get("Content-Type",
                            "application/json"), resp_body)
        except Exception as e:
            print(f"[acestep-proxy] Proxy error: {e}", flush=True)
            self._json({"error": str(e)}, 502)
        finally:
            with _lock:
                _active_requests -= 1
                _last_request_end = time.time()

    def _json(self, data, code=200):
        try:
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _raw_response(self, code, content_type, body):
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def log_message(self, format, *args):
        pass


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print(f"[acestep-proxy] Proxy starting on {HOST}:{PORT}", flush=True)
    print(f"[acestep-proxy] Backend: {ACESTEP_DIR} -> :{BACKEND_PORT}", flush=True)
    print(f"[acestep-proxy] GPU: {GPU_UUID}", flush=True)
    print(f"[acestep-proxy] Grace period: {GRACE_PERIOD}s (unloads after last job completes)", flush=True)

    t = threading.Thread(target=_idle_watcher, daemon=True)
    t.start()

    server = ThreadingHTTPServer((HOST, PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[acestep-proxy] Shutting down...", flush=True)
        with _lock:
            _stop_backend()


if __name__ == "__main__":
    main()