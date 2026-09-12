#!/usr/bin/env python3
"""flux_server.py — FLUX.2-klein-9B image generation API for Aion.

Full idle-unload: after IDLE_TIMEOUT seconds of inactivity, the entire
process exits. Zero VRAM, zero system RAM, zero extra power when idle.
The process is started on-demand by the art creation pipeline.

  POST /generate  {"prompt": "...", "seed": 42}  → {"image_path": "..."}
  GET  /health    → {"status": "idle"|"loading"|"ready"|"error"}
"""
import json
import os
import sys
import time
import hashlib
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

AION = os.environ.get("AION_HOME", os.path.expanduser("~/aikio"))
PORT = int(os.environ.get("FLUX_PORT", "8116"))
HOST = os.environ.get("FLUX_HOST", "0.0.0.0")
OUTPUT_DIR = f"{AION}/gallery/visual"
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEVICE = "cuda:0"
DTYPE = "bfloat16"
IDLE_TIMEOUT = 180  # exit process after 3 minutes idle

os.makedirs(OUTPUT_DIR, exist_ok=True)

_state = "idle"
_pipe = None
_last_use = 0
_lock = threading.Lock()
_busy = False


def _load_model():
    global _pipe, _state
    _state = "loading"
    print(f"[flux] Loading {MODEL_ID}...", flush=True)
    import torch
    from diffusers import Flux2KleinPipeline
    dt = torch.bfloat16 if DTYPE == "bfloat16" else torch.float32
    _pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dt)
    _pipe.enable_sequential_cpu_offload()
    try:
        _pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        _pipe.vae.to(DEVICE)
    except Exception:
        pass
    _state = "ready"
    print(f"[flux] Ready.", flush=True)


def _full_shutdown():
    """Unload model and exit the process entirely.

    This frees all VRAM, all system RAM (model weights in CPU offload),
    and stops the Python process from drawing power on the GPU.
    The process will be restarted on-demand by the art creation pipeline.
    """
    global _pipe, _state
    print("[flux] Full shutdown (idle timeout) — exiting process...", flush=True)
    try:
        del _pipe
        _pipe = None
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    _state = "exiting"
    # Flush and exit
    sys.stdout.flush()
    os._exit(0)


def _idle_watcher():
    """Background thread that exits the process after IDLE_TIMEOUT."""
    global _last_use
    while True:
        time.sleep(15)
        if _state == "ready" and _pipe is not None and not _busy:
            if time.time() - _last_use > IDLE_TIMEOUT:
                with _lock:
                    if _state == "ready" and _pipe is not None:
                        _full_shutdown()


def generate_image(prompt, seed=None, width=1024, height=1024, steps=12):
    """Generate an image. Loads model if not loaded."""
    global _last_use
    import torch

    global _busy, _last_use
    with _lock:
        _busy = True
        if _pipe is None:
            _load_model()

        _last_use = time.time()
        if seed is None:
            seed = int(time.time()) % 2**32

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        print(f"[flux] Generating: {prompt[:80]}... (seed={seed})", flush=True)

        image = _pipe(
            prompt=prompt,
            width=min(width, 1024),
            height=min(height, 1024),
            num_inference_steps=min(steps, 20),
            generator=generator,
            guidance_scale=4.0,
        ).images[0]

        _last_use = time.time()
        _busy = False

    ts = int(time.time())
    h = hashlib.sha256(f"{ts}{seed}".encode()).hexdigest()[:8]
    filename = f"flux_{ts}_{h}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    print(f"[flux] Saved: {filepath}", flush=True)
    return {"image_path": filepath, "filename": filename, "seed": seed}


class FluxHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": _state,
                "model": MODEL_ID,
                "idle_timeout": IDLE_TIMEOUT,
            }).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/generate":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                req = json.loads(raw)
            except Exception:
                self._json({"error": "invalid JSON"}, 400)
                return

            prompt = req.get("prompt", "").strip()
            if not prompt:
                self._json({"error": "missing prompt"}, 400)
                return

            try:
                result = generate_image(
                    prompt=prompt,
                    seed=req.get("seed"),
                    width=req.get("width", 1024),
                    height=req.get("height", 1024),
                    steps=req.get("num_inference_steps", 12),
                )
                self._json(result)
            except Exception as e:
                print(f"[flux] Error: {e}", flush=True)
                self._json({"error": str(e)}, 500)
        else:
            self.send_error(404)

    def _json(self, data, code=200):
        try:
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        pass


def main():
    print(f"[flux] Server starting on {HOST}:{PORT} (full idle-exit mode)", flush=True)
    print(f"[flux] Model: {MODEL_ID}", flush=True)
    print(f"[flux] Process will exit after {IDLE_TIMEOUT}s idle — restarted on demand", flush=True)

    t = threading.Thread(target=_idle_watcher, daemon=True)
    t.start()

    server = ThreadingHTTPServer((HOST, PORT), FluxHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()