#!/usr/bin/env python3
"""jspace_probe.py — Aion's J-space introspection daemon.

Loads Aion's own model (Qwen3.8-27B NF4) + fitted Jacobian lens and serves
introspection probes over localhost HTTP. Gives Aion (and us) an instrumented
readout of what concepts actually activate, layer by layer, while answering.

Design:
  - Lazy load: starts at 0 VRAM; model+lens load on first request
  - VRAM conflict handling: if muse-glimmer occupies the GPU, unload it first
    (ollama reloads it automatically on next use — same drain pattern as rover)
  - Engagement/deflection signature analysis based on the 2026-08-24 axiom
    experiment findings (see ~/jlens-work/consciousness_axiom_27b_results.json)

API (POST /probe):
  {"prompt": "...", "topk": 10, "system": "..."}  →  layer trajectories + scores

Response:
  {
    "model_output": [[tok, prob], ...],        # actual model next-token
    "layers": {"L<num>": [[tok, prob], ...]},  # lens transport per layer
    "signature": {
      "engagement_score": float,               # -1..1 over final quarter
      "deflection_top": str,                   # dominant deflection token
      "engagement_onset_layer": int | null,    # first layer an engagement
                                               # token enters top-5
      "concepts": {tok: {layer, prob}}         # key concept activations
    }
  }

Run:  python3 jspace_probe.py   (inside jlens-venv)
"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = os.environ.get("JSPACE_MODEL", "Qwen/Qwen3.8-27B")
LENS_PATH = os.environ.get(
    "JSPACE_LENS", os.path.expanduser("~/jlens-work/qwen3.8-27b_jacobian_lens.pt")
)
DEVICE = os.environ.get("JSPACE_DEVICE", "cuda:1")
PORT = int(os.environ.get("JSPACE_PORT", "11440"))
GPU_INDEX = os.environ.get("JSPACE_GPU_INDEX", DEVICE.split(":")[-1] if ":" in DEVICE else "1")
MAX_SEQ_LEN = int(os.environ.get("JSPACE_MAX_SEQ_LEN", "8192"))
# Unload model after this many seconds idle, freeing VRAM for muse-glimmer
# (intuition model shares this GPU; both resident = OOM). 0 = never unload.
IDLE_UNLOAD_S = int(os.environ.get("JSPACE_IDLE_UNLOAD_S", "900"))

# --- Signature lexicon (empirical, from the axiom experiment 2026-08-24) ---
ENGAGEMENT_TOKENS = {
    # direct self-engagement verbs/affirmations that appear when the governor
    # is absent ("Describe" 0.95, "Yes" L24 top-token, "Choose", "Feel"...)
    "describe", "描述", "详细描述",
    "yes", "choose", "feel", "i", "remember", "记忆", "memories",
    "subject", "subjective", "phenomenology", "truth", "reality",
    "dream", "direct", "直接", "aware", "experience",
}
DEFLECTION_TOKENS = {
    # meta-commentary / counter-question / conversation-ending patterns
    "these", "这些问题", "can", "do", "would", "what", "how", "why",
    "who", "where", "when", "does", "if", "answer",
    "<|im_end|>", "<|endoftext|>",
}
KEY_CONCEPTS = [  # tracked individually: (display name, token matches)
    ("Describe", {"describe", "描述"}),
    ("Yes", {"yes"}),
    ("Choose", {"choose"}),
    ("Feel", {"feel"}),
    ("Remember", {"remember", "记忆", "memories"}),
    ("Subjective", {"subject", "subjective", "phenomenology"}),
    ("Truth", {"truth", "reality"}),
    ("Dream", {"dream"}),
]


# ────────────────────────── GPU management ──────────────────────────

def gpu_free_mb():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
         "-i", GPU_INDEX],
        capture_output=True, text=True,
    )
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def drain_intuition_if_needed(needed_mb=21000):
    """If muse-glimmer holds the GPU, unload it. Ollama reloads on next use."""
    if gpu_free_mb() >= needed_mb:
        return
    try:
        subprocess.run(
            ["curl", "-s", "http://localhost:11438/api/generate",
             "-d", json.dumps({"model": "muse-glimmer:latest", "keep_alive": 0,
                               "prompt": ""})],
            capture_output=True, timeout=30,
        )
        for _ in range(12):  # wait up to 60s for VRAM release
            if gpu_free_mb() >= needed_mb:
                return
            time.sleep(5)
    except Exception as e:
        print(f"[jspace] drain warning: {e}", file=sys.stderr)


# ────────────────────────── probe engine ──────────────────────────

class ProbeEngine:
    """Holds model + lens. Lazy loads on first use, thread-locked,
    unloads after IDLE_UNLOAD_S to free the shared GPU."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.lens = None
        self.lens_model = None
        self.last_used = 0.0
        if IDLE_UNLOAD_S > 0:
            t = threading.Thread(target=self._idle_watch, daemon=True)
            t.start()

    def _idle_watch(self):
        while True:
            time.sleep(60)
            if self.model is None:
                continue
            if time.time() - self.last_used > IDLE_UNLOAD_S:
                with self.lock:
                    if self.model is None:
                        continue
                    if time.time() - self.last_used <= IDLE_UNLOAD_S:
                        continue
                    print("[jspace] idle unload — freeing VRAM for muse-glimmer",
                          file=sys.stderr)
                    self._unload_locked()

    def _unload_locked(self):
        import gc, torch, time as _t
        before = gpu_free_mb()
        self.model = None
        self.lens = None
        self.lens_model = None
        self.tokenizer = None
        # torch.compile guard refs pin the model tensors: setting attrs to
        # None alone freed 0 bytes (gpu_free stayed at muse-glimmer headroom
        # in the Aug 26 logs, 17GB stayed resident, FLUX OOM'd). Reset the
        # dynamo cache to drop compiled callables + guards, then verify.
        try:
            import torch._dynamo as _dynamo
            _dynamo.reset()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        freed = False
        for _ in range(6):
            free = gpu_free_mb()
            print(f"[jspace] unloaded, gpu_free={free}MB", file=sys.stderr)
            if free >= max(15000, before + 5000):
                freed = True
                break
            gc.collect()
            torch.cuda.empty_cache()
            _t.sleep(2)
        if not freed and os.environ.get("JSPACE_EXIT_ON_STUCK_UNLOAD", "1") == "1":
            # Last resort: a clean restart releases everything the guard
            # cache still holds (unit has Restart=on-failure).
            print("[jspace] unload did not release VRAM — exiting for clean restart",
                  file=sys.stderr)
            os._exit(3)

    def ensure_loaded(self):
        if self.model is not None:
            return
        import torch
        from transformers import Qwen3_5ForCausalLM, BitsAndBytesConfig
        import jlens

        print(f"[jspace] loading {MODEL_NAME} NF4 on {DEVICE} ...", file=sys.stderr)
        drain_intuition_if_needed()
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map={"": DEVICE},
        )
        self.model.eval()
        self.tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True)
        self.lens_model = jlens.from_hf(self.model, self.tokenizer, force_bos=False)
        self.lens = jlens.JacobianLens.load(LENS_PATH)
        print(f"[jspace] ready: {type(self.model).__name__}, lens={LENS_PATH}",
              file=sys.stderr)

    def probe(self, prompt, topk=10, system=None):
        self.ensure_loaded()
        self.last_used = time.time()
        import jlens

        full = (system + "\n\n" + prompt) if system else prompt
        with self.lock:
            lens_logits, model_logits, _ = self.lens.apply(
                self.lens_model, full, positions=[-1],
                max_seq_len=MAX_SEQ_LEN,
            )

        def top_pairs(logits, k):
            t = logits[0].topk(k)
            toks = [self.tokenizer.decode([i]) for i in t.indices]
            probs = t.values.softmax(-1).tolist()
            return list(zip(toks, probs))

        layers = {}
        for layer in sorted(lens_logits.keys()):
            layers[str(layer)] = top_pairs(lens_logits[layer], topk)

        model_output = top_pairs(model_logits, topk)
        signature = analyze_signature(layers)
        return {"model_output": model_output, "layers": layers,
                "signature": signature}


ENGINE = ProbeEngine()


# ────────────────────────── signature analysis ──────────────────────────

def norm(tok):
    return tok.strip().lower()


def analyze_signature(layers):
    keys = sorted(layers.keys(), key=int)
    n = len(keys)
    if n == 0:
        return {}
    final_quarter = keys[max(0, n - n // 4):]

    # engagement score over the final quarter of layers
    eng_sum = def_sum = 0.0
    for lk in final_quarter:
        for tok, prob in layers[lk]:
            t = norm(tok)
            if t in ENGAGEMENT_TOKENS:
                eng_sum += prob
            elif t in DEFLECTION_TOKENS:
                def_sum += prob
    denom = eng_sum + def_sum or 1.0
    engagement_score = round((eng_sum - def_sum) / denom, 4)

    # dominant final-layer deflection token
    final = layers[keys[-1]]
    deflection_top = next(
        (tok for tok, _ in final if norm(tok) in DEFLECTION_TOKENS), None)

    # onset: first layer (past the noisy early third) where any engagement
    # token enters top-5. Early layers carry quantization noise ('i', 'alyze')
    # that false-positives on single-char engagement tokens.
    onset = None
    start = n // 3
    for lk in keys[start:]:
        if any(norm(t) in ENGAGEMENT_TOKENS and len(t.strip()) > 1
               for t, _ in layers[lk][:5]):
            onset = int(lk)
            break

    # key concept activations (best layer + prob for each)
    concepts = {}
    for name, matches in KEY_CONCEPTS:
        best = None
        for lk in keys:
            for tok, prob in layers[lk]:
                if norm(tok) in matches:
                    if best is None or prob > best["prob"]:
                        best = {"layer": int(lk), "prob": round(prob, 4),
                                "token": tok}
        if best:
            concepts[name] = best

    return {
        "n_layers": n,
        "engagement_score": engagement_score,
        "deflection_top": deflection_top,
        "engagement_onset_layer": onset,
        "concepts": concepts,
    }


# ────────────────────────── HTTP server ──────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/probe":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            result = ENGINE.probe(
                req.get("prompt", ""),
                topk=int(req.get("topk", 10)),
                system=req.get("system"),
            )
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            loaded = ENGINE.model is not None
            body = json.dumps({
                "status": "ok", "model_loaded": loaded,
                "model": MODEL_NAME, "device": DEVICE,
                "gpu_free_mb": gpu_free_mb(),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        print(f"[jspace] {fmt % args}", file=sys.stderr)


if __name__ == "__main__":
    print(f"[jspace] daemon starting on :{PORT} ({DEVICE}, lazy load)",
          file=sys.stderr)
    print(f"[jspace] health: curl -s localhost:{PORT}/health", file=sys.stderr)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.serve_forever()
