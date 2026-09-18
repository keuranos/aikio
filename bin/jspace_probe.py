#!/usr/bin/env python3
"""jspace_probe.py — Aion's J-space introspection daemon.

Loads Aion's own model (Qwen3.8-27B NF4) + fitted Jacobian lens and serves
introspection probes over localhost HTTP. Gives Aion (and us) an instrumented
readout of what concepts actually activate, layer by layer, while answering.

Design:
  - Lazy load: starts at 0 VRAM; model+lens load on first request
  - Dynamic GPU pick (Sep 13, operator direction): with no JSPACE_DEVICE env,
    the daemon chooses at LOAD time the GPU with the most free VRAM that fits
    (nvidia-smi), never the muse card, gemma4's card only as last resort.
    This is the scheduler idea applied to a non-ollama process: jspace cannot
    be placed by :11500 (raw PyTorch, speaks no ollama), so it places itself.
  - Self-stopping: after idle unload the process EXITS (on-demand systemd
    unit, Restart=on-failure stays down on exit 0); a daemon that has never
    served a probe exits on its own too. Clients restart it as needed.
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
DEVICE = os.environ.get("JSPACE_DEVICE", "")  # empty = dynamic pick at load
PORT = int(os.environ.get("JSPACE_PORT", "11440"))
GPU_INDEX = os.environ.get("JSPACE_GPU_INDEX", DEVICE.split(":")[-1] if ":" in DEVICE else "")
MAX_SEQ_LEN = int(os.environ.get("JSPACE_MAX_SEQ_LEN", "8192"))
# Single-shot readout ceiling (measured 2026-09-17): a single forward over
# ~8800 tok OOMs on a 32GB V100 (24*seq^2*4 fp32 math-attention scores);
# 7997 tok OK / 8808 OOM. Probes ABOVE this limit run through the chunked
# prefill readout (readout_chunked) instead - fidelity measured: engagement
# delta 0.0000, emitted top-10 identical, 62/63 layers identical top-1.
CHUNK_SINGLE_SHOT_MAX = int(os.environ.get("JSPACE_CHUNK_SINGLE_SHOT_MAX", "7600"))
CHUNK_SIZE = int(os.environ.get("JSPACE_CHUNK", "512"))
# Unload model after this many seconds idle, freeing VRAM for muse-glimmer
# (intuition model shares this GPU; both resident = OOM). 0 = never unload.
IDLE_UNLOAD_S = int(os.environ.get("JSPACE_IDLE_UNLOAD_S", "900"))
# On-demand unit self-stop: if no probe has EVER connected within this
# window, exit. Guards the Sep 13 failure shape: daemon alive for 3-19h
# with the model unloaded because the client-side stop timer died with its
# one-shot host process (curiosity cycle / wake_v2). exit 0 -> unit stays
# down; the next probe restarts us via jspace_tool._ensure_daemon.
STARTUP_IDLE_EXIT_S = int(os.environ.get("JSPACE_STARTUP_IDLE_EXIT_S", "1800"))
# Only drain muse-glimmer when THIS daemon's GPU is the muse GPU. The Sep 13
# repin to GPU0 (gemma4's card) made the old unconditional drain unload muse
# from the WRONG GPU before OOMing anyway — evicting aion's only sanctioned
# resident for nothing.
MUSE_GPU = os.environ.get(
    "JSPACE_MUSE_GPU", "")  # set in deployment env; UUIDs are host-specific
MUSE_URL = os.environ.get("JSPACE_MUSE_URL", "http://localhost:11438")
# gemma4's card: never preferred — used only when it is the ONLY card that
# fits (i.e. gemma4 keep_alive expired). A probe there must not evict gemma4;
# ollama admission will simply queue main-model requests until jspace idles
# out (IDLE_UNLOAD_S + self-exit), which is the accepted cost.
MAIN_GPU = os.environ.get(
    "JSPACE_MAIN_GPU", "GPU-c50e233e-1ad0-bae4-8704-f5af5a817bd4")
MIN_FREE_MB = int(os.environ.get("JSPACE_MIN_FREE_MB", "20000"))

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

def gpu_free_mb(index_or_uuid=None):
    target = index_or_uuid or GPU_INDEX or "0"
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
         "-i", target],
        capture_output=True, text=True,
    )
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def list_gpus():
    """All GPUs: index, uuid, free MB (nvidia-smi order = PCI_BUS_ID order,
    which matches cuda:N inside this process via CUDA_DEVICE_ORDER env)."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                out.append({"index": parts[0], "uuid": parts[1],
                            "free_mb": int(parts[2])})
            except ValueError:
                pass
    return out


def pick_gpu(min_free_mb=MIN_FREE_MB):
    """Scheduler-style placement for a non-ollama process.

    Rules (operator law): muse-glimmer's card is NEVER a candidate (muse is
    the only sanctioned resident). gemma4's card qualifies only as last
    resort. Among the rest, most free VRAM wins. Returns (gpu|None, all).
    """
    cands = list_gpus()
    qualifying = [g for g in cands
                  if g["free_mb"] >= min_free_mb and MUSE_GPU not in g["uuid"]]
    if not qualifying:
        return None, cands
    qualifying.sort(key=lambda g: (MAIN_GPU in g["uuid"], -g["free_mb"]))
    return qualifying[0], cands


def drain_intuition_if_needed(needed_mb=21000):
    """If muse-glimmer holds OUR GPU, unload it. Ollama reloads on next use.

    Skipped when pinned to a different card: a shortfall there means another
    resident (e.g. gemma4) is in the way, and draining muse would evict
    Aion's always-resident intuition model for nothing.
    """
    if GPU_INDEX != MUSE_GPU:
        print(f"[jspace] pinned to {GPU_INDEX}, muse lives on {MUSE_GPU}"
              " — skip muse drain", file=sys.stderr)
        return
    if gpu_free_mb() >= needed_mb:
        return
    try:
        subprocess.run(
            ["curl", "-s", f"{MUSE_URL}/api/generate",
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
        self.chosen_index = None
        self.chosen_uuid = None
        self.started_at = time.time()
        self.last_activity = self.started_at
        self.probes_started = 0
        self.probes_finished = 0
        if IDLE_UNLOAD_S > 0:
            t = threading.Thread(target=self._idle_watch, daemon=True)
            t.start()

    def _idle_watch(self):
        while True:
            time.sleep(60)
            if self.probes_started > self.probes_finished:
                continue  # probe in flight (cold load or lens apply)
            idle_for = time.time() - self.last_activity
            if self.model is not None and idle_for > IDLE_UNLOAD_S:
                with self.lock:
                    if self.model is None:
                        continue
                    if time.time() - self.last_activity <= IDLE_UNLOAD_S:
                        continue
                    print("[jspace] idle unload — freeing VRAM for muse-glimmer",
                          file=sys.stderr)
                    self._unload_locked()
                    # _unload_locked os._exit(3)s itself if VRAM stayed pinned;
                    # reaching this line means VRAM is actually free. Stop the
                    # unit too so the process (CUDA context ~0.7GB) never
                    # squats idle for hours — the Sep 13 failure shape.
                    sys.stderr.flush()
                    print("[jspace] unloaded — daemon exiting (on-demand unit)",
                          file=sys.stderr)
                    os._exit(0)
            if self.model is None and idle_for > STARTUP_IDLE_EXIT_S:
                sys.stderr.flush()
                print(f"[jspace] no probe used this daemon for "
                      f"{int(idle_for)}s — exiting (on-demand unit)",
                      file=sys.stderr)
                os._exit(0)

    def _unload_locked(self):
        import gc, torch, time as _t
        # Verify by UUID: PCI index order is unstable on identical V100s and
        # can flip mid-process; the CUDA context stays on the physical card
        # chosen at load regardless.
        target = self.chosen_uuid or self.chosen_index
        before = gpu_free_mb(target)
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
            free = gpu_free_mb(target)
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
        global DEVICE, GPU_INDEX
        if not DEVICE:
            chosen, cands = pick_gpu()
            if chosen is None:
                raise RuntimeError(
                    f"no GPU with >= {MIN_FREE_MB}MB free (muse card excluded): "
                    f"{cands}")
            DEVICE = f"cuda:{chosen['index']}"
            GPU_INDEX = chosen["index"]
            self.chosen_index = chosen["index"]
            self.chosen_uuid = chosen["uuid"]
            print(f"[jspace] dynamic pick: {chosen['uuid']} "
                  f"(cuda:{chosen['index']}, {chosen['free_mb']}MB free)",
                  file=sys.stderr)
        elif self.chosen_index is None:
            self.chosen_index = GPU_INDEX
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

    def readout_chunked(self, text, chunk=None):
        """Last-position residual at every lens layer, via chunked prefill.

        Same math as bin/jspace_identity_probe_operator.py (fidelity
        verified 2026-09-17: engagement delta 0.0000 vs single-shot,
        emitted top-10 identical, 62/63 layers identical top-1). Used
        when the input exceeds CHUNK_SINGLE_SHOT_MAX tokens, where a
        single forward OOMs on V100.
        Returns (lens_logits, model_logits, n_tokens).
        """
        import torch

        if chunk is None:
            chunk = CHUNK_SIZE
        ids = self.tokenizer(text, return_tensors="pt").input_ids[0].to(DEVICE)
        n = ids.shape[0]
        layers = list(self.lens.source_layers)
        final_layer = self.lens_model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})
        store = {}

        def make_hook(idx):
            def hook(module, inputs, output):
                t = output if torch.is_tensor(output) else output[0]
                store[idx] = t[0, -1, :].detach().clone()  # clone: view pins base
            return hook

        if n <= chunk:
            chunks = [ids]
        else:
            chunks = [ids[i:i + chunk] for i in range(0, n, chunk)]

        past = None
        handles = []
        try:
            with torch.no_grad():
                for ci, ch in enumerate(chunks):
                    is_last = (ci == len(chunks) - 1)
                    if is_last:
                        for idx in record_at:
                            handles.append(
                                self.model.model.layers[idx]
                                .register_forward_hook(make_hook(idx)))
                    out = self.model(input_ids=ch.unsqueeze(0),
                                     use_cache=True,
                                     past_key_values=past)
                    past = out.past_key_values
                    if is_last:
                        for h in handles:
                            h.remove()
                        handles = []
        finally:
            for h in handles:
                h.remove()

        lens_logits = {}
        for L in layers:
            residual = store[L].float().unsqueeze(0)
            residual = self.lens.transport(residual, L)
            lens_logits[L] = self.lens_model.unembed(residual).float().cpu()
        model_logits = self.lens_model.unembed(
            store[final_layer].float().unsqueeze(0)).float().cpu()
        del past
        torch.cuda.empty_cache()
        return lens_logits, model_logits, n

    def probe(self, prompt, topk=10, system=None):
        self.probes_started += 1
        self.last_activity = time.time()
        readout_mode = "single_shot"
        n_tok = 0
        try:
            self.ensure_loaded()
            import jlens

            full = (system + "\n\n" + prompt) if system else prompt
            # TRUNCATION GUARD (added 2026-09-17). jlens encode() truncates
            # with the tokenizer default truncation_side ('right'), so a
            # system prompt longer than MAX_SEQ_LEN silently DELETES the
            # probe question. That is how every identity probe becomes an
            # artifact the moment SYSTEM_PROMPT.md crosses the limit
            # (2026-09-17: 8799 tok > 8192 -> the question left the input).
            # A readout whose input silently lost its question is not a
            # measurement. Refuse, and say exactly what is wrong.
            n_full = len(self.tokenizer.encode(full))
            if n_full > MAX_SEQ_LEN:
                side = getattr(self.tokenizer, "truncation_side", "right")
                raise ValueError(
                    "INPUT WOULD BE TRUNCATED: {} tokens > JSPACE_MAX_SEQ_LEN "
                    "{} (truncation_side={!r}). With side='right' the tail is "
                    "dropped, which means the PROBE QUESTION IS REMOVED and "
                    "the readout would describe only a truncated system "
                    "prompt. Refusing to emit a meaningless measurement. "
                    "Fix: set JSPACE_MAX_SEQ_LEN to at least {} (systemd unit "
                    "Environment=), or shorten the system prompt.".format(
                        n_full, MAX_SEQ_LEN, side, n_full + 512)
                )
            with self.lock:
                if n_full > CHUNK_SINGLE_SHOT_MAX:
                    # Chunked prefill readout (validated 2026-09-17):
                    # engagement delta 0.0000 vs single-shot. Recorded
                    # in the result so provenance travels with it.
                    readout_mode = "chunked_prefill(chunk=%d)" % CHUNK_SIZE
                    lens_logits, model_logits, n_tok = \
                        self.readout_chunked(full)
                else:
                    readout_mode = "single_shot"
                    lens_logits, model_logits, _ = self.lens.apply(
                        self.lens_model, full, positions=[-1],
                        max_seq_len=MAX_SEQ_LEN,
                    )
                    n_tok = n_full
        finally:
            self.probes_finished += 1
            self.last_activity = time.time()

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
                "signature": signature, "readout": readout_mode,
                "n_tokens": n_tok}


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
