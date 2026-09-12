#!/usr/bin/env python3
"""aion_models.py — model routing helpers (code vs conscious).

Roster (2026-09-07):
  conscious/narrative/vision: gemma4:31b-65k on :11436 (V100 #1)
  code:                       qwen3.8:27b  on :11439 (V100 #3, loaned)
  intuition:                  muse-glimmer on :11438 (V100 #2)

code_endpoint() returns the (url, model) tuple for code-generation calls.
When the code instance is unreachable (e.g. the loaned V100 #3 is removed),
it falls back to the conscious main model — gemma4 does the coding.
Result is cached briefly so per-call health probes stay cheap.
"""
import os
import time
import urllib.request

try:
    import aion_env  # noqa: F401  (loads config/aion.env into os.environ)
except Exception:
    pass

_cache = (0.0, None)


def _main_endpoint():
    return (os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436"),
            os.environ.get("MAIN_MODEL", "gemma4:31b-65k"))


def code_endpoint(timeout=2.0, cache_s=60.0, probe=True):
    """Return (url, model) for code generation, with main-model fallback."""
    global _cache
    now = time.time()
    ts, cached = _cache
    if cached is not None and now - ts < cache_s:
        return cached

    url = os.environ.get("OLLAMA_CODE_URL", "")
    model = os.environ.get("CODE_MODEL", "")
    result = None
    if url and model:
        ok = True
        if probe:
            try:
                urllib.request.urlopen(
                    url.rstrip("/") + "/api/tags", timeout=timeout).read()
            except Exception:
                ok = False
        if ok:
            result = (url, model)

    if result is None:
        result = _main_endpoint()

    _cache = (now, result)
    return result


def code_num_ctx():
    return int(os.environ.get("CODE_NUM_CTX",
                              os.environ.get("MAIN_NUM_CTX", "65536")))


if __name__ == "__main__":
    print("code endpoint:", code_endpoint(probe=False))
    print("main endpoint:", _main_endpoint())
    print("code num_ctx:", code_num_ctx())
