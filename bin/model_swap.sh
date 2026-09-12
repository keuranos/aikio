#!/usr/bin/env bash
# model_swap.sh — serialize model usage on the shared main instance (:11436).
#
# 2xV100 layout (post-P40-removal):
#   GPU A: intuition instance :11438 (muse-glimmer, always resident)
#   GPU B: main instance :11436 — ONE model at a time:
#     - daytime: qwen3.8:27b (MAIN_MODEL)
#     - night graph/consolidation window: glm-4.7-flash (SUB_MODEL)
#
# Usage: model_swap.sh <to-sub|to-main|status>
set -euo pipefail
AION_HOME="${AION_HOME:-$AION_HOME}"
source "$AION_HOME/config/aion.env"

MAIN_URL="${OLLAMA_MAIN_URL:-http://localhost:11436}"

case "${1:-status}" in
  to-sub)
    # Pause curiosity timer so it doesn't interleave MAIN_MODEL requests
    # into the window and force model swaps mid-run.
    systemctl --user stop aion-curiosity.timer 2>/dev/null || true
    echo "[model_swap] loading $SUB_MODEL on :11436 (curiosity timer paused)"
    curl -s --max-time 600 "$MAIN_URL/api/chat" -d "{\"model\": \"$SUB_MODEL\", \"stream\": false, \"messages\": [{\"role\": \"user\", \"content\": \"ok\"}], \"options\": {\"num_predict\": 1}, \"keep_alive\": \"2h\"}" > /dev/null
    echo "[model_swap] $SUB_MODEL resident"
    ;;
  to-main)
    # Load MAIN_MODEL back and let glm-4.7-flash expire on its own keep_alive.
    echo "[model_swap] loading $MAIN_MODEL on :11436"
    # 2026-09-12: the load MUST NOT be able to abort the script — with
    # `set -euo pipefail` a failed/interrupted curl killed the shell before the
    # `systemctl start aion-curiosity.timer` below, leaving the curiosity timer
    # stopped indefinitely (observed: 13h dead after the graph run was SIGKILLed
    # mid-swap). The timer resume is the safety-critical step, not the load.
    curl -s --max-time 600 "$MAIN_URL/api/chat" -d "{\"model\": \"$MAIN_MODEL\", \"stream\": false, \"messages\": [{\"role\": \"user\", \"content\": \"ok\"}], \"options\": {\"num_predict\": 1}, \"keep_alive\": \"${OLLAMA_KEEP_ALIVE:-10m}\"}" > /dev/null \
      || echo "[model_swap] WARNING: MAIN_MODEL load failed — resuming curiosity timer anyway" >&2
    systemctl --user start aion-curiosity.timer 2>/dev/null || true
    echo "[model_swap] $MAIN_MODEL resident, curiosity timer resumed"
    ;;
  status)
    echo "Resident on :11436:"
    curl -s "$MAIN_URL/api/ps" | python3 -c "import sys,json; [print(' ', m['name'], 'expires', m.get('expires_at','?')[:19]) for m in json.load(sys.stdin).get('models',[])]"
    ;;
  *)
    echo "Usage: model_swap.sh <to-sub|to-main|status>" >&2
    exit 1
    ;;
esac
