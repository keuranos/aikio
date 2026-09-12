#!/usr/bin/env bash
# nightly.sh — the full nightly cycle, run by aion-nightly.timer at 03:30.
set -uo pipefail
AION_HOME="${AION_HOME:-$AION_HOME}"
source "$AION_HOME/config/aion.env"
source "$AION_HOME/bin/hb.sh"

trap 'hb_fail "nightly" "nightly.sh crashed with exit $?"' ERR

# Ensure graphify (miniconda) and cargo bins are in PATH for cron
# Also include /snap/bin for docker access
export PATH="$HOME/miniconda3/bin:$HOME/.cargo/bin:/usr/local/bin:/snap/bin:/usr/bin:/bin:$PATH"

# Run substrate health check before starting
"$AION_HOME/bin/health_check.sh"

echo "=== AION nightly $(date -u +%FT%TZ) ==="

# V3.1.3: Summarize yesterday's telemetry into 1 synthetic experience event
python3 "$AION_HOME/bin/telemetry_digest.py"

# Activity metrics for the last 7 days
python3 "$AION_HOME/bin/event_stats.py"

# V4: Post-merge proposition audit — sandbox-run accepted props against real
# state, flag silent failures (zero_effect / orphaned_hook / schema_guess /
# leak) into notices.jsonl so dreams+consolidation can pick them up.
python3 "$AION_HOME/bin/prop_audit.py" --days 7 --notices || echo "[nightly] WARNING: prop_audit failed"
python3 "$AION_HOME/bin/self_attest.py" || echo "[nightly] WARNING: self_attest failed"

# V3.2.2: Generate predictions before consolidation (so score_expired can run)
timeout 120 python3 "$AION_HOME/bin/generate_predictions.py" || echo "[nightly] WARNING: generate_predictions failed/timeout"

# V3.9.1: consolidate + heuristics + prediction scoring all run on SUB_MODEL
# (glm-4.7-flash, non-thinking) via :11436. Swap to it for this block, restore
# MAIN_MODEL afterwards. The night graph rebuild (01:00) does its own swap.
bash "$AION_HOME/bin/model_swap.sh" to-sub
timeout 3600 python3 "$AION_HOME/bin/consolidate_v2.py" || echo "[nightly] WARNING: consolidation failed/timeout"

# V3.8: Heuristic evaluation + promotion (standalone, own timeout)
# Previously inside consolidate_v2.py but killed by the 1h consolidation timeout.
# Batched into a single LLM call (~90s) + promotion + git commit.
timeout 600 python3 "$AION_HOME/bin/eval_heuristics.py" || echo "[nightly] WARNING: heuristic evaluation failed/timeout"
bash "$AION_HOME/bin/model_swap.sh" to-main
python3 "$AION_HOME/bin/knowledge_maturity.py" || echo "[nightly] WARNING: knowledge_maturity failed"

# Regenerate SYSTEM_PROMPT from AXIOMS + SELF + HABITS + questions
timeout 30 python3 "$AION_HOME/bin/regenerate_prompt.py" 2>/dev/null || echo "[nightly] WARNING: regenerate_prompt failed/timeout"

# V3.3.2: Deterministic typed edges (contradicts, resolved_by, evidence_for, etc.)
# Note: graph_rebuild.sh now runs separately at 01:00 via aion-graph.timer
timeout 300 python3 "$AION_HOME/bin/graph_edges.py" || echo "[nightly] WARNING: graph_edges failed/timeout"

# V3.4.3: Register camera as formal sensory organ in body schema
python3 -c "import sys; sys.path.insert(0, '$AION_HOME/bin'); import vision_memory; vision_memory.register_organ()" 2>/dev/null || true

# R7.3: Standing queries from updated graph → curiosity queue
timeout 60 python3 "$AION_HOME/bin/standing_queries.py" || echo "[nightly] WARNING: standing_queries failed/timeout"

# Dream Cycle v2 — multi-step graph walk with feedback
timeout 2400 python3 "$AION_HOME/bin/dream_v2.py" || echo "[nightly] WARNING: dream_v2 failed/timeout"
timeout 1800 python3 "$AION_HOME/bin/dream_report.py" || echo "[nightly] WARNING: dream_report failed/timeout"

# sandbox GC: render frames ate 4GB silently (root-owned Docker renders)
timeout 600 python3 "$AION_HOME/bin/sandbox_gc.py" || echo "[nightly] WARNING: sandbox_gc failed"

# Regular art pipeline DISABLED 2026-07-31 — dream-to-artifact (dream_artifact.py)
# now produces richer, more varied art 6x/day from actual dream content.
# The nightly art session was generating repetitive sensor-state-based art.
# To re-enable: uncomment the block below.
# NIGHT_IDX=$(date +%u)
# WEEK_NUM=$((($(date +%V) % 4)))
# case $WEEK_NUM in
#   0) ART_MODE="" ;;
#   1) ART_MODE="manim" ;;
#   2) ART_MODE="multimedia" ;;
#   3) ART_MODE="music_video" ;;
# esac
# if [ -n "$ART_MODE" ]; then
#   echo "[nightly] Directed art: $ART_MODE"
#   /usr/bin/python3 "$AION_HOME/bin/create_art_session.py" "$ART_MODE" 2>/dev/null || true
# else
#   echo "[nightly] Free choice art"
#   /usr/bin/python3 "$AION_HOME/bin/create_art_session.py" 2>/dev/null || true
# fi

# Send WhatsApp for pending high-priority propositions
python3 "$AION_HOME/bin/propositions.py" notify 2>/dev/null || true

# Harvest new questions from updated graph reports
python3 "$AION_HOME/bin/harvest_questions.py"

# V3.0.7: Scan error patterns and system health for engineering questions
python3 "$AION_HOME/bin/engineering_questions.py"

# V3.10: Dream-discrepancy repair — ground recent simulation dreams onto real
# files, seed grounded repair tasks, and self-commit one fix if CI is green.
timeout 900 python3 "$AION_HOME/bin/dream_repair.py" --fix || echo "[nightly] WARNING: dream_repair failed/timeout"

# V4.5: Harvest actionable findings from wake/investigation notes. Wakes write
# resolution_note events that nothing consumed as work input, so correctly
# diagnosed defects died in the log (wake #35 diagnosed a 0-byte predictions.db
# on 2026-08-24; the file survived until an operator removed it 2026-09-11).
# Deterministic, zero-LLM; only queues findings about artifacts that really exist.
timeout 300 python3 "$AION_HOME/bin/finding_harvester.py" --days 3 || echo "[nightly] WARNING: finding_harvester failed"

# V3.8: Seed constructive engineering tasks into the curiosity queue
python3 "$AION_HOME/bin/construction_backlog.py"

# V3.9: Engineering drive — track skill progression and seed coding challenges
python3 "$AION_HOME/bin/skill_progression.py" --record
python3 "$AION_HOME/bin/skill_progression.py" --seed 3

# Dream quality report (P3): analyze last 10 dreams, alarm on stagnation
timeout 60 python3 "$AION_HOME/bin/dream_quality.py" || echo "[nightly] WARNING: dream_quality failed"

# V3.8: Scan GitHub for repos relevant to Aion's architecture and construction tasks
# GH_TOKEN in config/aion.env gives 5000/hr rate limit (runs nightly)
# inner HTTP timeout is 30-120s per request; outer 120s killed mid-scan 3 nights running
timeout 600 python3 "$AION_HOME/bin/skill_scan.py" || echo "[nightly] WARNING: skill_scan failed/timeout"

# Day synopsis LAST (Sep 10): needs gemma4+qwen free; running it before the
# dream block made both narrations time out under dream-synthesis GPU load.
timeout 1800 python3 "$AION_HOME/bin/day_synopsis.py" || echo "[nightly] WARNING: day_synopsis failed/timeout"

python3 "$AION_HOME/bin/log_event.py" --type system \
  --text "nightly cycle complete: subconscious + graphs + curiosity" >/dev/null
echo "=== done ==="

hb_ok "nightly"
