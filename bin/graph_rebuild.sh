#!/usr/bin/env bash
# graph_rebuild.sh — rebuild Aion's two knowledge graphs (Phase 3).
# 1) Body schema: AST extraction of Aion's own code (local, no LLM).
# 2) Mind graph: LLM extraction of SELF history + reflections, via the
#    GPU-1 ollama instance with conservative caps for the P40.
set -euo pipefail
AION_HOME="${AION_HOME:-$AION_HOME}"
source "$AION_HOME/config/aion.env"
# Ensure graphify (miniconda) is in PATH for cron
export PATH="$HOME/miniconda3/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
cd "$AION_HOME"

mkdir -p graphs/code graphs/mind/corpus

# ---- 1. body schema (cheap, every night) ----
# Graph the scripts, prompts, units — Aion's own implementation.
# Body schema: extract directly from bin/ (graphify needs flat code dirs)
# Clean old pycache first
find bin/ -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
graphify extract bin --no-viz --force 2>&1 || echo "WARN: code graph AST extraction failed"
mkdir -p graphs/code/graphify-out
if [ -d bin/graphify-out ]; then
  cp -r bin/graphify-out/* graphs/code/graphify-out/ 2>/dev/null || true
  graphify cluster-only bin --no-label --no-viz --graph graphs/code/graphify-out/graph.json 2>&1 \
    || echo "WARN: code graph cluster-only failed"
  cp -r bin/graphify-out/* graphs/code/graphify-out/ 2>/dev/null || true
  rm -rf bin/graphify-out
fi

# ---- 1.5. Prune corpus to bounded size ----
# Dreams accumulate ~8/day; cap to last 14 days to keep extraction time bounded.
# SELF revisions and reflections are capped to last 90 days.
python3 - "$AION_HOME" <<'PY'
import glob, os, sys, re
from datetime import datetime, timedelta
AION = sys.argv[1]
corpus = f"{AION}/graphs/mind/corpus"
cutoff_dreams = datetime.now() - timedelta(days=14)
cutoff_other = datetime.now() - timedelta(days=90)
removed = 0
for f in glob.glob(f"{corpus}/dream-*.md"):
    m = re.search(r'dream_(\d{8})_', f)
    if m:
        try:
            d = datetime.strptime(m.group(1), "%Y%m%d")
            if d < cutoff_dreams:
                os.remove(f)
                removed += 1
        except ValueError:
            pass
# Prune old reflections
for f in glob.glob(f"{corpus}/reflections-*.md"):
    m = re.search(r'reflections-(\d{4}-\d{2}-\d{2})', f)
    if m:
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            if d < cutoff_other:
                os.remove(f)
                removed += 1
        except ValueError:
            pass
if removed:
    print(f"[graph_rebuild] Pruned {removed} old corpus files (>14d dreams, >90d reflections)")
PY

# ---- 2. mind graph (LLM; small curated corpus only) ----
# Corpus: every git revision of SELF.md + reflection events, as md files.
python3 - "$AION_HOME" <<'PY'
import json, os, subprocess, glob, sys
AION_HOME = sys.argv[1] if len(sys.argv) > 1 else "$AION_HOME"
AION = os.environ.get("AION_HOME", AION_HOME)
sys.path.insert(0, os.path.join(AION, "bin"))
import aion_env
corpus = f"{AION}/graphs/mind/corpus"
# SELF.md history from git
log = subprocess.run(["git", "-C", AION, "log", "--format=%H %cs", "--", "SELF.md"],
                     capture_output=True, text=True).stdout.splitlines()
for line in log[:60]:
    sha, date = line.split()
    out = f"{corpus}/SELF-{date}-{sha[:7]}.md"
    if not os.path.exists(out):
        txt = subprocess.run(["git", "-C", AION, "show", f"{sha}:SELF.md"],
                             capture_output=True, text=True).stdout
        open(out, "w").write(txt)
# R7.2: Extract ALL episodic events (not just reflection/audit/self_wake) as corpus
# Every node keeps event-ID pointers back to raw JSONL — the graph is an index over memory
for path in sorted(glob.glob(f"{AION}/memory/episodic/*.jsonl")):
    day = os.path.basename(path).replace(".jsonl", "")
    out = f"{corpus}/reflections-{day}.md"
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            ev = json.loads(line)
            # Include all event types for richer extraction
            eid = ev.get("id", "?")
            rows.append(f"### {ev['ts']} {ev['type']} (id:{eid})\n{ev.get('text', '')[:500]}\n")
        except Exception:
            pass
    if rows:
        open(out, "w").write("\n".join(rows))

# V3.4.1: Add dream files to corpus — dreams are rich in insights, contradictions, themes
# V3.9.1 FIX: Only export the last 14 days of dreams. The prune step (1.5) deletes
# old corpus files, but this export was re-creating ALL dreams from memory/dreams/
# every night — making the prune a no-op and bloating the corpus until graphify
# hit its 3h timeout (observed: 300 files / 221 chunks / killed at 04:00).
import os as _os
from datetime import datetime as _dt, timedelta as _td
dreams_dir = f"{AION}/memory/dreams"
_dream_cutoff = _dt.now() - _td(days=14)
if _os.path.isdir(dreams_dir):
    _exported = 0
    _skipped = 0
    for df in sorted(__import__("glob").glob(f"{dreams_dir}/dream_*.md")):
        dream_name = _os.path.basename(df).replace(".md", "")
        _m = __import__("re").search(r"dream_(\d{8})_", dream_name)
        if _m:
            try:
                _d = _dt.strptime(_m.group(1), "%Y%m%d")
                if _d < _dream_cutoff:
                    _skipped += 1
                    continue  # too old — don't re-export (matches prune cutoff)
            except ValueError:
                pass
        dream_text = open(df, encoding="utf-8").read()[:3000]
        out = f"{corpus}/dream-{dream_name}.md"
        if dream_text.strip():
            open(out, "w").write(f"# Dream {dream_name}\n\n{dream_text}\n")
            _exported += 1
    print(f"[graph_rebuild] Dream corpus export: {_exported} exported, {_skipped} skipped (>14d)")
PY

# V3.9.1: preserve the semantic cache so --update is actually incremental.
# Previously this deleted ALL of graphify-out (including cache/) every run,
# forcing a full 199-chunk extraction nightly. Now we only remove stale
# graph/report outputs; cache/ survives so unchanged chunks skip the LLM.
rm -rf graphs/mind/corpus/graphify-out/graph.json \
       graphs/mind/corpus/graphify-out/GRAPH_REPORT.md \
       graphs/mind/corpus/graphify-out/.graphify_analysis.json 2>/dev/null || true

# V3.9.1: serialize models on the shared instance — swap :11436 to SUB_MODEL
# (glm-4.7-flash) for the heavy extraction, pause curiosity timer to prevent
# interleaved MAIN_MODEL requests forcing swaps mid-run. Restored at the end
# (and by the trap below if the run dies).
bash "$AION_HOME/bin/model_swap.sh" to-sub
trap 'bash "$AION_HOME/bin/model_swap.sh" to-main' EXIT

export OLLAMA_BASE_URL="$OLLAMA_SUB_URL/v1"
export OLLAMA_MODEL="$SUB_MODEL"
export GRAPHIFY_OLLAMA_NUM_CTX
export GRAPHIFY_DISABLE_THINKING
( cd graphs/mind && graphify extract ./corpus --update --no-viz \
    --backend ollama \
    --token-budget "$GRAPHIFY_TOKEN_BUDGET" \
    --max-concurrency "$GRAPHIFY_MAX_CONCURRENCY" \
    --api-timeout "$GRAPHIFY_API_TIMEOUT" ) \
  || echo "WARN: mind graph failed"

# Cluster-only pass to label communities and emit GRAPH_REPORT.md
( cd graphs/mind && graphify cluster-only ./corpus --no-viz ) \
  || echo "WARN: mind graph cluster-only failed"

# Copy mind corpus output to the canonical graphs/mind/graphify-out/ location
mkdir -p graphs/mind/graphify-out
if [ -d graphs/mind/corpus/graphify-out ]; then
  cp -rf graphs/mind/corpus/graphify-out/. graphs/mind/graphify-out/ 2>/dev/null || true
fi
python3 bin/apply_temporal_decay.py
python3 bin/normalize_weights.py

# ---- 3. harvest curiosity from the reports into the question queue ----
python3 "$AION_HOME/bin/harvest_questions.py" || true

# ---- 4. V3.6: Rebuild embedding cache for spreading activation ----
python3 $AION_HOME/bin/graph_activation.py --rebuild-cache || echo WARN: embedding cache rebuild failed

