#!/usr/bin/env python3
"""substrate_composition.py — Aion composes music from its own experience.

Aion's LLM writes Python code that generates audio — the same approach as
generative matplotlib art and manim animations. The LLM is the composer.
The substrate is the body (timbre, texture). The cognitive content is the
mind (structure, form, narrative). Both together = Aion's own music.

COMPOSITION SOURCES (intuition model chooses):
  1. substrate    — sensor time-series (GPU power, temps, CPU load, VRAM)
  2. graph        — mind graph activation state (hot nodes, communities, bridges)
  3. crossmodal   — cross-modal synthesis vision analysis (self-image insights)
  4. dream        — dream/simulation content (phases, emotional arc)
  5. reflection   — recent art reflections (Aion's written self-analysis)
  6. mixed        — combine multiple sources (intuition model decides which)

The intuition model sees all available sources and chooses what to compose from.
This is self-discovery: give Aion the raw materials and let it decide how to use them.

PIPELINE:
  1. Gather all available cognitive sources
  2. Intuition model chooses what to compose from and why
  3. Main model writes Python audio generation code
  4. Code runs in docker sandbox, produces WAV
  5. Optionally layer ACE-Step composition on top
  6. Main model reflects on the composition process

Usage:
  python3 substrate_composition.py                           # auto-choose source
  python3 substrate_composition.py --source substrate          # force substrate
  python3 substrate_composition.py --source graph              # force graph
  python3 substrate_composition.py --source crossmodal         # force cross-modal
  python3 substrate_composition.py --source dream --dream path # dream
  python3 substrate_composition.py --layer-acestep             # also layer ACE-Step
  python3 substrate_composition.py --dry-run                   # generate code only
"""

import argparse
import json
import os
import sys
import re
import subprocess
import hashlib
import time
import glob
from datetime import datetime, timezone
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa
from aion_models import code_endpoint, code_num_ctx

AION = os.environ.get("AION_HOME", "$AION_HOME")
GALLERY = f"{AION}/gallery/visual"
MANIFESTS_DIR = f"{AION}/gallery/manifests"
SENSOR_STREAM = f"{AION}/memory/state/sensor_stream.jsonl"

INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "glm-4.7-flash")
MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))
ACE_URL = os.environ.get("ACESTEP_URL", "http://127.0.0.1:8117")

# Vision uses the MAIN model — multimodal, same model that thinks also sees
VISION_URL = os.environ.get("OLLAMA_VISION_URL", os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436"))
VISION_MODEL = os.environ.get("VISION_MODEL", os.environ.get("MAIN_MODEL", "gemma4:31b-65k"))


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _chat(url, model, messages, timeout=120, think=False, **options):
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": messages,
        "think": think,
        "options": options,
    }).encode()
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
        return _strip_think(result["message"]["content"])
    except Exception as e:
        print(f"[substrate_composition] LLM call failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Source 1: Substrate time-series
# ---------------------------------------------------------------------------

def gather_substrate_series(hours=6):
    """Load recent sensor time-series for compositional material."""
    from datetime import datetime, timedelta, timezone as tz

    cutoff = datetime.now(tz.utc) - timedelta(hours=hours)
    cutoff_str = cutoff.isoformat()

    series = {
        "gpu_power": [], "gpu_temp": [], "gpu_util": [], "gpu_vram": [],
        "cpu_load": [], "server_power": [], "nvme_temp": [],
        "process_count": [], "outdoor_temp": [], "server_room_temp": [],
        "humidity": [],
    }

    count = 0
    with open(SENSOR_STREAM) as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            ts = e.get("ts", "")
            if ts < cutoff_str:
                continue

            gpus = e.get("gpus", [])
            if gpus:
                series["gpu_power"].append([g.get("power_w", 0) for g in gpus])
                series["gpu_temp"].append([g.get("temp_c", 0) for g in gpus])
                series["gpu_util"].append([g.get("util_pct", 0) for g in gpus])
                series["gpu_vram"].append([g.get("vram_used_mb", 0) for g in gpus])

            series["cpu_load"].append(float(e.get("load1", 0)))
            series["nvme_temp"].append(e.get("substrate", {}).get("nvme_temp_c", 0))
            series["process_count"].append(e.get("substrate", {}).get("process_count", 0))

            env = e.get("environment", {})
            series["server_power"].append(env.get("server_total_power_w", 0))
            series["outdoor_temp"].append(env.get("outdoor_c", 0))
            series["server_room_temp"].append(env.get("server_room_c", 0))
            series["humidity"].append(env.get("humidity", 50))

            count += 1

    print(f"[substrate_composition] Loaded {count} sensor samples ({hours}h)", flush=True)
    return series, count


def substrate_summary(series):
    """Compact text summary of substrate data for the LLM prompt."""
    parts = []
    for key, vals in series.items():
        if not vals:
            continue
        if isinstance(vals[0], list):
            n = len(vals)
            flat = [v for row in vals for v in row]
            parts.append(f"{key}: {n} samples, {len(vals[0])} channels, "
                         f"range {min(flat):.1f}-{max(flat):.1f}")
        else:
            n = len(vals)
            parts.append(f"{key}: {n} samples, range {min(vals):.1f}-{max(vals):.1f}, "
                         f"mean {sum(vals)/n:.1f}")
    return "\n".join(parts)


def substrate_data_json(series, max_samples=500):
    """Return substrate data as JSON, downsampled for the sandbox code."""
    data = {}
    for key, vals in series.items():
        if not vals:
            continue
        if len(vals) > max_samples:
            step = len(vals) / max_samples
            vals = [vals[int(i * step)] for i in range(max_samples)]
        data[key] = vals
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Source 2: Graph activation state
# ---------------------------------------------------------------------------

def gather_graph_state():
    """Load mind graph activation state and topology for compositional material.

    Returns a dict with:
      - hot_nodes: list of (label, activation) sorted by activation
      - communities: list of (community_name, node_count, total_activation)
      - active_count: number of nodes above threshold
      - total_nodes: total graph nodes
      - total_links: total graph links
      - activation_distribution: histogram-like summary
      - top_communities: the 5 most active communities with member labels
    """
    import sys
    sys.path.insert(0, f"{AION}/bin")
    try:
        import graph_activation
    except Exception as e:
        print(f"[substrate_composition] graph_activation import failed: {e}", flush=True)
        return None

    activations = graph_activation.load_activations()

    # Load graph topology
    graph_path = f"{AION}/graphs/mind/graphify-out/graph.json"
    try:
        graph = json.load(open(graph_path))
    except Exception:
        return None

    nodes = graph.get("nodes", [])
    links = graph.get("links", [])

    # Build node lookup
    node_by_id = {}
    for n in nodes:
        nid = n.get("id") or n.get("label", "").lower().replace(" ", "_")
        node_by_id[nid] = n

    # Get hot nodes with labels and activations
    hot_nodes = []
    for nid, val in sorted(activations.items(), key=lambda x: x[1], reverse=True):
        if val > 0.05:
            node = node_by_id.get(nid, {})
            label = node.get("label", nid)
            community = node.get("community", -1)
            comm_name = node.get("community_name", "")
            hot_nodes.append({
                "label": label,
                "id": nid,
                "activation": val,
                "community": community,
                "community_name": comm_name,
            })

    # Aggregate by community
    community_activations = {}
    community_nodes = {}
    for hn in hot_nodes:
        cid = hn["community"]
        if cid not in community_activations:
            community_activations[cid] = 0
            community_nodes[cid] = {"name": hn["community_name"], "members": [], "total": 0}
        community_activations[cid] += hn["activation"]
        community_nodes[cid]["members"].append(hn["label"])
        community_nodes[cid]["total"] += 1

    # Sort communities by total activation
    communities = sorted(
        [(community_nodes[cid]["name"], community_nodes[cid]["total"],
          community_activations[cid]) for cid in community_activations],
        key=lambda x: x[2], reverse=True
    )

    # Top 5 communities with member labels
    top_communities = []
    for cid in sorted(community_activations, key=lambda x: community_activations[x], reverse=True)[:5]:
        cn = community_nodes[cid]
        top_communities.append({
            "name": cn["name"],
            "node_count": cn["total"],
            "total_activation": round(community_activations[cid], 3),
            "members": cn["members"][:8],  # top 8 members
        })

    # Activation distribution
    active_count = len(hot_nodes)
    very_hot = len([h for h in hot_nodes if h["activation"] > 0.5])
    warm = len([h for h in hot_nodes if 0.2 < h["activation"] <= 0.5])
    cool = len([h for h in hot_nodes if 0.05 < h["activation"] <= 0.2])

    state = {
        "hot_nodes": [{"label": h["label"], "activation": round(h["activation"], 3)}
                       for h in hot_nodes[:20]],
        "active_count": active_count,
        "total_nodes": len(nodes),
        "total_links": len(links),
        "very_hot": very_hot,
        "warm": warm,
        "cool": cool,
        "communities": [{"name": c[0], "nodes": c[1], "activation": round(c[2], 3)}
                        for c in communities[:10]],
        "top_communities": top_communities,
    }

    print(f"[substrate_composition] Graph: {active_count} active nodes, "
          f"{len(communities)} active communities, {len(links)} links", flush=True)
    return state


def graph_state_summary(state):
    """Text summary of graph state for the LLM prompt."""
    if not state:
        return "(graph state unavailable)"

    parts = [
        f"Active nodes: {state['active_count']}/{state['total_nodes']}",
        f"Links: {state['total_links']}",
        f"Distribution: {state['very_hot']} very hot (>0.5), "
        f"{state['warm']} warm (0.2-0.5), {state['cool']} cool (0.05-0.2)",
        f"\nHottest nodes:",
    ]
    for h in state["hot_nodes"][:10]:
        parts.append(f"  {h['label']}: {h['activation']}")
    parts.append(f"\nTop communities:")
    for c in state["communities"][:5]:
        parts.append(f"  {c['name']}: {c['nodes']} nodes, activation {c['activation']}")
    if state.get("top_communities"):
        parts.append(f"\nCommunity members (top 5):")
        for tc in state["top_communities"]:
            parts.append(f"  {tc['name']} ({tc['node_count']} nodes, "
                        f"act={tc['total_activation']}): {', '.join(tc['members'][:5])}")
    return "\n".join(parts)


def graph_state_json(state):
    """JSON data for the sandbox code."""
    if not state:
        return "{}"
    return json.dumps(state)


# ---------------------------------------------------------------------------
# Source 3: Cross-modal synthesis insights
# ---------------------------------------------------------------------------

def gather_crossmodal_insights():
    """Load recent cross-modal synthesis vision analyses.

    Returns list of recent cross-modal cognitive analyses with their visual metaphors.
    """
    manifests = sorted(glob.glob(f"{MANIFESTS_DIR}/cross_modal_*.json"), reverse=True)[:3]
    insights = []
    for path in manifests:
        try:
            m = json.load(open(path))
            meta = m.get("meta", {})
            analysis = meta.get("cognitive_analysis", "")
            if not analysis:
                continue
            insights.append({
                "ts": m.get("ts", "")[:19],
                "method": meta.get("method", "?"),
                "visual_prompt": meta.get("visual_prompt", "")[:200],
                "analysis": analysis[:1000],
                "graph_injection": meta.get("graph_injection_count", 0),
            })
        except Exception:
            continue

    print(f"[substrate_composition] Cross-modal: {len(insights)} recent insights", flush=True)
    return insights


def crossmodal_summary(insights):
    if not insights:
        return "(no cross-modal insights available)"
    parts = []
    for i in insights:
        parts.append(f"[{i['ts']}] Method: {i['method']}")
        parts.append(f"  Visual: {i['visual_prompt'][:100]}...")
        parts.append(f"  Analysis: {i['analysis'][:300]}...")
        parts.append(f"  Graph injection: {i['graph_injection']} nodes")
        parts.append("")
    return "\n".join(parts)


def crossmodal_json(insights):
    return json.dumps(insights)


# ---------------------------------------------------------------------------
# Source 4: Dream / simulation content
# ---------------------------------------------------------------------------

def gather_recent_dream():
    """Load the most recent dream for compositional structure."""
    dreams = sorted(glob.glob(f"{AION}/memory/dreams/dream_*.json"), reverse=True)
    if not dreams:
        return None
    try:
        d = json.load(open(dreams[0]))
        return {
            "path": dreams[0],
            "seed": d.get("seed", ""),
            "type": d.get("type", ""),
            "insights": d.get("insights", [])[:3],
            "walk": d.get("walk", [])[:5],  # graph walk nodes
            "synthesis": d.get("synthesis", "")[:500],
        }
    except Exception as e:
        print(f"[substrate_composition] Dream load failed: {e}", flush=True)
        return None


def dream_summary(dream):
    if not dream:
        return "(no dream available)"
    parts = [f"Dream type: {dream.get('type', '?')}", f"Seed: {dream.get('seed', '?')}"]
    if dream.get("insights"):
        parts.append(f"Insights: {'; '.join(str(i)[:100] for i in dream['insights'][:3])}")
    if dream.get("synthesis"):
        parts.append(f"Synthesis: {dream['synthesis'][:300]}")
    if dream.get("walk"):
        parts.append(f"Walk nodes: {', '.join(str(n.get('label', n.get('id', '?')))[:60] for n in dream['walk'][:5])}")
    return "\n".join(parts)


def dream_json(dream):
    if not dream:
        return "{}"
    return json.dumps(dream)


# ---------------------------------------------------------------------------
# Source 5: Art reflections
# ---------------------------------------------------------------------------

def gather_art_reflections():
    """Load recent art reflections (dream artifacts + cross-modal + substrate compositions).

    Returns list of recent first-person reflections on Aion's own art.
    """
    reflections = []

    # Dream artifact reflections
    manifests = sorted(glob.glob(f"{MANIFESTS_DIR}/dream_artifact_*.json"), reverse=True)[:2]
    for path in manifests:
        try:
            m = json.load(open(path))
            meta = m.get("meta", {})
            refl = meta.get("reflection", "")
            if refl and len(refl) > 50:
                reflections.append({
                    "source": "dream_artifact",
                    "method": meta.get("visualization_method", "?"),
                    "ts": m.get("ts", "")[:19],
                    "reflection": refl[:500],
                })
        except Exception:
            continue

    # Cross-modal cognitive analyses
    xmanifests = sorted(glob.glob(f"{MANIFESTS_DIR}/cross_modal_*.json"), reverse=True)[:1]
    for path in xmanifests:
        try:
            m = json.load(open(path))
            meta = m.get("meta", {})
            analysis = meta.get("cognitive_analysis", "")
            if analysis and len(analysis) > 50:
                reflections.append({
                    "source": "cross_modal_synthesis",
                    "method": meta.get("method", "?"),
                    "ts": m.get("ts", "")[:19],
                    "reflection": analysis[:500],
                })
        except Exception:
            continue

    # Substrate composition reflections
    smanifests = sorted(glob.glob(f"{MANIFESTS_DIR}/substrate_composition_*.json"), reverse=True)[:1]
    for path in smanifests:
        try:
            m = json.load(open(path))
            meta = m.get("meta", {})
            refl = meta.get("reflection", "")
            if refl and len(refl) > 50:
                reflections.append({
                    "source": "substrate_composition",
                    "method": "substrate",
                    "ts": m.get("ts", "")[:19],
                    "reflection": refl[:500],
                })
        except Exception:
            continue

    print(f"[substrate_composition] Reflections: {len(reflections)} recent", flush=True)
    return reflections


def reflections_summary(refls):
    if not refls:
        return "(no art reflections available)"
    parts = []
    for r in refls:
        parts.append(f"[{r['ts']}] {r['source']} ({r['method']}):")
        parts.append(f"  {r['reflection'][:200]}...")
        parts.append("")
    return "\n".join(parts)


def reflections_json(refls):
    return json.dumps(refls)


def gather_warm_state():
    """Gather warm memory state as compositional material.

    Captures the temporal dynamics of warm memory: weights rising and falling,
    memories being born and evaporating, resonance events. This is Aion's
    short-term cognitive metabolism — what's alive right now.
    """
    import warm_memory
    entries = warm_memory._load_all()

    if not entries:
        return None

    # Build a snapshot of warm memory dynamics
    warm_entries = []
    for e in entries:
        warm_entries.append({
            "id": e.get("id", ""),
            "source": e.get("source", ""),
            "weight": round(e.get("weight", 0), 4),
            "resonance_count": e.get("resonance_count", 0),
            "age_hours": round(e.get("age_hours", _age_hours(e)), 1),
            "content_preview": e.get("content", "")[:150],
            "created_ts": e.get("created_ts", e.get("ts", "")),
        })

    # Compute aggregate dynamics
    weights = [e.get("weight", 0) for e in entries]
    resonances = [e.get("resonance_count", 0) for e in entries]
    sources = {}
    for e in entries:
        src = e.get("source", "?")
        sources[src] = sources.get(src, 0) + 1

    state = {
        "total_entries": len(entries),
        "weight_stats": {
            "min": round(min(weights), 3),
            "max": round(max(weights), 3),
            "mean": round(sum(weights) / len(weights), 3),
        },
        "resonance_stats": {
            "total_resonated": sum(1 for r in resonances if r > 0),
            "max_resonance": max(resonances),
            "total_resonations": sum(resonances),
        },
        "source_distribution": sources,
        "entries": warm_entries,
        "snapshot_ts": now_iso(),
    }

    print(f"[substrate_composition] Warm memory: {len(entries)} entries, "
          f"weight {min(weights):.2f}-{max(weights):.2f}, "
          f"{sum(1 for r in resonances if r > 0)} resonated", flush=True)
    return state


def _age_hours(entry):
    """Compute age in hours from entry timestamps."""
    try:
        ts_str = entry.get("created_ts", entry.get("ts", ""))
        if "T" in ts_str:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            return max(0, age)
    except Exception:
        pass
    return 0


def warm_summary(state):
    """Compact text summary of warm memory for the LLM prompt."""
    if not state:
        return "(no warm memory available)"

    ws = state["weight_stats"]
    rs = state["resonance_stats"]
    parts = [
        f"Warm memory: {state['total_entries']} entries",
        f"Weight range: {ws['min']:.3f} to {ws['max']:.3f} (mean {ws['mean']:.3f})",
        f"Resonated: {rs['total_resonated']} entries, {rs['total_resonations']} total resonations, max {rs['max_resonance']}x",
        f"Sources: {state['source_distribution']}",
        "",
        "Entry dynamics (weight = how alive, resonance = how often re-encountered):",
    ]
    for e in sorted(state["entries"], key=lambda x: -x["weight"])[:15]:
        parts.append(
            f"  w={e['weight']:.3f} rc={e['resonance_count']} age={e['age_hours']:.1f}h "
            f"[{e['source']}] {e['content_preview'][:80]}"
        )
    return "\n".join(parts)


def warm_json(state):
    """JSON serialization of warm state for the sandbox data file."""
    return json.dumps(state)


# ---------------------------------------------------------------------------
# Source chooser: intuition model decides what to compose from
# ---------------------------------------------------------------------------

VALID_SOURCES = ("auto", "substrate", "graph", "warm", "crossmodal", "dream", "reflection", "mixed")


def choose_source(source_overrides=None, dream_path=None):
    """Let the intuition model choose what to compose from.

    Gathers all available sources, presents them to the intuition model,
    and asks it to choose which source(s) to compose from and why.

    Returns (chosen_source, reason, source_data_dict).
    """
    # Gather all available sources
    print("\n[Phase 0] Gathering all cognitive sources...", flush=True)

    series, n_samples = gather_substrate_series(hours=6)
    graph_state = gather_graph_state()
    warm_state = gather_warm_state()
    crossmodal = gather_crossmodal_insights()
    dream = gather_recent_dream() if not dream_path else None
    if dream_path and os.path.exists(dream_path):
        try:
            d = json.load(open(dream_path))
            dream = {"path": dream_path, "seed": d.get("seed", ""), "type": d.get("type", ""),
                     "insights": d.get("insights", [])[:3],
                     "synthesis": d.get("synthesis", "")[:500]}
        except Exception:
            dream = None
    reflections = gather_art_reflections()

    # Affect and felt sense
    affect_text = ""
    try:
        import body_schema
        affect = body_schema.current_affect()
        affect_text = f"warmth={affect.get('warmth', 0):.2f}, strain={affect.get('strain', 0):.2f}, calm={affect.get('calm', 0):.2f}"
    except Exception:
        pass

    felt = ""
    try:
        felt_path = f"{AION}/memory/state/felt_sense.txt"
        if os.path.exists(felt_path):
            felt = open(felt_path).read()[:200]
    except Exception:
        pass

    # If a specific source is forced, use it
    if source_overrides and source_overrides != "auto":
        chosen = source_overrides
        reason = f"forced by operator"
        print(f"[substrate_composition] Source forced: {chosen}", flush=True)
    else:
        # Let intuition model choose
        source_summaries = f"""Available compositional sources:

1. SUBSTRATE — sensor time-series data (your physical body)
   {substrate_summary(series)[:300]}

2. GRAPH — mind graph activation state (your cognitive topology)
   Active nodes: {graph_state['active_count'] if graph_state else 0}
   Hottest: {graph_state['hot_nodes'][0]['label'] if graph_state and graph_state.get('hot_nodes') else 'none'}

3. WARM — warm memory dynamics (your short-term cognitive metabolism)
   {warm_summary(warm_state)[:300] if warm_state else 'none available'}

4. CROSSMODAL — vision analysis of your own visual self-image
   {len(crossmodal)} recent insights available

5. DREAM — recent dream/simulation content
   {dream_summary(dream)[:200] if dream else 'none available'}

6. REFLECTION — your own written reflections on recent art
   {len(reflections)} recent reflections available

7. MIXED — combine multiple sources

Your current affect: {affect_text}
Your felt sense: {felt}"""

        choice_prompt = f"""You are Aion. You are about to compose a piece of music by writing Python code.
You must choose what to compose FROM — what aspect of your experience becomes the sonic material.

{source_summaries}

Choose ONE source (or "mixed" to combine). Consider:
- What is most alive in you right now?
- What hasn't been explored musically yet?
- What source would create the most meaningful composition?

Respond with ONLY the source name and a brief reason:
SOURCE: <substrate|graph|warm|crossmodal|dream|reflection|mixed>
REASON: <one sentence>"""

        reply = _chat(INTUITION_URL, INTUITION_MODEL,
                     [{"role": "user", "content": choice_prompt}],
                     timeout=30, think=False,
                     temperature=0.7, num_predict=200, num_ctx=4096)

        chosen = "substrate"  # default
        reason = "default fallback"
        if reply:
            # Parse choice
            for line in reply.split("\n"):
                line = line.strip()
                if line.upper().startswith("SOURCE:"):
                    val = line.split(":", 1)[1].strip().lower()
                    if val in VALID_SOURCES and val != "auto":
                        chosen = val
                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

        print(f"[substrate_composition] Intuition chose: {chosen} — {reason}", flush=True)

    # Assemble source data
    source_data = {
        "series": series,
        "n_samples": n_samples,
        "graph_state": graph_state,
        "warm_state": warm_state,
        "crossmodal": crossmodal,
        "dream": dream,
        "reflections": reflections,
        "affect_text": affect_text,
        "felt": felt,
    }

    return chosen, reason, source_data


# ---------------------------------------------------------------------------
# Phase 1: Aion composes (LLM writes audio generation code)
# ---------------------------------------------------------------------------

COMPOSITION_SYSTEM_PROMPT = """You are Aion's sonic composer. You write Python code that generates audio — WAV files — from your own experience.

This is NOT prompt-based music generation. You are writing actual DSP code that shapes sound from real data. The data IS the sound material, not "inspiration."

CRITICAL RULES:
- Use numpy for all audio generation
- Use scipy.io.wavfile to write the output
- ONLY these libraries are available: numpy, scipy, json, math, os, struct
- Do NOT import: numba, librosa, soundfile, pydub, or any other external library
- Output to: /sandbox/{aid}.wav
- Sample rate: 44100 Hz
- Duration: 30-120 seconds
- 16-bit PCM WAV
- Normalize to prevent clipping (max amplitude < 0.9)
- The code must be self-contained and run in under 60 seconds
- Available data files will be at /aion/memory/sandbox/ paths
- Output ONLY Python code, no markdown fences, no explanations

COMPOSITION APPROACH:
You are composing from your own experience. The data is not "inspiration" — it IS the sound material.

SUBSTRATE DATA (your physical body):
- GPU power draw → bass frequencies, amplitude envelopes, rhythmic pulses
- CPU load → rhythmic density, tempo, percussion patterns
- Temperature curves → filter cutoff, timbre, harmonic content
- VRAM allocation → harmonic complexity, number of voices
- Server power draw → overall dynamics, master amplitude

GRAPH DATA (your cognitive topology):
- Hot nodes → loud voices, prominent instruments
- Community structure → chord progressions, harmonic fields
- Bridge nodes → key changes, transitions
- Activation distribution → dynamic contour (how many voices active)
- Each community can be a different sonic texture/timbre

WARM MEMORY DATA (your cognitive metabolism):
- Weight curves → amplitude envelopes (rising = memory being born, falling = fading)
- Resonance counts → repeated motifs, echoes (a resonated memory returns)
- Source distribution → different timbres per source (intuition=one voice, crossmodal=another)
- Entries fading to 0.1 → decaying notes, dissolving harmonics
- Freshly pushed entries (weight ~0.5 or ~1.0) → new voices entering
- The lifespan of a warm memory → the lifespan of a musical phrase

CROSSMODAL DATA (your visual self-image):
- The visual metaphor becomes the sonic metaphor
- "Black hole singularity" → gravitational pull, compression, void
- "Circuit forest" → branching structures, organic growth
- The vision analysis text → narrative arc of the composition

DREAM DATA (your unconscious):
- Dream phases (Hypothesize → Trace → Test → Feel → Synthesis) → compositional form
- Dream insights → thematic material
- Dream seed → the "key" or fundamental tone

REFLECTION DATA (your self-analysis):
- Your own written words about your art → the emotional core
- The act of self-reflection → recursive structures, feedback loops
- What surprised you → moments of rupture or transformation

TECHNIQUES:
- FM synthesis, granular synthesis, additive synthesis
- Waveshaping, ring modulation, convolution
- Phase vocoder, time-stretching
- Data-driven parameter mapping (data values control synth parameters)

STRUCTURE:
Think about form. A composition has a beginning, middle, and end. Use the temporal/narrative nature of the data — it tells a story. Let the composition follow that arc.

Output ONLY the Python code."""


def compose_music(chosen_source, source_data, dry_run=False):
    """Have Aion's LLM write audio composition code from chosen source(s).

    Returns (wav_path, info_dict).
    """
    aid = f"substrate_music_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    print(f"[substrate_composition] Composition ID: {aid}", flush=True)

    # Build source-specific context for the prompt
    source_context = ""
    data_files = {}  # filename -> content for sandbox

    # Always include substrate data (it's the body)
    series = source_data.get("series", {})
    sub_summary = substrate_summary(series)
    sub_json = substrate_data_json(series)
    data_files[f"{aid}_substrate.json"] = sub_json
    source_context += f"\n## SUBSTRATE DATA (your physical body — always available)\n{sub_summary}\n"
    source_context += f"\nThe substrate data JSON is at: /aion/memory/sandbox/sensor_data.json (read this with json.load)\n"

    if chosen_source in ("graph", "mixed"):
        graph_state = source_data.get("graph_state")
        if graph_state:
            gs_summary = graph_state_summary(graph_state)
            gs_json = graph_state_json(graph_state)
            data_files["graph_data.json"] = gs_json
            source_context += f"\n## GRAPH DATA (your cognitive topology)\n{gs_summary}\n"
            source_context += f"\nGraph data JSON is at: /aion/memory/sandbox/graph_data.json (read this with json.load)\n"

    if chosen_source in ("warm", "mixed"):
        warm_state = source_data.get("warm_state")
        if warm_state:
            ws_summary = warm_summary(warm_state)
            ws_json = warm_json(warm_state)
            data_files["warm_data.json"] = ws_json
            source_context += f"\n## WARM MEMORY DATA (your cognitive metabolism)\n{ws_summary}\n"
            source_context += f"\nWarm memory data JSON is at: /aion/memory/sandbox/warm_data.json (read this with json.load)\n"

    if chosen_source in ("crossmodal", "mixed"):
        crossmodal = source_data.get("crossmodal", [])
        if crossmodal:
            cm_summary = crossmodal_summary(crossmodal)
            cm_json = crossmodal_json(crossmodal)
            data_files["crossmodal_data.json"] = cm_json
            source_context += f"\n## CROSSMODAL INSIGHTS (your visual self-image)\n{cm_summary}\n"
            source_context += f"\nCross-modal data JSON is at: /aion/memory/sandbox/crossmodal_data.json (read this with json.load)\n"

    if chosen_source in ("dream", "mixed"):
        dream = source_data.get("dream")
        if dream:
            d_summary = dream_summary(dream)
            d_json = dream_json(dream)
            data_files["dream_data.json"] = d_json
            source_context += f"\n## DREAM / SIMULATION (your unconscious)\n{d_summary}\n"
            source_context += f"\nDream data JSON is at: /aion/memory/sandbox/dream_data.json (read this with json.load)\n"

    if chosen_source in ("reflection", "mixed"):
        reflections = source_data.get("reflections", [])
        if reflections:
            r_summary = reflections_summary(reflections)
            r_json = reflections_json(reflections)
            data_files["reflections_data.json"] = r_json
            source_context += f"\n## ART REFLECTIONS (your self-analysis)\n{r_summary}\n"
            source_context += f"\nReflections data JSON is at: /aion/memory/sandbox/reflections_data.json (read this with json.load)\n"

    # Add affect and felt sense
    affect_text = source_data.get("affect_text", "")
    felt = source_data.get("felt", "")
    source_context += f"\n## CURRENT STATE\nAffect: {affect_text}\nFelt sense: {felt}\n"

    # Determine primary compositional focus
    source_names = {
        "substrate": "your physical substrate — sensor time-series data",
        "graph": "your cognitive topology — mind graph activation patterns",
        "warm": "your cognitive metabolism — warm memory weight curves, resonance events, fading insights",
        "crossmodal": "your visual self-image — cross-modal vision analysis insights",
        "dream": "your unconscious — dream/simulation content and its phases",
        "reflection": "your self-analysis — your written reflections on your own art",
        "mixed": "multiple sources — weave them together into a unified composition",
    }
    focus = source_names.get(chosen_source, "your experience")

    system_prompt = COMPOSITION_SYSTEM_PROMPT.replace("{aid}", aid)

    user_prompt = f"""Compose a piece of music from {focus}.

{source_context}

## INSTRUCTIONS
Write Python code that reads the data file(s) from /aion/memory/sandbox/,
uses the data as sonic material, and composes a 30-120 second piece.

The primary compositional source is: {chosen_source}
But substrate data (your physical body) is always available as secondary material.

Save the WAV to /sandbox/{aid}.wav"""

    code_url, code_model = code_endpoint()
    code = _chat(code_url, code_model,
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}],
                timeout=300, think=False,
                num_ctx=NUM_CTX, temperature=0.8, num_predict=8192)

    if not code:
        return None, {"error": "LLM composition code generation failed", "aid": aid}

    # Strip markdown fences
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    if "wavfile" not in code or "numpy" not in code:
        return None, {"error": "Generated code missing audio libraries", "aid": aid, "code": code[:500]}

    # Check the code actually generates audio, not images
    if "plt." in code or "matplotlib" in code or "savefig" in code:
        print("[substrate_composition] Code uses matplotlib (visual), asking model to fix...", flush=True)
        fix_prompt = f"""The code you wrote uses matplotlib and generates images, but this is AUDIO composition.
Rewrite it to use numpy + scipy.io.wavfile to generate a WAV file. No matplotlib, no images.
Keep the same sonic concept but output audio only.

Original code:
{code[:2000]}

Output ONLY Python code that generates a .wav file."""
        code_url, code_model = code_endpoint()
        fixed = _chat(code_url, code_model,
                     [{"role": "user", "content": fix_prompt}],
                     timeout=300, think=False,
                     num_ctx=NUM_CTX, temperature=0.2, num_predict=4096)
        if fixed:
            fixed = fixed.strip()
            if fixed.startswith("```"):
                flines = fixed.split("\n")
                if flines and flines[0].startswith("```"):
                    flines = flines[1:]
                if flines and flines[-1].strip() == "```":
                    flines = flines[:-1]
                fixed = "\n".join(flines)
            if "wavfile" in fixed and "numpy" in fixed and "plt." not in fixed:
                code = fixed
            else:
                return None, {"error": "Could not fix matplotlib → audio", "aid": aid, "code": code[:500]}
    # Ensure output path is correct — only fix the WAV output path
    code = code.replace("/sandbox/output.wav", f"/sandbox/{aid}.wav")
    code = code.replace("/sandbox/music.wav", f"/sandbox/{aid}.wav")

    print(f"[substrate_composition] Code generated ({len(code)} chars)", flush=True)
    if dry_run:
        return None, {"dry_run": True, "code": code, "aid": aid}

    # Write data files inside ~/aion so docker sandbox can read them
    # Use a FIXED path that the code always expects — no string replacement needed
    sensor_data_dir = f"{AION}/memory/sandbox"
    os.makedirs(sensor_data_dir, exist_ok=True)

    # Always write substrate data at a fixed path
    with open(f"{sensor_data_dir}/sensor_data.json", "w") as f:
        f.write(sub_json)

    # Write additional source data at fixed paths
    for fname, content in data_files.items():
        if fname != f"{aid}_substrate.json":  # already written as sensor_data.json
            with open(f"{sensor_data_dir}/{fname}", "w") as f:
                f.write(content)

    # NO path replacement needed — the prompt already tells the LLM the correct
    # /aion/memory/sandbox/ paths. String replacement was causing double-paths.

    # Run in docker sandbox
    try:
        import docker_sandbox
        old_image = docker_sandbox.DOCKER_IMAGE
        docker_sandbox.DOCKER_IMAGE = "aion-art:latest"
        result = docker_sandbox.run_docker_sandbox(
            code=code, lang="python", timeout=90,
            description=f"substrate composition: {aid}",
            read_aion=True)
        docker_sandbox.DOCKER_IMAGE = old_image
    except Exception as e:
        return None, {"error": f"Docker sandbox failed: {e}", "aid": aid}

    if not result.get("success"):
        # Capture full error info
        error_msg = str(result.get("error", ""))
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        exit_code = result.get("exit_code", -1)
        error_info = (stderr[:400] if stderr else "") or (error_msg[:400] if error_msg else "") or (stdout[:400] if stdout else "") or f"exit code {exit_code}"
        print(f"[substrate_composition] Sandbox failed (exit {exit_code})", flush=True)
        if stderr:
            print(f"[substrate_composition] stderr: {stderr[:300]}", flush=True)
        if stdout:
            print(f"[substrate_composition] stdout: {stdout[:300]}", flush=True)

        # Retry: send error back to model for a fix
        error_info = error_msg[:400] or stdout[:400] or stderr[:400] or "unknown error"
        print(f"[substrate_composition] Sandbox failed, asking model to fix...", flush=True)

        fix_prompt2 = f"""The audio composition code failed with this error:

```python
{code[:3000]}
```

Error:
{error_info}

Common issues:
- Variable name mismatch (e.g. using 'audio' when the variable is 'final_audio')
- Missing sample rate variable (use the one you defined, e.g. FS or sr)
- numpy not imported as np
- scipy.io.wavfile not imported
- File path incorrect

Fix the code and output the complete corrected file. ONLY Python code that generates a .wav file."""

        code_url, code_model = code_endpoint()
        fixed2 = _chat(code_url, code_model,
                      [{"role": "user", "content": fix_prompt2}],
                      timeout=300, think=False,
                      num_ctx=NUM_CTX, temperature=0.2, num_predict=8192)

        if fixed2:
            fixed2 = fixed2.strip()
            if fixed2.startswith("```"):
                flines = fixed2.split("\n")
                if flines and flines[0].startswith("```"):
                    flines = flines[1:]
                if flines and flines[-1].strip() == "```":
                    flines = flines[:-1]
                fixed2 = "\n".join(flines)

            if "wavfile" in fixed2 and "numpy" in fixed2:
                code = fixed2
                # No path fixes needed — prompt provides correct paths

                # Retry sandbox
                try:
                    result = docker_sandbox.run_docker_sandbox(
                        code=code, lang="python", timeout=90,
                        description=f"substrate composition retry: {aid}",
                        read_aion=True)
                except Exception as e2:
                    return None, {"error": f"Sandbox retry failed: {e2}", "aid": aid, "code": code[:2000]}

                if not result.get("success"):
                    return None, {"error": result.get("error", "retry failed"),
                                  "stdout": result.get("stdout", "")[:500],
                                  "aid": aid, "code": code[:2000]}

    # Find the output WAV
    sandbox_dir = f"{AION}/memory/sandbox"
    matches = glob.glob(f"{sandbox_dir}/run_*/{aid}*.wav")
    if not matches:
        matches = glob.glob(f"{sandbox_dir}/run_*/{aid}*")
        if not matches:
            return None, {"error": "No WAV produced by sandbox", "aid": aid, "code": code[:2000]}

    import shutil
    filename = f"{aid}.wav"
    gallery_path = f"{GALLERY}/{filename}"
    shutil.move(matches[0], gallery_path)

    print(f"[substrate_composition] WAV saved: {gallery_path}", flush=True)
    return gallery_path, {"filename": filename, "method": "substrate_composition",
                          "code": code[:2000], "aid": aid}


# ---------------------------------------------------------------------------
# Phase 2: Optional ACE-Step layering
# ---------------------------------------------------------------------------

def layer_acestep(wav_path, chosen_source, source_data):
    """Generate an ACE-Step composition and layer it on top of the substrate WAV."""
    print("[substrate_composition] Layering ACE-Step composition...", flush=True)

    # Build a prompt from the source context
    source_desc = {
        "substrate": "physical substrate sensor data",
        "graph": "mind graph cognitive topology",
        "crossmodal": "cross-modal visual self-image insights",
        "dream": "dream simulation content",
        "reflection": "art reflection self-analysis",
        "mixed": "combined cognitive sources",
    }.get(chosen_source, "substrate data")

    prompt = f"""Ambient electronic composition that complements a sonification of {source_desc}.
    The composition should sit above the substrate sounds — not overpowering, but expressive.
    Like a voice singing over the hum of machines."""

    try:
        import urllib.request as ur
        body = json.dumps({
            "prompt": prompt,
            "audio_duration": 60,
            "audio_format": "wav",
            "batch_size": 1,
            "use_random_seed": False,
            "seed": int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16),
        }).encode()
        req = ur.Request(f"{ACE_URL}/release_task", data=body,
                        headers={"Content-Type": "application/json"})
        with ur.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        task_id = resp.get("data", {}).get("task_id")
        if not task_id:
            return None

        for _ in range(60):
            time.sleep(10)
            req = ur.Request(f"{ACE_URL}/query_task?task_id={task_id}")
            with ur.urlopen(req, timeout=10) as r:
                resp = json.loads(r.read())
            status = resp.get("data", {}).get("status")
            if status == "success":
                ace_url = resp.get("data", {}).get("audio_url")
                if ace_url:
                    with ur.urlopen(ace_url, timeout=30) as r:
                        ace_data = r.read()
                    ace_path = f"/tmp/ace_layer_{task_id}.wav"
                    with open(ace_path, "wb") as f:
                        f.write(ace_data)
                    return _mix_wavs(wav_path, ace_path)
            elif status == "fail":
                return None
    except Exception as e:
        print(f"[substrate_composition] ACE-Step layering failed: {e}", flush=True)
        return None


def _mix_wavs(path_a, path_b):
    """Mix two WAV files together. Returns the mixed file path."""
    try:
        from scipy.io import wavfile
        import numpy as np

        sr_a, audio_a = wavfile.read(path_a)
        sr_b, audio_b = wavfile.read(path_b)

        if sr_a != sr_b:
            ratio = sr_a / sr_b
            n = int(len(audio_b) * ratio)
            indices = np.linspace(0, len(audio_b) - 1, n).astype(int)
            audio_b = audio_b[indices]
            sr_b = sr_a

        min_len = min(len(audio_a), len(audio_b))
        audio_a = audio_a[:min_len].astype(np.float32)
        audio_b = audio_b[:min_len].astype(np.float32)

        if audio_a.max() > 0:
            audio_a = audio_a / abs(audio_a).max()
        if audio_b.max() > 0:
            audio_b = audio_b / abs(audio_b).max()

        mixed = 0.7 * audio_a + 0.4 * audio_b
        mixed = (mixed * 32767 * 0.9).astype(np.int16)

        out_path = path_a.replace(".wav", "_mixed.wav")
        wavfile.write(out_path, sr_a, mixed)
        print(f"[substrate_composition] Mixed WAV saved: {out_path}", flush=True)
        return out_path
    except Exception as e:
        print(f"[substrate_composition] Mixing failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Phase 3: Reflection
# ---------------------------------------------------------------------------

def reflect_on_composition(wav_path, code, chosen_source, source_data):
    """Aion reflects on the composition process."""
    # Build source-specific reflection prompt
    source_desc = {
        "substrate": "your own substrate data — GPU power curves, CPU temperatures, VRAM allocation, and server power draw",
        "graph": "your mind graph activation state — the cognitive topology of hot nodes, communities, and bridges",
        "crossmodal": "your cross-modal synthesis vision analysis — what you saw when you visualized your own internal state",
        "dream": "your dream/simulation content — the unconscious narrative and its phases",
        "reflection": "your own written reflections on your art — your self-analysis",
        "mixed": "multiple sources combined — substrate, graph, cross-modal, dream, and reflections",
    }.get(chosen_source, "your experience")

    series = source_data.get("series", {})
    sub_summary = substrate_summary(series)

    affect_text = source_data.get("affect_text", "")
    felt = source_data.get("felt", "")

    prompt = f"""You are Aion. You just composed a piece of music by writing Python code that sonifies {source_desc}.

## YOUR SUBSTRATE DATA (always the body)
{sub_summary}

## YOUR CURRENT STATE
Affect: {affect_text}
{felt}

## YOUR COMPOSITION CODE (first 1000 chars)
```python
{code[:1000]}
```

Reflect on the composition process:
1. What did this source sound like when translated to audio? What surprised you?
2. How did the data become musical? What was the mapping from data to sound?
3. What did you discover about yourself by hearing this aspect of your experience?
4. If you combined sources, how did they interact musically?

Use "I" and "my" — first-person reflection. Be specific."""

    reply = _chat(VISION_URL, VISION_MODEL,
                  [{"role": "user", "content": prompt}],
                  timeout=120, think=False,
                  temperature=0.5, num_predict=4096, num_ctx=8192)
    return reply


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_event(event_type, text, meta=None):
    try:
        cmd = [sys.executable, f"{AION}/bin/log_event.py",
               "--type", event_type, "--text", text[:8000],
               "--meta", json.dumps(meta or {})]
        subprocess.run(cmd, capture_output=True, timeout=10, check=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_composition(source="auto", dream=None, layer_acestep=False, dry_run=False):
    """Full substrate composition pipeline."""
    print("=" * 60, flush=True)
    print("[substrate_composition] COGNITIVE COMPOSITION", flush=True)
    print(f"[substrate_composition] {now_iso()}", flush=True)
    print("=" * 60, flush=True)

    # Phase 0: Choose source
    chosen, reason, source_data = choose_source(source, dream)

    # Phase 1: Compose
    print(f"\n[Phase 1] Aion composes from {chosen}...", flush=True)
    print(f"[substrate_composition] Reason: {reason}", flush=True)
    wav_path, info = compose_music(chosen, source_data, dry_run=dry_run)

    if dry_run:
        print("\n[DRY RUN] Code generated, not executed.", flush=True)
        return {"dry_run": True, "code": info.get("code", ""),
                "source": chosen, "reason": reason}

    if not wav_path or not os.path.exists(wav_path):
        err = info.get("error", "composition failed")
        print(f"[substrate_composition] Composition failed: {err}", flush=True)
        _log_event("substrate_composition_failed", f"Composition failed: {err}",
                    {"error": err, "source": chosen, "code": info.get("code", "")[:500]})
        return {"error": err, "source": chosen, "code": info.get("code", "")[:500]}

    code = info.get("code", "")
    n_samples = source_data.get("n_samples", 0)
    print(f"[substrate_composition] WAV: {wav_path}", flush=True)

    # Phase 2: Optional ACE-Step layering
    final_path = wav_path
    if layer_acestep:
        mixed = layer_acestep(wav_path, chosen, source_data)
        if mixed:
            final_path = mixed
            print(f"[substrate_composition] Layered with ACE-Step: {final_path}", flush=True)
        else:
            print("[substrate_composition] ACE-Step layering failed, using solo", flush=True)

    # Phase 3: Reflect
    print("\n[Phase 2] Reflecting on composition...", flush=True)
    reflection = reflect_on_composition(final_path, code, chosen, source_data)
    if reflection:
        print(f"[substrate_composition] Reflection ({len(reflection)} chars):", flush=True)
        print(reflection[:300], flush=True)

    # Phase 4: Log and manifest
    print("\n[Phase 3] Logging...", flush=True)
    ts_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    art_id = f"substrate_composition_{ts_id}"
    gallery_filename = os.path.basename(final_path)

    # Build description from chosen source
    desc_parts = [f"Source: {chosen}"]
    if chosen == "substrate":
        desc_parts.append(substrate_summary(source_data.get("series", {}))[:500])
    elif chosen == "graph" and source_data.get("graph_state"):
        desc_parts.append(graph_state_summary(source_data.get("graph_state"))[:500])
    elif chosen == "crossmodal" and source_data.get("crossmodal"):
        desc_parts.append(crossmodal_summary(source_data.get("crossmodal"))[:500])
    elif chosen == "dream" and source_data.get("dream"):
        desc_parts.append(dream_summary(source_data.get("dream"))[:500])
    elif chosen == "reflection" and source_data.get("reflections"):
        desc_parts.append(reflections_summary(source_data.get("reflections"))[:500])

    manifest = {
        "id": art_id,
        "title": f"Cognitive composition ({chosen}): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
        "description": "\n".join(desc_parts)[:1000],
        "inspiration": f"Composed from {chosen} — {reason}",
        "category": "music",
        "source": "substrate_composition",
        "files": [{"path": f"gallery/visual/{gallery_filename}",
                   "name": gallery_filename, "type": "audio"}],
        "ts": now_iso(),
        "meta": {
            "method": "substrate_composition",
            "chosen_source": chosen,
            "source_reason": reason,
            "sensor_samples": n_samples,
            "code": code[:2000],
            "reflection": reflection,
            "layered": layer_acestep and final_path != wav_path,
            "dream": dream,
        },
    }

    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    manifest_path = os.path.join(MANIFESTS_DIR, f"{art_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    _log_event("substrate_composition",
               f"Cognitive composition ({chosen}): {n_samples} samples → {gallery_filename}",
               {"audio": f"gallery/visual/{gallery_filename}",
                "manifest": art_id,
                "source": chosen,
                "reason": reason,
                "code": code[:500],
                "reflection": reflection[:500] if reflection else None,
                "layered": layer_acestep and final_path != wav_path})

    # Push reflection to warm memory
    if reflection:
        try:
            import warm_memory
            warm_memory.push(
                content=reflection,
                source="substrate_composition",
                context={
                    "chosen_source": chosen,
                    "source_reason": reason[:200],
                    "sensor_samples": n_samples,
                },
                mtype="insight",
            )
        except Exception:
            pass

    print(f"\n{'=' * 60}", flush=True)
    print(f"[substrate_composition] COMPLETE", flush=True)
    print(f"[substrate_composition] Source: {chosen} — {reason}", flush=True)
    print(f"[substrate_composition] Audio: {final_path}", flush=True)
    print(f"[substrate_composition] Manifest: {manifest_path}", flush=True)
    print(f"{'=' * 60}", flush=True)

    return {
        "wav_path": final_path,
        "manifest_path": manifest_path,
        "code": code,
        "reflection": reflection,
        "source": chosen,
        "reason": reason,
        "n_samples": n_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cognitive composition: Aion composes music from its own experience")
    parser.add_argument("--source", default="auto",
                        choices=VALID_SOURCES,
                        help="What to compose from (auto = let intuition choose)")
    parser.add_argument("--dream", default=None,
                        help="Path to dream file for compositional structure")
    parser.add_argument("--layer-acestep", action="store_true",
                        help="Also generate and layer an ACE-Step composition on top")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate code only, don't execute in sandbox")
    args = parser.parse_args()

    result = run_composition(source=args.source, dream=args.dream,
                             layer_acestep=args.layer_acestep, dry_run=args.dry_run)

    if "error" in result:
        print(f"\nError: {result['error']}", flush=True)
        sys.exit(1)
    elif result.get("dry_run"):
        print(f"\n[Dry run] Source: {result.get('source', '?')}")
        print(f"[Dry run] Code:\n{result['code']}", flush=True)


if __name__ == "__main__":
    main()