# The Dream–Artifact Pipeline and Supporting Systems

This framework core ships the *cognitive loop* (see README). The papers also
describe systems that turn the loop's outputs into artifacts — dream imagery,
animation, music, and host-fit telemetry. Those modules live in the
deployment layer and are summarized here; where an upstream is public, it is
linked.

## The pipeline (as described in Paper 1, §Memory and §Art)

```
dream cycle (dream_v2)
  ├── graph-walk dreams      → insight types feed construction backlog
  ├── simulation dreams      → eng insight types (FAILURE_MODE, GAP, ...)
  └── synthesis
        │
        ▼
dream_artifact  — the intuition model CHOOSES the medium per dream:
  ├── FLUX            (diffusion image; atmospheric/emotional dreams)
  ├── generative      (LLM-written matplotlib; structural dreams)
  ├── graph           (mind-graph subgraph render; connectivity dreams)
  ├── hybrid          (FLUX + graph overlay)
  └── manim           (animation; selected for process/sequence dreams)
        │
        ▼
art-learning loop — artifacts are scored for alignment with the source
dream (mean alignment 0.5–0.7 in Paper 1); scores feed back into the
prompt-generation heuristics.

cross_modal_synthesis — a daemon on the intuition model that periodically
renders visual representations of cognitive state (not tied to a dream).

substrate_composition — GPU telemetry (power/temperature/utilization) is
rendered as audible structure via AceStep; included in this repo
(`bin/substrate_composition.py`).
```

## Upstream components and links

| System | What it does | Status / link |
|---|---|---|
| `fitctl` | Rust "host capability contract" tool (`survey`/`contract`/`classify`); the sensor stream polls it for a physical FIT verdict that enters the agent's self-model as a first-class fact | **public**: [github.com/tznurmin/fitctl](https://github.com/tznurmin/fitctl) · crates.io `fitctl` (v0.8+) |
| FLUX | local diffusion image server (`flux_server.py`, ~200 lines wrapping a FLUX checkpoint); dream_artifact's richest medium | deployment layer; standard FLUX weights required |
| AceStep | local music-generation model; `acestep_proxy.py` wraps it for dream sonification and substrate compositions | deployment layer |
| matplotlib/manim paths | LLM-authored rendering code with a fix-retry loop (render errors return the traceback to the model) | in `dream_artifact.py`/`dream_matplotlib.py`, deployment layer |
| Mind-graph renders | graphify-derived graph renders (networkx) | graph rebuild scripts in this repo (`bin/graph_rebuild.sh`) |

## The artifact modules in this repo

As of this revision the artifact layer ships in `artifacts/` next to the core:

| File | Role |
|---|---|
| `dream_artifact.py` | dream → artifact transformer; the intuition model selects the medium (FLUX / generative matplotlib / graph render / hybrid / manim) per dream, with an LLM fix-retry loop for render errors |
| `dream_matplotlib.py` | the LLM-authored matplotlib rendering path |
| `flux_server.py` | local FLUX diffusion image server (wraps a FLUX checkpoint) |
| `flux_prompt_v2.py` | FLUX prompt generation from dream content |
| `acestep_proxy.py` | AceStep music-generation proxy (dream sonification, substrate compositions) |
| `cross_modal_synthesis.py` | periodic cognitive-state visualizations on the intuition model |
| `art_tools.py`, `visual_manifestation.py`, `create_art_session.py` | gallery/art session tooling |

Upstream model requirements (not included): a FLUX diffusion checkpoint
(~12 GB VRAM for inference) and an AceStep installation (`ACESTEP_DIR`
environment variable). Paths are environment-driven — see
`config/aion.env.example` and the module headers.

## Experiment-critical vs. descriptive

To reproduce the papers you need: the loop (this repo), Ollama models per
`config/aion.env.example`, and — for the introspection experiments — the
probe/ablation code and data in
[`keuranos/aion-jspace`](https://github.com/keuranos/aion-jspace). The art
pipeline and embodiment (rover, cameras, Home Assistant sensors) are
descriptive context, not experiment dependencies.
