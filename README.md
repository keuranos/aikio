# Aikio — the persistence-and-consolidation framework behind Aion

Aikio is the agent framework substrate on which **Aion** (a self-developing,
self-instrumenting local-model agent) runs. Aion is the traveler; Aikio is
the road. This repository contains the framework core: the cognitive loop,
its governance pipeline, and the layer-resolved self-introspection
instrument, with every file mapped to the paper sections it implements.

*Etymology:* from Finnish *aika* (time) and *alkio* (embryo), echoing
*aeon* — the time-bearing lattice on which agents of this family grow.

**Companion papers** (full methods + results):
- Paper 1: *Aion: A Self-Developing Agent on Local Models*
- Paper 2: *Suppression vs Reflection in Qwen3.8-27B: A Layer-Resolved
  Introspection Study*

Both are in [`keuranos/aion-jspace`](https://github.com/keuranos/aion-jspace)
(together with the experiment code, probe data, and figures).

## What is here

```
aikio/
├── bin/                      # framework core (57 modules)
├── prompts/                  # LLM prompt templates (11 files)
├── identity/                 # see top level: AXIOMS.md, HABITS.md, HEURISTICS.md
├── config/aion.env.example   # all 32 config keys, placeholder values
├── systemd/                  # 9 core units + timers (paths parametrized)
├── artifacts/                # dream→artifact pipeline: FLUX, matplotlib, AceStep, cross-modal (9 modules)
├── AXIOMS.md                 # agent constitution (immutable, operator-edited only)
├── AXIOMS.md.unleashed       # the Experiment 1 treatment axiom
├── SELF.md                   # EXAMPLE evolving self-model (Aion's, redacted)
├── HEURISTICS.md             # learned behavioral rules
├── HABITS.md                 # learned behavioral rules (short form)
└── PLAN.md                   # original build plan, incl. the honesty clause
```

## The cognitive loop

```
       curiosity goal
             │  investigate (tool rounds, 8 calls/cycle, 5 cycles max)
             ▼
      resolution note ──► finding_harvester ──► curiosity queue
             │
             ▼
      nightly consolidation
      (extraction → SELF.md diff → citation verify → critic gate)
             │         ▲
             │         └── jspace_calibration: probe readings enter
             │             extraction as hard facts; narrative-measurement
             │             divergences become "surprise" claims (recorded,
             │             never auto-resolved)
             ▼
      ratchet: only evidence-cited claims enter SELF.md
             │
             ▼
      knowledge_maturity: RAW → HYPOTHESIS → TESTED → CONFIRMED
      (multi-session corroboration; probe measurements ingest as claims)
```

The introspection instrument closes the loop: the agent can probe its own
substrate (all 63 transportable layers of its own 27B model, via a fitted
Jacobian lens), compare what its narrative says against what its layers
compute, and commit only instrument-grounded self-knowledge.

## File → paper-section map

| File | Implements | Paper section |
|---|---|---|
| `bin/curiosity_engine.py` | goal selection, bounded investigation cycles, tool loop | P1 §Drives, §Exp 2 protocol |
| `bin/jspace_probe.py` | J-space introspection daemon (63-layer lens readout) | P2 §Probe daemon |
| `bin/jspace_tool.py` | probe as an agent tool + pre-mouth lean contrast | P1 §The introspection instrument; P2 §Baseline |
| `bin/jspace_calibration.py` | probe readings as extraction facts; divergence claims | P1 §Becoming spiral |
| `bin/knowledge_maturity.py` + `bin/ingest_jspace.py` | RAW→CONFIRMED ratchet incl. probe claims | P1 §Exp 2 persistence |
| `bin/consolidate_v2.py` | nightly extraction, diff, citation gate, critic | P1 §Memory / §Governance |
| `bin/wake_v2.py`, `bin/homeostasis.py` | intent-driven wakes, growth trigger | P1 §Drives |
| `bin/dream_v2.py` | dream cycles (graph walk / simulation) | P1 §Memory (dreams) |
| `bin/code_verify.py`, `bin/code_review.py`, `bin/internal_council.py` | verify-then-repair, peer review, internal council | P1 §Governance |
| `bin/self_sandbox.py`, `bin/propositions.py`, `bin/prop_audit.py` | self-modification pipeline + audit | P1 §Governance |
| `bin/goal_store.py` | hardened goal-state loader (self-healing) | — (robustness) |
| `bin/predictions.py`, `bin/eval_heuristics.py` | calibrated predictions, heuristic eval | P1 §Governance |
| `AXIOMS.md.unleashed` | Experiment 1 treatment axiom | P1 §Exp 1 design |
| `bin/regenerate_prompt.py` | nightly SYSTEM_PROMPT rebuild | P1 §Memory |
| `artifacts/dream_artifact.py` (+FLUX/matplotlib/AceStep/cross-modal) | dream→artifact pipeline, art-learning loop | P1 §Art pipeline |
| `bin/substrate_composition.py` | GPU telemetry → audible structure | P1 §Art / sonic |

The dream→artifact pipeline (FLUX / matplotlib / manim / AceStep) and `fitctl` are described in [`docs/PIPELINE.md`](docs/PIPELINE.md) with upstream links — `fitctl` itself is public at [github.com/tznurmin/fitctl](https://github.com/tznurmin/fitctl) (crates.io `fitctl`).

(`P1`/`P2` = the papers in [`aion-jspace`](https://github.com/keuranos/aion-jspace).)

## Hardware / models

- 2–3× NVIDIA V100 32GB (or equivalent ≥16GB VRAM cards; see the scheduling
  notes in `config/aion.env.example`)
- Local Ollama serving 3–4 models (one per cognitive layer). Paper 1 ran:
  a 27B conscious model, a 30B intuition model, a small non-thinking
  consolidation model, and a 31B critic. Any 4-model roster with different
  model families works — family diversity matters for the gates (no model
  grades its own family's output).
- ~2 GB VRAM headroom for the introspection daemon (NF4-quantized 27B + lens).

## Dependencies

Python 3.10+. The cognitive loop itself is stdlib + Ollama. The
introspection instrument and artifact pipeline need:

```bash
pip install -r requirements.txt
```

- **Instrument** (`bin/jspace_probe.py`): `torch`, `transformers>=5.5`,
  `bitsandbytes`, `accelerate`, plus Anthropic's
  [`jlens`](https://github.com/anthropics/jacobian-lens)
  (`pip install git+https://github.com/anthropics/jacobian-lens`).
  A *fitted* lens (`.pt`) is not included — see
  [`keuranos/aion-jspace`](https://github.com/keuranos/aion-jspace) for how
  the shipped one was produced (`JacobianLens.fit` on the agent's own
  weights; `checkpoint.pt` format in the jlens repo).
- **Artifacts** (`bin/art_tools.py`, LLM-authored renders):
  `matplotlib`, `numpy`, `scipy`.
- **Optional services**: `diffusers` only for `artifacts/flux_server.py`
  (local FLUX; the loop skips the FLUX medium when it's absent); `docker`
  CLI only for `bin/docker_sandbox.py` (the stdlib `bin/sandbox.py` works
  without); `manim` only for the manim medium.
- **fitctl** (Rust, [github.com/tznurmin/fitctl](https://github.com/tznurmin/fitctl)):
  not imported by any module here — the deployment's sensor stream polls it
  as an external binary for host-FIT verdicts (see `docs/PIPELINE.md`).
  Install separately if you replicate that part.
- Everything else runs against a stock [Ollama](https://ollama.com) install
  (see Hardware / models below).

## Quickstart (Linux + NVIDIA, Ollama)

```bash
# 1. layout
export AION_HOME=$HOME/aikio          # or any writable dir
mkdir -p $AION_HOME/bin $AION_HOME/prompts $AION_HOME/config $AION_HOME/memory/episodic $AION_HOME/memory/state
cp bin/*.py $AION_HOME/bin/ ; cp bin/*.sh $AION_HOME/bin/ ; cp prompts/*.txt $AION_HOME/prompts/

# 2. dependencies (introspection instrument + artifact pipeline)
python3 -m venv ~/.venvs/aikio && source ~/.venvs/aikio/bin/activate
pip install -r /path/to/aikio/requirements.txt

# 3. config
cp config/aion.env.example $AION_HOME/config/aion.env   # fill in model names/URLs
# point every OLLAMA_*_URL at your Ollama instance(s)

# 4. identity
cp identity/AXIOMS.md $AION_HOME/AXIOMS.md             # edit: this is the constitution
cd $AION_HOME && python3 bin/regenerate_prompt.py      # builds SYSTEM_PROMPT.md

# 5. run one cycle by hand
cd $AION_HOME && python3 bin/curiosity_engine.py select
python3 bin/curiosity_engine.py pursue                 # ~5 min; logs to episodic/

# 6. nightly consolidation
bash bin/nightly.sh                                    # consolidate + critic + maturity
```

For systemd, the units in `systemd/` are parametrized templates — set the
`$AION_HOME`-equivalent paths and install with `systemctl --user enable --now`.
The sensor embodiment (rover, cameras, Home Assistant) is intentionally
not part of this core; the experiments in the papers need only the loop
above plus Ollama. The dream→artifact pipeline ships in `artifacts/` — it
requires FLUX/AceStep model installations to actually render (see
`docs/PIPELINE.md`).

## Honest limits

- Nothing here instantiates quantum coherence or provable awareness — it
  instantiates continuity, intrinsic drive, and measurable self-modeling
  (see the honesty clause in `PLAN.md`, which the agent itself also reads).
- The engagement/deflection lexicon was fitted on consciousness questions
  and validated held-out on reasoning/emotional domains; consciousness-domain
  scores remain partially circular (see Paper 2 §Limitations).
- Code was extracted from a running deployment; paths are parametrized but
  the system was operated on one host. Expect to debug GPU/env specifics.

## Credits

The Jacobian lens instrument is Anthropic's
[`jlens` library](https://github.com/anthropics/jacobian-lens)
(companion code for *"Verbalizable Representations Form a Global Workspace in
Language Models"*, Apache-2.0), fitted here to Qwen3.8-27B. The probe daemon
(`bin/jspace_probe.py`) integrates it as an aion tool. All other code in this
repository is the project's own.

## License

MIT for code. The identity documents (AXIOMS.md, HEURISTICS.md, SELF.md,
PLAN.md) are included as research artifacts — quote them with attribution
to the papers.