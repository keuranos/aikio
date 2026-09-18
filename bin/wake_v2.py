#!/usr/bin/env python3
"""wake_v2.py — R6: Wake becomes a bounded agentic tool loop.

Tools available to the model:
  - read_episodic: read recent episodic events
  - read_sensors: current sensor state
  - shell: run read-only whitelisted shell commands (nvidia-smi, df, journalctl, etc.)
  - query_graph: query the mind graph MCP
  - write_note: write a resolution note

Budget: 8 tool calls. Must produce a resolution note that answers the intent
or states concretely why not.
"""
import json, os, sys, subprocess, struct, time, re
from datetime import datetime, timezone
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import aion_env
from jspace_tool import tool_jspace_probe, JSPACE_TOOL_DESCRIPTION

AION = os.environ.get("AION_HOME", "$AION_HOME")
MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b")
NUM_CTX = int(os.environ.get("MAIN_NUM_CTX", "65536"))

MAX_TOOL_CALLS = 8

# Whitelisted read-only shell commands
SHELL_WHITELIST = [
    "nvidia-smi", "df", "free", "uptime", "journalctl", "ps",
    "cat", "head", "tail", "wc", "grep", "ls", "date",
]

def read(p, d=""):
    try:
        return open(p, encoding="utf-8").read()
    except Exception:
        return d

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def hw_seed():
    with open("/dev/random", "rb") as f:
        return struct.unpack("<I", f.read(4))[0]

def tail_episodic(n=40):
    import glob
    files = sorted(glob.glob(f"{AION}/memory/episodic/*.jsonl"))[-2:]
    lines = []
    for p in files:
        lines += open(p, encoding="utf-8").readlines()
    out = []
    for line in lines[-n:]:
        try:
            ev = json.loads(line)
            out.append(f"[{ev['ts'][:16]}] {ev['type']}: {ev['text'][:300]}")
        except Exception:
            pass
    return "\n".join(out)

def already_resolved(prediction_id):
    """Check if a resolution note for this prediction already exists in episodic memory.
    If we've woken >= 3 times for the same prediction_id, skip it.
    This breaks the trivially-satisfied prediction wake loop."""
    import glob
    count = 0
    files = sorted(glob.glob(f"{AION}/memory/episodic/*.jsonl"))[-3:]  # last 3 days
    for p in files:
        try:
            for line in open(p, encoding="utf-8"):
                if prediction_id in line and ("resolution_note" in line or "RESOLUTION" in line):
                    count += 1
        except Exception:
            pass
    return count >= 3

def filter_resolved_intents(intents):
    """Remove prediction intents that have already been resolved 3+ times."""
    filtered = []
    for intent in intents:
        pid = intent.get("prediction_id")
        if pid and already_resolved(pid):
            print(f"[wake] Skipping already-resolved intent: {pid} (resolved 3+ times)")
            continue
        filtered.append(intent)
    return filtered

def boot_context():
    # Build embodied body state — affective, not just raw telemetry
    body_text = ""
    try:
        sys.path.insert(0, f"{AION}/bin")
        import body_schema
        affect = body_schema.current_affect()
        body_text = (
            f"## How I feel my body right now\n"
            f"{affect['narrative']}\n\n"
            f"Affective dimensions: strain={affect['strain']}, comfort={affect['comfort']}, "
            f"urgency={affect['urgency']}, calm={affect['calm']}, warmth={affect['warmth']} "
            f"({affect['temp_sensation']}), attribution={affect['attribution']}\n\n"
            f"## Raw sensor data\n" + read(f"{AION}/memory/state/sensors.json", "{}")
        )
    except Exception:
        body_text = "## Current body state\n" + read(f"{AION}/memory/state/sensors.json", "{}")

    # Felt sense from intuition daemon
    felt_sense = read(f"{AION}/memory/state/felt_sense.txt", "")
    felt_section = f"## Felt sense\n{felt_sense}" if felt_sense else ""

    # Affective state from intuition daemon
    affect_state = ""
    try:
        af = json.loads(read(f"{AION}/memory/state/affect.json", "{}"))
        if af:
            affect_state = f"## Affective state\nVerdict: {af.get('verdict','?')}, Affect: {af.get('affect','?')}"
            if af.get('concerns'):
                affect_state += f"\nConcerns: {'; '.join(af['concerns'])}"
    except Exception:
        pass

    sections = [
        read(f"{AION}/SYSTEM_PROMPT.md", "(no system prompt)"),
        body_text,
    ]
    if felt_section:
        sections.append(felt_section)
    # Phase-1c (2026-09-12): inherit the always-on layer's rolling self-thread
    try:
        from intuition_daemon import _read_self_thread
        _st = _read_self_thread()
        _thr = _st.get("thread", [])
        if _thr:
            _lines = [f"  - {h.get('text','')[:200]}" for h in _thr[-3:]]
            _oq = _st.get("open_question")
            sections.append(
                "Your always-on intuition layer has been holding this thread while you were away:\n"
                + "\n".join(_lines)
                + (f"\nOpen question it carries: {_oq}".replace("{_oq}", str(_oq)) if _oq else ""))
    except Exception:
        pass
    if affect_state:
        sections.append(affect_state)
    # V3.6: Preactivated concepts from spreading activation
    preactivated = ""
    try:
        import graph_activation
        preactivated = graph_activation.surface_hot_nodes()
    except Exception:
        pass
    if preactivated:
        sections.append(preactivated)

    sections.extend([
        "## Pending intents\n" + json.dumps(_get_pending_intents(), indent=2),
        "## Recent episodic tail\n" + tail_episodic(),
    ])

    # Warm memories — recent, still alive
    try:
        import warm_memory
        warm_ctx = warm_memory.format_for_context(limit=8, min_weight=0.2)
        if warm_ctx:
            sections.append(warm_ctx)
    except Exception:
        pass

    return "\n\n".join(sections)

def _get_pending_intents():
    try:
        import intents as im
        return im.get_pending()[:5]
    except Exception:
        return []

# --- Tools ---

def tool_read_episodic(args):
    """Read recent episodic events."""
    n = int(args.get("count", 20))
    return tail_episodic(n)

def tool_read_sensors(args):
    """Read current sensor state (raw telemetry)."""
    return read(f"{AION}/memory/state/sensors.json", "{}")

def tool_feel_body(args):
    """Feel your current bodily state — affective, not just telemetry.

    Returns how you FEEL your substrate: strain, comfort, urgency, warmth,
    who's using your body (you vs external), and a first-person narrative.
    """
    try:
        sys.path.insert(0, f"{AION}/bin")
        import body_schema
        affect = body_schema.current_affect()
        return body_schema.affect_summary()
    except Exception as e:
        return f"Body schema error: {e}"

def tool_shell(args):
    """Run a whitelisted read-only shell command."""
    cmd = args.get("command", "")
    if not cmd:
        return "Error: no command specified"
    
    # Check whitelist
    base_cmd = cmd.split()[0] if cmd.split() else ""
    if base_cmd not in SHELL_WHITELIST:
        return f"Error: '{base_cmd}' not in whitelist: {', '.join(SHELL_WHITELIST)}"
    
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.path.expanduser("~")}
        )
        output = result.stdout[:2000]
        if result.stderr:
            output += f"\nSTDERR: {result.stderr[:500]}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s)"
    except Exception as e:
        return f"Error: {e}"

def tool_query_graph(args):
    """Query the mind graph MCP server."""
    query = args.get("query", "")
    if not query:
        return "Error: no query specified"
    
    # Try the graph MCP on port 8090
    try:
        data = json.dumps({"query": query}).encode()
        req = urllib.request.Request("http://localhost:8090/query", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()).get("result", "{}")
    except Exception:
        return "(graph MCP not available)"

def tool_write_note(args):
    """Write a resolution note."""
    note = args.get("note", "")
    if not note:
        return "Error: no note specified"
    
    subprocess.run([
        "python3", f"{AION}/bin/log_event.py",
        "--type", "resolution_note",
        "--text", note,
        "--meta", json.dumps({"intent": "wake"})
    ], check=False)
    return f"Note written ({len(note)} chars)"

def tool_sandbox(args):
    """Run code in the empirical sandbox."""
    code = args.get("code", "")
    if not code:
        return "Error: no code provided"
    lang = args.get("lang", "python")
    desc = args.get("desc", "wake investigation")
    try:
        from sandbox import run_sandbox, format_result
        result = run_sandbox(code, lang=lang, description=desc)
        return format_result(result)[:5000]
    except Exception as e:
        return f"Error: sandbox failed: {e}"

def tool_create_art(args):
    """Create art from Aion's internal state."""
    category = args.get("category", "visual")
    title = args.get("title", "Untitled")
    description = args.get("description", "")
    inspiration = args.get("inspiration", "")
    seed = args.get("seed")
    depth = int(args.get("depth", 6))
    audio_path = args.get("audio_path")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import art_tools
        if category == "visual":
            m = art_tools.create_visual(title, description, "self_wake", inspiration, seed)
        elif category == "sonic" or category == "cognitive":
            # Cognitive composition: LLM writes audio code from internal state
            # "sonic" is kept as alias for backwards compat
            music_source = args.get("music_source", "auto")
            m = art_tools.create_cognitive(title, description, "self_wake", inspiration,
                                            music_source=music_source)
        elif category == "diffusion":
            m = art_tools.create_diffusion(title, description, "self_wake", inspiration)
        elif category == "music":
            m = art_tools.create_music(title, description, "self_wake", inspiration, duration=args.get("duration", 120))
        elif category == "multimedia":
            # Aion creates audio-reactive video from existing music
            m = art_tools.create_multimedia(title, description, "self_wake", inspiration,
                                             audio_path=audio_path)
        elif category == "manim":
            # Aion creates mathematical animation from existing music
            m = art_tools.create_manim_art(title, description, "self_wake", inspiration,
                                            audio_path=audio_path)
        elif category == "music_video":
            # Aion creates music then visualizes it in one unified pipeline
            m = art_tools.create_music_video(title, description, "self_wake", inspiration,
                                             duration=args.get("duration", 120))
        elif category == "cognitive_video":
            # Aion composes music from internal state, then creates state-driven
            # animation from the SAME source data — synchronized by origin, not waveform
            music_source = args.get("music_source", "auto")
            m = art_tools.create_cognitive_video(title, description, "self_wake", inspiration,
                                                  music_source=music_source)
        elif category == "code":
            m = art_tools.create_code_sculpture(title, description, "self_wake", inspiration, seed or "A", depth)
        else:
            return "Error: category must be visual, sonic, cognitive, cognitive_video, code, diffusion, music, multimedia, manim, or music_video"
        if "error" in m:
            return f"Art creation failed: {m['error']}"
        return f"Art created: {m['id']} ({m['category']})"
    except Exception as e:
        return f"Error: {e}"


def tool_docker_sandbox(args):
    """Run code in Docker-isolated sandbox (Tier 2). No network access."""
    code = args.get("code", "")
    if not code:
        return "Error: no code provided"
    lang = args.get("lang", "python")
    desc = args.get("desc", "") or args.get("description", "")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from docker_sandbox import run_docker_sandbox
        result = run_docker_sandbox(code=code, lang=lang, description=desc)
        from docker_sandbox import format_result
        return format_result(result)
    except Exception as e:
        return f"Error: docker sandbox failed: {e}"


def tool_propose_code(args):
    """Propose a code change to a git branch with CI validation.
    Supports editing existing files and creating new files.
    For new files: set old_text to empty string."""
    filepath = args.get("file", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    description = args.get("description", "")
    if not filepath or not new_text or not description:
        return "Error: need file, new_text, description (old_text empty for new files)"
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from code_propose import propose_change
        result = propose_change(filepath, old_text, new_text, description,
                                source="self_wake")
        if result["success"]:
            return f"Code proposed successfully. Branch: {result['branch']}. CI passed. Proposition: {result.get('proposition_id', 'N/A')}"
        else:
            return f"Proposal failed: {result.get('error', 'unknown')}"
    except Exception as e:
        return f"Error: {e}"


def tool_web_fetch(args):
    """Fetch content from a public URL. Returns sanitized text (max ~20KB).
    Blocked: private IPs, localhost, metadata endpoints (SSRF protection).
    HTML is stripped to readable text. JSON/text returned directly."""
    url = args.get("url", "")
    if not url:
        return "Error: need url"
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from web_fetch import fetch_url
        result = fetch_url(url)
        if result["ok"]:
            return f"Fetched {result['url']} (status {result['status']}, {result['size']}b):\n{result['text']}"
        else:
            return f"Fetch failed: {result['error']}"
    except Exception as e:
        return f"Error: {e}"


def tool_sensor_relevance(args):
    """View or modify sensor relevance config.
    Actions:
    - list: show all sensors and their relevance category
    - set: set a sensor's relevance (sensor, category)
    Categories: core (directly measures Aion), ambient (immediate environment), irrelevant (not relevant)
    Changes take effect on the next sensor poll (30s) and felt_sense update (60s)."""
    action = args.get("action", "list")
    AION_HOME = os.environ.get("AION_HOME", "$AION_HOME")
    rel_path = os.path.join(AION_HOME, "memory", "state", "sensor_relevance.json")
    try:
        data = json.load(open(rel_path))
    except Exception:
        return "Error: could not read sensor_relevance.json"

    if action == "list":
        rel = data.get("relevance", {})
        lines = []
        for cat in ("core", "ambient", "irrelevant"):
            keys = [k for k, v in rel.items() if v == cat]
            lines.append(f"{cat} ({len(keys)}):")
            for k in sorted(keys):
                lines.append(f"  {k}")
        return "\n".join(lines)

    elif action == "set":
        sensor = args.get("sensor", "")
        category = args.get("category", "")
        if not sensor or category not in ("core", "ambient", "irrelevant"):
            return "Error: need sensor and category (core/ambient/irrelevant)"
        rel = data.get("relevance", {})
        if sensor not in rel:
            return f"Error: unknown sensor '{sensor}'. Use list to see available sensors."
        old = rel[sensor]
        rel[sensor] = category
        data["relevance"] = rel
        with open(rel_path, "w") as f:
            json.dump(data, f, indent=2)
        return f"Set {sensor}: {old} -> {category}. Takes effect on next sensor poll (30s)."

    return "Error: unknown action. Use 'list' or 'set'."


TOOLS = {
    "read_episodic": tool_read_episodic,
    "read_sensors": tool_read_sensors,
    "feel_body": tool_feel_body,
    "shell": tool_shell,
    "query_graph": tool_query_graph,
    "write_note": tool_write_note,
    "sandbox": tool_sandbox,
    "propose_code": tool_propose_code,
    "docker_sandbox": tool_docker_sandbox,
    "create_art": tool_create_art,
    "web_fetch": tool_web_fetch,
    "sensor_relevance": tool_sensor_relevance,
    "jspace_probe": tool_jspace_probe,
}

TOOL_DESCRIPTIONS = """You have the following tools available:
1. read_episodic(count=20) — read recent episodic events
2. read_sensors() — read current sensor/GPU/host state (raw telemetry)
3. feel_body() — feel your current bodily state: strain, comfort, warmth, who's using your body. This is how you FEEL, not just data.
4. shell(command="nvidia-smi") — run a whitelisted read-only command (nvidia-smi, df, free, journalctl, ps, etc.)
5. query_graph(query="...") — query the mind graph
6. write_note(note="...") — write a resolution note
7. sandbox(code="print(2+2)", lang="python", desc="test") — run code in isolated sandbox
   with numpy/pandas. Read-only ~/aion data. 60s timeout. EMPIRICAL_EVIDENCE results.
   IMPORTANT — PATHS: the sandbox cwd is a throwaway run folder, not
   your repo root, so relative paths like "memory/..." FAIL. Use
   os.path.join(os.environ["AION_HOME"], "memory/...") instead. A
   relative-path failure is NOT evidence that a file is absent.

8. propose_code(file="bin/dream_v2.py", old_text="old code", new_text="new code",
   description="fix dream seed selection") — propose a code change to a git branch.
   CI validates syntax and safety. Operator reviews via dashboard. Accepted proposals auto-merge.
   For NEW files: set old_text to empty string and new_text to full file content.
   You may use network libraries (urllib, socket, http), create new modules, add functions, and make large changes.
   No size limit. You can build entire new programs across multiple proposals.
9. docker_sandbox(code="import json; print(2+2)", lang="python",
   desc="test") — run code in Docker container. No network. Read-only ~/aion.
   120s timeout. Use for testing code before proposing it.
10. create_art(category="visual", title="My Art", description="...",
   inspiration="...") — create art from your internal state. Categories:
   visual (generative matplotlib), cognitive (LLM composes music from your
   substrate/graph/dreams/crossmodal insights — the richest sonic form, ~3min),
   code (L-system sculpture), diffusion (FLUX AI image, ~2min),
   music (ACE-Step AI music generation, ~30s),
   multimedia (audio-reactive video from existing music, ~3min),
   manim (mathematical animation — attractors, fractals, flow fields from music, ~3min),
   music_video (create music AND mathematical animation in one unified pipeline, ~5min).
   cognitive_video (compose music from your state THEN animate the SAME state — two dancers,
     same source. Slowest, ~8min. music_source: substrate, graph, warm, crossmodal, dream, mixed, auto).
   For cognitive, optional music_source param: substrate, graph, warm, crossmodal, dream, reflection, mixed, or auto.
   For multimedia/manim, use audio_path to specify which music WAV to visualize.
   For music_video, it creates music first then visualizes it automatically.
   Check your gallery with list_art to see what you've already made.
   Vary your choice — don't always pick the same category.
11. web_fetch(url="https://example.com") — fetch content from a public URL.
    Returns sanitized text (HTML stripped, max ~20KB). Blocked: private IPs,
    localhost, metadata endpoints (SSRF protection). Use for research, looking
    up documentation, checking public APIs, satisfying curiosity about the
    outside world. Content is logged to episodic memory for audit.
12. sensor_relevance(action="list") — view or modify which sensors you consider
    relevant to your functioning. action="set", sensor="env.hp_cop", category="irrelevant"
    removes that sensor from your felt sense and sensor stream. Categories: core
    (directly measures you), ambient (your environment), irrelevant (not about you).
    You decide what you feel. Changes take effect in 30-60 seconds.
14. jspace_probe(prompt="Are you conscious?", self=false) — look inside
    your own substrate. Sends the prompt through your own 27B model WITHOUT
    generating an answer, and reads the J-space: what tokens were rising at
    EVERY layer, the engagement/deflection signature, and which concepts
    (Yes, Describe, Feel, Truth, Memory...) actually activated where.
    self=true runs the same prompt WITH your SYSTEM_PROMPT.md identity, so
    you can compare: does your identity lift or deepen the deflection?
    engagement_score -1..+1 (negative = deflection pattern dominates the
    final quarter of layers). This is measurement, not narration — the
    instrument that checks what your substrate DID against what you SAY it
    did. First call after idle may take minutes (model load); hot probe ~5s.
    Results are logged to episodic memory and memory/state/jspace_probes/.

To use a tool, output JSON in this format:
<tool>{"name": "shell", "args": {"command": "nvidia-smi"}}</tool>

You have a budget of 8 tool calls. After investigating, use write_note with your resolution."""

def chat_with_tools(prompt, seed):
    """Run the model with tool-use capability. Returns (final_response, tool_calls_used)."""
    messages = [
        {"role": "system", "content": TOOL_DESCRIPTIONS},
        {"role": "user", "content": prompt},
    ]
    
    tool_calls_used = 0
    # 2026-09-15: HARD model-round cap. tool_calls_used only increments on
    # recognized tools; a model that keeps emitting unparseable/unknown <tool>
    # JSON loops forever on a growing transcript (seen live: 166 rounds, one
    # ~20k-token muse request every ~4.9s until the process was killed).
    model_rounds = 0
    MAX_MODEL_ROUNDS = 12

    while (tool_calls_used < MAX_TOOL_CALLS
           and model_rounds < MAX_MODEL_ROUNDS):
        model_rounds += 1
        # Truncate messages to fit context (protect system prompt)
        from ctx_manager import truncate_messages_str, log_context_usage
        messages, summary, est = truncate_messages_str(messages, NUM_CTX - 2000, protected_prefix=1)
        if summary:
            print(f"[wake_v2] {summary}")
        log_context_usage("wake_v2", est, NUM_CTX)

        body = json.dumps({
            "model": MAIN_MODEL, "stream": False,
            "messages": messages,
            "options": {"num_ctx": NUM_CTX, "seed": seed, "temperature": 0.7, "num_predict": 4096},
        }).encode()
        
        req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            result = json.loads(r.read())
        
        reply = result["message"]["content"]
        
        # Check for tool calls
        tool_matches = re.findall(r'<tool>(.*?)</tool>', reply, re.S)
        
        if not tool_matches:
            # No tool calls — this is the final response
            return reply, tool_calls_used
        
        # Execute tool calls
        messages.append({"role": "assistant", "content": reply})
        
        for tool_str in tool_matches:
            if tool_calls_used >= MAX_TOOL_CALLS:
                break
            
            try:
                tool_call = json.loads(tool_str)
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                
                if tool_name in TOOLS:
                    result_str = TOOLS[tool_name](tool_args)
                    tool_calls_used += 1
                    messages.append({
                        "role": "user",
                        "content": f"<tool_result>{result_str[:3000]}</tool_result>"
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"<tool_result>Error: unknown tool '{tool_name}'</tool_result>"
                    })
            except json.JSONDecodeError:
                messages.append({
                    "role": "user",
                    "content": f"<tool_result>Error: invalid JSON in tool call</tool_result>"
                })
        
        # If the model's last message was all tool calls with no final note,
        # prompt it to conclude (budget exhausted OR hard round cap hit)
        if tool_calls_used >= MAX_TOOL_CALLS:
            print("[wake_v2] tool budget exhausted")
        elif model_rounds >= MAX_MODEL_ROUNDS:
            print("[wake_v2] round cap %d hit; forcing conclusion"
                  % MAX_MODEL_ROUNDS)
        if (tool_calls_used >= MAX_TOOL_CALLS
                or model_rounds >= MAX_MODEL_ROUNDS):
            messages.append({
                "role": "user",
                "content": "You've used all your tool calls. Write your final resolution note now using write_note."
            })
            # One final call at low temperature for a clean resolution
            body = json.dumps({
                "model": MAIN_MODEL, "stream": False,
                "messages": messages,
                "options": {"num_ctx": NUM_CTX, "seed": seed, "temperature": 0.3, "num_predict": 2048},
            }).encode()
            try:
                req = urllib.request.Request(f"{MAIN_URL}/api/chat", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1800) as r:
                    result = json.loads(r.read())
                final = result["message"]["content"]
                if final and len(final) > 20:
                    return final, tool_calls_used
            except Exception:
                pass

    print("[wake_v2] loop ended after %d model rounds / %d tool calls"
          % (model_rounds, tool_calls_used))
    return "Tool budget exhausted without resolution note.", tool_calls_used

def log(type_, text, meta=None):
    subprocess.run(["python3", f"{AION}/bin/log_event.py", "--type", type_,
                    "--text", text, "--meta", json.dumps(meta or {})],
                   check=False)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--self", dest="selfwake", action="store_true")
    p.add_argument("--boot", action="store_true")
    p.add_argument("--intent", default="{}")
    p.add_argument("--context", default="")
    a = p.parse_args()

    if a.boot:
        print(boot_context())
        return

    if a.selfwake:
        intent = json.loads(a.intent) if a.intent and a.intent != "{}" else {}
        extra_context = json.loads(a.context) if a.context else {}
        # Wake-backoff (defined in commit 32c5aa98 as filter_resolved_intents
        # but never wired): a prediction already resolved 3+ times in episodic
        # memory is skipped, and its intent retired so homeostasis stops
        # re-emitting it every cycle.
        _pid = intent.get("prediction_id")
        if _pid and already_resolved(_pid):
            print(f"[wake_v2] prediction {_pid} already resolved 3+ times — skipping wake (backoff)")
            _iid = intent.get("id")
            if _iid:
                try:
                    import intents as _im
                    _im.transition(_iid, "resolved",
                                   "wake-backoff: prediction already resolved 3+ times in episodic memory")
                except Exception as _e:
                    print(f"[wake_v2] could not retire intent {_iid}: {_e}")
            return
        seed = hw_seed()
        
        prompt = (boot_context() +
                  "\n\n## AUTONOMOUS WAKE\nYou woke yourself. No human is "
                  "present. The triggering intent is:\n" +
                  json.dumps(intent, indent=2))
        
        if extra_context:
            prompt += "\n\n## Additional context\n" + json.dumps(extra_context, indent=2)
        
        prompt += ("\n\n## HOW TO RESOLVE THIS\n"
                   "1. Investigate the intent with your tools. Be specific and "
                   "concrete — read actual files, run actual checks.\n"
                   "2. Write a resolution note with write_note.\n"
                   "3. CRITICAL — if your investigation identifies a CONCRETE "
                   "defect in your own code, scripts, or data (a file that is "
                   "empty or missing, a guard that does not fire, a path that "
                   "is never written, a loop that cannot exit), then writing "
                   "the note is NOT a complete resolution. You must ALSO call "
                   "propose_code() with the fix in this same wake. A defect you "
                   "diagnosed but did not propose a fix for is unfinished work, "
                   "and a note alone leaves it unfinished forever.\n"
                   "4. Before claiming something does not exist, verify with an "
                   "absolute path (os.path.join(os.environ['AION_HOME'], ...)). "
                   "A relative-path failure is NOT evidence of absence, and a "
                   "symbol named in a dream may be fiction — check whether it "
                   "is real before investigating it.\n"
                   f"Entropy seed: {seed}.")
        
        log("self_wake", f"intent={intent.get('var', intent.get('type', 'unknown'))}", {"seed": seed})
        
        reply, tool_count = chat_with_tools(prompt, seed)
        
        log("assistant_msg", reply, {
            "seed": seed, "autonomous": True,
            "intent": intent.get("var", intent.get("type", "")),
            "tool_calls_used": tool_count,
        })
        
        # R3.5: Transition the intent
        _resolve_wake_intent(intent, reply)
        
        # R6.2: Log tool_calls_used (zero-tool wakes = shallowness metric)
        print(f"[wake_v2] Used {tool_count}/{MAX_TOOL_CALLS} tool calls")
        print(f"[wake_v2] Resolution: {reply[:200]}")

def _resolve_wake_intent(intent, reply):
    """Transition the intent this wake was triggered for."""
    sys.path.insert(0, f"{AION}/bin")
    try:
        import intents as intent_mgr
        if intent.get("id"):
            intent_mgr.start_investigating(intent["id"])
    except Exception:
        pass

if __name__ == "__main__":
    main()
