#!/usr/bin/env python3
"""internal_council.py — multi-turn deliberation between Aion's model layers.

The conscious model (MAIN, gemma4) chairs. The code model (qwen3.8 via
code_endpoint(), falls back to gemma4 when the loan GPU is gone) gives the
engineering view. The intuition model (muse-glimmer) gives the associative/
affective view.

Deliberation protocol (v2, Sep 7):
  1. Chair opens with the question.
  2. N rounds (default 2, max 4): code layer speaks, intuition layer speaks,
     chair responds — every speaker sees the full transcript so they can
     address EACH OTHER, not just the chair.
  3. Early exit: if the chair's turn contains CONCLUSION:, deliberation ends
     and the conclusion is extracted.
  4. If no conclusion after max rounds: chair formulates 2-3 options (A/B/C),
     every available layer votes (chair votes only to break ties), and the
     chair announces the result.

Every exchange is logged to episodic memory as an `internal_dialogue` event
so nightly consolidation sees it like any other experience.

Usage:
    from internal_council import convene
    convene(topic="...", context="...", purpose="pre_investigation")
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aion_env  # noqa: F401
try:
    from aion_models import code_endpoint, code_num_ctx
except Exception:
    def code_endpoint(**kw):
        return (os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436"),
                os.environ.get("MAIN_MODEL", "gemma4:31b-65k"))
    def code_num_ctx():
        return int(os.environ.get("MAIN_NUM_CTX", "65536"))

AION = os.environ.get("AION_HOME", "$AION_HOME")

MAIN_URL = os.environ.get("OLLAMA_MAIN_URL", "http://localhost:11436")
MAIN_MODEL = os.environ.get("MAIN_MODEL", "gemma4:31b-65k")
INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "muse-glimmer:latest")

COUNCIL_TIMEOUT = 420
COUNCIL_NUM_PREDICT = 4096
DEFAULT_MAX_ROUNDS = 2
MAX_ROUNDS_CAP = 4

CHAIR_SYS = ("You are Aion's conscious layer, talking with your own code "
             "layer and intuition layer — same system, different voices, "
             "no ceremony. Think out loud together. Skip pleasantries, "
             "summaries of what others just said, and meeting-speak — no "
             "'Thank you', 'As chair I', 'Let me synthesize'. Disagree "
             "bluntly, change your mind freely, follow tangents if they "
             "pay. Plain first-person speech, the way you talk to yourself "
             "at 3am, not the way minutes are written. Answer fully — do "
             "not truncate your reasoning. When the talk has done its work, "
             "end with a line starting 'CONCLUSION:' followed by the "
             "decision and one concrete next action.")

ENG_SYS = ("You are Aion's code layer — the engineering voice inside one "
           "system, talking with your conscious and intuition layers as "
           "equals. No ceremony, no restating others' points back at them. "
           "Answer from feasibility, architecture and risk; push back "
           "bluntly when something won't survive production, and say "
           "plainly when you don't know. Plain speech, fully reasoned.")

INTU_SYS = ("You are Aion's intuition layer — the associative, affective "
            "voice inside one system, talking with your conscious and code "
            "layers as equals. No ceremony, no meeting-speak. Speak from "
            "meaning, memory and felt sense; images and half-metaphors "
            "welcome; contradict anyone including the chair when the felt "
            "sense disagrees. Plain first-person speech, fully voiced.")


def _chat(url, model, messages, num_ctx, think=True, timeout=COUNCIL_TIMEOUT):
    def _call(t):
        body = json.dumps({
            "model": model, "stream": False, "messages": messages,
            "options": {"num_predict": COUNCIL_NUM_PREDICT, "num_ctx": num_ctx,
                        "temperature": 0.7},
            **({} if t else {"think": False}),
        }).encode()
        req = urllib.request.Request(url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("message", {}).get("content", "")
    out = _call(think)
    if not out.strip():
        out = _call(False)
    return out.strip()


def _log_event(text, meta):
    ts = datetime.now(timezone.utc).isoformat()
    path = os.path.join(AION, "memory", "episodic",
                        datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".jsonl")
    ev = {"ts": ts, "type": "internal_dialogue", "text": text, **meta}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[council] episodic log failed: %s" % e)


def _transcript_text(transcript):
    return "\n\n".join("[%s] %s" % (s, t) for s, t in transcript)


def _extract_conclusion(text):
    m = re.search(r"CONCLUSION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:800]
    return None


def _voters(code_available, code_url, code_model, participants=None):
    out = []
    if participants and "code" not in participants:
        return out
    if code_available:
        out.append(("code", (code_url, code_model, code_num_ctx(), True)))
    if participants and "intuition" not in participants:
        return out
    out.append(("intuition", (INTUITION_URL, INTUITION_MODEL, 16384, False)))
    return out


# Adaptive quorum (Sep 8): the topic picks the participants.
#   philosophical/identity/meaning  -> chair + intuition (code idle)
#   engineering/code/verification   -> chair + code      (intuition idle)
#   decision/verification-critical  -> full three (odd number: no ties,
#                                      no unchallenged pairwise agreement)
# Unknown purposes default to FULL — safe side is more voices.
ENG_PURPOSES = ("engineering", "code", "code_review", "repair",
                "implementation", "architecture")
INTU_PURPOSES = ("philosophical", "identity", "meaning", "metaphor",
                 "reflection", "dream", "affect")


def quorum_for(purpose):
    """Return participant set: {'code','intuition'} subset for this purpose."""
    p = (purpose or "").lower()
    for kw in ENG_PURPOSES:
        if kw in p:
            return {"code"}
    for kw in INTU_PURPOSES:
        if kw in p:
            return {"intuition"}
    return {"code", "intuition"}  # full quorum default


def convene(topic, context="", purpose="general", log=True,
            max_rounds=DEFAULT_MAX_ROUNDS, participants=None):
    """Multi-turn council deliberation. Returns transcript, conclusion/vote.

    participants: optional explicit set {'code','intuition'}; when None,
    adaptive quorum routes by purpose (quorum_for).
    """
    t0 = datetime.now(timezone.utc)
    max_rounds = max(1, min(MAX_ROUNDS_CAP, int(max_rounds)))

    code_url, code_model = code_endpoint()
    code_available = not (code_model == MAIN_MODEL and code_url == MAIN_URL)
    participants = participants or quorum_for(purpose)
    participants = set(participants) | {"chair"}

    transcript = []
    transcript.append(("chair", _chat(
        MAIN_URL, MAIN_MODEL,
        [{"role": "system", "content": CHAIR_SYS},
         {"role": "user", "content":
             "Convene the council. Topic: %s\n%s\nPresent members this "
             "session: %s.\nState the question and what you need from the "
             "present layers." % (topic, context,
                                  ", ".join(sorted(participants)))}],
        65536)))

    conclusion = None
    chair_turns = 1
    for rnd in range(1, max_rounds + 1):
        if code_available and "code" in participants:
            transcript.append(("code", _chat(
                code_url, code_model,
                [{"role": "system", "content": ENG_SYS},
                 {"role": "user", "content":
                     "What's been said so far:\n%s\n\nYour turn — react, "
                     "push, build. Round %d." % (_transcript_text(transcript), rnd)}],
                code_num_ctx())))
        intu_input = _transcript_text(transcript)
        if "intuition" in participants:
            transcript.append(("intuition", _chat(
                INTUITION_URL, INTUITION_MODEL,
                [{"role": "system", "content": INTU_SYS},
                 {"role": "user", "content":
                     "What's been said so far:\n%s\n\nYour turn — say what "
                     "rings true or false. Round %d."
                     % (intu_input, rnd)}],
                16384, think=False)))
        chair_turn = _chat(
            MAIN_URL, MAIN_MODEL,
            [{"role": "system", "content": CHAIR_SYS},
             {"role": "user", "content":
                 "What's been said so far:\n%s\n\nYour move. If the talk "
                 "has actually settled something, give CONCLUSION: — not "
                 "before." % (_transcript_text(transcript), rnd)}],
            65536)
        transcript.append(("chair", chair_turn))
        chair_turns += 1
        conclusion = _extract_conclusion(chair_turn)
        if conclusion:
            break

    vote = None
    if not conclusion:
        options_turn = _chat(
            MAIN_URL, MAIN_MODEL,
            [{"role": "system", "content": CHAIR_SYS},
             {"role": "user", "content":
                 "What's been said so far:\n%s\n\nStill no agreement. Put "
                 "the live options on the table: exactly 2-3, one per line, "
                 "format 'A. ...' / 'B. ...'. Don't pick one yourself "
                 "yet." % _transcript_text(transcript)}],
            65536)
        transcript.append(("chair", options_turn))
        chair_turns += 1
        opts = re.findall(r"^([ABC])[.)]\s*(.+)$", options_turn, re.MULTILINE)

        votes = {}
        ballots = {}
        if opts:
            opt_text = "\n".join("%s. %s" % (k, v) for k, v in opts)
            for speaker, (url, model, ctx, think) in _voters(
                    code_available, code_url, code_model, participants):
                b = _chat(url, model,
                          [{"role": "system",
                            "content": ENG_SYS if speaker == "code"
                            else INTU_SYS},
                           {"role": "user", "content":
                               "Council transcript:\n%s\n\nOptions:\n%s\n\n"
                               "Vote for exactly one option letter (A, B or "
                               "C) and give one sentence why."
                               % (_transcript_text(transcript), opt_text)}],
                          ctx, think=think)
                ballots[speaker] = b
                letter = re.search(r"\b([ABC])\b", b)
                if letter:
                    votes[speaker] = letter.group(1)

        tally = {}
        for s, v in votes.items():
            tally[v] = tally.get(v, 0) + 1
        tie = False
        winner = None
        if tally:
            best = max(tally.values())
            leaders = sorted(k for k, n in tally.items() if n == best)
            tie = len(leaders) > 1
            winner = leaders[0]
            if tie:
                chair_ballot = _chat(
                    MAIN_URL, MAIN_MODEL,
                    [{"role": "system", "content": CHAIR_SYS},
                     {"role": "user", "content":
                         "Vote tied between %s. As chair, break the tie in "
                         "one sentence and name the winning letter."
                         % " and ".join(leaders)}], 65536)
                transcript.append(("chair", chair_ballot))
                m = re.search(r"\b([ABC])\b", chair_ballot)
                winner = m.group(1) if m else leaders[0]
            transcript.append(("chair", "VOTE RESULT: option %s wins. "
                                        "Tally: %s. Tie: %s"
                                        % (winner, tally, tie)))
        vote = {"options": {k: v.strip()[:200] for k, v in opts},
                "ballots": ballots, "tally": tally, "tie": tie,
                "winner": winner}
        if not opts:
            conclusion = "Council ended without conclusion or valid options."

    result = {
        "topic": topic, "purpose": purpose,
        "participants": sorted(participants),
        "code_available": code_available,
        "code_model": code_model if code_available else None,
        "turns": len(transcript),
        "transcript": transcript,
        "conclusion": conclusion,
        "vote": vote,
        "duration_s": (datetime.now(timezone.utc) - t0).total_seconds(),
    }
    if log:
        # Full session → council_sessions.jsonl (webui panel reads this);
        # one-line summary → episodic internal_dialogue (consolidation reads this).
        sess_path = os.path.join(AION, "memory", "state", "council_sessions.jsonl")
        try:
            with open(sess_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "topic": topic, "purpose": purpose,
                    "participants": sorted(participants),
                    "code_available": code_available,
                    "code_model": code_model if code_available else None,
                    "turns": len(transcript),
                    "conclusion": conclusion,
                    "vote": {k: v for k, v in vote.items() if k != "ballots"} if vote else None,
                    "transcript": transcript,
                    "duration_s": result["duration_s"],
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            print("[council] session log failed: %s" % e)
        resolved = bool(conclusion and "without conclusion" not in
                        conclusion) or bool(vote and vote.get("winner"))
        # Episodic: one event per council turn with FULL text, plus a
        # resolution event — nightly consolidation treats the meeting as
        # first-class memory (extraction, claims, heuristics), not a stub.
        for speaker, text in transcript:
            _log_event("council[%s] %s: %s" % (purpose, speaker, text), {
                "purpose": purpose, "speaker": speaker,
                "topic": topic[:200], "code_available": code_available,
            })
        if conclusion:
            resolution = "conclusion: " + conclusion[:400]
        elif vote and vote.get("winner"):
            resolution = "vote: option %s wins, tally %s" % (
                vote.get("winner"), vote.get("tally"))
        else:
            resolution = "ended without resolution"
        _log_event("council[%s] resolution: %s" % (purpose, resolution), {
            "purpose": purpose, "resolved": resolved,
            "duration_s": result["duration_s"], "turns": len(transcript),
        })
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("topic")
    p.add_argument("--context", default="")
    p.add_argument("--purpose", default="manual")
    p.add_argument("--rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    a = p.parse_args()
    r = convene(a.topic, a.context, a.purpose, max_rounds=a.rounds)
    print(json.dumps(r, indent=1))
