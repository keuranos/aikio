#!/usr/bin/env python3
"""eval_heuristics.py — Standalone heuristic evaluation + promotion.

Called from nightly.sh AFTER consolidation, with its own timeout.
This prevents the consolidation 1-hour timeout from killing heuristic
evaluation before it runs.

Two phases:
  1. evaluate: check active heuristics against recent events via LLM
  2. promote: auto-promote any candidates into SELF.md tier:recent

Usage:
  python3 eval_heuristics.py
"""
import os, sys, json, subprocess
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aion_env  # noqa

AION = os.environ.get("AION_HOME", "$AION_HOME")

# Reuse the LLM and event infrastructure from consolidate_v2
from consolidate_v2 import llm, evaluate_and_promote_heuristics, auto_promote_heuristics, log_event


def main():
    print(f"[eval_heuristics] Starting at {datetime.now(timezone.utc).isoformat()}")
    
    # Phase 1: Evaluate active heuristics against recent events
    print("[eval_heuristics] Phase 1: Evaluating heuristics...")
    evaluate_and_promote_heuristics()
    
    # Phase 2: Auto-promote any candidates
    print("[eval_heuristics] Phase 2: Auto-promoting candidates...")
    auto_promote_heuristics()
    
    # Git commit if SELF.md or HEURISTICS.md changed
    try:
        result = subprocess.run(
            ["git", "-C", AION, "status", "--porcelain", "SELF.md", "HEURISTICS.md", "memory/state/heuristics.json"],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            subprocess.run(["git", "-C", AION, "add", "SELF.md", "HEURISTICS.md", "memory/state/heuristics.json"],
                         timeout=10)
            subprocess.run(["git", "-C", AION, "commit", "-m", "nightly: heuristic evaluation + promotion"],
                         timeout=10)
            print("[eval_heuristics] Committed changes to SELF.md/HEURISTICS.md")
        else:
            print("[eval_heuristics] No changes to commit")
    except Exception as e:
        print(f"[eval_heuristics] Git commit failed: {e}")
    
    print(f"[eval_heuristics] Done at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[eval_heuristics] FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)