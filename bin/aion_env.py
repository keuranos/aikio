#!/usr/bin/env python3
"""aion_env.py — load AION_HOME/config/aion.env into os.environ.

Imported at the top of every Aion script so that cron/systemd/manual runs
all see the same single source of truth, even when the shell environment
does not export the variables explicitly.
"""
import os
from pathlib import Path


def load_aion_env(path=None):
    """Load key=value pairs from aion.env, stripping comments and quotes."""
    if path is None:
        aion = os.environ.get("AION_HOME", "$AION_HOME")
        path = Path(aion) / "config" / "aion.env"
    else:
        path = Path(path)

    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # strip inline comments: only when preceded by whitespace, unless quoted
            quote = None
            i = 0
            cleaned = []
            while i < len(value):
                ch = value[i]
                if ch in ('"', "'"):
                    if quote is None:
                        quote = ch
                        i += 1
                        continue
                    elif quote == ch:
                        quote = None
                        i += 1
                        continue
                if quote is None and ch == "#" and (i == 0 or value[i - 1].isspace()):
                    break
                cleaned.append(ch)
                i += 1
            value = "".join(cleaned).rstrip()

            # strip surrounding quotes
            if len(value) >= 2 and value[0] in ('"', "'") and value[0] == value[-1]:
                value = value[1:-1]

            if key and key not in os.environ:
                os.environ[key] = value


# Auto-load on import so scripts only need: import aion_env
load_aion_env()
