#!/usr/bin/env python3
"""harvest_questions.py — pull 'suggested questions' and 'surprising
connections' from GRAPH_REPORT.md files into the question queue.
Deduplicates against existing queue; caps queue at 40 items."""
import json, os, re
from datetime import datetime, timezone
import aion_env  # loads config/aion.env

AION = os.environ.get("AION_HOME", "$AION_HOME")
REPORTS = [
    f"{AION}/graphs/mind/graphify-out/GRAPH_REPORT.md",
    f"{AION}/graphs/code/src/graphify-out/GRAPH_REPORT.md",
]
QPATH = f"{AION}/memory/state/questions.json"

def load():
    try:
        return json.load(open(QPATH))
    except Exception:
        return {"queue": []}

def harvest(text):
    items = []
    # questions: lines ending in '?' inside list items
    for m in re.finditer(r"^\s*[-*\d.]+\s*(.+\?)\s*$", text, re.M):
        items.append(m.group(1).strip())
    # surprising-connections section: turn each into a question
    sec = re.search(r"#+\s*Surprising connections(.+?)(\n#+|\Z)", text, re.S | re.I)
    if sec:
        for m in re.finditer(r"^\s*[-*]\s*(.+)$", sec.group(1), re.M):
            line = m.group(1).strip()
            if line and not line.endswith("?"):
                items.append(f"Why are these connected: {line}?")
    return items

def main():
    q = load()
    existing = {x["q"] for x in q["queue"]}
    now = datetime.now(timezone.utc).isoformat()
    added = 0
    for rp in REPORTS:
        if not os.path.exists(rp):
            continue
        src = "mind_graph" if "/mind/" in rp else "code_graph"
        for item in harvest(open(rp, encoding="utf-8").read()):
            if item not in existing and len(q["queue"]) < 40:
                q["queue"].append({"q": item, "added": now, "source": src})
                existing.add(item)
                added += 1
    os.makedirs(os.path.dirname(QPATH), exist_ok=True)
    json.dump(q, open(QPATH, "w"), indent=2)
    print(f"harvested {added} new questions")

if __name__ == "__main__":
    main()
