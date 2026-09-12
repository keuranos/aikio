#!/usr/bin/env python3
"""standing_queries.py — R7.3: Standing queries feeding the curiosity queue.

Queries:
  - contradicts pairs → auto-question
  - Question nodes with no resolved_by older than 7 days → staleness
  - orphan clusters → "why did I stop caring about X?"
"""
import json, os, sys, glob
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))

AION = os.environ.get("AION_HOME", "$AION_HOME")
QUESTIONS_FILE = f"{AION}/memory/state/questions.json"
GRAPH_FILE = f"{AION}/graphs/mind/graphify-out/graph.json"

def load_json(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default

def load_graph():
    return load_json(GRAPH_FILE, {"nodes": [], "links": []})

def graph_links(graph):
    """Get edges from graph, handling both 'links' (graphify) and 'edges' (legacy) keys."""
    return graph.get("links", graph.get("edges", []))

def edge_relation(edge):
    """Get the relation type from an edge, handling 'relation' (graphify), 'type', and 'label' keys."""
    return edge.get("relation", edge.get("type", edge.get("label", "")))

def node_type(node):
    """Get the type of a node, handling 'file_type' (graphify) and 'type' (legacy) keys."""
    return node.get("file_type", node.get("type", ""))

def load_questions():
    return load_json(QUESTIONS_FILE, {"queue": []})

def save_questions(data):
    with open(QUESTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def find_contradictions(graph):
    """Find pairs of nodes connected by 'contradicts' edges."""
    contradictions = []
    links = graph_links(graph)
    nodes = {n.get("id", ""): n for n in graph.get("nodes", [])}
    
    for edge in links:
        if edge_relation(edge) == "contradicts":
            src = nodes.get(edge.get("source", ""), {})
            tgt = nodes.get(edge.get("target", ""), {})
            if src and tgt:
                contradictions.append({
                    "node_a": src.get("label", src.get("id", "?")),
                    "node_b": tgt.get("label", tgt.get("id", "?")),
                })
    return contradictions

def find_stale_questions(graph, max_age_days=7):
    """Find Question nodes with no resolved_by edge, older than max_age_days."""
    now = datetime.now(timezone.utc)
    stale = []
    
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    
    # Find question nodes
    question_nodes = [n for n in nodes if node_type(n) == "Question" or "question" in n.get("label", "").lower()]
    
    # Find resolved_by edges
    resolved_targets = set()
    for edge in links:
        if edge_relation(edge) == "resolved_by":
            resolved_targets.add(edge.get("source", ""))
    
    for q in question_nodes:
        qid = q.get("id", "")
        if qid in resolved_targets:
            continue
        
        # Check age
        ts = q.get("ts") or q.get("created_at")
        if ts:
            try:
                created = datetime.fromisoformat(str(ts))
                age = (now - created).days
                if age >= max_age_days:
                    stale.append({
                        "question": q.get("label", qid),
                        "age_days": age,
                    })
            except Exception:
                pass
        else:
            # No timestamp — assume old
            stale.append({"question": q.get("label", qid), "age_days": "unknown"})
    
    return stale

def find_orphan_clusters(graph):
    """Find nodes with very few connections that might indicate abandoned topics."""
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    
    # Count connections per node
    connections = {}
    for edge in links:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        connections[src] = connections.get(src, 0) + 1
        connections[tgt] = connections.get(tgt, 0) + 1
    
    orphans = []
    for node in nodes:
        nid = node.get("id", "")
        conn_count = connections.get(nid, 0)
        if conn_count <= 1:  # Orphan or near-orphan
            label = node.get("label", nid)
            # Skip system nodes
            if label and not any(s in label.lower() for s in ["system", "genesis", "seed", "origin"]):
                orphans.append({"node": label, "connections": conn_count})
    
    return orphans[:5]  # Top 5

def generate_questions():
    """Run standing queries and add results to the question queue."""
    graph = load_graph()
    questions = load_questions()
    existing_qs = {q.get("q", "").lower() for q in questions.get("queue", [])}
    
    new_questions = []
    now = now_iso()
    
    # 1. Contradictions → auto-questions
    contradictions = find_contradictions(graph)
    for c in contradictions:
        q = f"How do I reconcile the contradiction between '{c['node_a']}' and '{c['node_b']}'?"
        if q.lower() not in existing_qs:
            new_questions.append({"q": q, "added": now, "source": "standing_query:contradicts"})
    
    # 2. Stale questions
    stale = find_stale_questions(graph)
    for s in stale:
        q = f"Why is the question '{s['question']}' still unresolved after {s.get('age_days', '?')} days?"
        if q.lower() not in existing_qs:
            new_questions.append({"q": q, "added": now, "source": "standing_query:stale"})
    
    # 3. Orphan clusters
    orphans = find_orphan_clusters(graph)
    for o in orphans:
        q = f"Why did I stop caring about '{o['node']}'? (only {o['connections']} connections)"
        if q.lower() not in existing_qs:
            new_questions.append({"q": q, "added": now, "source": "standing_query:orphan"})
    
    if new_questions:
        questions.setdefault("queue", []).extend(new_questions)
        save_questions(questions)
        print(f"[standing_queries] Added {len(new_questions)} questions from standing queries")
        for q in new_questions:
            print(f"  [{q['source'].split(':')[1]}] {q['q'][:80]}")
    else:
        print("[standing_queries] No new questions from standing queries")
    
    return new_questions

if __name__ == "__main__":
    generate_questions()
