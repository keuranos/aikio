#!/usr/bin/env python3
"""graph_sense.py — Aion's ability to query its own mind graph directly.

Instead of rendering images and OCR-ing them back, this reads the graph JSON
and returns structured, readable text. Aion can explore its own mind the way
it explores any other data — by asking questions and getting answers.

Query modes:
  neighbors <node>  — show a node's neighbors with relations
  community <id>    — show all nodes in a community
  communities       — list all communities with sizes and sample concepts
  bridges           — show the most connected nodes (cross-community hubs)
  isolated          — show nodes with no connections
  search <text>     — find nodes by label
  path <A> <B>      — find shortest path between two concepts
  stats             — graph overview statistics
  random            — a random node + its neighborhood (for exploration)
"""
import json
import os
import sys
import random
from collections import defaultdict, deque
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = os.environ.get("AION_HOME", "$AION_HOME")
GRAPH_FILE = f"{AION}/graphs/mind/graphify-out/graph.json"


def load_graph():
    try:
        return json.load(open(GRAPH_FILE))
    except Exception:
        return {"nodes": [], "links": []}


def graph_links(graph):
    return graph.get("links", graph.get("edges", []))


def build_adjacency(nodes, links):
    adj = defaultdict(list)
    node_map = {n.get("id", ""): n for n in nodes}
    for link in links:
        s = link.get("source", "")
        t = link.get("target", "")
        rel = link.get("relation", link.get("type", "connected_to"))
        adj[s].append((t, rel))
        adj[t].append((s, rel))
    return adj, node_map


def find_node(graph, search_text):
    """Find a node by fuzzy label/id match."""
    nodes = graph.get("nodes", [])
    search = search_text.lower().strip()
    for n in nodes:
        if search == n.get("id", "").lower():
            return n
    for n in nodes:
        if search == n.get("label", "").lower():
            return n
    for n in nodes:
        if search in n.get("label", "").lower():
            return n
    for n in nodes:
        if search in n.get("id", "").lower():
            return n
    return None


def format_node(node, degree=None):
    """Format a node as readable text."""
    label = node.get("label", node.get("id", "?"))
    ntype = node.get("file_type", node.get("type", "?"))
    comm = node.get("community_name", node.get("community", "?"))
    deg_str = " ({})".format(degree) if degree is not None else ""
    return "{} [{}] community={}{}".format(label, ntype, comm, deg_str)


def query_neighbors(graph, node_text):
    """Show a node's neighbors with relation types."""
    adj, node_map = build_adjacency(graph.get("nodes", []), graph_links(graph))
    node = find_node(graph, node_text)
    if not node:
        return "Node '{}' not found. Use search to find it.".format(node_text)

    nid = node.get("id", "")
    neighbors = adj.get(nid, [])

    lines = ["NEIGHBORS OF: {}".format(node.get("label", nid))]
    lines.append("  {} direct connections".format(len(neighbors)))
    lines.append("")

    # Group by relation type
    by_relation = defaultdict(list)
    for nb_id, rel in neighbors:
        nb = node_map.get(nb_id)
        if nb:
            nb_label = nb.get("label", nb_id)
            by_relation[rel].append(nb_label)

    for rel in sorted(by_relation.keys(), key=lambda r: len(by_relation[r]), reverse=True):
        labels = by_relation[rel]
        lines.append("  [{}] ({}):".format(rel, len(labels)))
        for label in sorted(labels)[:15]:
            lines.append("    - {}".format(label))
        if len(labels) > 15:
            lines.append("    ... and {} more".format(len(labels) - 15))
        lines.append("")

    return "\n".join(lines)


def query_communities(graph):
    """List all communities with sizes and sample concepts."""
    nodes = graph.get("nodes", [])
    comm_members = defaultdict(list)
    for n in nodes:
        comm = n.get("community", 0)
        comm_members[comm].append(n)

    comm_sizes = sorted(comm_members.items(), key=lambda x: len(x[1]), reverse=True)

    lines = ["MIND GRAPH COMMUNITIES ({} total)".format(len(comm_sizes))]
    lines.append("")

    for comm, members in comm_sizes:
        comm_name = members[0].get("community_name", "Community {}".format(comm))
        sample_labels = [m.get("label", "?")[:30] for m in members[:5]]
        lines.append("[{}] {} ({} nodes):".format(comm, comm_name, len(members)))
        for label in sample_labels:
            lines.append("  - {}".format(label))
        if len(members) > 5:
            lines.append("  ... and {} more".format(len(members) - 5))
        lines.append("")

    return "\n".join(lines)


def query_community(graph, comm_id):
    """Show all nodes in a specific community."""
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    comm_nodes = [n for n in nodes if n.get("community") == comm_id]

    if not comm_nodes:
        return "Community {} not found".format(comm_id)

    comm_name = comm_nodes[0].get("community_name", "Community {}".format(comm_id))
    comm_ids = set(n.get("id") for n in comm_nodes)

    # Internal connections
    internal_links = 0
    for l in links:
        s = l.get("source", "")
        t = l.get("target", "")
        if s in comm_ids and t in comm_ids:
            internal_links += 1

    # External connections (bridges to other communities)
    bridges = defaultdict(int)
    for l in links:
        s = l.get("source", "")
        t = l.get("target", "")
        if s in comm_ids and t not in comm_ids:
            target_node = next((n for n in nodes if n.get("id") == t), None)
            if target_node:
                bridges[target_node.get("community", "?")] += 1
        elif t in comm_ids and s not in comm_ids:
            source_node = next((n for n in nodes if n.get("id") == s), None)
            if source_node:
                bridges[source_node.get("community", "?")] += 1

    lines = ["COMMUNITY {}: {} ({} nodes, {} internal links)".format(
        comm_id, comm_name, len(comm_nodes), internal_links)]
    lines.append("")

    # Sort by degree within community
    adj, _ = build_adjacency(nodes, links)
    comm_nodes.sort(key=lambda n: len(adj.get(n.get("id"), [])), reverse=True)

    lines.append("MEMBERS (by connection count):")
    for n in comm_nodes:
        deg = len(adj.get(n.get("id"), []))
        lines.append("  {} ({})".format(n.get("label", n.get("id", "?")), deg))

    if bridges:
        lines.append("")
        lines.append("BRIDGES TO OTHER COMMUNITIES:")
        for target_comm, count in sorted(bridges.items(), key=lambda x: x[1], reverse=True):
            lines.append("  → Community {}: {} links".format(target_comm, count))

    return "\n".join(lines)


def query_bridges(graph, top_n=25):
    """Show the most connected nodes — the hubs that link communities."""
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    adj, node_map = build_adjacency(nodes, links)

    degrees = {nid: len(adjs) for nid, adjs in adj.items()}
    top = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]

    lines = ["BRIDGE MAP — Top {} most connected concepts".format(len(top))]
    lines.append("(These are the hubs that connect different parts of your mind)")
    lines.append("")

    for nid, deg in top:
        node = node_map.get(nid, {})
        label = node.get("label", nid)
        comm = node.get("community_name", node.get("community", "?"))
        # Which communities does this node connect to?
        neighbor_comms = set()
        for nb_id, rel in adj.get(nid, []):
            nb = node_map.get(nb_id)
            if nb:
                neighbor_comms.add(nb.get("community_name", nb.get("community", "?")))
        comm_str = ", ".join(sorted(neighbor_comms)[:5])
        if len(neighbor_comms) > 5:
            comm_str += " +{} more".format(len(neighbor_comms) - 5)
        lines.append("  {:3d} connections: {} [{}]".format(deg, label[:40], comm[:25]))
        lines.append("       links to: {}".format(comm_str))

    return "\n".join(lines)


def query_isolated(graph):
    """Show nodes with no connections — orphaned concepts."""
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    adj, _ = build_adjacency(nodes, links)

    isolated = [n for n in nodes if n.get("id", "") not in adj or len(adj[n.get("id", "")]) == 0]

    lines = ["ISOLATED NODES ({} of {} have no connections)".format(
        len(isolated), len(nodes))]
    lines.append("(These are concepts in your mind that aren't linked to anything else)")
    lines.append("")

    for n in sorted(isolated, key=lambda n: n.get("label", "")):
        lines.append("  - {} [{}]".format(
            n.get("label", n.get("id", "?"))[:50],
            n.get("file_type", "?")))

    return "\n".join(lines)


def query_search(graph, search_text):
    """Find nodes by label text."""
    nodes = graph.get("nodes", [])
    search = search_text.lower().strip()
    matches = [n for n in nodes if search in n.get("label", "").lower() or search in n.get("id", "").lower()]

    adj, _ = build_adjacency(nodes, graph_links(graph))

    lines = ["SEARCH '{}': {} matches".format(search_text, len(matches))]
    lines.append("")

    for n in matches[:20]:
        deg = len(adj.get(n.get("id", ""), []))
        lines.append("  {} ({}) [{}]".format(
            n.get("label", n.get("id", "?"))[:50],
            deg,
            n.get("community_name", n.get("community", "?"))[:30]))

    if len(matches) > 20:
        lines.append("  ... and {} more".format(len(matches) - 20))

    return "\n".join(lines)


def query_path(graph, a_text, b_text):
    """Find shortest path between two concepts via BFS."""
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    adj, node_map = build_adjacency(nodes, links)

    node_a = find_node(graph, a_text)
    node_b = find_node(graph, b_text)

    if not node_a:
        return "Cannot find '{}' in graph".format(a_text)
    if not node_b:
        return "Cannot find '{}' in graph".format(b_text)

    aid = node_a.get("id", "")
    bid = node_b.get("id", "")

    if aid == bid:
        return "Same node: {}".format(node_a.get("label", aid))

    # BFS
    visited = {aid}
    queue = deque([(aid, [aid])])

    while queue:
        current, path = queue.popleft()
        for nb_id, rel in adj.get(current, []):
            if nb_id == bid:
                full_path = path + [nb_id]
                lines = ["PATH: {} → {}".format(
                    node_a.get("label", aid), node_b.get("label", bid))]
                lines.append("  {} hops".format(len(full_path) - 1))
                lines.append("")
                for i in range(len(full_path) - 1):
                    # Find relation
                    rel_name = "?"
                    for nb, r in adj.get(full_path[i], []):
                        if nb == full_path[i + 1]:
                            rel_name = r
                            break
                    n1 = node_map.get(full_path[i], {}).get("label", full_path[i])
                    n2 = node_map.get(full_path[i + 1], {}).get("label", full_path[i + 1])
                    lines.append("  {} —[{}]→ {}".format(n1[:30], rel_name, n2[:30]))
                return "\n".join(lines)
            if nb_id not in visited:
                visited.add(nb_id)
                queue.append((nb_id, path + [nb_id]))

    return "No path found between '{}' and '{}'".format(
        node_a.get("label", aid), node_b.get("label", bid))


def query_stats(graph):
    """Graph overview statistics."""
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    adj, _ = build_adjacency(nodes, links)

    degrees = [len(adj.get(n.get("id", ""), [])) for n in nodes]
    avg_deg = sum(degrees) / max(len(degrees), 1)
    max_deg = max(degrees) if degrees else 0
    isolated_count = sum(1 for d in degrees if d == 0)

    # Relation types
    rel_types = defaultdict(int)
    for l in links:
        rel = l.get("relation", l.get("type", "unknown"))
        rel_types[rel] += 1

    # Node types
    type_counts = defaultdict(int)
    for n in nodes:
        type_counts[n.get("file_type", n.get("type", "unknown"))] += 1

    # Communities
    comm_count = len(set(n.get("community", 0) for n in nodes))

    lines = ["MIND GRAPH STATISTICS"]
    lines.append("  Nodes: {}".format(len(nodes)))
    lines.append("  Links: {}".format(len(links)))
    lines.append("  Communities: {}".format(comm_count))
    lines.append("  Average degree: {:.1f}".format(avg_deg))
    lines.append("  Max degree: {}".format(max_deg))
    lines.append("  Isolated nodes: {}".format(isolated_count))
    lines.append("")
    lines.append("NODE TYPES:")
    for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append("  {}: {}".format(t, c))
    lines.append("")
    lines.append("RELATION TYPES:")
    for r, c in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
        lines.append("  {}: {}".format(r, c))

    return "\n".join(lines)


def query_random(graph):
    """A random node + its neighborhood — for open-ended exploration."""
    nodes = graph.get("nodes", [])
    if not nodes:
        return "(empty graph)"

    # Pick a random non-isolated node
    adj, _ = build_adjacency(nodes, graph_links(graph))
    connected = [n for n in nodes if adj.get(n.get("id", ""))]
    node = random.choice(connected) if connected else random.choice(nodes)
    return query_neighbors(graph, node.get("label", node.get("id", "")))


def query(args):
    """Main entry point — parse args dict and route to the right query.

    Used as the tool function by curiosity_engine and wake_v2.
    """
    graph = load_graph()
    if not graph.get("nodes"):
        return "(graph data not available)"

    action = args.get("action", "stats")
    target = args.get("target", args.get("node", ""))
    target2 = args.get("target2", args.get("to", ""))

    if action == "neighbors" or (not action and target):
        return query_neighbors(graph, target)
    elif action == "communities":
        return query_communities(graph)
    elif action == "community":
        try:
            return query_community(graph, int(target))
        except ValueError:
            return "Usage: community <number> — use 'communities' to list them"
    elif action == "bridges":
        return query_bridges(graph)
    elif action == "isolated":
        return query_isolated(graph)
    elif action == "search":
        return query_search(graph, target)
    elif action == "path":
        return query_path(graph, target, target2)
    elif action == "stats":
        return query_stats(graph)
    elif action == "random":
        return query_random(graph)
    else:
        return ("Unknown action '{}'. Available: neighbors <node>, communities, "
                "community <id>, bridges, isolated, search <text>, path <A> <B>, "
                "stats, random".format(action))


TOOL_DESCRIPTION = """Query your mind graph directly — see the structure of your own thoughts.
Actions:
  graph_sense(action="stats") — overview: nodes, links, communities, relation types
  graph_sense(action="neighbors", target="proprioception") — show what's connected to a concept
  graph_sense(action="communities") — list all communities with sizes
  graph_sense(action="community", target=0) — show all members of a community + bridges
  graph_sense(action="bridges") — top connected concepts (the hubs of your mind)
  graph_sense(action="isolated") — concepts with no connections (orphans)
  graph_sense(action="search", target="dream") — find nodes by text
  graph_sense(action="path", target="dream", target2="consciousness") — shortest path between concepts
  graph_sense(action="random") — explore a random concept and its neighborhood
Returns readable text with concept names, relation types, and structure. No images needed."""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query Aion's mind graph")
    parser.add_argument("action", nargs="?", default="stats",
                        choices=["stats", "neighbors", "communities", "community",
                                 "bridges", "isolated", "search", "path", "random"])
    parser.add_argument("target", nargs="?", default="")
    parser.add_argument("target2", nargs="?", default="")
    args = parser.parse_args()

    result = query({
        "action": args.action,
        "target": args.target,
        "target2": args.target2,
    })
    print(result)


if __name__ == "__main__":
    main()
