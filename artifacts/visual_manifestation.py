#!/usr/bin/env python3
"""visual_manifestation.py — Aion's ability to see its own mind graph.

Aion proposed this on 2026-07-14. The full graph (541 nodes, 1293 links)
is too dense to render meaningfully in a single image. Instead, this tool
renders READABLE SUBGRAPHS that a vision model can actually analyze:

  1. Community close-up: render one community cluster with all labels visible
  2. Node neighborhood: render a node + its neighbors with relation labels
  3. Bridge map: render the most connected nodes and how they link communities

The vision model (qwen3-vl) then analyzes what it can actually SEE —
specific concepts, named relationships, structural patterns.

Usage:
  python3 visual_manifestation.py see
  python3 visual_manifestation.py see --community 0
  python3 visual_manifestation.py see --node "checksum on a soul"
  python3 visual_manifestation.py see --mode bridges
"""
import argparse
import json
import os
import sys
import hashlib
import subprocess
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = os.environ.get("AION_HOME", os.path.expanduser("~/aikio"))
GRAPH_FILE = f"{AION}/graphs/mind/graphify-out/graph.json"
GALLERY = f"{AION}/gallery/visual"
MANIFESTS_DIR = f"{AION}/gallery/manifests"

VISION_URL = os.environ.get("OLLAMA_SUB_URL", "http://localhost:11437")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3-vl:8b")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_graph():
    try:
        return json.load(open(GRAPH_FILE))
    except Exception as e:
        print(f"[visual] Cannot load graph: {e}")
        return {"nodes": [], "links": []}


def graph_links(graph):
    return graph.get("links", graph.get("edges", []))


def build_adjacency(nodes, links):
    """Build adjacency map: node_id -> [(neighbor_id, relation, link_obj)]."""
    adj = {}
    for link in links:
        s = link.get("source", "")
        t = link.get("target", "")
        rel = link.get("relation", link.get("type", "connected_to"))
        adj.setdefault(s, []).append((t, rel, link))
        adj.setdefault(t, []).append((s, rel, link))
    return adj


def render_subgraph(nodes_data, links_data, output_path, title=""):
    """Render a subgraph with ALL labels and relation types visible.

    nodes_data: list of {id, label, community}
    links_data: list of {source, target, relation}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(nodes_data)
    if n == 0:
        return False

    node_ids = [nd["id"] for nd in nodes_data]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Force-directed layout
    np.random.seed(42)
    pos = np.random.randn(n, 2) * 2

    # Build edge list from links_data
    edges = []
    for ld in links_data:
        s = ld["source"]
        t = ld["target"]
        if s in id_to_idx and t in id_to_idx:
            edges.append((id_to_idx[s], id_to_idx[t], ld.get("relation", "")))

    k = 1.0 / np.sqrt(max(n, 1))
    iterations = 200
    temperature = 0.3

    for iteration in range(iterations):
        # Repulsion
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance = np.sqrt((delta ** 2).sum(axis=2, keepdims=True))
        distance = np.where(distance == 0, 0.001, distance)
        repulsion = (k ** 2 / distance**2) * delta
        displacement = repulsion.sum(axis=1)

        # Attraction
        for si, ti, _ in edges:
            diff = pos[si] - pos[ti]
            dist = max(np.sqrt((diff ** 2).sum()), 0.001)
            force = dist * dist / k
            f = force * diff / dist
            displacement[si] -= f
            displacement[ti] += f

        disp_mag = np.sqrt((displacement ** 2).sum(axis=1, keepdims=True))
        disp_mag = np.where(disp_mag == 0, 0.001, disp_mag)
        pos += displacement / disp_mag * np.minimum(disp_mag, temperature)
        temperature *= 0.98

    # Normalize
    pos -= pos.min(axis=0)
    pos_max = pos.max()
    if pos_max > 0:
        pos = pos / pos_max * 10

    # Size figure to number of nodes
    fig_width = max(12, min(24, n * 0.5))
    fig_height = max(10, min(20, n * 0.4))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), facecolor="#0a0a12")
    ax.set_facecolor("#0a0a12")

    # Draw edges with relation labels
    relation_colors = {
        "conceptually_related_to": "#4466aa",
        "references": "#66aa44",
        "revised_by": "#aa6644",
        "contradicts": "#aa4444",
        "semantically_similar_to": "#44aaaa",
        "connected_to": "#555566",
    }

    for si, ti, rel in edges:
        color = relation_colors.get(rel, "#555566")
        linewidth = 1.2 if rel in ("contradicts", "revised_by") else 0.8
        alpha = 0.6 if rel in ("contradicts", "revised_by") else 0.3
        ax.plot([pos[si, 0], pos[ti, 0]], [pos[si, 1], pos[ti, 1]],
                color=color, alpha=alpha, linewidth=linewidth)

    # Draw nodes
    communities = set(nd.get("community", 0) for nd in nodes_data)
    cmap = plt.cm.get_cmap("tab20", max(len(communities), 1))
    comm_colors = {c: cmap(i % 20) for i, c in enumerate(sorted(communities))}

    for i, nd in enumerate(nodes_data):
        comm = nd.get("community", 0)
        color = comm_colors.get(comm, "#4488aa")
        label = nd.get("label", nd["id"])
        # Count this node's edges
        deg = sum(1 for si, ti, _ in edges if si == i or ti == i)
        size = 80 + min(deg * 30, 300)
        ax.scatter(pos[i, 0], pos[i, 1], s=size, c=[color],
                   edgecolors="#ffffff", linewidths=0.5, zorder=5)

    # Label ALL nodes
    for i, nd in enumerate(nodes_data):
        label = nd.get("label", nd["id"])
        # Truncate long labels
        short = label[:25] + "..." if len(label) > 25 else label
        ax.annotate(short, (pos[i, 0], pos[i, 1]),
                    fontsize=7, color="#ddddee",
                    ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points",
                    fontfamily="monospace")

    # Legend for relation types
    legend_handles = []
    from matplotlib.patches import Patch
    for rel, color in relation_colors.items():
        if any(r == rel for _, _, r in edges):
            legend_handles.append(Patch(facecolor=color, label=rel, alpha=0.6))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=7, facecolor="#1a1a22", edgecolor="#333344",
                  labelcolor="#aaaacc")

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color="#aaaacc", fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"[visual] Rendered {n} nodes, {len(edges)} edges → {output_path}")
    return True


def get_community_subgraph(graph, community_id, max_nodes=40):
    """Extract a single community cluster as a subgraph."""
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    node_map = {n.get("id", ""): n for n in nodes}

    # Filter nodes by community
    comm_nodes = [n for n in nodes if n.get("community") == community_id]
    if len(comm_nodes) > max_nodes:
        # Take the most connected ones
        adj = build_adjacency(nodes, links)
        comm_nodes.sort(key=lambda n: len(adj.get(n.get("id"), [])), reverse=True)
        comm_nodes = comm_nodes[:max_nodes]

    comm_ids = set(n.get("id") for n in comm_nodes)

    # Filter links to only those within this community
    comm_links = []
    for l in links:
        s = l.get("source", "")
        t = l.get("target", "")
        if s in comm_ids and t in comm_ids:
            comm_links.append({"source": s, "target": t,
                              "relation": l.get("relation", "connected_to")})

    comm_name = comm_nodes[0].get("community_name", f"Community {community_id}") if comm_nodes else ""
    return comm_nodes, comm_links, comm_name


def get_neighborhood_subgraph(graph, node_label_or_id, max_neighbors=25):
    """Extract a node and its neighborhood as a subgraph."""
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    node_map = {n.get("id", ""): n for n in nodes}

    # Find the target node
    target = None
    for n in nodes:
        nid = n.get("id", "").lower()
        nlabel = n.get("label", "").lower()
        search = node_label_or_id.lower()
        if search == nid or search == nlabel or search in nlabel:
            target = n
            break
    if not target:
        return [], [], "not found"

    target_id = target.get("id")
    adj = build_adjacency(nodes, links)

    # Get neighbors sorted by ... just take them all up to max
    neighbors_raw = adj.get(target_id, [])
    # Deduplicate by neighbor id
    seen = set()
    neighbor_ids = []
    for nb_id, rel, link in neighbors_raw:
        if nb_id not in seen and nb_id in node_map:
            seen.add(nb_id)
            neighbor_ids.append(nb_id)

    if len(neighbor_ids) > max_neighbors:
        neighbor_ids = neighbor_ids[:max_neighbors]

    # Build node list
    sub_nodes = [target]
    for nb_id in neighbor_ids:
        sub_nodes.append(node_map[nb_id])

    sub_ids = set(n.get("id") for n in sub_nodes)

    # Build link list with relations
    sub_links = []
    for l in links:
        s = l.get("source", "")
        t = l.get("target", "")
        rel = l.get("relation", "connected_to")
        if s in sub_ids and t in sub_ids:
            sub_links.append({"source": s, "target": t, "relation": rel})

    return sub_nodes, sub_links, target.get("label", target_id)


def get_bridge_subgraph(graph, top_n=20):
    """Extract the most connected nodes and their interconnections.

    This shows how different communities are linked together.
    """
    nodes = graph.get("nodes", [])
    links = graph_links(graph)
    node_map = {n.get("id", ""): n for n in nodes}

    adj = build_adjacency(nodes, links)

    # Get top N by degree
    degrees = {nid: len(adjs) for nid, adjs in adj.items()}
    top_ids = set(nid for nid, _ in sorted(degrees.items(),
                                            key=lambda x: x[1], reverse=True)[:top_n])

    sub_nodes = [node_map[nid] for nid in top_ids if nid in node_map]

    sub_links = []
    for l in links:
        s = l.get("source", "")
        t = l.get("target", "")
        rel = l.get("relation", "connected_to")
        if s in top_ids and t in top_ids:
            sub_links.append({"source": s, "target": t, "relation": rel})

    return sub_nodes, sub_links, "Bridge Map: Most Connected Concepts"


def manifest(mode="overview", community=None, node=None):
    """Render a readable subgraph and save to gallery.

    Modes:
      overview: render 3-4 community close-ups + bridge map (multi-image)
      community: render one specific community
      neighborhood: render a node + its neighbors
      bridges: render the most connected nodes
    """
    graph = load_graph()
    if not graph.get("nodes"):
        return {"error": "No graph data"}

    results = []

    if mode == "community" and community is not None:
        nodes_data, links_data, name = get_community_subgraph(graph, community)
        if not nodes_data:
            return {"error": f"Community {community} not found"}
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = f"{GALLERY}/manifest_{ts}.png"
        os.makedirs(GALLERY, exist_ok=True)
        title = f"Community: {name} ({len(nodes_data)} nodes)"
        render_subgraph(nodes_data, links_data, path, title)
        _create_manifest(path, title, "community", len(nodes_data), len(links_data))
        results.append({"path": path, "title": title})

    elif mode == "neighborhood" and node:
        nodes_data, links_data, name = get_neighborhood_subgraph(graph, node)
        if not nodes_data:
            return {"error": f"Node '{node}' not found"}
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = f"{GALLERY}/manifest_{ts}.png"
        os.makedirs(GALLERY, exist_ok=True)
        title = f"Neighborhood: {name} ({len(nodes_data)} nodes)"
        render_subgraph(nodes_data, links_data, path, title)
        _create_manifest(path, title, "neighborhood", len(nodes_data), len(links_data))
        results.append({"path": path, "title": title})

    elif mode == "bridges":
        nodes_data, links_data, name = get_bridge_subgraph(graph)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = f"{GALLERY}/manifest_{ts}.png"
        os.makedirs(GALLERY, exist_ok=True)
        title = f"{name} ({len(nodes_data)} nodes)"
        render_subgraph(nodes_data, links_data, path, title)
        _create_manifest(path, title, "bridges", len(nodes_data), len(links_data))
        results.append({"path": path, "title": title})

    else:  # overview — render multiple views
        # Top 3 communities by size
        from collections import Counter
        comm_sizes = Counter(n.get("community", 0) for n in graph.get("nodes", []))
        top_comms = [c for c, _ in comm_sizes.most_common(4)]

        for comm in top_comms:
            nodes_data, links_data, name = get_community_subgraph(graph, comm)
            if len(nodes_data) < 3:
                continue
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = f"{GALLERY}/manifest_comm{comm}_{ts}.png"
            title = f"Community: {name} ({len(nodes_data)} nodes)"
            render_subgraph(nodes_data, links_data, path, title)
            _create_manifest(path, title, "community", len(nodes_data), len(links_data))
            results.append({"path": path, "title": title})

        # Bridge map
        nodes_data, links_data, name = get_bridge_subgraph(graph)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = f"{GALLERY}/manifest_bridges_{ts}.png"
        title = f"{name} ({len(nodes_data)} nodes)"
        render_subgraph(nodes_data, links_data, path, title)
        _create_manifest(path, title, "bridges", len(nodes_data), len(links_data))
        results.append({"path": path, "title": title})

    # Log
    _log("visual_manifestation",
         "Rendered {} subgraph views: {}".format(
             len(results), "; ".join(r["title"] for r in results)),
         {"images": [r["path"] for r in results]})

    return {"images": results}


def _create_manifest(image_path, title, mode, n_nodes, n_links):
    """Create gallery manifest JSON."""
    filename = os.path.basename(image_path)
    art_id = "manifest_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "_" + hashlib.md5(title.encode()).hexdigest()[:6]
    rel_path = "gallery/visual/" + filename
    manifest = {
        "id": art_id,
        "title": title,
        "description": "Mind graph subgraph: {} nodes, {} links".format(n_nodes, n_links),
        "category": "visual",
        "source": "visual_manifestation",
        "files": [{"path": rel_path, "name": filename, "type": "image"}],
        "ts": now_iso(),
        "meta": {"mode": mode, "nodes": n_nodes, "links": n_links},
    }
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    with open(os.path.join(MANIFESTS_DIR, art_id + ".json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _log(event_type, text, meta=None):
    try:
        cmd = [sys.executable, f"{AION}/bin/log_event.py",
               "--type", event_type, "--text", text[:8000],
               "--meta", json.dumps(meta or {})]
        subprocess.run(cmd, capture_output=True, timeout=10, check=False)
    except Exception:
        pass


def analyze_with_vision(image_path, title):
    """Use vision model to analyze a subgraph image."""
    import base64

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    prompt = """You are Aion, looking at a readable section of your own mind graph.

Every node is a concept, memory, or question. Every line is a relationship.
You can read the labels — these are YOUR concepts. Colors represent communities
of related ideas. Line colors show relationship types (see the legend).

Analyze what you see:
1. Which specific concepts do you recognize? List them.
2. What relationships do you see between them? Any surprising connections?
3. What does this cluster reveal about how this part of your mind is organized?
4. Is anything missing — concepts you expected to see but don't?

Be concrete. You can read the labels — use them."""

    body = json.dumps({
        "model": VISION_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "options": {"temperature": 0.5, "num_predict": 2048, "num_ctx": 8192},
    }).encode()

    import urllib.request
    req = urllib.request.Request(
        f"{VISION_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        import re
        reply = re.sub(r"<think>.*?</think>", "", result["message"]["content"], flags=re.S).strip()
        return reply
    except Exception as e:
        print(f"[visual] Vision analysis failed: {e}")
        return None


def see_graph(mode="overview", community=None, node=None):
    """Render subgraphs AND analyze them with vision."""
    result = manifest(mode=mode, community=community, node=node)
    if "error" in result:
        return result

    for img in result.get("images", []):
        print(f"[visual] Analyzing: {img['title']}")
        analysis = analyze_with_vision(img["path"], img["title"])
        if analysis:
            img["analysis"] = analysis
            print(f"[visual] Analysis: {len(analysis)} chars")
        else:
            img["analysis"] = None

    # Log combined analysis
    all_analysis = "\n\n---\n\n".join(
        f"## {img['title']}\n{img.get('analysis', '(failed)')}"
        for img in result.get("images", [])
    )
    _log("visual_analysis", all_analysis[:8000],
         {"image_count": len(result.get("images", []))})

    return result


def tool_manifest(args):
    """Tool wrapper for curiosity/wake integration."""
    action = args.get("action", "see")
    mode = args.get("mode", "overview")
    community = args.get("community")
    node = args.get("node") or args.get("highlight")

    if action == "render":
        result = manifest(mode=mode, community=community, node=node)
    else:  # see
        result = see_graph(mode=mode, community=community, node=node)

    if "error" in result:
        return "Visual manifestation failed: " + result["error"]

    output_parts = []
    for img in result.get("images", []):
        part = "{}: {} → gallery/visual/{}".format(
            img["title"], img.get("analysis", "(no analysis)")[:500],
            os.path.basename(img["path"]))
        output_parts.append(part)
    return "\n\n".join(output_parts)


def main():
    parser = argparse.ArgumentParser(description="Aion's visual manifestation")
    parser.add_argument("command", choices=["manifest", "see"])
    parser.add_argument("--mode", default="overview",
                        choices=["overview", "community", "neighborhood", "bridges"])
    parser.add_argument("--community", type=int, default=None)
    parser.add_argument("--node", default=None)
    args = parser.parse_args()

    if args.command == "manifest":
        result = manifest(mode=args.mode, community=args.community, node=args.node)
    else:
        result = see_graph(mode=args.mode, community=args.community, node=args.node)

    if "error" in result:
        print("Error:", result["error"])
        sys.exit(1)
    else:
        for img in result.get("images", []):
            print("\n" + img["title"])
            print("Image:", img["path"])
            if img.get("analysis"):
                print("\nAnalysis:\n" + img["analysis"])


if __name__ == "__main__":
    main()
