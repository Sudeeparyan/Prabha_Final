"""
Run this after editing any file in emg_data/nodes/ to regenerate all_nodes.json.
Usage:  python emg_data/rebuild.py
"""
import json, os, datetime

base = os.path.join(os.path.dirname(__file__), "nodes")
categories = [
    ("faq",           "faq.json"),
    ("preference",    "preference.json"),
    ("event",         "event.json"),
    ("account_state", "account_state.json"),
]

all_nodes   = []
type_counts = {}

for _, fname in categories:
    path = os.path.join(base, fname)
    with open(path, encoding="utf-8") as f:
        nodes = json.load(f)
    all_nodes.extend(nodes)
    t = nodes[0]["type"] if nodes else "Unknown"
    type_counts[t] = len(nodes)

edges_path = os.path.join(os.path.dirname(__file__), "edges.json")
with open(edges_path, encoding="utf-8") as f:
    edges = json.load(f)

# validate all edge node IDs exist
node_ids = {n["id"] for n in all_nodes}
bad_edges = [(s, t) for s, t in edges if s not in node_ids or t not in node_ids]
if bad_edges:
    print(f"[WARN] {len(bad_edges)} edges reference missing node IDs:")
    for s, t in bad_edges:
        print(f"  {s} -> {t}  ({'src missing' if s not in node_ids else 'tgt missing'})")

master = {
    "meta": {
        "persona":        "Priya Sharma",
        "location":       "Dublin, Ireland",
        "occupation":     "Senior Data Analyst, TechCorp Dublin",
        "education":      "MSc Open Data Practice, NCI Ireland (Year 2)",
        "total_nodes":    len(all_nodes),
        "total_edges":    len(edges),
        "node_type_counts": type_counts,
        "last_updated":   datetime.date.today().isoformat(),
        "version":        "3.0",
    },
    "nodes": all_nodes,
    "edges": edges,
}

out = os.path.join(os.path.dirname(__file__), "all_nodes.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(master, f, indent=2, ensure_ascii=False)

print(f"Rebuilt all_nodes.json")
print(f"  Nodes: {len(all_nodes)}  {type_counts}")
print(f"  Edges: {len(edges)}  (bad: {len(bad_edges)})")
print(f"  Saved: {out}")
