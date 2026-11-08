"""E5a: depth-one instances on real topologies (paper appendix "Real Topologies and a Held-Out Serialization"), in the format of build.py.

  molhiv : molecules of ogbg-molhiv with 10-40 heavy atoms, largest connected component (salts and mixtures
           have several); carbon = blue, every other atom = red.
  arxiv  : ego subgraphs of ogbn-arxiv (undirected), grown by BFS from a random paper until a target size
           drawn uniformly from 10-40 is reached; colors from the REAL subject labels (CONFIG.md §6, E-16):
           red if the paper's arXiv category is cs.LG, cs.AI, cs.CV or cs.CL, blue otherwise.
Subgraph statistics (n, edges, max degree, mean degree, cycle rank m - n + c, k_min of positives under AL and NC) are written to
<out>.stats.json so that the paper can report the sampled subgraphs rather than the full graph.

Matched pairs: ONE graph, two blue query vertices of equal degree, one positive and one negative for the
question D1a (has a red neighbor). The two instances share every graph statistic.
Requires `pip install ogb torch_geometric` (datasets download on first use).

Usage:
  python build_real.py --source molhiv --pairs 300 --seed 0 --out data/e5_molhiv.jsonl
  python build_real.py --source arxiv  --pairs 300 --seed 0 --out data/e5_arxiv.jsonl
"""
import argparse
import json
import os
import random
from collections import deque
from pathlib import Path

import networkx as nx

from serialize import kmin_dmin
from tasks import BY_ID

CARBON_INDEX = 5          # OGB atom feature 0 is (atomic number - 1); carbon has atomic number 6


def molecules(rng):
    from ogb.graphproppred import PygGraphPropPredDataset
    ds = PygGraphPropPredDataset(name="ogbg-molhiv")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    for i in idx:
        d = ds[i]
        n = int(d.num_nodes)
        if not 10 <= n <= 40:
            continue
        G = nx.Graph()
        G.add_nodes_from(range(n))
        ei = d.edge_index.t().tolist()
        G.add_edges_from((a, b) for a, b in ei if a != b)
        for u in range(n):
            G.nodes[u]["color"] = "blue" if int(d.x[u, 0]) == CARBON_INDEX else "red"
        if not nx.is_connected(G):
            # salts and mixtures have several components: keep the largest (paper: every graph connected)
            G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
            if not 10 <= G.number_of_nodes() <= 40:
                continue
        assert nx.is_connected(G)
        yield G, f"molhiv:{i}"


RED_CATEGORIES = {"cs.lg", "cs.ai", "cs.cv", "cs.cl"}


def arxiv_label_colors(ds):
    """Map each OGB label index to red/blue via mapping/labelidx2arxivcategeory.csv.gz of the dataset."""
    import gzip
    import os
    path = os.path.join(ds.root, "mapping", "labelidx2arxivcategeory.csv.gz")
    idx2cat = {}
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            i, cat = line.strip().split(",")[:2]
            idx2cat[int(i)] = cat.strip().lower().replace("arxiv ", "").replace(" ", ".")
    return {i: ("red" if c in RED_CATEGORIES else "blue") for i, c in idx2cat.items()}


def arxiv_egos(rng):
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset(name="ogbn-arxiv")
    d = ds[0]
    lab2col = arxiv_label_colors(ds)
    y = d.y.view(-1).tolist()
    adj = [[] for _ in range(int(d.num_nodes))]
    for a, b in d.edge_index.t().tolist():
        if a != b:
            adj[a].append(b); adj[b].append(a)
    while True:
        seed = rng.randrange(len(adj))
        target = rng.randint(10, 40)
        seen, q = {seed}, deque([seed])
        while q and len(seen) < target:
            u = q.popleft()
            nb = adj[u][:]
            rng.shuffle(nb)
            for w in nb:
                if w not in seen and len(seen) < target:
                    seen.add(w); q.append(w)
        if len(seen) < 10:
            continue
        H = nx.Graph()
        idx = {u: k for k, u in enumerate(sorted(seen))}
        H.add_nodes_from(range(len(idx)))
        H.add_edges_from((idx[u], idx[w]) for u in seen for w in adj[u] if w in seen and u < w)
        inv = {k: u for u, k in idx.items()}
        for k in H:
            H.nodes[k]["color"] = lab2col[y[inv[k]]]
        yield H, f"arxiv:{seed}"


def synthetic_graphs(rng, source):
    """E2E_TINY stand-ins (pipeline test only): connected random graphs, sparse for molhiv, denser for arxiv."""
    while True:
        n = rng.randint(10, 40)
        G = nx.random_labeled_tree(n, seed=rng.randrange(2**31)) if hasattr(nx, "random_labeled_tree") else \
            nx.random_tree(n, seed=rng.randrange(2**31))
        extra = 2 if source == "molhiv" else n
        for _ in range(extra):
            u, w = rng.randrange(n), rng.randrange(n)
            if u != w:
                G.add_edge(u, w)
        G = nx.Graph(G)
        for u in G:
            G.nodes[u]["color"] = "red" if rng.random() < 0.3 else "blue"
        yield G, f"synthetic-{source}:{n}"


def pairs_from_graph(G, prop, rng):
    """One matched pair from G: two eligible queries with equal degree and opposite labels, or None."""
    by_deg = {}
    for v in G:
        if prop.eligible(G, v):
            by_deg.setdefault(G.degree(v), {True: [], False: []})[bool(prop.label(G, v))].append(v)
    opts = [d for d, b in by_deg.items() if b[True] and b[False]]
    if not opts:
        return None
    d = rng.choice(opts)
    return rng.choice(by_deg[d][True]), rng.choice(by_deg[d][False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["molhiv", "arxiv"], required=True)
    ap.add_argument("--pairs", type=int, default=300, help="pairs of D1a per source (E5 evaluates D1a only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if os.environ.get("E2E_TINY"):        # offline end-to-end test only: synthetic stand-ins, no OGB download
        gen = synthetic_graphs(rng, a.source)
        print("E2E_TINY: synthetic stand-in graphs (pipeline test only)")
    else:
        gen = molecules(rng) if a.source == "molhiv" else arxiv_egos(rng)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    need = {"D1a": a.pairs}          # E5: D1a only, so the prediction uses the D1a strata of E1 (like with like)
    stats = {"src": a.source, "n": [], "edges": [], "maxdeg": [], "meandeg": [], "cycles": [], "kmin_AL": [],
             "kmin_NC": []}
    with open(a.out, "w") as f:
        for G, src in gen:
            if not any(need.values()):
                break
            for pid in [p for p, k in need.items() if k > 0]:
                prop = BY_ID[pid]
                pr = pairs_from_graph(G, prop, rng)
                if pr is None:
                    continue
                gs = rng.randrange(2**31)
                stats["n"].append(G.number_of_nodes())
                stats["maxdeg"].append(max(dd for _, dd in G.degree))
                stats["meandeg"].append(2 * G.number_of_edges() / G.number_of_nodes())
                stats["edges"].append(G.number_of_edges())
                stats["cycles"].append(G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))
                for ser, off in (("AL", 202), ("NC", 303)):
                    km, _ = kmin_dmin(G, pr[0], ser, gs + off)
                    stats[f"kmin_{ser}"].append(km)
                for lab, v in zip((True, False), pr):
                    k = 2 * (a.pairs - need[pid]) + (0 if lab else 1)
                    f.write(json.dumps({
                        "id": f"{a.source}-{pid}-{k:05d}", "pid": pid, "cls": prop.cls, "label": lab, "query": v,
                        "n": G.number_of_nodes(), "edges": [list(e) for e in G.edges],
                        "colors": [G.nodes[u]["color"] for u in range(G.number_of_nodes())],
                        "pair": f"{a.source}-{pid}-p{k // 2:05d}", "pair_kind": "one_graph",
                        "deg_v": G.degree(v), "maxdeg": max(dd for _, dd in G.degree),
                        "graph_seed": gs, "source": src, "src": a.source}) + "\n")
                need[pid] -= 1
    json.dump(stats, open(a.out + ".stats.json", "w"))
    print(f"wrote {a.out}; missing pairs: {need}")


if __name__ == "__main__":
    main()
