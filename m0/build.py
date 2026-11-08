"""Build the LLM instance set (E1; CONFIG.md §1.3, §5.4-5.5; F-3).

Global questions : pairs of two different graphs with equal (|V|, |E|).
Query questions  : ONE graph and two query vertices with opposite labels, matched on the query key of
                   tasks.py (deg v; for depth two also #red neighbors and the sum of neighbor degrees).
                   Both instances share the graph, its names (graph_seed) and its order choices, so every
                   graph-level statistic is shared exactly and the two prompts differ only in the query name.
Graphs: G(n, p) with n ~ U{10..40}, expected degree ~ U[2, 8], p_red ~ U[0.1, 0.4]
(CONFIG §5.5). With --colors 3 (E8) every vertex is red with p_red and otherwise blue or green with equal
probability; labels still concern red.

Usage:
  python build.py --pairs 400 --seed 0 --out data/m0_instances.jsonl
  python build.py --pairs 400 --seed 7 --out data/demo_instances.jsonl          # E6 demonstrations
  python build.py --pairs 200 --seed 3 --colors 3 --out data/e8_colors3.jsonl --pids D1a,D1b,D1c,G1,G2,G3
  python build.py --pairs 400 --seed 5 --pids Q0,D1a --out data/e7_instances.jsonl
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx

from names import POOL as NAME_POOL   # noqa: F401  (re-exported for older scripts)
from tasks import BY_ID, PROPS


def random_graph(rng, n_min=10, n_max=40, mean_deg=(2.0, 8.0), p_red=(0.1, 0.4), colors=2):
    n = rng.randint(n_min, n_max)
    d = rng.uniform(*mean_deg)
    G = nx.gnp_random_graph(n, min(1.0, d / (n - 1)), seed=rng.randrange(2**31))
    pr = rng.uniform(*p_red)
    for u in G:
        if rng.random() < pr:
            G.nodes[u]["color"] = "red"
        else:
            G.nodes[u]["color"] = "blue" if colors == 2 or rng.random() < 0.5 else "green"
    return G


def record(prop, G, v, lab, pair_id, k, graph_seed, kind):
    return {"id": f"{prop.pid}-{k:05d}", "pid": prop.pid, "cls": prop.cls, "label": bool(lab),
            "query": v, "pair": f"{prop.pid}-p{pair_id:05d}", "pair_kind": kind,
            "n": G.number_of_nodes(), "edges": [list(e) for e in G.edges],
            "colors": [G.nodes[u]["color"] for u in range(G.number_of_nodes())],
            "deg_v": G.degree(v) if v is not None else None,
            "maxdeg": max((d for _, d in G.degree), default=0), "graph_seed": graph_seed}


def pairs_two_graphs(prop, n_pairs, rng, colors, max_draws=3_000_000):
    pools = defaultdict(lambda: {True: [], False: []})
    out, draws = [], 0
    while len(out) < n_pairs and draws < max_draws:
        draws += 1
        G = random_graph(rng, colors=colors)
        key, lab = prop.graph_key(G), bool(prop.label(G, None))
        pools[key][lab].append(G)
        if pools[key][True] and pools[key][False]:
            out.append((pools[key][True].pop(), None, pools[key][False].pop(), None))
    return out, draws


def pairs_one_graph(prop, n_pairs, rng, colors, max_draws=3_000_000):
    out, draws = [], 0
    while len(out) < n_pairs and draws < max_draws:
        draws += 1
        G = random_graph(rng, colors=colors)
        by = defaultdict(lambda: {True: [], False: []})
        for v in G:
            if prop.eligible(G, v):
                by[prop.query_key(G, v)][bool(prop.label(G, v))].append(v)
        keys = [k for k, b in by.items() if b[True] and b[False]]
        if keys:
            k = rng.choice(sorted(keys))
            out.append((G, rng.choice(by[k][True]), G, rng.choice(by[k][False])))
    return out, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--colors", type=int, choices=[2, 3], default=2)
    ap.add_argument("--pids", default=None, help="comma-separated question ids (default: the 12 questions)")
    ap.add_argument("--out", default="data/m0_instances.jsonl")
    a = ap.parse_args()
    if os.environ.get("E2E_TINY"):               # offline end-to-end test only (a copy of the repository)
        a.pairs = min(a.pairs, 8)
    props = [BY_ID[p] for p in a.pids.split(",")] if a.pids else PROPS
    rng = random.Random(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w") as f:
        for prop in props:
            t0 = time.time()
            if prop.needs_query:
                pairs, draws = pairs_one_graph(prop, a.pairs, rng, a.colors)
                kind = "one_graph"
            else:
                pairs, draws = pairs_two_graphs(prop, a.pairs, rng, a.colors)
                kind = "two_graphs"
            if len(pairs) < a.pairs:
                raise RuntimeError(f"{prop.pid}: only {len(pairs)} pairs after {draws} draws")
            for i, (Gp, vp, Gn, vn) in enumerate(pairs):
                gs_p = rng.randrange(2**31)
                gs_n = gs_p if kind == "one_graph" else rng.randrange(2**31)
                f.write(json.dumps(record(prop, Gp, vp, True, i, 2 * i, gs_p, kind)) + "\n")
                f.write(json.dumps(record(prop, Gn, vn, False, i, 2 * i + 1, gs_n, kind)) + "\n")
                n += 2
            print(f"{prop.pid:4s} {prop.cls:7s} pairs={len(pairs)} draws={draws} {time.time() - t0:.1f}s", flush=True)
    print(f"wrote {n} instances to {a.out}")


if __name__ == "__main__":
    main()
