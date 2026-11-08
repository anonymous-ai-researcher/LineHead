"""E2 positive controls of the name-blind experiment (CONFIG.md §2, E-3).

Query-vertex questions need no separate control set: in build.py every query pair is ONE graph with two query
vertices and one order choice, so after masking all names the two prompts are token-for-token identical. A
name-blind model therefore scores exactly 0.500 balanced accuracy on them (Theorem 1 and its distributional
version, paper appendix "Proofs for Name-Blind Models"); any deviation is an implementation error, not a finding.

Positive controls (graph properties that the theorem says ARE decidable name-blind under some formats):
  X2 "some vertex has degree >= 3": positive and negative share sigma_EL (#red, #blue, |E|) exactly;
     decidable under AL and NC, not under EL.
  X1 "some red vertex has a red neighbor": positive and negative share sigma_AL (multiset of (color, degree))
     exactly, via degree-preserving double edge swaps; decidable under NC only.
Graphs: n ~ U{10..40} as in E1. X2 negatives are disjoint unions of paths and cycles (max degree 2) with
m ~ U{ceil(n/2) .. n-1} edges; positives are G(n, m) graphs with a vertex of degree >= 3 and the same colors
permuted. X1 uses p_red ~ U[0.1, 0.25] and mean degree ~ U[1.5, 3] so that swaps can remove all red-red edges.

Usage: python build_blind.py --pairs 400 --seed 11 --out data/e2_controls.jsonl
"""
import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import networkx as nx

from tasks import BY_ID


def red(G, u):
    return G.nodes[u]["color"] == "red"


def sigma(G, kind):
    if kind == "EL":
        c = Counter(G.nodes[u]["color"] for u in G)
        return (c["red"], c["blue"], G.number_of_edges())
    if kind == "AL":
        return tuple(sorted((G.nodes[u]["color"], G.degree(u)) for u in G))
    if kind == "NC":
        return tuple(sorted((G.nodes[u]["color"], tuple(sorted(G.nodes[w]["color"] for w in G[u]))) for u in G))
    raise ValueError(kind)


def colorize(G, cols):
    for u, c in zip(sorted(G.nodes), cols):
        G.nodes[u]["color"] = c
    return G


def x1(G):
    return BY_ID["X1"].label(G, None)


def x2(G):
    return BY_ID["X2"].label(G, None)


def maxdeg2_graph(n, m, rng):
    for _ in range(200):
        G = nx.empty_graph(n)
        tries = 0
        while G.number_of_edges() < m and tries < 50 * n:
            tries += 1
            a, b = rng.sample(range(n), 2)
            if not G.has_edge(a, b) and G.degree(a) < 2 and G.degree(b) < 2:
                G.add_edge(a, b)
        if G.number_of_edges() == m:
            return G
    return None


def pair_x2(rng):
    while True:
        n = rng.randint(10, 40)
        m = rng.randint((n + 1) // 2, n - 1)
        Gn = maxdeg2_graph(n, m, rng)
        if Gn is None:
            continue
        for _ in range(200):
            Gp = nx.gnm_random_graph(n, m, seed=rng.randrange(2**31))
            if x2(Gp):
                break
        else:
            continue
        pr = rng.uniform(0.1, 0.4)
        cols = ["red" if rng.random() < pr else "blue" for _ in range(n)]
        colorize(Gp, rng.sample(cols, n)); colorize(Gn, rng.sample(cols, n))
        assert sigma(Gp, "EL") == sigma(Gn, "EL") and x2(Gp) and not x2(Gn)
        return Gp, Gn


def pair_x1(rng):
    while True:
        n = rng.randint(10, 40)
        m = max(n // 2, int(n * rng.uniform(1.5, 3.0) / 2))
        G = nx.gnm_random_graph(n, m, seed=rng.randrange(2**31))
        pr = rng.uniform(0.1, 0.25)
        colorize(G, ["red" if rng.random() < pr else "blue" for _ in range(n)])
        if sum(red(G, u) for u in G) < 2:
            continue
        H = G.copy()
        for _ in range(400):
            try:
                nx.double_edge_swap(H, nswap=1, max_tries=200, seed=rng.randrange(2**31))
            except nx.NetworkXException:
                break
            if x1(H) != x1(G):
                assert sigma(H, "AL") == sigma(G, "AL")
                return (G, H) if x1(G) else (H, G)


def rec(pid, G, lab, k, rng):
    return {"id": f"{pid}-{k:05d}", "pid": pid, "cls": "control", "label": lab, "query": None,
            "pair": f"{pid}-p{k // 2:05d}", "pair_kind": "two_graphs", "n": G.number_of_nodes(),
            "edges": [list(e) for e in G.edges], "colors": [G.nodes[u]["color"] for u in sorted(G.nodes)],
            "deg_v": None, "maxdeg": max((d for _, d in G.degree), default=0), "graph_seed": rng.randrange(2**31)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/e2_controls.jsonl")
    a = ap.parse_args()
    if os.environ.get("E2E_TINY"):               # offline end-to-end test only (a copy of the repository)
        a.pairs = min(a.pairs, 8)
    rng = random.Random(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for pid, fn in (("X2", pair_x2), ("X1", pair_x1)):
            for i in range(a.pairs):
                Gp, Gn = fn(rng)
                f.write(json.dumps(rec(pid, Gp, True, 2 * i, rng)) + "\n")
                f.write(json.dumps(rec(pid, Gn, False, 2 * i + 1, rng)) + "\n")
            print(pid, "done", flush=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
