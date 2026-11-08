"""Twin-graph Q0 pairs for the AL heads-masked condition of E7 (Proposition new:prop-mask).

A pair is two colorings of one graph G in which the query vertex v has a twin w (N(w) = N(v), v and w not
adjacent): in G, v is red and w blue (label Yes for "Is node v red?"); in G', the colors of v and w are exchanged
(label No). The order of G' is that of G with the LINES of v and w exchanged and every neighbor list unchanged
(v's and w's own lists are identical). With line-head names masked, the AL prompts of the two instances are then
identical token by token: the line at v's position reads "kit (red): L" in G (v's line) and in G' (w's line), and
every entry is the same name at the same position. A heads-masked model therefore answers both alike and its
balanced accuracy is exactly 0.5 by construction, as for the name-blind control (V-B1); the unmasked prompts
differ (v's head color), so the unmasked and entries-masked conditions remain informative.

Graphs as E1 (G(n, p), n ~ U{10..40}, expected degree ~ U[2, 8], p_red ~ U[0.1, 0.4]); w is a vertex other than v
whose edges are replaced by edges to N(v). Every record stores the explicit list order ("order") used by
serialize.py for the list formats; edge formats are not run on these pairs.
Usage: python build_twin.py --pairs 400 --seed 11 --out data/e7_twin.jsonl
"""
import argparse
import json
import os
import random
from pathlib import Path

import networkx as nx

from instances import MASK_WORD, SER_OFFSET, prompts_for
from serialize import list_positions


def make_pair(rng, k):
    while True:
        n = rng.randint(10, 40)
        deg = rng.uniform(2.0, 8.0)
        G = nx.gnp_random_graph(n, min(1.0, deg / (n - 1)), seed=rng.randrange(2**31))
        cand = [u for u in G if G.degree(u) >= 1]
        if len(cand) < 2:
            continue
        v = rng.choice(cand)
        others = [u for u in G if u != v and u not in G[v]]
        if not others:
            continue
        w = rng.choice(others)
        G.remove_edges_from(list(G.edges(w)))
        G.add_edges_from((w, x) for x in list(G[v]) if x != w)
        if set(G[v]) != set(G[w]) or G.has_edge(v, w):
            continue
        p_red = rng.uniform(0.1, 0.4)
        col = ["red" if rng.random() < p_red else "blue" for _ in range(n)]
        col[v], col[w] = "red", "blue"
        gseed = rng.randrange(2**31)
        Gc = G.copy()
        for u in Gc:
            Gc.nodes[u]["color"] = col[u]
        lp = list_positions(Gc, "AL", gseed + SER_OFFSET["AL"])
        nb = dict(lp)
        nb[w] = list(nb[v])                                     # identical own lists
        order = [(u, nb[u]) for u, _ in lp]
        heads = [u for u, _ in order]
        iv, iw = heads.index(v), heads.index(w)
        order2 = list(order)
        order2[iv], order2[iw] = order[iw], order[iv]           # exchange the lines of v and w
        col2 = list(col)
        col2[v], col2[w] = col[w], col[v]
        base = dict(pid="Q0", cls="diag", query=v, pair=f"Q0t-p{k:05d}", pair_kind="twin", n=n,
                    edges=[list(e) for e in G.edges], deg_v=G.degree(v), maxdeg=max(d for _, d in G.degree),
                    graph_seed=gseed, src="twin")
        pos = dict(base, id=f"Q0t-{2 * k:05d}", label=True, colors=col, order=[[u, ws] for u, ws in order])
        neg = dict(base, id=f"Q0t-{2 * k + 1:05d}", label=False, colors=col2, order=[[u, ws] for u, ws in order2])
        a = [p["prompt"] for p in prompts_for(pos, serializations=("AL",), mask="heads")]
        b = [p["prompt"] for p in prompts_for(neg, serializations=("AL",), mask="heads")]
        u0 = [p["prompt"] for p in prompts_for(pos, serializations=("AL",))]
        u1 = [p["prompt"] for p in prompts_for(neg, serializations=("AL",))]
        assert a == b and u0 != u1 and MASK_WORD in a[0], "twin construction failed"
        return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("E2E_TINY"):
        a.pairs = min(a.pairs, 8)
    rng = random.Random(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for k in range(a.pairs):
            for r in make_pair(rng, k):
                f.write(json.dumps(r) + "\n")
    print("wrote", a.out, 2 * a.pairs, "instances (heads-masked AL prompts of each pair identical: checked)")


if __name__ == "__main__":
    main()
