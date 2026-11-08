"""Matched pairs for pilot 1 (attainability of the ID-blind information bound).

Statistics a serialization exposes when every node name is masked (Theorem 1):
  sigma_EL(G) = (#red, #blue, |E|)
  sigma_AL(G) = multiset of (color, degree)
  sigma_NC(G) = multiset of (color, multiset of neighbor colors)

Tasks, each built so that positive and negative share the statistic that the
theorem says is NOT sufficient, making chance level a sharp prediction:
  G1  "is some node red?"                        pairs share n and |E|
  X2  "has some node degree >= 3?"               pairs share sigma_EL exactly
  X1  "has some red node a red neighbor?"        pairs share sigma_AL exactly (degree-preserving swaps)
  D1a "has query node v (blue) a red neighbor?"  pairs are the same graph with two different queries
Predicted (ID-blind):   EL  AL  NC
  G1                    yes yes yes
  X2                    -   yes yes
  X1                    -   -   yes
  D1a                   -   -   -
"""
import random
from collections import Counter

import networkx as nx


def red(G, u):
    return G.nodes[u]["color"] == "red"


def sigma(G, kind):
    if kind == "EL":
        c = Counter(G.nodes[u]["color"] for u in G)
        return (c["red"], c["blue"], G.number_of_edges())
    if kind == "AL":
        return tuple(sorted((G.nodes[u]["color"], G.degree(u)) for u in G))
    if kind == "NC":
        return tuple(sorted((G.nodes[u]["color"],
                             tuple(sorted(G.nodes[w]["color"] for w in G[u]))) for u in G))
    raise ValueError(kind)


def colorize(G, colors):
    for u, c in zip(G.nodes, colors):
        G.nodes[u]["color"] = c
    return G


def rand_colors(n, rng, p=(0.2, 0.45)):
    pr = rng.uniform(*p)
    return ["red" if rng.random() < pr else "blue" for _ in range(n)]


def x1(G):
    return any(red(G, u) and any(red(G, w) for w in G[u]) for u in G)


def x2(G):
    return max((d for _, d in G.degree), default=0) >= 3


def pair_G1(rng):
    n = rng.randint(5, 9)
    m = rng.randint(n - 2, min(2 * n, n * (n - 1) // 2))
    seed = rng.randrange(2**31)
    Gp = colorize(nx.gnm_random_graph(n, m, seed=seed), rand_colors(n, rng))
    if not any(red(Gp, u) for u in Gp):
        Gp.nodes[rng.randrange(n)]["color"] = "red"
    Gn = colorize(nx.gnm_random_graph(n, m, seed=rng.randrange(2**31)), ["blue"] * n)
    return (Gp, None, True), (Gn, None, False)


def _maxdeg2_graph(n, m, rng):
    G = nx.empty_graph(n)
    tries = 0
    while G.number_of_edges() < m and tries < 10_000:
        tries += 1
        a, b = rng.sample(range(n), 2)
        if not G.has_edge(a, b) and G.degree(a) < 2 and G.degree(b) < 2:
            G.add_edge(a, b)
    return G if G.number_of_edges() == m else None


def pair_X2(rng):
    while True:
        n = rng.randint(5, 9)
        m = rng.randint(n // 2, n - 1)
        Gn = _maxdeg2_graph(n, m, rng)
        if Gn is None:
            continue
        for _ in range(200):
            Gp = nx.gnm_random_graph(n, m, seed=rng.randrange(2**31))
            if x2(Gp):
                break
        else:
            continue
        cols = rand_colors(n, rng)
        colorize(Gp, rng.sample(cols, n))
        colorize(Gn, rng.sample(cols, n))
        assert sigma(Gp, "EL") == sigma(Gn, "EL") and x2(Gp) and not x2(Gn)
        return (Gp, None, True), (Gn, None, False)


def pair_X1(rng):
    while True:
        n = rng.randint(5, 9)
        m = rng.randint(n, min(2 * n, n * (n - 1) // 2))
        G = colorize(nx.gnm_random_graph(n, m, seed=rng.randrange(2**31)), rand_colors(n, rng))
        if sum(red(G, u) for u in G) < 2 or m < 2:
            continue
        H = G.copy()
        for _ in range(60):                       # degree-preserving swaps keep sigma_AL
            try:
                nx.double_edge_swap(H, nswap=1, max_tries=100, seed=rng.randrange(2**31))
            except nx.NetworkXException:
                break
            if x1(H) != x1(G):
                assert sigma(H, "AL") == sigma(G, "AL")
                return ((G, None, x1(G)), (H, None, x1(H))) if x1(G) else ((H, None, True), (G, None, False))
        continue


def pair_D1a(rng):
    while True:
        n = rng.randint(5, 9)
        m = rng.randint(n - 1, min(2 * n, n * (n - 1) // 2))
        G = colorize(nx.gnm_random_graph(n, m, seed=rng.randrange(2**31)), rand_colors(n, rng))
        blue = [u for u in G if not red(G, u) and G.degree(u) >= 1]
        pos = [u for u in blue if any(red(G, w) for w in G[u])]
        neg = [u for u in blue if not any(red(G, w) for w in G[u])]
        if pos and neg:
            return (G, rng.choice(pos), True), (G, rng.choice(neg), False)


PAIRS = {"G1": pair_G1, "X2": pair_X2, "X1": pair_X1, "D1a": pair_D1a}
QUESTION = {
    "G1": "is some node red ?",
    "X2": "does some node have at least three neighbors ?",
    "X1": "does some red node have a red neighbor ?",
    "D1a": "does node {v} have a red neighbor ?",
}
