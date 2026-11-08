"""E3 data (CONFIG.md §6.14: E22, E26, E27). Same token format, name pool and serializations as gen_e3.py.

Tasks
  g1  (E22)  "some vertex is red" (global, no query name):  <BOS> <serialization> <NL> Q : ANY
             pair = one graph and one order choice; the positive has exactly one red vertex, the negative is the same
             graph with that vertex blue (all vertices blue). Decided by P(red) in one round under every format.
  c3  (E22)  "at least three vertices are red" (global):  ... Q : THREE
             positive exactly three red, negative the same graph with one of them blue (exactly two). Needs a
             threshold P^{>=3}; NoPE decides it (counting by a global head), so it controls for E22's reading.
  dd  (E27)  "v has a neighbor that has a red neighbor other than v" (two hops):  ... Q : NBR2 @v
             blue query v, the pair's two queries have equal degree, max degree <= 3 (Delta <= 3; Corollary
             ehat:window predicts the D-local frontier for depth-two questions only through the rounds).
  c2  (E27)  "v has at least two red neighbors":  ... Q : TWO @v
             positive exactly two red neighbors, negative exactly one, equal degree, Delta <= 3.
Test split pm<E> (E26, task dia only): a perfect matching with E edges (2E vertices, every degree 1); the queries are
  two blue vertices whose partners are red (positive) and blue (negative). Every vertex has degree 1, so the
  match-nesting-depth bound of Theorem md:el is the only obstacle for EL; the arm lastglobal has one matching layer.
Training split pmtrain (E26): perfect matchings with E ~ U{8..32} edges; r20_e3.sh mixes it half and half with the
  E3 dia training set.
Splits: g1 / c3 use the graphs of E3 (gen_e3.graph_id for train / val / id, graph_ood for ood); dd / c2 use
  n ~ U{8..16} (train / val / id) and n ~ U{17..40} (ood), Delta <= 3 in every split. Graphs depend on
  (task, split, seed) only, so the serializations share them.
Usage
  python gen_r20.py --task g1 --ser AL --split train --pairs 250000 --seed 0 --out data/g1_AL_train.jsonl
  python gen_r20.py --task dia --ser EL --split pm32 --pairs 2000 --seed 7 --out data/dia_EL_pm32.jsonl
"""
import argparse
import json
import os
import random
from pathlib import Path

import networkx as nx

from gen_e3 import TOK, colorize, graph_id, graph_ood, name_pool, red, serialize

TASK_TOKEN = {"g1": "ANY", "c3": "THREE", "dd": "NBR2", "c2": "TWO", "dia": "NBR"}
GLOBAL = ("g1", "c3")


def tokens(G, names, ser, task, seed, v=None):
    text = serialize(G, names, ser, seed=seed)
    nameset = set(names.values())
    out = ["<BOS>"]
    for t in TOK.findall(text):
        out.append("<NL>" if t == "\n" else "@" + t if t in nameset else t)
    out += ["<NL>", "Q", ":", TASK_TOKEN[task]]
    return out + (["@" + names[v]] if v is not None else [])


def graph(rng, split, maxdeg):
    lo, hi = (17, 40) if split == "ood" else (8, 16)
    while True:
        n = rng.randint(lo, hi)
        m = max(n // 2, round(n * rng.uniform(1.5, min(4.0, maxdeg)) / 2))
        G = nx.gnm_random_graph(n, m, seed=rng.randrange(2**31))
        if max(d for _, d in G.degree) <= maxdeg:
            return colorize(G, rng)


def matching(rng, E):
    G = nx.Graph()
    G.add_nodes_from(range(2 * E))
    perm = rng.sample(range(2 * E), 2 * E)
    G.add_edges_from((perm[2 * i], perm[2 * i + 1]) for i in range(E))
    return colorize(G, rng)


def two_hop(G, v):
    return any(red(G, x) for w in G[v] for x in G[w] if x != v)


def recolor(G, reds):
    H = G.copy()
    for u in H:
        H.nodes[u]["color"] = "red" if u in reds else "blue"
    return H


def pairs(task, split, seed, n, pool=None):
    """Yields (G+, G-, v+, v-, names, order seed): G+ = G- except in the global tasks (one vertex recolored)."""
    pool = pool or name_pool()
    rng = random.Random(f"r20-{task}-{split}-{seed}")
    k = 0
    while k < n:
        if split.startswith("pm"):
            G = matching(rng, rng.randint(8, 32) if split == "pmtrain" else int(split[2:]))
            pos = [u for u in G if not red(G, u) and red(G, next(iter(G[u])))]
            neg = [u for u in G if not red(G, u) and not red(G, next(iter(G[u])))]
            if not pos or not neg:
                continue
            Gp = Gn = G
            vp, vn = rng.choice(pos), rng.choice(neg)
        elif task in GLOBAL:
            G = (graph_ood if split == "ood" else graph_id)(rng)
            want = 1 if task == "g1" else 3
            if G.number_of_nodes() < want:
                continue
            reds = set(rng.sample(sorted(G), want))
            Gp = recolor(G, reds)
            Gn = recolor(G, reds - {rng.choice(sorted(reds))})
            vp = vn = None
        else:
            G = graph(rng, split, 3)
            by = {}
            for u in G:
                if red(G, u) or G.degree(u) == 0:
                    continue
                nr = sum(red(G, w) for w in G[u])
                lab = two_hop(G, u) if task == "dd" else nr >= 2 if nr in (1, 2) else None
                if task == "c2" and lab is None:
                    continue
                by.setdefault(G.degree(u), {True: [], False: []})[lab].append(u)
            by = {d: b for d, b in by.items() if b[True] and b[False]}
            if not by:
                continue
            d = rng.choice(sorted(by))
            vp, vn = rng.choice(by[d][True]), rng.choice(by[d][False])
            Gp = Gn = G
        names = dict(zip(Gp.nodes, rng.sample(pool, Gp.number_of_nodes())))
        yield Gp, Gn, vp, vn, names, rng.randrange(2**31)
        k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["g1", "c3", "dd", "c2", "dia"], required=True)
    ap.add_argument("--ser", choices=["EL", "AL", "NC", "ECL", "NCH", "ELB"], required=True)
    ap.add_argument("--split", required=True, help="train, val, id, ood, or pmtrain / pm<E> (task dia)")
    ap.add_argument("--pairs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.split.startswith("pm") != (a.task == "dia"):
        raise SystemExit("pm<E> splits go with task dia, and task dia here only with pm<E>")
    if os.environ.get("E2E_TINY"):
        a.pairs = min(a.pairs, 12)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for k, (Gp, Gn, vp, vn, names, s) in enumerate(pairs(a.task, a.split, a.seed, a.pairs)):
            for G, v, lab in ((Gp, vp, 1), (Gn, vn, 0)):
                rec = {"tokens": tokens(G, names, a.ser, a.task, s, v), "label": lab, "pair": k, "split": a.split,
                       "n": G.number_of_nodes(), "maxdeg": max(d for _, d in G.degree),
                       "deg_v": G.degree(v) if v is not None else None, "k_min": None, "d_min": None}
                f.write(json.dumps(rec) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
