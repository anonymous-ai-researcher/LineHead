"""E3 data (CONFIG.md §3.3): matched pairs for "v is red" (task red) and "v has a red neighbor" (task dia).

Each pair is ONE graph and ONE order choice with two query vertices of equal degree, so the two sequences differ
only in the final query token. Sequences are word-level tokens:
    <BOS> <serialization tokens> <NL> Q : RED @v        (task red)
    <BOS> <serialization tokens> <NL> Q : NBR @v        (task dia)
Names are written as "@name"; train_e3.py replaces every name by a frozen random vector drawn per sequence
(F-2), so name identity within a sequence is all a model can use. The label is read at the last token.

Splits
  train / id : n ~ U{8..16}, max degree <= 6 (rejection), p_red ~ U[0.2, 0.5], blue query with deg >= 1 (dia),
               query with deg >= 1 (red).
  ood        : n ~ U{17..64}, Delta ~ U{8..32}: a hub joined to Delta random vertices, then random edges between
               vertices of degree < Delta up to mean degree ~ U[2, 4].
  kset       : OOD graphs, dia only, positives binned by k_min (smallest position of v in the neighbor list of a
               red neighbor, under the AL/NC line order of the pair's order choice; AL and NC share line and
               neighbor orders for a given seed) into strata 1, 2, 3, 4-6, 7-12, 13-32; --pairs pairs per stratum,
               filled by rejection sampling. EL/ECL k-sets reuse the same graphs and order seeds (k_min is then a
               covariate only). Every record stores k_min and d_min (token distance from the entry to its head
               color: AL 2k+1, NC 5k-1).
               Position control (own-line routes, App. new:prop-rounds): the list order is the seed's order with the
               lines of the two query vertices moved forward, before every other line that names a red neighbor of v+
               (v+ first if v- has a red neighbor of v+ in its list, otherwise the two in random order); since 
               all lines that name no red neighbor of v+ precede the query lines, in random order, and zero or one
               of them separates the two; at least 11 (kset) or 5 (kid) of them precede the first query line, else
               the pair is rejected (Prop. lb:top: with both query lines on top, line-index formulas of round depth 4
               (AL) and 3 (NC) decided every pair; shortcut_check.py checks that they are at chance now), and in v+'s
               own list a red neighbor at position 1 is exchanged with the first blue neighbor (rejected if v+ has no
               blue neighbor). This blocks the AL own-line route and both NC own-line routes for every L <= 12; no
               listed construction other than the head route applies (checked by ownline_check.py). Positive records also store
               line_len (degree of the red neighbor that attains k_min) and the own-line access positions.
  kid        : as kset on in-distribution graphs, strata 1, 2, 3, 4-6 
  d256       : hub degree Delta in {64, 128, 192, 256} (uniform), n ~ U{Delta+1..Delta+64}; random orders; AL and NC, dia only;
               the hub is red and is the positive's only red neighbor, so v+ is read through a list of length Delta
               (the asymptotic regime of Theorem 8: no since-free formula of bounded depth succeeds at high degree)
  kown       : own-line set (P2b): OOD graphs, positives with k_min >= 4 (beyond the head route for L <= 8) whose
               own list starts with a red neighbor whose line precedes v+'s line (AL own-line route at position 1)
Usage
  python gen_e3.py --task dia --ser NC --split train --pairs 250000 --seed 0 --out data/dia_NC_train.jsonl
  python gen_e3.py --task dia --ser NC --split id    --pairs 5000   --seed 1 --out data/dia_NC_id.jsonl
  python gen_e3.py --task dia --ser NC --split ood   --pairs 5000   --seed 2 --out data/dia_NC_ood.jsonl
  python gen_e3.py --task dia --ser NC --split kset  --pairs 2000   --seed 3 --out data/dia_NC_kset.jsonl
"""
import argparse
import os
import json
import random
import re
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "m0"))
from serialize import kmin_dmin, list_positions, ownline_access, serialize  # noqa: E402

TOK = re.compile(r"[A-Za-z]+|--|\n|[^\sA-Za-z]")
TASK_TOKEN = {"red": "RED", "dia": "NBR"}
KSTRATA = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 12), (13, 32)]
KSTRATA_ID = [(1, 1), (2, 2), (3, 3), (4, 6)]


def name_pool(n=10_000, seed=12345):
    """Alphabetic CVCVC names; identity only matters, since embeddings are random per sequence."""
    rng = random.Random(seed)
    C, V = "bdfgklmnprstvz", "aeiou"
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(C if i % 2 == 0 else V) for i in range(5)))
    return sorted(out)


def colorize(G, rng, p=(0.2, 0.5)):
    pr = rng.uniform(*p)
    for u in G:
        G.nodes[u]["color"] = "red" if rng.random() < pr else "blue"
    return G


def graph_id(rng):
    """train / id: n ~ U{8..16}, Delta <= 6, mean degree ~ U[1.5, 4]."""
    while True:
        n = rng.randint(8, 16)
        m = max(n // 2, round(n * rng.uniform(1.5, 4.0) / 2))
        G = nx.gnm_random_graph(n, m, seed=rng.randrange(2**31))
        if max(d for _, d in G.degree) <= 6:
            return colorize(G, rng)


def graph_ood(rng):
    """ood: n ~ U{17..64}, Delta ~ U{8..32}."""
    while True:
        D = rng.randint(8, 32)
        n = rng.randint(max(17, D + 1), 64)
        G = nx.empty_graph(n)
        hub = rng.randrange(n)
        for w in rng.sample([u for u in range(n) if u != hub], D):
            G.add_edge(hub, w)
        target = int(n * rng.uniform(2.0, 4.0) / 2)
        tries = 0
        while G.number_of_edges() < target and tries < 50 * n:
            tries += 1
            a, b = rng.sample(range(n), 2)
            if not G.has_edge(a, b) and G.degree(a) < D and G.degree(b) < D:
                G.add_edge(a, b)
        if max(d for _, d in G.degree) == D:
            return colorize(G, rng)


def red(G, u):
    return G.nodes[u]["color"] == "red"


def candidates(G, task):
    by_deg = {}
    for u in G:
        if G.degree(u) == 0:
            continue
        if task == "red":
            lab = red(G, u)
        else:
            if red(G, u):
                continue
            lab = any(red(G, w) for w in G[u])
        by_deg.setdefault(G.degree(u), {True: [], False: []})[lab].append(u)
    return {d: s for d, s in by_deg.items() if s[True] and s[False]}


def encode(G, v, names, ser, task, seed, order=None):
    text = serialize(G, names, ser, seed=seed, order=order)
    nameset = set(names.values())
    toks = []
    for t in TOK.findall(text):
        if t == "\n":
            toks.append("<NL>")
        elif t in nameset:
            toks.append("@" + t)
        else:
            toks.append(t)
    return ["<BOS>"] + toks + ["<NL>", "Q", ":", TASK_TOKEN[task], "@" + names[v]]


def stratum_of(k, strata=KSTRATA):
    for i, (lo, hi) in enumerate(strata):
        if lo <= k <= hi:
            return i
    return None


# a formula that finds the query's line by counting line breaks needs about one round per
# line, so a query line with index m is reachable by line-index formulas of round depth about m+1. The first query
# line therefore follows at least L_max - 1 lines, where L_max is the largest depth tested on the set: 12 for kset
# (depth sweep), 6 for kid (main runs); pairs whose graph has fewer such lines are rejected.
MIN_PRE = {"kset": 11, "kid": 5}


def blocked_order(G, vp, vn, s, rng, min_pre=0):
    """Position-controlled list order (kset, kid): see the module docstring. None if v+ has no blue neighbor."""
    lp = [(u, list(ws)) for u, ws in list_positions(G, "AL", s)]
    nb = dict(lp)
    own = nb[vp]
    if red(G, own[0]):
        blue = [i for i, u in enumerate(own) if not red(G, u)]
        if not blue:
            return None
        i = blue[0]
        own[0], own[i] = own[i], own[0]
    reds = {u for u in G[vp] if red(G, u)}
    pair_ = [vp, vn] if (reds & set(G[vn]) or rng.random() < 0.5) else [vn, vp]
    # (proofs12, Prop. lb:top): the query lines are no longer on top. Every line that names no red neighbor
    # of v+ (neither as head nor as entry) precedes them, in random order, so (B1) still holds and the query lines
    # sit as deep as the graph allows; zero or one of these lines separates the two query lines. The pair shares one
    # serialization and the two query lines are adjacent up to that filler, so the line index carries no
    # information on which query is positive, and a formula that locates the query's line by counting lines needs
    # round depth about its index (shortcut_check.py).
    free = [u for u, ws in lp if u not in pair_ and u not in reds and not (reds & set(ws))]
    rng.shuffle(free)
    f1 = rng.randint(0, min(1, len(free)))
    pre, mid = free[f1:], free[:f1]
    if len(pre) < min_pre:              # the query lines must follow at least min_pre lines (see MIN_PRE)
        return None
    head = pre + [pair_[0]] + mid + [pair_[1]]
    order = [(u, nb[u]) for u in head] + [(u, nb[u]) for u, _ in lp if u not in head]
    acc = ownline_access(G, vp, order)
    assert acc["j_prec"] is None and acc["j_occ"] is None and acc["j_inl"] >= 2, acc
    return order


def ownline_order(G, vp, s, rng):
    """kown: a red neighbor u of v+ at position 1 of v+'s list and u's line before v+'s line."""
    lp = [(u, list(ws)) for u, ws in list_positions(G, "AL", s)]
    nb = dict(lp)
    u = rng.choice(sorted(w for w in G[vp] if red(G, w)))
    nb[vp].remove(u)
    nb[vp].insert(0, u)
    heads = [h for h, _ in lp]
    if heads.index(u) > heads.index(vp):
        heads.remove(vp)
        heads.insert(heads.index(u) + 1, vp)
    order = [(h, nb[h]) for h in heads]
    assert ownline_access(G, vp, order)["j_prec"] == 1
    return order


def write_pair(f, G, vp, vn, names, ser, task, s, k, split, order=None):
    md = max(d for _, d in G.degree)
    km, dm = kmin_dmin(G, vp, ser if ser in ("AL", "NC") else "AL", s) if task == "dia" else (None, None)
    extra = {}
    if task == "dia":
        lp = order or list_positions(G, "AL", s)
        att = [(ws.index(vp) + 1, len(ws)) for u, ws in lp if red(G, u) and vp in ws]
        extra = dict(line_len=min(att)[1] if att else None, **ownline_access(G, vp, lp))
    if task == "dia" and ser in ("AL", "NC"):
        dmin = dm
    elif task == "dia":                       # EL / ECL: k_min of the AL order as a covariate, no head distance
        dmin = None
    else:
        km = dmin = None
    for v, lab in ((vp, 1), (vn, 0)):
        rec = {"tokens": encode(G, v, names, ser, task, s, order), "label": lab, "pair": k, "split": split,
               "n": G.number_of_nodes(), "maxdeg": md, "deg_v": G.degree(v),
               "k_min": km if lab else None, "d_min": dmin if lab else None}
        if lab:
            rec.update(extra)
        f.write(json.dumps(rec) + "\n")


def graph_d256(rng):
    """d256 (hub-degree strata): a hub of degree Delta drawn uniformly from {64, 128, 192, 256} (so a
    quarter of the hubs exceed the degree 247 that Theorem 8 names for six rounds), n ~ U{Delta+1..Delta+64}, other
    edges as in ood (mean degree ~ U[2, 4], degrees below Delta)."""
    while True:
        D = rng.choice((64, 128, 192, 256))
        n = rng.randint(D + 1, D + 64)
        G = nx.empty_graph(n)
        hub = rng.randrange(n)
        for w in rng.sample([u for u in range(n) if u != hub], D):
            G.add_edge(hub, w)
        target = int(n * rng.uniform(2.0, 4.0) / 2)
        tries = 0
        while G.number_of_edges() < target and tries < 50 * n:
            tries += 1
            a, b = rng.sample(range(n), 2)
            if not G.has_edge(a, b) and G.degree(a) < D and G.degree(b) < D:
                G.add_edge(a, b)
        if max(d for _, d in G.degree) == D:
            return colorize(G, rng)


def pairs(task, split, seed, n, pool=None):
    """The pairs of one (task, split, seed), in file order: yields (G, v+, v-, names, order seed, list order or None,
    split label). Shared by main() and kdist.py, which therefore measures
    exactly the instances of the released test sets; the k_min distribution of Table T_kdist is the generator's."""
    pool = pool or name_pool()
    # graphs depend on (task, split, seed) only, so the four serializations see the same graphs and orders
    rng = random.Random(f"{task}-{split}-{seed}")
    if split not in ("kset", "kid", "kown"):
        gen = {"ood": graph_ood, "d256": graph_d256}.get(split, graph_id)
        k = 0
        while k < n:
            G = gen(rng)
            c = candidates(G, task)
            if split == "d256":            # the hub is red and is the positive's only red neighbor (Theorem 8)
                hub = max(G.nodes, key=G.degree)
                if G.nodes[hub]["color"] != "red":
                    continue
                c = {d_: {True: [v for v in b[True] if [u for u in G[v] if G.nodes[u]["color"] == "red"] == [hub]],
                          False: b[False]} for d_, b in c.items()}
                c = {d_: b for d_, b in c.items() if b[True] and b[False]}
            if not c:
                continue
            d = rng.choice(sorted(c))
            vp, vn = rng.choice(c[d][True]), rng.choice(c[d][False])
            names = dict(zip(G.nodes, rng.sample(pool, G.number_of_nodes())))
            s = rng.randrange(2**31)
            yield G, vp, vn, names, s, None, split
            k += 1
    elif split == "kown":
        k = 0
        while k < n:
            G = graph_ood(rng)
            c = candidates(G, "dia")
            if not c:
                continue
            s = rng.randrange(2**31)
            opts = [(dd, u) for dd in sorted(c) for u in c[dd][True] if kmin_dmin(G, u, "AL", s)[0] >= 4]
            if not opts:
                continue
            dd, vp = rng.choice(opts)
            vn = rng.choice(c[dd][False])
            names = dict(zip(G.nodes, rng.sample(pool, G.number_of_nodes())))
            yield G, vp, vn, names, s, ownline_order(G, vp, s, rng), "kown"
            k += 1
    else:
        strata = KSTRATA if split == "kset" else KSTRATA_ID
        gen = graph_ood if split == "kset" else graph_id
        need = [n] * len(strata)
        while any(need):
            G = gen(rng)
            c = candidates(G, "dia")
            if not c:
                continue
            s = rng.randrange(2**31)
            # all positive candidates (any degree with a matching negative), binned by k_min
            opts = [(stratum_of(kmin_dmin(G, u, "AL", s)[0], strata), dd, u) for dd in sorted(c) for u in c[dd][True]]
            opts = [o for o in opts if o[0] is not None and need[o[0]] > 0]
            if not opts:
                continue
            top = max(o[0] for o in opts)                  # prefer the rarest (largest-k) stratum still needed
            st, dd, vp = rng.choice([o for o in opts if o[0] == top])
            vn = rng.choice(c[dd][False])
            order = blocked_order(G, vp, vn, s, rng, MIN_PRE[split])
            if order is None:
                continue
            names = dict(zip(G.nodes, rng.sample(pool, G.number_of_nodes())))
            yield G, vp, vn, names, s, order, f"{split}{st}"
            need[st] -= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["red", "dia"], required=True)
    ap.add_argument("--ser", choices=["EL", "AL", "NC", "ECL", "NCH", "ELB"], required=True)
    ap.add_argument("--split", choices=["train", "val", "id", "ood", "kset", "kid", "kown", "d256"], required=True,
                    help="val: in-distribution validation split, disjoint from the ID test set (own seed)")
    ap.add_argument("--pairs", type=int, required=True, help="pairs (per stratum for kset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("E2E_TINY"):        # offline end-to-end test only
        a.pairs = min(a.pairs, 3 if a.split in ("kset", "kid", "kown") else 9 if a.split == "d256" else 12)   # d256: 9 pairs cover all four hub degrees
    if a.split in ("kset", "kid", "kown") and a.task != "dia":
        raise SystemExit("kset, kid and kown are defined for task dia only")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for k, (G, vp, vn, names, s, order, lab) in enumerate(pairs(a.task, a.split, a.seed, a.pairs)):
            write_pair(f, G, vp, vn, names, a.ser, a.task if lab == a.split else "dia", s, k, lab, order=order)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
