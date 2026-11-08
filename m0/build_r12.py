"""Instance files (CONFIG.md §6.2-§6.5).

e11b  Degree sweep (E11b): graphs with maximum degree D in {6, 12, 24, 48} and two hubs of degree D (hub_graph). A
      pair is ONE graph with two blue query vertices of equal query key (degree and degree sum of the neighbors, as
      in E1): v+ a neighbor of the red hub (its only red neighbor) and v- a neighbor of the blue hub (no red neighbor). Positives are kept only if v+ sits late in the hub's list under both list formats
      (k_min >= max(3, D/2) for AL and NC, with the order seeds the prompts use), so the list formats need a long
      since step. Questions D1a and D1c, 150 pairs per (D, question). n <= 2D + 18 keeps every prompt (EL, AL, NC,
      ECL; templates T1-T3) under Gemma-2's 4,096-token window (checked: --max-words, word-level count x 1.6).
e12   Since probe (E12): AL- and NC-style lists in which a query name occurs once, as the k-th entry of some line;
      question: "Is the color written at the head of the line that lists node X red?". A pair is ONE prompt with
      the head color red vs blue (everything else equal). k in {1, 2, 3, 4, 6, 8, 12, 16, 24}, 60 pairs per (format,
      k). r(d) = balanced accuracy by word distance d from the entry to its head color (AL 2k+1, NC 5k-1).
e13   GraphQA node tasks of Fatemi et al. (2024) (E13): node degree (generated integer), edge existence (yes/no,
      matched pairs: an edge and a non-edge of the same graph), node neighbors (generated list, exact set
      match). Graphs: G(n, p), n ~ U{5..20}, p ~ U[0.1, 0.5], as in GraphQA; 200 questions per task.
      (Records keep the colors field so the serializers apply unchanged; colors are not asked about.)
Usage: python build_r12.py e11b --out data/e11b_instances.jsonl
       python build_r12.py e12  --out data/e12_probe.jsonl
       python build_r12.py e13  --out data/e13_graphqa.jsonl
"""
import argparse
import json
import os
import random
from pathlib import Path

import networkx as nx

from instances import SER_OFFSET
from serialize import kmin_dmin, serialize
from tasks import BY_ID

DEGREES = (6, 12, 24, 48)
KS = (1, 2, 3, 4, 6, 8, 12, 16, 24)


def tiny():
    return bool(os.environ.get("E2E_TINY"))


def hub_graph(rng, D):
    """Two hubs of degree D with disjoint neighbor sets: hub 0 red, hub 1 blue; every other vertex blue; a few
    extra edges among the non-hub vertices (degrees stay below D). A positive query vertex is a neighbor of the red
    hub, the negative one a neighbor of the blue hub, so the pair matches on degree and on the degree sum of the
    neighbors (the E1 query key, verify_data.py) and differs in the hub's color only."""
    extra = rng.randint(4, 16)
    n = 2 * D + 2 + extra
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rest = list(range(2, n))
    rng.shuffle(rest)
    A, B = rest[:D], rest[D:2 * D]
    G.add_edges_from((0, u) for u in A)
    G.add_edges_from((1, u) for u in B)
    tries = 0
    while G.number_of_edges() < 2 * D + n // 3 and tries < 50 * n:
        tries += 1
        a, b = rng.sample(rest, 2)
        if not G.has_edge(a, b) and G.degree(a) < D - 1 and G.degree(b) < D - 1:
            G.add_edge(a, b)
    for u in G:
        G.nodes[u]["color"] = "red" if u == 0 else "blue"
    return G


def rec(pid, cls, G, v, lab, pair, k, gs, extra=None):
    r = {"id": f"{pid}-{k:05d}", "pid": pid, "cls": cls, "label": bool(lab), "query": v, "pair": pair,
         "pair_kind": "one_graph", "n": G.number_of_nodes(), "edges": [list(e) for e in G.edges],
         "colors": [G.nodes[u]["color"] for u in range(G.number_of_nodes())], "deg_v": G.degree(v) if v is not None else None,
         "maxdeg": max(d for _, d in G.degree), "graph_seed": gs}
    r.update(extra or {})
    return r


def words(G, kind, gs):
    names = {u: f"w{u}" for u in G}
    return len(serialize(G, names, kind, seed=gs + SER_OFFSET[kind]).split()) + 40


def cmd_e11b(a):
    rng = random.Random(a.seed)
    n_pairs = 4 if tiny() else a.pairs
    out, k = [], 0
    for D in DEGREES:
        for pid in ("D1a", "D1c"):
            got = 0
            while got < n_pairs:
                G = hub_graph(rng, D)
                gs = rng.randrange(2**31)
                prop = BY_ID[pid]
                cand = [(u, w) for u in G[0] for w in G[1] if prop.query_key(G, u) == prop.query_key(G, w)
                        and not any(G.nodes[x]["color"] == "red" for x in G[w])]
                if not cand:
                    continue
                vp, vn = rng.choice(cand)
                kk = [kmin_dmin(G, vp, f, gs + SER_OFFSET[f])[0] for f in ("AL", "NC")]
                if min(kk) < max(3, D // 2):
                    continue
                if max(words(G, f, gs) for f in ("EL", "AL", "NC", "ECL")) * 1.6 > a.max_tokens:
                    continue
                pair = f"E11b-{pid}-D{D}-p{got:04d}"
                for v, lab in ((vp, True), (vn, False)):
                    out.append(rec(pid, "depth1", G, v, lab, pair, k, gs, {"degree_cell": D}))
                    k += 1
                got += 1
    return out


AL_LINE = "{h} ({c}): {ents}"
NC_LINE = "{h} is {c}. Neighbors: {ents}."


def probe_prompt(rng, ser, kq, head_red, names):
    """Lines of random heads with random entry lists; the query name is the kq-th entry of one line."""
    nl = rng.randint(4, 8)
    lines, qline = [], rng.randrange(nl)
    used = iter(names)
    q = next(used)
    for i in range(nl):
        h = next(used)
        m = max(kq, rng.randint(2, 6)) if i == qline else rng.randint(2, 6)
        ents = [next(used) for _ in range(m)]
        if i == qline:
            ents[kq - 1] = q
        c = ("red" if head_red else "blue") if i == qline else rng.choice(["red", "blue"])
        if ser == "AL":
            lines.append(AL_LINE.format(h=h, c=c, ents=", ".join(ents)))
        else:
            ents_c = ", ".join(f"{e} ({rng.choice(['red', 'blue'])})" for e in ents)
            lines.append(NC_LINE.format(h=h, c=c, ents=ents_c))
    text = "\n".join(" " + ln for ln in lines)
    return q, text


def cmd_e12(a):
    from names import POOL
    rng = random.Random(a.seed)
    n = 4 if tiny() else a.pairs
    out = []
    for ser in ("AL", "NC"):
        for kq in KS:
            for i in range(n):
                s = rng.randrange(2**31)
                names = random.Random(s).sample(POOL, 400)
                for lab in (True, False):
                    q, text = probe_prompt(random.Random(s), ser, kq, lab, names)
                    prompt = ("Here is a list of nodes. Every line starts with a node and its color.\n\n" + text +
                              f"\n\nQuestion: Is the color at the start of the line that lists node {q} red?\n"
                              "Answer with Yes or No.")
                    d = 2 * kq + 1 if ser == "AL" else 5 * kq - 1
                    out.append({"id": f"E12-{ser}-k{kq}-{i:03d}-{int(lab)}", "pid": "E12", "cls": "probe",
                                "pair": f"E12-{ser}-k{kq}-{i:03d}", "label": lab, "ser": ser, "tpl": "T1", "k": kq,
                                "d": d, "prompt": prompt})
    return out


def cmd_e13(a):
    rng = random.Random(a.seed)
    n = 6 if tiny() else a.pairs
    out = []
    for task in ("degree", "edge", "neighbors"):
        i = 0
        while i < n:
            nn = rng.randint(5, 20)
            G = nx.gnp_random_graph(nn, rng.uniform(0.1, 0.5), seed=rng.randrange(2**31))
            if G.number_of_edges() == 0:
                continue
            for u in G:
                G.nodes[u]["color"] = "blue"
            gs = rng.randrange(2**31)
            if task == "edge":
                es = list(G.edges)
                non = [(u, w) for u in G for w in G if u < w and not G.has_edge(u, w)]
                if not non:
                    continue
                for (u, w), lab in ((rng.choice(es), True), (rng.choice(non), False)):
                    out.append(dict(rec("E13edge", "graphqa", G, u, lab, f"E13-edge-p{i:04d}", len(out), gs),
                                    task=task, other=w, answer=lab))
            else:
                v = rng.choice([u for u in G if G.degree(u) > 0])
                ans = G.degree(v) if task == "degree" else sorted(G[v])
                out.append(dict(rec(f"E13{task}", "graphqa", G, v, True, f"E13-{task}-p{i:04d}", len(out), gs),
                                task=task, answer=ans))
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e11b", "e12", "e13"])
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=3800, help="E11b: bound on the estimated prompt length")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.pairs = a.pairs or {"e11b": 150, "e12": 60, "e13": 200}[a.cmd]
    recs = {"e11b": cmd_e11b, "e12": cmd_e12, "e13": cmd_e13}[a.cmd](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    print(f"wrote {len(recs)} records to {a.out}")


if __name__ == "__main__":
    main()
