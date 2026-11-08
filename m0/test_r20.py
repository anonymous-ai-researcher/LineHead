"""CONFIG §6.14: unit tests of r20.py without a model (builders, BFS order, between filler, stability) and
of the ELB serialization of E33."""
import networkx as nx

import r20
from e9_mechanism import list_structure
from instances import load, prompts_for, to_graph
from names import POOL
from serialize import kmin_dmin
from instances import SER_OFFSET


def test_between_filler():
    r = [x for x in load("data/m0_instances.jsonl") if x["pid"] == "D1a"][:3]
    for ser in ("AL", "NC"):
        for p in (p for x in r for p in prompts_for(x, POOL, serializations=(ser,), templates=("T1",))):
            a = r20.between_filler(p["prompt"], ser, 24)
            n = len(list_structure(p["prompt"], ser))
            assert len(a.split()) == len(p["prompt"].split()) + 24 * n
            assert list_structure(a, ser) and len(list_structure(a, ser)) == n     # list lines themselves unchanged
            assert all(l in a.split("\n") for l in p["prompt"].split("\n"))


def test_bfs_edges():
    r = [x for x in load("data/m0_instances.jsonl") if x["pid"] == "D1a"][:5]
    for x in r:
        p = next(prompts_for(x, POOL, serializations=("EL",), templates=("T1",)))["prompt"]
        b = r20.bfs_edges(p)
        assert sorted(p.split("\n")) == sorted(b.split("\n")) and len(b) == len(p)


def test_large_and_latek():
    recs = r20.build_large(n_pairs=2)
    assert len(recs) == 3 * 2 * 3 * 2
    for r in recs:
        G = to_graph(r)
        assert G.number_of_nodes() == r["size"]
        idx = [i for i, (u, _) in enumerate(r["order"]) if u == r["query"]][0]
        n = r["size"]
        assert r["third"] * n // 3 <= idx < (r["third"] + 1) * n // 3 + 1
        assert next(prompts_for(r, POOL, serializations=("AL",), templates=("T1",)))
    for r in r20.build_latek(n_pairs=3):
        if r["label"]:
            assert kmin_dmin(to_graph(r), r["query"], "AL", r["graph_seed"] + SER_OFFSET["AL"])[0] >= 5
        assert next(prompts_for(r, POOL, serializations=("AL",), templates=("T1",)))


def test_algo():
    recs = r20.build_algo(n_pairs=5)
    by = {}
    for r in recs:
        G = to_graph(r)
        assert r20.algo_label(r["pid"], G, r["query"], r["query2"]) == r["label"]
        by.setdefault(r["pair"], []).append(r)
    for rs in by.values():
        assert sorted(x["label"] for x in rs) == [False, True]
        if rs[0]["pid"] != "A2":
            assert rs[0]["query"] == rs[1]["query"] and rs[0]["edges"] == rs[1]["edges"]
            G = to_graph(rs[0])
            assert G.degree(rs[0]["query2"]) == G.degree(rs[1]["query2"])
        else:
            assert (rs[0]["n"], len(rs[0]["edges"])) == (rs[1]["n"], len(rs[1]["edges"]))
    assert len(list(r20.algo_prompts(recs[0], POOL))) == 5


def test_stability_helpers():
    h1, h2 = r20.split_halves(100, 3)
    assert len(h1) == len(h2) == 50 and not set(h1) & set(h2)
    assert r20.jaccard([(0, 1), (1, 2)], [(1, 2), (3, 4)]) == 1 / 3


def test_elb_blocks():
    """ELB (E33): the edges of EL in blocks of 3, each preceded by a Colors line with exactly its endpoints' colors;
    every edge appears once and every vertex's color is the one of the graph."""
    import random
    from serialize import ELB_BLOCK, edge_blocks
    for x in load("data/m0_instances.jsonl")[:20]:
        G = to_graph(x)
        names = dict(zip(G.nodes, POOL))
        inv = {v: k for k, v in names.items()}
        lines = [ln.strip() for ln in edge_blocks(G, names, random.Random(0)).split("\n") if ln.strip()]
        seen, cols = set(), None
        n_edges_in_block = 0
        for ln in lines:
            if ln.startswith("Colors:"):
                assert cols is None or n_edges_in_block > 0
                items = [t.strip().split(" ") for t in ln[len("Colors:"):].split(",") if t.strip()]
                cols = {inv[a]: c for a, c in items}
                assert all(G.nodes[u]["color"] == c for u, c in cols.items())
                n_edges_in_block = 0
            else:
                a, b = (inv[t] for t in ln.split(" -- "))
                assert a in cols and b in cols and n_edges_in_block < ELB_BLOCK
                seen.add(frozenset((a, b)))
                n_edges_in_block += 1
        assert seen == {frozenset(e) for e in G.edges}


if __name__ == "__main__":
    test_between_filler(); test_bfs_edges(); test_large_and_latek(); test_algo(); test_stability_helpers(); test_elb_blocks()
    print("ok")
