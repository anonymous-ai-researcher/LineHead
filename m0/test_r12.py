"""Unit tests of the code (CONFIG.md §6.2-§6.6): e11_heads.random_sets, build_r12 (e11b, e12, e13 instance
invariants), e12_probe.predict (zero-parameter prediction on synthetic summaries), e13_graphqa.parse_answer and
token_cost.py (word-level counts and hashes).
  python test_r12.py"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:                                  # under pytest, the scratch directory is pytest's tmp_path
    import pytest

    @pytest.fixture
    def tmp(tmp_path):
        return tmp_path
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def test_random_sets():
    from e11_heads import random_sets
    top = [(3, 1), (3, 4), (7, 0), (10, 2), (10, 5), (10, 6), (12, 3), (20, 1)]
    for h in (4, 8):
        a, b = random_sets(top, 28, 28, h, 0), random_sets(top, 28, 28, h, 0)
        assert a == b, "deterministic"
        assert len(a) == h and not set(a) & set(top[:h]), "h heads outside the top set"
        per = lambda s: sorted({l: sum(1 for x, _ in s if x == l) for l, _ in s}.items())
        assert per(a) == per(top[:h]), "same heads per layer as the top set"
        assert random_sets(top, 28, 28, h, 1) != a, "seeds differ"


def run_build(cmd, out):
    subprocess.run([sys.executable, "build_r12.py", cmd, "--out", str(out)], check=True, cwd=HERE,
                   env=dict(os.environ, E2E_TINY="1"), stdout=subprocess.DEVNULL)
    return [json.loads(l) for l in open(out)]


def test_build(tmp):
    import networkx as nx
    from instances import SER_OFFSET, to_graph
    from serialize import kmin_dmin
    from tasks import BY_ID
    recs = run_build("e11b", tmp / "e11b.jsonl")
    by = {}
    for r in recs:
        by.setdefault(r["pair"], []).append(r)
    assert len(by) == 4 * 2 * 4, len(by)                     # 4 pairs per (degree, question) under E2E_TINY
    for pair, (a, b) in by.items():
        assert a["edges"] == b["edges"] and a["label"] != b["label"]
        G = to_graph(a)
        D = a["degree_cell"]
        assert max(d for _, d in G.degree) == D
        assert BY_ID[a["pid"]].query_key(G, a["query"]) == BY_ID[b["pid"]].query_key(G, b["query"])
        pos = a if a["label"] else b
        neg = b if a["label"] else a
        assert [G.nodes[u]["color"] for u in G[pos["query"]]].count("red") == 1
        assert all(G.nodes[u]["color"] == "blue" for u in G[neg["query"]])
        for f in ("AL", "NC"):
            assert kmin_dmin(G, pos["query"], f, pos["graph_seed"] + SER_OFFSET[f])[0] >= max(3, D // 2)
    recs = run_build("e12", tmp / "e12.jsonl")
    by = {}
    for r in recs:
        by.setdefault(r["pair"], []).append(r)
    for pair, (a, b) in by.items():
        assert a["d"] == (2 * a["k"] + 1 if a["ser"] == "AL" else 5 * a["k"] - 1)
        wa, wb = a["prompt"].split(), b["prompt"].split()
        diff = [(x, y) for x, y in zip(wa, wb) if x != y]
        assert len(wa) == len(wb) and len(diff) == 1 and {d.strip("():.,") for d in diff[0]} == {"red", "blue"}, diff
    recs = run_build("e13", tmp / "e13.jsonl")
    for r in recs:
        G = to_graph(r)
        if r["task"] == "degree":
            assert r["answer"] == G.degree(r["query"])
        elif r["task"] == "neighbors":
            assert r["answer"] == sorted(G[r["query"]])
        else:
            assert r["answer"] == G.has_edge(r["query"], r["other"]) == r["label"]


def test_predict():
    from e12_probe import predict
    # probe: t(k) falls linearly from 0.9 at k=1 to 0.5 at k=5; TNR 0.9 everywhere
    P = {"cells": [dict(model="M", ser=s, k=k, tpr=0.9 - 0.1 * (k - 1), tnr=0.9) for s in ("AL", "NC") for k in (1, 3, 5)]}
    pos = {("AL", "a"): [1, 4], ("AL", "b"): [2], ("AL", "c"): [7], ("NC", "a"): [1], ("NC", "b"): [3, 9]}
    exact = {("AL", "1"): 0.9, ("AL", "2"): 0.85, ("AL", ">=5"): 0.7, ("NC", "1"): 0.9, ("NC", "3-4"): 0.8}
    lab = {"1": "k1", "2": "k2", "3-4": "k3-4", ">=5": "k5+"}
    S = {"cells": [dict(model="M", ser=s, strata=[dict(stratum=lab[st], bal_acc=v) for (s_, st), v in exact.items() if s_ == s])
                   for s in ("AL", "NC")]}
    out = predict(P, S, pos)
    assert out["error_pts"] is not None and out["error_pts"] < 1e-9, out
    S["cells"][0]["strata"][0]["bal_acc"] = 0.8                 # one stratum off by 10 points
    out = predict(P, S, pos)
    assert abs(out["max_error_pts"] - 10) < 1e-9 and abs(out["error_pts"] - 2) < 1e-9, out


def test_parse():
    from e13_graphqa import parse_answer
    names = {0: "cat", 1: "dog", 2: "elk"}
    assert parse_answer("degree", " The degree is 3.", names) == 3
    assert parse_answer("degree", "none", names) is None
    assert parse_answer("neighbors", " dog, elk and cat.", names) == [0, 1, 2]
    assert parse_answer("neighbors", " Dog, cow", names) == []


def test_token_cost(tmp):
    import random
    src = [json.loads(l) for l in open(HERE / "data" / "m0_instances.jsonl")]
    rng = random.Random(0)
    sub = rng.sample(src, 40)
    f = tmp / "inst.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in sub))
    out = tmp / "tc.json"
    subprocess.run([sys.executable, "token_cost.py", "--instances", str(f), "--out", str(out)], check=True, cwd=HERE,
                   stdout=subprocess.DEVNULL)
    r = json.load(open(out))
    import hashlib
    assert r["sha256"]["serialize.py"] == hashlib.sha256((HERE / "serialize.py").read_bytes()).hexdigest()
    assert r["sha256"]["inst.jsonl"] == hashlib.sha256(f.read_bytes()).hexdigest()
    w = r["word_tokens"]
    for c in {x["cls"] for x in sub} | {"all"}:
        assert w[f"{c}|EL2"]["mean"] > w[f"{c}|EL"]["mean"], "EL-2 writes every edge twice"
        assert w[f"{c}|NC"]["mean"] > w[f"{c}|AL"]["mean"], "NC adds a color to every entry"
        assert w[f"{c}|EL"]["max"] >= w[f"{c}|EL"]["mean"]


def test_corrupt():
    """The corrupted prompts of E11a patching change only the line heads."""
    import random
    from e11_heads import corrupt
    pal = " oak (red): hen, jar\n hen (blue): oak\n jar (blue): oak"
    r = corrupt(pal, "AL", "recolor", [], random.Random(0))
    assert r == " oak (blue): hen, jar\n hen (red): oak\n jar (red): oak"
    r = corrupt(pal, "AL", "rename", ["cat", "dog", "elk", "fox", "oak"], random.Random(0))
    heads = [ln.split()[0] for ln in r.split("\n")]
    assert len(set(heads)) == 3 and not set(heads) & {"oak", "hen", "jar"} and r.count("hen,") == 1, r
    pnc = " oak is red. Neighbors: hen (blue).\n hen is blue. Neighbors: oak (red)."
    assert corrupt(pnc, "NC", "recolor", [], random.Random(0)) == " oak is blue. Neighbors: hen (blue).\n hen is red. Neighbors: oak (red)."
    assert corrupt(" oak -- hen", "EL", "recolor", [], random.Random(0)) is None


def test_role_means():
    """E11a means are taken per format and token role; the pooled mean is their count-weighted average."""
    import os
    tiny = os.environ.get("E9_MODEL") or str(Path(os.environ.get("TINY_DIR", "/nonexistent")))
    if not Path(tiny, "config.json").exists():
        print("test_role_means: skipped (no tiny model)")
        return
    import torch
    from e11_heads import Ablator, load_model, token_roles, chat
    tok, m, _ = load_model(tiny)
    ps = [" oak (red): hen, jar\n hen (blue): oak\n jar (blue): oak", " oak -- hen\n hen -- jar"]
    ab = Ablator(m, torch)
    ab.calibrate(tok, ps, fmts=["AL", "EL"], names={"oak", "hen", "jar"})
    roles = token_roles(tok, tok(chat(tok, ps[0]), add_special_tokens=False)["input_ids"], {"oak", "hen", "jar"})
    assert roles.count("N") == 7 and roles.count("C") == 3, roles
    n = {k: 0 for k in ab.keys}
    for p, f in zip(ps, ["AL", "EL"]):
        for r in token_roles(tok, tok(chat(tok, p), add_special_tokens=False)["input_ids"], ab.names):
            n[(f, r)] += 1
            n[(f, "*")] += 1
    for l in (0, ab.L - 1):
        w = sum(n[k] * ab.table[l][i] for k, i in ab.keys.items() if k[1] != "*") / sum(v for k, v in n.items() if k[1] != "*")
        assert torch.allclose(w, ab.mean[l], atol=1e-6)
        assert not torch.allclose(ab.table[l][ab.keys[("AL", "N")]], ab.table[l][ab.keys[("AL", "C")]])
    ab.prepare(tok, ps[0], "AL")
    assert ab.idx.shape[0] == len(roles)
    print("test_role_means: ok")


def main():
    tmp = Path(tempfile.mkdtemp())
    test_corrupt()
    test_role_means()
    test_random_sets()
    test_build(tmp)
    test_predict()
    test_parse()
    test_token_cost(tmp)
    print("test_r12: ok")


if __name__ == "__main__":
    main()
