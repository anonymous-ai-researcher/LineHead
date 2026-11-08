"""Unit tests (run: python test_r14.py): E14 filler placement, E16 identifiers, E11b-L records."""
import re

from instances import FILLER, add_filler, assign_names, load, prompts_for, to_graph
from tasks import BY_ID

WORD = re.compile(r"[A-Za-z0-9]+|--|[^\sA-Za-z0-9]")


def test_filler():
    r = [x for x in load("data/m0_instances.jsonl") if x["pid"] == "D1a"][0]
    for f in ("EL", "ECL"):
        base = next(prompts_for(r, serializations=(f,), templates=("T1",)))["prompt"]
        for pos in ("between", "before"):
            q = dict(r, filler={"pos": pos, "words": 64})
            p = next(prompts_for(q, serializations=(f,), templates=("T1",)))["prompt"]
            assert p.count(FILLER) == 8 and len(WORD.findall(p)) == len(WORD.findall(base)) + 64
            i, c, e = p.index(FILLER), p.index("Colors:"), p.index("Edges:")
            assert (c < i < e) if pos == "between" else (i < c)
            assert p.replace(FILLER + "\n ", "") == base          # nothing else changes
    q = dict(r, filler={"pos": "between", "words": 64})
    assert FILLER not in next(prompts_for(q, serializations=("AL",), templates=("T1",)))["prompt"]   # EL, ECL only


def test_ids():
    r = load("data/m0_instances.jsonl")[0]
    for ids in ("compound", "num4"):
        nm = assign_names(r, __import__("names").POOL, ids)
        assert len(set(nm.values())) == r["n"]
        assert nm == assign_names(r, __import__("names").POOL, ids)          # deterministic per graph
    assert all(re.fullmatch(r"\d{4}", v) for v in assign_names(r, __import__("names").POOL, "num4").values())


def test_e11bL():
    recs = load("data/e11bL_instances.jsonl")
    for r in recs[:200]:
        G = to_graph(r)
        assert bool(BY_ID[r["pid"]].label(G, r["query"])) == r["label"]
        assert all(G.degree(u) == 0 for u in range(r["n"] - r["pad"], r["n"]))   # padding is isolated
        ps = list(prompts_for(r, serializations=("EL", "AL", "NC", "ECL"), templates=("T1",)))
        assert {p["ser"] for p in ps} == {r["ser_only"]}
    assert all(r["pad"] == 0 for r in recs if r["degree_cell"] == 48)


if __name__ == "__main__":
    test_filler(); test_ids(); test_e11bL()
    print("test_r14: all passed")
