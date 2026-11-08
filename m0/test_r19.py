"""CONFIG §6.13: unit tests of r19.py without a model (structure of knockout, filler and tables)."""
import random

import r19
from e9_mechanism import list_structure
from instances import load, prompts_for
from names import POOL


def _prompts(ser):
    r = [x for x in load("data/m0_instances.jsonl") if x["pid"] == "D1a"][:3]
    return [p for x in r for p in prompts_for(x, POOL, serializations=(ser,), templates=("T1",))]


def test_filler():
    for ser in ("AL", "NC"):
        for p in _prompts(ser):
            a = r19.line_filler(p["prompt"], ser, 8, "inside")
            b = r19.line_filler(p["prompt"], ser, 8, "outside")
            n = len(list_structure(p["prompt"], ser))
            w0 = len(p["prompt"].split())
            assert len(a.split()) == len(b.split()) == w0 + 8 * n, (ser, len(a.split()), w0, n)
            assert not set(r19.FILL_WORDS) & set(POOL)
            # outside keeps every entry at its distance from the line head; inside moves the first entry by 8 words
            la = [l for l in a.split("\n") if r19.FILL_WORDS[1] in l][0]
            lb = [l for l in b.split("\n") if r19.FILL_WORDS[1] in l][0]
            assert la.split()[-1] != "content" and lb.rstrip(".").split()[-1] in r19.FILL_WORDS


def test_knockout_edges():
    for ser in ("AL", "NC"):
        p = _prompts(ser)[0]["prompt"]
        words = [(i, i + len(w)) for i, w in ((p.index(w, 0), w) for w in [])]
        # character-level stand-in tokenizer: one token per whitespace-separated word
        offs, pos = [], 0
        for w in p.split(" "):
            offs.append((pos, pos + len(w)))
            pos += len(w) + 1
        own = r19.knockout_edges(offs, 0, p, ser, "own")
        oth = r19.knockout_edges(offs, 0, p, ser, "other")
        assert own and oth and all(k < q for q, k in own + oth)
        assert not set(own) & set(oth)


def test_tables():
    recs = r19.tables_build(n_pairs=20)
    assert sum(r["pid"] == "T1" for r in recs) == 40 and sum(r["pid"] == "G1" for r in recs) == 40
    by = {}
    for r in recs:
        by.setdefault(r["pair"], []).append(r)
    for pr in by.values():
        assert sorted(r["label"] for r in pr) == [False, True]
        if pr[0]["pid"] == "T1":
            a, b = pr
            assert len(a["members"][str(a["query"])]) == len(b["members"][str(b["query"])])
            for r in pr:
                ms = r["members"][str(r["query"])]
                assert any(r["level"][str(u)] == "senior" for u in ms) == r["label"]
        else:
            for r in pr:
                assert (sum(v == "senior" for v in r["level"].values()) >= 4) == r["label"]
    ps = r19.tables_prompts(recs[0], POOL)
    assert len(ps) == 8 and {p["ser"] for p in ps} == set(r19.TFMTS)
    for p in ps:
        assert "senior" in p["prompt"] or "junior" in p["prompt"]


if __name__ == "__main__":
    test_filler(); test_knockout_edges(); test_tables()
    print("test_r19: ok")
