"""Checks on an instance set before any model sees it (run after build.py / build_blind.py / build_real.py).

1. Every stored label equals the label recomputed from the graph (instances.prompts_for asserts it).
2. Positives and negatives are exactly balanced per question, and every pair has one of each.
3. Pair structure: one_graph pairs share the graph, the graph seed (names, orders) and the query key of
   tasks.py; two_graphs pairs share the graph key.
4. Token rule (F-1): every name occurrence in every prompt is preceded by a space (string-level check; the
   tokenizer-level check runs inside run_llm.py).
5. Shortcut ceilings: run probe_shortcuts.py on the same file (gradient boosting, grouped 5-fold CV).
Usage: python verify_data.py data/m0_instances.jsonl
"""
import re
import sys
from collections import Counter, defaultdict

from instances import load, prompts_for, to_graph
from serialize import ALL_FORMATS
from tasks import BY_ID


def main(path):
    recs = load(path)
    by_pair = defaultdict(list)
    for r in recs:
        by_pair[r["pair"]].append(r)
    ok = True
    for pid in sorted({r["pid"] for r in recs}):
        c = Counter(r["label"] for r in recs if r["pid"] == pid)
        print(f"{pid:4s} pos={c[True]} neg={c[False]}")
        ok &= c[True] == c[False]
    for pair, rs in by_pair.items():
        assert len(rs) == 2 and {r["label"] for r in rs} == {True, False}, pair
        p = BY_ID[rs[0]["pid"]]
        if rs[0]["pair_kind"] == "one_graph":
            a, b = rs
            assert a["edges"] == b["edges"] and a["colors"] == b["colors"] and a["graph_seed"] == b["graph_seed"], pair
            G = to_graph(a)
            assert p.query_key(G, a["query"]) == p.query_key(G, b["query"]), pair
        elif p.graph_key is not None:
            assert p.graph_key(to_graph(rs[0])) == p.graph_key(to_graph(rs[1])), pair
    print("checks 1-3:", "passed" if ok else "IMBALANCED")
    bad = 0
    sers = [s for s in ALL_FORMATS]
    for r in recs[:200]:
        for pr in prompts_for(r, serializations=sers, templates=("T1",)):
            for w in pr["names"]:
                for m in re.finditer(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", pr["prompt"]):
                    bad += pr["prompt"][m.start() - 1] != " "
    print(f"check 4: {bad} name occurrences without a leading space in the first 200 instances x {len(sers)} formats")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/m0_instances.jsonl")
