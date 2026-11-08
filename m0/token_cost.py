"""Token cost of every serialization on the E1 instances (decision table): mean and maximum number of
word-level tokens (the tokens of the theory: names, color words, punctuation, "--"; serialize.py) of the graph text,
per question class, for the frozen instance file. Measured without a model; the SHA-256 values of the
serializer, this script and the instance file are stored in the output, so it can be checked against the current files. Model-token lengths are
measured with each tokenizer in the LLM runs (r9_stats.py lengths) and are not part of this file.
Usage: python token_cost.py --instances data/m0_instances.jsonl --out results/token_cost.json
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from instances import SER_OFFSET, assign_names, load, to_graph
from names import POOL
from serialize import serialize

FMTS = ["EL", "AL", "NC", "ECL", "EL2", "ELfar", "NCH", "Incident", "Friendship", "GraphML"]
WORD = re.compile(r"[A-Za-z0-9]+|--|[^\sA-Za-z0-9]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--out", default="results/token_cost.json")
    a = ap.parse_args()
    recs = load(a.instances)
    seen, by = set(), {}
    for r in recs:
        if r["pair"] in seen and r["pair_kind"] == "one_graph":
            continue                                   # one graph per query pair
        seen.add(r["pair"])
        G, names = to_graph(r), assign_names(r, POOL)
        for f in FMTS:
            n = len(WORD.findall(serialize(G, names, f, seed=r["graph_seed"] + SER_OFFSET[f])))
            by.setdefault((r["cls"], f), []).append(n)
            by.setdefault(("all", f), []).append(n)
    here = Path(__file__).resolve().parent
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out = {"word_tokens": {f"{c}|{f}": {"mean": round(float(np.mean(v)), 1), "max": int(max(v)), "n_graphs": len(v)}
                           for (c, f), v in sorted(by.items())},
           "sha256": {"serialize.py": sha(here / "serialize.py"), "token_cost.py": sha(here / "token_cost.py"),
                      Path(a.instances).name: sha(a.instances)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v["mean"] for k, v in out["word_tokens"].items() if k.startswith("depth1")}))


if __name__ == "__main__":
    main()
