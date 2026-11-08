"""k_min shares of the frozen E1 instance set (measured, no model needed).

For the positives of D1a and D1c (and of D1a alone), k_min under the order that serialize() uses for AL and NC
(seed = graph_seed + SER_OFFSET); EL, ECL and GraphML take the matched NC k_min (CONFIG §5.5). Shares by the
paper's bins 1, 2, 3-4, >=5. The output records the SHA-256 of the instance file it was computed from.

Usage: python kmin_shares.py data/m0_instances.jsonl --out results/kmin_shares.json
"""
import argparse
import hashlib
import json

from instances import SER_OFFSET, load, to_graph
from serialize import kmin_dmin

BINS = (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), (">=5", 5, 10 ** 9))


def shares(ks):
    n = len(ks)
    return {lab: sum(lo <= k <= hi for k in ks) / n for lab, lo, hi in BINS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances")
    ap.add_argument("--out", default="results/kmin_shares.json")
    a = ap.parse_args()
    recs = load(a.instances)
    out = {"instances": a.instances, "instances_sha256": hashlib.sha256(open(a.instances, "rb").read()).hexdigest()}
    for subset, pids in (("D1a+D1c", ("D1a", "D1c")), ("D1a", ("D1a",))):
        out[subset] = {}
        for s in ("AL", "NC"):
            ks = []
            for r in recs:
                if r["pid"] in pids and r["label"]:
                    k, _ = kmin_dmin(to_graph(r), r["query"], s, r["graph_seed"] + SER_OFFSET[s])
                    ks.append(k)
            out[subset][s] = {"n_positives": len(ks), "kmin_share": shares(ks)}
        for s in ("EL", "ECL", "GraphML"):
            out[subset][s] = dict(out[subset]["NC"], matched="NC")
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "instances"}, indent=1)[:900])


if __name__ == "__main__":
    main()
