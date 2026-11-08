"""E15 summary (CONFIG.md §6.10): seeds, mean and s.d. of balanced accuracy per (format, N, split) from
results/e15.jsonl (train_e3.py --train-size) and, for N = 500k, from the E3 runs (results/e3.jsonl, arm
local_hardsince, task dia, seeds 0-2); examples to reach 0.95 in distribution by log-linear interpolation between
adjacent sizes of the seed means.
Usage: python e15_summary.py results/e15.jsonl results/e3.jsonl --out ../m0/results/summary_e15.json
"""
import argparse
import json
import math
import statistics as st

SIZES = (25000, 50000, 125000, 250000, 500000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e15")
    ap.add_argument("e3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [json.loads(l) for l in open(a.e15) if l.strip()]
    runs += [dict(r, train_size=500000) for r in map(json.loads, open(a.e3)) if r.get("pos") == "local_hardsince"
             and r.get("task") == "dia" and r.get("layers", 6) == 6 and r.get("seed") in (0, 1, 2)]
    out = {}
    for f in ("EL", "AL", "NC", "ECL"):
        for N in SIZES:
            for sp, key in (("id", "acc_id"), ("ood", "acc_ood")):
                sd = [r[key] for r in sorted(runs, key=lambda r: r["seed"])
                      if r["ser"] == f and r.get("train_size") == N and r.get(key) is not None]
                if sd:
                    out[f"{f}|{N}|{sp}"] = {"seeds": sd, "mean": st.mean(sd), "sd": st.stdev(sd) if len(sd) > 1 else 0.0}
        ys = [(N, out[f"{f}|{N}|id"]["mean"]) for N in SIZES if f"{f}|{N}|id" in out]
        n95 = ys[0][0] if ys and ys[0][1] >= 0.95 else None
        for (x0, y0), (x1, y1) in zip(ys, ys[1:]):
            if n95 is None and y0 < 0.95 <= y1:
                n95 = round(math.exp(math.log(x0) + (0.95 - y0) / (y1 - y0) * (math.log(x1) - math.log(x0))))
        out[f"{f}|n95"] = n95
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(out)} entries")


if __name__ == "__main__":
    main()
