"""Summary of the controlled experiments (CONFIG.md §6.14; rules C50-C57, C66). Reads the
results/r20_<mode>.jsonl files written by r20_e3.sh and writes one JSON with seed means, s.d. and the rule statistics.
Keys: "<exp>|<arm>|<format>|<setting>|<split>" -> {"seeds", "mean", "sd"}; derived statistics under "<exp>|stat|...".
  E22  pos-control   setting = task (g1, c3)
  E23  data-limited  setting = N; plus n95 (log-linear interpolation of the seed means of the final ID accuracy
                     between adjacent N, as e15_summary.py) and steps95 (first evaluation step, every 250 steps, with
                     val_bal_acc >= 0.95 at N = 500k, per seed; mean and 95% bootstrap interval of EL minus AL / NC)
  E25  window        setting = w<w>L<L>; split = k-strata of the k control set (k_min strata of gen_e3.KSTRATA)
  E26  nest          setting = pm<E>
  E27  depth2        setting = task (dd, c2)
  E32  robust        setting = L<L> (depth sweep seeds 3-4, pooled with results/e3d.jsonl seeds 0-2 if given) or d<d>
Usage: python r20_summary.py --results results --e3d results/e3d.jsonl --out ../m0/results/summary_r20.json
"""
import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

SIZES = (25000, 50000, 100000, 200000, 500000)


def agg(xs):
    return {"seeds": xs, "mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0}


def n95(points):
    """points: [(N, mean ID accuracy)] in increasing N; None if 0.95 is never reached."""
    if points and points[0][1] >= 0.95:
        return points[0][0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 < 0.95 <= y1:
            return round(math.exp(math.log(x0) + (0.95 - y0) / (y1 - y0) * (math.log(x1) - math.log(x0))))
    return None


def steps95(run):
    for rec in run.get("curve", []):
        if rec.get("val_bal_acc", 0) >= 0.95:
            return rec["step"]
    return None


def boot_diff(a, b, B=10_000, seed=0):
    rng = random.Random(seed)
    d = sorted(st.mean(rng.choices(a, k=len(a))) - st.mean(rng.choices(b, k=len(b))) for _ in range(B))
    return {"mean": st.mean(a) - st.mean(b), "lo": d[int(0.025 * B)], "hi": d[int(0.975 * B) - 1]}


def setting(exp, r):
    if exp in ("E22", "E27"):
        return r["task"]
    if exp == "E23":
        return str(r["train_size"])
    if exp == "E25":
        return f"w{r['window']}L{r['layers']}"
    if exp == "E32":
        return f"d{r['d']}" if r["d"] != 256 else f"L{r['layers']}"
    return "all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--e3d", default=None, help="E3d runs (seeds 0-2) pooled into the E32 depth sweep")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = {"E22": "pos-control", "E23": "data-limited", "E25": "window", "E26": "nest", "E27": "depth2", "E32": "robust"}
    out = {}
    for exp, mode in files.items():
        p = Path(a.results) / f"r20_{mode}.jsonl"
        runs = [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
        if exp == "E32" and a.e3d and Path(a.e3d).exists():
            runs += [r for r in map(json.loads, open(a.e3d)) if r.get("seed") in (0, 1, 2)]
        cells = {}
        for r in runs:
            base = (exp, r["pos"], r["ser"], setting(exp, r))
            for key, v in r.items():
                if key.startswith("acc_") and isinstance(v, float):
                    cells.setdefault(base + (key[4:],), []).append(v)
                elif key == "acc_k" and isinstance(v, dict):
                    for s, acc in v.items():
                        cells.setdefault(base + (f"k{s}",), []).append(acc)
        for k, xs in cells.items():
            out["|".join(k)] = agg(xs)
        if exp == "E23":
            for pos in sorted({r["pos"] for r in runs}):
                for f in ("EL", "AL", "NC", "ECL"):
                    pts = [(N, out[f"E23|{pos}|{f}|{N}|id"]["mean"]) for N in SIZES if f"E23|{pos}|{f}|{N}|id" in out]
                    out[f"E23|stat|{pos}|{f}|n95"] = n95(pts)
                    out[f"E23|stat|{pos}|{f}|steps95"] = [steps95(r) for r in runs if r["pos"] == pos and r["ser"] == f
                                                         and r["train_size"] == 500000]
                el = [s for s in out[f"E23|stat|{pos}|EL|steps95"] if s is not None]
                for g in ("AL", "NC"):
                    oth = [s for s in out[f"E23|stat|{pos}|{g}|steps95"] if s is not None]
                    if el and oth:
                        out[f"E23|stat|{pos}|EL-{g}|steps95"] = boot_diff(el, oth)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(out)} entries")


if __name__ == "__main__":
    main()
