"""Quantities that combine models or cells (paper Figures 1b and 4a; name-blind control).

mean-ci   Mean over models of balanced accuracy with a JOINT pair bootstrap: the models answer the same
          instances, so one resample of matched pairs (stratified by question) is drawn per replicate and applied
          to every model; 95% percentile interval of the mean (2,000 replicates). Groups: --by columns other
          than model (e.g. cls,ser for Figure 1b; ser,mask,pid for Figure 4a).
blind     E2: the 12 undecidable positive-control cells (X2 under EL, X1 under EL and AL, 4 models) tested against
          0.5, two-sided, normal approximation with the pair-bootstrap se, Holm over the 12; the 36 query-vertex
          cells are exactly 0.5 by construction (checked, not tested).

Usage
  python joint_stats.py mean-ci results/e1_*.jsonl --by cls,ser --out results/summary_e1_meanci.json
  python joint_stats.py mean-ci results/e7_*.jsonl results/e7q0_*.jsonl results/e1_*.jsonl results/e4_*.jsonl \
      --pids D1a,Q0 --by ser,mask,pid --out results/summary_e7_meanci.json
  python joint_stats.py blind results/e2_*.jsonl --out results/summary_e2_holm.json   (query and control files)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from stats import _pair_counts, _ba_from_counts, bal_acc, load, pair_bootstrap


def _counts_by_pair(d):
    c, pids = _pair_counts(d)
    pairs = d.groupby("pair")["pid"].first().index.values
    return dict(zip(pairs, c)), pids, pairs


def mean_ci(d, by, reps, rng):
    out = []
    models = sorted(d.model.unique())
    for key, g in d.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        per = {m: _counts_by_pair(g[g.model == m]) for m in models if (g.model == m).any()}
        if len(per) < 2:
            continue
        common = sorted(set.intersection(*(set(v[0]) for v in per.values())))      # pairs shared by all models
        pid_of = g.groupby("pair")["pid"].first()
        strata = {q: np.array([p for p in common if pid_of[p] == q]) for q in sorted(set(pid_of[common]))}
        arr = {m: np.array([v[0][p] for p in common]) for m, v in per.items()}
        pos = {p: i for i, p in enumerate(common)}
        est = float(np.mean([_ba_from_counts(a) for a in arr.values()]))
        vals = np.empty(reps)
        for r in range(reps):
            idx = np.concatenate([[pos[p] for p in s[rng.integers(0, len(s), len(s))]] for s in strata.values()])
            vals[r] = np.mean([_ba_from_counts(a[idx]) for a in arr.values()])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out.append(dict(zip(by, key), mean_bal_acc=est, ci_low=float(lo), ci_high=float(hi),
                        n_models=len(arr), n_pairs=len(common)))
    return out


UNDECIDABLE = {("X2", "EL"), ("X1", "EL"), ("X1", "AL")}     # positive controls the theorem calls undecidable


def blind(d, reps, rng):
    """E2 Holm family: the undecidable positive-control cells (X2 under EL, X1 under EL and AL; 3 per model, 12 in
    all), each tested against 0.5, two-sided, normal approximation with the pair-bootstrap se. The query-vertex
    cells are exactly 0.5 by construction (identical masked prompts in a pair) and are reported, not tested."""
    cells = []
    for (m, pid, ser), g in d[d.pid.isin(["X1", "X2"])].groupby(["model", "pid", "ser"]):
        if (pid, ser) not in UNDECIDABLE:
            continue
        c, pids = _pair_counts(g)
        b = np.array([_ba_from_counts(c[_ridx(pids, rng)]) for _ in range(reps)])
        est, se = bal_acc(g), float(np.std(b, ddof=1))
        p = 1.0 if se == 0 else float(2 * norm.sf(abs(est - 0.5) / se))
        cells.append(dict(model=m, pid=pid, ser=ser, bal_acc=est, se=se, p=p))
    if cells:
        for x, q in zip(cells, multipletests([x["p"] for x in cells], method="holm")[1]):
            x["p_holm"] = float(q)
            x["off_chance"] = bool(q < 0.05)
    exact = d[d.cls.isin(["depth1", "depth2", "count"])].groupby(["model", "cls", "ser"]).apply(bal_acc)
    return {"controls": cells, "n_cells": len(cells), "n_off_chance": sum(x["off_chance"] for x in cells),
            "query_cells_exactly_half": bool((exact == 0.5).all()), "n_query_cells": int(len(exact))}


def _ridx(pids, rng):
    from stats import _resample_idx
    return _resample_idx(pids, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("mean-ci", "blind"))
    ap.add_argument("results", nargs="+")
    ap.add_argument("--by", default="cls,ser")
    ap.add_argument("--pids", default=None)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("E2E_TINY"):               # offline end-to-end test only: few resamples
        a.reps = 50
        if hasattr(a, "perm_reps"):
            a.perm_reps = 50
    rng = np.random.default_rng(a.seed)
    d = load(a.results)
    if a.pids:
        d = d[d.pid.isin(a.pids.split(","))]
    if "mask" not in d:
        d["mask"] = "none"
    d["mask"] = d["mask"].fillna("none")
    out = mean_ci(d, a.by.split(","), a.reps, rng) if a.mode == "mean-ci" else blind(d, a.reps, rng)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1)[:1500])


if __name__ == "__main__":
    main()
