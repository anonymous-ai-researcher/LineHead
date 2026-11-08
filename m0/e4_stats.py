"""Cell summaries for E2, E4, E5, E6, E7, E8, E10 (CONFIG.md §5-6).

Balanced accuracy with a 95% pair-bootstrap interval for every combination of the grouping columns, plus
depth-one strata by k_min (1, 2, 3-4, >=5; as in analysis/schema.json) and by maximum degree (<=2, 3-7, >=8), and
graph-level paired permutation tests between the listed format pairs, Holm-corrected.

Examples
  E4 (formats):  python e4_stats.py results/e4_*.jsonl --by model,ser --cls depth1 \
                     --tests ECL:NC,EL2:EL,ELfar:EL,NCH:NC,ECL:ELfar --out results/e4_summary.json
  E7 (partial):  python e4_stats.py results/e7_*.jsonl results/e7q0_*.jsonl results/e1_<model>.jsonl --pids D1a,Q0 \
                     --by model,ser,mask,pid --out results/summary_e7.json   (E7 none for D1a = E1)
  E2 (control):  python e4_stats.py results/e2_*.jsonl --by model,ser,cls --out results/e2_summary.json
  E5 / GraphML:  python e4_stats.py results/e5_molhiv.jsonl --by model,ser --cls depth1 --out ...
  E8:            python e4_stats.py results/e8_*.jsonl --by model,variant,ser,cls --out ...
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from stats import bal_acc, load, paired_permutation, pair_bootstrap

KSTRATA = {"k1": lambda k: k == 1, "k2": lambda k: k == 2, "k3-4": lambda k: (k >= 3) & (k <= 4),
           "k5+": lambda k: k >= 5}
DSTRATA = {"le2": lambda d: d <= 2, "3to7": lambda d: (d >= 3) & (d <= 7), "ge8": lambda d: d >= 8}


def holm_within(tests, group):
    """Holm correction within each group (e.g. within each model: the seven E4 comparisons of one model form one
    family, as the paper states), never pooled across groups."""
    fams = {}
    for t in tests:
        fams.setdefault(tuple(t.get(g) for g in group), []).append(t)
    for fam in fams.values():
        for t, q in zip(fam, multipletests([t["p"] for t in fam], method="holm")[1]):
            t["p_holm"] = float(q)
    return tests


def strata_cells(c, reps, rng):
    """Depth-one strata. k_min is defined for positives only, so a k stratum pairs the true-positive rate of its
    positives with the true-negative rate of ALL negatives of the condition (as in the paper's Table T_kstrata):
    the cell holds the stratum's positives and every negative, and balanced accuracy = (TPR_k + TNR) / 2."""
    out = []
    kpair = c[c.label == True].groupby("pair")["k_min"].first()               # noqa: E712
    neg = c[c.label == False]                                                 # noqa: E712
    for name, f in KSTRATA.items():
        if kpair.isna().all():
            break
        pairs = set(kpair[f(kpair.fillna(-1))].index)
        cc = pd.concat([c[(c.label == True) & c.pair.isin(pairs)], neg])         # noqa: E712
        if len(cc):
            lo, hi = pair_bootstrap(cc, reps, rng)
            out.append({"stratum": name, "bal_acc": bal_acc(cc), "ci_low": float(lo), "ci_high": float(hi),
                        "n_pairs": len(pairs)})
    for name, f in DSTRATA.items():
        cc = c[f(c.maxdeg)]
        if len(cc):
            lo, hi = pair_bootstrap(cc, reps, rng)
            out.append({"stratum": name, "bal_acc": bal_acc(cc), "ci_low": float(lo), "ci_high": float(hi),
                        "n_pairs": int(cc.pair.nunique())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--by", default="model,ser")
    ap.add_argument("--cls", default=None, help="keep one class, e.g. depth1")
    ap.add_argument("--pids", default=None, help="keep these question ids, e.g. D1a,Q0 (E7)")
    ap.add_argument("--tests", default="", help="format pairs A:B for paired permutation tests (within model)")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--perm-reps", type=int, default=10_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("E2E_TINY"):               # offline end-to-end test only: few resamples
        a.reps = 50
        if hasattr(a, "perm_reps"):
            a.perm_reps = 50
    rng = np.random.default_rng(0)
    d = load(a.results)
    if "variant" not in d:
        d["variant"] = "default"
    if a.cls:
        d = d[d.cls == a.cls]
    if a.pids:
        d = d[d.pid.isin(a.pids.split(","))]
    if "mask" not in d:
        d["mask"] = "none"
    d["mask"] = d["mask"].fillna("none")         # E1 records carry no mask: they are the E7 "none" condition
    by = a.by.split(",")
    out = {"cells": [], "tests": []}
    for key, c in d.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        lo, hi = pair_bootstrap(c, a.reps, rng)
        cell = dict(zip(by, key), bal_acc=bal_acc(c), ci_low=float(lo), ci_high=float(hi),
                    n_pairs=int(c.pair.nunique()))
        if "reasoning_tokens" in c and c["reasoning_tokens"].notna().any():          # E6 CoT truncation rate
            cell["truncated"] = float((c["reasoning_tokens"] >= 512).mean())
        if "fallback" in c and c["fallback"].notna().any():             # E10: answers parsed from text (B4)
            cell["fallback"] = float(c["fallback"].fillna(False).astype(bool).mean())
        if (c.cls == "depth1").all():
            cell["strata"] = strata_cells(c, a.reps, rng)
        out["cells"].append(cell)
    if a.tests:
        group = [b for b in by if b != "ser"]
        for key, c in d.groupby(group) if group else [((), d)]:
            for t in a.tests.split(","):
                x, y = t.split(":")
                r = paired_permutation(c, x, y, reps=a.perm_reps)
                if r:
                    r.update(dict(zip(group, key if isinstance(key, tuple) else (key,))))
                    out["tests"].append(r)
        holm_within(out["tests"], group)
    json.dump(out, open(a.out, "w"), indent=1)
    for c in out["cells"]:
        tag = " ".join(f"{b}={c[b]}" for b in by)
        print(f"{tag:40s} {c['bal_acc']:.3f} [{c['ci_low']:.3f},{c['ci_high']:.3f}] pairs={c['n_pairs']}")


if __name__ == "__main__":
    main()
