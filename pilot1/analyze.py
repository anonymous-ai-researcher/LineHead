"""Summarize pilot 1: accuracy per cell (mean, std over seeds), exact binomial tests vs 0.5 per seed (Holm over
the 18 seed-level tests of the 6 decidable cells; the paper reports the weakest) and pooled,
Wilson 95% CI on pooled accuracy, Holm correction over the 6 name-blind cells the theorem calls decidable
(the family of the paper's Table 7; chance cells are judged by their Wilson intervals); compare with the theorem.

Usage: python analyze.py results.jsonl --tex table.tex
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

# Theorem 1 predictions for ID-blind models: True = decidable, False = chance
PRED = {("G1", s): True for s in ("EL", "AL", "NC")}
PRED.update({("X2", "EL"): False, ("X2", "AL"): True, ("X2", "NC"): True,
             ("X1", "EL"): False, ("X1", "AL"): False, ("X1", "NC"): True,
             ("D1a", "EL"): False, ("D1a", "AL"): False, ("D1a", "NC"): False})
TASK_TEXT = {"G1": "some node is red", "X2": "some node has degree $\\geq 3$",
             "X1": "some red node has a red neighbor", "D1a": "query node has a red neighbor"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.results)]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["task"], r["ser"])].append(r)
    keys = sorted(cells, key=lambda k: (k[0] != "blind", ["G1", "X2", "X1", "D1a"].index(k[1]),
                                        ["EL", "AL", "NC"].index(k[2])))
    out, pvals = [], []
    for k in keys:
        rs = cells[k]
        accs = np.array([r["acc"] for r in rs])
        n = sum(r["n_test"] for r in rs)
        c = int(round(sum(r["acc"] * r["n_test"] for r in rs)))
        p = binomtest(c, n, 0.5, alternative="greater").pvalue
        lo, hi = proportion_confint(c, n, method="wilson")
        out.append(dict(arm=k[0], task=k[1], ser=k[2], seeds=len(rs), mean=accs.mean(),
                        std=accs.std(ddof=1) if len(rs) > 1 else float("nan"),
                        lo=lo, hi=hi, n=n, p=p, pred=PRED.get(k[1:]) if k[0] == "blind" else None))
        pvals.append(p)
    # Holm family = the cells the paper's Table 7 tests: the 6 name-blind cells the theorem calls decidable
    # (chance cells are checked by their Wilson intervals; the names-kept arm is reported separately)
    fam = [i for i, o in enumerate(out) if o["arm"] == "blind" and o["pred"]]
    adj = multipletests([pvals[i] for i in fam], method="holm")[1]
    for o in out:
        o["p_holm"] = float("nan")
    for i, q in zip(fam, adj):
        out[i]["p_holm"] = q
    for o in out:
        o["above_chance"] = (o["p_holm"] < 0.05) if o["p_holm"] == o["p_holm"] else (not (o["lo"] <= 0.5 <= o["hi"]))
        o["agrees"] = None if o["pred"] is None else (o["pred"] == o["above_chance"])
    # per-seed exact tests (the paper's Table 7): every seed of every decidable cell, Holm over all of them
    seed_p = [(k, r["seed"], binomtest(int(round(r["acc"] * r["n_test"])), r["n_test"], 0.5, alternative="greater").pvalue)
              for k in keys if k[0] == "blind" and PRED.get(k[1:]) for r in cells[k]]
    adj_s = multipletests([p for _, _, p in seed_p], method="holm")[1]
    w = max(range(len(seed_p)), key=lambda i: adj_s[i])
    print(f"per-seed exact tests: {len(seed_p)} (6 cells x seeds), Holm; weakest: {seed_p[w][0][1:]} seed {seed_p[w][1]} "
          f"p_holm={adj_s[w]:.3g}")
    for o in out:
        print(f"{o['arm']:5s} {o['task']:4s} {o['ser']}  acc {o['mean']:.3f} +/- {o['std']:.3f} "
              f"(seeds={o['seeds']}) CI[{o['lo']:.3f},{o['hi']:.3f}] p_holm={o['p_holm']:.2g} "
              f"pred={o['pred']} agrees={o['agrees']}")
    if a.tex:
        lines = []
        for o in out:
            pred = "--" if o["pred"] is None else ("decidable" if o["pred"] else "chance")
            std = "" if np.isnan(o["std"]) else f" $\\pm$ {o['std']:.3f}"
            arm = "ID-blind" if o["arm"] == "blind" else "names kept"
            above = "yes" if o["above_chance"] else "no"
            lines.append(f"{arm} & {o['task']} & {o['ser']} & {pred} & "
                         f"{o['mean']:.3f}{std} & [{o['lo']:.3f}, {o['hi']:.3f}] & {o['seeds']} & {above} \\\\")
        open(a.tex, "w").write("\n".join(lines) + "\n")
        print("wrote", a.tex)


if __name__ == "__main__":
    main()
