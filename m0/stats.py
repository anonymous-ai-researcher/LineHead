"""Statistics for the LLM experiments (CONFIG.md §7; paper appendix "Statistical Analysis").

Per (class, serialization) cell
  balanced accuracy, pooled over the class's questions and the 3 templates; 95% percentile interval from
  2,000 bootstrap resamples of MATCHED PAIRS (S-4): a resampled pair enters with both instances and all
  their prompts (3 templates; all serializations when formats are compared).
Format comparisons (S-3)
  graph-level paired permutation test: per pair, the difference in correctness between two formats is
  averaged over the pair's instances and templates; the sign of each pair's difference is flipped at random
  (10,000 permutations), two-sided p. Holm within model over classes x format pairs.
Interaction
  logistic GEE  correct ~ C(ser)*C(cls) + C(tpl) [+ loglen], clustered by graph (a query-vertex pair is one
  graph; each graph of a global pair is its own cluster), exchangeable; Wald test of the 6 interaction
  coefficients. Robustness: the same model with question fixed effects (class x question within class, coded
  C(cls) + C(qi) + C(cls):C(qi), which avoids collinear columns), reported as "interaction_qfe".
Planned contrasts (S-1, primary, preregistered)
  Delta_c(F) = (F - EL)_c - (F - EL)_global for F in {NC, AL}, c in {depth1, depth2, count};
  one-sided (Delta > 0): z = Delta / se with se the sd of 2,000 pair-bootstrap resamples (stratified by
  question), normal p, Holm over the 6 contrasts of a model; the empirical bootstrap p is also reported.
Variance explained (S-2)
  McFadden pseudo-R2 increment of the ser x cls INTERACTION only: full model ser*cls + model + tpl versus
  ser + cls + model + tpl (both pooled over models).
Distance regressions (S-5), depth-one questions, per serialization (AL, NC; EL with log length only)
  logit(correct) ~ log d_min + k_min + log n_tokens + n + deg_v (+ C(model)), positives only (k_min and
  d_min are defined for positives), cluster-robust SE by pair.

Usage: python stats.py results/e1_*.jsonl --out results/summary.json
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from serialize import SERIALIZATIONS
from tasks import CLASSES


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f]
    df = pd.DataFrame(rows)
    keys = [k for k in ("model", "id", "ser", "tpl", "mask", "variant", "mode", "ids", "question_first", "sort_edges")
            if k in df]
    dup = df.astype({k: str for k in keys}).duplicated(keys, keep="last")
    if dup.any():                      # a resumed run that wrote a prompt twice: keep the last record, warn
        import warnings
        warnings.warn(f"{int(dup.sum())} duplicate records (same model, prompt and condition); the last one is used")
        df = df[~dup.values].reset_index(drop=True)
    df["correct"] = (df["pred"] == df["label"]).astype(int)   # unparseable (None) counts as wrong
    df["unparsed"] = df["pred"].isna()
    if "pair" not in df or df["pair"].isna().all():
        df["pair"] = df["id"]
    # GEE clusters are graphs: a query-vertex pair is one graph (two query vertices); a global pair is two
    # graphs, so each global instance is its own graph
    df["graph"] = np.where(df["cls"] == "global", df["id"], df["pair"])
    return df


def bal_acc(d):
    return float(np.mean([d.loc[d.label == lab, "correct"].mean() for lab in (True, False)
                          if (d.label == lab).any()]))


def _pair_counts(d):
    """Per matched pair: (#correct positives, #positives, #correct negatives, #negatives) and the question id."""
    g = d.assign(cp=d.correct * d.label, np_=d.label.astype(int), cn=d.correct * (~d.label.astype(bool)),
                 nn=(~d.label.astype(bool)).astype(int)).groupby("pair")
    c = g[["cp", "np_", "cn", "nn"]].sum()
    pid = g["pid"].first()
    return c.values.astype(float), pid.values


def _pair_ids(d):
    """Pair ids in the row order of _pair_counts (groupby sorts them)."""
    return d.groupby("pair")["pid"].first().index.values


def _ba_from_counts(c):
    tp, P, tn, N = c.sum(0)
    rates = [r for r, n in ((tp / P if P else None, P), (tn / N if N else None, N)) if n]
    return float(np.mean(rates))


def _resample_idx(pids, rng):
    """Indices of a pair bootstrap sample, stratified by question."""
    idx = []
    for q in np.unique(pids):
        w = np.flatnonzero(pids == q)
        idx.append(w[rng.integers(0, len(w), len(w))])
    return np.concatenate(idx)


def cell_rng(d):
    """Bootstrap generator seeded by the cell's content (its pairs and their correctness counts), so the same
    records give the same interval in every script (E1 cells reused as E6 zero-shot and E8 default cells)."""
    import zlib
    c, pids = _pair_counts(d)
    key = ",".join(sorted(map(str, d["pair"].unique()))) + "|" + ",".join(map(str, c.sum(0).astype(int)))
    return np.random.default_rng(zlib.crc32(key.encode()))


def pair_bootstrap(d, reps, rng, stat=None):
    """Resample matched pairs with replacement, stratified by question; 95% percentile interval of the
    balanced accuracy (vectorised on per-pair counts). The generator is seeded by the cell's content (cell_rng);
    the `rng` argument is kept for compatibility."""
    rng = cell_rng(d)
    c, pids = _pair_counts(d)
    vals = [_ba_from_counts(c[_resample_idx(pids, rng)]) for _ in range(reps)]
    return np.percentile(vals, [2.5, 97.5])


cluster_bootstrap = pair_bootstrap      # backward-compatible name used by e4_stats.py


def paired_permutation(d, a, b, reps=10_000, seed=0):
    """Graph-level (pair-level) paired permutation test of formats a vs b on correctness."""
    piv = d.pivot_table(index=["pair", "id", "tpl"], columns="ser", values="correct")
    if a not in piv or b not in piv:
        return None
    diff = (piv[a] - piv[b]).groupby(level="pair").mean().values
    obs = diff.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(reps, len(diff)))
    null = (signs * diff).mean(1)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (reps + 1)
    return {"a": a, "b": b, "mean_diff": float(obs), "p": float(p), "n_pairs": int(len(diff))}


def mcnemar(a, b):
    """Exact McNemar on paired 0/1 vectors (kept for reference; the paper uses paired_permutation)."""
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return binomtest(n01, n01 + n10, 0.5).pvalue, n01, n10


def _wald_interaction(m):
    names = list(m.params.index)
    idx = [i for i, n in enumerate(names) if n.startswith("C(ser)") and n.count(":") == 1 and "C(cls)" in n]
    R = np.zeros((len(idx), len(names)))
    for r, i in enumerate(idx):
        R[r, i] = 1
    w = m.wald_test(R, scalar=True)
    return {"wald_chi2": float(w.statistic), "df": len(idx), "p": float(w.pvalue)}


def interaction_test(d, with_length=False, question_fe=False):
    d = d.copy()
    d["ser"] = pd.Categorical(d["ser"], SERIALIZATIONS)
    d["cls"] = pd.Categorical(d["cls"], CLASSES)
    rhs = "C(ser) * C(cls) + C(tpl)"
    if question_fe:          # question fixed effects = class x question-within-class (no collinear columns)
        d["qi"] = d.groupby("cls", observed=True)["pid"].transform(lambda x: pd.factorize(x, sort=True)[0])
        rhs = "C(ser) * C(cls) + C(qi) + C(cls):C(qi) + C(tpl)"
    if with_length:
        ll = np.log(d["n_tokens"].astype(float))
        d["loglen"] = (ll - ll.mean()) / (ll.std() if ll.std() > 0 else 1.0)       # standardized (paper F.4)
        rhs += " + loglen"
    m = smf.gee(f"correct ~ {rhs}", groups="graph", data=d, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    return _wald_interaction(m)


def planned_contrasts(d, reps, rng):
    """Planned contrasts (preregistered, primary): Delta_c(F) = (F-EL)_c - (F-EL)_global for F in {NC, AL}.

    Exactly as in the paper (Statistical Analysis): the pair bootstrap (stratified by question, `reps`
    resamples of matched pairs) gives the percentile interval and the standard error se = sd of the bootstrap
    distribution; the test is the normal approximation z = Delta / se, one-sided (H1: Delta > 0, as
    preregistered), p = 1 - Phi(z); Holm over the six contrasts of a model. The empirical bootstrap p
    = (1 + b) / (reps + 1), with b the number of resamples with Delta <= 0, is reported as well (p_boot)."""
    from scipy.stats import norm
    out = []
    have = set(zip(d.cls, d.ser))
    for f in ("NC", "AL"):
        for cls in ("depth1", "depth2", "count"):
            if not {(cls, f), (cls, "EL"), ("global", f), ("global", "EL")} <= have:
                continue                                   # a class or format absent from this file
            cells = {}
            for cl in (cls, "global"):
                sub = {ser: d[(d.cls == cl) & (d.ser == ser)] for ser in (f, "EL")}
                common = sorted(set(sub[f]["pair"]) & set(sub["EL"]["pair"]))      # inner join on pair ids
                miss = set(sub[f]["pair"]) ^ set(sub["EL"]["pair"])
                if miss:
                    import warnings
                    warnings.warn(f"planned contrast {f}-EL {cl}: {len(miss)} pairs present in one format only; dropped")
                for ser in (f, "EL"):
                    cells[(cl, ser)] = _pair_counts(sub[ser][sub[ser]["pair"].isin(common)])
                assert list(_pair_ids(sub[f][sub[f]["pair"].isin(common)])) == list(_pair_ids(sub["EL"][sub["EL"]["pair"].isin(common)]))

            def delta(sel):
                g = {k: _ba_from_counts(v[0][sel[k[0]]]) for k, v in cells.items()}
                return (g[(cls, f)] - g[(cls, "EL")]) - (g[("global", f)] - g[("global", "EL")])

            full = {cl: np.arange(len(cells[(cl, "EL")][1])) for cl in (cls, "global")}
            obs = delta(full)
            vals = np.array([delta({cl: _resample_idx(cells[(cl, "EL")][1], rng) for cl in (cls, "global")})
                             for _ in range(reps)])
            se = float(np.std(vals, ddof=1))
            z = obs / se if se > 0 else float("inf") * np.sign(obs)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out.append({"format": f, "cls": cls, "delta": float(obs), "ci_low": float(lo), "ci_high": float(hi),
                        "se": se, "z": float(z), "p_one_sided": float(norm.sf(z)),
                        "p_boot": float((1 + np.sum(vals <= 0)) / (reps + 1))})
    if out:
        adj = multipletests([x["p_one_sided"] for x in out], method="holm")[1]
        for x, q in zip(out, adj):
            x["p_holm"] = float(q)
    return out


def distance_regressions(d):
    out = {}
    pos = d[d.pid.isin(["D1a", "D1c"]) & (d.label == True)].copy()      # noqa: E712  (D1a, D1c as in the caption)
    for ser in sorted(pos.ser.unique()):
        c = pos[pos.ser == ser].copy()
        c["loglen"] = np.log(c["n_tokens"].astype(float))
        if c["d_min"].notna().all() and ser in ("AL", "NC", "NCH"):
            c["logd"] = np.log(c["d_min"].astype(float))
            f = "correct ~ logd + k_min + loglen + n + deg_v"
        else:
            f = "correct ~ loglen + n + deg_v"
        if "model" in c and c["model"].nunique() > 1:
            f += " + C(model)"
        try:
            m = smf.logit(f, data=c).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(c["pair"])[0]})
            out[ser] = {k: {"coef": float(m.params[k]), "se": float(m.bse[k]), "p": float(m.pvalues[k])}
                        for k in m.params.index if k != "Intercept" and not k.startswith("C(model)")}
        except Exception as e:                                       # singular design on tiny smoke data
            out[ser] = {"error": str(e)}
    return out


def analyse_model(d, reps, rng, perm_reps):
    out = {"n_prompts": int(len(d)), "unparsed": int(d["unparsed"].sum()),
           "mock": bool(d.get("mock", pd.Series(False)).fillna(False).any()), "cells": []}
    for cls in CLASSES:
        for ser in SERIALIZATIONS:
            c = d[(d.cls == cls) & (d.ser == ser)]
            if c.empty:
                continue
            lo, hi = pair_bootstrap(c, reps, rng)
            out["cells"].append({"cls": cls, "ser": ser, "bal_acc": bal_acc(c), "ci_low": float(lo),
                                 "ci_high": float(hi), "n_pairs": int(c["pair"].nunique()), "n_prompts": int(len(c))})
        for pid in sorted(d.loc[d.cls == cls, "pid"].unique()):         # per question (E-12 reporting)
            for ser in SERIALIZATIONS:
                c = d[(d.pid == pid) & (d.ser == ser)]
                if not c.empty:
                    out.setdefault("per_question", []).append({"pid": pid, "ser": ser, "bal_acc": bal_acc(c)})
    for tpl in sorted(d.tpl.unique()):                                    # per template (E-13)
        for cls in CLASSES:
            for ser in SERIALIZATIONS:
                c = d[(d.tpl == tpl) & (d.cls == cls) & (d.ser == ser)]
                if not c.empty:
                    out.setdefault("per_template", []).append({"tpl": tpl, "cls": cls, "ser": ser, "bal_acc": bal_acc(c)})
    if "gen_pred" in d:                                                    # E8 agreement, needs no other analysis
        g = d[d.gen_pred.notna()]
        out["generation_agreement"] = float((g.gen_pred == g.pred).mean()) if len(g) else None
    if d.cls.nunique() >= 2 and d.ser.nunique() >= 2:       # the GEE interaction needs two classes and two formats
        out["interaction"] = interaction_test(d)
        if d["n_tokens"].notna().all():
            out["interaction_length_adjusted"] = interaction_test(d, with_length=True)
        out["interaction_qfe"] = interaction_test(d, question_fe=True)
    pw = []
    for cls in CLASSES:
        for b, a in itertools.combinations(SERIALIZATIONS, 2):        # later format first: AL-EL, NC-EL, NC-AL
            r = paired_permutation(d[d.cls == cls], a, b, reps=perm_reps)
            if r:
                r["cls"] = cls
                pw.append(r)
    if pw:
        adj = multipletests([x["p"] for x in pw], method="holm")[1]
        for x, q in zip(pw, adj):
            x["p_holm"] = float(q)
    out["pairwise"] = pw
    out["planned_contrasts"] = planned_contrasts(d, reps, rng)
    if "k_min" in d and (d.cls == "depth1").any():
        out["distance_regression"] = distance_regressions(d)
    return out


def pooled_r2_gain(df):
    """S-2: McFadden pseudo-R2 increment of the ser x cls interaction only (pooled over models)."""
    d = df.copy()
    base = smf.logit("correct ~ C(ser) + C(cls) + C(model) + C(tpl)", data=d).fit(disp=0)
    full = smf.logit("correct ~ C(ser) * C(cls) + C(model) + C(tpl)", data=d).fit(disp=0)
    return {"pseudo_r2_base": float(base.prsquared), "pseudo_r2_full": float(full.prsquared),
            "gain_interaction": float(full.prsquared - base.prsquared)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="results/summary.json")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--perm-reps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("E2E_TINY"):               # offline end-to-end test only: few resamples
        args.reps = 50
        if hasattr(args, "perm_reps"):
            args.perm_reps = 50
    rng = np.random.default_rng(args.seed)
    df = load(args.results)
    summary = {m: analyse_model(d, args.reps, rng, args.perm_reps) for m, d in df.groupby("model")}
    models = list(summary)
    for key in ("interaction", "interaction_length_adjusted", "interaction_qfe"):     # Holm across models
        ms = [m for m in models if key in summary[m]]
        if ms:
            adj = multipletests([summary[m][key]["p"] for m in ms], method="holm")[1]
            for m, q in zip(ms, adj):
                summary[m][key]["p_holm_across_models"] = float(q)
    if len(models) > 1 and df.cls.nunique() >= 2 and df.ser.nunique() >= 2:
        summary["_pooled"] = pooled_r2_gain(df)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    for m, s in summary.items():
        if m.startswith("_"):
            continue
        tag = "  [MOCK DATA: pipeline test only]" if s["mock"] else ""
        print(f"\n{m}{tag}  prompts={s['n_prompts']} unparsed={s['unparsed']}")
        if "interaction" in s:
            it = s["interaction"]
            print(f"  ser x cls interaction: chi2={it['wald_chi2']:.2f} df={it['df']} p={it['p']:.3g}")
        if s.get("generation_agreement") is not None:
            print(f"  generation agreement: {s['generation_agreement']:.3f}")
        for c in s["cells"]:
            print(f"  {c['cls']:7s} {c['ser']}  {c['bal_acc']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]")
        for c in s["planned_contrasts"]:
            print(f"  contrast {c['format']}-EL {c['cls']:7s} delta={c['delta']:+.3f} p_holm={c['p_holm']:.3g}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
