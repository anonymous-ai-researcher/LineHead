"""Additional analyses of the LLM records (paper appendix "Statistical Analysis" and "Additional
Analyses"). Every subcommand reads run_llm.py records and writes one JSON summary.

  paired    D11, D13: threshold-free paired metric: for every matched pair and template, P(score(pos) > score(neg))
                with score = log p(Yes) - log p(No) (ties count 1/2), and the TPR and TNR, per model x ser x cls
  position  depth-one positives: logistic GEE (clusters = pairs) correct ~ C(ser) [+ rel + rel^2], rel = relative
                position of the needed line in the serialization (instances.rel_pos); the NC-EL and AL-EL log-odds
                gaps with and without the position terms, and the Wald p of the position terms, per model
  tost      D10: equivalence (two one-sided tests at alpha = 0.05 <=> the 90% pair-bootstrap interval inside the
                margin) for the chance and tie claims, margins from the preregistered rules
  lengths   mean and maximum prompt length in model tokens per model x ser
  kdiff     V-C8: ECL's change across k_min against NC's: Delta_f = BA(k_min = 1) - BA(k_min >= 5) =
                (TPR_1 - TPR_>=5) / 2 on the D1a and D1c positives (EL and ECL use the matched NC prompt's k_min),
                T = (Delta_NC - Delta_EL) - |Delta_ECL - Delta_EL| for the four-model mean, joint pair bootstrap
                (pairs resampled once per question for every model and format), one-sided 90% bound and p
  p4ni      V-C41, V-C42: P4 as non-inferiority: per k_min stratum (1, 2, 3-4, >= 5), the four-model mean shortfall of ECL
                behind the better list format of each model, max(BA_AL, BA_NC) - BA_ECL with BA = (TPR in the stratum +
                TNR) / 2 on D1a and D1c, joint pair bootstrap as kdiff; upper 95% bound (P4 holds if below 0.02).
                kdiff also reports the direct contrast Delta_NC - Delta_ECL with its lower 5% bound.
  qre       question random effects: linear mixed model of correctness on C(ser)*C(cls) + C(tpl) with a random
                intercept per question (statsmodels MixedLM); Wald chi^2 of the 6 interaction terms, per model

Usage (RUN_PROTOCOL.md §4.12):
  python r9_stats.py paired   results/e1_*.jsonl --out results/summary_paired.json
  python r9_stats.py position results/e1_*.jsonl --out results/summary_position.json
  python r9_stats.py tost     --results results --out results/summary_tost.json
  python r9_stats.py lengths  results/e1_*.jsonl results/e4_*.jsonl --out results/summary_lengths.json
  python r9_stats.py qre      results/e1_*.jsonl --out results/summary_qre.json
  python r9_stats.py kdiff    results/e1_*.jsonl results/e4_*_ecl.jsonl --out results/summary_kdiff.json
  python r9_stats.py p4ni     results/e1_*.jsonl results/e4_*_ecl.jsonl --out results/summary_p4ni.json
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

from stats import _pair_counts, _resample_idx, bal_acc, cell_rng, load

warnings.filterwarnings("ignore")
REPS = 2000


def score(d):
    s = d["lp_yes"].astype(float) - d["lp_no"].astype(float) if "lp_yes" in d else pd.Series(np.nan, index=d.index)
    return s.fillna(d["pred"].astype(float) * 2 - 1)        # parsed answers (no log-probabilities): +-1


def paired_auc(c):
    """Mean over (pair, template) of 1[s+ > s-] + 1/2 1[s+ = s-]; query pairs and global pairs alike."""
    c = c.assign(s=score(c))
    pos = c[c.label.astype(bool)].groupby(["pair", "tpl"])["s"].mean()
    neg = c[~c.label.astype(bool)].groupby(["pair", "tpl"])["s"].mean()
    j = pd.concat([pos.rename("p"), neg.rename("n")], axis=1).dropna()
    if j.empty:
        return float("nan")
    return float(((j.p > j.n) + 0.5 * (j.p == j.n)).mean())


def cmd_paired(a):
    d = load(a.results)
    out = {"cells": []}
    for (m, s, c), x in d.groupby(["model", "ser", "cls"]):
        out["cells"].append(dict(model=m, ser=s, cls=c, auc=paired_auc(x), bal_acc=bal_acc(x),
                                 tpr=float(x[x.label.astype(bool)].correct.mean()),
                                 tnr=float(x[~x.label.astype(bool)].correct.mean())))
    return out


def gee_gap(x, with_pos):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    x = x.copy()
    x["ser"] = pd.Categorical(x["ser"], ["EL", "AL", "NC"])
    rhs = "C(ser)" + (" + rel + I(rel ** 2)" if with_pos else "")
    m = smf.gee(f"correct ~ {rhs}", groups="pair", data=x, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Independence()).fit()
    r = {"NC_EL": float(m.params.get("C(ser)[T.NC]", np.nan)), "AL_EL": float(m.params.get("C(ser)[T.AL]", np.nan))}
    if with_pos:
        names = list(m.params.index)
        R = np.zeros((2, len(names)))
        for i, n in enumerate([n for n in names if "rel" in n]):
            R[i, names.index(n)] = 1
        w = m.wald_test(R, scalar=True)
        r.update(pos_chi2=float(w.statistic), pos_p=float(w.pvalue), rel=float(m.params.get("rel", np.nan)))
    return r


def cmd_position(a):
    d = load(a.results)
    d = d[(d.cls == "depth1") & d.label.astype(bool) & d.ser.isin(["EL", "AL", "NC"])].dropna(subset=["rel_pos"])
    d = d.assign(rel=d["rel_pos"].astype(float) - 0.5)
    out = {"models": {}}
    for m, x in d.groupby("model"):
        try:
            base, adj = gee_gap(x, False), gee_gap(x, True)
        except Exception as e:                       # tiny test data: report the failure, never invent numbers
            base, adj = {"error": str(e)}, {"error": str(e)}
        out["models"][m] = {"base": base, "adjusted": adj, "n": int(len(x))}
    return out


def boot_diff(xa, xb, reps, rng):
    """90% pair-bootstrap interval of BA(a) - BA(b); xa, xb share their pairs (same instances) or are
    independent (resampled separately)."""
    ca, pa = _pair_counts(xa)
    cb, pb = _pair_counts(xb)
    from stats import _ba_from_counts
    same = set(xa.pair) == set(xb.pair)
    vals = []
    for _ in range(reps):
        ia = _resample_idx(pa, rng)
        ib = ia if same else _resample_idx(pb, rng)
        vals.append(_ba_from_counts(ca[ia]) - (_ba_from_counts(cb[ib]) if xb is not None else 0.5))
    return [float(v) for v in np.percentile(vals, [5, 95])]


def boot_ci90(x, reps, rng):
    c, p = _pair_counts(x)
    from stats import _ba_from_counts
    vals = [_ba_from_counts(c[_resample_idx(p, rng)]) for _ in range(reps)]
    return [float(v) for v in np.percentile(vals, [5, 95])]


def _files(res, pat):
    return sorted(glob.glob(os.path.join(res, pat)))


def cmd_tost(a):
    """Equivalence claims of the paper and their preregistered margins (VALIDATION_RULES.md)."""
    reps = 50 if os.environ.get("E2E_TINY") else REPS
    R = a.results
    out = {"tests": []}

    def add(name, model, lo, hi, margin, centre=0.0):
        out["tests"].append(dict(claim=name, model=model, ci90=[lo, hi], margin=margin,
                                 equivalent=bool(centre - margin <= lo and hi <= centre + margin)))

    e1 = load(_files(R, "e1_*.jsonl"))
    for m, x in e1[e1.cls == "global"].groupby("model"):            # V-C5: formats tie on global questions
        for f1, f2 in (("NC", "EL"), ("AL", "EL"), ("NC", "AL")):
            lo, hi = boot_diff(x[x.ser == f1], x[x.ser == f2], reps, cell_rng(x[x.ser == f1]))
            add(f"global {f1}-{f2}", m, lo, hi, 0.03)
    q = _files(R, "e2_*_query.jsonl")
    if q:                                                         # V-B3: masking costs global questions <= 0.015
        e2 = load(q)
        for m, x in e2[e2.cls == "global"].groupby("model"):
            for f in ("EL", "AL", "NC"):
                xb = e1[(e1.model == m) & (e1.cls == "global") & (e1.ser == f)]
                xa = x[x.ser == f]
                if len(xa) and len(xb):
                    lo, hi = boot_diff(xa, xb, reps, cell_rng(xa))
                    add(f"masked global {f}", m, lo, hi, 0.015)
    cfiles = _files(R, "e2_*_controls.jsonl")
    if cfiles:                                                    # V-B2: undecidable positive-control cells at chance
        ct = load(cfiles)
        for (m, pid, f), x in ct.groupby(["model", "pid", "ser"]):
            if (pid == "X2" and f == "EL") or (pid == "X1" and f in ("EL", "AL")):
                lo, hi = boot_ci90(x, reps, cell_rng(x))
                add(f"control {pid} {f} at chance", m, lo, hi, 0.03, centre=0.5)
    g = _files(R, "e5_*_graphml.jsonl")
    if g:                                                         # V-C10: GraphML within 0.03 of EL
        gm = load(g)
        for m, x in gm.groupby("model"):
            xb = e1[(e1.model == m) & (e1.cls == "depth1") & (e1.ser == "EL")]
            lo, hi = boot_diff(x, xb, reps, cell_rng(x))
            add("GraphML-EL depth one", m, lo, hi, 0.03)
    e4 = _files(R, "e4_*.jsonl")
    e4 = [p for p in e4 if not p.endswith("_ecl.jsonl")]
    if e4:                                                        # V-C16: NC-nw within 0.03 of NC
        d4 = load(e4)
        for m, x in d4[d4.ser == "NCnw"].groupby("model"):
            xb = e1[(e1.model == m) & (e1.cls == "depth1") & (e1.ser == "NC")]
            lo, hi = boot_diff(x, xb, reps, cell_rng(x))
            add("NC-nw - NC depth one", m, lo, hi, 0.03)
    return out


def cmd_lengths(a):
    d = load(a.results)
    out = {"cells": []}
    for (m, s), x in d.groupby(["model", "ser"]):
        out["cells"].append(dict(model=m, ser=s, mean=float(x.n_tokens.mean()), max=int(x.n_tokens.max())))
    return out


def cmd_qre(a):
    import statsmodels.formula.api as smf
    d = load(a.results)
    out = {"models": {}}
    for m, x in d.groupby("model"):
        x = x.copy()
        x["ser"] = pd.Categorical(x["ser"], ["EL", "AL", "NC"])
        x["cls"] = pd.Categorical(x["cls"], ["global", "depth1", "depth2", "count"])
        try:
            r = smf.mixedlm("correct ~ C(ser) * C(cls) + C(tpl)", x, groups=x["pid"]).fit(reml=False)
            names = list(r.fe_params.index)
            idx = [i for i, n in enumerate(names) if n.startswith("C(ser)") and ":" in n]
            b = r.fe_params.values[idx]
            V = r.cov_params().loc[[names[i] for i in idx], [names[i] for i in idx]].values
            chi2 = float(b @ np.linalg.pinv(V) @ b)
            from scipy.stats import chi2 as C2
            out["models"][m] = {"wald_chi2": chi2, "df": len(idx), "p": float(C2.sf(chi2, len(idx))),
                                "question_sd": float(np.sqrt(max(0.0, float(r.cov_re.iloc[0, 0]))))}
        except Exception as e:
            out["models"][m] = {"error": str(e)}
    return out


def cmd_kdiff(a):
    reps = 50 if os.environ.get("E2E_TINY") else REPS
    d = load(a.results)
    d = d[(d.cls == "depth1") & d.pid.isin(["D1a", "D1c"]) & d.ser.isin(["EL", "NC", "ECL"]) & d.label.astype(bool)]
    models = sorted(d.model.unique())
    # per (model, ser, pair): mean correctness of the positive prompts (templates) and the pair's k_min
    g = d.groupby(["model", "ser", "pair"]).agg(c=("correct", "mean"), k=("k_min", "first"), q=("pid", "first")).reset_index()
    pairs = g[["pair", "q"]].drop_duplicates("pair").sort_values("pair")
    rng = np.random.default_rng(12)
    draws = [np.concatenate([rng.choice(pairs.pair[pairs.q == q].values, (pairs.q == q).sum()) for q in sorted(pairs.q.unique())])
             for _ in range(reps)]

    def delta(x, sample=None):
        x = x.set_index("pair")
        if sample is not None:
            x = x.reindex(sample).dropna(subset=["c"])
        one, five = x[x.k == 1].c.mean(), x[x.k >= 5].c.mean()
        return 0.5 * (one - five)

    def stat(sample=None):
        D = {f: np.mean([delta(g[(g.model == m) & (g.ser == f)], sample) for m in models]) for f in ("EL", "NC", "ECL")}
        return (D["NC"] - D["EL"]) - abs(D["ECL"] - D["EL"]), D
    t0, D0 = stat()
    st = [stat(s_) for s_ in draws]
    T = np.array([x[0] for x in st])
    T = T[~np.isnan(T)]
    DD = np.array([x[1]["NC"] - x[1]["ECL"] for x in st])      # direct contrast NC minus ECL
    DD = DD[~np.isnan(DD)]
    return dict(nc_minus_el=float(D0["NC"] - D0["EL"]), ecl_minus_el_abs=float(abs(D0["ECL"] - D0["EL"])), stat=float(t0),
                nc_minus_ecl=float(D0["NC"] - D0["ECL"]), nc_minus_ecl_lo90=float(np.percentile(DD, 5)),
                lo90=float(np.percentile(T, 5)), hi90=float(np.percentile(T, 95)), p_one_sided=float(np.mean(T <= 0)),
                by_model={m: {f: float(delta(g[(g.model == m) & (g.ser == f)])) for f in ("EL", "NC", "ECL")} for m in models})


def cmd_p4ni(a):
    """P4 as a non-inferiority test per k_min stratum (see the module docstring)."""
    reps = 50 if os.environ.get("E2E_TINY") else REPS
    d = load(a.results)
    d = d[(d.cls == "depth1") & d.pid.isin(["D1a", "D1c"]) & d.ser.isin(["AL", "NC", "ECL"])]
    models = sorted(d.model.unique())
    g = d.groupby(["model", "ser", "pair", "label"]).agg(c=("correct", "mean"), k=("k_min", "first"), q=("pid", "first")).reset_index()
    pairs = g[["pair", "q"]].drop_duplicates("pair").sort_values("pair")
    rng = np.random.default_rng(15)
    draws = [np.concatenate([rng.choice(pairs.pair[pairs.q == q].values, (pairs.q == q).sum()) for q in sorted(pairs.q.unique())])
             for _ in range(reps)]
    SB = {"1": lambda k: k == 1, "2": lambda k: k == 2, "3-4": lambda k: (k >= 3) & (k <= 4), ">=5": lambda k: k >= 5}

    def short(sample=None):
        out = {s_: 0.0 for s_ in SB}
        for m in models:
            ba = {}
            for f in ("AL", "NC", "ECL"):
                x = g[(g.model == m) & (g.ser == f)].set_index("pair")
                pos, neg = x[x.label.astype(bool)], x[~x.label.astype(bool)]
                if sample is not None:
                    pos, neg = pos.reindex(sample).dropna(subset=["c"]), neg.reindex(sample).dropna(subset=["c"])
                tnr = neg.c.mean()
                for s_, fn in SB.items():
                    ba[(f, s_)] = 0.5 * (pos[fn(pos.k.astype(float))].c.mean() + tnr)
            for s_ in SB:
                out[s_] += (max(ba[("AL", s_)], ba[("NC", s_)]) - ba[("ECL", s_)]) / len(models)
        return out
    s0 = short()
    bs = [short(s_) for s_ in draws]
    return {s_: dict(shortfall=float(s0[s_]), hi95=float(np.nanpercentile([b[s_] for b in bs], 95))) for s_ in SB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["paired", "position", "tost", "lengths", "qre", "kdiff", "p4ni"])
    ap.add_argument("results", nargs="*")
    ap.add_argument("--results", dest="results_dir", default="results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "tost":
        a.results = a.results_dir
    out = globals()["cmd_" + a.cmd](a)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
