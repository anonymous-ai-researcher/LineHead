"""Analyses of the E3 runs (paper appendix "Positional Mechanisms" and "Additional Analyses").

  curves  C18 learning speed under exact since (local_hardsince, task dia): per format and seed, the first
              evaluation step at which the ID-validation balanced accuracy reaches 0.9 (runs that never reach it
              count as the last step plus one evaluation interval, censored) and the mean validation accuracy over
              the curve; OLS of log(steps to 0.9) on the number of nested matches of the format's construction
              (EL 2; AL, NC, ECL 1), with a seed-bootstrap 95% interval of the slope.
  kreg    C20 k-controlled sets, positives: two logistic regressions, A on log k_min and log line length (b_logk),
              B on one indicator per k_min and log line length (b_loglen); SEs clustered by test instance
              (degree of the red neighbor attaining k_min) per arm with approximate since (local_alibi,
              alibi_only, rope) and format (AL, NC), pooled over seeds; Wald z of each coefficient.
  extra   D12, B16  in-distribution k_min strata (acc_kid) per arm and format, the own-line set (acc_kown,
              1-local AL at L = 6 from e3.jsonl and L = 8 from e3d.jsonl), the NoPE / RoPE learning-rate check,
              and the NC-H arm.
Usage: python r9_e3.py --e3 results/e3.jsonl --e3d results/e3d.jsonl --lr results/lr_check.jsonl \
           --nch results/e3_nch.jsonl --kx results/kset_examples.jsonl --out results/summary_r9_e3.json
"""
import argparse
import json
import math
import os

import numpy as np

NESTED = {"EL": 2, "AL": 1, "NC": 1, "ECL": 1}
ARMS = {"nope": "NoPE", "local": "1-local", "alibi_only": "ALiBi-only", "local_alibi": "1-local+ALiBi",
        "local_hardsince": "1-local+hard-since", "rope": "RoPE"}


def load(p):
    return [json.loads(l) for l in open(p)] if p and os.path.exists(p) else []


def steps_to(curve, thr=0.9):
    for c in curve:
        if c.get("val_bal_acc", 0) >= thr:
            return c["step"], False
    last = curve[-1]["step"] if curve else 0
    step = curve[1]["step"] - curve[0]["step"] if len(curve) > 1 else last
    return last + step, True


def curves(runs):
    rows = [r for r in runs if r["pos"] == "local_hardsince" and r["task"] == "dia" and r.get("layers", 6) == 6]
    out = {"by_format": {}}
    xs, ys, seeds = [], [], []
    for f in NESTED:
        fr = [r for r in rows if r["ser"] == f]
        st = [steps_to(r["curve"]) for r in fr]
        out["by_format"][f] = dict(steps90=[s for s, _ in st], censored=sum(c for _, c in st),
                                   mean_steps90=float(np.mean([s for s, _ in st])) if st else None,
                                   auc=float(np.mean([np.mean([c["val_bal_acc"] for c in r["curve"]]) for r in fr])) if fr else None)
        for r, (s, _) in zip(fr, st):
            xs.append(NESTED[f]); ys.append(math.log(s)); seeds.append(r["seed"])
    xs, ys, seeds = map(np.array, (xs, ys, seeds))

    def slope(x, y):
        return float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else float("nan")
    b = slope(xs, ys)
    rng = np.random.default_rng(0)
    us = np.unique(seeds)
    boots = []
    for _ in range(50 if os.environ.get("E2E_TINY") else 2000):
        pick = rng.choice(us, len(us))
        idx = np.concatenate([np.flatnonzero(seeds == s) for s in pick])
        boots.append(slope(xs[idx], ys[idx]))
    boots = [v for v in boots if not math.isnan(v)]
    out["slope_log_steps"] = b
    out["slope_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [b, b]
    out["ratio_EL"] = float(math.exp(b)) if not math.isnan(b) else None
    return out


def kreg(kx):
    import statsmodels.api as sm
    out = {}
    for pos in ("local_alibi", "alibi_only", "rope"):
        for f in ("AL", "NC"):
            T = [t for r in kx if r["pos"] == pos and r["ser"] == f and r.get("layers", 6) == 6
                 for t in r["pos_k_len_correct"] if t[1]]
            if len(T) < 10:
                continue
            k, L, c = (np.array([t[j] for t in T], dtype=float) for j in range(3))
            # standard errors clustered by test instance (the 4th field, the index of the example in the
            # k-controlled set), since every seed of an arm scores the same instances; without ids each positive
            # is its own cluster (sandwich standard errors)
            g = np.array([t[3] if len(t) > 3 else i for i, t in enumerate(T)])
            ck = dict(cov_type="cluster", cov_kwds={"groups": np.unique(g, return_inverse=True)[1]})
            X = sm.add_constant(np.column_stack([np.log(k), np.log(L)]))
            # the line-length coefficient is estimated within each value of k_min (one indicator per value),
            # so a k_min effect that is not linear in log k cannot load on the line length, which is >= k_min
            ks = sorted(set(k.tolist()))
            XW = np.column_stack([(k == v).astype(float) for v in ks] + [np.log(L)])
            try:
                m = sm.GLM(c, X, family=sm.families.Binomial()).fit(**ck)       # model A: log k and log line length
                w = sm.GLM(c, XW, family=sm.families.Binomial()).fit(**ck)      # model B: k indicators and log line length
                out[f"{ARMS[pos]}|{f}"] = dict(b_logk=float(m.params[1]), z_logk=float(m.tvalues[1]),
                                               b_loglen=float(w.params[-1]), z_loglen=float(w.tvalues[-1]), n=int(len(c)))
            except Exception as e:       # no variation in correctness (tiny test data): no effect is estimable
                print(f"WARNING kreg {pos} {f}: {e}; coefficients set to 0")
                out[f"{ARMS[pos]}|{f}"] = dict(b_logk=0.0, z_logk=0.0, b_loglen=0.0, z_loglen=0.0, n=int(len(c)))
    return out


def ms(v):
    v = [x for x in v if x is not None]
    return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, seeds=v) if v else None


def extra(e3, e3d, lr, nch):
    out = {"kid": {}, "kown": {}, "lr_check": {}, "nch": {}, "d256": {}, "d256_by": {}, "hard": {}, "hardp": {}}
    for pos in ARMS:                       # Delta <= 256 test set, every arm under AL and NC (6 layers)
        for f in ("AL", "NC"):
            rs = [r for r in e3 if r["pos"] == pos and r["ser"] == f and r["task"] == "dia" and r.get("layers", 6) == 6
                  and "acc_d256" in r]
            if rs:
                out["d256"][f"{ARMS[pos]}|{f}"] = ms([r["acc_d256"] for r in rs])
                for D_ in (64, 128, 192, 256):     # by hub degree (equal shares in gen_e3.py)
                    out["d256_by"][f"{ARMS[pos]}|{f}|{D_}"] = ms([r.get("acc_d256_by", {}).get(str(D_)) for r in rs])
            rs = [r for r in e3 if r["pos"] == pos and r["ser"] == f and r["task"] == "dia" and r.get("layers", 6) == 6
                  and "acc_hard" in r]
            if rs:                         # hard pairs (Theorem lb:main)
                out["hard"][f"{ARMS[pos]}|{f}"] = ms([r["acc_hard"] for r in rs])
            for L_ in (4, 6, 8):           # proof's construction at the model's depth (main runs: L = 6)
                rs = [r for r in e3 + e3d if r["pos"] == pos and r["ser"] == f and r["task"] == "dia"
                      and r.get("layers", 6) == L_ and "acc_hardp" in r]
                if rs:
                    out["hardp"][f"{ARMS[pos]}|{f}|{L_}"] = ms([r["acc_hardp"] for r in rs])
    for pos, arm in ARMS.items():
        for f in ("EL", "AL", "NC", "ECL"):
            rs = [r for r in e3 if r["pos"] == pos and r["ser"] == f and r["task"] == "dia" and "acc_kid" in r]
            if rs:
                out["kid"][f"{arm}|{f}"] = {k: ms([r["acc_kid"].get(k) for r in rs]) for k in ("1", "2", "3", "4-6")}
    for L, src in ((6, e3), (8, e3d)):
        rs = [r for r in src if r["pos"] == "local" and r["ser"] == "AL" and r["task"] == "dia"
              and r.get("layers", 6) == L and "acc_kown" in r]
        if rs:
            out["kown"][str(L)] = ms([r["acc_kown"] for r in rs])
    for r in lr:
        LRK = {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}
        out["lr_check"][f"{ARMS[r['pos']]}|{LRK.get(float(r['lr']), str(r['lr']))}"] = dict(id=r.get("acc_id"), ood=r.get("acc_ood"),
                                                                val=r["curve"][-1].get("val_bal_acc") if r["curve"] else None)
    for pos in ("local", "local_hardsince"):
        for task in ("red", "dia"):
            rs = [r for r in nch if r["pos"] == pos and r["task"] == task]
            if rs:
                out["nch"][f"{ARMS[pos]}|{task}"] = dict(id=ms([r.get("acc_id") for r in rs]), ood=ms([r.get("acc_ood") for r in rs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", default="results/e3.jsonl")
    ap.add_argument("--e3d", default="results/e3d.jsonl")
    ap.add_argument("--lr", default="results/lr_check.jsonl")
    ap.add_argument("--nch", default="results/e3_nch.jsonl")
    ap.add_argument("--kx", default="results/kset_examples.jsonl")
    ap.add_argument("--out", default="results/summary_r9_e3.json")
    a = ap.parse_args()
    e3 = load(a.e3)
    res = {"curves": curves(e3), "kreg": kreg(load(a.kx)), **extra(e3, load(a.e3d), load(a.lr), load(a.nch))}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res)[:500])


if __name__ == "__main__":
    main()
