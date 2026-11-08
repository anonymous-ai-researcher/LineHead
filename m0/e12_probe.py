"""E12: since probe and a zero-parameter prediction of the E1 k_min strata (CONFIG.md §6.3).

run      Score the probe prompts of data/e12_probe.jsonl (build_r12.py e12) with run_llm.py's backends: one record
         per prompt (label, ser, k, d, lp_yes, lp_no, pred).
summary  r(k): per model and format, balanced accuracy of the probe pairs at each k (95% pair-bootstrap interval),
         and its true positive and true negative rates.
predict  Zero-parameter prediction of the E1 depth-one strata under AL and NC (the formats whose route is a since
         step through a line head; EL and ECL have none, so the probe predicts no change across k_min for them).
         For every D1a/D1c positive of the frozen instance file, with red neighbors u_1..u_m in whose lists v sits at
         positions k_1..k_m (the list order the E1 prompt uses), the probability of answering Yes is
         t(k_min) (shared reading: the earliest entry decides, as in the route model), t = the probe's true positive
         rate at k (linear in k between the probed values, constant beyond the last); negatives are answered No with the probe's true negative rate (pooled over k).
         The predicted balanced accuracy of a stratum is (mean predicted TPR of its positives + TNR) / 2. No
         parameter is fitted to E1. The error is the mean |predicted - measured| over model x format x stratum
         (measured: results/summary_e1_strata.json); it is reported, not tested against a threshold.
Usage:
  python e12_probe.py run --backend hf --model Qwen/Qwen2.5-7B-Instruct --probe data/e12_probe.jsonl --out results/e12_qwen7b.jsonl
  python e12_probe.py summary results/e12_*.jsonl --out results/summary_e12.json
  python e12_probe.py predict --probe-summary results/summary_e12.json --strata results/summary_e1_strata.json \
      --instances data/m0_instances.jsonl --out results/summary_e12_predict.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from instances import SER_OFFSET, load, to_graph
from serialize import list_positions

KBINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 99, ">=5")]


def backend(a):
    from run_llm import HF, Mock
    return HF(a.model, batch=a.batch) if a.backend == "hf" else Mock()


def cmd_run(a):
    recs = load(a.probe)
    m = backend(a)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        import time
        for i in range(0, len(recs), 64):
            ch = recs[i:i + 64]
            t0 = time.time()
            res = m([r["prompt"] for r in ch])
            sec = round((time.time() - t0) / len(ch), 4)
            for r, s in zip(ch, res):
                f.write(json.dumps({**{k: r[k] for k in ("id", "pid", "cls", "pair", "label", "ser", "tpl", "k", "d")},
                                    **{k: s.get(k) for k in ("pred", "lp_yes", "lp_no", "n_tokens")}, "model": a.model,
                                    "backend": a.backend, "sec": sec, "n_gpus": a.n_gpus}) + "\n")
    print("wrote", out)


def cmd_summary(a):
    from stats import bal_acc, cell_rng, load as load_recs, pair_bootstrap
    d = load_recs(a.results)
    reps = 50 if os.environ.get("E2E_TINY") else 2000
    out = {"cells": []}
    for (m, s, k), x in d.groupby(["model", "ser", "k"]):
        lo, hi = pair_bootstrap(x, reps, cell_rng(x))
        out["cells"].append(dict(model=m, ser=s, k=int(k), d=int(x.d.iloc[0]), bal_acc=bal_acc(x), ci_low=float(lo),
                                 ci_high=float(hi), tpr=float(x[x.label.astype(bool)].correct.mean()),
                                 tnr=float(x[~x.label.astype(bool)].correct.mean()), n_pairs=int(x.pair.nunique())))
    return out


def t_of(curve, k):
    ks = sorted(curve)
    if k >= ks[-1]:
        return curve[ks[-1]]
    return float(np.interp(k, ks, [curve[x] for x in ks]))


def kpositions(recs):
    """(ser, id) -> positions of v in the lists of its red neighbors, under the list order of the E1 prompt."""
    pos = {}
    for r in recs:
        G = to_graph(r)
        for s in ("AL", "NC"):
            lists = list_positions(G, s, r["graph_seed"] + SER_OFFSET[s])
            pos[(s, r["id"])] = [ws.index(r["query"]) + 1 for u, ws in lists if G.nodes[u]["color"] == "red" and r["query"] in ws]
    return pos


def predict(P, S, pos):
    """P: probe summary (cells: model, ser, k, tpr, tnr); S: E1 strata summary (cells: model, ser, strata: [{stratum,
    bal_acc}]). Shared reading, as in the route model: a positive is read through its earliest entry, so the
    probability of Yes is t(k_min). Returns the predicted and measured strata and the mean absolute error."""
    meas = {}
    for c in S["cells"]:
        for st in c.get("strata", []):
            meas[(c["model"], c["ser"], st["stratum"])] = st["bal_acc"]
    out, errs = {"cells": []}, []
    for m in sorted({c["model"] for c in P["cells"]}):
        for s in ("AL", "NC"):
            cs = [c for c in P["cells"] if c["model"] == m and c["ser"] == s]
            if not cs:
                continue
            tpr = {c["k"]: c["tpr"] for c in cs}
            tnr = float(np.mean([c["tnr"] for c in cs]))
            for lo, hi, lab in KBINS:
                ps = [t_of(tpr, min(ks)) for (s_, _), ks in pos.items() if s_ == s and ks and lo <= min(ks) <= hi]
                if not ps:
                    continue
                pred = 0.5 * (float(np.mean(ps)) + tnr)
                ms = meas.get((m, s, {"1": "k1", "2": "k2", "3-4": "k3-4", ">=5": "k5+"}[lab]))
                cell = dict(model=m, ser=s, stratum=lab, pred=pred, measured=ms)
                if ms is not None:
                    cell["err"] = pred - ms
                    errs.append(abs(pred - ms))
                out["cells"].append(cell)
    out["error_pts"] = 100 * float(np.mean(errs)) if errs else None
    out["max_error_pts"] = 100 * float(np.max(errs)) if errs else None
    return out


def cmd_predict(a):
    recs = [r for r in load(a.instances) if r["pid"] in ("D1a", "D1c") and r["label"]]
    return predict(json.load(open(a.probe_summary)), json.load(open(a.strata)), kpositions(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "predict"])
    ap.add_argument("results", nargs="*")
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-gpus", type=int, default=1, help="GPUs the run occupies (72B: 4), for T_compute")
    ap.add_argument("--probe", default="data/e12_probe.jsonl")
    ap.add_argument("--probe-summary", default="results/summary_e12.json")
    ap.add_argument("--strata", default="results/summary_e1_strata.json")
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
        return
    res = cmd_summary(a) if a.cmd == "summary" else cmd_predict(a)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
