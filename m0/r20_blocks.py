"""CONFIG.md §6.14: build the measured blocks e22-e33 from the raw result files, in the structure of
analysis/schema.json, for analysis/import_measured.py --r20. Only cells whose files exist are written; cells
that restate another block (the E1, E10, E11a, E11b-L and E20 cells beside the new conditions, the T1 losses and the
k_min-share predictions) are filled by the importer from the merged data, not here.

Inputs (default folders: m0/results and e3/results)
  e3/results/r20_pos-control.jsonl, r20_data-limited.jsonl, r20_window.jsonl, r20_nest.jsonl, r20_depth2.jsonl,
  r20_robust.jsonl, r20_nest-dist.jsonl (train_e3.py, one line per run), e3/results/e3d.jsonl (seeds 0-2 of the
  depth sweep)
  m0/results/e24.jsonl (lora_r20.py; its ELB runs go to block e33), e28a_*.jsonl, e28b_*.json, e28c_qwen72b.jsonl, e28d_*.jsonl, e29_*.jsonl,
  e30_*.jsonl, e31e_*.jsonl, e31h_*.jsonl (r20.py), and the run_llm.py records e31a_*.jsonl (chain of thought,
  variant cot), e31a_r1.jsonl, e31b_gpt4o.jsonl, e31c_*.jsonl (variants num4, compound), e31d_*.jsonl (pairs
  L<n>-p<k>-<third>), e31e_shots<K>_*.jsonl and e31e_sorted_*.jsonl, e31f_*.jsonl (variant <size>|<kind>),
  e31g_*.jsonl, e31i_<src>_*.jsonl, e31j_*.jsonl, e33c_*.jsonl (ELB, global and depth-one questions)
Usage: python r20_blocks.py --out results/blocks_r20.json
"""
import argparse
import glob
import json
import math
import random
import re
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

from stats import pair_bootstrap

HERE = Path(__file__).resolve().parent
ARM = {"rope": "RoPE", "alibi_only": "ALiBi-only", "local_alibi": "1-local+ALiBi", "local_hardsince": "1-local+hard-since",
       "local": "1-local", "nope": "NoPE", "dlocal": "8-local", "lastglobal": "last-global"}
F4 = ("EL", "AL", "NC", "ECL")
F5 = F4 + ("NCH",)
NS = (25000, 50000, 100000, 200000, 500000)
KS = ("1", "2", "3", "4-6", "7-12", "13-32")
MODEL = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B", "qwen2.5-7b": "Qwen2.5-7B",
         "llama-3.1-8b": "Llama-3.1-8B", "gemma-2-9b": "Gemma-2-9B", "qwen2.5-72b": "Qwen2.5-72B", "gpt-4o": "GPT-4o",
         "bloom-7b1": "BLOOM-7B1", "mpt-7b": "MPT-7B"}


def mkey(name):
    s = str(name).split("/")[-1].lower()
    for k in sorted(MODEL, key=len, reverse=True):
        if s.startswith(k):
            return MODEL[k]
    return str(name)


def fkey(s):
    return {"nc-h": "NCH", "nch": "NCH", "elb": "ELB"}.get(str(s).lower(), str(s).upper() if str(s).upper() in F4 else str(s))


def runs(path):
    p = Path(path)
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def seeds(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    return {"seeds": xs, "mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0}


RES = [HERE / "results"]


def records(pattern):
    rows = []
    for fn in sorted(glob.glob(str(RES[0] / pattern))):
        rows += [json.loads(l) for l in open(fn) if l.strip()]
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["correct"] = (d["pred"] == d["label"]).astype(int)
    if "pid" not in d:
        d["pid"] = "x"
    return d


def ba_cell(d):
    if d is None or not len(d) or d.label.nunique() < 2:
        return None
    ba = float(np.mean([d.loc[d.label == lab, "correct"].mean() for lab in (True, False)]))
    lo, hi = pair_bootstrap(d, 2000, None)
    return {"ba": round(ba, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def per_pair(d):
    g = d.groupby(["pair", "label"])["correct"].mean().unstack()
    return 0.5 * (g[True] + g[False])


def pdiff(da, db, seed=0):
    a, b = per_pair(da), per_pair(db)
    ks = sorted(set(a.index) & set(b.index))
    x = np.array([a[k] - b[k] for k in ks])
    rs = np.random.default_rng(seed)
    bs = x[rs.integers(0, len(x), (2000, len(x)))].mean(1)
    return {"diff": round(float(x.mean()), 4), "lo95": round(float(np.quantile(bs, 0.025)), 4),
            "hi95": round(float(np.quantile(bs, 0.975)), 4), "hi95_1s": round(float(np.quantile(bs, 0.95)), 4)}


def n95(points):
    if points and points[0][1] >= 0.95:
        return points[0][0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 < 0.95 <= y1:
            return round(math.exp(math.log(x0) + (0.95 - y0) / (y1 - y0) * (math.log(x1) - math.log(x0))))
    return None


def e3_blocks(E3, out):
    # E22
    R = runs(E3 / "r20_pos-control.jsonl")
    for task in ("g1", "c3"):
        for f in F4:
            for sp in ("id", "ood"):
                v = seeds([r[f"acc_{sp}"] for r in R if r["task"] == task and r["ser"] == f])
                if v:
                    out["e22"][f"{task}|{f}|{sp}"] = v
    # E23
    R = runs(E3 / "r20_data-limited.jsonl")
    for pos, arm in ARM.items():
        if not any(r["pos"] == pos for r in R):
            continue
        for f in F4:
            for N in NS:
                for sp in ("id", "ood"):
                    v = seeds([r[f"acc_{sp}"] for r in R if r["pos"] == pos and r["ser"] == f and r.get("train_size") == N])
                    if v:
                        out["e23"]["cells"][f"{arm}|{f}|{N}|{sp}"] = v
            pts = [(N, out["e23"]["cells"][f"{arm}|{f}|{N}|id"]["mean"]) for N in NS if f"{arm}|{f}|{N}|id" in out["e23"]["cells"]]
            out["e23"]["n95"][f"{arm}|{f}"] = n95(pts)
            st95 = []
            for r in sorted((r for r in R if r["pos"] == pos and r["ser"] == f and r.get("train_size") == 500000), key=lambda r: r["seed"]):
                st95.append(next((c["step"] for c in r.get("curve", []) if c.get("val_bal_acc", 0) >= 0.95), None))
            out["e23"]["steps95"][f"{arm}|{f}"] = st95
        el = [x for x in out["e23"]["steps95"].get(f"{arm}|EL", []) if x is not None]
        for g in ("AL", "NC"):
            ot = [x for x in out["e23"]["steps95"].get(f"{arm}|{g}", []) if x is not None]
            if el and ot:
                rng = random.Random(f"e23|{arm}|{g}")
                bs = sorted(st.mean(rng.choices(el, k=len(el))) - st.mean(rng.choices(ot, k=len(ot))) for _ in range(10000))
                out["e23"]["steps_diff"][f"{arm}|EL-{g}"] = {"mean": st.mean(el) - st.mean(ot), "lo": bs[250], "hi": bs[9749]}
    # E33a: ELB under the four arms of E23 (the other formats are the E23 cells)
    R = runs(E3 / "r20_nest-dist.jsonl")
    for pos, arm in ARM.items():
        if not any(r["pos"] == pos and r["ser"] == "ELB" for r in R):
            continue
        for N in NS:
            for sp in ("id", "ood"):
                v = seeds([r[f"acc_{sp}"] for r in R if r["pos"] == pos and r["ser"] == "ELB" and r.get("train_size") == N])
                if v:
                    out["e33"]["cells"][f"{arm}|ELB|{N}|{sp}"] = v
        pts = [(N, out["e33"]["cells"][f"{arm}|ELB|{N}|id"]["mean"]) for N in NS if f"{arm}|ELB|{N}|id" in out["e33"]["cells"]]
        out["e33"]["n95"][f"{arm}|ELB"] = n95(pts)
        out["e33"]["steps95"][f"{arm}|ELB"] = [
            next((c["step"] for c in r.get("curve", []) if c.get("val_bal_acc", 0) >= 0.95), None)
            for r in sorted((r for r in R if r["pos"] == pos and r["ser"] == "ELB" and r.get("train_size") == 500000), key=lambda r: r["seed"])]
    # E25
    R = runs(E3 / "r20_window.jsonl")
    for w in (2, 4, 8):
        for L in (3, 4, 6):
            for f in ("AL", "NC"):
                rr = [r for r in R if r.get("window") == w and r["layers"] == L and r["ser"] == f]
                for lab in KS:
                    v = seeds([r["acc_k"].get(lab) for r in rr if r.get("acc_k")])
                    if v:
                        out["e25"]["cells"][f"w{w}|L{L}|{f}|{lab}"] = v
    # E26
    R = runs(E3 / "r20_nest.jsonl")
    for pos in ("lastglobal", "local"):
        for f in ("EL", "ECL"):
            for E in (8, 16, 32, 64, 128):
                v = seeds([r.get(f"acc_pm{E}") for r in R if r["pos"] == pos and r["ser"] == f])
                if v:
                    out["e26"][f"{ARM[pos]}|{f}|{E}"] = v
    # E27
    R = runs(E3 / "r20_depth2.jsonl")
    for pos in ("dlocal", "local_hardsince", "rope", "local"):
        for task in ("dd", "c2"):
            for f in F4:
                for sp in ("id", "ood"):
                    v = seeds([r[f"acc_{sp}"] for r in R if r["pos"] == pos and r["task"] == task and r["ser"] == f])
                    if v:
                        out["e27"][f"{ARM[pos]}|{task}|{f}|{sp}"] = v
    # E32
    R = runs(E3 / "r20_robust.jsonl")
    D = runs(E3 / "e3d.jsonl")
    for L in (2, 3, 4, 8, 12):
        for f in F4:
            for lab in KS:
                rr = [r for r in D if r["layers"] == L and r["ser"] == f and r.get("seed") in (0, 1, 2)] + \
                     [r for r in R if r["pos"] == "local" and r.get("d", 256) == 256 and r["layers"] == L and r["ser"] == f]
                v = seeds([r["acc_k"].get(lab) for r in rr if r.get("acc_k")])
                if v:
                    out["e32"]["depth"][f"{L}|{f}|{lab}"] = v
    for d in (128, 512):
        for pos in ("local", "local_hardsince"):
            for f in F4:
                for lab in KS:
                    v = seeds([r["acc_k"].get(lab) for r in R if r.get("d") == d and r["pos"] == pos and r["ser"] == f and r.get("acc_k")])
                    if v:
                        out["e32"]["width"][f"{d}|{ARM[pos]}|{f}|{lab}"] = v


def llm_blocks(out):
    # E24 (LoRA): balanced accuracy per seed, then mean and s.d. over seeds
    d = records("e24.jsonl")
    if d is not None:
        for (m, f, N, sp), g in d.groupby(["model", "ser", "n_train", "split"]):
            v = seeds([ba_cell(gs)["ba"] for _, gs in g.groupby("seed")])
            blk = "e33" if fkey(f) == "ELB" else "e24"          # E33b: the ELB runs of lora_r20.py
            (out[blk]["lora"] if blk == "e33" else out[blk])[f"{mkey(m)}|{fkey(f)}|{int(N)}|{sp}"] = v
    # E33c: ELB on the global and depth-one questions (run_llm.py records, T1)
    d = records("e33c_*.jsonl")
    if d is not None:
        for (m, pid), g in d[d.ser.str.upper() == "ELB"].groupby(["model", "pid"]):
            out["e33"]["llm"][mkey(m)][pid] = ba_cell(g)
    # E28
    d = records("e28a_*.jsonl")
    if d is not None:
        for (m, tpl, v, f, pid), g in d.groupby(["model", "tpl", "variant", "ser", "pid"]):
            c = ba_cell(g)
            cond = v.replace("abl-", "")
            if cond.startswith("rand16s"):
                continue
            out["e28"]["repl"][mkey(m)][f"{tpl}|{cond}|{fkey(f)}|{pid}"] = c
        for (m, tpl, f, pid), g in d[d.variant.str.startswith("abl-rand16s")].groupby(["model", "tpl", "ser", "pid"]):
            xs = [ba_cell(gs)["ba"] for _, gs in g.groupby("variant")]
            out["e28"]["repl"][mkey(m)][f"{tpl}|rand16|{fkey(f)}|{pid}"] = {"mean": round(st.mean(xs), 4), "lo": min(xs), "hi": max(xs)}
    for fn in sorted(glob.glob(str(RES[0] / "e28b_*.json"))):
        S = json.load(open(fn))
        sp = S["splits"]
        out["e28"]["stability"][mkey(S["model"])] = {"splits": sp, "jac16": round(st.mean(x["jac16"] for x in sp), 4),
                                                     "jac128": round(st.mean(x["jac128"] for x in sp), 4)}
    d = records("e28c_*.jsonl")
    if d is not None:
        for (v, f, pid), g in d.groupby(["variant", "ser", "pid"]):
            out["e28"]["ko72"][f"{v.replace('ko-', '')}|{fkey(f)}|{pid}"] = ba_cell(g)
    d = records("e28d_*.jsonl")
    if d is not None:
        for m, gm in d.groupby("model"):
            for f, gf in gm.groupby("ser"):
                c = {v: ba_cell(g) for v, g in gf.groupby("variant")}
                R_ = out["e28"]["restore"][mkey(m)]
                for v in ("clean", "flip", "restore128"):
                    if v in c:
                        R_[f"{v}|{fkey(f)}"] = c[v]
                rs = [c[v]["ba"] for v in c if v.startswith("restore-rand128s")]
                if rs:
                    R_[f"restore-rand128|{fkey(f)}"] = {"mean": round(st.mean(rs), 4), "lo": min(rs), "hi": max(rs)}
                if all(v in c for v in ("clean", "flip", "restore128")):
                    den = c["clean"]["ba"] - c["flip"]["ba"]
                    out["e28"]["recovery"][f"{mkey(m)}|{fkey(f)}"] = {
                        "top": round((c["restore128"]["ba"] - c["flip"]["ba"]) / den, 4) if den else None,
                        "rand": round((st.mean(rs) - c["flip"]["ba"]) / den, 4) if den and rs else None}
    # E29
    d = records("e29_*.jsonl")
    if d is not None:
        d["D"] = d["degree_cell"].astype(str).str.extract(r"(\d+)")[0]
        for (m, v, D), g in d.groupby(["model", "variant", "D"]):
            out["e29"][mkey(m)][f"{v}|{int(D)}"] = ba_cell(g)
    # E30
    d = records("e30_*.jsonl")
    if d is not None:
        for (m, f), g in d.groupby(["model", "ser"]):
            gi, gb = g[g.variant == "fill-inside24"], g[g.variant == "fill-between24"]
            out["e30"][mkey(m)]["cells"][f"inside|{fkey(f)}"] = ba_cell(gi)
            out["e30"][mkey(m)]["cells"][f"between|{fkey(f)}"] = ba_cell(gb)
            out["e30"][mkey(m)]["contrast"][fkey(f)] = pdiff(gi, gb)
    # E31
    E = out["e31"]
    d = records("e31a_*.jsonl")
    if d is not None:
        for (m, c, f), g in d.groupby(["model", "cls", "ser"]):
            key = "r1" if "r1" in str(m).lower() or "distill" in str(m).lower() else None
            if key:
                E["r1"][f"{c}|{fkey(f)}"] = ba_cell(g)
            else:
                E["cot"][mkey(m)][f"{c}|{fkey(f)}"] = ba_cell(g)
    d = records("e31b_*.jsonl")
    if d is not None:
        for (c, f), g in d.groupby(["cls", "ser"]):
            E["gpt4o"][f"{c}|{fkey(f)}"] = ba_cell(g)
    d = records("e31c_*.jsonl")
    if d is not None:
        for (m, v, c, f), g in d.groupby(["model", "variant", "cls", "ser"]):
            E["ids"][mkey(m)][f"{v}|{c}|{fkey(f)}"] = ba_cell(g)
    d = records("e31d_*.jsonl")
    if d is not None:
        x = d["pair"].str.extract(r"L(\d+)-p\d+-(\d)")
        d["n"], d["third"] = x[0], x[1]
        for (m, n, f), g in d.groupby(["model", "n", "ser"]):
            th = {t: ba_cell(gt) for t, gt in g.groupby("third")}
            for t, c in th.items():
                E["large"][mkey(m)][f"{n}|{fkey(f)}|{t}"] = c
            vals = [c["ba"] for c in th.values()]
            E["large"][mkey(m)][f"{n}|{fkey(f)}"] = {"ba": round(st.mean(vals), 4), "spread": round(max(vals) - min(vals), 4)}
    d = records("e31e_*.jsonl")
    if d is not None:
        for (m, v, pid, f), g in d.groupby(["model", "variant", "pid", "ser"]):
            E["prompting"][mkey(m)][f"{v}|{pid}|{fkey(f)}"] = ba_cell(g)
    d = records("e31f_*.jsonl")
    if d is not None:
        for (v, c, f), g in d.groupby(["variant", "cls", "ser"]):
            E["scale"][v][f"{c}|{fkey(f)}"] = ba_cell(g)
    d = records("e31g_*.jsonl")
    if d is not None:
        d = d[d.pid == "D1a"]
        d["kb"] = d["k_min"].map(lambda k: None if k is None or (isinstance(k, float) and math.isnan(k)) else
                                 "1" if k == 1 else "2" if k == 2 else "3-4" if k <= 4 else ">=5")
        for (m, f), g in d.groupby(["model", "ser"]):
            neg = g[~g.label.astype(bool)]
            name = "Qwen2.5-7B (E11c pool)" if mkey(m) == "Qwen2.5-7B" else mkey(m)
            for kb in ("1", "2", "3-4", ">=5"):
                pos = g[g.label.astype(bool) & (g.kb == kb)]
                gs = pd.concat([pos, neg[neg.pair.isin(pos.pair)]])
                c = ba_cell(gs)
                if c:
                    E["alibi"][name][f"{fkey(f)}|{kb}"] = c["ba"]
            if f"{fkey(f)}|1" in E["alibi"][name] and f"{fkey(f)}|>=5" in E["alibi"][name]:
                E["alibi"][name][f"{fkey(f)}|drop"] = round(E["alibi"][name][f"{fkey(f)}|1"] - E["alibi"][name][f"{fkey(f)}|>=5"], 4)
    d = records("e31h_*.jsonl")
    if d is not None:
        for (m, t, f), g in d.groupby(["model", "pid", "ser"]):
            E["algo"][mkey(m)][f"{t}|{fkey(f)}"] = ba_cell(g)
    for src in ("molecules", "citation"):
        d = records(f"e31i_{src}_*.jsonl")
        if d is not None:
            for (m, pid, f), g in d.groupby(["model", "pid", "ser"]):
                E["real"][mkey(m)][f"{src}|{pid}|{fkey(f)}"] = ba_cell(g)
    d = records("e31j_*.jsonl")
    if d is not None:
        for m, gm in d.groupby("model"):
            for f, g in gm.groupby("ser"):
                E["latek"][mkey(m)][fkey(f)] = ba_cell(g)
            if {"NC", "EL"} <= set(gm.ser):
                E["latek"][mkey(m)]["NC-EL"] = pdiff(gm[gm.ser == "NC"], gm[gm.ser == "EL"])


def tree():
    from collections import defaultdict
    return defaultdict(tree)


def plain(d):
    return {k: plain(v) for k, v in d.items()} if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e3", default=str(HERE.parent / "e3" / "results"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    RES[0] = Path(a.results)
    out = tree()
    e3_blocks(Path(a.e3), out)
    llm_blocks(out)
    res = plain(out)
    json.dump(res, open(a.out, "w"), indent=1)
    print(a.out, {k: len(v) for k, v in res.items()})


if __name__ == "__main__":
    main()
