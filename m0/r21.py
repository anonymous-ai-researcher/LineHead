"""CONFIG.md §6.15: the seventh preregistered set, three tests of the since-head evidence, fixed before
any of their runs (rules V-C68 to V-C74, VALIDATION_RULES.md, group C).

E34  Binding control (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B). GraphQA edge existence ("Is there an edge between
     nodes u and w?", the E13 pairs of data/e13_graphqa.jsonl, 200 per format, template T1 layout of e13_graphqa.py,
     scored by log p(Yes) against log p(No) as E1). Under AL and NC the answer needs the line that lists the second
     name, not a head color.
     edge     the top-128 since heads of E11a's ranking (results/e11_head_scores<_tag>.json), mean-ablated with the
              role means of E11a at every position, and 5 random sets of 128 heads with the per-layer counts of the
              top 128 (e11_heads.random_sets); formats EL, AL, NC, ECL. Variants since128, rand128s<seed>. The
              unablated cells are those of E13 (rerun with --with-base, they must equal them).
     edge-ko  the E19 knockout (r19.score_knockout) on the same prompts, AL and NC: in every head the attention from
              each list-entry name to the head segment of its own line (ko-own) or of the previous line (ko-other).
E35  Norm-matched null (same three models; the E11a/E17 prompts: the first 300 E1 D1a and E7 Q0 pairs, T1, four
     formats). 100 random sets of 128 heads, each head drawn from the heads outside the top 128 in the same layer
     and the same decile of the head-output norm (the mean L2 norm of the head's slice of the o_proj input over all
     positions of the 64 calibration prompts of E11a, deciles over all heads of the model); a stratum without a free
     head falls back to the same layer, then to the nearest layer with the same decile. Variants null128s<seed>.
     The top-128 cells are those of E17.
E36  Replication of h = 128.
     tpl      templates T2 and T3 (the E28 prompts), three models: top-128 since heads and 5 random sets of 128;
              variants top128, rand128s<seed>; the unablated cells are those of E28 (E28a).
     d1c      the paraphrase D1c (the first 300 E1 D1c pairs, T1), three models: unablated and top 128.
     q72      Qwen2.5-72B-Instruct (D1a, T1, the first 300 pairs), heads ranked by its own attention exactly as E11a
              (e11_heads.py score with this model); h = 128 (2.5% of its 5,120 heads) and h = 640 (12.5%, within
              the 12 to 19% that 128 heads make up in the smaller models), top and 5 random sets each; four GPUs.
summary  results/blocks_r21.json with the blocks e34, e35, e36 in the structure of analysis/schema.json (cells with
         95% pair-bootstrap intervals; flips with the one-sided 95% lower bound of EL minus the mean of AL and NC);
         analysis/import_measured.py --r21 imports it and fills the cells that restate earlier blocks.
Usage (one model per call; TAG = qwen7b, llama8b, gemma9b, qwen72b):
  python r21.py edge     --model M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e34_<tag>.jsonl
  python r21.py edge-ko  --model M $NAMES_ARG --out results/e34ko_<tag>.jsonl
  python r21.py null     --model M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e35_<tag>.jsonl
  python r21.py tpl      --model M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e36tpl_<tag>.jsonl
  python r21.py d1c      --model M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e36d1c_<tag>.jsonl
  python e11_heads.py score --model Qwen/Qwen2.5-72B-Instruct $NAMES_ARG --out results/e11_head_scores_qwen72b.json
  python r21.py q72      --model Qwen/Qwen2.5-72B-Instruct $NAMES_ARG --scores results/e11_head_scores_qwen72b.json \
      --out results/e36q72.jsonl
  python r21.py summary results/e34_*.jsonl results/e34ko_*.jsonl results/e35_*.jsonl results/e36*.jsonl \
      --out results/blocks_r21.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from e11_heads import (FMTS, SEEDS, Ablator, chat, load_model, random_sets, score_prompts, test_records, tiny)
from instances import load, prompts_for
from names import POOL, single_token_names
from r19 import score_knockout

MODELS = {"Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B", "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
          "google/gemma-2-9b-it": "Gemma-2-9B", "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B"}
NULL_SETS = 100


def setup(a):
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    ab = Ablator(m, torch)
    cal = [r for r in load(a.instances) if r["cls"] == "depth1"][-(8 if tiny() else 64):]
    cfmt = [FMTS[i % 4] for i in range(len(cal))]
    cal_prompts = [next(prompts_for(r, pool, serializations=(f,), templates=("T1",)))["prompt"] for r, f in zip(cal, cfmt)]
    ab.calibrate(tok, cal_prompts, fmts=cfmt, names=pool)
    return tok, m, torch, pool, ab, cal_prompts


def top_heads(a):
    return [(l, j) for l, j, _ in json.load(open(a.scores))["scores"]]


def run(f, tok, m, torch, ab, conds, prompts, model, extra=None):
    for cond, heads in conds:
        ab.set(heads, "role")
        for p in prompts:
            ab.prepare(tok, p["prompt"], p["ser"])
            ly, ln = score_prompts(tok, m, torch, [p["prompt"]])[0]
            rec = {k: p[k] for k in ("id", "pid", "pair", "label", "ser", "tpl") if k in p}
            rec.update(variant=cond, lp_yes=ly, lp_no=ln, pred=ly > ln, model=model, **(extra or {}))
            f.write(json.dumps(rec) + "\n")
        print(cond, len(prompts), flush=True)
    ab.set([])


def edge_prompts(a, pool, sers):
    from e13_graphqa import prompt_of
    recs = [r for r in load(a.e13_instances) if r["task"] == "edge"][: (16 if tiny() else 400)]
    out = []
    for s in sers:
        for r in recs:
            p, _ = prompt_of(r, s, pool)
            out.append(dict(id=r["id"], pid="edge", pair=r["pair"], label=r["label"], ser=s, tpl="T1", prompt=p))
    return out


def cmd_edge(a):
    t0 = time.time()
    tok, m, torch, pool, ab, _ = setup(a)
    top = top_heads(a)
    conds = ([("none", [])] if a.with_base else []) + [("since128", top[:128])] + \
            [(f"rand128s{s}", random_sets(top, ab.L, ab.H, 128, s)) for s in range(SEEDS)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        run(f, tok, m, torch, ab, conds, edge_prompts(a, pool, FMTS), a.model)
    timing(a, t0, torch)


def cmd_edge_ko(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for kind in ("own", "other"):
            for p in edge_prompts(a, pool, ("AL", "NC")):
                y, n, k = score_knockout(tok, m, torch, p["prompt"], p["ser"], kind)
                rec = {k_: p[k_] for k_ in ("id", "pid", "pair", "label", "ser", "tpl")}
                rec.update(variant=f"ko-{kind}", lp_yes=y, lp_no=n, pred=y > n, n_removed=k, model=a.model)
                f.write(json.dumps(rec) + "\n")
    timing(a, t0, torch)


def head_norms(tok, m, torch, prompts):
    """Mean L2 norm of every head's slice of the o_proj input over all positions of the calibration prompts."""
    L, H = m.config.num_hidden_layers, m.config.num_attention_heads
    dh = m.config.hidden_size // H
    tot, n = np.zeros((L, H)), [0]

    def mk(l):
        def f(mod, inp):
            x = inp[0][0].detach().float().cpu().view(-1, H, dh)
            tot[l] += x.norm(dim=-1).sum(0).numpy()
            if l == 0:
                n[0] += x.shape[0]
        return f
    hs = [m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(mk(l)) for l in range(L)]
    for p in prompts:
        enc = tok(chat(tok, p), return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            m(**{k: v.to(m.device) for k, v in enc.items()})
    for h in hs:
        h.remove()
    return tot / max(n[0], 1)


def norm_matched_sets(top, norms, h, n_sets):
    """n_sets random sets of h heads, each head of the top h replaced by a free head of its layer and norm decile."""
    L, H = norms.shape
    dec = np.digitize(norms, np.quantile(norms, np.linspace(0.1, 0.9, 9)))
    excl = set(map(tuple, top[:h]))
    sets = []
    for s in range(n_sets):
        rng = random.Random(f"e35|{h}|{s}")
        used, out = set(), []
        for l, j in top[:h]:
            d = dec[l, j]
            cands = [[(l, k) for k in range(H) if dec[l, k] == d],                       # same layer, same decile
                     [(l, k) for k in range(H)]]                                         # same layer
            for lay in sorted(range(L), key=lambda x: abs(x - l)):                         # nearest layer, same decile
                cands.append([(lay, k) for k in range(H) if dec[lay, k] == d])
            for c in cands:
                c = [x for x in c if x not in excl and x not in used]
                if c:
                    x = rng.choice(c)
                    used.add(x)
                    out.append(x)
                    break
        sets.append(out)
    return sets


def cmd_null(a):
    t0 = time.time()
    tok, m, torch, pool, ab, cal_prompts = setup(a)
    top = top_heads(a)
    norms = head_norms(tok, m, torch, cal_prompts)
    sets = norm_matched_sets(top, norms, 128, 4 if tiny() else NULL_SETS)
    recs = test_records(a, pool)
    prompts = [p for r in recs for p in prompts_for(r, pool, serializations=FMTS, templates=("T1",))]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"norms": norms.round(5).tolist(), "sets": [[list(x) for x in s] for s in sets]},
              open(Path(a.out).with_name(Path(a.out).stem + "_sets.json"), "w"))
    with open(a.out, "w") as f:
        run(f, tok, m, torch, ab, [(f"null128s{s}", x) for s, x in enumerate(sets)], prompts, a.model)
    timing(a, t0, torch)


def cmd_tpl(a):
    t0 = time.time()
    tok, m, torch, pool, ab, _ = setup(a)
    top = top_heads(a)
    recs = test_records(a, pool)
    prompts = [p for r in recs for p in prompts_for(r, pool, serializations=FMTS, templates=("T2", "T3"))]
    conds = [("top128", top[:128])] + [(f"rand128s{s}", random_sets(top, ab.L, ab.H, 128, s)) for s in range(SEEDS)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        run(f, tok, m, torch, ab, conds, prompts, a.model)
    timing(a, t0, torch)


def cmd_d1c(a):
    t0 = time.time()
    tok, m, torch, pool, ab, _ = setup(a)
    top = top_heads(a)
    d1c = [r for r in load(a.instances) if r["pid"] == "D1c"]
    keep = set(sorted({r["pair"] for r in d1c})[: (8 if tiny() else a.pairs)])
    prompts = [p for r in d1c if r["pair"] in keep for p in prompts_for(r, pool, serializations=FMTS, templates=("T1",))]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        run(f, tok, m, torch, ab, [("none", []), ("top128", top[:128])], prompts, a.model, extra={"pid_src": "D1c"})
    timing(a, t0, torch)


def cmd_q72(a):
    t0 = time.time()
    tok, m, torch, pool, ab, _ = setup(a)
    top = top_heads(a)
    recs = [r for r in test_records(a, pool) if r["pid"] == "D1a"]
    prompts = [p for r in recs for p in prompts_for(r, pool, serializations=FMTS, templates=("T1",))]
    hs = (128, 640) if not tiny() else (2, 4)
    conds = [("none", [])] + [(f"top{h}", top[:h]) for h in hs] + \
            [(f"rand{h}s{s}", random_sets(top, ab.L, ab.H, h, s)) for h in hs for s in range(SEEDS)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        run(f, tok, m, torch, ab, conds, prompts, a.model)
    timing(a, t0, torch)


def timing(a, t0, torch):
    json.dump({"sec": round(time.time() - t0, 1), "n_gpus": max(1, torch.cuda.device_count())},
              open(Path(a.out).with_name(Path(a.out).stem + "_timing.json"), "w"))


# ------------------------------------------------------------------ summary
def per_pair(rows):
    by = {}
    for r in rows:
        by.setdefault(r["pair"], []).append((bool(r["label"]), bool(r["pred"]) == bool(r["label"])))
    return {k: 0.5 * (np.mean([c for l, c in v if l]) + np.mean([c for l, c in v if not l]))
            for k, v in by.items() if any(l for l, _ in v) and any(not l for l, _ in v)}


def cell(rows, seed=0, reps=2000):
    pp = np.array(list(per_pair(rows).values()))
    bs = pp[np.random.default_rng(seed).integers(0, len(pp), (reps, len(pp)))].mean(1)
    return dict(ba=round(float(pp.mean()), 4), lo=round(float(np.quantile(bs, 0.025)), 4), hi=round(float(np.quantile(bs, 0.975)), 4))


def flip(el, al, nc, seed=0, reps=2000):
    a, b, c = per_pair(el), per_pair(al), per_pair(nc)
    ks = sorted(set(a) & set(b) & set(c))
    d = np.array([a[k] - 0.5 * (b[k] + c[k]) for k in ks])
    bs = d[np.random.default_rng(seed).integers(0, len(d), (reps, len(d)))].mean(1)
    return dict(diff=round(float(d.mean()), 4), lo05=round(float(np.quantile(bs, 0.05)), 4))


def summ(xs):
    return dict(mean=round(float(np.mean(xs)), 4), min=round(float(min(xs)), 4), max=round(float(max(xs)), 4))


def cmd_summary(a):
    rows = [json.loads(l) for fn in a.results for l in open(fn)]
    G = {}
    for r in rows:
        src = "edge" if r["pid"] == "edge" else r.get("pid_src", r["pid"])
        G.setdefault((MODELS.get(r["model"], r["model"]), r["variant"], r["ser"], src, r["tpl"]), []).append(r)
    ba = lambda *k: cell(G[k])["ba"] if k in G else None
    out = {"e34": {}, "e35": {}, "e36": {"tpl": {}, "d1c": {}, "q72": {}}}
    for m in ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B"):
        E = {"cells": {}, "rand128": {}}
        for f in FMTS:
            for v in ("none", "since128") + (("ko-own", "ko-other") if f in ("AL", "NC") else ()):
                if (m, v, f, "edge", "T1") in G:
                    E["cells"][f"{v}|{f}"] = cell(G[(m, v, f, "edge", "T1")])
            rs = [ba(m, f"rand128s{s}", f, "edge", "T1") for s in range(SEEDS)]
            if all(x is not None for x in rs):
                E["rand128"][f] = summ(rs)
        out["e34"][m] = E
        N = {}
        for f in FMTS:
            for pid in ("D1a", "Q0"):
                ns = [ba(m, f"null128s{s}", f, pid, "T1") for s in range(NULL_SETS)]
                ns = [x for x in ns if x is not None]
                if ns:          # losses against the unablated E11a cell are formed at import (top from E17)
                    N[f"{f}|{pid}"] = dict(null_ba=[round(x, 4) for x in ns], n_sets=len(ns))
        out["e35"][m] = N
        T = {}
        for tpl in ("T2", "T3"):
            for f in FMTS:
                for pid in ("D1a", "Q0"):
                    if (m, "top128", f, pid, tpl) in G:
                        T[f"{tpl}|top128|{f}|{pid}"] = cell(G[(m, "top128", f, pid, tpl)])
                        T[f"{tpl}|rand128|{f}|{pid}"] = summ([ba(m, f"rand128s{s}", f, pid, tpl) for s in range(SEEDS)])
            if (m, "top128", "EL", "D1a", tpl) in G:
                T[f"{tpl}|flip"] = flip(*(G[(m, "top128", f, "D1a", tpl)] for f in ("EL", "AL", "NC")))
        out["e36"]["tpl"][m] = T
        D = {}
        for f in FMTS:
            for v in ("none", "top128"):
                if (m, v, f, "D1c", "T1") in G:
                    D[f"{v}|{f}"] = cell(G[(m, v, f, "D1c", "T1")])
        if (m, "top128", "EL", "D1c", "T1") in G:
            D["flip"] = flip(*(G[(m, "top128", f, "D1c", "T1")] for f in ("EL", "AL", "NC")))
        out["e36"]["d1c"][m] = D
    m, Q = "Qwen2.5-72B", {}
    for f in FMTS:
        for v in ("none", "top128", "top640"):
            if (m, v, f, "D1a", "T1") in G:
                Q[f"{v}|{f}"] = cell(G[(m, v, f, "D1a", "T1")])
        for h in ("128", "640"):
            rs = [ba(m, f"rand{h}s{s}", f, "D1a", "T1") for s in range(SEEDS)]
            if all(x is not None for x in rs):
                Q[f"rand{h}|{f}"] = summ(rs)
    for h in ("128", "640"):
        if (m, f"top{h}", "EL", "D1a", "T1") in G:
            Q[f"flip{h}"] = flip(*(G[(m, f"top{h}", f, "D1a", "T1")] for f in ("EL", "AL", "NC")))
    out["e36"]["q72"] = Q
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["edge", "edge-ko", "null", "tpl", "d1c", "q72", "summary"])
    ap.add_argument("results", nargs="*")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--q0-instances", default="data/e7_instances.jsonl")
    ap.add_argument("--e13-instances", default="data/e13_graphqa.jsonl")
    ap.add_argument("--names-from", default=None)
    ap.add_argument("--scores", default="results/e11_head_scores.json")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--with-base", action="store_true", help="E34: also rerun the unablated cells (identity check)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"edge": cmd_edge, "edge-ko": cmd_edge_ko, "null": cmd_null, "tpl": cmd_tpl, "d1c": cmd_d1c, "q72": cmd_q72,
     "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
