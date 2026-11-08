"""E13: GraphQA node tasks (Fatemi et al., 2024) under our serializations (CONFIG.md §6.4).

Tasks (data/e13_graphqa.jsonl, build_r12.py e13): node degree ("What is the degree of node v?", generated integer),
edge existence ("Is there an edge between nodes u and w?", Yes/No scored as in E1, matched pairs), node neighbors
("List all neighbors of node v.", generated list, exact set match). Formats EL, AL, NC, ECL, Incident and
Friendship, template T1 layout (instruction, graph, question). Yes/No questions are scored by the log-probability
comparison of run_llm.py; generated answers use greedy decoding of at most 48 tokens, and the first integer (degree)
or the set of pool names (neighbors) in the answer is compared with the truth. The two kinds are reported apart.
Usage:
  python e13_graphqa.py run --backend hf --model Qwen/Qwen2.5-7B-Instruct $NAMES_ARG --instances data/e13_graphqa.jsonl \
      --out results/e13_qwen7b.jsonl
  python e13_graphqa.py summary results/e13_*.jsonl --out results/summary_e13.json
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from instances import SER_OFFSET, assign_names, load, to_graph
from names import POOL, single_token_names
from serialize import serialize

FMTS = ("EL", "AL", "NC", "ECL", "Incident", "Friendship")
HEAD = "Here is an undirected graph. Every node is colored red or blue.\n\n"
Q = {"degree": "Question: What is the degree of node {v} (its number of neighbors)?\nAnswer with a number.",
     "edge": "Question: Is there an edge between nodes {v} and {w}?\nAnswer with Yes or No.",
     "neighbors": "Question: List all neighbors of node {v}.\nAnswer with the node names, separated by commas."}


def prompt_of(r, s, pool):
    G = to_graph(r)
    names = assign_names(r, pool)
    text = serialize(G, names, s, seed=r["graph_seed"] + SER_OFFSET[s])
    q = Q[r["task"]].format(v=names[r["query"]], w=names.get(r.get("other"), ""))
    return HEAD + text + "\n\n" + q, names


def parse_answer(task, text, names):
    if task == "degree":
        m = re.search(r"\d+", text)
        return int(m.group()) if m else None
    inv = {w: u for u, w in names.items()}
    return sorted({inv[w] for w in re.findall(r"[A-Za-z]+", text) if w in inv})


def cmd_run(a):
    from run_llm import HF, Mock
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    m = HF(a.model, batch=a.batch) if a.backend == "hf" else Mock()
    recs = load(a.instances)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in FMTS:
            items = [(r, *prompt_of(r, s, pool)) for r in recs]
            yn = [x for x in items if x[0]["task"] == "edge"]
            gen = [x for x in items if x[0]["task"] != "edge"]
            import time
            t0 = time.time()
            scs = m([p for _, p, _ in yn])
            sec = round((time.time() - t0) / max(1, len(yn)), 4)
            for (r, p, _), sc in zip(yn, scs):
                f.write(json.dumps({"id": r["id"], "pid": r["pid"], "cls": "graphqa", "pair": r["pair"], "label": r["label"],
                                    "ser": s, "tpl": "T1", "task": r["task"], "kind": "yesno", "pred": sc["pred"],
                                    "lp_yes": sc.get("lp_yes"), "lp_no": sc.get("lp_no"), "n_tokens": sc.get("n_tokens"),
                                    "model": a.model, "backend": a.backend, "sec": sec, "n_gpus": a.n_gpus}) + "\n")
            if gen:
                t0 = time.time()
                ans = (m.generate_answers([p for _, p, _ in gen], max_new=48) if a.backend == "hf"
                       else m.generate_answers([p for _, p, _ in gen]))
                sec = round((time.time() - t0) / len(gen), 4)
                for (r, p, names), g in zip(gen, ans):
                    text = g["gen_text"]
                    got = parse_answer(r["task"], text, names)
                    f.write(json.dumps({"id": r["id"], "pid": r["pid"], "cls": "graphqa", "pair": r["pair"], "label": True,
                                        "ser": s, "tpl": "T1", "task": r["task"], "kind": "generate", "gen_text": text,
                                        "parsed": got, "exact": got == r["answer"], "pred": got == r["answer"],
                                        "model": a.model, "backend": a.backend, "sec": sec, "n_gpus": a.n_gpus}) + "\n")
    print("wrote", out)


def cmd_summary(a):
    from stats import bal_acc, cell_rng, load as load_recs, pair_bootstrap
    d = load_recs(a.results)
    reps = 50 if os.environ.get("E2E_TINY") else 2000
    out = {"cells": []}
    for (m, s, t), x in d.groupby(["model", "ser", "task"]):
        if t == "edge":
            lo, hi = pair_bootstrap(x, reps, cell_rng(x))
            out["cells"].append(dict(model=m, ser=s, task=t, kind="yesno", metric="bal_acc", value=bal_acc(x),
                                     ci_low=float(lo), ci_high=float(hi), n=int(x.pair.nunique())))
        else:
            v = x["exact"].astype(float).values
            rng = np.random.default_rng(0)
            bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)]
            out["cells"].append(dict(model=m, ser=s, task=t, kind="generate", metric="exact", value=float(v.mean()),
                                     ci_low=float(np.percentile(bs, 2.5)), ci_high=float(np.percentile(bs, 97.5)), n=int(len(v))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("results", nargs="*")
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--names-from", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-gpus", type=int, default=1, help="GPUs the run occupies (72B: 4), for T_compute")
    ap.add_argument("--instances", default="data/e13_graphqa.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        json.dump(cmd_summary(a), open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
