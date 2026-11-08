"""Name matching measured inside the trained E3 models (the realizability theorem's conditions are not met
by E3: pre-LayerNorm, 32-dimensional heads; Section mr:sec). For every attention head of a saved model
(train_e3.py --save-dir) and every name token i of a test prompt that has an earlier occurrence of the same name,
the head's attention weights from i are computed exactly (scores, causal or local mask, ALiBi bias, RoPE as in the
model; the hard head of the hard-since arm by rightmost selection). A head matches at i if its largest weight is on an
earlier occurrence of the same name (the operation M_{0,0}); its mass is the weight on those occurrences. Per model the
best matching head (highest accuracy over all positions) is reported by the number of distinct names in the prompt
(bins 1-16, 17-32, 33-48, 49-64), with the number of heads whose accuracy exceeds 0.9 ("M-like heads").
Usage: python match_heads.py --ckpt ckpt/dia_AL_local_L6_s0.pt --data data --n 500 --out results/match_heads.jsonl
       python match_heads.py summary results/match_heads.jsonl --out results/summary_match.json
"""
import argparse
import time
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from train_e3 import Model, batchify, load, rope

BINS = ((1, 16), (17, 32), (33, 48), (49, 64))


def bin_of(k):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= k <= hi:
            return f"{lo}-{hi}"
    return None


def attn_probs(blk, model, h, bias, glob):
    """Attention weights (B, H, T, T) of one block on the residual h, as the block computes them."""
    B, T, D = h.shape
    q, k, v = blk.qkv(blk.ln1(h)).view(B, T, 3, blk.h, D // blk.h).permute(2, 0, 3, 1, 4)
    if blk.pos == "rope":
        q, k = rope(q), rope(k)
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1]) + bias.float()
    p = torch.softmax(s, -1)
    if blk.pos == "local_hardsince":
        hh = 3
        s3 = (q[:, hh].float() @ k[:, hh].float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        s3 = s3.masked_fill(~glob, float("-inf"))
        is_max = s3 == s3.max(-1, keepdim=True).values
        idx = (T - 1) - torch.flip(is_max, dims=[-1]).int().argmax(-1)
        p[:, hh] = torch.nn.functional.one_hot(idx, T).float()
    return p


def run_one(a):
    t_start = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu")
    A = ck["args"]
    vocab = ck["vocab"]
    model = Model(len(vocab) + 2, A["d"], A["layers"], A["heads"], A["pos"]).eval()
    model.load_state_dict(ck["state"])
    model.to(dev)
    # prompts per name bin: the in-distribution test set (8 to 16 vertices) gives bin 1-16, the OOD set the others;
    # the first n/4 prompts of each bin in file order (2 per bin in the offline test)
    per = 2 if os.environ.get("E2E_TINY") else a.n // len(BINS)
    pool = load(str(Path(a.data) / f"{A['task']}_{A['ser']}_id.jsonl")) + load(str(Path(a.data) / f"{A['task']}_{A['ser']}_ood.jsonl"))
    take = {f"{lo}-{hi}": [] for lo, hi in BINS}
    for e in pool:
        b_ = bin_of(len({t for t in e["tokens"] if t.startswith("@")}))
        if b_ is not None and len(take[b_]) < per:
            take[b_].append(e)
    exs = [e for b_ in take for e in take[b_]]
    gen = torch.Generator().manual_seed(777)
    L, H = A["layers"], A["heads"]
    hit = np.zeros((L, H, len(BINS)))
    mass = np.zeros((L, H, len(BINS)))
    cnt = np.zeros(len(BINS))
    with torch.no_grad():
        for e in exs:
            x, nv, isn, ln, _ = batchify([e], vocab, A["d"], dev, gen)
            toks = e["tokens"]
            names = [t for t in toks if t.startswith("@")]
            b = bin_of(len(set(names)))
            if b is None:
                continue
            bi = [f"{lo}-{hi}" for lo, hi in BINS].index(b)
            prev = {}
            targets = []
            for i, t in enumerate(toks):
                if t.startswith("@"):
                    if t in prev:
                        targets.append((i, list(prev[t])))
                    prev.setdefault(t, []).append(i)
            if not targets:
                continue
            hres = model.emb(x)
            hres = torch.where(isn[..., None], nv.to(hres.dtype), hres)
            bias, glob = model.bias(x.shape[1], x.device)
            for l, blk in enumerate(model.blocks):
                p = attn_probs(blk, model, hres, bias, glob)[0].cpu()    # (H, T, T)
                for i, js in targets:
                    row = p[:, i, :]
                    hit[l, :, bi] += np.isin(row.argmax(-1).numpy(), js)
                    mass[l, :, bi] += row[:, js].sum(-1).numpy()
                hres = blk(hres, bias, glob)
            cnt[bi] += len(targets)
    acc = hit / np.maximum(cnt, 1)
    tot = hit.sum(-1) / max(cnt.sum(), 1)
    l_, h_ = np.unravel_index(np.argmax(tot), tot.shape)
    rec = {"task": A["task"], "ser": A["ser"], "pos": A["pos"], "layers": A["layers"], "seed": A["seed"],
           "best_head": [int(l_), int(h_)], "best_acc": float(tot[l_, h_]),
           "acc_by_bin": {f"{lo}-{hi}": (float(acc[l_, h_, i]) if cnt[i] else None) for i, (lo, hi) in enumerate(BINS)},
           "mass_by_bin": {f"{lo}-{hi}": (float(mass[l_, h_, i] / cnt[i]) if cnt[i] else None) for i, (lo, hi) in enumerate(BINS)},
           "positions_by_bin": {f"{lo}-{hi}": int(cnt[i]) for i, (lo, hi) in enumerate(BINS)},
           "n_mlike": int((tot > 0.9).sum()), "n_prompts": len(exs)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rec["sec"] = round(time.time() - t_start, 1)           # wall clock (T_compute: e3_match_hours)
    rec["n_gpus"] = 1 if dev == "cuda" else 0
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("pos", "ser", "seed", "best_head", "best_acc", "n_mlike")}))


ARMS = {"nope": "NoPE", "local": "1-local", "alibi_only": "ALiBi-only", "local_alibi": "1-local+ALiBi",
        "local_hardsince": "1-local+hard-since", "rope": "RoPE"}


def summary(paths, out):
    recs = [json.loads(l) for p in paths for l in open(p)]
    res = {}
    for pos, arm in ARMS.items():
        for f in ("EL", "AL", "NC", "ECL"):
            rs = [r for r in recs if r["pos"] == pos and r["ser"] == f]
            if not rs:
                continue
            cell = {}
            for lo, hi in BINS:
                k = f"{lo}-{hi}"
                v = [1 - r["acc_by_bin"][k] for r in rs if r["acc_by_bin"].get(k) is not None]
                if v:
                    cell[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "seeds": v}
            cell["n_mlike"] = float(np.mean([r["n_mlike"] for r in rs]))
            res[f"{arm}|{f}"] = cell
    json.dump({"error": res}, open(out, "w"), indent=1)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "summary"])
    ap.add_argument("results", nargs="*")
    ap.add_argument("--ckpt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "summary":
        summary(a.results, a.out)
    else:
        run_one(a)


if __name__ == "__main__":
    main()
