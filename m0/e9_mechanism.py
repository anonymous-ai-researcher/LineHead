"""E9: mechanism evidence in Qwen2.5-7B-Instruct (28 layers; CONFIG.md §6).

Four analyses on the 400 graphs of the D1a pairs in data/m0_instances.jsonl (one prompt per pair, template T1,
names filtered exactly as in E1 with --names-from):
  probe     logistic-regression probes on the residual stream at list-entry name tokens (AL and NC) that
            decode the color of the entry's LINE HEAD; per layer (0..28) and per distance bin of the entry to
            its head color token (word-level distance d: 1-8, 9-24, >24; as in analysis/schema.json); 5-fold CV grouped by graph;
            classes reweighted, balanced accuracy (chance 0.5); every entry with d > 24 is kept (few lines are long).
  retrieve  head-name retrieval: for each entry token, cosine similarity between its residual at layer l and
            the residuals at every line-head name position of the same prompt; top-1 accuracy of retrieving its
            own line head, per layer (the "name transport" of the paper's appendix section on routes, D.7).
  patch     activation patching (NC and AL): corrupt the prompt by changing the head color of every red
            neighbor's line from red to blue (keeping entries), run clean and corrupted, and for each residual
            index i = 0..28 (0: embedding output, i: output of block i) patch the clean residual at the corrupted head-color tokens into the corrupted run; report the
            fraction of the clean-vs-corrupted Yes-No logit difference restored, per layer, as a ratio of sums over
            graphs: sum (patched - corrupted) / sum (clean - corrupted).
  el_inner  under EL: at the second endpoint b of every edge line, probe b's OWN color (from the color line),
            per layer: the inner match M_{0,1} red of Proposition 6(b).
Controls 
  control   Hewitt-Liang control task for the probe: the same probe (same tokens, layers, folds, regularization)
            trained on a label that is a fixed random function of the entry's NAME (crc32 of the name, mod 2), which
            a probe can only fit by memorizing name types; selectivity = probe accuracy - control accuracy, per layer.
  chance    retrieval chance: the top-1 accuracy of a uniformly random choice among the prompt's line heads
            (mean of 1 / #lines over the scored entries), reported beside the retrieval accuracy.
Outputs results/e9_{probe,retrieve,patch,el_inner}.json (schema in RUN_PROTOCOL.md).
Usage: python e9_mechanism.py --model Qwen/Qwen2.5-7B-Instruct --names-from $NAMES --out-dir results
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from instances import assign_names, load, prompts_for, to_graph
from names import POOL, single_token_names

DBINS = [(1, 8), (9, 24), (25, 10**6)]     # E1 degree <= 17, so AL d <= 35 and only 24 NC entries have d > 64
BIN_LABEL = {(1, 8): "1-8", (9, 24): "9-24", (25, 10**6): ">24"}
YES_WORDS, NO_WORDS = ["Yes", " Yes", "yes", " yes"], ["No", " No", "no", " no"]    # as in run_llm.py scoring


def answer_ids(tok, words):
    return sorted({t[0] for t in (tok.encode(w, add_special_tokens=False) for w in words) if t})


def dist_of(ser, k):
    """Word-level distance from the k-th entry to its line's head color (2k+1 under AL, 5k-1 under NC)."""
    return 2 * k + 1 if ser == "AL" else 5 * k - 1


def bin_of(d):
    return next(BIN_LABEL[b] for b in DBINS if b[0] <= d <= b[1])


def dedupe_pairs(recs):
    """The two query vertices of a pair share one graph and one serialization; the prompts differ only in the
    question, so residuals at list entries coincide. Keep one record per pair (the positive): n = #graphs."""
    seen, out = set(), []
    for r in sorted(recs, key=lambda r: (r["pair"], not r["label"])):
        if r["pair"] not in seen:
            seen.add(r["pair"]); out.append(r)
    return out


def word_spans(prompt):
    """Word-level tokens (as in the theory) with character spans."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9]+|--|[^\sA-Za-z0-9]", prompt)]


def list_structure(prompt, ser):
    """For AL/NC prompts: list of lines, each (head_name_span, head_color_span or None, [entry spans])."""
    words = word_spans(prompt)
    lines, cur = [], None
    text_lines = prompt.split("\n")
    pos = 0
    for line in text_lines:
        start = pos
        pos += len(line) + 1
        lw = [w for w in words if start <= w[1] < pos]
        if ser == "AL" and len(lw) >= 5 and lw[1][0] == "(" and lw[3][0] == ")" and lw[4][0] == ":":
            entries = [w for w in lw[5:] if w[0] not in (",", "none")]
            lines.append((lw[0], lw[2], entries))
        elif ser == "NC" and len(lw) >= 6 and lw[1][0] == "is" and lw[4][0] == "Neighbors":
            entries = [lw[i] for i in range(6, len(lw)) if i + 1 < len(lw) and lw[i + 1][0] == "("]
            lines.append((lw[0], lw[2], entries))
    return lines


def char_to_token(enc_offsets, a, b):
    idx = [i for i, (s, e) in enumerate(enc_offsets) if e > a and s < b]
    return idx[-1] if idx else None


def collect(model, tok, prompts, ser, torch, per_graph=8, seed=0):
    """Residual streams (all layers, float16) at up to `per_graph` randomly chosen entry tokens per prompt, with
    labels: head color, distance, line id; every entry more than 24 tokens from its head color is added (147 under AL,
    8,277 under NC on the E1 D1a graphs). Memory: about 11,500 entries x 29 x 3,584 x 2 bytes = 2.4 GB (NC)."""
    rs = np.random.default_rng(seed)
    X = {l: [] for l in range(model.config.num_hidden_layers + 1)}
    meta = []
    for gi, p in enumerate(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        off = text.index(p)
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            hs = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
        lines = list_structure(p, ser)
        cand = [(li, k, e, c) for li, (h, c, entries) in enumerate(lines) for k, e in enumerate(entries, start=1)]
        pick = list(rs.permutation(len(cand))[:per_graph])
        pick += [j for j, (_, k, _, _) in enumerate(cand) if dist_of(ser, k) > 24 and j not in pick]   # all far entries
        for j in pick:
            li, k, e, c = cand[j]
            t = char_to_token(offsets, off + e[1], off + e[2])
            if t is None:
                continue
            d = dist_of(ser, k)
            if True:
                for l in X:
                    X[l].append(hs[l][0, t].to(torch.float16).cpu().numpy())
                import zlib
                meta.append({"graph": gi, "line": li, "k": k, "d": d, "head_red": c[0] == "red", "tok": t,
                             "ctrl": zlib.crc32(e[0].encode()) % 2 == 1})
    return {l: np.stack(v).astype(np.float32) for l, v in X.items()}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--graphs", type=int, default=None, help="cap on graphs (default: all D1a pairs)")
    ap.add_argument("--names-from", default=None, help="same single-token name filter as run_llm.py (prompts = E1)")
    ap.add_argument("--analyses", default="probe,retrieve,patch,el_inner")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    import os
    import time
    t_start = time.time()
    tiny = bool(os.environ.get("E2E_TINY"))       # offline end-to-end test only: 40 graphs, fp32 on CPU
    if tiny:
        a.graphs = 40
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32 if tiny else torch.bfloat16,
                                                 device_map=None if tiny else "auto",
                                                 attn_implementation="eager").eval()
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    recs = dedupe_pairs([r for r in load(a.instances) if r["pid"] == "D1a"])[: a.graphs]      # one per graph
    print(f"E9: {len(recs)} graphs (D1a pairs of {a.instances}, one prompt per pair)")
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    todo = a.analyses.split(",")
    for ser in ("AL", "NC"):
        prompts = [next(prompts_for(r, pool, serializations=(ser,), templates=("T1",)))["prompt"] for r in recs]
        if "probe" in todo or "retrieve" in todo:
            X, meta = collect(model, tok, prompts, ser, torch)
            y = np.array([m["head_red"] for m in meta]); g = np.array([m["graph"] for m in meta])
            dist = np.array([m["d"] for m in meta])
            if "probe" in todo:
                res = {}
                for l, Xl in X.items():
                    pred = np.zeros_like(y)
                    for tr, te in GroupKFold(5).split(Xl, y, g):
                        pred[te] = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(Xl[tr], y[tr]).predict(Xl[te])
                    def bacc(m_):                               # balanced accuracy (head colors are red in about 20% of lines)
                        yt, yp = y[m_], pred[m_]
                        if not (yt.any() and (~yt).any()):
                            return None
                        return float(0.5 * (np.mean(yp[yt]) + np.mean(~yp[~yt])))
                    res[l] = {BIN_LABEL[(lo, hi)]: bacc((dist >= lo) & (dist <= hi)) for lo, hi in DBINS}   # null: empty bin
                    res[l]["all"] = bacc(np.ones_like(y, dtype=bool))
                    res[l]["n"] = {BIN_LABEL[(lo, hi)]: int(((dist >= lo) & (dist <= hi)).sum()) for lo, hi in DBINS}
                    yc = np.array([m["ctrl"] for m in meta])            # control task (Hewitt and Liang, 2019)
                    pc = np.zeros_like(yc)
                    for tr, te in GroupKFold(5).split(Xl, yc, g):
                        pc[te] = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(Xl[tr], yc[tr]).predict(Xl[te])
                    res[l]["control"] = float(0.5 * (np.mean(pc[yc]) + np.mean(~pc[~yc])))
                    res[l]["selectivity"] = res[l]["all"] - res[l]["control"]
                json.dump(res, open(out_dir / f"e9_probe_{ser}.json", "w"), indent=1)
        if "retrieve" in todo:
            res = retrieve_pass(model, tok, prompts, ser, torch)
            json.dump(res, open(out_dir / f"e9_retrieve_{ser}.json", "w"), indent=1)
        if "patch" in todo:
            res = patch_pass(model, tok, recs, ser, torch, pool)
            json.dump(res, open(out_dir / f"e9_patch_{ser}.json", "w"), indent=1)
    if "el_inner" in todo:
        res = el_inner_pass(model, tok, recs, torch, pool)
        json.dump(res, open(out_dir / "e9_el_inner.json", "w"), indent=1)
    json.dump({"sec": round(time.time() - t_start, 1), "graphs": len(recs), "n_gpus": torch.cuda.device_count()},
              open(out_dir / "e9_timing.json", "w"))


def _forward(model, tok, p, torch):
    text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    off = text.index(p)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        o = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True)
    return o, offsets, off, enc


def retrieve_pass(model, tok, prompts, ser, torch):
    """Top-1 retrieval of an entry's own line head among all line-head name tokens of the prompt, per layer and
    per distance bin of the entry to its head color (1-8, 9-24, >24). The reported layer is the one with
    the best top-1 accuracy in the nearest bin; by_bin gives every bin at that layer."""
    L = model.config.num_hidden_layers + 1
    hits = {b: np.zeros(L) for b in BIN_LABEL.values()}; cnt = {b: 0 for b in BIN_LABEL.values()}
    chance = []                           # 1 / #line heads per scored entry: top-1 accuracy of a random choice
    for p in prompts:
        o, offsets, off, _ = _forward(model, tok, p, torch)
        lines = list_structure(p, ser)
        htok = [char_to_token(offsets, off + h[1], off + h[2]) for h, _, _ in lines]
        if len(lines) < 2 or None in htok:
            continue
        for li, (h, c, entries) in enumerate(lines):
            for k, e in enumerate(entries, start=1):
                t = char_to_token(offsets, off + e[1], off + e[2])
                if t is None:
                    continue
                b = bin_of(dist_of(ser, k)); cnt[b] += 1
                chance.append(1.0 / len(lines))
                for l in range(L):
                    H = o.hidden_states[l][0, htok].float()
                    q = o.hidden_states[l][0, t].float()
                    sim = torch.nn.functional.cosine_similarity(H, q[None], dim=-1)
                    hits[b][l] += int(sim.argmax().item() == li)
    by_layer = {b: (hits[b] / max(cnt[b], 1)).tolist() for b in hits}
    layer = int(np.argmax(by_layer["1-8"]))
    return {"top1_by_layer_bin": by_layer, "n_by_bin": cnt, "layer": layer,
            "by_bin": {b: (by_layer[b][layer] if cnt[b] else None) for b in by_layer},     # null: no entry in the bin
            "top1_by_layer": (sum(hits.values()) / max(sum(cnt.values()), 1)).tolist(), "n_entries": sum(cnt.values()),
            "chance": float(np.mean(chance)) if chance else None}


def patch_pass(model, tok, recs, ser, torch, pool=POOL):
    """Fraction of the Yes-No logit difference restored by patching clean residuals at head-color tokens."""
    yes, no = answer_ids(tok, YES_WORDS), answer_ids(tok, NO_WORDS)        # same token sets as the main scoring
    L = model.config.num_hidden_layers
    # index i = 0..L of the residual stream: i = 0 is the embedding output, i >= 1 the output of block i (the
    # same indexing as hidden_states[i]); L + 1 values (29 for the 28-layer Qwen2.5-7B). Patching i = 0 replaces
    # the corrupted color embeddings by the clean ones (restores the clean run exactly); after block L the color
    # positions no longer feed the answer position (restores nothing).
    num = np.zeros(L + 1); den = 0.0; n = 0       # ratio of sums: sum_g (patched - corrupted) / sum_g (clean - corrupted)
    for r in recs:
        if not r["label"]:
            continue
        G = to_graph(r)
        clean = next(prompts_for(r, pool, serializations=(ser,), templates=("T1",)))["prompt"]
        lines = list_structure(clean, ser)
        # corrupt exactly the lines of v's red neighbors (CONFIG §6): their head color red -> blue, entries kept
        names = assign_names(r, pool)
        red_nbr = {names[u] for u in G[r["query"]] if G.nodes[u]["color"] == "red"}
        target = [(h, c, e) for h, c, e in lines if h[0] in red_nbr and c[0] == "red"]
        corr = clean
        for h, c, _ in sorted(target, key=lambda x: -x[1][1]):
            corr = corr[:c[1]] + "blue" + corr[c[2]:]
        oc, offc, offset_c, encc = _forward(model, tok, clean, torch)
        ok, offk, offset_k, enck = _forward(model, tok, corr, torch)
        if encc["input_ids"].shape != enck["input_ids"].shape:
            continue                                       # "red"/"blue" tokenized to different lengths
        diff = lambda lg: (torch.logsumexp(lg[list(yes)], 0) - torch.logsumexp(lg[list(no)], 0)).item()
        dc, dk = diff(oc.logits[0, -1].float()), diff(ok.logits[0, -1].float())
        if abs(dc - dk) < 1e-3:
            continue
        # token positions of the changed color words; each color word is one token, so token indices agree
        ctoks = [char_to_token(offc, offset_c + c[1], offset_c + c[2]) for h, c, _ in target]
        # clean residuals by index i: 0 = embedding output, i >= 1 = output of block i, captured by hooks (HF's
        # hidden_states[L] is taken after the final norm, so it is not the block-L output)
        mods = [model.model.embed_tokens] + list(model.model.layers)
        clean_out = {}
        hs_ = [m.register_forward_hook(lambda mod, inp, out, i=i: clean_out.__setitem__(
            i, (out[0] if isinstance(out, tuple) else out)[0, ctoks].detach().clone())) for i, m in enumerate(mods)]
        with torch.no_grad():
            model(**{k: v.to(model.device) for k, v in encc.items()})
        for h_ in hs_:
            h_.remove()
        n += 1
        for i in range(L + 1):
            def hook(mod, inp, out, i=i):
                hs = out[0] if isinstance(out, tuple) else out
                hs[0, ctoks] = clean_out[i].to(hs.dtype)
                return out
            mod_ = mods[i]
            hdl = mod_.register_forward_hook(hook)
            with torch.no_grad():
                lp = model(**{k: v.to(model.device) for k, v in enck.items()}).logits[0, -1].float()
            hdl.remove()
            num[i] += diff(lp) - dk
        den += dc - dk
    # ratio of sums per layer (not the mean of per-graph ratios, which graphs with a tiny clean-corrupted gap would
    # dominate); index 0 restores the clean run (1) and index L nothing (0) exactly as before
    return {"restored_by_layer": (num / den if den else num).tolist(), "n": n, "aggregation": "ratio_of_sums"}


def el_inner_pass(model, tok, recs, torch, pool=POOL):
    """Probe the own color of the second endpoint b at its token in every EL edge line, per layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    L = model.config.num_hidden_layers + 1
    X = {l: [] for l in range(L)}; y = []; g = []
    for gi, r in enumerate(recs):
        G = to_graph(r)
        p = next(prompts_for(r, pool, serializations=("EL",), templates=("T1",)))["prompt"]
        o, offsets, off, _ = _forward(model, tok, p, torch)
        colors = dict(re.findall(r" ([a-z]+) (red|blue)[,\n]", p.split("Edges:")[0] + "\n"))
        for m in re.finditer(r"-- ([a-z]+)\n", p.split("Edges:")[1] + "\n"):
            b = m.group(1)
            a0 = p.index("Edges:") + 6 + m.start(1)
            t = char_to_token(offsets, off + a0, off + a0 + len(b))
            if t is None or b not in colors:
                continue
            for l in range(L):
                X[l].append(o.hidden_states[l][0, t].float().cpu().numpy())
            y.append(colors[b] == "red"); g.append(gi)
    y, g = np.array(y), np.array(g)
    res = {}
    for l in range(L):
        Xl = np.stack(X[l]); pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(Xl, y, g):
            pred[te] = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(Xl[tr], y[tr]).predict(Xl[te])
        res[l] = float(0.5 * (np.mean(pred[y]) + np.mean(~pred[y == False])))      # balanced accuracy
    return {"acc_by_layer": res, "n": int(len(y))}


if __name__ == "__main__":
    main()
