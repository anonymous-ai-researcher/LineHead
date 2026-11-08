"""E11a: since-head ablation in Qwen2.5-7B-Instruct (CONFIG.md §6.2). Reuses the E9 tooling (e9_mechanism.py:
list structure of AL/NC prompts, chat template, eager attention).

score   For every attention head (layer l, head j) the "line-head attention mass": on the D1a prompts of E1 (template
        T1, AL and NC, one prompt per pair), the mean over list-entry name tokens of the attention weight from the
        entry to the tokens of ITS OWN line head (head name and head color), minus the mean weight to the head
        tokens of the other lines (so a head that attends to every line head equally scores 0). A since step
        reads the most recent head color before an entry; a head that implements it puts its mass there.
        Output: results/e11_head_scores.json {"scores": [[l, j, score], ...] sorted by score, "n_prompts": ...}.
ablate  Mean ablation: the output of an ablated head (its slice of the input of o_proj) is replaced at every
        position by its mean over a calibration set (64 E1 prompts, 16 per format, disjoint from the test pairs).
        the mean is taken separately per format and per token role (names, color words, and every
        other token by its id), as Wang et al. (2023) compute means per template, so that the ablated value is the
        typical output of the head at a token of the same kind in the same format; a role not seen in calibration
        falls back to the format's mean over all its tokens. The earlier definition, one mean over all positions
        of all formats, is kept as a sensitivity check (variants pooled-top16 and pooled-rand16s<seed>). Conditions: none; the top-h heads of the ranking (h in 4, 8, 16); and for every h, 5 random head
        sets with the same number of heads per layer as the top-h set, drawn from the other heads (seeds 0-4).
        Test prompts: D1a and Q0 ("v is red") pairs (the first 300 E1 D1a pairs and E7 Q0 pairs; the head scores use the last 100 D1a pairs), formats EL, AL, NC, ECL, template
        T1; scored as run_llm.py (log p Yes vs log p No at the first answer token).
        Output: results/e11a_<model>.jsonl, one record per prompt and condition (field variant = none | top<h> |
        rand<h>s<seed> | pooled-top16 | pooled-rand16s<seed>).
E18     (CONFIG.md §6.12) controls of the since-head ablation:
        score-edge  edge-head scores on the EL scoring prompts: the attention weight from the second endpoint b of an
                    edge line to b's own entry in the color line (name and color), minus the mean weight to the other
                    color-line entries (the inner match of Proposition 4(b)). Output: results/e18_edge_scores.json.
        ablate --scores results/e18_edge_scores.json --prefix e --hs 16,128 --no-base: the top edge heads and 5 random
                    sets in their layers (variants etop<h>, erand<h>s<seed>).
        ablate --prefix n --positions entries --hs 16,128 --no-base --no-random: the since heads ablated only at
                    list-entry name tokens (edge-line names under EL and ECL); variants ntop<h>.
        ablate --task degree --hs 128 --no-base: GraphQA node degree (data/e13_graphqa.jsonl, 200 questions per
                    format, greedy answers as e13_graphqa.py) under the top-128 since heads and 5 random sets of 128
                    heads; the unablated cells are those of E13. Positions generated after the prompt use the pooled mean.
E17     (CONFIG.md §6.11) the same command with --hs 32,64,128 --no-base: the dose of the ablation beyond
        16 heads on the same prompts, same scores and same calibration; only top<h> and rand<h>s<seed> are run (the
        unablated and top-16 conditions are those of E11a). Output: results/e17_<model>.jsonl.
summary Balanced accuracy per (cond, ser, pid) with 95% pair-bootstrap intervals (stats.pair_bootstrap); the drop
        from "none" per format; the random-head rows are pooled over the 5 seeds (mean and range), the pooled-mean
        sensitivity rows in "pooled_random".
Usage:
  python e11_heads.py score  --model Qwen/Qwen2.5-7B-Instruct $NAMES_ARG --out results/e11_head_scores.json
  python e11_heads.py ablate --model Qwen/Qwen2.5-7B-Instruct $NAMES_ARG --scores results/e11_head_scores.json \
      --out results/e11a_qwen7b.jsonl
  python e11_heads.py summary results/e11a_*.jsonl --out results/summary_e11a.json
  python e11_heads.py ablate --model Qwen/Qwen2.5-7B-Instruct $NAMES_ARG --scores results/e11_head_scores.json \
      --hs 32,64,128 --no-base --out results/e17_qwen7b.jsonl                      # E17 (CONFIG §6.11)
  python e11_heads.py summary results/e11a_*.jsonl results/e17_*.jsonl --out results/summary_e17.json
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from e9_mechanism import char_to_token, dedupe_pairs, list_structure, word_spans
from instances import load, prompts_for
from names import POOL, single_token_names

HS = (4, 8, 16)
SEEDS = 5
FMTS = ("EL", "AL", "NC", "ECL")


def tiny():
    return bool(os.environ.get("E2E_TINY"))


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32 if tiny() else torch.bfloat16,
                                             device_map=None if tiny() else "auto", attn_implementation="eager").eval()
    return tok, m, torch


def chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)


def head_scores(tok, m, torch, prompts_by_ser):
    L, H = m.config.num_hidden_layers, m.config.num_attention_heads
    acc, n = np.zeros((L, H)), 0
    for ser, prompts in prompts_by_ser.items():
        for p in prompts:
            text = chat(tok, p)
            off = text.index(p)
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
            offs = enc.pop("offset_mapping")[0].tolist()
            lines = list_structure(p, ser)
            heads = []
            for hn, hc, ents in lines:
                ht = [char_to_token(offs, off + s[1], off + s[2]) for s in (hn, hc) if s is not None]
                heads.append(([t for t in ht if t is not None], [char_to_token(offs, off + e[1], off + e[2]) for e in ents]))
            with torch.no_grad():
                att = m(**{k: v.to(m.device) for k, v in enc.items()}, output_attentions=True).attentions
            A = torch.stack([a[0].float() for a in att]).cpu().numpy()          # L x H x T x T
            sc, cnt = np.zeros((L, H)), 0
            for i, (own, ents) in enumerate(heads):
                other = [t for k, (o, _) in enumerate(heads) if k != i for t in o]
                for e in ents:
                    if e is None or not own:
                        continue
                    w_own = A[:, :, e, own].sum(-1)
                    w_oth = A[:, :, e, other].mean(-1) * len(own) if other else 0.0
                    sc += w_own - w_oth
                    cnt += 1
            if cnt:
                acc += sc / cnt
                n += 1
    acc /= max(n, 1)
    order = sorted(((l, j, float(acc[l, j])) for l in range(L) for j in range(H)), key=lambda x: -x[2])
    return {"scores": [list(x) for x in order], "n_prompts": n, "layers": L, "heads": H}


def el_structure(prompt):
    """For EL and ECL prompts: the color line as {name: (name_span, color_span)} and the edge lines as
    [(first_endpoint_span, second_endpoint_span)] (ECL lines "a c -- b c")."""
    words = word_spans(prompt)
    colors, edges, pos = {}, [], 0
    for line in prompt.split("\n"):
        start = pos
        pos += len(line) + 1
        lw = [w for w in words if start <= w[1] < pos]
        if lw and lw[0][0] == "Colors" and len(lw) > 2 and lw[1][0] == ":":
            rest = [w for w in lw[2:] if w[0] != ","]
            for nm, col in zip(rest[0::2], rest[1::2]):
                colors[nm[0]] = (nm, col)
        elif "--" in [w[0] for w in lw]:
            i = [w[0] for w in lw].index("--")
            if i >= 1 and i + 1 < len(lw):
                edges.append((lw[0], lw[i + 1]))
    return colors, edges


def edge_scores(tok, m, torch, prompts):
    """E18: per head, the attention from an edge line's second endpoint to its own color-line entry (name
    and color tokens) minus the mean attention to the other color-line entries (scaled to the same number of tokens)."""
    L, H = m.config.num_hidden_layers, m.config.num_attention_heads
    acc, n = np.zeros((L, H)), 0
    for p in prompts:
        text = chat(tok, p)
        off = text.index(p)
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offs = enc.pop("offset_mapping")[0].tolist()
        colors, edges = el_structure(p)
        ctok = {nm: [t for t in (char_to_token(offs, off + s_[1], off + s_[2]) for s_ in sp) if t is not None]
                for nm, sp in colors.items()}
        with torch.no_grad():
            att = m(**{k: v.to(m.device) for k, v in enc.items()}, output_attentions=True).attentions
        A = torch.stack([a_[0].float() for a_ in att]).cpu().numpy()
        sc, cnt = np.zeros((L, H)), 0
        for _, b in edges:
            t = char_to_token(offs, off + b[1], off + b[2])
            own = ctok.get(b[0], [])
            if t is None or not own:
                continue
            other = [x for nm, ts in ctok.items() if nm != b[0] for x in ts]
            w_own = A[:, :, t, own].sum(-1)
            w_oth = A[:, :, t, other].mean(-1) * len(own) if other else 0.0
            sc += w_own - w_oth
            cnt += 1
        if cnt:
            acc += sc / cnt
            n += 1
    acc /= max(n, 1)
    order = sorted(((l, j, float(acc[l, j])) for l in range(L) for j in range(H)), key=lambda x: -x[2])
    return {"scores": [list(x) for x in order], "n_prompts": n, "layers": L, "heads": H, "kind": "edge"}


def entry_positions(tok, prompt, ser):
    """E18: token positions of list-entry names (AL, NC) or of edge-line names (EL, ECL) in the chat text."""
    text = chat(tok, prompt)
    off = text.index(prompt)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    if ser in ("AL", "NC"):
        spans = [e for _, _, ents in list_structure(prompt, ser) for e in ents]
    else:
        spans = [x for ab in el_structure(prompt)[1] for x in ab]
    return sorted({t for t in (char_to_token(offs, off + s_[1], off + s_[2]) for s_ in spans) if t is not None})


def random_sets(top, L, H, h, seed):
    """h random heads with the per-layer counts of the top-h set, drawn from heads outside the top-h set."""
    rng = random.Random(f"e11a|{h}|{seed}")
    per = {}
    for l, j in top[:h]:
        per[l] = per.get(l, 0) + 1
    excl = {(l, j) for l, j in top[:h]}
    out, short = [], 0
    for l, c in sorted(per.items()):
        pool = [(l, j) for j in range(H) if (l, j) not in excl]
        out += rng.sample(pool, min(c, len(pool)))
        short += max(0, c - len(pool))
    # E17 (h up to 128): a layer may hold more top heads than it has other heads; the missing ones are drawn from the
    # nearest layers (outside the top-h set), so every random set has h heads (never needed for h <= 16 in E11a)
    if short:
        layers = sorted(range(L), key=lambda x: min(abs(x - l) for l in per))
        for l in layers:
            pool = [(l, j) for j in range(H) if (l, j) not in excl and (l, j) not in out]
            k = min(short, len(pool))
            out += rng.sample(pool, k)
            short -= k
            if not short:
                break
    return out


COLORS = {"red", "blue", "green"}


def token_roles(tok, ids, names):
    """The role of every token of a prompt: "N" for a name, "C" for a color word, else "t<id>"."""
    out = []
    for i in ids:
        w = tok.decode([int(i)]).strip().strip("(),.:;")
        out.append("N" if w in names else "C" if w.lower() in COLORS else f"t{int(i)}")
    return out


class Ablator:
    """Forward pre-hooks on o_proj: the slices of ablated heads are replaced by their calibration means, per format
    and token role or, with mode "pooled", by one mean over all calibration positions."""
    def __init__(self, m, torch):
        self.m, self.torch = m, torch
        self.L, self.H = m.config.num_hidden_layers, m.config.num_attention_heads
        self.dh = m.config.hidden_size // self.H
        self.mean = None            # pooled: L tensors H x dh
        self.table, self.keys = None, None   # per format: L tensors K x H x dh and {role: row}
        self.idx = None             # rows of the current prompt (one per position)
        self.mode = "role"
        self.pos = None             # positions to ablate (None: every position)
        self.hooks = []
        self.names = set()

    def _proj(self, l):
        return self.m.model.layers[l].self_attn.o_proj

    def calibrate(self, tok, prompts, fmts=None, names=()):
        torch = self.torch
        self.names = set(names)
        fmts = fmts or ["all"] * len(prompts)
        tot = [torch.zeros(self.H, self.dh, dtype=torch.float64) for _ in range(self.L)]
        n_tot = 0
        acc = {}                    # (fmt, role) -> [L tensors], count
        cur = {}

        def mk(l):
            def f(mod, inp):
                x = inp[0][0].detach().double().cpu().view(-1, self.H, self.dh)
                tot[l] += x.sum(0)
                for t, r in enumerate(cur["roles"]):
                    for key in ((cur["fmt"], r), (cur["fmt"], "*")):
                        if key not in acc:
                            acc[key] = [[torch.zeros(self.H, self.dh, dtype=torch.float64) for _ in range(self.L)], 0]
                        acc[key][0][l] += x[t]
                        if l == 0:
                            acc[key][1] += 1
            return f
        hs = [self._proj(l).register_forward_pre_hook(mk(l)) for l in range(self.L)]
        for p, f in zip(prompts, fmts):
            enc = tok(chat(tok, p), return_tensors="pt", add_special_tokens=False)
            cur["fmt"], cur["roles"] = f, token_roles(tok, enc["input_ids"][0].tolist(), self.names)
            n_tot += enc["input_ids"].shape[1]
            with torch.no_grad():
                self.m(**{k: v.to(self.m.device) for k, v in enc.items()})
        for h in hs:
            h.remove()
        self.mean = [t / max(n_tot, 1) for t in tot]
        self.keys = {k: i for i, k in enumerate(sorted(acc))}
        self.table = [torch.stack([acc[k][0][l] / max(acc[k][1], 1) for k in sorted(acc)]) for l in range(self.L)]

    def prepare(self, tok, prompt, fmt):
        """Rows of the per-format role means for the positions of one prompt (unseen roles: the format's mean)."""
        enc = tok(chat(tok, prompt), return_tensors="pt", add_special_tokens=False)
        roles = token_roles(tok, enc["input_ids"][0].tolist(), self.names)
        self.idx = self.torch.tensor([self.keys.get((fmt, r), self.keys.get((fmt, "*"), 0)) for r in roles])

    def set(self, heads, mode="role"):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.mode = mode
        by = {}
        for l, j in heads:
            by.setdefault(l, []).append(j)
        for l, js in by.items():
            def f(mod, inp, l=l, js=js):
                x = inp[0]
                y = x.view(*x.shape[:-1], self.H, self.dh).clone()
                for j in js:
                    if self.mode == "pooled" or self.idx is None or self.idx.shape[0] != y.shape[-3]:
                        if self.pos is None:            # (also every position generated after the prompt)
                            y[..., j, :] = self.mean[l][j].to(y.dtype).to(y.device)
                    elif self.pos is None:
                        y[0, :, j, :] = self.table[l][self.idx, j, :].to(y.dtype).to(y.device)
                    else:                               # entry positions only
                        y[0, self.pos, j, :] = self.table[l][self.idx[self.pos], j, :].to(y.dtype).to(y.device)
                return (y.view(*x.shape),) + tuple(inp[1:])
            self.hooks.append(self._proj(l).register_forward_pre_hook(f))


def corrupt(prompt, ser, kind, pool, rng):
    """Activation patching: the prompt with every line head renamed (kind "rename": a fresh pool name not
    in the prompt, so entries keep the old names) or with every head color flipped (kind "recolor"). Only line
    heads change; None if the line structure is not found (EL, ECL: no line heads)."""
    lines = list_structure(prompt, ser)
    if not lines:
        return None
    import re
    used = set(re.findall(r"[A-Za-z]+", prompt))
    fresh = iter(rng.sample([w for w in pool if w not in used], len(lines)) if kind == "rename" else [])
    edits = []
    for hn, hc, _ in lines:
        if kind == "rename":
            edits.append((hn[1], hn[2], next(fresh)))
        elif hc is not None:
            edits.append((hc[1], hc[2], {"red": "blue", "blue": "red"}.get(hc[0], hc[0])))
    out = prompt
    for a, b, w in sorted(edits, reverse=True):
        out = out[:a] + w + out[b:]
    return out


class Patcher(Ablator):
    """Activation patching. The slices of the chosen heads (the input of o_proj) are recorded on a
    corrupted prompt and written into the run on the clean prompt; prompts whose two token sequences differ in
    length are skipped (they cannot be aligned position by position)."""
    def run(self, tok, heads, clean, bad):
        torch = self.torch
        by = {}
        for l, j in heads:
            by.setdefault(l, []).append(j)
        e_c = tok(chat(tok, clean), return_tensors="pt", add_special_tokens=False)
        e_b = tok(chat(tok, bad), return_tensors="pt", add_special_tokens=False)
        if e_c["input_ids"].shape != e_b["input_ids"].shape:
            return None
        rec = {}
        hs = []
        for l, js in by.items():
            def grab(mod, inp, l=l, js=js):
                x = inp[0][0].view(-1, self.H, self.dh)
                rec[l] = {j: x[:, j, :].detach().clone() for j in js}
            hs.append(self._proj(l).register_forward_pre_hook(grab))
        with torch.no_grad():
            self.m(**{k: v.to(self.m.device) for k, v in e_b.items()})
        for h in hs:
            h.remove()
        hs = []
        for l, js in by.items():
            def put(mod, inp, l=l, js=js):
                x = inp[0]
                y = x.view(*x.shape[:-1], self.H, self.dh).clone()
                for j in js:
                    y[0, :, j, :] = rec[l][j].to(y.dtype)
                return (y.view(*x.shape),) + tuple(inp[1:])
            hs.append(self._proj(l).register_forward_pre_hook(put))
        try:
            return score_prompts(tok, self.m, torch, [clean])[0]
        finally:
            for h in hs:
                h.remove()


def cmd_patch(a):
    """Conditions patch-<kind><h> for the top-h heads (h = 4, 8, 16) and prand-<kind>16s<seed> for random
    head sets of the same layers (5 seeds), kind in {rename, recolor}; formats AL and NC (EL and ECL have no line
    heads). Records as in ablate, with field variant and skipped prompts counted in e11a_patch_timing.json."""
    import time
    t_start = time.time()
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    S = json.load(open(a.scores))
    top = [(l, j) for l, j, _ in S["scores"]]
    pt = Patcher(m, torch)
    recs = test_records(a, pool)
    prompts = [p for r in recs for p in prompts_for(r, pool, serializations=("AL", "NC"), templates=("T1",))]
    rng = random.Random(f"e11a-patch|{a.model}")
    conds = [(f"patch-{k}{h}", k, top[:h]) for k in ("rename", "recolor") for h in HS] + \
            [(f"prand-{k}16s{s}", k, random_sets(top, pt.L, pt.H, 16, s)) for k in ("rename", "recolor") for s in range(SEEDS)]
    skipped = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for cond, kind, heads in conds:
            for p in prompts:
                bad = corrupt(p["prompt"], p["ser"], kind, pool, rng)
                sc = pt.run(tok, heads, p["prompt"], bad) if bad else None
                if sc is None:
                    skipped += 1
                    continue
                rec = {k: p[k] for k in ("id", "pid", "cls", "pair", "label", "ser", "tpl", "k_min")}
                rec.update(variant=cond, heads=[list(x) for x in heads], lp_yes=sc[0], lp_no=sc[1], pred=sc[0] > sc[1],
                           model=a.model)
                f.write(json.dumps(rec) + "\n")
            print(cond, flush=True)
    json.dump({"sec": round(time.time() - t_start, 1), "n_gpus": max(1, torch.cuda.device_count()), "skipped": skipped},
              open(Path(a.out).with_name(Path(a.out).stem + "_timing.json"), "w"))


def score_prompts(tok, m, torch, prompts):
    yes = sorted({t[0] for t in (tok.encode(w, add_special_tokens=False) for w in ("Yes", " Yes", "yes", " yes")) if t})
    no = sorted({t[0] for t in (tok.encode(w, add_special_tokens=False) for w in ("No", " No", "no", " no")) if t})
    out = []
    for p in prompts:
        enc = tok(chat(tok, p), return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            lg = m(**{k: v.to(m.device) for k, v in enc.items()}).logits[0, -1].float()
        lp = torch.log_softmax(lg, -1)
        a, b = float(torch.logsumexp(lp[yes], -1)), float(torch.logsumexp(lp[no], -1))
        out.append((a, b))
    return out


def test_records(a, pool):
    d1 = [r for r in load(a.instances) if r["pid"] == "D1a"]
    q0 = [r for r in load(a.q0_instances) if r["pid"] == "Q0"]
    n = 8 if tiny() else a.pairs
    pairs = sorted({r["pair"] for r in d1})[:n] + sorted({r["pair"] for r in q0})[:n]
    keep = set(pairs)
    return [r for r in d1 + q0 if r["pair"] in keep]


def cmd_score(a):
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    recs = dedupe_pairs([r for r in load(a.instances) if r["pid"] == "D1a"])
    recs = recs[-(16 if tiny() else a.score_pairs):]           # scoring graphs: the last pairs, not the test pairs
    P = {s: [next(prompts_for(r, pool, serializations=(s,), templates=("T1",)))["prompt"] for r in recs] for s in ("AL", "NC")}
    res = head_scores(tok, m, torch, P)
    res["model"] = a.model
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("top heads:", res["scores"][:8])


def cmd_score_edge(a):
    """E18: edge-head scores on the EL prompts of the scoring pairs (the last D1a pairs, as cmd_score)."""
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    recs = dedupe_pairs([r for r in load(a.instances) if r["pid"] == "D1a"])
    recs = recs[-(16 if tiny() else a.score_pairs):]
    P = [next(prompts_for(r, pool, serializations=("EL",), templates=("T1",)))["prompt"] for r in recs]
    res = edge_scores(tok, m, torch, P)
    res["model"] = a.model
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("top edge heads:", res["scores"][:8])


def cmd_ablate_degree(a, tok, m, torch, ab, top, hs, pool):
    """E18c: GraphQA node degree under the top-h heads and 5 random sets (greedy answers, at most 8 new
    tokens; the first integer is compared with the truth, as e13_graphqa.py)."""
    from e13_graphqa import parse_answer, prompt_of
    recs = [r for r in load(a.e13_instances) if r["task"] == "degree"][: (8 if tiny() else 200)]
    conds = [(f"top{h}", top[:h]) for h in hs] + [(f"rand{h}s{s}", random_sets(top, ab.L, ab.H, h, s))
                                                  for h in hs for s in range(SEEDS)]
    with open(a.out, "w") as f:
        for cond, heads in conds:
            ab.set(heads, "role")
            for ser in FMTS:
                for r in recs:
                    p, names = prompt_of(r, ser, pool)
                    ab.prepare(tok, p, ser)
                    enc = tok(chat(tok, p), return_tensors="pt", add_special_tokens=False)
                    with torch.no_grad():
                        g = m.generate(**{k: v.to(m.device) for k, v in enc.items()}, max_new_tokens=8, do_sample=False)
                    text = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    got = parse_answer("degree", text, names)
                    f.write(json.dumps({"id": r["id"], "pid": "degree", "cls": "graphqa", "pair": r["pair"], "label": True,
                                        "ser": ser, "tpl": "T1", "task": "degree", "variant": cond, "gen_text": text,
                                        "parsed": got, "exact": got == r["answer"], "pred": got == r["answer"],
                                        "model": a.model}) + "\n")
            print(cond, flush=True)
    ab.set([])


def cmd_ablate(a):
    import time
    t_start = time.time()
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    S = json.load(open(a.scores))
    top = [(l, j) for l, j, _ in S["scores"]]
    ab = Ablator(m, torch)
    cal = [r for r in load(a.instances) if r["cls"] == "depth1"][-(8 if tiny() else 64):]
    cfmt = [FMTS[i % 4] for i in range(len(cal))]
    ab.calibrate(tok, [next(prompts_for(r, pool, serializations=(f,), templates=("T1",)))["prompt"]
                       for r, f in zip(cal, cfmt)], fmts=cfmt, names=pool)
    recs = test_records(a, pool)
    prompts = [p for r in recs for p in prompts_for(r, pool, serializations=FMTS, templates=("T1",))]
    hs = tuple(int(x) for x in a.hs.split(","))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.task == "degree":                          # E18c
        cmd_ablate_degree(a, tok, m, torch, ab, top, hs, pool)
        return
    pre = a.prefix                                  # E18: "e" edge heads, "n" entry positions only
    conds = ([] if a.no_base else [("none", [], "role")]) + [(f"{pre}top{h}", top[:h], "role") for h in hs] + \
            ([] if a.no_random else [(f"{pre}rand{h}s{s}", random_sets(top, ab.L, ab.H, h, s), "role")
                                     for h in hs for s in range(SEEDS)])
    if not a.no_base:                   # E11a only: the pooled-mean sensitivity check of 
        conds += [("pooled-top16", top[:16], "pooled")] + \
                 [(f"pooled-rand16s{s}", random_sets(top, ab.L, ab.H, 16, s), "pooled") for s in range(SEEDS)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for cond, heads, mode in conds:
            ab.set(heads, mode)
            for p in prompts:
                ab.prepare(tok, p["prompt"], p["ser"])
                ab.pos = (torch.tensor(entry_positions(tok, p["prompt"], p["ser"]), dtype=torch.long)
                          if a.positions == "entries" and heads else None)
                ly, ln = score_prompts(tok, m, torch, [p["prompt"]])[0]
                rec = {k: p[k] for k in ("id", "pid", "cls", "pair", "label", "ser", "tpl", "k_min")}
                rec.update(variant=cond, heads=[list(x) for x in heads], lp_yes=ly, lp_no=ln, pred=ly > ln, model=a.model)
                f.write(json.dumps(rec) + "\n")
            print(cond, len(prompts), flush=True)
    ab.set([])
    ab.pos = None
    json.dump({"sec": round(time.time() - t_start, 1), "n_gpus": max(1, torch.cuda.device_count())},
              open(Path(a.out).with_name(Path(a.out).stem + "_timing.json"), "w"))    # one file per model


def cmd_summary(a):
    from stats import bal_acc, cell_rng, load as load_recs, pair_bootstrap
    d = load_recs(a.results)
    reps = 50 if tiny() else 2000
    out = {"cells": []}
    deg = d[d.pid == "degree"] if "pid" in d and (d.pid == "degree").any() else None
    d = d[d.pid != "degree"]
    for (m, cond, ser, pid), x in d.groupby(["model", "variant", "ser", "pid"]):
        lo, hi = pair_bootstrap(x, reps, cell_rng(x))
        out["cells"].append(dict(model=m, cond=cond, ser=ser, pid=pid, bal_acc=bal_acc(x), ci_low=float(lo),
                                 ci_high=float(hi), n_pairs=int(x.pair.nunique())))
    base = {(c["model"], c["ser"], c["pid"]): c["bal_acc"] for c in out["cells"] if c["cond"] == "none"}
    erd = {}
    rnd, prn, pool = {}, {}, {}
    for c in out["cells"]:
        c["drop"] = base.get((c["model"], c["ser"], c["pid"]), float("nan")) - c["bal_acc"]
        if c["cond"].startswith("erand"):              # E18: random sets in the layers of the edge heads
            h = c["cond"][5:].split("s")[0]
            erd.setdefault((c["model"], h, c["ser"], c["pid"]), []).append(c["bal_acc"])
        elif c["cond"].startswith("rand"):
            h = c["cond"][4:].split("s")[0]
            rnd.setdefault((c["model"], h, c["ser"], c["pid"]), []).append(c["bal_acc"])
        elif c["cond"].startswith("pooled-rand"):        # pooled-mean sensitivity check
            pool.setdefault((c["model"], c["ser"], c["pid"]), []).append(c["bal_acc"])   # kept apart from random_patch
        elif c["cond"].startswith("prand-"):           # random-head patching, pooled over seeds
            kind = c["cond"][6:].rsplit("16s", 1)[0]
            prn.setdefault((c["model"], kind, c["ser"], c["pid"]), []).append(c["bal_acc"])
    out["random"] = [dict(model=m, h=int(h), ser=s, pid=p, mean=float(np.mean(v)), min=float(min(v)), max=float(max(v)),
                          n_sets=len(v)) for (m, h, s, p), v in sorted(rnd.items())]
    out["edge_random"] = [dict(model=m, h=int(h), ser=s, pid=p, mean=float(np.mean(v)), min=float(min(v)),
                               max=float(max(v)), n_sets=len(v)) for (m, h, s, p), v in sorted(erd.items())]
    out["random_patch"] = [dict(model=m, kind=k, h=16, ser=s, pid=p, mean=float(np.mean(v)), min=float(min(v)),
                                max=float(max(v)), n_sets=len(v)) for (m, k, s, p), v in sorted(prn.items())]
    # sensitivity check (old pooled-mean definition): its top-16 rows stay in "cells" (cond pooled-top16), its
    # random-head rows are pooled over the 5 seeds here; neither enters the paper's tables
    out["pooled_random"] = [dict(model=m, h=16, ser=s, pid=p, mean=float(np.mean(v)), min=float(min(v)),
                                 max=float(max(v)), n_sets=len(v)) for (m, s, p), v in sorted(pool.items())]
    # E17 (CONFIG §6.11): the flip contrast under top-h, EL minus AL, minus NC and minus the mean of AL and NC, on
    # the D1a pairs (per-pair balanced accuracy, the same pairs under every format), with the one-sided 95%
    # pair-bootstrap bound (5th percentile of 2,000 resamples of the pairs)
    out["flip"] = []
    dd = d[d.pid == "D1a"].assign(correct=lambda x: (x.pred == x.label).astype(float))
    for (m, cond), x in dd.groupby(["model", "variant"]):
        if not (cond.startswith("top") or cond.startswith("ntop")):     # also entries only (E18)
            continue
        kind = "top" if cond.startswith("top") else "ntop"
        pp = x.groupby(["ser", "pair", "label"]).correct.mean().unstack("label").mean(axis=1).unstack("ser")
        if not {"EL", "AL", "NC"} <= set(pp.columns):
            continue
        pp = pp.dropna(subset=["EL", "AL", "NC"])
        rng = np.random.default_rng(sum(map(ord, f"{m}|{cond}")))          # deterministic in the cell
        idx = rng.integers(0, len(pp), size=(50 if tiny() else 2000, len(pp)))
        for name, diff in (("AL", pp.EL - pp.AL), ("NC", pp.EL - pp.NC), ("list", pp.EL - 0.5 * (pp.AL + pp.NC))):
            v = diff.to_numpy()
            out["flip"].append(dict(model=m, h=int(cond[len(kind):]), kind=kind, contrast=f"EL-{name}", diff=float(v.mean()),
                                    lo05=float(np.quantile(v[idx].mean(axis=1), 0.05)), n_pairs=len(v)))
    # E18c: GraphQA degree under ablation: exact-answer accuracy with the item bootstrap; random sets pooled
    out["degree"], out["degree_random"] = [], []
    if deg is not None:
        rb = np.random.default_rng(18)
        rd = {}
        for (m, cond, ser), x in deg.groupby(["model", "variant", "ser"]):
            y = x.exact.astype(float).to_numpy()
            bs = y[rb.integers(0, len(y), size=(50 if tiny() else 2000, len(y)))].mean(axis=1)
            cell = dict(model=m, cond=cond, ser=ser, acc=float(y.mean()), ci_low=float(np.quantile(bs, 0.025)),
                        ci_high=float(np.quantile(bs, 0.975)), n=len(y))
            if cond.startswith("rand"):
                rd.setdefault((m, int(cond[4:].split("s")[0]), ser), []).append(cell["acc"])
            else:
                out["degree"].append(cell)
        out["degree_random"] = [dict(model=m, h=h, ser=s, mean=float(np.mean(v)), min=float(min(v)), max=float(max(v)),
                                     n_sets=len(v)) for (m, h, s), v in sorted(rd.items())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "score-edge", "ablate", "patch", "summary"])
    ap.add_argument("results", nargs="*")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--q0-instances", default="data/e7_instances.jsonl")
    ap.add_argument("--names-from", default=None)
    ap.add_argument("--scores", default="results/e11_head_scores.json")
    ap.add_argument("--pairs", type=int, default=300, help="D1a and Q0 test pairs each (the first ones)")
    ap.add_argument("--score-pairs", type=int, default=100)
    ap.add_argument("--hs", default="4,8,16", help="numbers of top heads to ablate (E11a: 4,8,16; E17: 32,64,128)")
    ap.add_argument("--no-base", action="store_true", help="E17: skip the unablated and pooled-mean conditions")
    ap.add_argument("--prefix", default="", help="E18: variant prefix (e: edge heads, n: entry positions only)")
    ap.add_argument("--positions", default="all", choices=["all", "entries"], help="E18: ablate at every position or only at entry names")
    ap.add_argument("--no-random", action="store_true", help="E18: skip the random head sets")
    ap.add_argument("--task", default="d1q0", choices=["d1q0", "degree"], help="E18c: GraphQA node degree")
    ap.add_argument("--e13-instances", default="data/e13_graphqa.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "summary":
        res = cmd_summary(a)
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out)
    else:
        {"score": cmd_score, "score-edge": cmd_score_edge, "ablate": cmd_ablate, "patch": cmd_patch}[a.cmd](a)


if __name__ == "__main__":
    main()
