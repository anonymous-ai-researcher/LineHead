"""CONFIG.md §6.13: three further tests of the route account, preregistered before any of their runs.

E19  Attention knockout and entry patching (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B; the E11a test prompts: the first
     300 E1 D1a pairs and E7 Q0 pairs, template T1, formats AL and NC).
     knockout   In every layer and every head, the attention from each list-entry name token to the head segment of
                its OWN line (the tokens from the start of the line to the first entry: head name, head color and
                the delimiters) is removed by an additive mask of -inf before the softmax (attention knockout); no
                head is selected. Control ("other"): the same entries lose their attention to the head segment of
                the PREVIOUS line instead (same kind and about the same number of key tokens; entries of the first
                line keep all their attention). Variants ko-own, ko-other.
     patch      The top-128 since heads of E11a (scores file of each model), recorded on the prompt with every line
                head color flipped (e11_heads.corrupt "recolor") and written into the clean run only at the
                list-entry name positions (e11_heads.entry_positions). Variant patch-recolor-entries128.
E20  List-line filler (same models and prompts, formats AL and NC): F in {8, 24} neutral words in every list line,
     either between the line head and its first entry ("inside": every entry moves F words away from its head) or
     after the last entry ("outside": the same words, the same prompt length, no distance added). Variants
     fill-inside<F>, fill-outside<F>. The edge-format counterpart is E14 (filler between the color line and the edges).
E21  Relational tables (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, Qwen2.5-72B): projects and people, a membership
     relation and one binary attribute of people (senior or junior), i.e. a bipartite colored graph whose project
     vertices carry no color. Four serializations, the counterparts of the four graph formats:
        rows     a line "Levels: <person> <level>, ..." and one line "<project> -- <person>" per membership  (EL)
        person   one line "<person> (<level>): <project>, ..." per person                                 (AL)
        project  one line "<project> is a project. Members: <person> (<level>), ..." per project          (NC)
        inline   the Levels line and one line "<project> -- <person> <level>" per membership               (ECL)
     Questions: T1 "Does project p have a senior member?" (depth one, ◇senior at a project vertex) and G1 "Are at least
     four people senior?" (global, PL+GC); 200 matched pairs each (one graph and two query projects with the same
     number of members for T1; one graph and a single flipped level for G1), templates T1 and T2.

Commands:
  python r19.py knockout --model M $NAMES_ARG --out results/e19_ko_<tag>.jsonl
  python r19.py patch    --model M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e19_patch_<tag>.jsonl
  python r19.py filler   --model M $NAMES_ARG --out results/e20_<tag>.jsonl
  python r19.py tables-build --out data/e21_instances.jsonl
  python r19.py tables-run   --model M $NAMES_ARG --out results/e21_<tag>.jsonl
  python r19.py summary results/e19_*.jsonl results/e20_*.jsonl results/e21_*.jsonl --out results/summary_r19.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from e9_mechanism import char_to_token, list_structure
from e11_heads import Patcher, chat, corrupt, entry_positions, load_model, score_prompts, test_records, tiny
from instances import prompts_for
from names import POOL, single_token_names

FILL_WORDS = ("this", "filler", "text", "carries", "no", "content")      # no pool name, no color, no answer word
FILLS = (8, 24)
LEVELS = ("senior", "junior")
TFMTS = ("rows", "person", "project", "inline")
TQ = {"T1": {"T1": "Question: Does project {p} have at least one senior member? Answer with Yes or No.",
             "T2": "Question: Is any member of project {p} senior? Answer with Yes or No."},
      "G1": {"T1": "Question: Are at least four people senior? Answer with Yes or No.",
             "T2": "Question: Is the number of senior people four or more? Answer with Yes or No."}}
INTRO = {"T1": "Here is a table of projects and their members. Every person is senior or junior.",
         "T2": "The following lists projects and people; each person has a level, senior or junior."}


# ---------------------------------------------------------------- E19: knockout
def line_parts(prompt, ser):
    """For each AL/NC line: (character start of the line, character start of its first entry or None, entry spans)."""
    out, pos = [], 0
    lines = list_structure(prompt, ser)
    starts = []
    for line in prompt.split("\n"):
        starts.append(pos)
        pos += len(line) + 1
    for hn, hc, ents in lines:
        ls = max(s for s in starts if s <= hn[1])
        out.append((ls, ents[0][1] if ents else None, ents))
    return out


def knockout_edges(offs, off, prompt, ser, kind):
    """Token pairs (query, key) to remove: list-entry name tokens -> the head segment of their own line (kind "own")
    or of the previous line (kind "other")."""
    parts = line_parts(prompt, ser)
    segs = []
    for ls, fe, ents in parts:
        a, b = off + ls, off + (fe if fe is not None else ls)
        segs.append([i for i, (s, e) in enumerate(offs) if s >= a and e <= b and e > s])
    pairs = []
    for i, (_, _, ents) in enumerate(parts):
        keys = segs[i] if kind == "own" else (segs[i - 1] if i > 0 else [])
        if not keys:
            continue
        for e in ents:
            q = char_to_token(offs, off + e[1], off + e[2])
            if q is not None:
                pairs += [(q, k) for k in keys if k < q]
    return pairs


def knockout_mask(T, pairs, torch, dtype):
    """Causal additive mask (1 x 1 x T x T) with -inf at the removed pairs (a custom 4D attention mask)."""
    neg = torch.finfo(dtype).min
    M = torch.triu(torch.full((T, T), neg, dtype=dtype), diagonal=1)
    if pairs:
        q, k = zip(*pairs)
        M[list(q), list(k)] = neg
    return M[None, None]


def score_knockout(tok, m, torch, prompt, ser, kind):
    text = chat(tok, prompt)
    off = text.index(prompt)
    enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist()
    pairs = knockout_edges(offs, off, prompt, ser, kind)
    T = enc["input_ids"].shape[1]
    mask = knockout_mask(T, pairs, torch, m.dtype).to(m.device)
    yes = sorted({t[0] for t in (tok.encode(w, add_special_tokens=False) for w in ("Yes", " Yes", "yes", " yes")) if t})
    no = sorted({t[0] for t in (tok.encode(w, add_special_tokens=False) for w in ("No", " No", "no", " no")) if t})
    with torch.no_grad():
        lg = m(input_ids=enc["input_ids"].to(m.device), attention_mask=mask).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    return float(torch.logsumexp(lp[yes], -1)), float(torch.logsumexp(lp[no], -1)), len(pairs)


class EntryPatcher(Patcher):
    """Patcher that writes the recorded head slices only at the given token positions (E19 patch)."""
    def run_at(self, tok, heads, clean, bad, positions):
        torch = self.torch
        by = {}
        for l, j in heads:
            by.setdefault(l, []).append(j)
        e_c = tok(chat(tok, clean), return_tensors="pt", add_special_tokens=False)
        e_b = tok(chat(tok, bad), return_tensors="pt", add_special_tokens=False)
        if e_c["input_ids"].shape != e_b["input_ids"].shape or not positions:
            return None
        rec, hs = {}, []
        for l, js in by.items():
            def grab(mod, inp, l=l, js=js):
                x = inp[0][0].view(-1, self.H, self.dh)
                rec[l] = {j: x[:, j, :].detach().clone() for j in js}
            hs.append(self._proj(l).register_forward_pre_hook(grab))
        with torch.no_grad():
            self.m(**{k: v.to(self.m.device) for k, v in e_b.items()})
        for h in hs:
            h.remove()
        idx = torch.tensor(positions)
        hs = []
        for l, js in by.items():
            def put(mod, inp, l=l, js=js):
                x = inp[0]
                y = x.view(*x.shape[:-1], self.H, self.dh).clone()
                for j in js:
                    y[0, idx, j, :] = rec[l][j][idx].to(y.dtype)
                return (y.view(*x.shape),) + tuple(inp[1:])
            hs.append(self._proj(l).register_forward_pre_hook(put))
        try:
            return score_prompts(tok, self.m, torch, [clean])[0]
        finally:
            for h in hs:
                h.remove()


# ---------------------------------------------------------------- E20: list-line filler
def filler_words(F):
    return " ".join(FILL_WORDS[i % len(FILL_WORDS)] for i in range(F))


def line_filler(prompt, ser, F, place):
    """Insert F neutral words in every AL/NC list line, before its first entry (inside) or after its last word
    (outside, before a final period under NC)."""
    if not F:
        return prompt
    w = filler_words(F)
    parts = line_parts(prompt, ser)
    edits = []
    starts, pos = [], 0
    for line in prompt.split("\n"):
        starts.append((pos, pos + len(line)))
        pos += len(line) + 1
    for ls, fe, _ in parts:
        le = next(e for s, e in starts if s == ls)
        if place == "inside":
            at = fe if fe is not None else prompt.rfind(" none", ls, le) + 1      # an empty list reads "none"
            edits.append((at, w + " "))
        else:
            at = le - 1 if prompt[le - 1] == "." else le
            edits.append((at, " " + w))
    out = prompt
    for at, s in sorted(edits, reverse=True):
        out = out[:at] + s + out[at:]
    return out


# ---------------------------------------------------------------- E21: relational tables
def tables_graph(rng, n_proj, n_people):
    level = {u: ("senior" if rng.random() < 0.25 else "junior") for u in range(n_people)}
    members = {p: sorted(rng.sample(range(n_people), rng.randint(2, 5))) for p in range(n_proj)}
    return level, members


def tables_build(seed=21, n_pairs=200):
    """Matched pairs. T1: one graph, two query projects with the same number of members, one with a senior member
    and one without. G1: one graph and the same graph with one level flipped, with 4 and 3 senior people."""
    rng = random.Random(seed)
    recs, pair = [], 0
    while sum(r["pid"] == "T1" for r in recs) < 2 * n_pairs:
        gs = rng.randrange(10 ** 9)
        g = random.Random(gs)
        level, members = tables_graph(g, g.randint(6, 9), g.randint(10, 14))
        pos = [p for p, ms in members.items() if any(level[u] == "senior" for u in ms)]
        neg = [p for p, ms in members.items() if all(level[u] == "junior" for u in ms)]
        cand = [(a, b) for a in pos for b in neg if len(members[a]) == len(members[b])]
        if not cand:
            continue
        a, b = g.choice(cand)
        for q, lab in ((a, True), (b, False)):
            recs.append(dict(pid="T1", pair=f"T1-{pair}", graph_seed=gs, level=level, members=members, query=q, label=lab))
        pair += 1
    pair = 0
    while sum(r["pid"] == "G1" for r in recs) < 2 * n_pairs:
        gs = rng.randrange(10 ** 9)
        g = random.Random(gs)
        level, members = tables_graph(g, g.randint(6, 9), g.randint(10, 14))
        sen = [u for u, v in level.items() if v == "senior"]
        if len(sen) != 4:
            continue
        lv2 = dict(level)
        lv2[g.choice(sen)] = "junior"
        for lv, lab in ((level, True), (lv2, False)):
            recs.append(dict(pid="G1", pair=f"G1-{pair}", graph_seed=gs, level=lv, members=members, query=None, label=lab))
        pair += 1
    for i, r in enumerate(recs):
        r["id"] = f"e21-{i}"
        r["level"] = {str(k): v for k, v in r["level"].items()}
        r["members"] = {str(k): v for k, v in r["members"].items()}
    return recs


def tables_names(rec, pool):
    g = random.Random(rec["graph_seed"] + 7)
    n_p, n_u = len(rec["members"]), len(rec["level"])
    w = g.sample(list(pool), n_p + n_u)
    return {f"p{i}": w[i] for i in range(n_p)} | {f"u{i}": w[n_p + i] for i in range(n_u)}


def tables_serialize(rec, names, fmt, order_seed):
    """The four serializations (orders shuffled by order_seed, the same for both prompts of a pair)."""
    g = random.Random(order_seed)
    lev = {int(k): v for k, v in rec["level"].items()}
    mem = {int(k): v for k, v in rec["members"].items()}
    P = lambda p: names[f"p{p}"]
    U = lambda u: names[f"u{u}"]
    edges = [(p, u) for p, ms in mem.items() for u in ms]
    g.shuffle(edges)
    people = sorted(lev, key=lambda u: g.random())
    projs = sorted(mem, key=lambda p: g.random())
    if fmt in ("rows", "inline"):
        head = " Levels: " + ", ".join(f"{U(u)} {lev[u]}" for u in people)
        body = [f" {P(p)} -- {U(u)}" + (f" {lev[u]}" if fmt == "inline" else "") for p, u in edges]
        return "\n".join([head, " Memberships:"] + body)
    if fmt == "person":
        of = {u: [p for p, ms in mem.items() if u in ms] for u in people}
        return "\n".join(f" {U(u)} ({lev[u]}): " + (", ".join(P(p) for p in sorted(of[u], key=lambda p: g.random())) or "none")
                         for u in people)
    if fmt == "project":
        return "\n".join(f" {P(p)} is a project. Members: " +
                         ", ".join(f"{U(u)} ({lev[u]})" for u in sorted(mem[p], key=lambda u: g.random())) + "." for p in projs)
    raise ValueError(fmt)


def tables_prompts(rec, pool):
    names = tables_names(rec, pool)
    out = []
    for fmt in TFMTS:
        body = tables_serialize(rec, names, fmt, rec["graph_seed"] + TFMTS.index(fmt))
        for tpl in ("T1", "T2"):
            q = TQ[rec["pid"]][tpl].format(p=names[f"p{rec['query']}"] if rec["query"] is not None else "")
            out.append(dict(id=rec["id"], pid=rec["pid"], pair=rec["pair"], label=rec["label"], ser=fmt, tpl=tpl,
                            prompt=f"{INTRO[tpl]}\n\n{body}\n\n{q}"))
    return out


# ---------------------------------------------------------------- commands
def _rec(p, **kw):
    r = {k: p[k] for k in ("id", "pid", "pair", "label", "ser", "tpl") if k in p}
    r.update(kw)
    return r


def cmd_knockout(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    prompts = [p for r in test_records(a, pool) for p in prompts_for(r, pool, serializations=("AL", "NC"), templates=("T1",))]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for kind in ("own", "other"):
            for p in prompts:
                y, n, k = score_knockout(tok, m, torch, p["prompt"], p["ser"], kind)
                f.write(json.dumps(_rec(p, k_min=p.get("k_min"), variant=f"ko-{kind}", lp_yes=y, lp_no=n, pred=y > n,
                                        n_removed=k, model=a.model)) + "\n")
    json.dump({"sec": round(time.time() - t0, 1), "n_gpus": max(1, torch.cuda.device_count())},
              open(Path(a.out).with_name(Path(a.out).stem + "_timing.json"), "w"))


def cmd_patch(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    top = [(l, j) for l, j, _ in json.load(open(a.scores))["scores"]][:128]
    pt = EntryPatcher(m, torch)
    rng = random.Random(f"e19-patch|{a.model}")
    prompts = [p for r in test_records(a, pool) for p in prompts_for(r, pool, serializations=("AL", "NC"), templates=("T1",))
               if p["pid"] == "D1a"]
    skipped = 0
    with open(a.out, "w") as f:
        for p in prompts:
            bad = corrupt(p["prompt"], p["ser"], "recolor", pool, rng)
            pos = entry_positions(tok, p["prompt"], p["ser"])
            sc = pt.run_at(tok, top, p["prompt"], bad, pos) if bad else None
            if sc is None:
                skipped += 1
                continue
            f.write(json.dumps(_rec(p, variant="patch-recolor-entries128", lp_yes=sc[0], lp_no=sc[1], pred=sc[0] > sc[1],
                                    model=a.model)) + "\n")
    json.dump({"sec": round(time.time() - t0, 1), "n_gpus": max(1, torch.cuda.device_count()), "skipped": skipped},
              open(Path(a.out).with_name(Path(a.out).stem + "_timing.json"), "w"))


def cmd_filler(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    prompts = [p for r in test_records(a, pool) for p in prompts_for(r, pool, serializations=("AL", "NC"), templates=("T1",))
               if p["pid"] == "D1a"]
    with open(a.out, "w") as f:
        for F in FILLS:
            for place in ("inside", "outside"):
                txt = [line_filler(p["prompt"], p["ser"], F, place) for p in prompts]
                for p, t, (y, n) in zip(prompts, txt, score_prompts(tok, m, torch, txt)):
                    f.write(json.dumps(_rec(p, k_min=p.get("k_min"), variant=f"fill-{place}{F}", lp_yes=y, lp_no=n,
                                            pred=y > n, model=a.model)) + "\n")
    json.dump({"sec": round(time.time() - t0, 1), "n_gpus": max(1, torch.cuda.device_count())},
              open(Path(a.out).with_name(Path(a.out).stem + "_timing.json"), "w"))


def cmd_tables_build(a):
    recs = tables_build()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(len(recs), "records")


def cmd_tables_run(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = single_token_names(a.names_from.split(",")) if a.names_from else POOL
    recs = [json.loads(l) for l in open(a.instances_e21)]
    if tiny():
        keep = {r["pair"] for r in recs[:8]} | {r["pair"] for r in recs if r["pid"] == "G1"}
        recs = [r for r in recs if r["pair"] in sorted(keep)[:8]]
    prompts = [p for r in recs for p in tables_prompts(r, pool)]
    with open(a.out, "w") as f:
        for p, (y, n) in zip(prompts, score_prompts(tok, m, torch, [p["prompt"] for p in prompts])):
            f.write(json.dumps(_rec(p, variant="e21", lp_yes=y, lp_no=n, pred=y > n, model=a.model)) + "\n")
    json.dump({"sec": round(time.time() - t0, 1), "n_gpus": max(1, torch.cuda.device_count())},
              open(Path(a.out).with_name(Path(a.out).stem + "_timing.json"), "w"))


def _per_pair(rows):
    by = {}
    for r in rows:
        by.setdefault(r["pair"], []).append((r["label"], bool(r["pred"]) == bool(r["label"])))
    out = {}
    for k, v in by.items():
        pos = [c for l, c in v if l]
        neg = [c for l, c in v if not l]
        if pos and neg:
            out[k] = 0.5 * (np.mean(pos) + np.mean(neg))
    return out


def ba_cell(rows, reps=2000, seed=0):
    """Balanced accuracy over pairs (each pair: one positive and one negative per template) with the 95% pair
    bootstrap of CONFIG §4."""
    per = np.array(list(_per_pair(rows).values()))
    if not len(per):
        return None
    rs = np.random.default_rng(seed)
    bs = per[rs.integers(0, len(per), (reps, len(per)))].mean(1)
    return dict(bal_acc=float(per.mean()), ci_low=float(np.quantile(bs, 0.025)), ci_high=float(np.quantile(bs, 0.975)),
                n_pairs=int(len(per)))


def paired_diff(a_rows, b_rows, reps=2000, seed=0):
    """Mean over pairs of BA(a) - BA(b) on the same pairs, with the one-sided upper 95% pair-bootstrap bound."""
    a, b = _per_pair(a_rows), _per_pair(b_rows)
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    rs = np.random.default_rng(seed)
    bs = d[rs.integers(0, len(d), (reps, len(d)))].mean(1)
    return dict(diff=float(d.mean()), hi95=float(np.quantile(bs, 0.95)), n_pairs=int(len(d)))


def cmd_summary(a):
    rows = [json.loads(l) for fn in a.results for l in open(fn)]
    G = {}
    for r in rows:
        G.setdefault((r["model"], r["variant"], r["ser"], r["pid"]), []).append(r)
    cells = [dict(model=m, variant=v, ser=s, pid=q, **ba_cell(rs)) for (m, v, s, q), rs in sorted(G.items())]
    diffs = []
    for m in sorted({k[0] for k in G}):
        for s in ("AL", "NC"):
            if (m, "ko-own", s, "D1a") in G and (m, "ko-other", s, "D1a") in G:
                diffs.append(dict(model=m, kind="ko", ser=s, **paired_diff(G[(m, "ko-own", s, "D1a")], G[(m, "ko-other", s, "D1a")])))
            for F in FILLS:
                ki, ko = (m, f"fill-inside{F}", s, "D1a"), (m, f"fill-outside{F}", s, "D1a")
                if ki in G and ko in G:
                    diffs.append(dict(model=m, kind="fill", ser=s, F=F, **paired_diff(G[ki], G[ko])))
    json.dump({"cells": cells, "diffs": diffs}, open(a.out, "w"), indent=1)
    print(a.out, len(cells), "cells", len(diffs), "paired differences")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["knockout", "patch", "filler", "tables-build", "tables-run", "summary"])
    ap.add_argument("results", nargs="*")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--q0-instances", default="data/e7_instances.jsonl")
    ap.add_argument("--instances-e21", default="data/e21_instances.jsonl")
    ap.add_argument("--names-from", default=None)
    ap.add_argument("--scores", default="results/e11_head_scores.json")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"knockout": cmd_knockout, "patch": cmd_patch, "filler": cmd_filler, "tables-build": cmd_tables_build,
     "tables-run": cmd_tables_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
