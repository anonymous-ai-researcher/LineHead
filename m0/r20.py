"""CONFIG.md §6.14: the LLM experiments E28-E31 that need code beyond run_llm.py, e11_heads.py and r19.py
(the others are run_llm.py runs listed in r20_llm.sh). Preregistered before any of their runs (rules C58-C65).

E28  Mechanism replication (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B; the E11a test prompts)
     replicate  (a) the E11a ablation (none, top16 since heads, 5 random sets of 16 heads with the per-layer counts of
                the top-16 set; per-format, per-role means calibrated on T1 as E11a) and the E19 knockout (ko-own,
                ko-other) on templates T2 and T3. Variants abl-<cond>, ko-<kind>; field tpl.
     stability  (b) the 100 scoring prompts of E11a (last 100 D1a pairs, AL and NC) split at random into two halves of
                50 (10 splits, seeds 0-9); the head ranking of each half (e11_heads.head_scores); Jaccard overlap of
                the two top-16 and the two top-128 sets. Output: one JSON per model.
     knockout-big  (c) the E19 knockout on Qwen2.5-72B, with the unablated condition scored in the same run.
     restore    (d) sufficiency: on the prompt with every line head color flipped (e11_heads.corrupt "recolor"), the
                outputs of the top-128 since heads recorded on the CLEAN prompt are written back only at the
                list-entry name positions (the reverse direction of E19 patch); controls: 5 random 128-head sets
                (e11_heads.random_sets). Variants clean, flip, restore128, restore-rand128s<seed>; every record keeps
                the clean label, so recovery = (BA(restore) - BA(flip)) / (BA(clean) - BA(flip)).
E29  routes-degree  E11b-L (data/e11bL_instances.jsonl: Delta in {6,12,24,48}, padded to a fixed length per format):
                the NC-padded records under NC, NC-H, NC with line heads masked and NC with entries masked
                (instances.prompts_for mask "heads" / "entries"). Variants nc, nch, nc-mask-heads, nc-mask-entries.
E30  filler-between  E20's prompts and models: F = 24 neutral words in every list line between the head and the first
                entry (inside, as E20) or as a separate line right after the list line (between: the within-line
                distances stay, the prompt grows by the same words). Variants fill-inside24, fill-between24.
E31  (d) build-large  D1a pairs on graphs with 100, 200, 500 vertices (G(n, 4/(n-1)), p_red ~ U[0.1, 0.4]), 200 pairs per
                size; every pair three times, with the two query lines (adjacent, random order) placed in the first,
                middle or last third of the list order (field third); run with run_llm.py.
     (e) prompting  Qwen2.5-7B and Llama-3.1-8B, D1a and G1 (first 200 pairs each), T1: two system prompts (sys1, sys2)
                under the five formats, and EL with the edges in BFS order from the first vertex of the color line
                (el-bfs). Few-shot K in {1, 5, 10} and source-sorted EL are run_llm.py --shots / --sort-edges.
     (h) build-algo / algo  connectivity A1 ("Is there a path between node u and node w?"; one graph with at least two
                components, w+ in u's component and w- not, equal degree), cycle A2 ("Does the graph contain a
                cycle?"; two graphs with the same n and m, one a forest), distance A3 ("Is the distance between node
                u and node w at most 2?"; one graph, w+ at distance 2 and w- at distance >= 3, equal degree); 200
                pairs each, graphs with 10-40 vertices and mean degree in [1.2, 2.4]; four models, five formats, T1.
     (j) build-latek  1,000 D1a pairs (E1 distribution) whose positive has k_min >= 5 under the AL order seed.
Commands:
  python r20.py replicate --model M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e28a_<tag>.jsonl
  python r20.py stability --model M $NAMES_ARG --out results/e28b_<tag>.json
  python r20.py knockout-big --model Qwen/Qwen2.5-72B-Instruct $NAMES_ARG --out results/e28c_qwen72b.jsonl
  python r20.py restore   --model M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e28d_<tag>.jsonl
  python r20.py routes-degree --model M $NAMES_ARG --out results/e29_<tag>.jsonl
  python r20.py filler-between --model M $NAMES_ARG --out results/e30_<tag>.jsonl
  python r20.py build-large --out data/e31d_instances.jsonl
  python r20.py build-algo  --out data/e31h_instances.jsonl
  python r20.py build-latek --out data/e31j_instances.jsonl
  python r20.py prompting --model M $NAMES_ARG --out results/e31e_<tag>.jsonl
  python r20.py algo      --model M $NAMES_ARG --out results/e31h_<tag>.jsonl
  python r20.py summary results/e28*.jsonl results/e29_*.jsonl results/e30_*.jsonl results/e31*.jsonl \
      --stability results/e28b_*.json --out results/summary_r20.json
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx

from build import pairs_one_graph, random_graph, record
from e9_mechanism import list_structure
from e11_heads import (FMTS, Ablator, chat, corrupt, entry_positions, head_scores, load_model, random_sets,
                       score_prompts, test_records, tiny)
from instances import SER_OFFSET, assign_names, load, prompts_for
from names import POOL, single_token_names
from prompts import render
from r19 import EntryPatcher, _rec, ba_cell, filler_words, line_filler, line_parts, paired_diff, score_knockout
from serialize import kmin_dmin, list_positions, serialize
from tasks import BY_ID

F5 = ("EL", "AL", "NC", "ECL", "NCH")
SYSTEM = {"sys1": "You are a helpful assistant.",
          "sys2": "You are a careful graph analyst. Read the graph description exactly as written and answer the "
                  "question about it."}
ALGO_Q = {"A1": "Is there a path between node {u} and node {w}?",
          "A2": "Does the graph contain a cycle?",
          "A3": "Is the distance between node {u} and node {w} at most 2?"}
SIZES = (100, 200, 500)


def _pool(a):
    return single_token_names(a.names_from.split(",")) if a.names_from else POOL


def _timing(a, t0, torch, **kw):
    json.dump(dict(sec=round(time.time() - t0, 1), n_gpus=max(1, torch.cuda.device_count()), **kw),
              open(Path(a.out).with_name(Path(a.out).stem + "_timing.json"), "w"))


def _write(f, p, **kw):
    f.write(json.dumps(_rec(p, k_min=p.get("k_min"), **kw)) + "\n")


# ---------------------------------------------------------------- E28 (a): T2 / T3 replication
def cmd_replicate(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = _pool(a)
    top = [(l, j) for l, j, _ in json.load(open(a.scores))["scores"]]
    ab = Ablator(m, torch)
    cal = [r for r in load(a.instances) if r["cls"] == "depth1"][-(8 if tiny() else 64):]
    cf = [FMTS[i % 4] for i in range(len(cal))]
    ab.calibrate(tok, [next(prompts_for(r, pool, serializations=(f,), templates=("T1",)))["prompt"]
                       for r, f in zip(cal, cf)], fmts=cf, names=pool)
    recs = test_records(a, pool)
    conds = [("none", [])] + [("top16", top[:16])] + [(f"rand16s{s}", random_sets(top, ab.L, ab.H, 16, s)) for s in range(5)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for tpl in ("T2", "T3"):
            prompts = [p for r in recs for p in prompts_for(r, pool, serializations=FMTS, templates=(tpl,))]
            for cond, heads in conds:
                ab.set(heads, "role")
                for p in prompts:
                    ab.prepare(tok, p["prompt"], p["ser"])
                    y, n = score_prompts(tok, m, torch, [p["prompt"]])[0]
                    _write(f, p, variant=f"abl-{cond}", lp_yes=y, lp_no=n, pred=y > n, model=a.model)
            ab.set([])
            for kind in ("own", "other"):
                for p in prompts:
                    if p["ser"] not in ("AL", "NC"):
                        continue
                    y, n, k = score_knockout(tok, m, torch, p["prompt"], p["ser"], kind)
                    _write(f, p, variant=f"ko-{kind}", lp_yes=y, lp_no=n, pred=y > n, n_removed=k, model=a.model)
    _timing(a, t0, torch)


# ---------------------------------------------------------------- E28 (c): knockout on Qwen2.5-72B, with its own baseline
def cmd_knockout_big(a):
    """The E19 knockout (own line head, previous line head) and the unablated condition on the E11a test prompts, AL
    and NC, T1 (E28c; the unablated cell is scored in the same run, so the losses need no other file)."""
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = _pool(a)
    prompts = [p for r in test_records(a, pool) for p in prompts_for(r, pool, serializations=("AL", "NC"), templates=("T1",))]
    with open(a.out, "w") as f:
        for p, (y, n) in zip(prompts, score_prompts(tok, m, torch, [p["prompt"] for p in prompts])):
            _write(f, p, variant="none", lp_yes=y, lp_no=n, pred=y > n, model=a.model)
        for kind in ("own", "other"):
            for p in prompts:
                y, n, k = score_knockout(tok, m, torch, p["prompt"], p["ser"], kind)
                _write(f, p, variant=f"ko-{kind}", lp_yes=y, lp_no=n, pred=y > n, n_removed=k, model=a.model)
    _timing(a, t0, torch)


# ---------------------------------------------------------------- E28 (b): ranking stability
def jaccard(a, b):
    a, b = set(map(tuple, a)), set(map(tuple, b))
    return len(a & b) / len(a | b) if a | b else 1.0


def split_halves(n, seed):
    idx = list(range(n))
    random.Random(f"e28b|{seed}").shuffle(idx)
    return sorted(idx[: n // 2]), sorted(idx[n // 2:])


def cmd_stability(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = _pool(a)
    from e9_mechanism import dedupe_pairs
    recs = dedupe_pairs([r for r in load(a.instances) if r["pid"] == "D1a"])[-(16 if tiny() else a.score_pairs):]
    P = {s: [next(prompts_for(r, pool, serializations=(s,), templates=("T1",)))["prompt"] for r in recs] for s in ("AL", "NC")}
    out = {"model": a.model, "splits": []}
    for seed in range(3 if tiny() else 10):
        h1, h2 = split_halves(len(recs), seed)
        r1 = head_scores(tok, m, torch, {s: [P[s][i] for i in h1] for s in P})["scores"]
        r2 = head_scores(tok, m, torch, {s: [P[s][i] for i in h2] for s in P})["scores"]
        out["splits"].append({"seed": seed, **{f"jac{h}": jaccard([x[:2] for x in r1[:h]], [x[:2] for x in r2[:h]])
                                               for h in (16, 128)}})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    _timing(a, t0, torch)


# ---------------------------------------------------------------- E28 (d): sufficiency (restore)
def cmd_restore(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = _pool(a)
    top = [(l, j) for l, j, _ in json.load(open(a.scores))["scores"]]
    pt = EntryPatcher(m, torch)
    rng = random.Random(f"e28d|{a.model}")
    prompts = [p for r in test_records(a, pool) for p in prompts_for(r, pool, serializations=("AL", "NC"), templates=("T1",))
               if p["pid"] == "D1a"]
    sets = [("restore128", top[:128])] + [(f"restore-rand128s{s}", random_sets(top, pt.L, pt.H, 128, s)) for s in range(5)]
    skipped = 0
    with open(a.out, "w") as f:
        for p in prompts:
            bad = corrupt(p["prompt"], p["ser"], "recolor", pool, rng)
            if not bad:
                skipped += 1
                continue
            (yc, nc), (yb, nb) = score_prompts(tok, m, torch, [p["prompt"], bad])
            pos = entry_positions(tok, bad, p["ser"])
            rows = [("clean", yc, nc), ("flip", yb, nb)]
            for name, heads in sets:
                sc = pt.run_at(tok, heads, bad, p["prompt"], pos)     # record on the clean prompt, write into the flipped run
                if sc is None:
                    break
                rows.append((name, sc[0], sc[1]))
            if len(rows) < 2 + len(sets):
                skipped += 1
                continue
            for name, y, n in rows:
                _write(f, p, variant=name, lp_yes=y, lp_no=n, pred=y > n, model=a.model)
    _timing(a, t0, torch, skipped=skipped)


# ---------------------------------------------------------------- E29: NC routes at high degree
def cmd_routes_degree(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = _pool(a)
    recs = [r for r in load(a.instances_e11bl) if r.get("ser_only") == "NC"]
    if tiny():
        recs = recs[:8]
    variants = [("nc", "NC", None), ("nch", "NCH", None), ("nc-mask-heads", "NC", "heads"), ("nc-mask-entries", "NC", "entries")]
    with open(a.out, "w") as f:
        for name, ser, mask in variants:
            ps = [p for r in recs for p in prompts_for(dict(r, ser_only=ser), pool, serializations=(ser,),
                                                       templates=("T1",), mask=mask)]
            for p, (y, n), r in zip(ps, score_prompts(tok, m, torch, [p["prompt"] for p in ps]), recs):
                _write(f, p, variant=name, degree_cell=r.get("degree_cell"), lp_yes=y, lp_no=n, pred=y > n, model=a.model)
    _timing(a, t0, torch)


# ---------------------------------------------------------------- E30: within-line vs between-line filler
def between_filler(prompt, ser, F):
    """F neutral words as a separate line right after every AL/NC list line (within-line distances unchanged)."""
    w = filler_words(F)
    ends, pos = {}, 0
    for line in prompt.split("\n"):
        ends[pos] = pos + len(line)
        pos += len(line) + 1
    out = prompt
    for ls in sorted({ls for ls, _, _ in line_parts(prompt, ser)}, reverse=True):
        at = ends[ls]
        out = out[:at] + "\n" + w + out[at:]
    return out


def cmd_filler_between(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = _pool(a)
    prompts = [p for r in test_records(a, pool) for p in prompts_for(r, pool, serializations=("AL", "NC"), templates=("T1",))
               if p["pid"] == "D1a"]
    with open(a.out, "w") as f:
        for name, fn in (("fill-inside24", lambda p: line_filler(p["prompt"], p["ser"], 24, "inside")),
                         ("fill-between24", lambda p: between_filler(p["prompt"], p["ser"], 24))):
            txt = [fn(p) for p in prompts]
            for p, (y, n) in zip(prompts, score_prompts(tok, m, torch, txt)):
                _write(f, p, variant=name, lp_yes=y, lp_no=n, pred=y > n, model=a.model)
    _timing(a, t0, torch)


# ---------------------------------------------------------------- E31 (d): large graphs
def large_graph(rng, n):
    G = nx.gnp_random_graph(n, 4.0 / (n - 1), seed=rng.randrange(2**31))
    pr = rng.uniform(0.1, 0.4)
    for u in G:
        G.nodes[u]["color"] = "red" if rng.random() < pr else "blue"
    return G


def place_lines(lists, qs, third, rng):
    """The list order with the query lines (adjacent, in random order) moved into the given third (0, 1, 2)."""
    rest = [x for x in lists if x[0] not in qs]
    q = [x for x in lists if x[0] in qs]
    rng.shuffle(q)
    n = len(lists)
    lo, hi = third * n // 3, max(third * n // 3, (third + 1) * n // 3 - len(q))
    at = rng.randint(lo, min(hi, len(rest)))
    return rest[:at] + q + rest[at:]


def build_large(seed=31, n_pairs=200):
    prop = BY_ID["D1a"]
    out = []
    for n in SIZES:
        rng = random.Random(f"e31d|{seed}|{n}")
        k = 0
        while k < n_pairs:
            G = large_graph(rng, n)
            by = defaultdict(lambda: {True: [], False: []})
            for v in G:
                if prop.eligible(G, v):
                    by[prop.query_key(G, v)][bool(prop.label(G, v))].append(v)
            keys = [q for q, b in by.items() if b[True] and b[False]]
            if not keys:
                continue
            key = rng.choice(sorted(keys))
            vp, vn = rng.choice(by[key][True]), rng.choice(by[key][False])
            gs = rng.randrange(2**31)
            for third in range(3):
                lists = list_positions(G, "AL", gs + SER_OFFSET["AL"])
                order = place_lines(lists, {vp, vn}, third, rng)
                for v, lab, kind in ((vp, True, "p"), (vn, False, "n")):
                    r = record(prop, G, v, lab, k, k, gs, "one_graph")
                    r.update(id=f"L{n}-{k:04d}-{third}{kind}", pair=f"L{n}-p{k:04d}-{third}", order=order,
                             third=third, size=n)
                    out.append(r)
            k += 1
    return out


# ---------------------------------------------------------------- E31 (h): algorithmic tasks
def sparse_graph(rng):
    n = rng.randint(10, 40)
    m = max(1, round(n * rng.uniform(1.2, 2.4) / 2))
    G = nx.gnm_random_graph(n, m, seed=rng.randrange(2**31))
    for u in G:
        G.nodes[u]["color"] = "red" if rng.random() < 0.3 else "blue"
    return G


def algo_label(pid, G, u, w):
    if pid == "A1":
        return nx.has_path(G, u, w)
    if pid == "A2":
        return not nx.is_forest(G)
    d = nx.shortest_path_length(G, u, w) if nx.has_path(G, u, w) else 99
    return d <= 2


def _algo_rec(pid, G, u, w, lab, k, gs, kind):
    return {"id": f"{pid}-{k:05d}{kind}", "pid": pid, "cls": "algo", "label": bool(lab), "query": u, "query2": w,
            "pair": f"{pid}-p{k:05d}", "n": G.number_of_nodes(), "edges": [list(e) for e in G.edges],
            "colors": [G.nodes[x]["color"] for x in range(G.number_of_nodes())], "graph_seed": gs}


def build_algo(seed=33, n_pairs=200):
    out = []
    for pid in ("A1", "A2", "A3"):
        rng = random.Random(f"e31h|{seed}|{pid}")
        k = 0
        pools = defaultdict(lambda: {True: [], False: []})
        while k < n_pairs:
            G = sparse_graph(rng)
            if pid == "A2":
                key = (G.number_of_nodes(), G.number_of_edges())
                pools[key][algo_label("A2", G, None, None)].append(G)
                if pools[key][True] and pools[key][False]:
                    for H, lab, kind in ((pools[key][True].pop(), True, "p"), (pools[key][False].pop(), False, "n")):
                        out.append(_algo_rec(pid, H, None, None, lab, k, rng.randrange(2**31), kind))
                    k += 1
                continue
            us = [u for u in G if G.degree(u) >= 1]
            rng.shuffle(us)
            for u in us:
                cand = defaultdict(lambda: {True: [], False: []})
                for w in G:
                    if w != u and G.degree(w) >= 1 and not G.has_edge(u, w):
                        if pid == "A3" and not nx.has_path(G, u, w):
                            continue                       # both queries connected to u (distance 2 vs >= 3)
                        cand[G.degree(w)][algo_label(pid, G, u, w)].append(w)
                ds = [d for d, b in cand.items() if b[True] and b[False]]
                if ds:
                    d = rng.choice(sorted(ds))
                    gs = rng.randrange(2**31)
                    for w, lab, kind in ((rng.choice(cand[d][True]), True, "p"), (rng.choice(cand[d][False]), False, "n")):
                        out.append(_algo_rec(pid, G, u, w, lab, k, gs, kind))
                    k += 1
                    break
    return out


def algo_prompts(rec, pool, sers=F5, tpl="T1"):
    G = nx.Graph()
    G.add_nodes_from(range(rec["n"]))
    G.add_edges_from(map(tuple, rec["edges"]))
    for u, c in enumerate(rec["colors"]):
        G.nodes[u]["color"] = c
    names = assign_names(rec, pool)
    q = ALGO_Q[rec["pid"]].format(u=names.get(rec["query"], ""), w=names.get(rec["query2"], ""))
    for s in sers:
        text = serialize(G, names, s, seed=rec["graph_seed"] + SER_OFFSET[s])
        yield {"id": rec["id"], "pid": rec["pid"], "pair": rec["pair"], "label": rec["label"], "ser": s, "tpl": tpl,
               "prompt": render(tpl, text, q)}


# ---------------------------------------------------------------- E31 (j): late positions
def build_latek(seed=35, n_pairs=1000, kmin=5):
    prop = BY_ID["D1a"]
    rng = random.Random(f"e31j|{seed}")
    out, k = [], 0
    while k < n_pairs:
        got, _ = pairs_one_graph(prop, 1, rng, 2, max_draws=10_000)
        if not got:
            continue
        Gp, vp, _, vn = got[0]
        gs = rng.randrange(2**31)
        if kmin_dmin(Gp, vp, "AL", gs + SER_OFFSET["AL"])[0] < kmin:
            continue
        for v, lab in ((vp, True), (vn, False)):
            r = record(prop, Gp, v, lab, k, 2 * k + (not lab), gs, "one_graph")
            r["id"] = f"J-{k:05d}{'p' if lab else 'n'}"
            r["pair"] = f"J-p{k:05d}"
            out.append(r)
        k += 1
    return out


def _dump(recs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(len(recs), "records ->", path)


# ---------------------------------------------------------------- E31 (e): system prompts and BFS edge order
def bfs_edges(prompt):
    """EL prompt with its edge lines reordered by breadth-first search from the first vertex of the color line
    (components in color-line order; within a vertex, neighbors in color-line order); endpoints keep their order."""
    lines = prompt.split("\n")
    ci = next(i for i, l in enumerate(lines) if l.lstrip().startswith("Colors:"))
    ei = next(i for i, l in enumerate(lines) if l.strip() == "Edges:")
    ej = ei + 1
    while ej < len(lines) and " -- " in lines[ej]:
        ej += 1
    order = [x.split()[0] for x in lines[ci].split(":", 1)[1].split(",")]
    rank = {v: i for i, v in enumerate(order)}
    edges = [(l, l.split(" -- ")[0].split()[-1], l.split(" -- ")[1].split()[0]) for l in lines[ei + 1: ej]]
    G = nx.Graph()
    G.add_nodes_from(order)
    G.add_edges_from((a, b) for _, a, b in edges)
    seen, pos = set(), {}
    for root in order:
        if root in seen:
            continue
        seen.add(root)
        queue = [root]
        while queue:
            u = queue.pop(0)
            pos[u] = len(pos)
            for w in sorted(G[u], key=rank.get):
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
    edges.sort(key=lambda e: (min(pos[e[1]], pos[e[2]]), max(pos[e[1]], pos[e[2]])))
    return "\n".join(lines[: ei + 1] + [e[0] for e in edges] + lines[ej:])


def score_system(tok, m, torch, prompts, system):
    yes = sorted({t[0] for t in (tok.encode(w, add_special_tokens=False) for w in ("Yes", " Yes", "yes", " yes")) if t})
    no = sorted({t[0] for t in (tok.encode(w, add_special_tokens=False) for w in ("No", " No", "no", " no")) if t})
    out = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            lg = m(**{k: v.to(m.device) for k, v in enc.items()}).logits[0, -1].float()
        lp = torch.log_softmax(lg, -1)
        out.append((float(torch.logsumexp(lp[yes], -1)), float(torch.logsumexp(lp[no], -1))))
    return out


def cmd_prompting(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = _pool(a)
    recs = load(a.instances)
    n = 8 if tiny() else a.pairs
    keep = set()
    for pid in ("D1a", "G1"):
        keep |= set(sorted({r["pair"] for r in recs if r["pid"] == pid})[:n])
    recs = [r for r in recs if r["pair"] in keep]
    with open(a.out, "w") as f:
        for sname, sys_ in SYSTEM.items():
            ps = [p for r in recs for p in prompts_for(r, pool, serializations=F5, templates=("T1",))]
            for p, (y, n_) in zip(ps, score_system(tok, m, torch, [p["prompt"] for p in ps], sys_)):
                _write(f, p, variant=sname, lp_yes=y, lp_no=n_, pred=y > n_, model=a.model)
        ps = [p for r in recs for p in prompts_for(r, pool, serializations=("EL",), templates=("T1",))]
        for p, (y, n_) in zip(ps, score_prompts(tok, m, torch, [bfs_edges(p["prompt"]) for p in ps])):
            _write(f, p, variant="el-bfs", lp_yes=y, lp_no=n_, pred=y > n_, model=a.model)
    _timing(a, t0, torch)


def cmd_algo(a):
    t0 = time.time()
    tok, m, torch = load_model(a.model)
    pool = _pool(a)
    recs = load(a.instances_algo)
    if tiny():
        recs = [r for r in recs if r["pair"] in {x["pair"] for x in recs[:2]} | {x["pair"] for x in recs if x["pid"] == "A2"}][:12]
    ps = [p for r in recs for p in algo_prompts(r, pool)]
    with open(a.out, "w") as f:
        for p, (y, n) in zip(ps, score_prompts(tok, m, torch, [p["prompt"] for p in ps])):
            _write(f, p, variant="algo", lp_yes=y, lp_no=n, pred=y > n, model=a.model)
    _timing(a, t0, torch)


# ---------------------------------------------------------------- summary
def cmd_summary(a):
    rows = [json.loads(l) for fn in a.results for l in open(fn) if l.strip()]
    G = defaultdict(list)
    for r in rows:
        G[(r["model"], r["variant"], r["ser"], r["pid"], r.get("tpl", "T1"), r.get("degree_cell"))].append(r)
    cells = [dict(model=m, variant=v, ser=s, pid=q, tpl=t, degree_cell=d, **(ba_cell(rs) or {}))
             for (m, v, s, q, t, d), rs in sorted(G.items(), key=lambda kv: tuple(map(str, kv[0])))]
    diffs = []
    for (m, v, s, q, t, d), rs in G.items():
        if v == "fill-inside24" and (m, "fill-between24", s, q, t, d) in G:
            diffs.append(dict(model=m, kind="inside-between", ser=s, **paired_diff(rs, G[(m, "fill-between24", s, q, t, d)])))
    rec = []
    by = {(c["model"], c["variant"], c["ser"]): c.get("bal_acc") for c in cells if c["pid"] == "D1a"}
    for (m, v, s), ba in by.items():
        if v == "restore128" and by.get((m, "clean", s)) is not None and by.get((m, "flip", s)) is not None:
            den = by[(m, "clean", s)] - by[(m, "flip", s)]
            fr = lambda x: (x - by[(m, "flip", s)]) / den if den else None
            rnd = [fr(by[(m, f"restore-rand128s{k}", s)]) for k in range(5) if by.get((m, f"restore-rand128s{k}", s)) is not None]
            rnd = [x for x in rnd if x is not None]          # None: clean and flipped prompts score alike
            rec.append(dict(model=m, ser=s, recovery=fr(ba), recovery_rand_mean=sum(rnd) / len(rnd) if rnd else None))
    stab = []
    for fn in a.stability or []:
        S = json.load(open(fn))
        stab.append({"model": S["model"], **{f"jac{h}": sum(x[f"jac{h}"] for x in S["splits"]) / len(S["splits"])
                                              for h in (16, 128)}})
    json.dump({"cells": cells, "diffs": diffs, "recovery": rec, "stability": stab}, open(a.out, "w"), indent=1)
    print(a.out, len(cells), "cells")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cmds = {"replicate": cmd_replicate, "stability": cmd_stability, "restore": cmd_restore, "knockout-big": cmd_knockout_big,
            "routes-degree": cmd_routes_degree, "filler-between": cmd_filler_between, "prompting": cmd_prompting,
            "algo": cmd_algo, "summary": cmd_summary,
            "build-large": lambda a: _dump(build_large(), a.out), "build-algo": lambda a: _dump(build_algo(), a.out),
            "build-latek": lambda a: _dump(build_latek(), a.out)}
    ap.add_argument("cmd", choices=list(cmds))
    ap.add_argument("results", nargs="*")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--q0-instances", default="data/e7_instances.jsonl")
    ap.add_argument("--instances-e11bl", default="data/e11bL_instances.jsonl")
    ap.add_argument("--instances-algo", default="data/e31h_instances.jsonl")
    ap.add_argument("--names-from", default=None)
    ap.add_argument("--scores", default="results/e11_head_scores.json")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--score-pairs", type=int, default=100)
    ap.add_argument("--stability", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "prompting" and a.pairs == 300:
        a.pairs = 200
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
