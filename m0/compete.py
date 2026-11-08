"""Competing explanations of the format effects (paper appendix "Competing Explanations").

static  (no model; measured on the frozen instance files) for every depth-one condition (format, mask) on the D1a
        pairs and for "v is red" (Q0) under the E7 masks, template T1:
        - window heuristic: answer Yes iff some unmasked occurrence of the query name has the word "red" within w
          tokens (w in 2, 4, 8, 16), and the same restricted to the occurrence's own line; balanced accuracy over
          the matched pairs;
        - relevant distance: word-level tokens between an unmasked occurrence of v and a color token of a red neighbor
          (Q0: of v itself); the owner of a color token is the vertex it describes (structure of the format).
fit     per-cell accuracies of the LLMs (depth one, all formats of E1/E4, and the E7 masks) explained by
        - a locality model:  logit BA = model effect + b1 log(1 + mean relevant distance) + b2 [Q0 cell]
        - the route model:   chance (0.5, no parameter) where the theory leaves no route; elsewhere the locality
          terms plus the route class of the cell from the theory (single match, nested match, transport only;
          App. new:prop-mask, Lemma ecl)
        both with chance (no parameter) for the conditions the theory proves undecidable; binomial quasi-likelihood GLM
        on the (prompt-weighted) accuracies (Pearson dispersion of the route model, which nests the locality model, QAIC; the AIC treats prompts as
        independent); the mean held-out log-likelihood per prompt and the held-out RMSE (all cells, divergent cells)
        when each condition is left out across models (leave-one-condition-out). Lower (Q)AIC / RMSE is better.
Usage: python compete.py static --instances data/m0_instances.jsonl --e7 data/e7_instances.jsonl --out results/compete_static.json
       python compete.py fit --static results/compete_static.json --e4 results/summary_e4.json \
           --e7 results/summary_e7.json --out results/summary_compete.json
"""
import argparse
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

from instances import MASK_WORD, SER_OFFSET, assign_names, load, prompts_for, to_graph
from names import POOL
from serialize import list_positions

WORD = re.compile(r"[A-Za-z]+|--|\n|[^\sA-Za-z]")
COLORS = {"red", "blue", "green"}
D1_FORMATS = ["EL", "AL", "NC", "ECL", "EL2", "ELfar", "NCH", "NCnw", "Incident", "Friendship", "GraphML"]
MASKED = [("AL", "heads"), ("AL", "entries"), ("NC", "heads"), ("NC", "entries")]
# route class per condition (theory): single = one match (since / inline color), nested = two nested matches,
# transport = only the name-transport (or head-anchored shift) route, none = no route (Prop. new:prop-mask)
ROUTE = {("EL", None): "nested", ("EL2", None): "nested", ("ELfar", None): "nested", ("GraphML", None): "nested",
         ("Friendship", None): "nested", ("AL", None): "single", ("NC", None): "single", ("NCnw", None): "single",
         ("Incident", None): "single", ("ECL", None): "single", ("NCH", None): "transport",
         ("AL", "heads"): "single", ("AL", "entries"): "none", ("NC", "heads"): "single", ("NC", "entries"): "transport"}
ROUTE_Q0 = {(None, "AL"): "single", (None, "NC"): "single", ("heads", "AL"): "none", ("entries", "AL"): "single",
            ("heads", "NC"): "single", ("entries", "NC"): "single"}


def annotate(text, names, rec, ser):
    """Word tokens of a serialization with, for every color token, the vertex it describes."""
    toks = [(m.group(0), m.start()) for m in WORD.finditer(text)]
    inv = {w: u for u, w in names.items()}
    owner = {}
    if ser == "ELfar":                         # j-th line of the color block describes the j-th edge's endpoints
        lines = text.split("\n ")
        blk = lines.index("Edge colors:")
        edge_lines = lines[2:blk]
        pos = 0
        line_of = []
        for i, ln in enumerate(lines):
            line_of += [i] * len(WORD.findall(ln + "\n"))
        for t, (w, _) in enumerate(toks):
            if w in COLORS and line_of[t] > blk:
                j = line_of[t] - blk - 1
                a_, b_ = [x.strip() for x in edge_lines[j].split("--")]
                k = sum(1 for tt in range(t) if line_of[tt] == line_of[t] and toks[tt][0] in COLORS)
                owner[t] = inv[(a_, b_)[k]]
    reach = 12 if ser == "GraphML" else 3
    for t, (w, _) in enumerate(toks):
        if w in COLORS and t not in owner:
            for s in range(t - 1, max(-1, t - reach - 1), -1):
                if toks[s][0] in inv:
                    owner[t] = inv[toks[s][0]]
                    break
    return [w for w, _ in toks], owner


def condition(rec, ser, mask, pool):
    p = next(prompts_for(rec, pool, serializations=(ser,), templates=("T1",)))
    names = assign_names(rec, pool)
    G = to_graph(rec)
    text = p["prompt"].split("\n\n")[1]
    words, owner = annotate(text, names, rec, ser)
    lines = np.cumsum([w == "\n" for w in words])
    nv = names[rec["query"]]
    # positions of the query name that survive the mask: heads = first name of a line, entries = the others (lists)
    occ = []
    for t, w in enumerate(words):
        if w != nv:
            continue
        is_head = (t == 0 or words[t - 1] == "\n" or (ser == "Incident" and words[t - 1] == "Node"))
        if (mask == "heads" and is_head) or (mask == "entries" and not is_head):
            continue
        occ.append(t)
    v = rec["query"]
    if rec["pid"] == "Q0":
        targets = [t for t, u in owner.items() if u == v]
    else:
        targets = [t for t, u in owner.items() if u in G[v] and words[t] == "red"]
    dist = min((abs(a - b) for a in occ for b in targets), default=len(words))
    yes = {w_: any(words[s] == "red" for t in occ for s in range(max(0, t - w_), min(len(words), t + w_ + 1)))
           for w_ in (2, 4, 8, 16)}
    yes.update({f"L{w_}": any(words[s] == "red" and lines[s] == lines[t] for t in occ
                              for s in range(max(0, t - w_), min(len(words), t + w_ + 1))) for w_ in (2, 4, 8, 16)})
    return dist, yes


def ba(rows, w):
    pos = [r["yes"][w] for r in rows if r["label"]]
    neg = [not r["yes"][w] for r in rows if not r["label"]]
    return 0.5 * (np.mean(pos) + np.mean(neg))


def cmd_static(a):
    pool = POOL
    recs = [r for r in load(a.instances) if r["pid"] == "D1a"]
    q0 = [r for r in load(a.e7) if r["pid"] == "Q0"]
    if os.environ.get("E2E_TINY"):
        recs, q0 = recs[:40], q0[:40]
    out = {"d1a": {}, "q0": {}}
    for ser, mask in [(f, None) for f in D1_FORMATS] + MASKED:
        rows = []
        for r in recs:
            d, y = condition(r, ser, mask, pool)
            rows.append(dict(label=r["label"], dist=d, yes=y))
        dpos = [r["dist"] for r in rows if r["label"]]
        out["d1a"][f"{ser}|{mask or 'none'}"] = dict(
            window={str(w): float(ba(rows, w)) for w in (2, 4, 8, 16)},
            window_line={str(w): float(ba(rows, f"L{w}")) for w in (2, 4, 8, 16)}, log_dist=float(np.mean(np.log1p(dpos))))
    # chance of the head-name retrieval of E9 (one prompt per D1a pair): mean of 1 / #line heads over entries
    seen, num, den = set(), 0.0, 0
    for r in recs:
        if r["pair"] in seen:
            continue
        seen.add(r["pair"])
        G = to_graph(r)
        num += 2 * G.number_of_edges() / G.number_of_nodes()
        den += 2 * G.number_of_edges()
    out["retrieval_chance"] = num / den
    for ser, mask in [("AL", None), ("NC", None)] + MASKED:
        rows = []
        for r in q0:
            d, y = condition(r, ser, mask, pool)
            rows.append(dict(label=r["label"], dist=d, yes=y))
        out["q0"][f"{ser}|{mask or 'none'}"] = dict(log_dist=float(np.mean(np.log1p([r["dist"] for r in rows]))))
    return out


def fit_models(cells):
    """cells: [dict(model, cond, q0, ba, n, log_dist, route)]. Returns the fit of both models.
    Both models predict chance (0.5, no parameter) for the cells of conditions the theory proves undecidable
    (route "none"; Theorem 5, Prop. mask), so they are compared on the same free cells (fairness). On the
    other cells: locality, logit BA = model effect + b1 log(1 + d) + b2 [Q0]; route model, logit BA = model effect +
    route class (nested, transport vs single) + Q0 + b log(1 + d) (approximate since and transport decay with distance).
    Prompts of a pair share a graph, so they are not independent: the binomial GLM is fit as a quasi-likelihood with
    the Pearson dispersion c of the route model (which contains every term of the locality model), and QAIC = -2 LL / c + 2 (k + 1) is reported next
    to the AIC that treats prompts as independent. Held-out: each condition left out across models
    (leave-one-condition-out); log-likelihood per prompt, and the RMSE in accuracy points over all cells and over the
    divergent cells (NC without head colors and the masked conditions), both from the held-out predictions."""
    import statsmodels.api as sm
    models = sorted({c["model"] for c in cells})
    routes = ["single", "nested", "transport"]

    def X(cs, kind):
        rows = []
        for c in cs:
            r = [1.0 * (c["model"] == m) for m in models]
            if kind == "locality":
                r += [c["log_dist"], 1.0 * c["q0"]]
            else:
                r += [1.0 * (c["route"] == k) for k in routes[1:]] + [1.0 * c["q0"], c["log_dist"]]
            rows.append(r)
        return np.array(rows)

    def free(cs):
        return [c for c in cs if c["route"] != "none"]

    def fit(cs, kind):
        cs = free(cs)
        y = np.array([[c["ba"] * c["n"], (1 - c["ba"]) * c["n"]] for c in cs])
        return sm.GLM(y, X(cs, kind), family=sm.families.Binomial()).fit()

    def predict(res, cs, kind):
        out_ = np.full(len(cs), 0.5)
        idx = [i for i, c in enumerate(cs) if c["route"] != "none"]
        if idx:
            out_[idx] = np.clip(res.predict(X([cs[i] for i in idx], kind)), 1e-6, 1 - 1e-6)
        return out_

    def ll(res, cs, kind):
        return sum(c["n"] * (c["ba"] * math.log(pp) + (1 - c["ba"]) * math.log(1 - pp))
                   for c, pp in zip(cs, predict(res, cs, kind)))

    is_div = lambda c: c["cond"] in ("NCH|none",) or "|heads" in c["cond"] or "|entries" in c["cond"]
    # the route model contains every term of the locality model (model effects, log distance, Q0) plus the route
    # classes, so it is the union model: its Pearson dispersion scales both likelihoods
    un = fit(cells, "route")
    nfree = len(free(cells))
    chat = max(1.0, float(un.pearson_chi2) / max(1, nfree - int(X(cells[:1], "route").shape[1])))
    out = {"dispersion": chat}
    conds = sorted({c["cond"] for c in cells})
    for kind in ("locality", "route"):
        res = fit(cells, kind)
        k = int(X(cells[:1], kind).shape[1])
        held, n, se, se_div = 0.0, 0, [], []     # leave-one-condition-out (all models), squared error in accuracy
        heldpred = {}
        for cd in conds:
            tr = [c for c in cells if c["cond"] != cd]
            te = [c for c in cells if c["cond"] == cd]
            f_ = fit(tr, kind)
            held += ll(f_, te, kind)
            n += sum(c["n"] for c in te)
            for c, p_ in zip(te, predict(f_, te, kind)):
                heldpred[f'{c["model"]}|{c["cond"]}'] = float(p_)
                se.append((float(p_) - c["ba"]) ** 2)
                if is_div(c):
                    se_div.append((float(p_) - c["ba"]) ** 2)
        LL = ll(res, cells, kind)
        out[kind] = dict(aic=float(-2 * LL + 2 * k), qaic=float(-2 * LL / chat + 2 * (k + 1)), heldout_ll=held / n,
                         heldout_rmse_pts=100 * math.sqrt(np.mean(se)), k=k,
                         divergent_rmse_pts=100 * math.sqrt(np.mean(se_div)) if se_div else float("nan"),
                         nondivergent_rmse_pts=100 * math.sqrt(np.mean([e for e, c in zip(se, [c for cd in conds for c in cells if c["cond"] == cd]) if not is_div(c)])),
                         pred=dict(zip([f'{c["model"]}|{c["cond"]}' for c in cells], map(float, predict(res, cells, kind)))),
                         heldout_pred=heldpred)
    out["delta_aic"] = out["locality"]["aic"] - out["route"]["aic"]
    out["delta_qaic"] = out["locality"]["qaic"] - out["route"]["qaic"]
    out["delta_heldout_ll"] = out["route"]["heldout_ll"] - out["locality"]["heldout_ll"]
    out["delta_rmse_pts"] = out["locality"]["heldout_rmse_pts"] - out["route"]["heldout_rmse_pts"]
    out["delta_divergent_rmse_pts"] = out["locality"]["divergent_rmse_pts"] - out["route"]["divergent_rmse_pts"]
    out["delta_nondivergent_rmse_pts"] = out["locality"]["nondivergent_rmse_pts"] - out["route"]["nondivergent_rmse_pts"]
    out["n_divergent"] = sum(map(is_div, cells))
    # RMSE over the free cells only (the cells of undecidable conditions are 0.5 under both models), and a condition
    # bootstrap of the RMSE differences (locality minus route): conditions resampled with replacement, the held-out
    # predictions kept (each was made with its condition left out)
    key = lambda c: f'{c["model"]}|{c["cond"]}'
    err = {k_: {key(c): (out[k_]["heldout_pred"][key(c)] - c["ba"]) ** 2 for c in cells} for k_ in ("locality", "route")}
    fr = [c for c in cells if c["route"] != "none"]
    for k_ in ("locality", "route"):
        out[k_]["free_rmse_pts"] = 100 * math.sqrt(np.mean([err[k_][key(c)] for c in fr]))
    out["n_free"] = len(fr)
    out["delta_free_rmse_pts"] = out["locality"]["free_rmse_pts"] - out["route"]["free_rmse_pts"]
    rng = np.random.default_rng(0)
    reps = 50 if os.environ.get("E2E_TINY") else 2000
    by = {cd: [c for c in cells if c["cond"] == cd] for cd in conds}
    dv = {"all": [], "free": [], "divergent": []}
    for _ in range(reps):
        pick = [cells_ for cd in rng.choice(conds, len(conds)) for cells_ in by[cd]]
        for name_, sel in (("all", pick), ("free", [c for c in pick if c["route"] != "none"]),
                           ("divergent", [c for c in pick if is_div(c)])):
            if sel:
                r_ = [100 * math.sqrt(np.mean([err[k_][key(c)] for c in sel])) for k_ in ("locality", "route")]
                dv[name_].append(r_[0] - r_[1])
    out["rmse_diff_ci"] = {k_: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k_, v in dv.items() if v}
    return out


def cells_from_summaries(static, e4, e7):
    cells = []
    for c in e4["cells"]:
        key = (c["ser"], None)
        if key in ROUTE and f"{c['ser']}|none" in static["d1a"]:
            cells.append(dict(model=c["model"], cond=f"{c['ser']}|none", q0=False, ba=c["bal_acc"],
                              n=6 * c["n_pairs"], log_dist=static["d1a"][f"{c['ser']}|none"]["log_dist"], route=ROUTE[key]))
    for c in e7["cells"]:
        mask = None if c.get("mask") in (None, "none") else c["mask"]
        if mask not in (None, "heads", "entries"):
            continue
        if c["pid"] == "D1a" and mask is not None and (c["ser"], mask) in ROUTE:
            k = f"{c['ser']}|{mask}"
            cells.append(dict(model=c["model"], cond=k, q0=False, ba=c["bal_acc"], n=6 * c["n_pairs"],
                              log_dist=static["d1a"][k]["log_dist"], route=ROUTE[(c["ser"], mask)]))
        if c["pid"] == "Q0" and (mask, c["ser"]) in ROUTE_Q0:
            k = f"{c['ser']}|{mask or 'none'}"
            cells.append(dict(model=c["model"], cond="Q0:" + k, q0=True, ba=c["bal_acc"], n=6 * c["n_pairs"],
                              log_dist=static["q0"][k]["log_dist"], route=ROUTE_Q0[(mask, c["ser"])]))
    return cells


def cmd_fit(a):
    static = json.load(open(a.static))
    cells = cells_from_summaries(static, json.load(open(a.e4)), json.load(open(a.e7)))
    r = fit_models(cells)
    r["n_cells"] = len(cells)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["static", "fit"])
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--e7", default="data/e7_instances.jsonl")
    ap.add_argument("--static")
    ap.add_argument("--e4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = cmd_static(a) if a.cmd == "static" else cmd_fit(a)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "pred"})[:600])


if __name__ == "__main__":
    main()
