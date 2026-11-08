"""Group R: computability and common sense, run on the imported summary by analysis/validate.py (run(chk, P);
documented in VALIDATION_RULES.md, group R). R1 to R6 and R8 to R11 hold for any correct data under their stated
premises (the same run, displayed values within the stated rounding), so a failure is a bug: a wrong sample size,
an interval from the wrong method, a summary that does not reproduce its parts. R7 states what the measurement
normally gives; validate.py reports a violation of R7 as a warning to inspect (configuration, seeds, code) and,
once the code is confirmed, as a finding.

R1  count granularity: a balanced accuracy over N pairs (one positive and one negative prompt per pair and
    template, t templates) is (a + b) / (2 N t) for integers a, b; an exact-answer accuracy over n items is k / n.
    Every stored cell of a known design, rounded to four decimals, must be within rounding of such a value.
R2  rates: where TPR and TNR are stored with a balanced accuracy, BA = (TPR + TNR) / 2, and each rate is a count
    over its side (E12: 60 pairs).
R3  interval method: CONFIG §7 prescribes the 95% pair bootstrap. In every family of at least 10 cells strictly
    inside (0.05, 0.95), at most 60% of the intervals are symmetric to four decimals, and the ratio of the
    half-width to the independent-prompt floor is not constant (coefficient of variation >= 0.01); a normal
    approximation (p +- 1.96 se) fails both.
R4  pooled = mean of parts: a class cell of E1 is the mean of its three questions (equal designs); a four-model
    mean is the mean of the four models; E16's one-token cell is the mean of D1a and D1c.
R5  stored differences: every decline (E11b, E11b-L: Delta 6 minus 48) and contrast (E14: between minus before)
    equals the difference of its cells.
R6  summaries of repeated runs: a mean lies within [min, max] of its runs, and five random head sets never all
    coincide.
R7  common sense of the measurement:
    a) a trained model is not better on the larger test graphs than in distribution (mean OOD <= mean ID + 0.01;
       chance cells excepted);
    b) more training examples are not worse (E15: mean at the next size >= mean - 0.01, both test sets);
    c) ablating more of the selected heads is not milder (top-16 drop >= top-4 drop - 0.01 under AL and NC), and
       the selected heads cost more than random heads of the same layers (AL, NC, D1a);
    d) a longer filler is not better (E14: BA at 1,024 words <= BA without filler + 0.01, both positions);
    e) no balanced yes/no cell is clearly below chance (upper interval end >= 0.49): a model cannot be
       systematically wrong on balanced pairs without a scoring bug (exact-answer accuracies of E13 excepted);
    f) the number of examples to reach 0.95 (E15 n95) is what log-linear interpolation of the displayed seed
       means gives (within 1%).
    g) the probe's reliability falls with distance: E12 mean over k in {1, 2} >= mean over k in {16, 24};
    h) harder questions are not easier: depth two and counting <= depth one + 0.02, and "v is red" >= the
       neighbor question - 0.01 without masking;
    i) a longer filler is never better than a shorter one by more than 0.01 (E14);
    j) text lengths follow the format definitions: NC > AL > EL and ECL > EL on the same graphs.
R8  p-values lie in [0, 1]; a Holm-adjusted p-value is at least its raw p-value and at most 1, and within a
    family it is non-decreasing in the raw p-value.
R1, R3, R6, R7c and R7e also cover E17 (CONFIG §6.11): its cells are over the E11a prompts (300 pairs, T1), a larger
    dose (more selected or more random heads) is never milder (tolerance 0.015), and selected heads cost the list
    formats more than random heads.
R10 one evaluation, two summaries: the E3 Delta <= 256 cell equals the mean of its degree strata.
R11 separate runs are not copies: the learning-rate check and the per-format M-like head counts are not identical.
R9  the same prompts give the same cell: the unfilled E14 prompts are the first 300 E1 D1a pairs under T1, the
    prompts of the unablated E11a cell of the same model (EL, ECL), and E11b-L at Delta = 48 is unpadded, so its
    cells are E11b's. Value and interval must agree (to 0.0005); a difference means two runs of one condition
    disagree or a table reads the wrong subset.
"""
import math
import statistics as st

F4 = ("EL", "AL", "NC", "ECL")


def _d(d):
    """The dict-valued entries of a block (skips status and config_ref)."""
    return [(k, v) for k, v in d.items() if isinstance(v, dict)] if isinstance(d, dict) else []


def _gran(x, n):
    """x (rounded to four decimals) is k / n for an integer k."""
    return abs(x * n - round(x * n)) <= n * 0.00005 + 1e-6


def _cells(d, path=()):
    if isinstance(d, dict):
        if isinstance(d.get("ba"), (int, float)) and "lo" in d:
            yield path, d
        for k, v in d.items():
            if isinstance(v, dict):
                yield from _cells(v, path + (k,))


def run(chk, P):
    MODELS = [m for m in P["llm_main"]["per_question"]]

    # ---------------------------------------------------------------- R1 granularity
    def g(tag, x, n):
        if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            chk("R1", _gran(x, n), f"{tag} = {x} is not a count over {n}")

    for m, qs in _d(P["llm_main"]["per_question"]):          # E1: 400 pairs x 3 templates per question
        for q, row in qs.items():
            for f, x in row.items():
                g(f"E1 {m} {q} {f}", x, 2400)
    for m, cls in _d(P["llm_main"]["models"]):              # E1 class cells: three questions pooled
        for c, row in cls.items():
            for f, v in row.items():
                g(f"E1 class {m} {c} {f}", v["ba"], 7200)
    for m, fm in _d(P["llm_formats"]):                      # E4 class cells
        for f, cls in fm.items():
            for c, v in cls.items():
                if isinstance(v, dict) and "ba" in v:
                    g(f"E4 {m} {f} {c}", v["ba"], 7200)
    for m, conds in _d(P["llm_partial"]):                   # E7: one question, 400 pairs x 3 templates
        for cond, qs in conds.items():
            if isinstance(qs, dict):
                for q, v in qs.items():
                    if isinstance(v, dict) and "ba" in v:
                        g(f"E7 {m} {cond} {q}", v["ba"], 2400)
    for m, row in _d(P["mask_controls"]):                   # random masks and twin pairs: as E7
        for cond, qs in row.items():
            for q, v in qs.items():
                if isinstance(v, dict) and "ba" in v:
                    g(f"mask control {m} {cond} {q}", v["ba"], 2400)
    for m, row in _d(P["nl_formats"]):
        for f, v in row.items():
            g(f"NL format {m} {f}", v["depth1"]["ba"], 7200)
    for blk in ("e11a", "e11a_llama", "e11a_gemma"):            # 300 pairs, T1
        B = P.get(blk)
        if not B:
            continue
        for sub in ("none", "top", "patch"):
            for k, v in B.get(sub, {}).items():
                g(f"{blk} {sub} {k}", v["ba"], 600)
        for sub in ("random", "random_patch"):
            for k, v in B.get(sub, {}).items():
                g(f"{blk} {sub} {k} min", v["min"], 600)
                g(f"{blk} {sub} {k} max", v["max"], 600)
                g(f"{blk} {sub} {k} mean", v["mean"], 3000)    # mean of five sets
    for m, B in _d(P.get("e17", {})):                           # E17: the E11a prompts (300 pairs, T1)
        for k, v in B.get("top", {}).items():
            g(f"e17 {m} top {k}", v["ba"], 600)
        for k, v in B.get("random", {}).items():
            g(f"e17 {m} random {k} min", v["min"], 600)
            g(f"e17 {m} random {k} max", v["max"], 600)
            g(f"e17 {m} random {k} mean", v["mean"], 3000)
    for m, B in _d(P.get("e18", {})):                           # E18: the E11a prompts (300 pairs, T1); degree 200 items
        for sub in ("edge", "entry"):
            for k, v in B.get(sub, {}).items():
                g(f"e18 {m} {sub} {k}", v["ba"], 600)
        for k, v in B.get("edge_random", {}).items():
            g(f"e18 {m} edge_random {k} min", v["min"], 600)
            g(f"e18 {m} edge_random {k} max", v["max"], 600)
            g(f"e18 {m} edge_random {k} mean", v["mean"], 3000)
        for k, v in B.get("degree", {}).items():
            g(f"e18 {m} degree {k}", v["ba"], 200)
        for k, v in B.get("degree_random", {}).items():
            g(f"e18 {m} degree_random {k} mean", v["mean"], 1000)
    for m, B in _d(P.get("e19", {})):                           # E19: the E11a prompts (300 pairs, T1)
        for sub in ("ko", "patch"):
            for k, v in B.get(sub, {}).items():
                g(f"e19 {m} {sub} {k}", v["ba"], 600)
    for m, B in _d(P.get("e20", {})):                           # E20: the E11a D1a prompts (300 pairs, T1)
        for k, v in B.get("cells", {}).items():
            g(f"e20 {m} {k}", v["ba"], 600)
    for m, B in _d(P.get("e21", {})):                           # E21: 200 pairs x 2 templates
        if m == "mean":
            continue
        for k, v in B.items():
            if isinstance(v, dict) and "ba" in v:
                g(f"e21 {m} {k}", v["ba"], 800)
    for blk in ("e11b", "e11bL"):                               # 300 pairs x 3 templates
        for m, row in P.get(blk, {}).items():
            if isinstance(row, dict):
                for k, v in row.items():
                    if isinstance(v, dict) and "ba" in v:
                        g(f"{blk} {m} {k}", v["ba"], 1800)
    for m, row in _d(P.get("e14", {})):                     # 300 pairs, T1
        if isinstance(row, dict) and m != "words_none":
            for k, v in row.items():
                if isinstance(v, dict) and "ba" in v:
                    g(f"E14 {m} {k}", v["ba"], 600)
    for k, v in _d(P.get("e16", {})):                        # D1a + D1c: 800 pairs x 3 templates
        if "|names|" not in k:
            g(f"E16 {k}", v["ba"], 4800)
    for m, row in _d(P.get("e11c", {})):
        for f, v in row.items():
            g(f"E11c {m} {f}", v["ba"], 4800)
    for m, row in _d(P.get("reasoning", {})):                # D1a, T1: 400 pairs
        for f, v in row.items():
            g(f"reasoning {m} {f}", v["ba"], 800)
    for m, row in _d(P.get("e13", {})):
        for k, v in row.items():
            g(f"E13 {m} {k}", v["ba"], 400 if k.startswith("edge|") else 200)
    for c in P.get("e12", {}).get("cells", []):                  # 60 pairs per (format, k)
        tag = f"E12 {c['model']} {c['ser']} k={c['k']}"
        g(tag, c["bal_acc"], 120)
        # ---------------------------------------------------------- R2 rates
        if "tpr" in c and "tnr" in c:
            g(tag + " TPR", c["tpr"], 60)
            g(tag + " TNR", c["tnr"], 60)
            chk("R2", abs(c["bal_acc"] - 0.5 * (c["tpr"] + c["tnr"])) <= 0.0001, f"{tag}: BA != (TPR + TNR) / 2")

    # ---------------------------------------------------------------- R3 interval method
    fams = {}
    for blk in ("llm_main", "llm_formats", "llm_partial", "mask_controls", "nl_formats", "e11a", "e11a_llama",
                "e11a_gemma", "e11b", "e11bL", "e13", "e14", "e16", "graphml", "real_graphs", "robustness", "prompting",
                "e17"):
        for path, v in _cells(P.get(blk, {})):
            if 0.05 < v["ba"] < 0.95 and v["hi"] > v["lo"]:
                fams.setdefault(blk, []).append(v)
    for blk, cs in fams.items():
        if len(cs) < 10:
            continue
        sym = sum(abs((v["hi"] - v["ba"]) - (v["ba"] - v["lo"])) <= 0.0001 for v in cs) / len(cs)
        r = [(v["hi"] - v["lo"]) / (2 * math.sqrt(v["ba"] * (1 - v["ba"]))) for v in cs]
        cv = st.pstdev(r) / st.mean(r)
        chk("R3", sym <= 0.60, f"{blk}: {sym:.0%} of {len(cs)} intervals symmetric (normal approximation, not the pair bootstrap?)")
        chk("R3", cv >= 0.01, f"{blk}: interval / sqrt(p(1-p)) constant (CV {cv:.4f}): a formula, not a bootstrap")

    # ---------------------------------------------------------------- R4 pooled = mean of parts
    cls_q = {"global": ("G1", "G2", "G3"), "depth1": ("D1a", "D1b", "D1c"), "depth2": ("D2a", "D2b", "D2c"),
             "count": ("C2", "C3", "C2p")}
    for m in MODELS:
        pq, mc = P["llm_main"]["per_question"][m], P["llm_main"]["models"][m]
        for c, qs in cls_q.items():
            for f in ("EL", "AL", "NC"):
                if c in mc and all(q in pq for q in qs):
                    mean = sum(pq[q][f] for q in qs) / 3
                    chk("R4", abs(mc[c][f]["ba"] - mean) <= 0.00011,
                        f"E1 {m} {c} {f}: class {mc[c][f]['ba']} != mean of questions {mean:.5f}")
    mci = P["llm_main"].get("mean_ci", {})
    for c, row in mci.items():
        for f, v in row.items():
            if isinstance(v, dict) and "ba" in v and all(c in P["llm_main"]["models"][m] for m in MODELS):
                mean = sum(P["llm_main"]["models"][m][c][f]["ba"] for m in MODELS) / len(MODELS)
                chk("R4", abs(v["ba"] - mean) <= 0.00011, f"four-model mean {c} {f}: {v['ba']} != {mean:.5f}")
    for k, v in _d(P.get("e16", {})):
        if "|names|" in k:
            m, _, f = k.split("|")
            pq = P["llm_main"]["per_question"][m]
            chk("R4", abs(v["ba"] - 0.5 * (pq["D1a"][f] + pq["D1c"][f])) <= 0.00006, f"E16 {k}: not the E1 mean of D1a, D1c")

    # ---------------------------------------------------------------- R5 stored differences
    for blk in ("e11b", "e11bL"):
        for m, row in P.get(blk, {}).items():
            if isinstance(row, dict):
                for f in F4:
                    if f"{f}|decline" in row:
                        d = row[f"{f}|6"]["ba"] - row[f"{f}|48"]["ba"]
                        chk("R5", abs(row[f"{f}|decline"] - d) <= 0.00011, f"{blk} {m} {f}: decline != cell 6 - cell 48")
    for m, row in _d(P.get("e14", {})):
        if isinstance(row, dict) and m != "words_none":
            for f in ("EL", "ECL"):
                for F in (64, 256, 1024):
                    k = f"{f}|contrast|{F}"
                    if k in row:
                        d = row[f"{f}|between|{F}"]["ba"] - row[f"{f}|before|{F}"]["ba"]
                        chk("R5", abs(row[k] - d) <= 0.00011, f"E14 {m} {k} != between - before")

    # ---------------------------------------------------------------- R6 repeated runs
    for blk in ("e11a", "e11a_llama", "e11a_gemma"):
        for sub in ("random", "random_patch"):
            for k, v in P.get(blk, {}).get(sub, {}).items():
                chk("R6", v["min"] - 1e-9 <= v["mean"] <= v["max"] + 1e-9, f"{blk} {sub} {k}: mean outside [min, max]")
                chk("R6", v["max"] > v["min"], f"{blk} {sub} {k}: five random head sets coincide")
    for m, B in _d(P.get("e17", {})):
        for k, v in B.get("random", {}).items():
            chk("R6", v["min"] - 1e-9 <= v["mean"] <= v["max"] + 1e-9, f"e17 {m} random {k}: mean outside [min, max]")
            chk("R6", v["max"] > v["min"], f"e17 {m} random {k}: five random head sets coincide")
    for m, B in _d(P.get("e18", {})):
        for sub in ("edge_random", "degree_random"):
            for k, v in B.get(sub, {}).items():
                chk("R6", v["min"] - 1e-9 <= v["mean"] <= v["max"] + 1e-9, f"e18 {m} {sub} {k}: mean outside [min, max]")
                chk("R6", v["max"] > v["min"], f"e18 {m} {sub} {k}: five random head sets coincide")
    for path, d in _seedlists(P):
        s = d["seeds"]
        chk("R6", min(s) - 1e-9 <= d["mean"] <= max(s) + 1e-9, f"{'/'.join(path)}: mean outside the seed range")

    # ---------------------------------------------------------------- R7 common sense
    S = P["synthetic"]["cells"]
    for k, v in S.items():
        if k.endswith("|id") and k[:-3] + "|ood" in S:
            o = S[k[:-3] + "|ood"]
            if v["mean"] > 0.56 or o["mean"] > 0.56:
                chk("R7a", o["mean"] <= v["mean"] + 0.01, f"E3 {k[:-3]}: larger graphs {o['mean']:.4f} > in distribution {v['mean']:.4f}")
    E15 = P.get("e15", {})
    sizes = sorted({int(k.split("|")[1]) for k in E15 if k.count("|") == 2})
    for f in F4:
        for N in sizes:
            i, o = E15.get(f"{f}|{N}|id"), E15.get(f"{f}|{N}|ood")
            if i and o:
                chk("R7a", o["mean"] <= i["mean"] + 0.01, f"E15 {f} {N}: larger graphs {o['mean']} > in distribution {i['mean']}")
        for sp in ("id", "ood"):
            ms = [E15[f"{f}|{N}|{sp}"]["mean"] for N in sizes if f"{f}|{N}|{sp}" in E15]
            for a, b in zip(ms, ms[1:]):
                chk("R7b", b >= a - 0.01, f"E15 {f} {sp}: more data is worse ({a} -> {b})")
        # f) n95 from the displayed means
        ms = [(N, E15[f"{f}|{N}|id"]["mean"]) for N in sizes if f"{f}|{N}|id" in E15]
        n95 = None
        for (a, ya), (b, yb) in zip(ms, ms[1:]):
            if ya < 0.95 <= yb:
                n95 = math.exp(math.log(a) + (0.95 - ya) / (yb - ya) * (math.log(b) - math.log(a)))
                break
        if n95 is not None and isinstance(E15.get(f"{f}|n95"), (int, float)):
            chk("R7f", abs(E15[f"{f}|n95"] - n95) <= 0.01 * n95, f"E15 {f}: n95 {E15[f'{f}|n95']} != interpolated {n95:.0f}")
    for blk in ("e11a", "e11a_llama", "e11a_gemma"):
        B = P.get(blk)
        if not B:
            continue
        for f in ("AL", "NC"):
            b0 = B["none"][f"{f}|D1a"]["ba"]
            d4, d16 = b0 - B["top"][f"4|{f}|D1a"]["ba"], b0 - B["top"][f"16|{f}|D1a"]["ba"]
            chk("R7c", d16 >= d4 - 0.01, f"{blk} {f}: 16 heads cost {d16:.4f} < 4 heads {d4:.4f}")
            d8 = b0 - B["top"][f"8|{f}|D1a"]["ba"]
            chk("R7c", d4 - 0.005 <= d8 <= d16 + 0.005, f"{blk} {f}: nested doses 4, 8, 16 cost {d4:.4f}, {d8:.4f}, {d16:.4f}")
            chk("R7c", d16 > b0 - B["random"][f"16|{f}|D1a"]["mean"], f"{blk} {f}: selected heads no worse than random")
    _e17b = {"Qwen2.5-7B": "e11a", "Llama-3.1-8B": "e11a_llama", "Gemma-2-9B": "e11a_gemma"}
    for m, B in _d(P.get("e17", {})):
        A = P.get(_e17b.get(m, ""), {})
        if not A:
            continue
        for f in ("EL", "AL", "NC", "ECL"):
            for q in ("D1a", "Q0"):
                b0 = A["none"][f"{f}|{q}"]["ba"]
                ds = [b0 - A["top"][f"16|{f}|{q}"]["ba"]] + [b0 - B["top"][f"{h}|{f}|{q}"]["ba"] for h in (32, 64, 128)]
                chk("R7c", all(y >= x - 0.015 for x, y in zip(ds, ds[1:])), f"e17 {m} {f} {q}: a larger dose is milder {ds}")
                rs = [b0 - B["random"][f"{h}|{f}|{q}"]["mean"] for h in (32, 64, 128)]
                chk("R7c", all(y >= x - 0.015 for x, y in zip(rs, rs[1:])), f"e17 {m} {f} {q}: more random heads milder {rs}")
            d128 = A["none"][f"{f}|D1a"]["ba"] - B["top"][f"128|{f}|D1a"]["ba"]
            if f in ("AL", "NC"):
                chk("R7c", d128 > A["none"][f"{f}|D1a"]["ba"] - B["random"][f"128|{f}|D1a"]["mean"],
                    f"e17 {m} {f}: selected heads no worse than random")
    for m, B in _d(P.get("e18", {})):
        A = P.get(_e17b.get(m, ""), {})
        if not A:
            continue
        for f in ("EL", "AL", "NC", "ECL"):
            for q in ("D1a", "Q0"):
                b0 = A["none"][f"{f}|{q}"]["ba"]
                for sub in ("edge", "entry"):
                    ds = [b0 - B[sub][f"{h}|{f}|{q}"]["ba"] for h in (16, 128)]
                    chk("R7c", ds[1] >= ds[0] - 0.015, f"e18 {m} {sub} {f} {q}: 128 heads milder than 16 {ds}")
        d_el = A["none"]["EL|D1a"]["ba"] - B["edge"]["128|EL|D1a"]["ba"]
        chk("R7c", d_el > A["none"]["EL|D1a"]["ba"] - B["edge_random"]["128|EL|D1a"]["mean"], f"e18 {m} EL: selected edge heads no worse than random")
        for f in ("EL", "AL", "NC", "ECL"):
            b0 = P["e13"][m][f"degree|{f}"]["ba"]
            chk("R7c", B["degree"][f"128|{f}"]["ba"] <= b0 + 0.005 and B["degree_random"][f"128|{f}"]["mean"] <= b0 + 0.005,
                f"e18 {m} degree {f}: ablation helps")
    for m, row in _d(P.get("e14", {})):
        if isinstance(row, dict) and m != "words_none":
            for f in ("EL", "ECL"):
                for pos in ("between", "before"):
                    if f"{f}|{pos}|1024" in row:
                        chk("R7d", row[f"{f}|{pos}|1024"]["ba"] <= row[f"{f}|none|0"]["ba"] + 0.01,
                            f"E14 {m} {f} {pos}: 1,024 filler words help")
    for blk in ("llm_main", "llm_formats", "llm_partial", "mask_controls", "nl_formats", "e11a", "e11a_llama",
                "e11a_gemma", "e11b", "e11bL", "e13", "e14", "e16", "e11c", "reasoning", "graphml", "real_graphs",
                "robustness", "prompting", "api_model", "llm_blind", "e17", "e18", "e19", "e20", "e21"):
        for path, v in _cells(P.get(blk, {})):
            if blk == "e18" and "degree" in path:
                continue            # exact-answer accuracy (GraphQA degree)
            if blk == "e13" and not path[-1].startswith("edge|"):
                continue            # degree and neighbors: exact-answer accuracy, whose chance level is not 0.5
            chk("R7e", v["hi"] >= 0.49, f"{blk} {'/'.join(path)}: clearly below chance (hi {v['hi']})")

    # g) the probe's reliability falls with distance (approximate since; 60 pairs per cell, so single steps are
    #    noisy): mean over k in {1, 2} >= mean over k in {16, 24}
    g12 = {}
    for c in P.get("e12", {}).get("cells", []):
        g12.setdefault((c["model"], c["ser"]), {})[c["k"]] = c["bal_acc"]
    for (m, s), v in g12.items():
        if all(k in v for k in (1, 2, 16, 24)):
            chk("R7g", (v[1] + v[2]) / 2 >= (v[16] + v[24]) / 2, f"E12 {m} {s}: reliability rises with distance")
    # h) harder questions are not easier: depth two and counting <= depth one + 0.02 (class cells, every model and
    #    format); "v is red" (one step) >= its neighbor question - 0.01 without masking
    for m in MODELS:
        c = P["llm_main"]["models"][m]
        for f in ("EL", "AL", "NC"):
            for hard in ("depth2", "count"):
                if hard in c and "depth1" in c:
                    chk("R7h", c[hard][f]["ba"] <= c["depth1"][f]["ba"] + 0.02, f"E1 {m} {f}: {hard} easier than depth one")
        for f in ("AL", "NC"):
            r = P["llm_partial"].get(m, {}).get(f"{f}-none")
            if r:
                chk("R7h", r["Q0"]["ba"] >= r["D1a"]["ba"] - 0.01, f"E7 {m} {f}: 'v is red' harder than its neighbor question")
    # i) a longer filler is never better than a shorter one by more than 0.01 (E14, both positions)
    for m, row in _d(P.get("e14", {})):
        if m == "words_none":
            continue
        for f in ("EL", "ECL"):
            for pos in ("between", "before"):
                seq = [row[f"{f}|none|0"]["ba"]] + [row[f"{f}|{pos}|{F}"]["ba"] for F in (64, 256, 1024) if f"{f}|{pos}|{F}" in row]
                for a, b in zip(seq, seq[1:]):
                    chk("R7i", b <= a + 0.01, f"E14 {m} {f} {pos}: a longer filler helps ({a} -> {b})")
    # j) text lengths follow the format definitions: NC > AL > EL and ECL > EL (same graphs)
    W = P.get("token_cost", {}).get("word_tokens", {})
    for grp in sorted({k.split("|")[0] for k in W}):
        w = {k.split("|")[1]: v["mean"] for k, v in W.items() if k.split("|")[0] == grp}
        if all(f in w for f in F4):
            chk("R7j", w["NC"] > w["AL"] > w["EL"] and w["ECL"] > w["EL"], f"word tokens ({grp}) out of the format order: {w}")
    for m, row in _d(P.get("e11bL", {})):
        for D in (6, 12, 24, 48):
            w = {f: row[f"{f}|{D}"].get("words") for f in F4 if f"{f}|{D}" in row}
            if len(w) == 4 and None not in w.values():
                chk("R7j", w["NC"] > w["AL"] > w["EL"] and w["ECL"] > w["EL"], f"E11b-L {m} {D}: words out of the format order")

    # ---------------------------------------------------------------- R9 same prompts, same cell
    _e11a = {"Qwen2.5-7B": "e11a", "Llama-3.1-8B": "e11a_llama", "Gemma-2-9B": "e11a_gemma"}
    for m, row in _d(P.get("e14", {})):
        base = P.get(_e11a.get(m, ""), {}).get("none", {})
        for f in ("EL", "ECL"):
            a, b = row.get(f"{f}|none|0"), base.get(f"{f}|D1a")
            if isinstance(a, dict) and isinstance(b, dict):
                chk("R9", all(abs(a[k] - b[k]) <= 5e-4 for k in ("ba", "lo", "hi")),
                    f"E14 {m} {f} unfilled {a} differs from the unablated E11a cell {b} (same prompts)")
    for m, row in _d(P.get("e11bL", {})):
        for f in F4:
            a, b = row.get(f"{f}|48"), P.get("e11b", {}).get(m, {}).get(f"{f}|48")
            if isinstance(a, dict) and isinstance(b, dict):
                chk("R9", all(abs(a[k] - b[k]) <= 5e-4 for k in ("ba", "lo", "hi")),
                    f"E11b-L {m} {f} at Delta = 48 differs from E11b (unpadded, same prompts)")

    # ---------------------------------------------------------------- R10 one evaluation, two summaries
    # the Delta <= 256 cell of E3 is the pair-weighted mean of its four degree strata (245 to 255 pairs each), seed by
    # seed, so its displayed mean is within 0.002 of the plain mean of the strata means
    X_ = P.get("e3_extra", {})
    for k, v in X_.get("d256", {}).items():
        parts = [X_["d256_by"].get(f"{k}|{D}") for D in (64, 128, 192, 256)]
        if all(isinstance(q, dict) for q in parts):
            chk("R10", abs(v["mean"] - sum(q["mean"] for q in parts) / 4) <= 0.002, f"E3 Delta <= 256 {k}: {v['mean']:.4f} vs strata")
    # ---------------------------------------------------------------- R11 separate runs are not copies
    # separately trained models or separate runs never give identical summaries: the learning-rate check differs
    # across learning rates, and the M-like head counts differ between AL and NC in at least one arm
    lr_ = X_.get("lr_check", {})
    for arm in ("NoPE", "RoPE"):
        for sp in ("id", "ood"):
            vs = [lr_.get(f"{arm}|{lr}", {}).get(sp) for lr in ("1e-4", "3e-4", "1e-3")]
            if None not in vs:
                chk("R11", len({round(x, 3) for x in vs}) >= 2, f"E3 lr check {arm} {sp}: identical at every learning rate")
    M_ = P.get("e3_match", {})
    arms_ = {k.split("|")[0] for k in M_ if "|" in k}
    if arms_:
        chk("R11", any(M_.get(f"{a}|AL", {}).get("n_mlike") != M_.get(f"{a}|NC", {}).get("n_mlike") for a in arms_),
            "E3 M-like head counts identical under AL and NC in every arm")

    # ---------------------------------------------------------------- R8 p-values
    for path, d in _pdicts(P):
        p = d["p"]
        chk("R8", 0.0 <= p <= 1.0, f"{'/'.join(path)}: p = {p} outside [0, 1]")
        if "p_holm" in d:
            chk("R8", p - 1e-12 <= d["p_holm"] <= 1.0 + 1e-12, f"{'/'.join(path)}: Holm p {d['p_holm']} < raw p {p}")
    for fam in _holm_families(P):
        fam = sorted(fam, key=lambda d: d["p"])
        for a, b in zip(fam, fam[1:]):
            chk("R8", b["p_holm"] >= a["p_holm"] - 1e-12, "Holm-adjusted p-values not monotone in the raw p-values")


def _seedlists(P, path=()):
    if isinstance(P, dict):
        if isinstance(P.get("seeds"), list) and len(P["seeds"]) >= 2 and isinstance(P.get("mean"), (int, float)) \
                and all(isinstance(x, (int, float)) for x in P["seeds"]):
            yield path, P
        for k, v in P.items():
            yield from _seedlists(v, path + (k,))


def _pdicts(P, path=()):
    if isinstance(P, dict):
        if isinstance(P.get("p"), (int, float)):
            yield path, P
        for k, v in P.items():
            yield from _pdicts(v, path + (k,))
    elif isinstance(P, list):
        for i, v in enumerate(P):
            yield from _pdicts(v, path + (str(i),))


def _holm_families(P):
    """Lists of dicts that all carry p and p_holm (one Holm family each)."""
    out = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            fam = [d for d in x if isinstance(d, dict) and isinstance(d.get("p"), (int, float)) and "p_holm" in d]
            if len(fam) >= 2:
                out.append(fam)
            for v in x:
                walk(v)
    walk(P)
    return out
