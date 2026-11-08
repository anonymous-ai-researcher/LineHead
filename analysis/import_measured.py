"""Import the experiment summaries into results/summary.json.

Each experiment script writes a summary file (formats in RUN_PROTOCOL.md §5). This script reads them and stores
their values in results/summary.json, a nested dict with one block per experiment whose structure is fixed by
analysis/schema.json. Every imported path is checked against the schema: an unknown key (for example a misspelt
variant tag) stops the import, so nothing lands where the paper does not read it.

The full command (one option per summary file, in this order) is in RUN_PROTOCOL.md §7.
Re-running adds to the existing summary.json (later imports of the same cell win).
  python analysis/import_measured.py --e1 m0/results/summary_e1.json --e1-strata ...      (see RUN_PROTOCOL §7)
  python analysis/import_measured.py --reset ...      start from an empty summary.json
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent
PATH = DATA / "schema.json"             # structure of the summary (read only)
MEASURED = DATA.parent / "results" / "summary.json"       # output
MODELS = ["Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B", "Qwen2.5-72B"]
ROB_MODELS = ["Qwen2.5-7B", "Llama-3.1-8B"]
F3 = ["EL", "AL", "NC"]
# k_min strata of the paper (CONFIG §5.5); the analysis scripts must use exactly these bins
KMAP = {"1": "1", "k1": "1", "2": "2", "k2": "2", "3-4": "3-4", "k3-4": "3-4", ">=5": ">=5", "k5+": ">=5", "5+": ">=5"}
DMAP = {"le2": "<=2", "3to7": "3-7", "ge8": ">=8"}
E3K = ["1", "2", "3", "4-6", "7-12", "13-32"]
ARM = {"nope": "NoPE", "local": "1-local", "1-local": "1-local", "alibi_only": "ALiBi-only", "alibi-only": "ALiBi-only",
       "local_alibi": "1-local+ALiBi", "local+alibi": "1-local+ALiBi", "1-local+alibi": "1-local+ALiBi",
       "local_hard": "1-local+hard-since", "hard": "1-local+hard-since", "1-local+hard-since": "1-local+hard-since",
       "local_hardsince": "1-local+hard-since", "rope": "RoPE"}
FMT = {"el": "EL", "al": "AL", "nc": "NC", "ecl": "ECL", "el2": "EL-2", "el-2": "EL-2", "elfar": "EL-far",
       "el-far": "EL-far", "nch": "NC-H", "nc-h": "NC-H", "graphml": "GraphML", "gml": "GraphML",
       "ncnw": "NC-nw", "nc-nw": "NC-nw", "incident": "Incident", "friendship": "Friendship", "elb": "ELB"}
NL = ("NC-nw", "Incident", "Friendship")          # their cells live in the block nl_formats
SRC = {"molhiv": "molecules", "molecules": "molecules", "arxiv": "citation", "citation": "citation"}
VARIANTS = ["default", "int_ids", "sorted_el", "question_first", "three_colors"]      # E8 tags = schema keys
WARN = []


def tree():
    return defaultdict(tree)


def plain(d):
    return {k: plain(v) for k, v in d.items()} if isinstance(d, dict) else d


def r4(x):
    return None if x is None else round(float(x), 4) + 0.0          # + 0.0: no negative zero


def model_key(name):
    s = str(name).split("/")[-1].lower()
    for m in MODELS:
        if s.startswith(m.lower()):
            return m
    if "gpt-4o" in s:
        return "GPT-4o"
    raise SystemExit(f"unknown model name {name!r}")


def fmt_key(s):
    k = re.sub(r"[^a-z0-9\-]", "", str(s).lower())
    if k not in FMT:
        raise SystemExit(f"unknown serialization {s!r}")
    return FMT[k]


def cellv(c):
    return {"ba": r4(c["bal_acc"]), "lo": r4(c["ci_low"]), "hi": r4(c["ci_high"])}


def cells_of(path):
    S = json.load(open(path))
    return S["cells"] if isinstance(S, dict) else S


def strata_of(c):
    """k_min strata (TPR of the stratum's positives with the TNR of all negatives), maximum-degree strata, and the
    measured share of positives per k_min stratum (unrounded, so that it can be compared with m0/results/kmin_shares.json)."""
    out = {"kmin": {}, "maxdeg": {}, "kmin_share": {}}
    ks = [s for s in c.get("strata", []) if s["stratum"] in KMAP]
    tot = sum(s.get("n_pairs", 0) for s in ks)
    for s in c.get("strata", []):
        lab = s["stratum"]
        if lab in KMAP:
            out["kmin"][KMAP[lab]] = r4(s["bal_acc"]) if s.get("n_pairs", 0) else None
            out["kmin_share"][KMAP[lab]] = s.get("n_pairs", 0) / tot if tot else None
        elif lab in DMAP:
            out["maxdeg"][DMAP[lab]] = r4(s["bal_acc"])
        else:
            WARN.append(f"stratum {lab!r} is not a paper bin (k: 1, 2, 3-4, >=5; maxdeg: le2, 3to7, ge8)")
    for k in ("1", "2", "3-4", ">=5"):                 # an empty stratum is reported as missing, not dropped
        out["kmin"].setdefault(k, None)
        out["kmin_share"].setdefault(k, 0.0 if tot else None)
    return out


def last_wins(rows, key, what):
    """Deduplicate runs by configuration (rerun after a crash): the last line wins, with a warning."""
    out = {}
    for r in rows:
        k = key(r)
        if k in out:
            WARN.append(f"{what}: duplicate run {k}; the last line is used")
        out[k] = r
    return list(out.values())


# ------------------------------------------------------------------ LLM experiments
def imp_e1(M, path):
    S = json.load(open(path))
    L = M["llm_main"]
    for name, s in S.items():
        if name.startswith("_"):
            continue
        m = model_key(name)
        for c in s["cells"]:
            L["models"][m][c["cls"]][fmt_key(c["ser"])] = cellv(c)
        for c in s.get("per_question", []):
            L["per_question"][m][c["pid"]][fmt_key(c["ser"])] = r4(c["bal_acc"])
        for k_in, k_out in (("interaction", "plain"), ("interaction_length_adjusted", "length"), ("interaction_qfe", "qfe")):
            if k_in in s:
                t = s[k_in]
                L["tests"][m][k_out] = {"chi2": round(float(t["wald_chi2"]), 1), "df": int(t["df"]), "p": float(t["p"])}
        if s.get("pairwise"):
            L["pairwise"][m] = [dict(cls=x["cls"], a=fmt_key(x["a"]), b=fmt_key(x["b"]), diff=r4(x["mean_diff"]),
                                     p=float(x["p"]), p_holm=float(x["p_holm"])) for x in s["pairwise"]]
        ren = {"logd": "logd", "k_min": "k", "loglen": "loglen", "n": "n", "deg_v": "degv"}
        for f, d_ in (s.get("distance_regression") or {}).items():
            if "error" in d_:
                WARN.append(f"E1 {m} distance regression {f}: {d_['error']}")
                continue
            if fmt_key(f) not in F3:
                continue
            for k, v in d_.items():
                if k in ren:
                    L["distance_regression"][m][fmt_key(f)][ren[k]] = {"coef": round(v["coef"], 3),
                                                                      "se": round(v["se"], 3), "p": v["p"]}
        for c in s.get("planned_contrasts", []):
            L["contrasts"][m][c["cls"]][f"{c['format']}_EL"] = {
                "est": r4(c["delta"]), "lo": r4(c["ci_low"]), "hi": r4(c["ci_high"]), "se": round(float(c["se"]), 5),
                "p": float(c["p_one_sided"]), "p_boot": float(c["p_boot"])}
        if m in ROB_MODELS and s.get("per_template"):
            pt = defaultdict(dict)
            for c in s["per_template"]:
                if c["cls"] == "depth1":
                    pt[fmt_key(c["ser"])][c["tpl"]] = r4(c["bal_acc"])
            for f in F3:
                if f in pt:
                    M["robustness"][m]["per_template"][f] = [pt[f][t] for t in sorted(pt[f])]
    if "_pooled" in S:
        p = S["_pooled"]
        L["pseudo_r2"] = {"interaction_increment": r4(p["gain_interaction"]), "base": r4(p["pseudo_r2_base"])}


def imp_strata(M, path, key, formats=None):
    """k_min strata from e4_stats.py --by model,ser --pids D1a,D1c (key strata) or --pids D1a (strata_d1a).
    E1 formats and ECL go to llm_main.<key>; the E4 ablations to llm_formats.<model>.<format>.strata (D1a+D1c);
    GraphML to graphml.by_model.<model>.<key> (and the Qwen2.5-7B headline graphml.<key>)."""
    for c in cells_of(path):
        m, f = model_key(c["model"]), fmt_key(c["ser"])
        if formats and f not in formats:
            continue
        if not c.get("strata"):
            continue
        sv = strata_of(c)
        if f in F3 or f == "ECL":
            M["llm_main"][key][m][f] = sv
        elif f == "GraphML":
            M["graphml"]["by_model"][m][key] = sv
            if m == "Qwen2.5-7B" and key == "strata":
                M["graphml"]["strata"] = sv
        elif key == "strata" and f not in NL:          # formats: no strata displayed
            M["llm_formats"][m][f]["strata"] = sv
        # the ablations' D1a-only strata are not displayed (E5 predicts from E1 formats, ECL and GraphML)


def imp_meanci(M, path, e7=False):
    """joint_stats.py mean-ci: four-model means with joint pair-bootstrap intervals (Figures 1b, 4a)."""
    for c in json.load(open(path)):
        v = {"ba": r4(c["mean_bal_acc"]), "lo": r4(c["ci_low"]), "hi": r4(c["ci_high"])}
        f = fmt_key(c["ser"])
        if e7:
            if str(c.get("mask", "none")) in ("rand_heads", "rand_entries"):
                continue                     # random-mask controls are not in Figure 4a
            mask = {"none": "none", "heads": "head", "head": "head", "entries": "entry", "entry": "entry"}[str(c.get("mask", "none"))]
            if f == "NC-H" and mask == "none" and c["pid"] == "D1a":
                M["llm_main"]["partial_mean_ci"]["NC-H"]["D1a"] = v
            elif f in ("AL", "NC"):
                M["llm_main"]["partial_mean_ci"][f"{f}-{mask}"][c["pid"]] = v
        else:
            M["llm_main"]["mean_ci"][c["cls"]][f] = v


def imp_e2(M, path, controls=False):
    """E2 cells: query file summary (--by model,ser,cls) or control file summary (--by model,ser,cls,pid)."""
    for c in cells_of(path):
        m, f = model_key(c["model"]), fmt_key(c["ser"])
        B = M["llm_blind"][m]
        if controls:
            if c.get("pid") not in ("X1", "X2"):
                raise SystemExit(f"E2 controls: cell without pid X1/X2 in {path}; run e4_stats.py with --by model,ser,cls,pid")
            B[c["pid"]][f] = cellv(c)
        elif c["cls"] in ("depth1", "depth2", "count"):
            B["query"][c["cls"]][f] = cellv(c)
        elif c["cls"] == "global":
            B["global"][f] = cellv(c)
        else:
            raise SystemExit(f"E2: unrecognised class {c['cls']!r} in {path}")


def imp_e2_holm(M, path):
    S = json.load(open(path))
    M["llm_main"]["blind_holm"] = {"n_cells": S["n_cells"], "n_off_chance": S["n_off_chance"],
                                   "controls": S.get("controls"),
                                   "query_cells_exactly_half": S.get("query_cells_exactly_half")}


def imp_e4(M, path):
    """summary_e4.json: depth-one cells of the ablations (EL, AL, NC rows are E1 and skipped) and the seven
    preregistered comparisons (Holm within model). Strata come from --e4-strata only (D1a+D1c)."""
    S = json.load(open(path))
    for c in S["cells"]:
        m, f = model_key(c["model"]), fmt_key(c["ser"])
        if f in F3:
            continue
        (M["nl_formats"] if f in NL else M["llm_formats"])[m][f]["depth1"] = cellv(c)
    tests = defaultdict(list)
    for t in S.get("tests", []):
        tests[model_key(t["model"])].append(dict(a=fmt_key(t["a"]), b=fmt_key(t["b"]), diff=r4(t["mean_diff"]),
                                                 p=float(t["p"]), p_holm=float(t["p_holm"])))
    for m, v in tests.items():
        M["llm_main"]["e4_tests"][m] = v


def imp_e4_ecl(M, path):
    for c in cells_of(path):
        M["llm_formats"][model_key(c["model"])]["ECL"][c["cls"]] = cellv(c)


def imp_e7(M, path):
    """summary_e7.json (--by model,ser,mask,pid; E1 records are the unmasked D1a condition, E4 records give NC-H)."""
    for c in cells_of(path):
        m, f = model_key(c["model"]), fmt_key(c["ser"])
        mk = str(c.get("mask", "none"))
        if mk in ("rand_heads", "rand_entries"):          # random masks of the targeted masks' size
            M["mask_controls"][m][f"{f}-{mk}"][c["pid"]] = cellv(c)
            continue
        mask = {"none": "none", "heads": "head", "head": "head", "entries": "entry", "entry": "entry"}[mk]
        if f == "NC-H" and mask == "none":
            M["llm_formats"][m]["NC-H"][c["pid"]] = cellv(c)
        elif f in ("AL", "NC"):
            M["llm_partial"][m][f"{f}-{mask}"][c["pid"]] = cellv(c)


def imp_e7_twin(M, path):
    """summary_e7_twin.json: twin Q0 pairs, AL, masks none / heads / entries."""
    for c in cells_of(path):
        mask = {"none": "none", "heads": "head", "entries": "entry"}[str(c.get("mask", "none"))]
        M["mask_controls"][model_key(c["model"])]["twin"][f"AL-{mask}"] = cellv(c)


def imp_r9(M, folder):
    """Statistics on the LLM records (r9_stats.py, dmin_tokens.py, compete.py fit, e4_stats fallback)."""
    d = Path(folder)
    S = M["llm_stats"]
    if (d / "summary_paired.json").exists():
        for c in cells_of(d / "summary_paired.json"):
            f = fmt_key(c["ser"])
            if f in F3 + ["ECL"] and c["cls"] in ("global", "depth1"):
                S["paired"][model_key(c["model"])][f"{f}|{c['cls']}"] = {"auc": r4(c["auc"]), "tpr": r4(c["tpr"]),
                                                                         "tnr": r4(c["tnr"])}
    if (d / "summary_position.json").exists():
        for m_, v in json.load(open(d / "summary_position.json"))["models"].items():
            S["position"][model_key(m_)] = {"base": {k: r4(x) for k, x in v["base"].items() if k in ("NC_EL", "AL_EL")},
                                            "adjusted": {k: r4(x) for k, x in v["adjusted"].items()
                                                         if k in ("NC_EL", "AL_EL", "rel", "pos_chi2", "pos_p")}}
    if (d / "summary_qre.json").exists():
        for m_, v in json.load(open(d / "summary_qre.json"))["models"].items():
            if "error" not in v:
                S["qre"][model_key(m_)] = {"wald_chi2": round(v["wald_chi2"], 1), "df": v["df"], "p": v["p"]}
    if (d / "summary_lengths.json").exists():
        for c in cells_of(d / "summary_lengths.json"):
            ser = {"EL-2": "EL2", "EL-far": "ELfar", "NC-H": "NCH", "NC-nw": "NCnw"}.get(fmt_key(c["ser"]), fmt_key(c["ser"]))
            if ser == "GraphML":                         # two models only; not in the lengths table
                continue
            S["lengths"][model_key(c["model"])][ser] = {"mean": round(c["mean"], 1), "max": int(c["max"])}
    if (d / "summary_kdiff.json").exists():            # V-C8 paired test
        k = json.load(open(d / "summary_kdiff.json"))
        S["kdiff"] = {x: r4(k[x]) for x in ("nc_minus_el", "ecl_minus_el_abs", "stat", "lo90", "hi90", "p_one_sided",
                                            "nc_minus_ecl", "nc_minus_ecl_lo90") if x in k}
        S["kdiff"]["by_model"] = {model_key(m_): {f: r4(v) for f, v in fv.items()} for m_, fv in k["by_model"].items()}
    if (d / "summary_p4ni.json").exists():             # P4 as non-inferiority per k_min stratum
        S["p4_noninf"] = {s_: {"shortfall": r4(v["shortfall"]), "hi95": r4(v["hi95"])}
                          for s_, v in json.load(open(d / "summary_p4ni.json")).items()}
    if (d / "summary_tost.json").exists():
        S["tost"] = [dict(t, model=model_key(t["model"])) for t in json.load(open(d / "summary_tost.json"))["tests"]]
    if (d / "dmin_tokens.json").exists():
        for tk, v in json.load(open(d / "dmin_tokens.json")).items():
            if model_key(tk):
                S["dmin_tok"][model_key(tk)] = {f: {"spearman": r4(x["spearman"]), "determined": r4(x["determined"])}
                                               for f, x in v.items()}
    if (d / "summary_e10.json").exists():
        fb = [c["fallback"] for c in cells_of(d / "summary_e10.json") if "fallback" in c]
        if fb:
            S["api_fallback"] = r4(st.mean(fb))
    for f in ("AL", "NC"):
        pr = d / f"e9_probe_{f}.json"
        if pr.exists():
            pr = json.load(open(pr))
            best = max(pr, key=lambda L_: pr[L_].get("all", 0))
            if "control" in pr[best]:
                S["probe_control"][f] = {"control": r4(pr[best]["control"]), "selectivity": r4(pr[best]["selectivity"])}
    if (d / "summary_compete.json").exists():
        c = json.load(open(d / "summary_compete.json"))
        for k in ("locality", "route"):
            M["competing"][k] = {x: c[k][x] for x in ("aic", "qaic", "heldout_ll", "heldout_rmse_pts", "k",
                                                      "divergent_rmse_pts", "nondivergent_rmse_pts", "free_rmse_pts")}
        M["competing"]["rmse_diff_ci"] = c["rmse_diff_ci"]
        for x in ("delta_aic", "delta_qaic", "delta_heldout_ll", "delta_rmse_pts", "delta_divergent_rmse_pts",
                  "delta_nondivergent_rmse_pts", "n_cells", "n_divergent", "dispersion", "n_free",
                  "delta_free_rmse_pts"):
            M["competing"][x] = c[x]


def imp_r9_e3(M, path):
    """summary_r9_e3.json (r9_e3.py): learning curves, line-length regression, ID strata, own-line set, lr check, NC-H."""
    R = json.load(open(path))
    X = M["e3_extra"]
    if R.get("curves", {}).get("by_format"):
        X["curves"] = R["curves"]
    for k, v in R.get("kreg", {}).items():
        X["kreg"][k] = v
    for k, row in R.get("kid", {}).items():
        for lab, v in row.items():
            if v:
                X["kid"][k][lab] = v
    for L_, v in R.get("kown", {}).items():
        X["kown"][L_] = v
    for k, v in R.get("lr_check", {}).items():
        X["lr_check"][k] = {"id": r4(v["id"]), "ood": r4(v["ood"])}
    for k, v in R.get("nch", {}).items():
        X["nch"][k] = {sp: x for sp, x in v.items() if x}
    for k, v in R.get("d256", {}).items():           # Delta <= 256 test set
        X["d256"][k] = v
    for k, v in R.get("d256_by", {}).items():        # by hub degree
        if v is not None:
            X["d256_by"][k] = v
    for k, v in R.get("hard", {}).items():           # hard pairs of Theorem lb:main
        X["hard"][k] = v
    for k, v in R.get("hardp", {}).items():          # instances of the proof's construction
        X["hardp"][k] = v


R12_NAMES = {"bloom-7b1": "BLOOM-7B1", "mpt-7b": "MPT-7B", "deepseek-r1-distill-qwen-7b": "R1-Distill-Qwen-7B"}


def r12_model(name, pool=False):
    s = str(name).split("/")[-1].lower()
    for k, v in R12_NAMES.items():
        if s.startswith(k):
            return v
    m = model_key(name)
    return f"{m} (E11c pool)" if pool else m


def imp_r12(M, folder):
    """Summaries (RUN_PROTOCOL §4.13): summary_e11a.json (e11_heads.py), summary_e11b.json (e4_stats --by
    model,ser,maxdeg), summary_e11c.json and summary_reasoning.json (e4_stats --by model,ser), summary_e12.json and
    summary_e12_predict.json (e12_probe.py), summary_e13.json (e13_graphqa.py)."""
    d = Path(folder)
    # summary_e11a.json: Qwen2.5-7B (block e11a); summary_e11a_llama.json (block e11a_llama); both may hold
    # the patching conditions patch-<kind><h> and the pooled random patching (random_patch)
    for name, blk in (("summary_e11a.json", "e11a"), ("summary_e11a_llama.json", "e11a_llama")):
        if not (d / name).exists():
            continue
        S = json.load(open(d / name))
        for c in S["cells"]:
            f, q, cond = fmt_key(c["ser"]), c["pid"], c["cond"]
            if cond == "none":
                M[blk]["none"][f"{f}|{q}"] = cellv(c)
            elif cond.startswith("top"):
                M[blk]["top"][f"{cond[3:]}|{f}|{q}"] = cellv(c)
            elif cond.startswith("patch-"):
                kind = cond[6:].rstrip("0123456789")
                M[blk]["patch"][f"{kind}|{cond[6 + len(kind):]}|{f}|{q}"] = cellv(c)
        for c in S["random"]:
            M[blk]["random"][f"{c['h']}|{fmt_key(c['ser'])}|{c['pid']}"] = {"mean": r4(c["mean"]), "min": r4(c["min"]),
                                                                            "max": r4(c["max"])}
        for c in S.get("random_patch", []):
            M[blk]["random_patch"][f"{c['kind']}|16|{fmt_key(c['ser'])}|{c['pid']}"] = {"mean": r4(c["mean"]),
                                                                                       "min": r4(c["min"]), "max": r4(c["max"])}
    if (d / "summary_match.json").exists():        # E3 name matching (e3/match_heads.py summary)
        for k, c in json.load(open(d / "summary_match.json"))["error"].items():
            M["e3_match"][k] = {b: ({"mean": r4(v["mean"]), "sd": r4(v["sd"]), "seeds": [r4(x) for x in v["seeds"]]}
                                    if isinstance(v, dict) else r4(v)) for b, v in c.items()}
    if (d / "summary_e11b.json").exists():
        for c in cells_of(d / "summary_e11b.json"):
            m, f, D = model_key(c["model"]), fmt_key(c["ser"]), int(c["maxdeg"])
            M["e11b"][m][f"{f}|{D}"] = cellv(c)
        for m, row in M["e11b"].items():
            for f in ("EL", "AL", "NC", "ECL"):
                if f"{f}|6" in row and f"{f}|48" in row:
                    row[f"{f}|decline"] = r4(row[f"{f}|6"]["ba"] - row[f"{f}|48"]["ba"])
    for name, blk in (("summary_e11c.json", "e11c"), ("summary_reasoning.json", "reasoning")):
        if (d / name).exists():
            for c in cells_of(d / name):
                M[blk][r12_model(c["model"], pool=blk == "e11c")][fmt_key(c["ser"])] = cellv(c)
    if (d / "summary_e12.json").exists():
        M["e12"]["cells"] = [dict(c, model=model_key(c["model"])) for c in cells_of(d / "summary_e12.json")]
    if (d / "summary_e12_predict.json").exists():
        R = json.load(open(d / "summary_e12_predict.json"))
        for c in R["cells"]:
            M["e12"]["predict"][f"{model_key(c['model'])}|{c['ser']}|{c['stratum']}"] = {"pred": r4(c["pred"]),
                                                                                        "measured": r4(c["measured"])}
        M["e12"]["error_pts"], M["e12"]["max_error_pts"] = R["error_pts"], R["max_error_pts"]
    if (d / "summary_e13.json").exists():
        for c in cells_of(d / "summary_e13.json"):
            M["e13"][model_key(c["model"])][f"{c['task']}|{fmt_key(c['ser'])}"] = {"ba": r4(c["value"]), "lo": r4(c["ci_low"]),
                                                                                  "hi": r4(c["ci_high"])}


def imp_r14(M, folder):
    """Summaries (RUN_PROTOCOL §4.15): summary_e11a_gemma.json (e11_heads.py; block e11a_gemma),
    summary_e11bL.json (e4_stats --by model,ser,maxdeg), summary_e14.json (e4_stats --by model,ser,filler_cell),
    summary_e16.json (e4_stats --by model,ids,ser), summary_e15.json (e3/e15_summary.py); e11bL_words.json and
    e14_words.json (prompt lengths, m0/build_r14.py)."""
    d = Path(folder)
    if (d / "summary_e11a_gemma.json").exists():
        S = json.load(open(d / "summary_e11a_gemma.json"))
        for c in S["cells"]:
            f, q, cond = fmt_key(c["ser"]), c["pid"], c["cond"]
            if cond == "none":
                M["e11a_gemma"]["none"][f"{f}|{q}"] = cellv(c)
            elif cond.startswith("top"):
                M["e11a_gemma"]["top"][f"{cond[3:]}|{f}|{q}"] = cellv(c)
        for c in S["random"]:
            M["e11a_gemma"]["random"][f"{c['h']}|{fmt_key(c['ser'])}|{c['pid']}"] = {"mean": r4(c["mean"]), "min": r4(c["min"]),
                                                                                     "max": r4(c["max"])}
    if (d / "summary_e11bL.json").exists():
        for c in cells_of(d / "summary_e11bL.json"):
            m, f, D = model_key(c["model"]), fmt_key(c["ser"]), int(c["maxdeg"])
            M["e11bL"][m][f"{f}|{D}"] = cellv(c)
        if (d / "e11bL_words.json").exists():     # prompt lengths of the instance file (m0/build_r14.py)
            W = json.load(open(d / "e11bL_words.json"))
            for m, row in M["e11bL"].items():
                for k, c in row.items():
                    if isinstance(c, dict) and k in W:
                        c["words"] = W[k]
        for m, row in M["e11bL"].items():
            for f in ("EL", "AL", "NC", "ECL"):
                if f"{f}|6" in row and f"{f}|48" in row:
                    row[f"{f}|decline"] = r4(row[f"{f}|6"]["ba"] - row[f"{f}|48"]["ba"])
    if (d / "summary_e14.json").exists():
        for c in cells_of(d / "summary_e14.json"):
            m, f = model_key(c["model"]), fmt_key(c["ser"])
            pos, F = c["filler_cell"].split("|")
            M["e14"][m][f"{f}|{pos}|{F}"] = cellv(c)
        if (d / "e14_words.json").exists():       # prompt lengths without filler (m0/build_r14.py)
            M["e14"]["words_none"] = {f: v for f, v in json.load(open(d / "e14_words.json")).items()}
        for m, row in M["e14"].items():
            for f in ("EL", "ECL"):
                for F in (64, 256, 1024):
                    if f"{f}|between|{F}" in row and f"{f}|before|{F}" in row:
                        row[f"{f}|contrast|{F}"] = r4(row[f"{f}|between|{F}"]["ba"] - row[f"{f}|before|{F}"]["ba"])
    if (d / "summary_e16.json").exists():
        for c in cells_of(d / "summary_e16.json"):
            M["e16"][f"{model_key(c['model'])}|{c['ids']}|{fmt_key(c['ser'])}"] = cellv(c)
    if (d / "summary_e15.json").exists():
        for k, v in json.load(open(d / "summary_e15.json")).items():
            M["e15"][k] = v if not isinstance(v, dict) else {"seeds": [r4(x) for x in v["seeds"]], "mean": r4(v["mean"]),
                                                             "sd": r4(v["sd"])}
    # E16 one-token names: the E1 prompts themselves (mean of D1a and D1c); needs --e1
    pq = M["llm_main"]["per_question"] if "llm_main" in M and "per_question" in M["llm_main"] else {}
    for m in ("Qwen2.5-7B", "Llama-3.1-8B"):
        for f in ("EL", "AL", "NC"):
            if m in pq and f in pq[m].get("D1a", {}) and f in pq[m].get("D1c", {}):
                M["e16"][f"{m}|names|{f}"] = {"ba": r4(0.5 * (pq[m]["D1a"][f] + pq[m]["D1c"][f]))}


def imp_r16(M, folder):
    """Summary (CONFIG §6.11; RUN_PROTOCOL §4.16): summary_e17.json (e11_heads.py summary over the E11a and E17
    records of the three models); block e17: top<h> and the 5 random sets for h in 32, 64, 128, and the flip
    contrast at h = 128 (EL minus AL, minus NC, minus their mean; mean over pairs and one-sided 95% bound)."""
    d = Path(folder)
    if not (d / "summary_e17.json").exists():
        return
    S = json.load(open(d / "summary_e17.json"))
    for c in S["cells"]:
        h = c["cond"][3:] if c["cond"].startswith("top") else None
        if h in ("32", "64", "128"):
            M["e17"][model_key(c["model"])]["top"][f"{h}|{fmt_key(c['ser'])}|{c['pid']}"] = cellv(c)
    for c in S["random"]:
        if c["h"] in (32, 64, 128):
            M["e17"][model_key(c["model"])]["random"][f"{c['h']}|{fmt_key(c['ser'])}|{c['pid']}"] = {
                "mean": r4(c["mean"]), "min": r4(c["min"]), "max": r4(c["max"])}
    for c in S.get("flip", []):
        if c["h"] == 128 and c.get("kind", "top") == "top":
            M["e17"][model_key(c["model"])]["flip"][c["contrast"][3:]] = {"diff": r4(c["diff"]), "lo05": r4(c["lo05"])}


def imp_r18(M, folder):
    """Summary (CONFIG §6.12; RUN_PROTOCOL §4.17): summary_e18.json (e11_heads.py summary over the E11a and E18
    records of the three models); block e18: edge heads (etop16, etop128) and their 5 random sets, since heads at
    entry positions only (ntop16, ntop128) with the flip contrast at h = 128, and GraphQA degree under the top-128
    since heads and 5 random sets."""
    d = Path(folder)
    if not (d / "summary_e18.json").exists():
        return
    S = json.load(open(d / "summary_e18.json"))
    for c in S["cells"]:
        for pre, sub in (("etop", "edge"), ("ntop", "entry")):
            if c["cond"].startswith(pre) and c["cond"][4:] in ("16", "128"):
                M["e18"][model_key(c["model"])][sub][f"{c['cond'][4:]}|{fmt_key(c['ser'])}|{c['pid']}"] = cellv(c)
    for c in S.get("edge_random", []):
        if c["h"] in (16, 128):
            M["e18"][model_key(c["model"])]["edge_random"][f"{c['h']}|{fmt_key(c['ser'])}|{c['pid']}"] = {
                "mean": r4(c["mean"]), "min": r4(c["min"]), "max": r4(c["max"])}
    for c in S.get("flip", []):
        if c["h"] == 128 and c.get("kind") == "ntop":
            M["e18"][model_key(c["model"])]["entry_flip"][c["contrast"][3:]] = {"diff": r4(c["diff"]), "lo05": r4(c["lo05"])}
    for c in S.get("degree", []):
        if c["cond"] == "top128":
            M["e18"][model_key(c["model"])]["degree"][f"128|{fmt_key(c['ser'])}"] = {
                "ba": r4(c["acc"]), "lo": r4(c["ci_low"]), "hi": r4(c["ci_high"])}
    for c in S.get("degree_random", []):
        if c["h"] == 128:
            M["e18"][model_key(c["model"])]["degree_random"][f"128|{fmt_key(c['ser'])}"] = {
                "mean": r4(c["mean"]), "min": r4(c["min"]), "max": r4(c["max"])}

def imp_r19(M, folder):
    """Summary (CONFIG §6.13; RUN_PROTOCOL §4.18): summary_r19.json (r19.py summary over the E19, E20 and E21
    records); blocks e19 (knockout of the entries' attention to their own or the previous line head, patching of the
    top-128 since heads at the entries), e20 (list-line filler inside or outside, F = 8, 24, with the paired
    inside-minus-outside contrast) and e21 (relational tables, four formats, questions T1 and G1, and their means)."""
    d = Path(folder)
    if not (d / "summary_r19.json").exists():
        return
    S = json.load(open(d / "summary_r19.json"))
    e21 = {}
    for c in S["cells"]:
        m, v, ser, pid = model_key(c["model"]), c["variant"], c["ser"], c["pid"]
        if v in ("ko-own", "ko-other"):
            M["e19"][m]["ko"][f"{v[3:]}|{fmt_key(ser)}|{pid}"] = cellv(c)
        elif v == "patch-recolor-entries128":
            M["e19"][m]["patch"][f"recolor128|{fmt_key(ser)}|{pid}"] = cellv(c)
        elif v.startswith("fill-"):
            place, F = ("inside", v[11:]) if v.startswith("fill-inside") else ("outside", v[12:])
            M["e20"][m]["cells"][f"{place}|{fmt_key(ser)}|{F}"] = cellv(c)
        elif v == "e21":
            M["e21"][m][f"{ser}|{pid}"] = cellv(c)
            e21.setdefault(f"{ser}|{pid}", []).append(cellv(c)["ba"])
    for c in S.get("diffs", []):
        m = model_key(c["model"])
        if c["kind"] == "ko":
            M["e19"][m]["ko_diff"][fmt_key(c["ser"])] = {"diff": r4(c["diff"]), "hi95": r4(c["hi95"])}
        else:
            M["e20"][m]["contrast"][f"{fmt_key(c['ser'])}|{c['F']}"] = {"diff": r4(c["diff"]), "hi95": r4(c["hi95"])}
    for k, v in e21.items():
        if len(v) == 4:
            M["e21"]["mean"][k] = r4(sum(v) / 4)


def imp_r20(M, path):
    """Blocks (CONFIG §6.14; RUN_PROTOCOL §4.19): results/blocks_r20.json (m0/r20_blocks.py) holds the cells of
    E22-E33 in the structure of schema.json; the cells that restate other blocks are filled here from the
    imported data: the T1 losses of E11a beside E28a, the E11b-L AL cells beside
    E29, the E11a and E20 cells beside E30, the E10 cells of GPT-4o, the k_min-share predictions of the real
    topologies and the E1 k_min = 1 levels beside the late-position set."""
    B = json.load(open(path))

    def put(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and not ({"ba", "mean", "diff", "top"} & set(v)):
                put(dst[k], v)
            elif v is not None:
                dst[k] = v
    for blk, v in B.items():
        put(M[blk], v)
    sys.path.insert(0, str(DATA))
    from load_data import load_merged
    P = load_merged()
    E11 = {"Qwen2.5-7B": P["e11a"], "Llama-3.1-8B": P["e11a_llama"], "Gemma-2-9B": P["e11a_gemma"]}
    for m, blk in E11.items():
        for k, c in blk["none"].items():
            if f"16|{k}" in blk["top"]:
                M["e28"]["t1_loss"][m][k] = r4(c["ba"] - blk["top"][f"16|{k}"]["ba"])
        if "e29" in B and m in B["e29"]:
            for D in (6, 12, 24, 48):
                M["e29"][m][f"al|{D}"] = {"ba": P["e11bL"][m][f"AL|{D}"]["ba"]}
        if "e30" in B and m in B["e30"]:
            for f in ("AL", "NC"):
                M["e30"][m]["cells"][f"none|{f}"] = {"ba": blk["none"][f"{f}|D1a"]["ba"]}
                M["e30"][m]["cells"][f"outside|{f}"] = {"ba": P["e20"][m]["cells"][f"outside|{f}|24"]["ba"]}
    E31 = B.get("e31", {})
    if "gpt4o" in E31:
        for c in ("global", "depth1"):
            for f, v in P["api_model"]["GPT-4o"][c].items():
                if f in ("EL", "AL", "NC", "ECL"):
                    M["e31"]["gpt4o"][f"{c}|{f}"] = dict({k: v[k] for k in ("ba", "lo", "hi")}, src="E10")
    if "real" in E31:
        for m, cells in E31["real"].items():
            for key in cells:
                src, pid, f = key.split("|")
                stt = P["llm_main"]["strata"][m].get(f)
                if pid == "D1a" and stt is not None:
                    sh = P["real_graphs"]["stats"][src]["kmin_share"]
                    M["e31"]["real"][m][key]["pred"] = r4(sum(sh[k] * (stt["kmin"][k] if stt["kmin"][k] is not None else stt["kmin"]["3-4"]) for k in sh))
    if "latek" in E31:
        for m in E31["latek"]:
            stt = P["llm_main"]["strata"][m]
            for f in ("EL", "AL", "NC"):
                M["e31"]["latek"][m][f"{f}|k1"] = r4(stt[f]["kmin"]["1"])
            M["e31"]["latek"][m]["ECL|k1"] = r4(stt["ECL"]["kmin"]["1"] if "ECL" in stt else P["llm_formats"][m]["ECL"]["depth1"]["ba"])


def imp_kset_blocking(M, path):
    for f, v in json.load(open(path)).items():
        M["kset_blocking"][f.replace(".jsonl", "")] = {"violations": int(v["violations"])}


def imp_e5(M, path):
    for c in cells_of(path):
        m, f, src = model_key(c["model"]), fmt_key(c["ser"]), SRC[c["src"]]
        M["real_graphs"]["results"][m][src][f] = dict(cellv(c), strata=strata_of(c))


def imp_e5_stats(M, paths):
    bins = {"1": lambda k: k == 1, "2": lambda k: k == 2, "3-4": lambda k: 3 <= k <= 4, ">=5": lambda k: k >= 5}
    for p in paths:
        s = json.load(open(p))
        src = SRC[s["src"]]
        km = [k for k in s.get("kmin_NC", []) if k is not None]       # NC order, as the LLM strata
        M["real_graphs"]["stats"][src] = {
            "n": round(st.mean(s["n"]), 1), "edges": round(st.mean(s["edges"]), 1), "maxdeg": round(st.mean(s["maxdeg"]), 1),
            "meandeg": round(st.mean(s["meandeg"]), 2), "cycles": round(st.mean(s["cycles"]), 2),
            "share_maxdeg_ge8": round(sum(x >= 8 for x in s["maxdeg"]) / len(s["maxdeg"]), 3),
            "kmin_share": {b: (sum(f(k) for k in km) / len(km) if km else None) for b, f in bins.items()}}


def imp_graphml(M, path):
    for c in cells_of(path):
        if fmt_key(c["ser"]) == "GraphML":
            m = model_key(c["model"])
            M["graphml"]["by_model"][m]["depth1"] = cellv(c)
            if m == "Qwen2.5-7B":
                M["graphml"]["depth1"] = cellv(c)


def imp_e6(M, path):
    trunc = defaultdict(list)
    for c in cells_of(path):
        m, f = model_key(c["model"]), fmt_key(c["ser"])
        mode = {"zero": "zero", "few": "few", "cot": "cot"}[str(c.get("mode") or "zero")]
        M["prompting"][m][mode][f] = cellv(c)
        if mode == "cot":
            if "truncated" not in c:
                raise SystemExit("E6: CoT cell without 'truncated' (e4_stats.py writes it from reasoning_tokens)")
            trunc[m].append(c["truncated"])
    for m, v in trunc.items():
        M["prompting"][m]["cot_truncation"] = r4(st.mean(v))


def imp_e8(M, path):
    for c in cells_of(path):
        if c.get("cls") != "depth1" or c.get("variant") == "generate":
            continue                                   # Table T_robust reports depth one; global cells stay in the summary
        m, f, v = model_key(c["model"]), fmt_key(c["ser"]), c["variant"]
        if v not in VARIANTS:
            raise SystemExit(f"E8: unknown variant tag {v!r} (use {VARIANTS})")
        M["robustness"][m][v][f] = cellv(c)


def imp_e8_gen(M, paths):
    for p in paths:
        for name, s in json.load(open(p)).items():
            if not name.startswith("_") and s.get("generation_agreement") is not None:
                M["robustness"][model_key(name)]["gen_agreement"] = r4(s["generation_agreement"])


def imp_e10(M, path):
    for c in cells_of(path):
        M["api_model"]["GPT-4o"][c["cls"]][fmt_key(c["ser"])] = cellv(c)


# ------------------------------------------------------------------ controlled transformers
def ms(v):
    return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "seeds": [float(x) for x in v]}      # unrounded: mean/sd are their exact statistics (rule D7)


def imp_e3(M, path, sweep=False):
    runs = [json.loads(l) for l in open(path) if l.strip()]
    runs = last_wins(runs, lambda r: (r["pos"], r["task"], r["ser"], int(r["layers"]), r["seed"]),
                     "E3d" if sweep else "E3")
    acc, kacc = defaultdict(list), defaultdict(list)
    for r in runs:
        f = fmt_key(r["ser"])
        if sweep:
            for lab in E3K:
                if lab in (r.get("acc_k") or {}):
                    kacc[(int(r["layers"]), f, lab)].append(r["acc_k"][lab])
            continue
        arm = ARM[str(r["pos"]).lower()]
        task = {"red": "red", "dia": "dia"}[str(r["task"]).lower()]
        acc[(arm, task, f, "id")].append(r["acc_id"])
        acc[(arm, task, f, "ood")].append(r["acc_ood"])
        for lab, v in (r.get("acc_k") or {}).items():
            kacc[(arm, f, lab)].append(v)
    if sweep:
        for (Lx, f, lab), v in kacc.items():
            M["depth_sweep"][f"{Lx}|{f}"][lab] = ms(v)
        if len(kacc) != 6 * 4 * 6:
            WARN.append(f"E3d: {len(kacc)} of 144 cells (6 depths x 4 formats x 6 strata); run export_l6.py for L=6")
        return
    for (arm, task, f, split), v in acc.items():
        M["synthetic"]["cells"][f"{arm}|{task}|{f}|{split}"] = ms(v)
    for (arm, f, lab), v in kacc.items():
        M["synthetic"]["kcontrolled"][f"{arm}|{f}|{lab}"] = ms(v)


def imp_kdist(M, path):
    k = json.load(open(path))
    for split in ("id", "ood"):
        for lab, v in k[split].items():
            M["kdist"][split][lab] = r4(v)
        M["kdist"][f"{split}_vfirst"] = r4(k[f"{split}_vfirst"])
        if f"{split}_own" in k:
            M["kdist"][f"{split}_own"] = r4(k[f"{split}_own"])
    M["kdist"]["kc_vfirst"], M["kdist"]["kc_vsecond"] = r4(k["kc_vfirst"]), r4(k["kc_vsecond"])
    for lab, d in k.get("kc_by", {}).items():
        M["kdist"]["kc_by"][lab] = {x: r4(y) for x, y in d.items()}


def imp_e0b(M, path):
    runs = [json.loads(l) for l in open(path) if l.strip()]
    runs = last_wins(runs, lambda r: (r["d"], r["layers"], r["train_pairs"], r["steps"], r["seed"]), "E0b")
    g = defaultdict(list)
    for r in runs:
        g[f"{r['d']}|{r['layers']}|{r['train_pairs'] // 1000}k|{r['steps'] // 1000}k"].append(r["acc"])
    for k, v in g.items():
        M["pilot_scaling"][k] = ms(v)


def imp_e0b_chance(M, path):
    """E0b chance-cell reruns (results_chance3.jsonl): the six undecidable pilot cells, seeds 0-2."""
    runs = last_wins([json.loads(l) for l in open(path) if l.strip()],
                     lambda r: (r["task"], r["ser"], r["steps"], r["seed"]), "E0b chance cells")
    g = defaultdict(list)
    for r in runs:
        g[f"{r['task']}|{r['ser']}"].append(r["acc"])
    for k, v in g.items():
        M["pilot_chance3"][k] = ms(v)


def imp_e9(M, folder):
    d = Path(folder)
    E = M["mechanism"]
    BIN = {"1-8": "1-8", "9-24": "9-24", ">24": ">24", "25-1000000": ">24"}
    for f in ("AL", "NC"):
        pr = json.load(open(d / f"e9_probe_{f}.json"))
        layers = sorted(pr, key=int)
        for lab in sorted({b for L_ in layers for b in pr[L_]}):
            if lab in BIN:
                E["probe"][f"{f}|{BIN[lab]}"] = [r4(pr[L_].get(lab)) for L_ in layers]
            elif lab not in ("all", "n", "control", "selectivity"):
                WARN.append(f"E9 probe bin {lab!r} is not a paper bin (1-8, 9-24, >24)")
        E["patch"][f] = [r4(x) for x in json.load(open(d / f"e9_patch_{f}.json"))["restored_by_layer"]]
        rt = json.load(open(d / f"e9_retrieve_{f}.json"))
        for lab, v in rt["by_bin"].items():           # top-1 at rt["layer"] (best for the nearest bin)
            E["retrieval"][f"{f}|{BIN[lab]}"] = r4(v)
    inner = json.load(open(d / "e9_el_inner.json"))["acc_by_layer"]
    E["el_inner"] = [r4(inner[k]) for k in sorted(inner, key=int)]


def imp_compute(M, folders):
    """Measured compute (Table T_compute) from the run records' wall-clock seconds."""
    C = M["compute"]
    files = {}
    for fo in folders:
        for p in Path(fo).glob("*.jsonl"):
            files[p.name] = p
    def gpu_h(r):                     # GPU hours of a record: wall-clock seconds x GPUs occupied (72B: 4)
        return (r.get("sec") or 0.0) * r.get("n_gpus", 1) / 3600

    def llm_hours(prefix):
        return sum(gpu_h(json.loads(l)) for n, p in files.items() if re.match(prefix, n) for l in open(p) if l.strip())
    if "lr_sweep.jsonl" in files:
        runs = [json.loads(l) for l in open(files["lr_sweep.jsonl"]) if l.strip()]
        C["e3_lr_runs"] = len(runs)
        C["e3_lr_hours"] = round(sum(gpu_h(r) for r in runs), 1)
    if "e3.jsonl" in files:
        runs = [json.loads(l) for l in open(files["e3.jsonl"]) if l.strip()]
        C["e3_runs"] = len(runs)
        C["e3_hours_per_run"] = round(st.mean(gpu_h(r) for r in runs), 2)
    if "e3d.jsonl" in files:
        runs = [json.loads(l) for l in open(files["e3d.jsonl"]) if l.strip() and "reused_from" not in l]
        C["e3d_runs"] = len(runs)
        byL = defaultdict(list)
        for r in runs:
            byL[str(r["layers"])].append(gpu_h(r))
        for Lx, v in byL.items():
            C["e3d_hours_by_L"][Lx] = round(st.mean(v), 2)
        C["e3d_hours"] = round(sum(sum(v) for v in byL.values()), 1)
    for m in MODELS:
        tag = {"Qwen2.5-7B": "Qwen2.5-7B", "Llama-3.1-8B": "Llama-3.1-8B", "Gemma-2-9B": "gemma-2-9b", "Qwen2.5-72B": "Qwen2.5-72B"}[m]
        C["e1_hours"][m] = round(llm_hours(rf"e1_{re.escape(tag)}"), 1)
    for k, pre in (("e2", r"e2_"), ("e4", r"e4_"), ("e7", r"e7(q0)?_"), ("e8", r"e8_"), ("e5", r"e5_"), ("e6", r"e6_")):
        C[f"{k}_hours"] = round(llm_hours(pre), 1)
    extra = [json.loads(l) for n in ("lr_check.jsonl", "e3_nch.jsonl") if n in files for l in open(files[n]) if l.strip()]
    if extra:
        C["e3_extra_runs"] = len(extra)
        C["e3_extra_hours"] = round(sum(gpu_h(r) for r in extra), 1)
    for fo in folders:
        t = Path(fo) / "e9_timing.json"
        if t.exists():
            C["e9_hours"] = round(gpu_h(json.load(open(t))), 1)
    if "e10_gpt4o.jsonl" in files:
        C["e10_calls"] = sum(1 for l in open(files["e10_gpt4o.jsonl"]) if l.strip())
    for fo in folders:
        t = Path(fo) / "e11a_qwen7b_timing.json"
        if not t.exists():
            t = Path(fo) / "e11a_timing.json"        # name
        if t.exists():
            C["e11a_hours"] = round(gpu_h(json.load(open(t))), 1)
        # patching (Qwen), the Llama replication (ablation and patching), E3 name matching
        t = Path(fo) / "e11a_patch_qwen7b_timing.json"
        if t.exists():
            C["e11a_patch_hours"] = round(gpu_h(json.load(open(t))), 1)
        ts = [Path(fo) / n_ for n_ in ("e11a_llama8b_timing.json", "e11a_patch_llama8b_timing.json")]
        if all(t.exists() for t in ts):
            C["e11a_llama_hours"] = round(sum(gpu_h(json.load(open(t))) for t in ts), 1)
        t = Path(fo) / "match_heads.jsonl"
        if t.exists():                               # GPU hours; on CPU (n_gpus 0) the wall clock counts as one device
            C["e3_match_hours"] = round(sum((r.get("sec") or 0.0) * max(1, r.get("n_gpus", 1)) / 3600
                                            for r in map(json.loads, open(t)) if r.get("sec") is not None), 1)
        t = Path(fo) / "e11a_gemma9b_timing.json"
        if t.exists():
            C["e11a_gemma_hours"] = round(gpu_h(json.load(open(t))), 1)
        ts = sorted(Path(fo).glob("e17_*_timing.json"))   # (E17, three models)
        if ts:
            C["e17_hours"] = round(sum(gpu_h(json.load(open(t))) for t in ts), 1)
        ts = sorted(Path(fo).glob("e18_*_timing.json"))   # (E18, three models)
        if ts:
            C["e18_hours"] = round(sum(gpu_h(json.load(open(t))) for t in ts), 1)
        for ex in ("e19", "e20", "e21"):                     # (E19 to E21)
            ts = sorted(Path(fo).glob(f"{ex}_*_timing.json"))
            if ts:
                C[f"{ex}_hours"] = round(sum(gpu_h(json.load(open(t))) for t in ts), 1)
        t = Path(fo) / "e15.jsonl"
        if t.exists():
            rs = [json.loads(l) for l in open(t) if l.strip()]
            C["e15_runs"] = len(rs)
            C["e15_hours"] = round(sum((r.get("sec") or 0.0) * max(1, r.get("n_gpus", 1)) / 3600 for r in rs), 1)
    for k, pre in (("e11b", r"e11b_"), ("e11c", r"e11c_"), ("reasoning", r"reasoning_"), ("e12", r"e12_"), ("e13", r"e13_"),
                   ("e11bL", r"e11bL_"), ("e14", r"e14_"), ("e16", r"e16_")):
        if any(re.match(pre, n) for n in files):
            C[f"{k}_hours"] = round(llm_hours(pre), 1)
    e0b = [json.loads(l) for n in ("results_e0b.jsonl", "results_chance3.jsonl") if n in files
           for l in open(files[n]) if l.strip()]
    if e0b:
        C["e0b_runs"] = len(e0b)
        C["e0b_cpu_hours"] = round(sum(r["sec"] for r in e0b) / 3600, 1)


# ------------------------------------------------------------------ schema check and main
def unknown_paths(P0, M, path=()):
    """Paths of M that schema.json does not have (an empty dict in P0 is an open container)."""
    bad = []
    for k, v in M.items():
        if not isinstance(P0, dict) or (P0 and k not in P0):
            bad.append(path + (k,))
            continue
        sub = P0.get(k) if isinstance(P0, dict) else None
        if isinstance(v, dict) and isinstance(sub, dict) and sub:
            bad += unknown_paths(sub, v, path + (k,))
    return bad


def imp_r21(M, path):
    """Blocks (CONFIG §6.15; RUN_PROTOCOL §4.20): results/blocks_r21.json (m0/r21.py summary) holds the cells
    of E34-E36 in the structure of schema.json; the cells that restate earlier blocks are filled here from the
    imported data: the unablated E34 cells from E13 (edge existence), the
    E35 top-128 losses from E11a minus E17 and the null statistics from the 100 null cells against the unablated
    E11a cell, the unablated E36 template cells from E28a and the unablated 72B list cells from E28c."""
    B = json.load(open(path))
    sys.path.insert(0, str(DATA))
    from load_data import load_merged
    P = load_merged()
    E11 = {"Qwen2.5-7B": P["e11a"], "Llama-3.1-8B": P["e11a_llama"], "Gemma-2-9B": P["e11a_gemma"]}
    for m, E in B.get("e34", {}).items():
        for c, v in E.get("cells", {}).items():
            M["e34"][m]["cells"][c] = v
        for f in ("EL", "AL", "NC", "ECL"):
            if f"none|{f}" not in E.get("cells", {}):
                M["e34"][m]["cells"][f"none|{f}"] = {k: P["e13"][m][f"edge|{f}"][k] for k in ("ba", "lo", "hi")}
        for f, v in E.get("rand128", {}).items():
            M["e34"][m]["rand128"][f] = v
    for m, E in B.get("e35", {}).items():
        for k, v in E.items():
            f, pid = k.split("|")
            base = E11[m]["none"][f"{f}|{pid}"]["ba"]
            top = base - P["e17"][m]["top"][f"128|{f}|{pid}"]["ba"]
            loss = sorted(base - x for x in v["null_ba"])
            q = 0.95 * (len(loss) - 1)                      # linear interpolation, as numpy.quantile
            q95 = loss[int(q)] + (q - int(q)) * (loss[min(int(q) + 1, len(loss) - 1)] - loss[int(q)])
            M["e35"][m][k] = dict(top=r4(top), null_mean=r4(st.mean(loss)), null_sd=r4(st.stdev(loss)),
                                  null_q95=r4(q95), null_max=r4(max(loss)),
                                  p_emp=r4((1 + sum(x >= top for x in loss)) / (len(loss) + 1)), n_sets=len(loss))
    for m, T in B.get("e36", {}).get("tpl", {}).items():
        for k, v in T.items():
            M["e36"]["tpl"][m][k] = v
        for tpl in ("T2", "T3"):
            for f in ("EL", "AL", "NC", "ECL"):
                for pid in ("D1a", "Q0"):
                    M["e36"]["tpl"][m][f"{tpl}|none|{f}|{pid}"] = {k: P["e28"]["repl"][m][f"{tpl}|none|{f}|{pid}"][k] for k in ("ba", "lo", "hi")}
    for m, D in B.get("e36", {}).get("d1c", {}).items():
        for k, v in D.items():
            M["e36"]["d1c"][m][k] = v
    for k, v in B.get("e36", {}).get("q72", {}).items():
        M["e36"]["q72"][k] = v
    for f in ("AL", "NC"):
        if B.get("e36", {}).get("q72") and f"none|{f}" not in B["e36"]["q72"]:
            M["e36"]["q72"][f"none|{f}"] = {k: P["e28"]["ko72"][f"none|{f}|D1a"][k] for k in ("ba", "lo", "hi")}


def main():
    ap = argparse.ArgumentParser()
    one = ["e1", "e1-strata", "e1-strata-d1a", "e1-meanci", "e2", "e2-controls", "e2-holm", "e4", "e4-ecl",
           "e4-strata", "e4-strata-d1a", "e7", "e7-meanci", "e5", "graphml", "graphml-strata", "graphml-strata-d1a",
           "e6", "e8", "e9", "e10", "e3", "e3d", "kdist", "e0b", "e0b-chance", "e7-twin", "r9", "r9-e3",
           "kset-blocking", "r12", "r14", "r16", "r18", "r19", "r20", "r21"]
    for k in one:
        ap.add_argument(f"--{k}")
    ap.add_argument("--e5-stats", nargs="*", default=[])
    ap.add_argument("--e8-generate", nargs="*", default=[])
    ap.add_argument("--compute", nargs="*", default=[])
    ap.add_argument("--reset", action="store_true", help="start from an empty summary.json")
    ap.add_argument("--out", default=str(MEASURED))
    a = ap.parse_args()
    sys.path.insert(0, str(DATA))
    from load_data import load_measured, overlay
    P0 = json.load(open(PATH))                       # schema; never written
    M = tree()
    # fixed order: cells first, strata last, so the D1a+D1c strata are never overwritten by a pooled summary
    steps = [("e1", imp_e1), ("e1_meanci", imp_meanci), ("e2", imp_e2), ("e2_controls", lambda M_, p: imp_e2(M_, p, True)),
             ("e2_holm", imp_e2_holm), ("e4", imp_e4), ("e4_ecl", imp_e4_ecl), ("e7", imp_e7),
             ("e7_meanci", lambda M_, p: imp_meanci(M_, p, e7=True)), ("e5", imp_e5), ("graphml", imp_graphml),
             ("e6", imp_e6), ("e8", imp_e8), ("e10", imp_e10), ("e3", imp_e3), ("e3d", lambda M_, p: imp_e3(M_, p, True)),
             ("kdist", imp_kdist), ("e0b", imp_e0b), ("e0b_chance", imp_e0b_chance), ("e9", imp_e9),
             ("e7_twin", imp_e7_twin), ("r9", imp_r9), ("r9_e3", imp_r9_e3), ("kset_blocking", imp_kset_blocking), ("r12", imp_r12), ("r14", imp_r14), ("r16", imp_r16), ("r18", imp_r18), ("r19", imp_r19), ("r20", imp_r20), ("r21", imp_r21),
             ("e1_strata", lambda M_, p: imp_strata(M_, p, "strata")),
             ("e1_strata_d1a", lambda M_, p: imp_strata(M_, p, "strata_d1a")),
             ("e4_strata", lambda M_, p: imp_strata(M_, p, "strata")),
             ("e4_strata_d1a", lambda M_, p: imp_strata(M_, p, "strata_d1a", formats={"ECL"})),
             ("graphml_strata", lambda M_, p: imp_strata(M_, p, "strata", formats={"GraphML"})),
             ("graphml_strata_d1a", lambda M_, p: imp_strata(M_, p, "strata_d1a", formats={"GraphML"}))]
    for k, fn in steps:
        v = getattr(a, k)
        if v:
            fn(M, v)
    if a.e5_stats:
        imp_e5_stats(M, a.e5_stats)
    if a.e8_generate:
        imp_e8_gen(M, a.e8_generate)
    if a.compute:
        imp_compute(M, a.compute)
    new = plain(M)
    bad = unknown_paths(P0, new)
    if bad:
        raise SystemExit("import would write paths that schema.json does not have (nothing written):\n  " +
                         "\n  ".join("/".join(map(str, p)) for p in bad[:40]))
    out = overlay({} if a.reset else load_measured(a.out), new)
    json.dump(out, open(a.out, "w"), indent=1)
    for w in dict.fromkeys(WARN):
        print("WARNING", w, file=sys.stderr)
    n = sum(1 for _ in __import__("load_data").leaves(new))
    print(f"imported {n} values into {a.out}")


if __name__ == "__main__":
    main()
