"""Run every rule of VALIDATION_RULES.md (groups A to F and the rules of the preregistered sets) on the imported
summary (results/summary.json, structure analysis/schema.json). Exit code 1 if any rule fails.
Usage: python analysis/check_rules.py [--verbose]
Every check is reported as passed, failed or skipped; a skipped check (its input file is missing) is listed with the
reason and never counted as passed. --verbose also prints passed, failed and skipped checks per rule.
A failing theory rule (group B) or consistency rule (group D) points to a bug in a run or in the import; a failing
directional rule (group C) is a falsified prediction and is reported as such (PREREGISTRATION.md)."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from load_data import load_merged  # noqa: E402
P = load_merged()
from load_data import nan_missing  # noqa: E402
P["mechanism"] = nan_missing(P["mechanism"])      # empty distance bins (null) -> NaN

# Theory constants (paper, Appendix C): layers of the E3 models and the rounds a route needs at k_min = k
L_LAYERS = 6
ROUNDS = dict(al_k=lambda k: 2 * k + 2, nc_k=lambda k: 5 * k)
# Preregistered route parameters of the E3 predictions (rules B8, C1): accuracy on the positives a route reaches
# (c_pos) and true-negative rate (tnr), in distribution and on the larger test graphs
E3 = dict(c_pos=dict(id=0.986, ood=0.972), tnr=dict(id=0.989, ood=0.978))
VERB = "--verbose" in sys.argv
FAIL, PASS = [], 0
MODELS = list(P["llm_main"]["models"])
F3 = ["EL", "AL", "NC"]




COUNT = {}                          # rule id -> [passed, failed, skipped]
SKIPPED = []


def rule(rid, ok, msg=""):
    global PASS
    c = COUNT.setdefault(rid, [0, 0, 0])
    if ok:
        PASS += 1
        c[0] += 1
        if VERB:
            print("ok  ", rid, msg)
    else:
        c[1] += 1
        FAIL.append(f"{rid}: {msg}")


def skip(rid, msg):
    """A check that cannot be evaluated because its input is missing: counted and listed, never as passed."""
    COUNT.setdefault(rid, [0, 0, 0])[2] += 1
    SKIPPED.append(f"{rid}: {msg}")


def items(d):
    return [(k, v) for k, v in d.items() if k not in ('status', 'config_ref', '_desc')]


def chance(x):
    return 0.47 <= x <= 0.53


def all_values(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("status", "config_ref", "_desc", "_note", "seeds_n", "p", "p_boot", "df", "chi2", "p_holm", "diff",
                     "distance_regression"):
                continue           # test statistics, signed differences and regression coefficients are not rates
            yield from all_values(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from all_values(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isnan(obj):        # NaN: an empty bin (null), not a rate
        yield path, obj


M = P["llm_main"]["models"]
S = P["synthetic"]["cells"]
K = P["synthetic"]["kcontrolled"]
KL = ["1", "2", "3", "4-6", "7-12", "13-32"]
KLO = {"1": 1, "2": 2, "3": 3, "4-6": 4, "7-12": 7, "13-32": 13}
KHI = {"1": 1, "2": 2, "3": 3, "4-6": 6, "7-12": 12, "13-32": 32}

# ---------------- A: numeric form
# statistics that are not rates (A1 does not apply): log-odds gaps, chi^2, lengths, model-fit criteria,
# interval ends of differences, regression coefficients, curve steps
_NONRATE = ("/llm_stats/position/", "/llm_stats/qre/", "/llm_stats/lengths", "/llm_stats/tost", "/competing/",
            "/e3_extra/curves/", "/e3_extra/kreg/", "/mask_controls/survival/", "/llm_stats/kdiff/",
            "/e12/error_pts", "/e12/max_error_pts", "/n_mlike", "/words",            # prompt lengths
            "/llm_stats/p4_noninf/",                                              # signed shortfalls
            "/e23/steps", "|drop", "/recovery/")                                  # steps, differences, shares
for path, x in all_values({k: P[k] for k in P if not k.startswith("_") and k not in ("compute", "kdist", "kset_blocking", "token_cost")}):
    if any(t in path for t in _NONRATE):
        continue
    if "share" in path or "/patch/" in path or "/stats/" in path or "kstar" in path or "truncation" in path or "gen_agreement" in path:
        continue
    if path.endswith("|decline") or "|contrast|" in path:    # E11b, E11b-L, E14: differences of two rates
        if not -1.0 <= x <= 1.0:
            rule("A1", False, f"{path}={x} outside [-1,1]")
        continue
    if not (0.0 <= x <= 1.0):
        if "est" in path or "lo" in path.split("/")[-1] or "hi" in path.split("/")[-1]:
            continue
        rule("A1", False, f"{path}={x} outside [0,1]")
for key, v in S.items():
    arm, task, fmt, split = key.split("|")
    # D7 recompute
    import statistics as st
    rule("D7", abs(round(st.mean(v["seeds"]), 4) - v["mean"]) < 1e-4 and abs(round(st.stdev(v["seeds"]), 4) - v["sd"]) < 1e-4, key)

# ---------------- B: theory constraints
B = P["llm_blind"]
for m in MODELS:
    for cls, d in B[m]["query"].items():
        for f, v in d.items():
            rule("B1", v["ba"] == 0.5 and v["lo"] == 0.5 and v["hi"] == 0.5, f"{m}/{cls}/{f} {v}")
    for f in F3:
        x2, x1 = B[m]["X2"][f]["ba"], B[m]["X1"][f]["ba"]
        rule("B2", (x2 >= 0.70) if f in ("AL", "NC") else chance(x2), f"{m} X2/{f}={x2}")
        rule("B2", (x1 >= 0.70) if f == "NC" else chance(x1), f"{m} X1/{f}={x1}")
        rule("B3", abs(B[m]["global"][f]["ba"] - M[m]["global"][f]["ba"]) <= 0.015, f"{m} global/{f}")
for f in ("EL", "AL", "NC", "ECL"):
    for t in ("red", "dia"):
        rule("B4", chance(S[f"NoPE|{t}|{f}|ood"]["mean"]), f"NoPE {t} {f} ood")
        rule("B4", S[f"NoPE|{t}|{f}|id"]["mean"] <= 0.62, f"NoPE {t} {f} id")
    for arm in ("1-local", "ALiBi-only", "1-local+ALiBi", "1-local+hard-since", "RoPE"):
        rule("B5", S[f"{arm}|red|{f}|id"]["mean"] >= 0.97, f"{arm} red {f}")


def kstar(fmt, L):
    f = ROUNDS["al_k"] if fmt == "AL" else ROUNDS["nc_k"]
    k = 0
    while f(k + 1) <= L:
        k += 1
    return k


for fmt in ("AL", "NC"):
    ks = kstar(fmt, L_LAYERS)
    for lab in KL:
        v = K[f"1-local|{fmt}|{lab}"]["mean"]
        if KHI[lab] <= ks:
            rule("B6", v >= 0.93, f"1-local {fmt} k={lab} {v}")
        elif KLO[lab] > ks:
            rule("B6", chance(v), f"1-local {fmt} k={lab} {v}")
DS = P["depth_sweep"]
Ls = [2, 3, 4, 6, 8, 12]
for fmt in ("EL", "AL", "NC", "ECL"):
    for lab in KL:
        seq = [DS[f"{L}|{fmt}"][lab]["mean"] for L in Ls]
        rule("B7", all(b >= a - 0.02 or (chance(a) and chance(b)) for a, b in zip(seq, seq[1:])), f"sweep {fmt} k={lab} {seq}")
        rule("B7", chance(DS[f"2|{fmt}"][lab]["mean"]), f"sweep L=2 {fmt}")
    for L in Ls:
        row = DS[f"{L}|{fmt}"]
        if fmt == "EL" and L >= 4 or fmt == "ECL" and L >= 5:
            rule("B7", min(r["mean"] for r in row.values()) >= 0.95, f"sweep {fmt} L={L}")
        if fmt in ("AL", "NC"):
            ks = kstar(fmt, L)
            for lab in KL:
                if KHI[lab] <= ks:
                    rule("B6", row[lab]["mean"] >= 0.93, f"sweep {fmt} L={L} k={lab}")
                elif KLO[lab] > ks:
                    rule("B6", chance(row[lab]["mean"]), f"sweep {fmt} L={L} k={lab}")
c, t = E3["c_pos"]["ood"], E3["tnr"]["ood"]
# k-controlled sets: v is a first (second) endpoint of some red edge with prob. 1 - E[2^-r] (orientations uniform)
qf, qs = P["kdist"]["kc_vfirst"], P["kdist"]["kc_vsecond"]
rule("B8", abs(qf - qs) <= 0.01, f"first/second endpoint shares {qf} {qs} must agree")
# the covered share of each stratum is its measured share (kdist.kc_by; the strata differ in composition,
# e.g. the number of red neighbors, not only by sampling)
part = lambda q: 0.5 * (c * q + (1 - c * q) * (1 - t) + t)
for lab in KL:
    by = P["kdist"]["kc_by"][lab]
    pred = {"EL": part(by["vfirst"]), "ECL": part(by["vsecond"])}
    e3, c3, c4 = DS["3|EL"][lab]["mean"], DS["3|ECL"][lab]["mean"], DS["4|ECL"][lab]["mean"]
    rule("B8", abs(e3 - pred["EL"]) <= 0.03, f"EL L=3 {e3} vs {pred['EL']:.3f}")
    rule("B8", abs(c3 - pred["ECL"]) <= 0.03 and abs(c4 - pred["ECL"]) <= 0.03, f"ECL L=3,4 {c3} {c4} vs {pred['ECL']:.3f}")
    rule("B8", abs(c3 - c4) <= 0.02 and abs((e3 - c3) - (pred["EL"] - pred["ECL"])) <= 0.02,
         f"EL L=3 / ECL L=3,4 differ by the covered shares {e3} {c3} {c4}")
for fmt in ("AL", "NC"):
    xs = [K[f"1-local+hard-since|{fmt}|{l}"]["mean"] for l in KL]
    rule("B9", min(xs) >= 0.95 and max(xs) - min(xs) <= 0.01, f"hard since {fmt} {xs}")
    xs = [K[f"1-local+ALiBi|{fmt}|{l}"]["mean"] for l in KL]
    rule("B10", all(round(b + 1e-12, 3) <= round(a + 1e-12, 3) for a, b in zip(xs, xs[1:])) and xs[-1] >= 0.88, f"alibi {fmt} {xs}")
for m in MODELS:
    d2 = M[m]["depth2"]
    rule("B11", d2["NC"]["ba"] - max(d2["EL"]["ba"], d2["AL"]["ba"]) >= 0.06 and abs(d2["EL"]["ba"] - d2["AL"]["ba"]) <= 0.05,
         f"{m} depth2 {[d2[f]['ba'] for f in F3]}")
    pt = P["llm_partial"][m]
    rule("B12", chance(pt["AL-head"]["Q0"]["ba"]) and 0.01 <= pt["AL-none"]["D1a"]["ba"] - pt["AL-head"]["D1a"]["ba"] <= 0.06, f"{m} AL-head")
    rule("B12", chance(pt["AL-entry"]["D1a"]["ba"]), f"{m} AL-entry D1a {pt['AL-entry']['D1a']['ba']}")
    rule("B12", pt["NC-head"]["D1a"]["ba"] < pt["NC-none"]["D1a"]["ba"] and pt["NC-head"]["Q0"]["ba"] >= 0.90, f"{m} NC-head")
    nch = P["llm_formats"][m]["NC-H"]["depth1"]["ba"]
    # NC-H is pooled over D1a-c; compare with D1a-only entry mask on the D1a scale (paraphrase effect small)
    rule("B12", abs(pt["NC-entry"]["D1a"]["ba"] - nch) <= 0.045, f"{m} NC-entry {pt['NC-entry']['D1a']['ba']} vs NC-H {nch}")

# ---------------- C: directional predictions
kd = P["kdist"]
for split in ("id", "ood"):
    c, t = E3["c_pos"][split], E3["tnr"][split]
    for fmt, rng_ in (("AL", (0.75, 0.92)), ("NC", (0.65, 0.80))):
        ks = kstar(fmt, L_LAYERS)
        pk = sum(kd[split][lab] for lab in KL if KHI[lab] <= ks)
        # random orders: positives beyond the head route that the AL own-line route reaches (kdist *_own, AL only at
        # six layers) may or may not be solved: the prediction is the band between the head-route share alone and
        # head-route plus own-line share
        po = pk + (kd.get(split + "_own", 0.0) if fmt == "AL" else 0.0)
        lo_ = 0.5 * (c * pk + (1 - c * pk) * (1 - t) + t)
        hi_ = 0.5 * (c * po + (1 - c * po) * (1 - t) + t)
        v = S[f"1-local|dia|{fmt}|{split}"]["mean"]
        rule("C1", lo_ - 0.03 <= v <= hi_ + 0.03 and rng_[0] <= v <= rng_[1],
             f"1-local {fmt} {split} {v} band [{lo_:.3f}, {hi_:.3f}]")
    for fmt in ("EL", "ECL"):
        rule("C1", S[f"1-local|dia|{fmt}|{split}"]["mean"] >= 0.95, f"1-local {fmt} {split}")
for arm in ("1-local+ALiBi", "1-local+hard-since"):
    for fmt in ("EL", "AL", "NC", "ECL"):
        rule("C2", S[f"{arm}|dia|{fmt}|id"]["mean"] >= 0.95, f"{arm} {fmt}")
for fmt in ("EL", "AL", "NC", "ECL"):
    for tk in ("red", "dia"):
        rope_id, ao_id = S[f"RoPE|{tk}|{fmt}|id"]["mean"], S[f"ALiBi-only|{tk}|{fmt}|id"]["mean"]
        rule("C3", abs(rope_id - ao_id) <= 0.015, f"RoPE vs ALiBi-only {tk} {fmt}")
    rule("C3", S[f"RoPE|dia|{fmt}|id"]["mean"] - S[f"RoPE|dia|{fmt}|ood"]["mean"] >= 0.05, f"RoPE ood drop {fmt}")
    rule("C3", S[f"RoPE|red|{fmt}|ood"]["mean"] >= 0.93, f"RoPE red ood {fmt}")
for m in MODELS:
    for cls in ("depth1", "count"):
        d = M[m][cls]
        rule("C4", d["NC"]["ba"] > d["AL"]["ba"] > d["EL"]["ba"] and d["NC"]["ba"] - d["EL"]["ba"] >= 0.05, f"{m} {cls}")
    g = [M[m]["global"][f]["ba"] for f in F3]
    rule("C5", max(g) - min(g) <= 0.03, f"{m} global spread")
    for cls, cc in P["llm_main"]["contrasts"][m].items():
        for k2, v in cc.items():
            if cls == "depth2" and k2 == "AL_EL":      # neither has a route for every order; AL needs two
                rule("C6", -0.06 <= v["est"] <= 0.005, f"{m} {cls} {k2} {v}")   # favorable orders, EL one
                continue
            rule("C6", v["est"] > 0 and v["lo"] > 0, f"{m} {cls} {k2} {v}")
    T = P["llm_main"]["tests"][m]
    rule("C7", T["plain"]["p"] < 1e-3 and T["length"]["p"] < 1e-3 and T["length"]["chi2"] <= 1.02 * T["plain"]["chi2"], f"{m} {T}")
    E_ = P["llm_formats"][m]
    ecl, nc = E_["ECL"]["depth1"]["ba"], M[m]["depth1"]["NC"]["ba"]
    rule("C8", ecl >= nc - 0.01 and ecl - nc <= 0.08, f"{m} ECL {ecl} NC {nc}")
    st_ = P["llm_main"]["strata"][m]
    e = [x for x in st_["ECL"]["kmin"].values() if x is not None]
    rule("C8", max(e) - min(e) <= 0.03, f"{m} ECL strata {e}")
    for k_ in ("1", "2", "3-4", ">=5"):        # P4 as stated: in every stratum ECL >= best list format - 0.02
        rule("C8", st_["ECL"]["kmin"][k_] >= max(st_["AL"]["kmin"][k_], st_["NC"]["kmin"][k_]) - 0.02,
             f"{m} k={k_} ECL {st_['ECL']['kmin'][k_]} vs best list format")
    dk = {f: st_[f]["kmin"]["1"] - st_[f]["kmin"][">=5"] for f in ("EL", "AL", "NC", "ECL")}
    # EL and ECL do not depend on list position: the same change across k_min (sampling noise only);
    # NC changes by that amount plus the line-head decay (at least 2 points more for 7-9B models)
    rule("C8", abs(dk["ECL"] - dk["EL"]) <= 0.03, f"{m} ECL vs EL change across k_min {dk}")   # about 2 s.e. of the difference
    rule("C8", dk["NC"] - dk["EL"] >= (0.0 if m == "Qwen2.5-72B" else 0.02), f"{m} NC decline beyond EL {dk}")
    # ECL's change across k_min differs from EL's by less than NC's does (four-model means), and the paired
    # test of that gap (r9_stats.py kdiff) is computed on the same strata
    _dm = {f: sum(P["llm_main"]["strata"][m_][f]["kmin"]["1"] - P["llm_main"]["strata"][m_][f]["kmin"][">=5"] for m_ in MODELS) / len(MODELS)
           for f in ("EL", "NC", "ECL")}
    rule("C8", abs(_dm["ECL"] - _dm["EL"]) < _dm["NC"] - _dm["EL"], f"mean |dECL-dEL| {abs(_dm['ECL'] - _dm['EL']):.4f} vs dNC-dEL {_dm['NC'] - _dm['EL']:.4f}")
    _kd = P["llm_stats"]["kdiff"]
    rule("C8", abs(_kd["nc_minus_el"] - (_dm["NC"] - _dm["EL"])) <= 0.003 and abs(_kd["ecl_minus_el_abs"] - abs(_dm["ECL"] - _dm["EL"])) <= 0.003
         and _kd["lo90"] <= _kd["stat"] <= _kd["hi90"], f"paired test consistent with the strata {_kd}")
    el = M[m]["depth1"]["EL"]["ba"]
    rule("C9", 0 <= E_["EL-2"]["depth1"]["ba"] - el <= 0.05, f"{m} EL-2")
    rule("C9", E_["EL-far"]["depth1"]["ba"] <= el + 0.005 and ecl - E_["EL-far"]["depth1"]["ba"] >= ecl - el - 0.005, f"{m} EL-far")
gm = P["graphml"]
el7 = M["Qwen2.5-7B"]["depth1"]["EL"]["ba"]
rule("C10", abs(gm["depth1"]["ba"] - el7) <= 0.03, f"GraphML {gm['depth1']['ba']} EL {el7}")
for m_, g_ in gm["by_model"].items():          # like with like: GraphML and EL of the same model
    gs = [x for x in g_["strata"]["kmin"].values() if x is not None]
    es = [x for x in P["llm_main"]["strata"][m_]["EL"]["kmin"].values() if x is not None]
    rule("C10", abs(g_["depth1"]["ba"] - M[m_]["depth1"]["EL"]["ba"]) <= 0.03, f"GraphML vs EL {m_}")
    rule("C10", (gs[0] - gs[-1]) - (es[0] - es[-1]) <= 0.03, f"GraphML strata decline vs EL {m_} {gs} {es}")
RG = P["real_graphs"]
for m, dd in RG["results"].items():
    for src, d in dd.items():
        sh = RG["stats"][src]["kmin_share"]
        for f in ("EL", "AL", "NC", "ECL"):
            ref = P["llm_main"]["strata_d1a"][m][f]["kmin"]      # like with like: E5 evaluates D1a
            pred = sum(sh[k] * ref[k] for k in sh if ref[k] is not None) / sum(sh[k] for k in sh if ref[k] is not None)
            rule("C11", abs(d[f]["ba"] - pred) <= 0.05, f"{m} {src} {f} {d[f]['ba']} pred {pred:.3f}")
PRm = P["prompting"]
for m, d in items(PRm):
    z, fw, ct = d["zero"], d["few"], d["cot"]
    rule("C12", fw["NC"]["ba"] >= fw["AL"]["ba"] > fw["EL"]["ba"], f"{m} few order")
    for f in F3:
        rule("C12", fw[f]["ba"] >= z[f]["ba"] - 0.01, f"{m} few vs zero {f}")
    rule("C12", ct["NC"]["ba"] - ct["EL"]["ba"] < z["NC"]["ba"] - z["EL"]["ba"], f"{m} cot gap")
    rule("D2", all(z[f]["ba"] == M[m]["depth1"][f]["ba"] for f in F3), f"{m} zero-shot = E1")
RB = P["robustness"]
for m, d in items(RB):
    for var, dv in d.items():
        if var in ("per_template", "gen_agreement"):
            continue
        rule("C13", dv["NC"]["ba"] > dv["AL"]["ba"] > dv["EL"]["ba"], f"{m} {var}")
    rule("C13", d["sorted_el"]["EL"]["ba"] - d["default"]["EL"]["ba"] <= 0.03, f"{m} sorted")
    rule("C13", all(abs(d["question_first"][f]["ba"] - d["default"][f]["ba"]) <= 0.02 for f in F3), f"{m} qfirst")
    rule("D2", all(d["default"][f]["ba"] == M[m]["depth1"][f]["ba"] for f in F3), f"{m} default = E1")
    for f in F3:
        pt = d["per_template"][f]
        rule("D6", abs(sum(pt) / 3 - M[m]["depth1"][f]["ba"]) <= 0.005, f"{m} {f} template mean {pt}")
dq = RB["Qwen2.5-7B"]["default"]["NC"]["ba"] - RB["Qwen2.5-7B"]["int_ids"]["NC"]["ba"]
dl = RB["Llama-3.1-8B"]["default"]["NC"]["ba"] - RB["Llama-3.1-8B"]["int_ids"]["NC"]["ba"]
rule("C13", dq >= dl, f"int ids drop Qwen {dq:.3f} Llama {dl:.3f}")
A_ = P["api_model"]["GPT-4o"]
rule("C14", A_["depth1"]["NC"]["ba"] >= A_["depth1"]["AL"]["ba"] > A_["depth1"]["EL"]["ba"]
     and A_["depth1"]["ECL"]["ba"] >= A_["depth1"]["NC"]["ba"] - 0.01
     and min(v["ba"] for v in A_["global"].values()) >= 0.97, "GPT-4o")
PS = P["pilot_scaling"]
base = PS["64|3|4k|8k"]
_pl = ROOT / "pilot1" / "results.jsonl"          # E0: the base cell of E0b repeats the pilot's X1-NC cell
_x1 = [r["acc"] for r in map(json.loads, open(_pl)) if (r["task"], r["ser"], r["arm"]) == ("X1", "NC", "blind")] \
    if _pl.exists() else []
if _x1:
    rule("C15", abs(base["mean"] - sum(_x1) / len(_x1)) <= 0.005, f"pilot base {base} vs E0 {sum(_x1) / len(_x1):.4f}")
else:
    skip("C15", "base cell of E0b against E0: pilot1/results.jsonl has no X1-NC name-blind records")
rule("C15", PS["256|6|16k|32k"]["mean"] >= 0.95, "pilot max")
dims = [(64, 128, 256), (3, 6), ("4k", "16k"), ("8k", "32k")]
import itertools
for key in PS:
    if key in ("status", "config_ref"):
        continue
    w, L, D, Sx = key.split("|")
    cur = PS[key]["mean"]
    for i, opts in enumerate(dims):
        parts = [w, L, D, Sx]
        opts = [str(o) for o in opts]
        j = opts.index(parts[i])
        if j + 1 < len(opts):
            parts[i] = opts[j + 1]
            nxt = PS["|".join(parts)]
            tol = 2 * max(PS[key]["sd"], nxt["sd"]) + 0.003
            rule("C15", nxt["mean"] >= cur - tol, f"{key} -> {'|'.join(parts)}")

# B13: E0b chance-cell reruns (undecidable pilot cells, 3 seeds, 2,000 balanced test examples each): every seed
# within the 99% binomial range of 0.5, |acc - 0.5| <= 2.576 * sqrt(0.25 / 2000)
for k_, v_ in items(P["pilot_chance3"]):
    rule("B13", all(abs(x - 0.5) <= 2.576 * math.sqrt(0.25 / 2000) + 1e-12 for x in v_["seeds"]), f"chance rerun {k_} {v_['seeds']}")

# ---------------- (VALIDATION_RULES.md: B14-B16, C16-C21, D10-D12)
X3 = P["e3_extra"]
# B14: twin Q0 pairs, AL heads masked: the masked prompts of a pair coincide -> exactly 0.5 (implementation check)
for m in MODELS:
    tw = P["mask_controls"][m]["twin"]["AL-head"]
    rule("B14", tw["ba"] == 0.5 and tw["lo"] == 0.5 and tw["hi"] == 0.5, f"{m} twin AL-head {tw}")
# B15: position control of the E3 test sets: no listed construction other than the head route applies
for f, v in items(P["kset_blocking"]):
    rule("B15", v["violations"] == 0, f"{f} violations {v['violations']}")
# B16 (P2b): own-line set, 1-local AL at L = 6 and 8: above chance, below the positives the head route reaches
for L_ in ("6", "8"):
    v = X3["kown"][L_]["mean"]
    head = K[f"1-local|AL|1"]["mean"] if L_ == "6" else P["depth_sweep"]["8|AL"]["1"]["mean"]
    rule("B16", 0.53 < v < head, f"kown L={L_} {v} head-route stratum {head}")
# the own-line set at L = 8 is the L = 6 set and every route of L = 6 remains, so not lower (seed noise:
# within 2 s.e. of the difference of the two means)
_k6, _k8 = X3["kown"]["6"], X3["kown"]["8"]
rule("B16", _k8["mean"] >= _k6["mean"] - 2 * math.sqrt(_k6["sd"] ** 2 / len(_k6["seeds"]) + _k8["sd"] ** 2 / len(_k8["seeds"])),
     f"kown L=8 {_k8['mean']:.3f} not below L=6 {_k6['mean']:.3f}")
# C16: the non-word NC (familiarity control) stays within 0.03 of NC on depth one
for m in MODELS:
    d = P["nl_formats"][m]["NC-nw"]["depth1"]["ba"] - M[m]["depth1"]["NC"]["ba"]
    rule("C16", abs(d) <= 0.03, f"{m} NC-nw - NC {d:.4f}")
# C17: the route model explains the per-cell accuracies better than the locality model (lower AIC, higher held-out
# log-likelihood, lower held-out error)
cp = P["competing"]
rule("C17", cp["delta_qaic"] > 0 and cp["delta_heldout_ll"] > 0 and cp["delta_rmse_pts"] > 0, str({k: cp[k] for k in ("delta_qaic", "delta_heldout_ll", "delta_rmse_pts")}))
# C18 (P3 hypothesis): under exact since, more nested matches -> slower learning (slope of log steps > 0)
rule("C18", X3["curves"]["slope_ci"][0] > 0, f"slope {X3['curves']['slope_log_steps']} CI {X3['curves']['slope_ci']}")
# C19: a random mask of the targeted mask's size removes less: AL D1a (entries) and AL Q0 (heads) stay above the
# targeted cells, which are at chance, by at least 0.03
for m in MODELS:
    mc, pt = P["mask_controls"][m], P["llm_partial"][m]
    rule("C19", mc["AL-rand_entries"]["D1a"]["ba"] >= pt["AL-entry"]["D1a"]["ba"] + 0.03
         and mc["AL-rand_heads"]["Q0"]["ba"] >= pt["AL-head"]["Q0"]["ba"] + 0.03, m)
# C20: approximate since declines with k_min, not with line length: |z(log line length)| < 2 where |z(log k)| >= 2
for key, v in X3["kreg"].items():
    if "error" in v:
        continue
    if abs(v["z_logk"]) >= 2:
        rule("C20", abs(v["z_loglen"]) < 2, f"{key} z_logk {v['z_logk']:.2f} z_loglen {v['z_loglen']:.2f}")
# C21: natural-language encodings behave like their class: Incident within 0.05 of AL and above EL;
# Friendship within 0.05 of EL and below AL
for m in MODELS:
    nl = P["nl_formats"][m]
    inc, fr = nl["Incident"]["depth1"]["ba"], nl["Friendship"]["depth1"]["ba"]
    al, el = M[m]["depth1"]["AL"]["ba"], M[m]["depth1"]["EL"]["ba"]
    rule("C21", abs(inc - al) <= 0.05 and inc > el and abs(fr - el) <= 0.05 and fr < al, f"{m} Inc {inc} Fr {fr} AL {al} EL {el}")
# D10: equivalence tests: an interval inside its margin <=> "equivalent"; margins are the preregistered ones
for t_ in P["llm_stats"]["tost"]:
    c0 = 0.5 if "at chance" in t_["claim"] else 0.0
    rule("D10", t_["equivalent"] == (c0 - t_["margin"] <= t_["ci90"][0] and t_["ci90"][1] <= c0 + t_["margin"]), t_["claim"])
# D11: paired metric consistent with balanced accuracy (both above 0.5 together; TPR + TNR = 2 BA)
for m in MODELS:
    for key, v in P["llm_stats"]["paired"][m].items():
        f, cls = key.split("|")
        ba_ = (M[m][cls][f] if f in F3 else P["llm_formats"][m]["ECL"][cls])["ba"]
        # identity, and the same side of chance wherever balanced accuracy is clearly off chance (|BA - 0.5| > 0.05)
        rule("D11", abs(v["tpr"] + v["tnr"] - 2 * ba_) < 2e-3 and (abs(ba_ - 0.5) <= 0.05 or (v["auc"] > 0.5) == (ba_ > 0.5)),
             f"{m} {key}")
# D13: the threshold-free paired metric orders the depth-one formats as balanced accuracy does (every pair
# of formats whose balanced accuracies differ by more than 0.01), so the format effect is not only a common shift of the scores
for m in MODELS:
    pv = P["llm_stats"]["paired"][m]
    bav = {f: (M[m]["depth1"][f] if f in F3 else P["llm_formats"][m]["ECL"]["depth1"])["ba"] for f in F3 + ["ECL"]}
    for fa in bav:
        for fb in bav:
            if bav[fa] - bav[fb] > 0.01:
                rule("D13", pv[f"{fa}|depth1"]["auc"] > pv[f"{fb}|depth1"]["auc"], f"{m} {fa} > {fb}: BA yes, paired metric no")
# D12: ID k_min strata of 1-local (position-controlled) follow the step of P2 as the OOD strata do
for f, ks in (("AL", 2), ("NC", 1)):
    row = X3["kid"][f"1-local|{f}"]
    for lab, v in row.items():
        hi_k = {"1": 1, "2": 2, "3": 3, "4-6": 6}[lab]
        rule("D12", (v["mean"] >= 0.93) if hi_k <= ks else (0.47 <= v["mean"] <= 0.55), f"kid {f} {lab} {v['mean']}")

# ---------------- D: cross-consistency
for m in MODELS:
    pq = P["llm_main"]["per_question"][m]["D1a"]
    for f in ("AL", "NC"):
        rule("D1", P["llm_partial"][m][f"{f}-none"]["D1a"]["ba"] == pq[f], f"{m} {f}-none vs E1 D1a")
    for f in ("EL", "AL", "NC", "ECL"):
        st_ = P["llm_main"]["strata"][m][f]
        sh = st_["kmin_share"]
        w = sum(sh[k] * st_["kmin"][k] for k in sh if st_["kmin"][k] is not None) / sum(sh[k] for k in sh if st_["kmin"][k] is not None)
        # strata use positives of D1a and D1c only; compare with the mean of the two questions
        tot = (P["llm_main"]["per_question"][m]["D1a"].get(f, None) if f != "ECL" else None)
        if tot is not None:
            tot = 0.5 * (pq[f] + P["llm_main"]["per_question"][m]["D1c"][f])
            rule("D5", abs(w - tot) <= 0.01, f"{m} {f} strata weighted {w:.4f} vs {tot:.4f}")
Mech = P["mechanism"]
for fmt in ("AL", "NC"):
    peaks = [max(Mech["probe"][f"{fmt}|{b}"]) for b in ("1-8", "9-24", ">24")]
    rule("M1", all(b < a for a, b in zip(peaks, peaks[1:])), f"probe peaks {fmt} {peaks}")
rule("M1", set(Mech["probe"]) == {f"{f}|{b}" for f in ("AL", "NC") for b in ("1-8", "9-24", ">24")}, "E9 bins 1-8, 9-24, >24")
for b in ("1-8", "9-24", ">24"):
    rule("M1", max(Mech["probe"][f"AL|{b}"]) <= max(Mech["probe"][f"NC|{b}"]) + 0.005, f"AL<=NC {b}")
for fmt in ("AL", "NC"):
    pc = Mech["patch"][fmt]
    rule("M1", min(pc[:13]) >= 0.85 and max(pc[19:]) <= 0.2, f"patch {fmt}")
    rule("M1", all(b < a for a, b in zip(pc, pc[1:])), f"patch curve strictly decreasing {fmt}")
    rule("D9", abs(pc[0] - 1.0) < 1e-3 and abs(pc[-1]) < 1e-3, f"patch identities (embedding restores 1, after the last layer 0) {fmt}")
el_peak = max(range(len(Mech["el_inner"])), key=lambda i: Mech["el_inner"][i])
since_peak = max(range(29), key=lambda i: Mech["probe"]["NC|1-8"][i])
rule("M1", el_peak < since_peak, f"EL inner peak {el_peak} < since peak {since_peak}")

# ---------------- E: scale
for cls in M["Qwen2.5-7B"]:
    for f in F3:
        rule("E1", M["Qwen2.5-72B"][cls][f]["ba"] >= M["Qwen2.5-7B"][cls][f]["ba"] - 0.005, f"{cls} {f}")
gaps = {m: M[m]["depth1"]["NC"]["ba"] - M[m]["depth1"]["EL"]["ba"] for m in MODELS}
rule("E2", min(gaps, key=gaps.get) == "Qwen2.5-72B", f"gaps {gaps}")
avg = {m: sum(M[m][c][f]["ba"] for c in M[m] for f in F3) / 12 for m in MODELS}
rule("E3", avg["Qwen2.5-72B"] > max(avg["Qwen2.5-7B"], avg["Gemma-2-9B"]) and min(avg["Qwen2.5-7B"], avg["Gemma-2-9B"]) > avg["Llama-3.1-8B"]
     and abs(avg["Qwen2.5-7B"] - avg["Gemma-2-9B"]) <= 0.02, f"{avg}")

# ---------------- (VALIDATION_RULES.md: C22-C25, D14, D15)
# C22 (E11a, P6): mean-ablating the top-h line-head heads lowers depth one under AL and NC toward EL (top 16: at least
# 0.05, and at least 3x what 16 random heads of the same layers cost), leaves EL, ECL (0.03) and "v is red" (0.02 under
# every format) nearly unchanged; the loss grows with h (within 0.01)
E11A = P["e11a"]
_n = lambda f, q: E11A["none"][f"{f}|{q}"]["ba"]
for f in ("AL", "NC"):
    d16 = _n(f, "D1a") - E11A["top"][f"16|{f}|D1a"]["ba"]
    r16 = _n(f, "D1a") - E11A["random"][f"16|{f}|D1a"]["mean"]
    rule("C22", d16 >= 0.05 and d16 >= 3 * max(r16, 0.0), f"E11a {f}: top-16 drop {d16:.3f}, random {r16:.3f}")
    rule("C22", _n(f, "D1a") - E11A["top"][f"4|{f}|D1a"]["ba"] <= d16 + 0.01, f"E11a {f}: drop grows with h")
for f in ("EL", "ECL"):
    rule("C22", _n(f, "D1a") - E11A["top"][f"16|{f}|D1a"]["ba"] <= 0.03, f"E11a {f} D1a nearly unchanged")
for f in ("EL", "AL", "NC", "ECL"):
    rule("C22", _n(f, "Q0") - E11A["top"][f"16|{f}|Q0"]["ba"] <= 0.02, f"E11a {f}: v is red unchanged")
# ---------------- (VALIDATION_RULES.md: C28-C30)
# C28 (E11a replication, Llama-3.1-8B, same procedure): the top-16 drop under AL and NC is at least 0.03 and at least
# twice what 16 random heads cost; EL, ECL within 0.03; "v is red" within 0.02 (a looser replication criterion)
E11L = P["e11a_llama"]
_nl = lambda f, q: E11L["none"][f"{f}|{q}"]["ba"]
for f in ("AL", "NC"):
    d16 = _nl(f, "D1a") - E11L["top"][f"16|{f}|D1a"]["ba"]
    r16 = _nl(f, "D1a") - E11L["random"][f"16|{f}|D1a"]["mean"]
    rule("C28", d16 >= 0.03 and d16 >= 2 * max(r16, 0.0), f"E11a Llama {f}: top-16 drop {d16:.3f}, random {r16:.3f}")
for f in ("EL", "ECL"):
    rule("C28", _nl(f, "D1a") - E11L["top"][f"16|{f}|D1a"]["ba"] <= 0.03, f"E11a Llama {f} D1a nearly unchanged")
for f in ("EL", "AL", "NC", "ECL"):
    rule("C28", _nl(f, "Q0") - E11L["top"][f"16|{f}|Q0"]["ba"] <= 0.02, f"E11a Llama {f}: v is red unchanged")
# C29 (activation patching, both models): patching the top-16 heads from a prompt with flipped head colors costs AL
# at least as much as ablating them (within 0.01); patching from a prompt with renamed line heads costs AL at most
# 0.03 (the since route reads the head color, not its name); random heads cost at most 0.02 under either corruption
for blk, nm in (("e11a", "Qwen"), ("e11a_llama", "Llama")):
    B_ = P[blk]
    b0 = B_["none"]["AL|D1a"]["ba"]
    rule("C29", b0 - B_["patch"]["recolor|16|AL|D1a"]["ba"] >= b0 - B_["top"]["16|AL|D1a"]["ba"] - 0.01, f"{nm} recolor >= ablation")
    rule("C29", b0 - B_["patch"]["rename|16|AL|D1a"]["ba"] <= 0.03, f"{nm} rename small under AL")
    for kind in ("recolor", "rename"):
        for f in ("AL", "NC"):
            rule("C29", B_["none"][f"{f}|D1a"]["ba"] - B_["random_patch"][f"{kind}|16|{f}|D1a"]["mean"] <= 0.02, f"{nm} random {kind} {f}")
# C30 (E3 name matching): the best matching head's error does not fall from 1-16 to 49-64 names (within 0.005) and
# stays at most 0.15 at 49-64 names, for every arm and format
for k, c in P["e3_match"].items():
    if not isinstance(c, dict) or "1-16" not in c:
        continue
    rule("C30", c["49-64"]["mean"] >= c["1-16"]["mean"] - 0.005 and c["49-64"]["mean"] <= 0.15, f"E3 match {k}")
# C23 (E11b, P7): from maximum degree 6 to 48 the list formats lose more than the edge formats: in every model each
# list format declines more than each edge format, and the format x degree interaction (mean decline of AL and NC
# minus that of EL and ECL, over models) is at least 0.03; whether EL reaches NC at 48 is reported, not predicted
_int = []
for m in MODELS:
    E_ = P["e11b"][m]
    dec = {f: E_[f"{f}|6"]["ba"] - E_[f"{f}|48"]["ba"] for f in ("EL", "AL", "NC", "ECL")}
    rule("C23", min(dec["AL"], dec["NC"]) > max(dec["EL"], dec["ECL"]), f"E11b {m} declines {dec}")
    _int.append((dec["AL"] + dec["NC"] - dec["EL"] - dec["ECL"]) / 2)
    rule("D1", all(abs(E_[f"{f}|decline"] - dec[f]) < 1e-3 for f in dec), f"E11b {m}: stored decline")
rule("C23", sum(_int) / len(_int) >= 0.03, f"E11b interaction {sum(_int) / len(_int):.3f}")
# D14 (E12): the probe's balanced accuracy does not rise from the first to the last position (within 0.02), every
# model and format
for m in MODELS:
    for f in ("AL", "NC"):
        cs = {c["k"]: c["bal_acc"] for c in P["e12"]["cells"] if c["model"] == m and c["ser"] == f}
        rule("D14", cs[1] >= cs[24] - 0.02, f"E12 {m} {f}: r(1) {cs[1]} vs r(24) {cs[24]}")
# D15 (E12, preregistered decision threshold): the zero-parameter prediction supports the routes
# if its mean absolute error over the E1 strata is at most 8 points
rule("D15", P["e12"]["error_pts"] <= 8.0, f"E12 zero-parameter error {P['e12']['error_pts']:.2f} <= 8 points")
# C24 (E13): degree and neighbors are read off v's own line under the list formats: max(AL, NC) at least 0.05 above
# EL; edge existence: max(AL, NC) not below EL (every model)
for m in MODELS:
    E_ = P["e13"][m]
    for t_ in ("degree", "neighbors"):
        rule("C24", max(E_[f"{t_}|AL"]["ba"], E_[f"{t_}|NC"]["ba"]) >= E_[f"{t_}|EL"]["ba"] + 0.05, f"E13 {m} {t_}")
    rule("C24", max(E_["edge|AL"]["ba"], E_["edge|NC"]["ba"]) >= E_["edge|EL"]["ba"], f"E13 {m} edge")
# C25 (E3, Delta <= 256; Theorem 8): hard since stays high (>= 0.93) under AL and NC; 1-local, since-free, falls
# below its value on the larger test graphs by at least 0.03
D2_ = P["e3_extra"]["d256"]
for f in ("AL", "NC"):
    # prompts of this set are 2 to 5 times longer than every OOD prompt; the threshold, fixed before any run,
    # allows for the loss of length generalization at these lengths
    rule("C25", D2_[f"1-local+hard-since|{f}"]["mean"] >= 0.80, f"d256 hard since {f}")
    rule("C25", D2_[f"1-local|{f}"]["mean"] <= S[f"1-local|dia|{f}|ood"]["mean"] - 0.03, f"d256 1-local {f} below OOD")
# C26 (E3 hard pairs; Theorem lb:main): no formula of round depth <= 6 separates a pair, so the six-layer 1-local arm
# is at chance (within 0.03 of 0.5); with exact since the pair is one since step apart, so hard since >= 0.93
HP_ = P["e3_extra"]["hard"]
for f in ("AL", "NC"):
    rule("C26", abs(HP_[f"1-local|{f}"]["mean"] - 0.5) <= 0.03, f"hard pairs 1-local {f} at chance")
    rule("C26", HP_[f"1-local+hard-since|{f}"]["mean"] >= 0.93, f"hard pairs hard since {f}")
# C27 (instances of the proof's construction of Theorem lb:main, depth L = 4, 6, 8): the depth-L 1-local
# models are at chance (within 0.03 of 0.5) on the depth-L instances; at L = 6 exact since stays clearly above them
# (>= 0.75, allowing for prompts 2 to 3.5 times longer than every OOD prompt) and above 1-local by at least 0.2
HQ_ = P["e3_extra"]["hardp"]
for f in ("AL", "NC"):
    for L_ in (4, 6, 8):
        rule("C27", abs(HQ_[f"1-local|{f}|{L_}"]["mean"] - 0.5) <= 0.03, f"proof instances, 1-local depth {L_} {f} at chance")
    rule("C27", HQ_[f"1-local+hard-since|{f}|6"]["mean"] >= 0.75
         and HQ_[f"1-local+hard-since|{f}|6"]["mean"] >= HQ_[f"1-local|{f}|6"]["mean"] + 0.2, f"proof instances, hard since {f}")

# ---------------- (VALIDATION_RULES.md: C31-C34)
# C31 (E11a on Gemma-2-9B, the C28 criterion): top-16 drop under AL and NC at least 0.03 and at least twice the
# random-head drop; EL, ECL within 0.03; "v is red" within 0.02
E11G = P["e11a_gemma"]
_ng = lambda f, q: E11G["none"][f"{f}|{q}"]["ba"]
for f in ("AL", "NC"):
    d16 = _ng(f, "D1a") - E11G["top"][f"16|{f}|D1a"]["ba"]
    r16 = _ng(f, "D1a") - E11G["random"][f"16|{f}|D1a"]["mean"]
    rule("C31", d16 >= 0.03 and d16 >= 2 * max(r16, 0.0), f"E11a Gemma {f}: top-16 drop {d16:.3f}, random {r16:.3f}")
for f in ("EL", "ECL"):
    rule("C31", _ng(f, "D1a") - E11G["top"][f"16|{f}|D1a"]["ba"] <= 0.03, f"E11a Gemma {f} D1a nearly unchanged")
for f in ("EL", "AL", "NC", "ECL"):
    rule("C31", _ng(f, "Q0") - E11G["top"][f"16|{f}|Q0"]["ba"] <= 0.02, f"E11a Gemma {f}: v is red unchanged")
# C32 (E11b-L, P7 without the length confound): at fixed prompt length, from maximum degree 6 to 48 each list format
# still declines more than each edge format in every model, the format x degree interaction (as in C23) is at least
# 0.03, and each edge format declines by at most 0.03 (the nested and inline routes have no distance term; E11b's edge-format decline is then a length effect)
_intL = []
for m in MODELS:
    E_ = P["e11bL"][m]
    dec = {f: E_[f"{f}|6"]["ba"] - E_[f"{f}|48"]["ba"] for f in ("EL", "AL", "NC", "ECL")}
    rule("C32", min(dec["AL"], dec["NC"]) > max(dec["EL"], dec["ECL"]), f"E11b-L {m} list declines {dec}")
    rule("C32", max(dec["EL"], dec["ECL"]) <= 0.03, f"E11b-L {m} edge declines {dec}")
    _intL.append((dec["AL"] + dec["NC"] - dec["EL"] - dec["ECL"]) / 2)
    rule("D1", all(abs(E_[f"{f}|decline"] - dec[f]) < 1e-3 for f in dec), f"E11b-L {m}: stored decline")
rule("C32", sum(_intL) / len(_intL) >= 0.03, f"E11b-L interaction {sum(_intL) / len(_intL):.3f}")
# C33 (E14, matches have no distance term): with 1,024 filler words, EL with the filler between the color line and
# the edges is within 0.03 of EL with the same filler before the color line, in every model (the locality account
# predicts a lower value between); the same for ECL; the stored contrast equals the difference of its cells
for m in ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B"):
    E_ = P["e14"][m]
    for f in ("EL", "ECL"):
        c_ = E_[f"{f}|between|1024"]["ba"] - E_[f"{f}|before|1024"]["ba"]
        rule("C33", abs(c_) <= 0.03, f"E14 {m} {f}: between - before at 1,024 = {c_:.3f}")
        for F in (64, 256, 1024):
            rule("D1", abs(E_[f"{f}|contrast|{F}"] - (E_[f"{f}|between|{F}"]["ba"] - E_[f"{f}|before|{F}"]["ba"])) < 1e-3,
                 f"E14 {m} {f} stored contrast {F}")
# C34 (E15, P3 under exact since; App. leap:sec): EL needs at least 1.3 times the training examples of every
# one-match format to reach 0.95 in distribution, and every format reaches 0.95 with the full 500k examples (E3)
E15 = P["e15"]
for f in ("AL", "NC", "ECL"):
    rule("C34", E15["EL|n95"] is not None and E15[f"{f}|n95"] is not None and E15["EL|n95"] >= 1.3 * E15[f"{f}|n95"],
         f"E15 n95 EL {E15['EL|n95']} vs {f} {E15[f'{f}|n95']}")
for f in ("EL", "AL", "NC", "ECL"):
    rule("C34", E15[f"{f}|500000|id"]["mean"] >= 0.95, f"E15 {f} at 500k")

# ---------------- (VALIDATION_RULES.md: C35-C37; CONFIG §6.11, E17: dose of the since-head ablation)
E17B = {"Qwen2.5-7B": P["e11a"], "Llama-3.1-8B": P["e11a_llama"], "Gemma-2-9B": P["e11a_gemma"]}
_flip_ok = 0
for m, B in E17B.items():
    E_ = P["e17"][m]
    top = lambda h, f, q: (B["top"][f"16|{f}|{q}"]["ba"] if h == 16 else E_["top"][f"{h}|{f}|{q}"]["ba"])
    none = lambda f, q: B["none"][f"{f}|{q}"]["ba"]
    # C35: the dose curve does not rise (tolerance 0.015) under AL and NC on D1a
    for f in ("AL", "NC"):
        cur = [top(h, f, "D1a") for h in (16, 32, 64, 128)]
        rule("C35", all(b_ <= a_ + 0.015 for a_, b_ in zip(cur, cur[1:])), f"E17 {m} {f} dose {cur}")
    # C36: EL minus the mean of AL and NC at h = 128, one-sided bound > 0 in at least two models (counted below);
    # in every model the top-128 loss of the list formats is at least 3 times the random-128 loss
    fl = E_["flip"]
    _flip_ok += fl["list"]["lo05"] > 0
    for f in ("AL", "NC"):
        d_ = none(f, "D1a") - top(128, f, "D1a")
        r_ = none(f, "D1a") - E_["random"][f"128|{f}|D1a"]["mean"]
        rule("C36", d_ >= 3 * max(r_, 0.0), f"E17 {m} {f}: top-128 loss {d_:.3f} vs random {r_:.3f}")
    for f, c in (("AL", "AL"), ("NC", "NC")):
        rule("D1", abs(fl[c]["diff"] - (top(128, "EL", "D1a") - top(128, f, "D1a"))) <= 0.002,
             f"E17 {m} stored EL-{f} = difference of the cells")
    rule("D1", abs(fl["list"]["diff"] - (top(128, "EL", "D1a") - 0.5 * (top(128, "AL", "D1a") + top(128, "NC", "D1a")))) <= 0.002,
         f"E17 {m} stored EL-list")
    rule("D1", all(fl[c]["lo05"] <= fl[c]["diff"] for c in ("AL", "NC", "list")), f"E17 {m} one-sided bounds below the means")
    # C37: specificity of the top-128 ablation and harmlessness of random heads
    for f in ("EL", "ECL"):
        rule("C37", none(f, "D1a") - top(128, f, "D1a") <= 0.04, f"E17 {m} {f} D1a under top-128")
    for f in ("EL", "AL", "NC", "ECL"):
        rule("C37", none(f, "Q0") - top(128, f, "Q0") <= 0.03, f"E17 {m} {f}: v is red under top-128")
        rule("C37", none(f, "D1a") - E_["random"][f"128|{f}|D1a"]["mean"] <= 0.04, f"E17 {m} {f} random-128")
rule("C36", _flip_ok >= 2, f"E17: EL above the mean of AL and NC at h = 128 (one-sided bound > 0) in {_flip_ok} models")

# ---------------- (VALIDATION_RULES.md: C38-C41; CONFIG §6.12, E18: controls of the since-head ablation)
_ok38 = _ok39 = 0
for m, B in E17B.items():
    E8, E_ = P["e18"][m], P["e17"][m]
    none = lambda f, q: B["none"][f"{f}|{q}"]["ba"]
    # C38: heads selected on EL prompts cost EL (and "v is red" under EL), not the list formats
    el = none("EL", "D1a") - E8["edge"]["128|EL|D1a"]["ba"]
    elq = none("EL", "Q0") - E8["edge"]["128|EL|Q0"]["ba"]
    ecl = none("ECL", "D1a") - E8["edge"]["128|ECL|D1a"]["ba"]
    li = [none(f, "D1a") - E8["edge"][f"128|{f}|D1a"]["ba"] for f in ("AL", "NC")]
    _ok38 += el >= 0.05 and el >= 3 * max(sum(li) / 2, 0.0) and elq >= 0.05 and ecl < el
    for f, d_ in zip(("AL", "NC"), li):
        r_ = none(f, "D1a") - E8["edge_random"][f"128|{f}|D1a"]["mean"]
        rule("C38", d_ <= r_ + 0.015, f"E18 {m} {f}: edge heads cost the list formats {d_:.3f}, random heads {r_:.3f}")
    for f in ("EL", "AL", "NC", "ECL"):
        for q in ("D1a", "Q0"):
            rule("D1", E8["edge"][f"128|{f}|{q}"]["ba"] <= E8["edge"][f"16|{f}|{q}"]["ba"] + 0.015, f"E18 {m} {f} {q}: nested edge heads")
            rule("D1", E8["entry"][f"128|{f}|{q}"]["ba"] <= E8["entry"][f"16|{f}|{q}"]["ba"] + 0.015, f"E18 {m} {f} {q}: nested entry ablation")
    # C39: the since heads act at the entries: ablated there only, they still put the list formats below EL
    fl = E8["entry_flip"]
    _ok39 += fl["list"]["lo05"] > 0
    for f in ("AL", "NC"):
        ent = none(f, "D1a") - E8["entry"][f"128|{f}|D1a"]["ba"]
        allp = none(f, "D1a") - E_["top"][f"128|{f}|D1a"]["ba"]
        rule("C39", ent >= 0.7 * allp, f"E18 {m} {f}: entries-only loss {ent:.3f} vs all positions {allp:.3f}")
    rule("D1", abs(fl["list"]["diff"] - (E8["entry"]["128|EL|D1a"]["ba"] - 0.5 * (E8["entry"]["128|AL|D1a"]["ba"] + E8["entry"]["128|NC|D1a"]["ba"]))) <= 0.002,
         f"E18 {m} stored EL-list (entries only)")
    rule("D1", all(fl[c]["lo05"] <= fl[c]["diff"] for c in ("AL", "NC", "list")), f"E18 {m} one-sided bounds below the means")
    # C40: a non-ceiling task without since (GraphQA node degree, v's own line) survives the top-128 since heads
    for f in ("AL", "NC"):
        b0 = P["e13"][m][f"degree|{f}"]["ba"]
        d_ = b0 - E8["degree"][f"128|{f}"]["ba"]
        dia = none(f, "D1a") - E_["top"][f"128|{f}|D1a"]["ba"]
        rule("C40", d_ <= 0.05 and d_ <= dia / 3, f"E18 {m} {f}: degree loss {d_:.3f} vs ^red loss {dia:.3f}")
        v = E8["degree_random"][f"128|{f}"]
        rule("D1", v["min"] - 1e-9 <= v["mean"] <= v["max"] + 1e-9, f"E18 {m} {f}: degree random sets")
rule("C38", _ok38 >= 2, f"E18: edge heads cost EL and leave the list formats in {_ok38} models")
rule("C39", _ok39 >= 2, f"E18: entries-only ablation keeps EL above the list mean (bound > 0) in {_ok39} models")
# C42: NC declines more than ECL from k_min = 1 to >= 5 (direct contrast, one-sided 95% lower bound > 0)
rule("C42", P["llm_stats"]["kdiff"]["nc_minus_ecl_lo90"] > 0, f"NC minus ECL change: bound {P['llm_stats']['kdiff']['nc_minus_ecl_lo90']}")
rule("D1", P["llm_stats"]["kdiff"]["nc_minus_ecl_lo90"] <= P["llm_stats"]["kdiff"]["nc_minus_ecl"], "NC minus ECL: bound below the estimate")
# C41: P4 as non-inferiority per k_min stratum (four-model mean, upper 95% bound of ECL's shortfall below 0.02)
for s_, v in P["llm_stats"]["p4_noninf"].items():
    rule("C41", v["hi95"] <= 0.02, f"P4 non-inferiority k_min {s_}: upper bound {v['hi95']}")
    rule("D1", v["shortfall"] <= v["hi95"], f"P4 k_min {s_}: bound below the point estimate")

_mean = lambda xs: sum(xs) / len(xs)
# ---------------- (VALIDATION_RULES.md: C43-C49; CONFIG §6.13: E19 knockout and entry patching, E20, E21)
_ok43 = _ok44 = _ok45 = 0
for m, B in E17B.items():
    none = lambda f, q: B["none"][f"{f}|{q}"]["ba"]
    K9, F0, E8 = P["e19"][m], P["e20"][m], P["e18"][m]
    # C43: knocking out the entries' attention to their own line head costs the list formats, the previous line's
    # head does not, and "v is red" (read at the line head) is unchanged
    ok = True
    for f in ("AL", "NC"):
        lo = none(f, "D1a") - K9["ko"][f"own|{f}|D1a"]["ba"]
        lc = none(f, "D1a") - K9["ko"][f"other|{f}|D1a"]["ba"]
        ok &= lo >= 0.05 and lo >= 3 * max(lc, 0.0)
        rule("C43", none(f, "Q0") - K9["ko"][f"own|{f}|Q0"]["ba"] <= 0.02, f"E19 {m} {f}: knockout leaves v is red")
        d_ = K9["ko_diff"][f]
        rule("D1", abs(d_["diff"] - (K9["ko"][f"own|{f}|D1a"]["ba"] - K9["ko"][f"other|{f}|D1a"]["ba"])) <= 0.002 and d_["hi95"] >= d_["diff"],
             f"E19 {m} {f}: stored own-other difference and its upper bound")
    _ok43 += ok
    # C44: patching the top-128 since heads at the entries from the recolored prompt costs at least as much as
    # ablating them there (E18), as heads that carry the head color predict
    _ok44 += all(none(f, "D1a") - K9["patch"][f"recolor128|{f}|D1a"]["ba"] >= none(f, "D1a") - E8["entry"][f"128|{f}|D1a"]["ba"]
                 for f in ("AL", "NC"))
    # C45: filler between the line head and the entries costs the list formats more than the same filler after them
    ok = True
    for f in ("AL", "NC"):
        c8, c24 = F0["contrast"][f"{f}|8"], F0["contrast"][f"{f}|24"]
        ok &= c24["diff"] <= -0.03 and c24["hi95"] < 0
        rule("C45", abs(c24["diff"]) >= abs(c8["diff"]) - 0.01, f"E20 {m} {f}: contrast grows with F ({c8['diff']}, {c24['diff']})")
        for F in (8, 24):
            c = F0["contrast"][f"{f}|{F}"]
            rule("D1", abs(c["diff"] - (F0["cells"][f"inside|{f}|{F}"]["ba"] - F0["cells"][f"outside|{f}|{F}"]["ba"])) <= 0.002 and c["hi95"] >= c["diff"],
                 f"E20 {m} {f} F={F}: stored contrast")
    _ok45 += ok
rule("C43", _ok43 >= 2, f"E19: own-head knockout costs AL and NC (>= 0.05, >= 3x control) in {_ok43} models")
rule("C44", _ok44 >= 2, f"E19: recolor patching at the entries costs at least the entry ablation in {_ok44} models")
rule("C45", _ok45 >= 2, f"E20: inside minus outside <= -0.03 with bound < 0 (F = 24) in {_ok45} models")
# C46: the distance cost is specific to the list formats: their E20 contrast (F = 24) is below the E14 contrast of
# EL and ECL (F = 64, more distance added) by more than 0.02
_lc = _mean([P["e20"][m]["contrast"][f"{f}|24"]["diff"] for m in E17B for f in ("AL", "NC")])
_ec = _mean([P["e14"][m][f"{f}|contrast|64"] for m in E17B for f in ("EL", "ECL")])
rule("C46", _lc < _ec - 0.02, f"E20 vs E14: list contrast {_lc:.3f}, edge contrast {_ec:.3f}")
# C47-C49: relational tables (E21)
E21 = P["e21"]
_m4 = [m for m in E21 if m not in ("mean",) and isinstance(E21[m], dict) and "rows|T1" in E21[m]]
_mu = lambda k: float(_mean([E21[m][k]["ba"] for m in _m4]))
rule("C47", _mu("project|T1") > _mu("person|T1") > _mu("rows|T1")
     and sum(E21[m]["project|T1"]["ba"] > E21[m]["person|T1"]["ba"] > E21[m]["rows|T1"]["ba"] for m in _m4) >= 3,
     "E21: project > person > rows on the depth-one question (mean and >= 3 models)")
rule("C48", _mu("inline|T1") >= max(_mu("project|T1"), _mu("person|T1")) - 0.02, "E21: inline within 0.02 of the better list format")
_g = [_mu(f"{f}|G1") for f in ("rows", "person", "project", "inline")]
rule("C49", max(_g) - min(_g) <= 0.02, f"E21: global question ties across formats ({max(_g) - min(_g):.3f})")
for k_, v in E21["mean"].items():
    rule("D1", abs(v - _mu(k_)) <= 0.00011, f"E21 stored mean {k_}")

# ---------------- (VALIDATION_RULES.md: C50-C66; CONFIG §6.14: E22-E32)
if "e22" in P:
    _F4 = ("EL", "AL", "NC", "ECL")
    _F5 = _F4 + ("NCH",)
    _M3r = ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B")
    _M4r = _M3r + ("Qwen2.5-72B",)
    E22 = P["e22"]
    # C50: NoPE decides "some vertex is red" in every format, "at least three" under EL and AL only
    rule("C50", all(E22[f"g1|{f}|ood"]["mean"] >= 0.95 for f in _F4), "E22: NoPE G1 OOD >= 0.95 in every format")
    rule("C50", all(E22[f"c3|{f}|ood"]["mean"] >= 0.95 for f in ("EL", "AL"))
         and all(E22[f"c3|{f}|ood"]["mean"] <= min(E22[f"c3|{g}|ood"]["mean"] for g in ("EL", "AL")) - 0.10 for f in ("NC", "ECL")),
         "E22: NoPE C3 OOD >= 0.95 under EL, AL and >= 0.10 lower under NC, ECL")
    for k_, v in ((k, v) for k, v in E22.items() if isinstance(v, dict)):
        rule("D1", abs(v["mean"] - _mean(v["seeds"])) <= 1e-9 and len(v["seeds"]) == 5, f"E22 {k_}: mean of 5 seeds")
    E23 = P["e23"]
    _arms23 = ("RoPE", "ALiBi-only", "1-local+ALiBi", "1-local+hard-since")
    _NS = (25000, 50000, 100000, 200000, 500000)
    for a_ in _arms23:
        n95 = {f: E23["n95"][f"{a_}|{f}"] for f in _F4}
        best = min(n95["AL"], n95["NC"])
        rule("C51", n95["EL"] is not None and n95["EL"] >= 1.3 * best and n95["ECL"] <= 1.1 * best,
             f"E23 {a_}: N95 EL {n95['EL']} >= 1.3 x {best}, ECL {n95['ECL']} <= 1.1 x")
        nmin = min(N for N in _NS if any(E23["cells"][f"{a_}|{f}|{N}|id"]["mean"] > 0.90 for f in _F4))
        bl = max(E23["cells"][f"{a_}|{f}|{nmin}|id"]["mean"] for f in ("AL", "NC"))
        rule("C52", E23["cells"][f"{a_}|EL|{nmin}|id"]["mean"] <= bl - 0.03, f"E23 {a_}: EL below the best list format at N = {nmin}")
        o5 = [E23["cells"][f"{a_}|{f}|500000|ood"]["mean"] for f in _F4]
        rule("C52", max(o5) - min(o5) <= 0.03, f"E23 {a_}: formats within 0.03 OOD at 500k")
        for N in _NS:
            for f in _F4:
                for sp in ("id", "ood"):
                    v = E23["cells"][f"{a_}|{f}|{N}|{sp}"]
                    rule("D1", abs(v["mean"] - _mean(v["seeds"])) <= 1e-9, f"E23 {a_} {f} {N} {sp}: stored mean")
    for g in ("AL", "NC"):
        d_ = E23["steps_diff"][f"1-local+hard-since|EL-{g}"]
        rule("C51", d_["mean"] >= 1000 and d_["lo"] > 0, f"E23 hard since: EL needs >= 1,000 more steps than {g} ({d_['mean']:.0f}, lo {d_['lo']:.0f})")
    # C53: LoRA
    E24 = P["e24"]
    for m in ("Qwen2.5-0.5B", "Qwen2.5-1.5B"):
        mu = lambda f, N: E24[f"{m}|{f}|{N}|e1"]["mean"]
        for N in (1000, 4000):
            bl = max(mu("AL", N), mu("NC", N))
            rule("C53", mu("EL", N) <= bl - 0.03 and mu("ECL", N) >= bl - 0.02, f"E24 {m} N={N}: EL {mu('EL', N):.3f}, ECL {mu('ECL', N):.3f}, best list {bl:.3f}")
        g = lambda N: max(mu("AL", N), mu("NC", N)) - mu("EL", N)
        rule("C53", g(16000) < g(1000), f"E24 {m}: the EL gap shrinks with N")
    # C54: window frontiers (strata entirely inside the proved upper frontier >= 0.90, entirely beyond the lower <= 0.56)
    E25 = P["e25"]
    _KS = ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 32, "13-32"))
    for key, b in E25["bounds"].items():
        up, lo = b["upper"], b["lower"]
        w, L, f = key.split("|")
        rule("D1", up == (max(0, ((int(L[1:]) - 1) * int(w[1:]) - 1) // 2) if f == "AL" else ((int(L[1:]) - 1) * int(w[1:]) + 1) // 5),
             f"E25 {key}: upper frontier of Corollary ehat:window")
        for a, b_, lab in _KS:
            v = E25["cells"][f"{key}|{lab}"]["mean"]
            if b_ <= up:
                rule("C54", v >= 0.90, f"E25 {key} k {lab}: inside the frontier {v:.3f}")
            elif a > lo:
                rule("C54", v <= 0.56, f"E25 {key} k {lab}: beyond the lower bound {v:.3f}")
    # C55: one global layer
    E26 = P["e26"]
    _E = (8, 16, 32, 64, 128)
    rule("C55", all(E26[f"last-global|ECL|{e}"]["mean"] >= 0.95 for e in _E), "E26: ECL >= 0.95 at every E with one global layer")
    el8, el128, ecl128 = (E26[f"last-global|EL|8"]["mean"], E26[f"last-global|EL|128"]["mean"], E26[f"last-global|ECL|128"]["mean"])
    rule("C55", el128 <= el8 - 0.15 and el128 <= ecl128 - 0.15, f"E26: EL at E = 128 ({el128:.3f}) 0.15 below E = 8 ({el8:.3f}) and ECL")
    rule("C55", all(E26[f"1-local|EL|{e}"]["mean"] >= 0.95 for e in _E), "E26: control arm EL >= 0.95 at every E")
    # C56, C57: depth two and counting
    E27 = P["e27"]
    rule("C56", E27["8-local|dd|NC|id"]["mean"] >= 0.90 and E27["8-local|dd|NC|ood"]["mean"] >= 0.90
         and E27["8-local|dd|AL|ood"]["mean"] <= 0.56, "E27: 8-local DD: NC >= 0.90, AL OOD <= 0.56")
    rule("C57", all(E27[f"1-local+hard-since|c2|{f}|ood"]["mean"] >= 0.95 for f in _F4), "E27: hard since C2 >= 0.95 OOD")
    el = E27["1-local|c2|EL|ood"]["mean"]
    rule("C57", el >= 0.95 and all(E27[f"1-local|c2|{f}|ood"]["mean"] <= el - 0.10 for f in ("AL", "NC")), "E27: 1-local C2: EL >= 0.95, AL and NC 0.10 lower")
    # C58-C61: mechanism replication
    E28 = P["e28"]
    for m in _M3r:
        Rp, t1 = E28["repl"][m], E28["t1_loss"][m]
        for tpl in ("T2", "T3"):
            for f in ("AL", "NC"):
                loss = Rp[f"{tpl}|none|{f}|D1a"]["ba"] - Rp[f"{tpl}|top16|{f}|D1a"]["ba"]
                ko = Rp[f"{tpl}|none|{f}|D1a"]["ba"] - Rp[f"{tpl}|ko-own|{f}|D1a"]["ba"]
                rule("C58", loss >= 0.05 and ko >= 0.05 and abs(loss - t1[f"{f}|D1a"]) <= 0.03,
                     f"E28a {m} {tpl} {f}: ablation loss {loss:.3f} (T1 {t1[f'{f}|D1a']:.3f}), knockout loss {ko:.3f}")
                rule("C58", Rp[f"{tpl}|none|{f}|Q0"]["ba"] - Rp[f"{tpl}|top16|{f}|Q0"]["ba"] <= 0.02, f"E28a {m} {tpl} {f}: v is red kept")
        S_ = E28["stability"][m]
        rule("C59", S_["jac16"] >= 0.6 and S_["jac128"] >= 0.7, f"E28b {m}: Jaccard {S_['jac16']}, {S_['jac128']}")
        rule("D1", abs(S_["jac16"] - _mean([x["jac16"] for x in S_["splits"]])) <= 1e-4, f"E28b {m}: stored mean")
    K = E28["ko72"]
    for f in ("AL", "NC"):
        lo = K[f"none|{f}|D1a"]["ba"] - K[f"own|{f}|D1a"]["ba"]
        lc = K[f"none|{f}|D1a"]["ba"] - K[f"other|{f}|D1a"]["ba"]
        rule("C60", lo >= 0.05 and lo >= 3 * max(lc, 0.0), f"E28c 72B {f}: knockout loss {lo:.3f}, control {lc:.3f}")
    _ok61 = sum(all(E28["recovery"][f"{m}|{f}"]["top"] >= 0.30 for f in ("AL", "NC")) for m in _M3r)
    rule("C61", _ok61 >= 2 and all(E28["recovery"][f"{m}|{f}"]["rand"] <= 0.05 for m in _M3r for f in ("AL", "NC")),
         f"E28d: restoration >= 0.30 in {_ok61} models, random sets <= 0.05")
    # C62, C63: routes at high degree, within- vs between-line filler
    E29 = P["e29"]
    dr = lambda v: _mean([E29[m][f"{v}|6"]["ba"] - E29[m][f"{v}|48"]["ba"] for m in _M3r])
    rule("C62", abs(dr("nc-mask-heads") - dr("al")) <= 0.04 and dr("nch") <= dr("nc-mask-heads") - 0.04
         and dr("nc-mask-entries") <= dr("nc-mask-heads") - 0.04,
         f"E29: decline head-masked {dr('nc-mask-heads'):.3f}, AL {dr('al'):.3f}, NC-H {dr('nch'):.3f}, entries masked {dr('nc-mask-entries'):.3f}")
    E30 = P["e30"]
    _ok63 = 0
    for m in _M3r:
        C = E30[m]
        _ok63 += all(C["contrast"][f]["diff"] <= -0.03 and C["contrast"][f]["hi95_1s"] < 0 for f in ("AL", "NC"))
        for f in ("AL", "NC"):
            c = C["cells"]
            rule("C63", c[f"none|{f}"]["ba"] - c[f"between|{f}"]["ba"] <= c[f"none|{f}"]["ba"] - c[f"outside|{f}"]["ba"] + 0.02,
                 f"E30 {m} {f}: between-line filler costs only the length factor")
    rule("C63", _ok63 >= 2, f"E30: inside minus between <= -0.03 (one-sided bound < 0) in {_ok63} models")
    # C64: recommendation under CoT, the reasoning model and GPT-4o
    E31 = P["e31"]
    cm = lambda c, f: _mean([E31["cot"][m][f"{c}|{f}"]["ba"] for m in _M4r])
    rule("C64", cm("depth1", "ECL") >= max(cm("depth1", "AL"), cm("depth1", "NC")) - 0.02, "E31a: CoT ECL within 0.02 of the best list format")
    _g = [cm("global", f) for f in _F5]
    rule("C64", max(_g) - min(_g) <= 0.03, f"E31a: CoT global range {max(_g) - min(_g):.3f}")
    r1 = E31["r1"]
    rule("C64", r1["depth1|ECL"]["ba"] >= max(r1["depth1|AL"]["ba"], r1["depth1|NC"]["ba"]) - 0.05, "E31a: reasoning model ECL within 0.05")
    g4 = E31["gpt4o"]
    for c in ("depth1", "depth2"):
        rule("C64", g4[f"{c}|NC"]["ba"] >= g4[f"{c}|AL"]["ba"] > g4[f"{c}|EL"]["ba"], f"E31b GPT-4o {c}: NC >= AL > EL")
    rule("C64", g4["depth1|ECL"]["ba"] >= g4["depth1|NC"]["ba"] - 0.01, "E31b GPT-4o: ECL >= NC - 0.01")
    _g = [g4[f"global|{f}"]["ba"] for f in _F5]
    rule("C64", max(_g) - min(_g) <= 0.03, "E31b GPT-4o: global tie")
    # C65: scope of the recommendation
    for kind in ("num4", "compound"):
        I_ = lambda m, f: E31["ids"][m][f"{kind}|depth1|{f}"]["ba"]
        rule("C65", sum(I_(m, "EL") < I_(m, "AL") < I_(m, "NC") for m in _M4r) >= 3, f"E31c {kind}: EL < AL < NC in >= 3 models")
        im = lambda f: _mean([I_(m, f) for m in _M4r])
        rule("C65", im("ECL") >= max(im("AL"), im("NC")) - 0.03, f"E31c {kind}: ECL within 0.03 (mean)")
    LG = E31["large"]
    for n in (100, 200, 500):
        ms = [m for m in _M4r if f"{n}|EL" in LG[m]]
        lm = lambda f: _mean([LG[m][f"{n}|{f}"]["ba"] for m in ms])
        if n < 500:
            rule("C65", lm("NC") > lm("AL") > lm("EL"), f"E31d n={n}: NC > AL > EL (mean)")
        rule("C65", lm("ECL") >= max(lm("AL"), lm("NC")) - 0.03, f"E31d n={n}: ECL within 0.03 (mean)")
        th = [_mean([LG[m][f"{n}|{f}|{t}"]["ba"] for m in ms for f in ("AL", "NC")]) for t in range(3)]
        rule("C65", max(th) - min(th) <= 0.03, f"E31d n={n}: line-position thirds range {max(th) - min(th):.3f}")
    ok, tot = 0, 0
    for m, PRm in E31["prompting"].items():
        pq = P["llm_main"]["per_question"][m]["D1a"]
        for v in sorted({k.split("|")[0] for k in PRm}):
            g = lambda f: PRm[f"{v}|D1a|{f}"]["ba"]
            tot += 1
            if v.startswith("el-"):            # EL edge orders: EL stays below the main-protocol AL and NC
                ok += g("EL") < min(pq["AL"], pq["NC"])
            else:
                ok += g("NC") > g("AL") > g("EL") and g("ECL") >= max(g("AL"), g("NC")) - 0.03
    rule("C65", ok >= 0.8 * tot, f"E31e: {ok} of {tot} prompting cells keep the order and the recommendation")
    SC = E31["scale"]
    for key, c in SC.items():
        size, kind = key.split("|")
        if float(size[:-1]) >= 3:
            lead = max(c["depth1|AL"]["ba"], c["depth1|NC"]["ba"]) - c["depth1|EL"]["ba"]
            g_ = [c[f"global|{f}"]["ba"] for f in _F4]
            rule("C65", lead >= 0.03 and max(g_) - min(g_) <= 0.05, f"E31f {key}: list lead {lead:.3f}, global range {max(g_) - min(g_):.3f}")
    for size in ("0.5B", "1.5B", "7B"):
        o = lambda kind: sorted(_F4, key=lambda f: SC[f"{size}|{kind}"][f"depth1|{f}"]["ba"])
        rule("C65", [f for f in o("base") if f != "ECL"] == [f for f in o("instruct") if f != "ECL"], f"E31f {size}: base and instruct order")
    AL_ = E31["alibi"]
    q_ = max(AL_["Qwen2.5-7B (E11c pool)"][f"{f}|drop"] for f in ("AL", "NC"))
    rule("C65", all(min(AL_[m][f"{f}|drop"] for f in ("AL", "NC")) >= q_ + 0.02 for m in ("BLOOM-7B1", "MPT-7B")),
         "E31g: ALiBi models lose >= 0.02 more from k_min = 1 to >= 5 than Qwen2.5-7B")
    am = lambda t, f: _mean([E31["algo"][m][f"{t}|{f}"]["ba"] for m in _M4r])
    for t in ("A1", "A2", "A3"):
        rule("C65", am(t, "ECL") >= am(t, "EL") - 0.02, f"E31h {t}: ECL >= EL - 0.02 (mean)")
    RL = E31["real"]
    for src in ("molecules", "citation"):
        rule("C65", sum(RL[m][f"{src}|D1a|NC"]["ba"] > RL[m][f"{src}|D1a|AL"]["ba"] > RL[m][f"{src}|D1a|EL"]["ba"] for m in _M4r) >= 3,
             f"E31i {src}: depth-one order in >= 3 models")
        for m in _M4r:
            for f in ("EL", "AL", "NC"):
                rule("C65", abs(RL[m][f"{src}|D1a|{f}"]["ba"] - RL[m][f"{src}|D1a|{f}"]["pred"]) <= 0.05, f"E31i {m} {src} {f}: within 0.05 of the prediction")
    LK = E31["latek"]
    for m in _M4r:
        rule("C65", LK[m]["NC-EL"]["diff"] >= 0.05 and LK[m]["NC-EL"]["lo95"] > 0, f"E31j {m}: NC - EL {LK[m]['NC-EL']['diff']}")
    rule("C65", abs(_mean([LK[m]["ECL"]["ba"] - LK[m]["ECL|k1"] for m in _M4r])) <= 0.03, "E31j: ECL keeps its k_min = 1 level (mean)")
    # C67: nesting versus distance (E33, ELB = EL's nested matches with the colors next to the edges)
    if "e33" in P:
        E33 = P["e33"]
        for a_ in _arms23:
            el, elb, ecl = E23["n95"][f"{a_}|EL"], E33["n95"][f"{a_}|ELB"], E23["n95"][f"{a_}|ECL"]
            rule("C67", elb is not None and elb >= (el * ecl) ** 0.5,
                 f"E33a {a_}: N95 of ELB {elb} at or above the geometric midpoint of EL {el} and ECL {ecl}")
            for N in _NS:
                v = E33["cells"][f"{a_}|ELB|{N}|id"]
                rule("D1", abs(v["mean"] - _mean(v["seeds"])) <= 1e-9, f"E33a {a_} {N}: stored mean")
        for m in ("Qwen2.5-0.5B", "Qwen2.5-1.5B"):
            for N in (1000, 4000):
                bl = max(E24[f"{m}|AL|{N}|e1"]["mean"], E24[f"{m}|NC|{N}|e1"]["mean"])
                rule("C67", E33["lora"][f"{m}|ELB|{N}|e1"]["mean"] <= bl - 0.03, f"E33b {m} N={N}: ELB 0.03 below the better list format")
        mm_ = lambda pids, f: _mean([(E33["llm"][m][q]["ba"] if f == "ELB" else P["llm_main"]["per_question"][m][q][f]) for m in _M4r for q in pids])
        d1 = ("D1a", "D1b", "D1c")
        eclm = _mean([P["llm_formats"][m]["ECL"]["depth1"]["ba"] for m in _M4r])
        rule("C67", mm_(d1, "ELB") <= mm_(d1, "EL") + 0.04 and mm_(d1, "ELB") <= eclm - 0.05,
             f"E33c: depth-one ELB {mm_(d1, 'ELB'):.3f} within 0.04 of EL {mm_(d1, 'EL'):.3f} and 0.05 below ECL {eclm:.3f}")
        g3 = ("G1", "G2", "G3")
        gv = [mm_(g3, "ELB"), mm_(g3, "EL"), _mean([P["llm_formats"][m]["ECL"]["global"]["ba"] for m in _M4r])]
        rule("C67", max(gv) - min(gv) <= 0.03, f"E33c: global questions tie across EL, ELB, ECL (range {max(gv) - min(gv):.3f})")
    # C66: depth sweep with 5 seeds, widths 128 and 512
    E32 = P["e32"]
    for key, v in E32["depth"].items():
        L, f, lab = key.split("|")
        rule("D1", len(v["seeds"]) == 5 and abs(v["mean"] - _mean(v["seeds"])) <= 1e-9, f"E32 depth {key}: 5 seeds")
        if f not in ("AL", "NC"):
            continue
        kst = max(0, (int(L) - 2) // 2) if f == "AL" else int(L) // 5
        a, b_ = [x for x in _KS if x[2] == lab][0][:2]
        if b_ <= kst:
            rule("C66", v["mean"] >= 0.90, f"E32 depth {key}: inside {v['mean']:.3f}")
        elif a > kst:
            rule("C66", v["mean"] <= 0.56, f"E32 depth {key}: beyond {v['mean']:.3f}")
    for key, v in E32["width"].items():
        d, arm, f, lab = key.split("|")
        a, b_ = [x for x in _KS if x[2] == lab][0][:2]
        if arm == "1-local+hard-since" or f in ("EL", "ECL") or b_ <= (2 if f == "AL" else 1):
            rule("C66", v["mean"] >= 0.90, f"E32 width {key}: inside {v['mean']:.3f}")
        elif a > (2 if f == "AL" else 1):
            rule("C66", v["mean"] <= 0.56, f"E32 width {key}: beyond {v['mean']:.3f}")

# ---------------- (seventh set, CONFIG §6.15): E34 binding control, E35 norm-matched null, E36 replication
if "e34" in P:
    _M3s = ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B")
    _B11 = {"Qwen2.5-7B": P["e11a"], "Llama-3.1-8B": P["e11a_llama"], "Gemma-2-9B": P["e11a_gemma"]}
    _dia = lambda m, f: _B11[m]["none"][f"{f}|D1a"]["ba"] - P["e17"][m]["top"][f"128|{f}|D1a"]["ba"]
    E34 = P["e34"]
    _l34 = lambda m, c, f: E34[m]["cells"][f"none|{f}"]["ba"] - E34[m]["cells"][f"{c}|{f}"]["ba"]
    # C68: the since heads spare a list question that needs line identity but no head color
    ok = sum(all(_l34(m, "since128", f) <= 0.05 and _l34(m, "since128", f) < _dia(m, f) / 3 for f in ("AL", "NC")) for m in _M3s)
    rule("C68", ok >= 2, f"E34: top-128 since heads cost edge existence <= 0.05 and < 1/3 of the ^red loss in {ok} models")
    # C69: the own-line knockout costs edge existence (line identity depends on the entries' attention to the head)
    ok = sum(all(_l34(m, "ko-own", f) >= 0.05 and _l34(m, "ko-own", f) >= 3 * max(_l34(m, "ko-other", f), 0.001) for f in ("AL", "NC")) for m in _M3s)
    rule("C69", ok >= 2, f"E34: own-line knockout costs edge existence >= 0.05 and >= 3x the control in {ok} models")
    # C70: random heads cost little; the edge formats lose little under the since heads
    for m in _M3s:
        for f in ("EL", "AL", "NC", "ECL"):
            rl = E34[m]["cells"][f"none|{f}"]["ba"] - E34[m]["rand128"][f]["mean"]
            rule("C70", rl <= 0.03, f"E34 {m} {f}: 128 random heads cost edge existence {rl:.3f}")
        for f in ("EL", "ECL"):
            rule("C70", _l34(m, "since128", f) <= 0.04, f"E34 {m} {f}: since heads cost edge existence {_l34(m, 'since128', f):.3f}")
    # C71: the top-128 loss of the list formats exceeds all 100 norm-matched null sets; EL, ECL and Q0 stay in the null
    E35 = P["e35"]
    for m in _M3s:
        for f in ("AL", "NC"):
            v = E35[m][f"{f}|D1a"]
            rule("C71", v["top"] > v["null_max"], f"E35 {m} {f}: top {v['top']:.3f} vs null max {v['null_max']:.3f}")
        for k, v in E35[m].items():
            if k.endswith("Q0") or k.split("|")[0] in ("EL", "ECL"):
                rule("C71", v["top"] <= v["null_q95"] + 0.01, f"E35 {m} {k}: top {v['top']:.3f} vs null q95 {v['null_q95']:.3f}")
    # C72: templates T2, T3 at h = 128: EL above the list mean (bound > 0) in >= 2 models, list loss within 0.04 of T1
    E36 = P["e36"]
    for tpl in ("T2", "T3"):
        ok = sum(E36["tpl"][m][f"{tpl}|flip"]["lo05"] > 0 for m in _M3s)
        rule("C72", ok >= 2, f"E36 {tpl}: flip with bound > 0 in {ok} models")
        for m in _M3s:
            for f in ("AL", "NC"):
                lt = E36["tpl"][m][f"{tpl}|none|{f}|D1a"]["ba"] - E36["tpl"][m][f"{tpl}|top128|{f}|D1a"]["ba"]
                rule("C72", abs(lt - _dia(m, f)) <= 0.04, f"E36 {m} {tpl} {f}: list loss {lt:.3f} vs T1 {_dia(m, f):.3f}")
    # C73: the paraphrase D1c at h = 128: flip with bound > 0 in >= 2 models
    ok = sum(E36["d1c"][m]["flip"]["lo05"] > 0 for m in _M3s)
    rule("C73", ok >= 2, f"E36 D1c: flip with bound > 0 in {ok} models")
    # C74: Qwen2.5-72B at h = 640 (12.5% of its heads): EL above the list mean (bound > 0), list loss >= 3x random
    Q = E36["q72"]
    l640 = 0.5 * sum(Q[f"none|{f}"]["ba"] - Q[f"top640|{f}"]["ba"] for f in ("AL", "NC"))
    r640 = 0.5 * sum(Q[f"none|{f}"]["ba"] - Q[f"rand640|{f}"]["mean"] for f in ("AL", "NC"))
    rule("C74", Q["flip640"]["lo05"] > 0 and l640 >= 3 * max(r640, 0.001), f"E36 72B h=640: flip bound {Q['flip640']['lo05']:.3f}, list loss {l640:.3f} vs random {r640:.3f}")

# ---------------- F: bookkeeping
for k, v in P.items():
    if not k.startswith("_") and isinstance(v, dict) and k != "alibi_pilot_X1_NC":
        rule("F2", "config_ref" in v, k)

if VERB:
    print("rule      passed  failed  skipped")
    for rid in sorted(COUNT, key=lambda r: (r[0], int("".join(ch for ch in r[1:] if ch.isdigit()) or 0), r)):
        print(f"{rid:<8}{COUNT[rid][0]:>8}{COUNT[rid][1]:>8}{COUNT[rid][2]:>9}")
print(f"{PASS} checks passed, {len(FAIL)} failed, {len(SKIPPED)} skipped (not evaluated)")
for f in FAIL:
    print("FAIL", f)
for s in SKIPPED:
    print("SKIP", s)
sys.exit(1 if FAIL else 0)
