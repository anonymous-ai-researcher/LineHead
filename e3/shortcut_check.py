"""Line-index shortcut check of the position-controlled E3 sets (proofs12, Prop. lb:top).

With the two query lines on top (the k-sets), Psi_AL (round depth 4) and Psi_NC (3) decided
"v has a red neighbor" on every kset/kid pair: "the query heads line m" plus "a red vertex is named in line m" for
m = 1, 2. The generator now puts a random number of filler lines before and between the query lines. This script
evaluates, with the exact evaluator of TL[P,Y,S,M] (tl_eval.py, from the logician's proofs12 scripts), the family
Psi_{<=M} = OR_{m<=M} (H_m AND P(E_F(line_m))) for every M whose round depth is at most 6 (the depth of the E3
models), on the records of a k-set file pair by pair, and reports the share of correctly answered records (1/2 is
chance on a matched pair set). The generator is accepted only if every such share is at most 0.55 on the kset and kid files.
Usage: python shortcut_check.py --split kset --seed 3 --pairs 30 --out results/kset_shortcut.json
       (pairs per k_min stratum, regenerated with gen_e3.pairs; the records equal those of make_data.sh)
"""
import argparse
import hashlib
import json
from pathlib import Path

import gen_e3
from tl_eval import AND, M, NOT, OR, P, T, Y, holds_final, rho

NL, BOS, RED = T("<NL>"), T("<BOS>"), T("red")


def at_least(j):
    """At least j line breaks before the position."""
    f = None
    for _ in range(j):
        f = P(NL) if f is None else P(AND(NL, f))
    return f


def line(m):
    if m == 1:
        return NOT(P(NL))
    return AND(at_least(m - 1), NOT(at_least(m)))


def head_of(m):
    """The query's earlier occurrence is the head (first name) of line m."""
    return M(0, 0, AND(line(m), Y(BOS) if m == 1 else Y(NL)))


def E(ser, ln):
    if ser == "AL":
        return AND(RED, M(2, 0, ln))                # a head color red whose head name occurs earlier in line ln
    return AND(RED, Y(T("(")), ln)                  # an entry color red in line ln


MAX_RHO = {"kset": 12, "kid": 6}       # the largest depth tested on each set (depth sweep up to 12 layers on kset)


def family(ser, max_rho=6):
    out = []
    for Mx in range(1, 12):
        f = OR(*[AND(head_of(m), P(E(ser, line(m)))) for m in range(1, Mx + 1)])
        if rho(f) > max_rho:
            break
        out.append((Mx, f))
    return out


def run(split, seed, n):
    res = {}
    recs = list(gen_e3.pairs("dia", split, seed, n))
    for ser in ("AL", "NC"):
        fam = family(ser, MAX_RHO.get(split, 6))
        cells = {}
        for Mx, f in fam:
            ok = tot = 0
            for G, vp, vn, names, s, order, lab in recs:
                for v, y in ((vp, 1), (vn, 0)):
                    toks = gen_e3.encode(G, v, names, ser, "dia", s, order)
                    ok += holds_final(f, toks) == bool(y)
                    tot += 1
            cells[str(Mx)] = {"rho": rho(f), "correct": ok, "records": tot, "share": ok / tot}
        res[ser] = cells
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="kset")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    out = {"split": a.split, "seed": a.seed, "pairs_per_stratum": a.pairs, "families": run(a.split, a.seed, a.pairs),
           "sha256": {f: hashlib.sha256((here / f).read_bytes()).hexdigest()
                      for f in ("gen_e3.py", "tl_eval.py", "shortcut_check.py")}}
    out["max_share"] = max(c["share"] for fam in out["families"].values() for c in fam.values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(a.split, "max share", round(out["max_share"], 3),
          {s: {m: round(c["share"], 3) for m, c in fam.items()} for s, fam in out["families"].items()})


if __name__ == "__main__":
    main()
