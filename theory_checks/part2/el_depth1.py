"""Item (B): match-nesting depth under EL (match_depth.tex). Three checks with tl_eval.py.

1. The pair of Theorem md:el (one serialization W, queries v+ / v-, maximum degree 1): the nested formula of
   Proposition prop:shift(b) (match-nesting depth 2) separates it; random formulas of match-nesting depth <= 1
   (with P^{>=k}, Y, S, M^{>=k}_{s,t}) never do (a sanity test of the proof, not a proof).
2. As a control, random formulas of depth 2 separate the pair at a positive rate (the test is not vacuous).
3. Bounded size: the depth-1 formula Theta_{E0} of Proposition md:bounded is correct on every query vertex of random
   EL prompts with at most E0 edges, and fails on some larger prompt (it is not a definition for unbounded size).
Usage: python3 -I el_depth1.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl_eval import AND, M, NAME, NOT, OR, P, T, Y, evaluate, holds_final  # noqa: E402

NL = "<NL>"


def el_tokens(order, colors, edges, q):
    t = ["<BOS>", "Colors", ":"]
    for i, u in enumerate(order):
        t += ([","] if i else []) + ["@" + u, colors[u]]
    t += [NL, "Edges", ":"]
    for a, b in edges:
        t += [NL, "@" + a, "--", "@" + b]
    return t + [NL, "Q", ":", "NBR", "@" + q]


def hard_pair(P=60, E=60):
    """u red; v+ -- u, v- -- w; padding edges between fresh blue vertices; every vertex has degree <= 1.
    Color line: u, then P blue padding vertices, v+, P/2 blue, v-, P/2 blue, w, then the remaining endpoints.
    Edge list: E padding edges, (u, v+), E/2 padding, (w, v-), E/2 padding (v+- second endpoint in both)."""
    colors = {"u": "red", "vp": "blue", "vm": "blue", "w": "blue"}
    pad = [f"p{i}" for i in range(2 * (2 * E) + 2 * P)]
    for p in pad:
        colors[p] = "blue"
    it = iter(pad)
    edges = []

    def padedge():
        edges.append((next(it), next(it)))

    for _ in range(E):
        padedge()
    edges.append(("u", "vp"))
    for _ in range(E // 2):
        padedge()
    edges.append(("w", "vm"))
    for _ in range(E // 2):
        padedge()
    used = [x for e in edges for x in e]
    rest = [x for x in pad if x not in set(used)]      # isolated padding vertices
    endpoints = [x for x in used if x not in ("u", "vp", "vm", "w")]
    allpad = endpoints + rest
    order = ["u"] + allpad[:P] + ["vp"] + allpad[P:P + P // 2] + ["vm"] + allpad[P + P // 2:2 * P] + ["w"] + allpad[2 * P:]
    assert sorted(order) == sorted(colors)
    W = el_tokens(order, colors, edges, "vp")[:-1]
    deg = {}
    for a, b in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    assert max(deg.values()) == 1
    return W + ["@vp"], W + ["@vm"]


# --- the nested EL formula of Prop. prop:shift(b) (match-nesting depth 2) ---
NESTED = OR(M(0, 2, AND(Y(T("--")), M(0, 1, T("red")))), M(0, 0, AND(Y(T("--")), M(2, 1, T("red")))))

ATOMS = [T(x) for x in ("red", "blue", ",", ":", "--", NL, "Colors", "Edges", "Q", "NBR")] + [NAME]


def rand_blind(rng, d):
    """random name-blind formula (no M), operator depth <= d."""
    if d == 0 or rng.random() < 0.25:
        return rng.choice(ATOMS)
    op = rng.choice(["not", "and", "or", "Y", "P", "S"])
    if op == "not":
        return NOT(rand_blind(rng, d - 1))
    if op in ("and", "or"):
        return (op, rand_blind(rng, d - 1), rand_blind(rng, d - 1))
    if op == "Y":
        return Y(rand_blind(rng, d - 1))
    if op == "P":
        return P(rand_blind(rng, d - 1), rng.choice([1, 1, 2, 3]))
    return ("S", rand_blind(rng, d - 1), rand_blind(rng, d - 1))


def rand_md(rng, d, md):
    """random formula of match-nesting depth <= md and operator depth <= d."""
    if md == 0:
        return rand_blind(rng, d)
    if d == 0 or rng.random() < 0.15:
        return rng.choice(ATOMS)
    op = rng.choice(["not", "and", "or", "Y", "P", "S", "M", "M", "M"])
    if op == "M":
        return M(rng.randint(0, 3), rng.randint(0, 3), rand_md(rng, d - 1, md - 1), rng.choice([1, 1, 2]))
    if op == "not":
        return NOT(rand_md(rng, d - 1, md))
    if op in ("and", "or"):
        return (op, rand_md(rng, d - 1, md), rand_md(rng, d - 1, md))
    if op == "Y":
        return Y(rand_md(rng, d - 1, md))
    if op == "P":
        return P(rand_md(rng, d - 1, md), rng.choice([1, 1, 2, 3]))
    return ("S", rand_md(rng, d - 1, md), rand_md(rng, d - 1, md))


def exactly_nl_before(m):
    return AND(P(T(NL), m), NOT(P(T(NL), m + 1))) if m else NOT(P(T(NL), 1))


def theta_bounded(E0):
    """Prop. md:bounded: match-nesting depth 1, correct when the graph has at most E0 edges."""
    ds = []
    red_here = M(0, 1, T("red"))
    for e in range(1, E0 + 1):
        lam = exactly_nl_before(e + 1)
        first = AND(NAME, Y(T(NL)), lam)
        second = AND(NAME, Y(T("--")), lam)
        ds.append(AND(M(0, 0, second), P(AND(first, red_here))))
        ds.append(AND(M(0, 0, first), P(AND(second, red_here))))
    return OR(*ds)


def random_el(rng, nmax, emax):
    n = rng.randint(2, nmax)
    V = [f"x{i}" for i in range(n)]
    colors = {u: ("red" if rng.random() < 0.4 else "blue") for u in V}
    allp = [(V[i], V[j]) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(allp)
    m = rng.randint(0, min(emax, len(allp)))
    edges = [(a, b) if rng.random() < 0.5 else (b, a) for a, b in allp[:m]]
    order = V[:]
    rng.shuffle(order)
    return V, colors, edges, order


def main():
    rng = random.Random(20261103)
    A, B = hard_pair()
    print(f"[1] hard pair: {len(A)} tokens; nested depth-2 formula: + {holds_final(NESTED, A)}, - {holds_final(NESTED, B)}")
    bad, tot = 0, 0
    for _ in range(3000):
        f = rand_md(rng, rng.randint(2, 6), 1)
        tot += 1
        bad += holds_final(f, A) != holds_final(f, B)
    print(f"[1] random formulas of match-nesting depth <= 1: {bad} of {tot} separate the pair (theorem: 0)")
    sep, tot2 = 0, 0
    for _ in range(3000):
        f = rand_md(rng, rng.randint(2, 6), 2)
        tot2 += 1
        sep += holds_final(f, A) != holds_final(f, B)
    print(f"[2] control, random formulas of match-nesting depth <= 2: {sep} of {tot2} separate the pair")
    sep, tot3 = 0, 0
    for _ in range(3000):        # shaped depth-2 family M_{0,a}(chi and M_{b,c}(color))
        f = M(0, rng.randint(0, 3), AND(rand_blind(rng, 2), M(rng.randint(0, 3), rng.randint(0, 3),
                                                         T(rng.choice(["red", "blue"])))))
        tot3 += 1
        sep += holds_final(f, A) != holds_final(f, B)
    print(f"[2] control, shaped depth-2 formulas M(chi and M(color)): {sep} of {tot3} separate the pair")
    for E0 in (3, 6):
        th = theta_bounded(E0)
        wrong, cases = 0, 0
        for _ in range(300):
            V, colors, edges, order = random_el(rng, 8, E0)
            nbr = {u: set() for u in V}
            for a, b in edges:
                nbr[a].add(b)
                nbr[b].add(a)
            for q in V:
                w = el_tokens(order, colors, edges, q)
                cases += 1
                wrong += holds_final(th, w) != any(colors[x] == "red" for x in nbr[q])
        # a prompt with more than E0 edges on which Theta_{E0} errs
        V = [f"y{i}" for i in range(2 * E0 + 4)]
        colors = {u: "blue" for u in V}
        colors["y1"] = "red"
        edges = [(V[2 * i + 2], V[2 * i + 3]) for i in range(E0)] + [("y1", "y0")]
        w = el_tokens(V, colors, edges, "y0")
        print(f"[3] Theta_{E0} (depth 1): {wrong} errors on {cases} (prompt, query) cases with <= {E0} edges; "
              f"on a positive with {E0 + 1} edges (red edge last) it gives {holds_final(th, w)} (truth True)")


if __name__ == "__main__":
    main()
