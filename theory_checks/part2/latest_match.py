"""Item (C): unique hard attention with a name-equality score and rightmost tie-breaking selects the LATEST earlier
match. Add the operator  ML_{s,t} psi  ("the latest j < i with w_{j-t} = w_{i-s} a name satisfies psi"), charged like
M_{s,t}: rho = max(rho psi, s, t) + 1.  A self-keyed ML is an exact shift by s-t (names do not repeat inside a line),
so the hard pairs of Theorem lb:main (construct12.py = proofs12/construct.py, S-units first) are separated with rho = 6 at L = 6 and rho = 7 at L = 12 (L: the round depth up to which lb:main proves them inseparable in TL).
Usage: python3 -I latest_match.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tl_eval  # noqa: E402
from tl_eval import T, Y, is_name  # noqa: E402
from construct12 import build, tokens  # noqa: E402  (copy of proofs12/construct.py: Theorem lb:main)


def evaluate(f, w, cache=None):
    cache = {} if cache is None else cache
    key = repr(f)
    if key in cache:
        return cache[key]
    if f[0] == "ML":
        s, t, g = f[1], f[2], f[3]
        a = evaluate(g, w, cache)
        n = len(w)
        last = {}            # name -> latest j (so far) with w_{j-t} = name
        v = [False]
        pending = []         # j's become available to i > j
        for i in range(1, n + 1):
            # make j = i-1 available
            j = i - 1
            if j >= t + 1 and is_name(w[j - t - 1]):
                last[w[j - t - 1]] = j
            if i > s and is_name(w[i - s - 1]) and w[i - s - 1] in last:
                v.append(a[last[w[i - s - 1]]])
            else:
                v.append(False)
        cache[key] = v
        return v
    # delegate other operators to tl_eval, but with ML-aware recursion
    op = f[0]
    if op in ("tok", "name", "true"):
        v = tl_eval.evaluate(f, w)
    elif op == "not":
        a = evaluate(f[1], w, cache)
        v = [False] + [not a[i] for i in range(1, len(w) + 1)]
    elif op in ("and", "or"):
        subs = [evaluate(g, w, cache) for g in f[1:]]
        comb = all if op == "and" else any
        v = [False] + [comb(x[i] for x in subs) for i in range(1, len(w) + 1)]
    elif op == "Y":
        a = evaluate(f[1], w, cache)
        v = [False] + [i > 1 and a[i - 1] for i in range(1, len(w) + 1)]
    else:
        raise ValueError(op)
    cache[key] = v
    return v


def rho(f):
    if f[0] == "ML":
        return max(rho(f[3]), f[1], f[2]) + 1
    if f[0] in ("tok", "name", "true"):
        return 0
    if f[0] == "not":
        return rho(f[1])
    if f[0] in ("and", "or"):
        return max(rho(g) for g in f[1:])
    if f[0] == "Y":
        return rho(f[1]) + 1
    raise ValueError(f[0])


def chain(target):
    """AL: a formula true at an entry at offset `target` (odd, >= 5) of a line iff the line's head color is red,
    built from Y^3 red at offset 5 and self-keyed ML shifts; dynamic programming over offsets for the fewest rounds."""
    best = {5: Y(T("red"), 3)}
    for p in range(7, target + 1, 2):
        cands = []
        for s in range(2, p - 4, 2):
            g = best[p - s]
            cands.append(("ML", s, 0, g))
        best[p] = min(cands, key=rho)
    return best[target]


def main():
    print("ser L k* k(=k*+1) N  rho(UHA-latest formula)  separates(+,-)  [lb:main: no TL formula with rho<=L separates]")
    for L in range(4, 17, 2):
        k = (L - 2) // 2 + 1
        N, buf, J = 2 * L + 6, L + 2, 2
        lines, col, nb, meta = build("AL", k, N, J, buf)
        ip = max(i for i in range(N) if col[meta["H"][i]] == "red")
        im = max(i for i in range(N) if col[meta["H"][i]] == "blue")
        A = tokens("AL", lines, col, meta["S"][ip])
        B = tokens("AL", lines, col, meta["S"][im])
        f = ("ML", 0, 0, chain(3 + 2 * k))
        a, b = evaluate(f, A)[len(A)], evaluate(f, B)[len(B)]
        print(f"AL {L:2d} {(L - 2) // 2:2d} {k:2d} {N:3d}  {rho(f):2d}  {a},{b}")


if __name__ == "__main__":
    main()
