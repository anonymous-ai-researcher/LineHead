"""Evaluator for TL[P,Y,S,M] over token sequences (names = tokens starting with '@'), with round depth rho.

Formulas are nested tuples:
  ('tok', 'red')  ('name',)  ('true',)  ('not', f)  ('and', f, g, ...)  ('or', f, g, ...)
  ('Y', f)  ('P', f, k=1)  ('S', f, g)  ('M', s, t, f, k=1)
Semantics (appendix.tex, Sec. A): positions 1..n; M^{>=k}_{s,t} f at i: #{j<i : j>t, i>s, w_{i-s}=w_{j-t} in N, f(j)} >= k.
rho as in Definition new:def-rounds (thresholds > 1 are charged here as 1 round, i.e. a LOWER bound on the paper's rho).
"""
from bisect import bisect_left


def is_name(t):
    return t.startswith("@")


def rho(f):
    op = f[0]
    if op in ("tok", "name", "true"):
        return 0
    if op == "not":
        return rho(f[1])
    if op in ("and", "or"):
        return max(rho(g) for g in f[1:])
    if op in ("Y", "P"):
        return rho(f[1]) + 1
    if op == "S":
        return max(rho(f[1]), rho(f[2])) + 1
    if op == "M":
        return max(rho(f[3]), f[1], f[2]) + 1
    raise ValueError(op)


def evaluate(f, w, cache=None):
    """Boolean list v[0..n] with v[i] = (w,i |= f) for 1-based i (v[0] unused)."""
    n = len(w)
    cache = {} if cache is None else cache
    key = repr(f)
    if key in cache:
        return cache[key]
    op = f[0]
    if op == "tok":
        v = [False] + [x == f[1] for x in w]
    elif op == "name":
        v = [False] + [is_name(x) for x in w]
    elif op == "true":
        v = [False] + [True] * n
    elif op == "not":
        a = evaluate(f[1], w, cache)
        v = [False] + [not a[i] for i in range(1, n + 1)]
    elif op in ("and", "or"):
        subs = [evaluate(g, w, cache) for g in f[1:]]
        comb = all if op == "and" else any
        v = [False] + [comb(s[i] for s in subs) for i in range(1, n + 1)]
    elif op == "Y":
        a = evaluate(f[1], w, cache)
        v = [False] + [i > 1 and a[i - 1] for i in range(1, n + 1)]
    elif op == "P":
        a = evaluate(f[1], w, cache)
        k = f[2] if len(f) > 2 else 1
        v, c = [False], 0
        for i in range(1, n + 1):
            v.append(c >= k)
            c += a[i]
    elif op == "S":
        a, b = evaluate(f[1], w, cache), evaluate(f[2], w, cache)
        v, ok = [False], False      # ok: some j<i with b(j) and a on (j,i)
        for i in range(1, n + 1):
            v.append(ok)
            ok = b[i] or (ok and a[i])
    elif op == "M":
        s, t, g = f[1], f[2], f[3]
        k = f[4] if len(f) > 4 else 1
        a = evaluate(g, w, cache)
        occ = {}                     # name -> sorted list of j (1-based) with w_{j-t}=name and a[j]
        for j in range(t + 1, n + 1):
            if is_name(w[j - t - 1]) and a[j]:
                occ.setdefault(w[j - t - 1], []).append(j)
        v = [False]
        for i in range(1, n + 1):
            if i > s and is_name(w[i - s - 1]):
                lst = occ.get(w[i - s - 1], [])
                v.append(bisect_left(lst, i) >= k)
            else:
                v.append(False)
    else:
        raise ValueError(op)
    cache[key] = v
    return v


def holds_final(f, w):
    return evaluate(f, w)[len(w)]


# ---- helpers -----------------------------------------------------------------------------------------
def T(x):
    return ("tok", x)


def AND(*a):
    return ("and",) + a


def OR(*a):
    return ("or",) + a


def NOT(a):
    return ("not", a)


def Y(a, m=1):
    for _ in range(m):
        a = ("Y", a)
    return a


def P(a, k=1):
    return ("P", a, k)


def M(s, t, a, k=1):
    return ("M", s, t, a, k)


NAME = ("name",)
