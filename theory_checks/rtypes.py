"""Round types (bisimulation for round depth) on token sequences.

tp_0(x)   = token class of w_x (names -> '*')
tp_l(x)   = ( tp_{l-1}(x), tp_{l-1}(x-1) or None, Past_{l-1}(x), (Match_{l-1}(x;s,t))_{s,t<=l-1} )
Past_{l-1}(x)      = capped multiset (cap kappa) of tp_{l-1}(j), j < x
Match_{l-1}(x;s,t) = capped multiset of tp_{l-1}(j) over j < x, j > t, x > s, w_{x-s} = w_{j-t} a name
(None if w_{x-s} is not a name).

Lemma (proved in lower_bound.tex): if tp_L of (A,x) equals tp_L of (B,y), then every formula of TL[P^{>=k},Y,M^{>=k}_{s,t}]
with thresholds <= kappa whose round depth is <= L -- counting every P^{>=k}, M^{>=k}_{s,t} like its threshold-1
version, i.e. rho(M^{>=k}_{s,t} f) = max(rho f, s, t)+1 -- has the same truth value at x in A and at y in B.
With kappa = 1 this covers exactly the formulas of Definition new:def-rounds (threshold 1).
Type ids are interned in a table shared by all words passed to the same Interner, so ids are comparable.
"""
from bisect import bisect_left


class Interner:
    def __init__(self):
        self.tab = {}

    def __call__(self, obj):
        i = self.tab.get(obj)
        if i is None:
            i = self.tab[obj] = len(self.tab)
        return i


def is_name(t):
    return t.startswith("@")


import random as _random
_ZOB = {}
_rng = _random.Random(20261103)


def _z(x, c):
    v = _ZOB.get((x, c))
    if v is None:
        v = _ZOB[(x, c)] = _rng.getrandbits(128)
    return v


EXACT = True      # exact canonical multisets (frozensets); False: 128-bit Zobrist hashes (collision prob ~2^-100)


class CappedMS:
    """Incrementally built capped multiset; .key() is a canonical id (exact or hashed, see EXACT)."""
    def __init__(self, intern, kappa):
        self.c, self.kappa, self.intern = {}, kappa, intern
        self.h = 0
        self._id = intern(("ms", frozenset())) if EXACT else ("h", 0)

    def add(self, x):
        c = self.c.get(x, 0)
        if c < self.kappa:
            self.c[x] = c + 1
            if EXACT:
                self._id = self.intern(("ms", frozenset(self.c.items())))
            else:
                self.h ^= (_z(x, c) if c else 0) ^ _z(x, c + 1)
                self._id = ("h", self.h)

    def key(self):
        return self._id


def round_types(w, L, intern, kappa=1, rmax=None):
    """w: list of tokens (1-based semantics; w[0] is position 1). Returns tps[l][x] for x in 1..n (index 0 unused).
    rmax=None: round-coupled offsets (s,t <= l-1 at level l). rmax=r: offsets s,t <= r at every level (then level l
    corresponds to operator depth l, as in Theorem new:thm-lower)."""
    n = len(w)
    pos_of = {}
    for x in range(1, n + 1):
        if is_name(w[x - 1]):
            pos_of.setdefault(w[x - 1], []).append(x)
    empty = intern(("ms", frozenset())) if EXACT else ("h", 0)
    tps = [[None] + [intern(("0", "*" if is_name(t) else t)) for t in w]]
    for l in range(1, L + 1):
        prev = tps[-1]
        # capped pasts
        past = [None] * (n + 1)
        ms = CappedMS(intern, kappa)
        for x in range(1, n + 1):
            past[x] = ms.key()
            ms.add(prev[x])
        # match tables: for each (name, t): sorted j's = o + t and cumulative capped-multiset ids
        tables = {}
        R = range(0, l) if rmax is None else range(0, rmax + 1)
        for t in R:
            for a, occ in pos_of.items():
                js = [o + t for o in occ if o + t <= n]
                cm = CappedMS(intern, kappa)
                cum = []
                for j in js:
                    cm.add(prev[j])
                    cum.append(cm.key())
                tables[(a, t)] = (js, cum)
        cur = [None] * (n + 1)
        for x in range(1, n + 1):
            comps = []
            for s in R:
                if x - s >= 1 and is_name(w[x - s - 1]):
                    a = w[x - s - 1]
                    for t in R:
                        js, cum = tables[(a, t)]
                        m = bisect_left(js, x)          # j < x  (j > t holds since j = o + t, o >= 1)
                        comps.append(cum[m - 1] if m else empty)
                else:
                    comps.extend([None] * len(R))
            cur[x] = intern((l, prev[x], prev[x - 1] if x > 1 else None, past[x], tuple(comps)))
        tps.append(cur)
    return tps
