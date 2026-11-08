"""Exact check of Lemma leap:fool via position types (no random sampling).

depth-e kappa-type of position i in a masked word (Booleans NOT counted in depth):
  T_0(i) = token(i)  (names masked)
  T_e(i) = (T_{e-1}(i), T_{e-1}(i-1) [Y], capped counts min(kappa, #{j<i: T_{e-1}(j)=tau}) for all tau [P^{>=k}, k<=kappa],
            recency order of T_{e-1}-types before i [S])
Two positions with equal T_d agree on every name-blind formula of operator depth <= d and thresholds <= kappa,
and (within one word) conversely. A depth-<=1 formula at the final position of A = W a vs B = W b differs iff for
some t <= r the multisets {T_d(o+t): o in Occ_W(a)} and {T_d(o+t): o in Occ_W(b)} differ (matches M_{0,t});
matches with s>0 see only W's tail (identical in A, B) provided no occurrence of a, b lies in the last r positions.
"""
import os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'part2'))
import el_depth1 as ed
from tl_eval import holds_final


def size_PE(s):
    Pp = 3 * s; Pp += Pp % 2
    E = max(s + 1, Pp // 2); E += E % 2
    return Pp, E


def types(word, d, kappa):
    """word: list of masked tokens (positions 0..n-1). returns list of ints T_d."""
    T = [hash(x) for x in word]
    intern = {}
    def iid(x):
        if x not in intern:
            intern[x] = len(intern)
        return intern[x]
    T = [iid(("a", x)) for x in word]
    for e in range(1, d + 1):
        newT = []
        cnt = Counter()
        rec = []          # recency list of types, most recent first
        for i in range(len(word)):
            prev = T[i - 1] if i > 0 else None
            capped = tuple(sorted((tau, min(kappa, c)) for tau, c in cnt.items()))
            newT.append(iid((e, T[i], prev, capped, tuple(rec))))
            tau = T[i]
            cnt[tau] += 1
            if tau in rec:
                rec.remove(tau)
            rec.insert(0, tau)
        T = newT
    return T


def mask(w):
    return ["_" if x.startswith("@") else x for x in w]


def separates(A, B, d, kappa, r):
    W = A[:-1]
    a, b = A[-1], B[-1]
    assert W == B[:-1]
    n = len(W)
    occ_a = [i for i, x in enumerate(W) if x == a]
    occ_b = [i for i, x in enumerate(W) if x == b]
    if any(o >= n - r for o in occ_a + occ_b):
        return "tail"
    T = types(mask(W), d, kappa)
    for t in range(r + 1):
        ma = Counter(T[o + t] for o in occ_a if o + t < n)
        mb = Counter(T[o + t] for o in occ_b if o + t < n)
        if ma != mb:
            return f"t={t}"
    return None


if __name__ == "__main__":
    Smax = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    bad = 0
    for s in range(1, Smax + 1):
        A, B = ed.hard_pair(*size_PE(s))
        # (i) every (d,kappa,r) with c_d + r <= s must be fooled
        checked = 0
        for d in range(0, 5):
            for kappa in range(1, s + 1):
                c = 1 + d * (kappa + 2)
                if c > s:
                    break
                for r in range(0, s - c + 1):
                    checked += 1
                    res = separates(A, B, d, kappa, r)
                    if res:
                        bad += 1
                        print(f"  VIOLATION s={s} d={d} kappa={kappa} r={r}: {res}")
                if d == 0:
                    break
        # (ii) smallest budget that separates (exact, over d<=3, kappa<=20, r<=3s+5)
        best = None
        for d in range(0, 4):
            for kappa in ([1] if d == 0 else range(1, 21)):
                c = 1 + d * (kappa + 2)
                for r in range(0, 3 * s + 8):
                    if best is not None and c + r >= best[0]:
                        break
                    res = separates(A, B, d, kappa, r)
                    if res and res != "tail":
                        best = (c + r, d, kappa, r, res)
                        break
        print(f"s={s}: |W|={len(A)-1}, edges={2*size_PE(s)[1]+2}, fooled-configs checked={checked}; "
              f"least separating budget found = {best}")
    print("violations:", bad)
