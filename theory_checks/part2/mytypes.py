"""Independent round-type computation and own-line pair construction (an implementation independent of construct13.py).
tp_l(x) = (tp_{l-1}(x), tp_{l-1}(x-1), capped past, (capped Match_{s,t}(x))_{s,t<=l-1}), exactly as in lb:main.
Positions are 1-based. Names are tokens starting with '@'.
"""
import sys
from bisect import bisect_left
from collections import defaultdict


def isname(t):
    return t.startswith("@")


def round_types(w, Lmax, kappa, intern=None):
    n = len(w)
    W = [None] + w
    if intern is None:
        intern = {}

    def I(key):
        v = intern.get(key)
        if v is None:
            v = len(intern)
            intern[key] = v
        return v

    occ = defaultdict(list)
    for x in range(1, n + 1):
        if isname(W[x]):
            occ[W[x]].append(x)
    tp = [[None] * (n + 1)]
    for x in range(1, n + 1):
        tp[0][x] = I(("tok", "*" if isname(W[x]) else W[x]))
    BOT = I(("bot",))
    for l in range(1, Lmax + 1):
        prev = tp[l - 1]
        cur = [None] * (n + 1)
        # capped past
        cnt = {}
        past = [None] * (n + 1)
        for x in range(1, n + 1):
            past[x] = I(("ms", frozenset(cnt.items())))
            k = prev[x]
            if cnt.get(k, 0) < kappa:
                cnt[k] = cnt.get(k, 0) + 1
        # prefix capped multisets per name and t: pref[name][t][m] = multiset of prev(o+t) for first m occurrences
        pref = {}
        for nm, os_ in occ.items():
            pt = []
            for t in range(l):
                c = {}
                lst = [I(("ms", frozenset()))]
                for o in os_:
                    j = o + t
                    if j <= n:
                        kk = prev[j]
                        if c.get(kk, 0) < kappa:
                            c[kk] = c.get(kk, 0) + 1
                    lst.append(I(("ms", frozenset(c.items()))))
                pt.append(lst)
            pref[nm] = pt
        for x in range(1, n + 1):
            mts = []
            for s in range(l):
                for t in range(l):
                    y = x - s
                    if y < 1 or not isname(W[y]):
                        mts.append(BOT)
                        continue
                    os_ = occ[W[y]]
                    # j = o + t with t < j < x  <=>  o >= 1 and o < x - t
                    m = bisect_left(os_, x - t)
                    mts.append(pref[W[y]][t][m])
            cur[x] = I((prev[x], prev[x - 1] if x > 1 else BOT, past[x], tuple(mts)))
        tp.append(cur)
    return tp


# ---------------- own-line pair construction from the text of ownline.tex ----------------
def line(ser, u, col, entries):
    if ser == "AL":
        body = []
        for i, e in enumerate(entries):
            body += ([","] if i else []) + ["@" + e]
        return ["@" + u, "(", col[u], ")", ":"] + (body or ["none"])
    body = []
    for i, e in enumerate(entries):
        body += ([","] if i else []) + ["@" + e, "(", col[e], ")"]
    return ["@" + u, "is", col[u], ".", "Neighbors", ":"] + (body or ["none"]) + ["."]


def build(case, j, kappa, Lover=None, Nover=None):
    ser = "AL" if case == "a" else "NC"
    L = {"a": 2 * j + 3, "b": 5 * j + 1, "c": 5 * j + 3}[case] if Lover is None else Lover
    J = j
    k = j + 2 if case == "a" else j + 1
    c = 1 + (L - 1) * (kappa + 1)
    N = 2 * c if Nover is None else Nover
    B = L * L + 2 * L
    col = {}
    lines = []
    bc = [0]

    def buffer():
        toks = 0
        while toks < B:
            z = f"z{bc[0]}"; bc[0] += 1
            col[z] = "blue"
            lines.append((z, []))
            toks += len(line(ser, z, col, [])) + 1
    H = [f"H{i}" for i in range(1, N + 1)]
    S = [f"s{i}" for i in range(1, N + 1)]
    F = [[f"f{i}_{m}" for m in range(1, J)] for i in range(1, N + 1)]
    Bn = [f"b{m}" for m in range(1, k)]
    for i in range(N):
        col[H[i]] = "red" if (i + 1) % 2 == 1 else "blue"
        col[S[i]] = "blue"
        for f in F[i]:
            col[f] = "blue"
    for b in Bn:
        col[b] = "blue"

    def U(i):
        for f in F[i]:
            lines.append((f, [S[i]]))
        lines.append((S[i], F[i] + [H[i]]))
        buffer()

    def V(i):
        lines.append((H[i], Bn + [S[i]]))
        buffer()
    if case in "ab":
        for i in range(N): V(i)
        for i in range(N): U(i)
    else:
        for i in range(N): U(i)
        for i in range(N): V(i)
    for b in Bn:
        lines.append((b, list(H)))
    # symmetric adjacency check
    adj = defaultdict(set)
    for u, es in lines:
        for e in es:
            adj[u].add(e)
    for u in list(adj):
        for e in adj[u]:
            assert u in adj[e], (u, e)
    W = ["<bos>"]
    for li, (u, es) in enumerate(lines):
        if li:
            W.append("<NL>")
        W += line(ser, u, col, es)
    W += ["<NL>", "Q", ":", "NBR"]
    vp, vm = S[N - 2], S[N - 1]
    return ser, L, N, W, vp, vm, col, adj, lines


def check_class(case, j, ser, lines, col, adj, v):
    """Return membership of query v in the own-line class (Definition own:def)."""
    order = {u: i for i, (u, _) in enumerate(lines)}
    lst = dict(lines)
    reds = [u for u in adj[v] if col[u] == "red"]
    # first occurrence index (line index) of each vertex: as head or entry
    first = {}
    for i, (u, es) in enumerate(lines):
        for x in [u] + es:
            first.setdefault(x, i)
    thr = j + 2 if case == "a" else j + 1
    ok = all(lst[u].index(v) + 1 >= thr for u in reds)
    ok &= all(lst[v].index(u) + 1 >= j for u in reds)
    if case == "c":
        ok &= all(first[u] >= order[v] for u in reds)  # no red nbr occurs before v's line
    if reds:
        if len(lst[v]) < j: return False
        u = lst[v][j - 1]
        ok &= col[u] == "red"
        if case == "a":
            ok &= order[u] < order[v]
        if case == "b":
            ok &= first[u] < order[v]
    return ok, bool(reds)


if __name__ == "__main__":
    case, j, kappa = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    Nover = int(sys.argv[4]) if len(sys.argv) > 4 else None
    ser, L, N, W, vp, vm, col, adj, lines = build(case, j, kappa, Nover=Nover)
    cp = check_class(case, j, ser, lines, col, adj, vp)
    cm = check_class(case, j, ser, lines, col, adj, vm)
    A = W + ["@" + vp]
    Bw = W + ["@" + vm]
    # compute types on concatenated? types are prefix-determined; compute separately (shared interner needed)
    # use one word with both endings by computing on A and B with a shared intern table: do it by merging
    intern = {}
    tpA = round_types(A, L + 1, kappa, intern)
    tpB = round_types(Bw, L + 1, kappa, intern)
    n = len(A)
    eq = [tpA[l][n] == tpB[l][n] for l in range(L + 2)]
    first = next((l for l in range(L + 2) if not eq[l]), None)
    print(case, j, kappa, "L=", L, "N=", N, "n=", n, "maxdeg=", max(len(s) for s in adj.values()),
          "class(+)=", cp, "class(-)=", cm, "equal_at_L=", eq[L], "first_diff=", first, flush=True)
