import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from rtypes import Interner, round_types
from construct import build, tokens
ser, L, k, N, J, buf, kappa = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), 2, int(sys.argv[5]), int(sys.argv[6])
lines, col, nb, meta = build(ser, k, N, J, buf)
ip = max(i for i in range(N) if col[meta["H"][i]] == "red"); im = max(i for i in range(N) if col[meta["H"][i]] == "blue")
qp, qm = meta["S"][ip], meta["S"][im]
A = tokens(ser, lines, col, qp)
I = Interner(); tp = round_types(A[:-1], L, I, kappa)   # shared prefix
inv = {v: kk for kk, v in I.tab.items()}
def occ(q): return [x for x in range(1, len(A)) if A[x-1] == "@" + q]
for l in range(L):
    for t in range(l + 1):
        sp = sorted(tp[l][o + t] for o in occ(qp)); sm = sorted(tp[l][o + t] for o in occ(qm))
        if sp != sm:
            print("round", l, "t", t, "occ+", occ(qp), "occ-", occ(qm))
            for o1, o2 in zip(occ(qp), occ(qm)):
                x, y = o1 + t, o2 + t
                if tp[l][x] != tp[l][y]:
                    print("  differ at", x, A[x-6:x+1], "vs", y, A[y-6:y+1])
                    a, b = inv[tp[l][x]], inv[tp[l][y]]
                    for ci, (u, v) in enumerate(zip(a, b)):
                        if u != v: print("   component", ci)
                    print('   x-s names:', [A[x-1-s] for s in range(l)])
                    if a[4] != b[4]:
                        for idx, (u, v) in enumerate(zip(a[4], b[4])):
                            if u != v: print("   match comp (s,t)=", divmod(idx, l), inv.get(u), inv.get(v))
            sys.exit()
