"""Verify hard pairs: for each L, build W (construct.py), the two queries, and compare round-L types at the final
position (rtypes.py). Also checks labels, k_min, and the blocking conditions from the tokens (e3/ownline_check.py).
Usage: python -I check_pairs.py AL|NC Lmin Lmax [--fresh] [--kappa K] [--N N] [--buf B] [--k K]"""
import argparse
import sys
import time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "e3"))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from rtypes import Interner, round_types  # noqa: E402
from construct import build, tokens  # noqa: E402
import ownline_check  # noqa: E402


def kstar(ser, L):
    return (L - 2) // 2 if ser == "AL" else L // 5


def jinl(ser, L):
    return 2 if ser == "AL" else max(2, (L - 4) // 5 + 1)


def check(ser, L, k, N, J, buf, kappa=1, fresh=False):
    lines, col, nb, meta = build(ser, k, N, J, buf, fresh_b=fresh)
    ip = max(i for i in range(N) if col[meta["H"][i]] == "red")
    im = max(i for i in range(N) if col[meta["H"][i]] == "blue")
    qp, qm = meta["S"][ip], meta["S"][im]
    lab = {q: any(col[u] == "red" for u in nb[q]) for q in (qp, qm)}
    assert lab[qp] and not lab[qm]
    A, B = tokens(ser, lines, col, qp), tokens(ser, lines, col, qm)
    acc = ownline_check.access(A, ser)
    assert acc["kmin"] == k and acc["j_prec"] is None and acc["j_occ"] is None and acc["j_inl"] == J, acc
    I = Interner()
    tA = round_types(A, L, I, kappa)
    tB = round_types(B, L, I, kappa)
    n = len(A)
    eq = [tA[l][n] == tB[l][n] for l in range(L + 1)]
    first_diff = next((l for l in range(L + 1) if not eq[l]), None)
    return dict(n=n, maxdeg=max(len(v) for v in nb.values()), equal=eq[L], first_diff_round=first_diff)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ser")
    ap.add_argument("Lmin", type=int)
    ap.add_argument("Lmax", type=int)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--kappa", type=int, default=1)
    ap.add_argument("--N", type=int, default=0)
    ap.add_argument("--buf", type=int, default=0)
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--proofparams", action="store_true",
                    help="N = 2+2(L-1)(kappa+1) hubs and buffers of ceil((L^2+2L)/7) lines, as in lower_bound.tex")
    ap.add_argument("--k", type=int, default=0, help="override k (default k*(L)+1); negative: k*(L)+1+k")
    a = ap.parse_args()
    import rtypes
    rtypes.EXACT = not a.hash
    for L in range(a.Lmin, a.Lmax + 1):
        k = kstar(a.ser, L) + 1
        if a.k > 0:
            k = a.k
        elif a.k < 0:
            k = k + a.k
        N = a.N or 2 * L + 6
        buf = a.buf or L + 2
        if a.proofparams:
            N = 2 + 2 * (L - 1) * (a.kappa + 1)
            N += N % 2
            buf = -(-(L * L + 2 * L) // 7)
        t0 = time.time()
        r = check(a.ser, L, k, N, jinl(a.ser, L), buf, a.kappa, a.fresh)
        print(f"{a.ser} L={L} k={k} (k*={kstar(a.ser, L)}) N={N} buf={buf} kappa={a.kappa} fresh={a.fresh}: "
              f"{r}  [{time.time() - t0:.1f}s]", flush=True)
