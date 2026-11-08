"""Item (A): tightness of the own-line routes. For each route and J, build the pair (construct13.py), check labels and
route conditions from the tokens (e3/ownline_check.access), and compare round types at the final position
(rtypes.py) at L = threshold-1 (must be equal) and L = threshold (must differ). Also evaluates the own-line formula of
Prop. new:prop-rounds with tl_eval.py (must be true on q+ and false on q-, with the stated round depth).

  route AL_Phi : hub_first, k = J+2, threshold 2J+4  (Phi^AL_J)
  route NC_Phi : hub_first, k = J+1, threshold 5J+2  (Phi^NC_J)
  route NC_Psi : s_first,   k = J+1, threshold 5J+4  (Psi^NC_J)
Usage: python3 -I ownline_pairs.py ROUTE Jmin Jmax [--kappa K] [--proofparams] [--hash] [--Lmax 12]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "..", "e3"))
import rtypes  # noqa: E402
from rtypes import Interner, round_types  # noqa: E402
from construct13 import build, tokens  # noqa: E402
from tl_eval import AND, M, NAME, NOT, T, Y, holds_final, rho  # noqa: E402
import ownline_check  # noqa: E402


def guard(m):
    return AND(*[Y(NOT(T("<NL>")), l) for l in range(1, m + 1)]) if m else ("true",)


def formula(route, j):
    if route == "AL_Phi":
        return M(0, 2 * j + 3, AND(NAME, Y(T("("), 2 * j + 2), guard(2 * j + 2), M(0, 2, T("red"))))
    if route == "NC_Phi":
        return M(0, 5 * j + 1, AND(NAME, Y(T("is"), 5 * j), guard(5 * j), M(0, 2, T("red"))))
    return M(0, 5 * j + 3, AND(T("red"), Y(T("(")), Y(T("is"), 5 * j + 2), guard(5 * j + 2)))


SPEC = {"AL_Phi": ("AL", "hub_first", lambda J: J + 2, lambda J: 2 * J + 4),
        "NC_Phi": ("NC", "hub_first", lambda J: J + 1, lambda J: 5 * J + 2),
        "NC_Psi": ("NC", "s_first", lambda J: J + 1, lambda J: 5 * J + 4)}


def kstar(ser, L):
    return max(0, (L - 2) // 2) if ser == "AL" else L // 5


def run(route, J, kappa=1, proofparams=False):
    ser, order, kf, thf = SPEC[route]
    k, thr = kf(J), thf(J)
    L = thr
    if proofparams:
        N = 2 + 2 * (L - 1) * (kappa + 1)
        N += N % 2
    else:
        N = 2 * L + 6
    toklen = 6 if ser == "AL" else 9
    buf = -(-(L * L // 2 + 3 * L) // toklen) + 1
    lines, col, nb, qp, qm = build(ser, k, N, J, buf, order)
    assert any(col[u] == "red" for u in nb[qp]) and not any(col[u] == "red" for u in nb[qm])
    A, B = tokens(ser, lines, col, qp), tokens(ser, lines, col, qm)
    acc = ownline_check.access(A, ser)
    accm = ownline_check.access(B, ser)
    assert acc["kmin"] == k and acc["j_inl"] == J, acc
    if order == "hub_first":
        assert acc["j_prec"] == J and acc["j_occ"] == J, acc
    else:
        assert acc["j_prec"] is None and acc["j_occ"] is None, acc
    assert accm["kmin"] is None and accm["j_inl"] is None
    # head route (2k+2 / 5k rounds) does not fit at the threshold
    assert k > kstar(ser, thr), (k, kstar(ser, thr))
    f = formula(route, J)
    assert rho(f) == thr, (rho(f), thr)
    fA, fB = holds_final(f, A), holds_final(f, B)
    I = Interner()
    tA = round_types(A, thr, I, kappa)
    tB = round_types(B, thr, I, kappa)
    n = len(A)
    eq = [tA[l][n] == tB[l][n] for l in range(thr + 1)]
    first = next((l for l in range(thr + 1) if not eq[l]), None)
    return dict(ser=ser, order=order, J=J, k=k, N=N, buf=buf, n=n, maxdeg=max(len(v) for v in nb.values()),
                threshold=thr, equal_below=eq[thr - 1], first_diff_round=first, formula_rho=rho(f),
                formula_pos=fA, formula_neg=fB, access=acc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("route", choices=sorted(SPEC))
    ap.add_argument("Jmin", type=int)
    ap.add_argument("Jmax", type=int)
    ap.add_argument("--kappa", type=int, default=1)
    ap.add_argument("--proofparams", action="store_true")
    ap.add_argument("--hash", action="store_true")
    a = ap.parse_args()
    rtypes.EXACT = not a.hash
    for J in range(a.Jmin, a.Jmax + 1):
        t0 = time.time()
        r = run(a.route, J, a.kappa, a.proofparams)
        ok = r["equal_below"] and r["first_diff_round"] == r["threshold"] and r["formula_pos"] and not r["formula_neg"]
        print(f"{a.route} J={J} kappa={a.kappa} hash={a.hash} proofparams={a.proofparams}: "
              f"{'OK' if ok else 'FAIL'} {r} [{time.time() - t0:.1f}s]", flush=True)
