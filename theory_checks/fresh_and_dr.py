"""(i) fresh (unshared) hub entries: minimal round depth that separates the pair, as a function of k (AL, NC);
(ii) operator-depth / offset version (d, r decoupled): smallest k for which the shared construction is
indistinguishable, for a few (d, r)."""
import os
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "e3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtypes
from rtypes import Interner, round_types
from construct import build, tokens
rtypes.EXACT = False


def pair(ser, k, N, J, buf, fresh):
    lines, col, nb, meta = build(ser, k, N, J, buf, fresh_b=fresh)
    ip = max(i for i in range(N) if col[meta["H"][i]] == "red")
    im = max(i for i in range(N) if col[meta["H"][i]] == "blue")
    return tokens(ser, lines, col, meta["S"][ip]), tokens(ser, lines, col, meta["S"][im])


def first_diff(A, B, L, kappa=1, rmax=None):
    I = Interner()
    tA, tB = round_types(A, L, I, kappa, rmax), round_types(B, L, I, kappa, rmax)
    n = len(A)
    return next((l for l in range(L + 1) if tA[l][n] != tB[l][n]), None)


if __name__ == "__main__":
    print("(i) fresh hub entries: minimal separating round depth vs. head route")
    for ser in ("AL", "NC"):
        for k in range(1, 11 if ser == "AL" else 6):
            head = 2 * k + 2 if ser == "AL" else 5 * k
            L = head
            J = 2 if ser == "AL" else max(2, (L - 4) // 5 + 1)
            A, B = pair(ser, k, 2 * L + 6, J, L + 2, True)
            print(f"  {ser} k={k}: head route {head}, fresh pair first separated at round {first_diff(A, B, L)}", flush=True)
    print("(ii) decoupled (operator depth d, offsets <= r), AL, kappa=1: smallest k with equal depth-d types")
    for d in (3, 6):
        for r in (0, 2, 5, 10, 20):
            for k in range(1, 30):
                A, B = pair("AL", k, 2 * d + 6, 2, max(d, r) + 4, False)
                if first_diff(A, B, d, 1, r) is None:
                    print(f"  d={d} r={r}: k_min hard = {k}", flush=True)
                    break
