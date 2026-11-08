"""Smallest hard pairs found by search (L = 6 and 12, AL and NC, kappa = 1, exact types): candidate stress tests
of the depth frontier that fit a transformer's context."""
import os
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "e3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtypes
from check_pairs import check, kstar, jinl
rtypes.EXACT = True
for ser, L in (("AL", 6), ("NC", 6), ("AL", 12), ("NC", 12)):
    best = None
    for N in range(4, 2 * L + 8, 2):
        for buf in range(0, L + 3):
            try:
                r = check(ser, L, kstar(ser, L) + 1, N, jinl(ser, L), buf)
            except AssertionError:
                continue
            if r["equal"] and (best is None or r["n"] < best[0]["n"]):
                best = (r, N, buf)
        if best:
            break
    print(ser, "L =", L, "smallest found:", best, flush=True)
