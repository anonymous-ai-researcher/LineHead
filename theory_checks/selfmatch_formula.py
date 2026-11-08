"""The self-matching shortcut made explicit. Theta = M_{0,0}( M_{2,0}( Y^3 red ) ), rho = 5 (< 6 = head route at k=2).
At an entry x, M_{2,0} attends to all earlier occurrences j of the name at x-2 -- including x-2 itself -- and Y^3 red
there reads the head color of the line of x-2 if x-2 is the first entry. So Theta separates the fresh-name pair at
k=2 with 5 rounds, but holds for both queries of the shared-name pair (ambiguity), and it is unsound on random
prompts (false positives), so it is not a decider."""
import os
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "e3"))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "m0"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl_eval import T, M, Y, rho, holds_final
from fresh_and_dr import pair
import gen_e3

THETA = M(0, 0, M(2, 0, Y(T("red"), 3)))
print("rho(Theta) =", rho(THETA))
for fresh in (True, False):
    A, B = pair("AL", 2, 16, 2, 7, fresh)
    print(f"  {'fresh' if fresh else 'shared'} pair, k=2: Theta(q+)={holds_final(THETA, A)}, Theta(q-)={holds_final(THETA, B)}")
fp = neg = tp = pos = 0
for G, vp, vn, names, s, order, lab in gen_e3.pairs("dia", "ood", 2, 300):
    for v, y in ((vp, 1), (vn, 0)):
        h = holds_final(THETA, gen_e3.encode(G, v, names, "AL", "dia", s, order))
        if y:
            pos += 1; tp += h
        else:
            neg += 1; fp += h
print(f"  random-order OOD AL prompts: Theta true on {tp}/{pos} positives and {fp}/{neg} negatives (unsound)")
