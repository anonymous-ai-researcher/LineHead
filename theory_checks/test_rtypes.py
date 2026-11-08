"""Soundness test of rtypes.round_types against tl_eval: on random words, random formulas of round depth <= L and
thresholds <= kappa never separate two positions (of the same or different words) that have equal round-L types."""
import random
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import rtypes
from rtypes import Interner, round_types
from tl_eval import evaluate, rho

ALPH = ["a", "b", "c"]
NAMES = ["@x", "@y", "@z", "@u"]


def rand_word(rng, n):
    return [rng.choice(ALPH + NAMES) for _ in range(n)]


def rand_formula(rng, L, kappa, depth=0):
    """random formula with rounds <= L (k-thresholds counted as one round)"""
    if L == 0 or rng.random() < 0.25:
        r = rng.random()
        if r < 0.15:
            return ("name",)
        return ("tok", rng.choice(ALPH))
    op = rng.choice(["not", "and", "or", "Y", "P", "M", "M", "M"])
    if op == "not":
        return ("not", rand_formula(rng, L, kappa))
    if op in ("and", "or"):
        return (op, rand_formula(rng, L, kappa), rand_formula(rng, L, kappa))
    if op == "Y":
        return ("Y", rand_formula(rng, L - 1, kappa))
    if op == "P":
        return ("P", rand_formula(rng, L - 1, kappa), rng.randint(1, kappa))
    s, t = rng.randint(0, L - 1), rng.randint(0, L - 1)
    return ("M", s, t, rand_formula(rng, L - 1, kappa), rng.randint(1, kappa))


def main(trials=200, seed=0):
    rng = random.Random(seed)
    viol = checked = 0
    for tr in range(trials):
        L, kappa = rng.randint(1, 4), rng.randint(1, 3)
        rtypes.EXACT = tr % 2 == 0
        words = [rand_word(rng, rng.randint(5, 40)) for _ in range(3)]
        I = Interner()
        tps = [round_types(w, L, I, kappa) for w in words]
        groups = {}
        for wi, w in enumerate(words):
            for x in range(1, len(w) + 1):
                groups.setdefault(tps[wi][L][x], []).append((wi, x))
        for _ in range(30):
            f = rand_formula(rng, L, kappa)
            assert rho(f) <= L
            vals = [evaluate(f, w) for w in words]
            for g in groups.values():
                tv = {vals[wi][x] for wi, x in g}
                checked += 1
                viol += len(tv) > 1
    print(f"groups checked {checked}, violations {viol}")
    return viol


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
