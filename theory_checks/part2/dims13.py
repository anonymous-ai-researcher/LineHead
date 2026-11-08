"""Dimension and width numbers quoted for Theorems mr:thm and ln:thm; writes results_dims13.json.
Closed forms of Lemma ln:conc2 and the three-term Chernoff sum; the exact-tail and Lemma mr:conc numbers are in results_D.txt."""
import json
import os
import numpy as np

a = lambda e: 0.5 * (e - np.log1p(e))
b = lambda e: 0.5 * (-e - np.log1p(-e))


def c(e):
    s = np.sqrt(1 + 4 * e * e)
    return 0.5 * (s - 1 - np.log((1 + s) / 2))


def least(f, dl):
    d = 1
    while f(d) > dl:
        d += 1
    return d


def three(m, e, dl):
    return least(lambda d: m * np.exp(-d * a(e)) + m * np.exp(-d * b(e)) + m * (m - 1) * np.exp(-d * c(e)), dl)


delta = 0.05
out = {"mr_thm_eps_1_4": {}, "ln_thm_eps_1_8": {}}
for m in (64, 150, 600):
    e = 0.25
    out["mr_thm_eps_1_4"][str(m)] = {
        "lemma_mr_conc": least(lambda d: 2 * m * m * np.exp(-d * e * e / 16), delta),
        "three_term": three(m, e, delta),
        "closed_12_over_eps2": int(np.ceil(12 / e ** 2 * np.log(2 * m * m / delta))),
        "closed_ln_over_a": int(np.ceil(np.log(2 * m * m / delta) / a(e))),
    }
for m in (64, 600):
    e = 0.125
    out["ln_thm_eps_1_8"][str(m)] = {
        "three_term_half_delta": three(m, e, delta / 2),
        "stated_12_over_eps2": int(np.ceil(12 / e ** 2 * np.log(4 * m * m / delta))),
        "width_D": int(np.ceil(8 * np.log(4 * m / delta) / e)),
    }
out["one_over_a_quarter"] = round(float(1 / a(0.25)), 1)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_dims13.json")
json.dump(out, open(p, "w"), indent=1, sort_keys=True)
print(json.dumps(out, indent=1, sort_keys=True))
