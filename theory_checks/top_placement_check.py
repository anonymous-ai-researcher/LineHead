"""Counterexample to a finite-depth lower bound on the generator's k-controlled sets (gen_e3.blocked_order puts the
lines of the two query vertices on top). Psi_AL (rho=4) and Psi_NC (rho=3) decide 'v has a red neighbor' on every
kset/kid record, at every k_min; on random-order (ood/id) records they are far from correct.
Usage: python -I top_placement_check.py [pairs_per_stratum]"""
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "e3"))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "m0"))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from tl_eval import T, AND, OR, NOT, Y, P, M, NAME, rho, holds_final  # noqa: E402
import gen_e3  # noqa: E402

NL = T("<NL>")
line1 = NOT(P(NL))                                   # no line break before: first line of the serialization
line2 = AND(P(NL), NOT(P(AND(NL, P(NL)))))           # exactly one line break before
H1 = M(0, 0, NOT(P(NAME)))                           # the query's first occurrence is the first name (head of line 1)
E_AL = lambda ln: AND(T("red"), M(2, 0, ln))         # a (head) color red whose head name occurs earlier in line ln
PSI_AL = OR(AND(H1, P(E_AL(line1))), AND(NOT(H1), P(E_AL(line2))))
E_NC = lambda ln: AND(T("red"), Y(T("(")), ln)       # an entry color red in line ln
PSI_NC = OR(AND(H1, P(E_NC(line1))), AND(NOT(H1), P(E_NC(line2))))


def run(split, n, seed):
    res = {}
    for ser, psi in (("AL", PSI_AL), ("NC", PSI_NC)):
        ok = tot = 0
        by_k = {}
        for G, vp, vn, names, s, order, lab in gen_e3.pairs("dia", split, seed, n):
            km = gen_e3.kmin_dmin(G, vp, ser, s, order)[0] if order else gen_e3.kmin_dmin(G, vp, ser, s)[0]
            for v, y in ((vp, 1), (vn, 0)):
                toks = gen_e3.encode(G, v, names, ser, "dia", s, order)
                c = holds_final(psi, toks) == bool(y)
                ok += c
                tot += 1
                if y:
                    by_k.setdefault(km, [0, 0])
                    by_k[km][0] += c
                    by_k[km][1] += 1
        res[ser] = (ok, tot, dict(sorted(by_k.items())))
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print("rho(Psi_AL) =", rho(PSI_AL), " rho(Psi_NC) =", rho(PSI_NC))
    for split, seed in (("kset", 3), ("kid", 4), ("ood", 2), ("id", 1)):
        r = run(split, n if split in ("kset", "kid") else 6 * n, seed)
        for ser, (ok, tot, byk) in r.items():
            print(f"{split:5s} {ser}: correct {ok}/{tot}; positives correct by k_min: "
                  + ", ".join(f"{k}:{a}/{b}" for k, (a, b) in byk.items()))
