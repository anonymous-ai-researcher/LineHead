"""Unit tests of the E3 test sets.
1. hard_pairs.py: for AL and NC, a generated pair (fresh names) has the sizes of proofs12 (465 / 624 tokens,
   maximum degree 10 / 12), k_min = k*(6)+1, blocked own-line routes, and equal round-6 types at the final position
   (rtypes.py, kappa = 1; Theorem lb:main); control: the same construction with k = k*(6) is separated by round 6.
2. shortcut_check.py on the position-controlled sets: the line-index formulas Psi_{<=M} of round depth <= 6 stay
   at most 0.55 on kset and kid samples (exactly 0.5 by the layout), and decide every pair of the old top placement (sanity).
  python test_hard_pairs.py"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen_e3  # noqa: E402
import hard_pairs as HP  # noqa: E402
import rtypes  # noqa: E402
import shortcut_check as SC  # noqa: E402
from tl_eval import holds_final  # noqa: E402

rtypes.EXACT = True


def final_equal(A, B, L, kappa=1):
    I = rtypes.Interner()
    ta, tb = rtypes.round_types(A, L, I, kappa), rtypes.round_types(B, L, I, kappa)
    return ta[L][len(A)] == tb[L][len(B)]


def test_hard():
    for ser, (n_tok, md) in (("AL", (465, 10)), ("NC", (624, 12))):
        G, vp, vn, names, s, order = next(HP.pairs(ser, 8, 1))
        A = gen_e3.encode(G, vp, names, ser, "dia", s, order)
        B = gen_e3.encode(G, vn, names, ser, "dia", s, order)
        assert len(A) == n_tok and max(d for _, d in G.degree) == md, (ser, len(A))
        assert A[:-1] == B[:-1] and A[-1] != B[-1]
        assert any(G.nodes[u]["color"] == "red" for u in G[vp]) and not any(G.nodes[u]["color"] == "red" for u in G[vn])
        km = HP.kmin_dmin(G, vp, ser, s, order)[0]
        kstar = 2 if ser == "AL" else 1
        assert km == kstar + 1
        acc = HP.ownline_access(G, vp, order)
        assert acc["j_prec"] is None and acc["j_occ"] is None and acc["j_inl"] == 2, acc
        assert final_equal(A, B, 6), f"{ser}: round-6 types differ"
        # control: k = k*(6) (inside the frontier) is separated within 6 rounds
        p = dict(HP.PARAMS[ser])
        lines, col, qp, qm = HP.build(kstar, p["N"], p["J"], p["buf"])
        ids = {u: i for i, (u, _) in enumerate(lines)}
        import networkx as nx
        G2 = nx.Graph()
        for u, i in ids.items():
            G2.add_node(i, color=col[u])
        for u, ws in lines:
            for w in ws:
                G2.add_edge(ids[u], ids[w])
        o2 = [(ids[u], [ids[w] for w in ws]) for u, ws in lines]
        nm = dict(zip(G2.nodes, random.Random(0).sample(gen_e3.name_pool(), G2.number_of_nodes())))
        A2 = gen_e3.encode(G2, ids[qp], nm, ser, "dia", 0, o2)
        B2 = gen_e3.encode(G2, ids[qm], nm, ser, "dia", 0, o2)
        assert not final_equal(A2, B2, 6), f"{ser}: control pair not separated"


def share(split, seed, n, ser):
    recs = list(gen_e3.pairs("dia", split, seed, n))
    out = []
    for Mx, f in SC.family(ser):
        ok = tot = 0
        for G, vp, vn, names, s, order, lab in recs:
            for v, y in ((vp, 1), (vn, 0)):
                ok += holds_final(f, gen_e3.encode(G, v, names, ser, "dia", s, order)) == bool(y)
                tot += 1
        out.append(ok / tot)
    return out


def test_shortcut():
    for split, seed in (("kset", 3), ("kid", 5)):
        for ser in ("AL", "NC"):
            sh = share(split, seed, 8, ser)
            assert max(sh) <= 0.55, (split, ser, sh)
    # the old top placement: Psi_{<=2} decides every pair
    G, vp, vn, names, s, order, lab = next(gen_e3.pairs("dia", "kset", 3, 1))
    top = [(u, ws) for u, ws in order if u in (vp, vn)] + [(u, ws) for u, ws in order if u not in (vp, vn)]
    f = dict(SC.family("AL"))[2]
    for v, y in ((vp, 1), (vn, 0)):
        assert holds_final(f, gen_e3.encode(G, v, names, "AL", "dia", s, top)) == bool(y)


def test_proof_instances():
    """Instances of the proof's construction (kappa = 1) for L = 4, 6, 8: sizes, k_min, blocking, buffers
    of at least L^2 + 2L tokens, and equal round-L types at the final position (exact)."""
    sizes = {("AL", 4): 1172, ("AL", 6): 2853, ("AL", 8): 6110, ("NC", 4): 1335, ("NC", 6): 3510, ("NC", 8): 6402}
    for (ser, L), n_tok in sizes.items():
        p = HP.proof_params(ser, L)
        assert p["N"] == 2 + 2 * (L - 1) * 2 and p["buf"] * (7 if ser == "AL" else 9) >= L * L + 2 * L
        G, vp, vn, names, s, order = next(HP.pairs(ser, 9, 1, L=L))
        A = gen_e3.encode(G, vp, names, ser, "dia", s, order)
        B = gen_e3.encode(G, vn, names, ser, "dia", s, order)
        assert len(A) == n_tok, (ser, L, len(A))
        kstar = max(0, (L - 2) // 2) if ser == "AL" else L // 5
        assert HP.kmin_dmin(G, vp, ser, s, order)[0] == kstar + 1
        acc = HP.ownline_access(G, vp, order)
        assert acc["j_prec"] is None and acc["j_occ"] is None and acc["j_inl"] >= 2, acc
        assert final_equal(A, B, L), f"{ser} L={L}: round types differ"


if __name__ == "__main__":
    test_proof_instances()
    test_hard()
    test_shortcut()
    print("test_hard_pairs: ok")
