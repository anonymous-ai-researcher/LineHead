"""Hard pairs: the adversarial E3 test set built from the construction of Theorem lb:main (finite-depth
lower bound; proofs12/construct.py and check_pairs.py by the logician, smallest instances of small_pairs.py).

One structure per format, for round depth L = 6 (the depth of the E3 models), threshold cap kappa = 1:
  AL: N = 10 hubs, k = k*_AL(6)+1 = 3, J = 2, one buffer line after every unit; 465 tokens, maximum degree 10.
  NC: N = 12 hubs, k = k*_NC(6)+1 = 2, J = 2, no buffer lines; 624 tokens, maximum degree 12.
Hubs H_0..H_{N-1} alternate red/blue and share their first k-1 entries b_1..b_{k-1}; hub H_i lists s_i k-th; s_i
lists J-1 fillers and then H_i; filler lines come right before s_i's line; S-units first, hub units next, b-lines
last. The queries are s_i of the last red hub (positive, k_min = k) and of the last blue hub (negative); the two
prompts are one serialization and differ only in the query token. By Theorem lb:main no formula of
TL[P^{>=k},Y,M^{>=k}] with thresholds <= kappa and round depth <= L separates them (checked exactly with rtypes.py
in test_hard_pairs.py); a since step separates them.
A pair of the test set is this structure with fresh names (5-letter names of gen_e3.name_pool, drawn per pair), so
all pairs of a format are isomorphic; the set measures the answer of a model on one hard structure under 1,000
namings. Records have the format of gen_e3.py (fields tokens, label, pair, split = "hard", k_min, ...).
--proof-L L writes instances of the proof's construction itself (proof_params: N = 2+2(L-1)(kappa+1),
buffers of at least L^2+2L tokens, kappa = 1) for L = 4, 6, 8: 1,172 / 2,853 / 6,110 tokens under AL and 1,335 /
3,510 / 6,402 under NC (L = 12 would need about 17,500 tokens and is not generated). Each is tested by the E3 models
of depth L (the depth sweep for L = 4, 8; the main runs for L = 6).
Usage: python hard_pairs.py --ser AL --pairs 1000 --seed 8 --out data/dia_AL_hard.jsonl
       python hard_pairs.py --ser AL --pairs 500 --seed 9 --proof-L 6 --out data/dia_AL_hardp6.jsonl
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "m0"))
from serialize import kmin_dmin, ownline_access  # noqa: E402

import gen_e3  # noqa: E402

PARAMS = {"AL": dict(L=6, k=3, N=10, J=2, buf=1), "NC": dict(L=6, k=2, N=12, J=2, buf=0)}


def proof_params(ser, L, kappa=1):
    """The parameters of the proof of Theorem lb:main for round depth L: k = k*(L)+1, J = J_F(L),
    N = 2 + 2(L-1)(kappa+1) hubs (even), and buffers of at least B = L^2 + 2L tokens (a buffer line has 7 tokens
    under AL, 9 under NC, line break included)."""
    kstar = max(0, (L - 2) // 2) if ser == "AL" else L // 5
    J = 2 if ser == "AL" else max(2, (L - 4) // 5 + 1)
    N = 2 + 2 * (L - 1) * (kappa + 1)
    N += N % 2
    per = 7 if ser == "AL" else 9
    buf = -(-(L * L + 2 * L) // per)
    return dict(L=L, k=kstar + 1, N=N, J=J, buf=buf)


def build(k, N, J, buf):
    """Lines [(u, [w, ...])] and colors of the construction (proofs12/construct.py, f_before=True, shared b)."""
    col, lines = {}, []
    z = [0]

    def buffer():
        for _ in range(buf):
            u = f"z{z[0]}"
            z[0] += 1
            col[u] = "blue"
            lines.append((u, []))
    H = [f"H{i}" for i in range(N)]
    S = [f"s{i}" for i in range(N)]
    F = [[f"f{i}x{j}" for j in range(J - 1)] for i in range(N)]
    B = [f"b{j}" for j in range(k - 1)]
    for i in range(N):
        col[H[i]] = ("red", "blue")[i % 2]
        col[S[i]] = "blue"
        for f in F[i]:
            col[f] = "blue"
    for b in B:
        col[b] = "blue"
    for i in range(N):
        for f in F[i]:
            lines.append((f, [S[i]]))
        lines.append((S[i], F[i] + [H[i]]))
        buffer()
    for i in range(N):
        lines.append((H[i], B + [S[i]]))
        buffer()
    for b in sorted(B, key=lambda b: (len(b), b)):
        lines.append((b, list(H)))
    ip = max(i for i in range(N) if col[H[i]] == "red")
    im = max(i for i in range(N) if col[H[i]] == "blue")
    return lines, col, S[ip], S[im]


def structure(ser, p=None):
    p = p or PARAMS[ser]
    lines, col, qp, qm = build(p["k"], p["N"], p["J"], p["buf"])
    ids = {u: i for i, (u, _) in enumerate(lines)}
    G = nx.Graph()
    for u, i in ids.items():
        G.add_node(i, color=col[u])
    for u, ws in lines:
        for w in ws:
            G.add_edge(ids[u], ids[w])
    order = [(ids[u], [ids[w] for w in ws]) for u, ws in lines]
    return G, order, ids[qp], ids[qm], ids


def pairs(ser, seed, n, pool=None, L=None):
    """L=None: the small machine-checked pairs; L = 4, 6, 8: instances of the proof's construction."""
    pool = pool or gen_e3.name_pool()
    G, order, vp, vn, _ = structure(ser, proof_params(ser, L) if L else None)
    rng = random.Random(f"hard-{ser}-{seed}" if L is None else f"hardp-{ser}-L{L}-{seed}")
    for _ in range(n):
        names = dict(zip(G.nodes, rng.sample(pool, G.number_of_nodes())))
        yield G, vp, vn, names, rng.randrange(2**31), order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ser", choices=["AL", "NC"], required=True)
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--proof-L", type=int, default=None, help="instances of the proof's construction for round depth L")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("E2E_TINY"):
        a.pairs = min(a.pairs, 4 if a.proof_L is None else 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for k, (G, vp, vn, names, s, order) in enumerate(pairs(a.ser, a.seed, a.pairs, L=a.proof_L)):
            km, dm = kmin_dmin(G, vp, a.ser, s, order)
            acc = ownline_access(G, vp, order)
            for v, lab in ((vp, 1), (vn, 0)):
                rec = {"tokens": gen_e3.encode(G, v, names, a.ser, "dia", s, order), "label": lab, "pair": k,
                       "split": "hard" if a.proof_L is None else f"hardp{a.proof_L}", "n": G.number_of_nodes(), "maxdeg": max(d for _, d in G.degree),
                       "deg_v": G.degree(v), "k_min": km if lab else None, "d_min": dm if lab else None}
                if lab:
                    rec.update(acc)
                f.write(json.dumps(rec) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
