"""Numeric checks for the budget/leap proposition (part3/leap.tex).

A. For each budget beta=(r,kappa,d) of the match arguments, build the pair (A_beta, B_beta) of Theorem md:el with the
   smallest P, E the proof allows; report its number of edges m_beta against the bound 3(c+r)+5 (c=1+d(kappa+2));
   sample random formulas of match-nesting depth <=1 within the budget: none may separate the pair; the depth-2
   formula of Prop. prop:shift(b) (constant budget) separates it; a shaped depth-2 control separates it at a positive rate.
B. Theta_{E0} (depth 1, budget r=1, kappa=E0+2, d=2) is correct on random EL prompts with <= E0 edges and, as the
   proposition requires, errs on the pair built for its own budget.
C. Random graphs (n in [8,16] and [17,32]): accuracy over all (prompt, query) cases of Theta_{E0} for growing E0 (EL),
   of the depth-2 formula (EL), and of the constant-budget depth-1 formulas of Prop. prop:since (AL, NC).
Usage: python3 -I check_leap.py
"""
import os, random, sys, math
R13 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part2')
sys.path.insert(0, R13)
from tl_eval import AND, M, NAME, NOT, OR, P, T, Y, holds_final  # noqa: E402
import el_depth1 as ed  # noqa: E402

NL = ed.NL
ATOMS = ed.ATOMS


def size_PE(s):
    """pair H_s of leap.tex: P_s = smallest even >= 3s, E_s = smallest even >= max(s+1, P_s/2)."""
    Pp = 3 * s; Pp += Pp % 2
    E = max(s + 1, Pp // 2); E += E % 2
    return Pp, E


def budget_PE(r, kappa, d):
    c = 1 + d * (kappa + 2)
    Pp, E = size_PE(c + r)
    return c, Pp, E


def rand_blind_b(rng, d, kappa):
    if d == 0 or rng.random() < 0.25:
        return rng.choice(ATOMS)
    op = rng.choice(["not", "and", "or", "Y", "P", "S"])
    if op == "not":
        return NOT(rand_blind_b(rng, d - 1, kappa))
    if op in ("and", "or"):
        return (op, rand_blind_b(rng, d - 1, kappa), rand_blind_b(rng, d - 1, kappa))
    if op == "Y":
        return Y(rand_blind_b(rng, d - 1, kappa))
    if op == "P":
        return P(rand_blind_b(rng, d - 1, kappa), rng.randint(1, kappa))
    return ("S", rand_blind_b(rng, d - 1, kappa), rand_blind_b(rng, d - 1, kappa))


def rand_d1(rng, depth, r, kappa, d):
    """match-nesting depth <= 1; every match argument is name-blind with operator depth <= d, thresholds <= kappa,
    offsets <= r; outer structure unrestricted (outer thresholds up to 5, match thresholds up to 3)."""
    if depth == 0 or rng.random() < 0.15:
        return rng.choice(ATOMS)
    op = rng.choice(["not", "and", "or", "Y", "P", "S", "M", "M", "M"])
    if op == "M":
        return M(rng.randint(0, r), rng.randint(0, r), rand_blind_b(rng, d, kappa), rng.randint(1, 3))
    if op == "not":
        return NOT(rand_d1(rng, depth - 1, r, kappa, d))
    if op in ("and", "or"):
        return (op, rand_d1(rng, depth - 1, r, kappa, d), rand_d1(rng, depth - 1, r, kappa, d))
    if op == "Y":
        return Y(rand_d1(rng, depth - 1, r, kappa, d))
    if op == "P":
        return P(rand_d1(rng, depth - 1, r, kappa, d), rng.randint(1, 5))
    return ("S", rand_d1(rng, depth - 1, r, kappa, d), rand_d1(rng, depth - 1, r, kappa, d))


def al_tokens(order, colors, nbrs, q, ser):
    t = ["<BOS>"]
    for i, u in enumerate(order):
        if i:
            t.append(NL)
        L = nbrs[u]
        if ser == "AL":
            t += ["@" + u, "(", colors[u], ")", ":"]
            if not L:
                t.append("none")
            for j, x in enumerate(L):
                t += ([","] if j else []) + ["@" + x]
        else:  # NC
            t += ["@" + u, "is", colors[u], ".", "Neighbors", ":"]
            if not L:
                t.append("none")
            for j, x in enumerate(L):
                t += ([","] if j else []) + ["@" + x, "(", colors[x], ")"]
            t.append(".")
    return t + [NL, "Q", ":", "NBR", "@" + q]


SINCE_AL = M(0, 0, ("S", NOT(T(NL)), AND(T("red"), Y(T("(")))))
SINCE_NC = M(0, 0, ("S", NOT(T(NL)), AND(T("red"), Y(T("is")))))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - h, ctr + h


def part_A(rng, nform):
    print("[A] budget pairs (theorem: 0 separations by depth-1 formulas within budget)")
    for (r, kappa, d) in [(0, 1, 1), (1, 1, 1), (1, 2, 1), (2, 1, 2), (2, 2, 2), (3, 3, 2), (1, 1, 3), (3, 2, 3)]:
        c, Pp, E = budget_PE(r, kappa, d)
        A, B = ed.hard_pair(Pp, E)
        m = 2 * E + 2
        bound = 3 * (c + r) + 5
        sep = sum(holds_final(f, A) != holds_final(f, B)
                  for f in (rand_d1(rng, rng.randint(2, 6), r, kappa, d) for _ in range(nform)))
        nested = (holds_final(ed.NESTED, A), holds_final(ed.NESTED, B))
        shaped = 0
        for _ in range(nform):
            f = M(0, rng.randint(0, 3), AND(ed.rand_blind(rng, 2), M(rng.randint(0, 3), rng.randint(0, 3),
                                                                 T(rng.choice(["red", "blue"])))))
            shaped += holds_final(f, A) != holds_final(f, B)
        print(f"  r={r} kappa={kappa} d={d}: c={c} P={Pp} E={E}; edges m={m} <= bound {bound}: {m <= bound}; "
              f"{len(A)} tokens; depth-1 separations {sep}/{nform}; depth-2 formula (+,-)={nested}; "
              f"shaped depth-2 control separations {shaped}/{nform}")


def part_B(rng):
    print("[B] Theta_{E0}: correct up to E0 edges, errs on the pair built for its own budget")
    for E0 in (2, 4, 6):
        th = ed.theta_bounded(E0)
        wrong = cases = 0
        for _ in range(150):
            V, colors, edges, order = ed.random_el(rng, 8, E0)
            nb = {u: set() for u in V}
            for a, b in edges:
                nb[a].add(b); nb[b].add(a)
            for q in V:
                cases += 1
                wrong += holds_final(th, ed.el_tokens(order, colors, edges, q)) != any(colors[x] == "red" for x in nb[q])
        Pp, E = size_PE(E0 + 6)          # budget of Theta_{E0}: r=1, kappa=E0+2, d=1 -> beta = E0+6
        A, B = ed.hard_pair(Pp, E)
        print(f"  E0={E0}: {wrong} errors on {cases} cases with <= {E0} edges; own-budget pair (m={2*E+2} edges): "
              f"Theta gives + {holds_final(th, A)}, - {holds_final(th, B)} (truth True, False)")


def rand_graph(rng, lo, hi):
    n = rng.randint(lo, hi)
    V = [f"x{i}" for i in range(n)]
    colors = {u: ("red" if rng.random() < 0.3 else "blue") for u in V}
    p = 2.2 / (n - 1)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((V[i], V[j]) if rng.random() < 0.5 else (V[j], V[i]))
    rng.shuffle(edges)
    order = V[:]; rng.shuffle(order)
    return V, colors, edges, order


def add(stats, key, pred, truth):
    st = stats.setdefault(key, [0, 0, 0, 0])      # TP, P, TN, N
    if truth:
        st[0] += pred == truth; st[1] += 1
    else:
        st[2] += pred == truth; st[3] += 1


def part_C(rng, ngraph):
    print("[C] random graphs: balanced accuracy over (prompt, query) cases (true positive and true negative rates)")
    for lo, hi in ((8, 16), (17, 32)):
        stats = {}
        mlist = []
        for _ in range(ngraph):
            V, colors, edges, order = rand_graph(rng, lo, hi)
            mlist.append(len(edges))
            nb = {u: [] for u in V}
            for a, b in edges:
                nb[a].append(b); nb[b].append(a)
            for u in V:
                rng.shuffle(nb[u])
            for q in rng.sample(V, min(4, len(V))):
                truth = any(colors[x] == "red" for x in nb[q])
                w = ed.el_tokens(order, colors, edges, q)
                for E0 in (2, 4, 8, 16, 32):
                    key = f"EL Theta_{E0} (depth 1, kappa={E0+2})"
                    add(stats, key, holds_final(ed.theta_bounded(E0), w), truth)
                key = "EL depth-2 formula (constant budget)"
                add(stats, key, holds_final(ed.NESTED, w), truth)
                for ser, f in (("AL", SINCE_AL), ("NC", SINCE_NC)):
                    key = f"{ser} since formula (depth 1, constant budget)"
                    wa = al_tokens(order, colors, nb, q, ser)
                    add(stats, key, holds_final(f, wa), truth)
        print(f"  n in [{lo},{hi}]: edges mean {sum(mlist)/len(mlist):.1f}, max {max(mlist)}")
        for k, (tp, np_, tn, nn) in stats.items():
            print(f"    {k}: TPR {tp}/{np_} = {tp/np_:.3f}, TNR {tn}/{nn} = {tn/nn:.3f}, balanced {(tp/np_+tn/nn)/2:.3f}")


def part_D(rng, J=12, nform=200):
    """mixture D_J: s uniform on 1..J; a depth-1 formula of budget b has |E[f y]| <= (b-1)/(2J) (y = +-1)."""
    print(f"[D] mixture D_J (J={J}): empirical |E[f y]| against the bound (b-1)/(2J)")
    pairs = {s: ed.hard_pair(*size_PE(s)) for s in range(1, J + 1)}
    worst = -1.0
    for _ in range(nform):
        r = rng.randint(0, 2); kappa = rng.randint(1, 2); d = rng.randint(1, 2)
        b = 1 + d * (kappa + 2) + r
        f = rand_d1(rng, rng.randint(2, 6), r, kappa, d)
        cov = sum((holds_final(f, A) - holds_final(f, B)) / 2 for A, B in pairs.values()) / J
        worst = max(worst, abs(cov) - (b - 1) / (2 * J))
    print(f"  {nform} random depth-1 formulas: max(|E[f y]| - bound) = {worst:.4f} (theorem: <= 0)")
    for E0 in (6, 10, 14):
        th = ed.theta_bounded(E0); b = E0 + 6
        cov = sum((holds_final(th, A) - holds_final(th, B)) / 2 for A, B in pairs.values()) / J
        print(f"  Theta_{E0} (budget {b}): E[f y] = {cov:.4f}, bound (b-1)/(2J) = {min(0.5,(b-1)/(2*J)):.4f}")
    cov2 = sum((holds_final(ed.NESTED, A) - holds_final(ed.NESTED, B)) / 2 for A, B in pairs.values()) / J
    print(f"  depth-2 formula: E[f y] = {cov2:.3f} (maximum 0.5)")


if __name__ == "__main__":
    rng = random.Random(20261104)
    if os.environ.get("ONLY_D"):
        part_D(rng); sys.exit()
    nform = int(os.environ.get("NFORM", "400"))
    part_A(rng, nform)
    part_B(rng)
    part_C(rng, int(os.environ.get("NGRAPH", "60")))
    part_D(rng)
