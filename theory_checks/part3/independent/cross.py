import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'part2'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import el_depth1 as ed
from tl_eval import holds_final, M, AND, Y, T, NAME, OR
from type_check import size_PE, separates, mask
from check_leap import al_tokens, SINCE_AL, SINCE_NC

# 1. Theta_{E0} vs type checker (d=1, kappa=E0+2, r=1) on H_s
for E0 in (4, 6, 10):
    th = ed.theta_bounded(E0)
    row = []
    for s in range(1, 13):
        A, B = ed.hard_pair(*size_PE(s))
        ev = holds_final(th, A) != holds_final(th, B)
        tc = separates(A, B, 1, E0 + 2, 1) is not None
        row.append((s, ev, tc, size_PE(s)[1] + 1 <= E0))
        assert (not ev) or tc, (E0, s)
    print("E0", E0, [(s, int(ev), int(tc), int(pred)) for s, ev, tc, pred in row])

# 2. other formats on the same graphs/orders
ECL_F = OR(M(0, 0, AND(Y(T("--")), Y(T("red"), 2))), M(0, 4, AND(T("red"), Y(NAME), Y(T("--"), 2))))
def parse(A):
    i = A.index("<NL>")
    cl = A[3:i]
    order, colors = [], {}
    for k in range(0, len(cl), 3):
        order.append(cl[k][1:]); colors[cl[k][1:]] = cl[k + 1]
    edges = []
    rest = A[i + 3:]
    j = 0
    while rest[j] == "<NL>" and rest[j + 2] == "--":
        edges.append((rest[j + 1][1:], rest[j + 3][1:])); j += 4
    return order, colors, edges
def ecl_tokens(order, colors, edges, q):
    t = ["<BOS>", "Colors", ":"]
    for i, u in enumerate(order):
        t += ([","] if i else []) + ["@" + u, colors[u]]
    t += ["<NL>", "Edges", ":"]
    for a, b in edges:
        t += ["<NL>", "@" + a, colors[a], "--", "@" + b, colors[b]]
    return t + ["<NL>", "Q", ":", "NBR", "@" + q]
for s in range(1, 9):
    A, B = ed.hard_pair(*size_PE(s))
    order, colors, edges = parse(A)
    nb = {u: [] for u in order}
    for a, b in edges:
        nb[a].append(b); nb[b].append(a)
    res = []
    for name, mk, f in (("AL", lambda q: al_tokens(order, colors, nb, q, "AL"), SINCE_AL),
                        ("NC", lambda q: al_tokens(order, colors, nb, q, "NC"), SINCE_NC),
                        ("ECL", lambda q: ecl_tokens(order, colors, edges, q), ECL_F)):
        a, b = mk("vp"), mk("vm")
        assert mask(a) == mask(b)
        res.append((name, holds_final(f, a), holds_final(f, b)))
    print("s", s, len(edges), "edges", res)
