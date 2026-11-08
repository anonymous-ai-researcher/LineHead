"""Hard pairs for the own-line routes (ownline.tex). Same gadget as Theorem lb:main (proofs12/construct.py), with the
order of the two blocks as a parameter.

  N hubs H_i, red for even i and blue for odd i (index from 0); shared entries b_1..b_{k-1}; H_i lists b_1..b_{k-1}, s_i;
  s_i lists fillers f_i^1..f_i^{J-1}, then H_i (so H_i is the J-th entry of s_i's own list);
  S-unit U_i : filler lines (each lists s_i), s_i's line, buffer;   hub unit V_i : H_i's line, buffer;  T : b-lines.

  order="s_first"  : U_1..U_N V_1..V_N T   (Theorem lb:main; NC inline route Psi is the only own-line route)
  order="hub_first": V_1..V_N U_1..U_N T   (H_i's line precedes s_i's line: AL Phi / NC Phi routes apply)
Queries: q+ = s_i with the last red hub, q- = s_i with the last blue hub; the two prompts share the serialization.
"""


def line_tokens(ser, u, color, entries, colors):
    if ser == "AL":
        body = []
        for i, e in enumerate(entries):
            body += ([","] if i else []) + ["@" + e]
        return ["@" + u, "(", color, ")", ":"] + (body or ["none"])
    body = []
    for i, e in enumerate(entries):
        body += ([","] if i else []) + ["@" + e, "(", colors[e], ")"]
    return ["@" + u, "is", color, ".", "Neighbors", ":"] + (body or ["none"]) + ["."]


def build(ser, k, N, J, buf, order="hub_first"):
    col, lines = {}, []
    bufc = [0]

    def buffer():
        for _ in range(buf):
            z = f"z{bufc[0]}"
            bufc[0] += 1
            col[z] = "blue"
            lines.append((z, []))

    H = [f"H{i}" for i in range(N)]
    S = [f"s{i}" for i in range(N)]
    F = [[f"f{i}x{j}" for j in range(J - 1)] for i in range(N)]
    Bn = [f"b{j}" for j in range(k - 1)]
    for i in range(N):
        col[H[i]] = "red" if i % 2 == 0 else "blue"
        col[S[i]] = "blue"
        for f in F[i]:
            col[f] = "blue"
    for b in Bn:
        col[b] = "blue"

    def s_units():
        for i in range(N):
            for f in F[i]:
                lines.append((f, [S[i]]))
            lines.append((S[i], F[i] + [H[i]]))
            buffer()

    def hub_units():
        for i in range(N):
            lines.append((H[i], Bn + [S[i]]))
            buffer()

    if order == "hub_first":
        hub_units()
        s_units()
    else:
        s_units()
        hub_units()
    for b in Bn:
        lines.append((b, list(H)))
    nb = {u: set(es) for u, es in lines}
    for u, es in lines:
        for e in es:
            assert u in nb[e], (u, e)
    ip = max(i for i in range(N) if col[H[i]] == "red")
    im = max(i for i in range(N) if col[H[i]] == "blue")
    return lines, col, nb, S[ip], S[im]


def tokens(ser, lines, col, q):
    toks = ["<BOS>"]
    for li, (u, es) in enumerate(lines):
        if li:
            toks.append("<NL>")
        toks += line_tokens(ser, u, col[u], es, col)
    return toks + ["<NL>", "Q", ":", "NBR", "@" + q]
