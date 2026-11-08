"""Hard pairs for the finite-depth lower bound (lower_bound.tex): one word W (AL or NC), two query vertices
q+ = s_i (red hub H_i, q+ at index k of H_i's list) and q- = s_i' (blue hub), sequences differ only in the last token.
Hubs share their first k-1 entries b_1..b_{k-1} (the 'ambiguity' device that defeats self-matching shortcuts)."""


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


def build(ser, k, N, J, buf, colors_pattern=("red", "blue"), fresh_b=False, f_before=True):
    """k: index of s_i in H_i's list; N hubs; J: index of H_i in s_i's list (J-1 fillers); buf: buffer lines
    after every line group. fresh_b: hub entries b_{i,j} local to each hub (no sharing) -- for comparison.
    Returns (lines, colors, adjacency, meta)."""
    col, adj, lines = {}, {}, []
    bufc = [0]

    def add_edge(a, b):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    def buffer():
        for _ in range(buf):
            z = f"z{bufc[0]}"
            bufc[0] += 1
            col[z] = "blue"
            adj.setdefault(z, [])
            lines.append((z, []))

    H = [f"H{i}" for i in range(N)]
    S = [f"s{i}" for i in range(N)]
    F = [[f"f{i}x{j}" for j in range(J - 1)] for i in range(N)]
    Bn = [[(f"b{i}x{j}" if fresh_b else f"b{j}") for j in range(k - 1)] for i in range(N)]
    for i in range(N):
        col[H[i]] = colors_pattern[i % len(colors_pattern)]
        col[S[i]] = "blue"
        for f in F[i]:
            col[f] = "blue"
        for b in Bn[i]:
            col[b] = "blue"
    hub_list = {i: Bn[i] + [S[i]] for i in range(N)}
    s_list = {i: F[i] + [H[i]] for i in range(N)}
    for i in range(N):
        if f_before:                     # filler lines directly before s_i's line (no route through them)
            for f in F[i]:
                lines.append((f, [S[i]]))
        lines.append((S[i], s_list[i]))
        buffer()
    for i in range(N):
        lines.append((H[i], hub_list[i]))
        buffer()
    for i in range(N):
        if f_before:
            continue
        for f in F[i]:
            lines.append((f, [S[i]]))
        if F[i]:
            buffer()
    bs = sorted({b for i in range(N) for b in Bn[i]}, key=lambda b: (len(b), b))
    for b in bs:
        lines.append((b, [H[i] for i in range(N) if b in Bn[i]]))
    # adjacency (undirected) check: every listed pair is symmetric
    nb = {u: set(es) for u, es in lines}
    for u, es in lines:
        for e in es:
            assert u in nb[e], (u, e)
    return lines, col, nb, dict(H=H, S=S)


def tokens(ser, lines, col, q):
    toks = ["<BOS>"]
    for li, (u, es) in enumerate(lines):
        if li:
            toks.append("<NL>")
        toks += line_tokens(ser, u, col[u], es, col)
    return toks + ["<NL>", "Q", ":", "NBR", "@" + q]
