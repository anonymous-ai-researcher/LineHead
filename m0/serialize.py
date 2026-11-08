"""Text serializations of a colored graph (paper §2 and appendix "Complete Preliminaries"; CONFIG.md §1.2, §5.6).

Main formats
  EL       edge list:      "Colors: a red, b blue, ..." line, then "Edges:" and one line "a -- b" per edge.
  AL       adjacency list: one line "u (red): w1, w2" per vertex ("none" if isolated).
  NC       node-centric:   one line "u is red. Neighbors: w1 (blue), w2 (red)." per vertex.
  ECL      edge list with inline colors (new syntax, D1): EL color line, edge lines "a red -- b blue".
Ablations (E4)
  EL2      EL with each edge listed in both orientations.
  ELfar    EL, then a color block "Edge colors:" whose j-th line gives the colors of edge j's endpoints.
  ELB      (E33) EL with block colors: the edges in blocks of 3, each block preceded by a line
           "Colors: ..." with the colors of the block's endpoints (and a last "Colors:" line for isolated
           vertices). Every color stands within a few lines of its edge, but reading it still matches the
           neighbor's name inside the edge line: the two nested matches of EL at the distance of ECL.
  NCH      NC without head colors: "u. Neighbors: w1 (blue), w2 (red)."
  NCnw     NC with the words "is" and "Neighbors" replaced by the non-words "dax" and "Wug" (familiarity
           control): "u dax red. Wug: w1 (blue), w2 (red)." Same token positions as NC.
Natural-language encodings of Fatemi et al. (2024), adapted to colored graphs and to the token rule
  Incident   (AL class) "Node u (red) is connected to nodes w1, w2." per vertex ("... connected to no nodes.")
  Friendship (EL class) "G describes a friendship graph among a (red), b (blue), ...", then "We have the following
             edges in G:" and one line "a and b are friends." per edge
Held out
  GraphML  node records with a color attribute, then edge records.

Token rule (F-1): every line starts with a single space, so every name occurrence is preceded by a space
(" ash") and is the same token wherever it occurs. Lines are joined with "\\n ". GraphML ids are written as
id=" ash" (a leading space inside the attribute value; this deviates from the NMTOKEN type of the GraphML
schema, see RUN_PROTOCOL.md).

Partial masking (E7): head_names / entry_names override the names used for line heads and for list
entries in AL, NC and NCH (default: `names`). Order choices are drawn from a seeded RNG, identically for all
name maps, so masked and unmasked prompts of a graph share the same order.
"""
import random

import networkx as nx

SERIALIZATIONS = ["EL", "AL", "NC"]
ALL_FORMATS = ["EL", "AL", "NC", "ECL", "EL2", "ELfar", "ELB", "NCH", "NCnw", "Incident", "Friendship", "GraphML"]
ELB_BLOCK = 3
NONWORD = {"is": "dax", "Neighbors": "Wug"}      # NCnw
NL = "\n "


def _join(lines):
    return " " + NL.join(lines)


def _ordered_nodes(G, rng, sort=False):
    nodes = list(G.nodes)
    if not sort:
        rng.shuffle(nodes)
    return nodes


def _ordered_edges(G, rng, sort_by=None):
    edges = [tuple(rng.sample(list(e), 2)) for e in G.edges]   # random orientation
    rng.shuffle(edges)
    if sort_by is not None:                                     # E8: EL sorted by source vertex
        pos = {u: i for i, u in enumerate(sort_by)}
        edges = [(a, b) if pos[a] <= pos[b] else (b, a) for a, b in edges]
        edges.sort(key=lambda e: (pos[e[0]], pos[e[1]]))
    return edges


def _col(G, u):
    return G.nodes[u]["color"]


def edge_list(G, names, rng, both=False, sort_edges=False, far=False, inline=False):
    nodes = _ordered_nodes(G, rng)
    edges = _ordered_edges(G, rng, sort_by=nodes if sort_edges else None)
    if both:
        edges = [e for a, b in edges for e in ((a, b), (b, a))]
    colors = ", ".join(f"{names[u]} {_col(G, u)}" for u in nodes)
    lines = [f"Colors: {colors}", "Edges:"]
    if inline:
        lines += [f"{names[a]} {_col(G, a)} -- {names[b]} {_col(G, b)}" for a, b in edges]
    else:
        lines += [f"{names[a]} -- {names[b]}" for a, b in edges]
    if far:
        lines += ["Edge colors:"] + [f"{_col(G, a)} {_col(G, b)}" for a, b in edges]
    return _join(lines)


def edge_blocks(G, names, rng, block=None):
    """ELB (E33): edges in blocks of ELB_BLOCK, each preceded by the colors of its endpoints (in random order)."""
    block = block or ELB_BLOCK
    edges = _ordered_edges(G, rng)
    lines = []
    for i in range(0, len(edges), block):
        chunk = edges[i:i + block]
        ends = list(dict.fromkeys(u for e in chunk for u in e))
        rng.shuffle(ends)
        lines.append("Colors: " + ", ".join(f"{names[u]} {_col(G, u)}" for u in ends))
        lines += [f"{names[a]} -- {names[b]}" for a, b in chunk]
    iso = [u for u in _ordered_nodes(G, rng) if G.degree(u) == 0]
    if iso or not edges:
        lines.append("Colors: " + ", ".join(f"{names[u]} {_col(G, u)}" for u in iso))
    return _join(lines)


def _lists(G, rng, order=None):
    """Vertex order and, per vertex, a neighbor order (shared by AL, NC, NCH). An explicit `order`
    [(u, [w...]), ...] (twin pairs of E7, position-controlled E3 sets) is used as given; the rng is still
    advanced as usual so that nothing else depends on whether an order was given."""
    if order is not None:
        _lists(G, rng)
        return [(u, list(ws)) for u, ws in order]
    order = []
    for u in _ordered_nodes(G, rng):
        nbrs = list(G[u])
        rng.shuffle(nbrs)
        order.append((u, nbrs))
    return order


def adjacency_list(G, names, rng, head_names=None, entry_names=None, order=None):
    hn, en = head_names or names, entry_names or names
    lines = []
    for u, nbrs in _lists(G, rng, order):
        nb = ", ".join(en[w] for w in nbrs) if nbrs else "none"
        lines.append(f"{hn[u]} ({_col(G, u)}): {nb}")
    return _join(lines)


def node_centric(G, names, rng, head_names=None, entry_names=None, head_colors=True, order=None, words=None):
    hn, en = head_names or names, entry_names or names
    w_is, w_nb = (words or {}).get("is", "is"), (words or {}).get("Neighbors", "Neighbors")
    lines = []
    for u, nbrs in _lists(G, rng, order):
        nb = ", ".join(f"{en[w]} ({_col(G, w)})" for w in nbrs) if nbrs else "none"
        head = f"{hn[u]} {w_is} {_col(G, u)}." if head_colors else f"{hn[u]}."
        lines.append(f"{head} {w_nb}: {nb}.")
    return _join(lines)


def incident(G, names, rng, order=None):
    lines = []
    for u, nbrs in _lists(G, rng, order):
        nb = ("nodes " + ", ".join(names[w] for w in nbrs)) if nbrs else "no nodes"
        lines.append(f"Node {names[u]} ({_col(G, u)}) is connected to {nb}.")
    return _join(lines)


def friendship(G, names, rng):
    nodes = _ordered_nodes(G, rng)
    edges = _ordered_edges(G, rng)
    among = ", ".join(f"{names[u]} ({_col(G, u)})" for u in nodes)
    lines = [f"G describes a friendship graph among {among}.", "We have the following edges in G:"]
    lines += [f"{names[a]} and {names[b]} are friends." for a, b in edges]
    return _join(lines)


def graphml(G, names, rng):
    nodes = _ordered_nodes(G, rng)
    edges = _ordered_edges(G, rng)
    lines = ['<graphml>', '<key id="color" for="node" attr.name="color" attr.type="string"/>',
             '<graph edgedefault="undirected">']
    lines += [f'<node id=" {names[u]}"><data key="color">{_col(G, u)}</data></node>' for u in nodes]
    lines += [f'<edge source=" {names[a]}" target=" {names[b]}"/>' for a, b in edges]
    lines += ['</graph>', '</graphml>']
    return _join(lines)


def serialize(G, names, kind, seed, head_names=None, entry_names=None, sort_edges=False, order=None):
    """`order`: explicit line and neighbor order for the list formats (AL, NC, NCH, NCnw); ignored otherwise."""
    rng = random.Random(seed)
    if kind == "EL":
        return edge_list(G, names, rng, sort_edges=sort_edges)
    if kind == "EL2":
        return edge_list(G, names, rng, both=True)
    if kind == "ELfar":
        return edge_list(G, names, rng, far=True)
    if kind == "ECL":
        return edge_list(G, names, rng, inline=True)
    if kind == "ELB":
        return edge_blocks(G, names, rng)
    if kind == "AL":
        return adjacency_list(G, names, rng, head_names, entry_names, order=order)
    if kind == "NC":
        return node_centric(G, names, rng, head_names, entry_names, order=order)
    if kind == "NCH":
        return node_centric(G, names, rng, head_names, entry_names, head_colors=False, order=order)
    if kind == "NCnw":
        return node_centric(G, names, rng, head_names, entry_names, order=order, words=NONWORD)
    if kind == "Incident":
        return incident(G, names, rng, order=order)
    if kind == "Friendship":
        return friendship(G, names, rng)
    if kind == "GraphML":
        return graphml(G, names, rng)
    raise ValueError(kind)


LIST_FORMATS = ("AL", "NC", "NCH", "NCnw", "Incident")


def list_positions(G, kind, seed, order=None):
    """For AL/NC/NCH/NCnw: the line order and neighbor orders that serialize() uses with this seed, as
    [(head, [entries...]), ...]. Used to compute k_min and d_min (CONFIG §3.3, §5.5)."""
    rng = random.Random(seed)
    if kind not in LIST_FORMATS:
        raise ValueError(kind)
    return _lists(G, rng, order)


def ownline_access(G, v, lists):
    """Own-line routes of a positive query v (App. new:prop-rounds) under a list order [(u, [w...])...]:
    j_prec: smallest position in v's own list of a red neighbor u whose line precedes v's line (AL own-line route);
    j_occ:  smallest position of a red neighbor u with some occurrence (head or entry) before v's line (NC nested);
    j_inl:  smallest position of any red neighbor in v's list (NC inline-color route). None if there is none."""
    line = {u: i for i, (u, _) in enumerate(lists)}
    nb = dict(lists)
    lv = line[v]
    early = {w for u, ws in lists[:lv] for w in ws} | {u for u, _ in lists[:lv]}
    reds = [(i, u) for i, u in enumerate(nb[v], start=1) if _col(G, u) == "red"]
    jp = [i for i, u in reds if line[u] < lv]
    jo = [i for i, u in reds if u in early]
    return dict(j_prec=min(jp) if jp else None, j_occ=min(jo) if jo else None, j_inl=reds[0][0] if reds else None)


def kmin_dmin(G, v, kind, seed, order=None):
    """k_min: smallest 1-based position of v in the neighbor list of a red neighbor (None if v has no red
    neighbor). d_min: token distance (word-level tokens) from that entry to its line's head color token,
    minimized over red neighbors. Word-level tokenization: names, color words and punctuation are tokens."""
    if kind not in LIST_FORMATS:
        return None, None      # EL-type formats have no lines with heads (instances.py uses the matched NC k_min)
    best = None
    for u, nbrs in list_positions(G, kind, seed, order):
        if _col(G, u) != "red" or v not in nbrs:
            continue
        k = nbrs.index(v) + 1
        if kind == "AL":       # "u ( c ) : w1 , w2 ...": color at h+2, entry k at h+3+2k
            d = 1 + 2 * k
        elif kind == "Incident":   # "Node u ( c ) is connected to nodes w1 , ...": color at h+2, entry k at h+6+2k
            d = 4 + 2 * k
        else:                  # "u is c . Neighbors : w1 ( c1 ) , ...": color at h+2, entry k at h+1+5k (NCnw alike)
            d = 5 * k - 1
        if best is None or k < best[0]:
            best = (k, d)
    return best if best else (None, None)
