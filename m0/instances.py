"""Turn stored instances into prompts: one prompt per (instance, serialization, template).

Options (CONFIG.md §5)
  mask        None | "all" (E2, every name incl. the query becomes MASK_WORD)
              | "heads" | "entries" (E7, only line-head names / only list-entry names, AL, NC, NCH; query kept)
              | "rand_heads" | "rand_entries" (E7 control: as many randomly chosen name occurrences of the
                serialization as "heads" / "entries" mask, drawn per graph, so both prompts of a pair share them)
  ids         "names" (default) | "int" (E8: integer identifiers 0..n-1)
              | "compound" (E16: two pool names joined by a hyphen, several tokens per name)
              | "num4" (E16: distinct random four-digit integers, several tokens per name in every tokenizer)
Record fields of (CONFIG §6.10): "filler" = {"pos": "between" | "before", "words": F} inserts F word tokens
of a neutral line ("Note: this line is filler text.") between the color line and the edge lines of EL and ECL, or
before the color line (E14); "ser_only" restricts a record to one serialization (E11b-L, whose padded graph
differs per format).
  question_first  E8
  sort_edges      E8 (EL sorted by source vertex)
Every prompt record carries k_min / d_min of the query for its serialization (CONFIG §3.3, §5.5).
"""
import json
import random

import networkx as nx

from names import POOL as NAME_POOL
from prompts import TEMPLATES, render
from serialize import NL, SERIALIZATIONS, kmin_dmin, serialize
from tasks import BY_ID

SER_OFFSET = {"EL": 101, "AL": 202, "NC": 303, "ECL": 404, "ELB": 808, "GraphML": 505, "EL2": 606, "ELfar": 707, "NCH": 303,
              "NCnw": 303, "Incident": 202, "Friendship": 101}
# NCH and NCnw use the NC offset so that their prompts of a graph share all order choices with NC (E4 comparison);
# Incident shares the AL orders and Friendship the EL orders (their classes).
LISTS = ("AL", "NC", "NCH", "NCnw", "Incident")


def random_mask(text, names, n_mask, seed):
    """Replace n_mask randomly chosen name occurrences of a serialization by MASK_WORD (E7 random-mask control)."""
    import re
    pat = re.compile(r"(?<= )(" + "|".join(sorted(map(re.escape, set(names.values())), key=len, reverse=True)) +
                     r")(?=[ ,.:()\n\"]|$)")
    occ = [m.span() for m in pat.finditer(text)]
    pick = set(random.Random(seed).sample(range(len(occ)), min(n_mask, len(occ))))
    out, last = [], 0
    for i, (a, b) in enumerate(occ):
        if i in pick:
            out += [text[last:a], MASK_WORD]
            last = b
    return "".join(out) + text[last:]


def rel_pos(G, v, kind, text, names, seed, order=None):
    """C2 covariate: relative position (0 = first line, 1 = last line of the serialization) of the line that a
    depth-one positive needs: under list formats the line of the red neighbor in whose list v is earliest (k_min);
    under edge formats and GraphML the first line naming v and a red neighbor. None for negatives."""
    from serialize import list_positions
    lines = text.split("\n ")
    reds = [u for u in G[v] if G.nodes[u]["color"] == "red"]
    if not reds:
        return None
    if kind in LISTS:
        lp = list_positions(G, kind, seed, order)
        best = min(((ws.index(v), i) for i, (u, ws) in enumerate(lp) if u in reds), default=None)
        return None if best is None else round(best[1] / max(1, len(lp) - 1), 4)
    nv = names[v]
    want = {names[u] for u in reds}
    import re
    hdr = [i for i, ln in enumerate(lines) if ln in ("Edges:", "We have the following edges in G:")]
    first = hdr[0] + 1 if hdr else 0                                       # skip the color line (edge formats)
    for i, ln in enumerate(lines):
        if i < first or (kind == "GraphML" and "<edge" not in ln):
            continue
        toks = set(re.findall(r"[^\s,.:()\"<>=/-]+", ln))
        if nv in toks and toks & want:
            return round(i / max(1, len(lines) - 1), 4)
    return None
MASK_WORD = "kit"


FILLER = "Note: this line is filler text."       # 8 word tokens, no name, no color word, no answer word


def add_filler(text, pos, words):
    """E14: insert ceil(words / 8) neutral lines between the color line and "Edges:" (pos="between") or before the
    color line (pos="before") of an EL or ECL serialization."""
    if not words:
        return text
    n = -(-words // 8)
    lines = text[1:].split(NL)
    i = lines.index("Edges:") if pos == "between" else 0
    lines[i:i] = [FILLER] * n
    return " " + NL.join(lines)


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def to_graph(rec):
    G = nx.Graph()
    G.add_nodes_from(range(rec["n"]))
    G.add_edges_from(map(tuple, rec["edges"]))
    for u, c in enumerate(rec["colors"]):
        G.nodes[u]["color"] = c
    return G


def assign_names(rec, name_pool, ids="names"):
    """Same names for a graph across all serializations and templates (and for both queries of a pair)."""
    if ids == "int":
        return {u: str(u) for u in range(rec["n"])}
    rng = random.Random(rec["graph_seed"])
    if ids == "compound":                  # E16: distinct hyphenated pairs of pool names
        out, seen = {}, set()
        while len(out) < rec["n"]:
            a, b = rng.sample(name_pool, 2)
            if f"{a}-{b}" not in seen:
                seen.add(f"{a}-{b}")
                out[len(out)] = f"{a}-{b}"
        return out
    if ids == "num4":                      # E16: distinct four-digit integers
        return dict(enumerate(str(x) for x in rng.sample(range(1000, 10000), rec["n"])))
    if rec["n"] > len(name_pool):
        raise ValueError("name pool smaller than graph")
    return dict(enumerate(rng.sample(name_pool, rec["n"])))


def prompts_for(rec, name_pool=NAME_POOL, serializations=SERIALIZATIONS, templates=tuple(TEMPLATES),
                mask=None, ids="names", question_first=False, sort_edges=False):
    G = to_graph(rec)
    prop = BY_ID[rec["pid"]]
    assert bool(prop.label(G, rec["query"])) == rec["label"], rec["id"]
    names = assign_names(rec, name_pool, ids)
    colors = sorted({c for c in rec["colors"]} | {"red", "blue"}, key=["red", "blue", "green"].index)
    head_names = entry_names = None
    qnames = names
    if mask == "all":
        names = qnames = {u: MASK_WORD for u in names}
    elif mask == "heads":
        head_names = {u: MASK_WORD for u in names}
    elif mask == "entries":
        entry_names = {u: MASK_WORD for u in names}
    q = prop.question.format(v=qnames[rec["query"]]) if prop.needs_query else prop.question
    order = rec.get("order")              # explicit list order (E7 twin pairs); None: drawn from the seed
    for s in serializations:
        if mask in ("heads", "entries", "rand_heads", "rand_entries") and s not in LISTS:
            continue
        if rec.get("ser_only") and s != rec["ser_only"]:          # E11b-L: one padded graph per format
            continue
        seed = rec["graph_seed"] + SER_OFFSET[s]
        text = serialize(G, names, s, seed=seed, head_names=head_names, entry_names=entry_names,
                         sort_edges=sort_edges and s == "EL", order=order)
        if rec.get("filler") and s in ("EL", "ECL"):
            text = add_filler(text, **rec["filler"])
        rp = None
        if rec["query"] is not None and rec["label"] and rec["cls"] == "depth1" and ids == "names" and mask is None:
            rp = rel_pos(G, rec["query"], s, text, names, seed, order)
        if mask in ("rand_heads", "rand_entries"):
            # as many occurrences as the targeted mask covers: n line heads, or 2|E| list entries
            n_mask = G.number_of_nodes() if mask == "rand_heads" else 2 * G.number_of_edges()
            text = random_mask(text, names, n_mask, rec["graph_seed"] + 9_000 + SER_OFFSET[s])
        if rec["query"] is None:
            k, d = None, None
        elif s in LISTS:
            k, d = kmin_dmin(G, rec["query"], s, seed, order)
        else:
            # edge formats have no line heads: their k_min is that of the matched NC prompt (same graph, naming
            # and NC order seed), so that every format is stratified by one quantity (CONFIG §5.5)
            k, _ = kmin_dmin(G, rec["query"], "NC", rec["graph_seed"] + SER_OFFSET["NC"])
            d = None
        for t in templates:
            yield {
                "id": rec["id"], "pid": rec["pid"], "cls": rec["cls"], "pair": rec.get("pair"),
                "label": rec["label"], "ser": s, "tpl": t, "n": rec["n"], "maxdeg": rec.get("maxdeg"),
                "deg_v": rec.get("deg_v"), "k_min": k, "d_min": d, "rel_pos": rp, "mask": mask, "ids": ids,
                "question_first": question_first, "sort_edges": sort_edges,
                "masked": mask == "all", "src": rec.get("src"),
                "filler_cell": rec.get("filler_cell"), "pad": rec.get("pad"),          # E14, E11b-L
                "names": sorted(set(names.values()) | set(qnames.values())) if ids == "names" else [],
                "prompt": render(t, text, q, colors=colors, question_first=question_first),
            }
