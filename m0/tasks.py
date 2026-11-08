"""Questions of the LLM experiments (CONFIG.md §5.4; paper appendix "Pretrained Language Models"), grouped by logical class.

Classes: global (PL+GC), depth1 (GML depth 1, threshold 1), depth2 (GML depth 2), count (depth 1, threshold >= 2).
Three questions per class (E-12): two formulas plus one paraphrase, so that question wording and logical class
are not confounded. Q0 ("is v red?") is a diagnostic used only in E7.

Matched pairs (CONFIG §1.3, F-3)
  global questions: two different graphs with equal (|V|, |E|)             -> pair_kind "two_graphs"
  query questions : ONE graph with two query vertices                       -> pair_kind "one_graph"
                    (so every graph-level statistic, e.g. #red and the sum of red degrees, is shared exactly),
                    matched on the query-level key below.
  query key       : depth1/count/Q0: deg v and the sum of neighbor degrees of v (2-hop size)
                    depth2         : deg v, #red neighbors of v, sum of neighbor degrees of v (2-hop size)
"""
from dataclasses import dataclass
from typing import Callable

import networkx as nx


def red(G: nx.Graph, u) -> bool:
    return G.nodes[u]["color"] == "red"


def n_red(G):
    return sum(red(G, u) for u in G)


def n_red_nbrs(G, v):
    return sum(red(G, u) for u in G[v])


def red_deg_sum(G):
    return sum(G.degree(u) for u in G if red(G, u))


def nbr_deg_sum(G, v):
    return sum(G.degree(u) for u in G[v])


@dataclass(frozen=True)
class Prop:
    pid: str
    cls: str
    question: str
    needs_query: bool
    label: Callable
    eligible: Callable
    graph_key: Callable      # (G) -> tuple, shared by the two graphs (two_graphs) / trivially (one_graph)
    query_key: Callable      # (G, v) -> tuple, shared by the two query vertices (one_graph)


def _any(G, v):
    return True


def _blue_deg(k):
    return lambda G, v: G.degree(v) >= k and not red(G, v)


GKEY = lambda G: (G.number_of_nodes(), G.number_of_edges())
GKEY_FULL = lambda G: (G.number_of_nodes(), G.number_of_edges(), n_red(G), red_deg_sum(G))
QKEY1 = lambda G, v: (G.degree(v), nbr_deg_sum(G, v))
QKEY2 = lambda G, v: (G.degree(v), n_red_nbrs(G, v), nbr_deg_sum(G, v))


def _red_at_two(G, v):
    return [u for u in G[v] if n_red_nbrs(G, u) >= 1]


PROPS = [
    # global
    Prop("G1", "global", "Is there at least one red node in the graph?", False,
         lambda G, v: n_red(G) >= 1, _any, GKEY, None),
    Prop("G2", "global", "Are there at least three red nodes in the graph?", False,
         lambda G, v: n_red(G) >= 3, _any, GKEY, None),
    Prop("G3", "global", "Are three or more nodes colored red?", False,      # paraphrase of G2
         lambda G, v: n_red(G) >= 3, _any, GKEY, None),
    # depth 1
    Prop("D1a", "depth1", "Does node {v} have at least one red neighbor?", True,
         lambda G, v: n_red_nbrs(G, v) >= 1, _blue_deg(1), GKEY_FULL, QKEY1),
    Prop("D1b", "depth1", "Are all neighbors of node {v} red?", True,
         lambda G, v: n_red_nbrs(G, v) == G.degree(v), _blue_deg(1), GKEY_FULL, QKEY1),
    Prop("D1c", "depth1", "Is node {v} adjacent to a red node?", True,
         lambda G, v: n_red_nbrs(G, v) >= 1, _blue_deg(1), GKEY_FULL, QKEY1),
    # depth 2
    Prop("D2a", "depth2", "Does node {v} have a neighbor that itself has at least one red neighbor?", True,
         lambda G, v: len(_red_at_two(G, v)) >= 1, _blue_deg(1), GKEY_FULL, QKEY2),
    Prop("D2b", "depth2", "Does every neighbor of node {v} have at least one red neighbor?", True,
         lambda G, v: len(_red_at_two(G, v)) == G.degree(v), _blue_deg(1), GKEY_FULL, QKEY2),
    Prop("D2c", "depth2", "Is node {v} connected through one intermediate node to a red node?", True,
         lambda G, v: len(_red_at_two(G, v)) >= 1, _blue_deg(1), GKEY_FULL, QKEY2),
    # counting
    Prop("C2", "count", "Does node {v} have at least two red neighbors?", True,
         lambda G, v: n_red_nbrs(G, v) >= 2, _blue_deg(2), GKEY_FULL, QKEY1),
    Prop("C3", "count", "Does node {v} have at least three red neighbors?", True,
         lambda G, v: n_red_nbrs(G, v) >= 3, _blue_deg(3), GKEY_FULL, QKEY1),
    Prop("C2p", "count", "Do two or more neighbors of node {v} have the color red?", True,
         lambda G, v: n_red_nbrs(G, v) >= 2, _blue_deg(2), GKEY_FULL, QKEY1),
]
# diagnostic (E7 only): any query color, degree >= 1
Q0 = Prop("Q0", "diag", "Is node {v} red?", True,
          lambda G, v: red(G, v), lambda G, v: G.degree(v) >= 1, GKEY_FULL, QKEY1)

# positive controls of the name-blind experiment E2 (graph properties; built by build_blind.py)
def _x1(G, v=None):
    return any(red(G, u) and any(red(G, w) for w in G[u]) for u in G)


X1 = Prop("X1", "control", "Does some red node have a red neighbor?", False, _x1, _any, None, None)
X2 = Prop("X2", "control", "Does some node have at least three neighbors?", False,
          lambda G, v: max((d for _, d in G.degree), default=0) >= 3, _any, None, None)

CLASSES = ["global", "depth1", "depth2", "count"]
BY_ID = {p.pid: p for p in PROPS + [Q0, X1, X2]}
