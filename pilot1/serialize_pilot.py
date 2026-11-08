"""Pilot-time serializer and name pool (frozen for reproducibility of the measured pilot).

The measured pilot (pilot1/results.jsonl, run by run_grid.sh) used the m0 serializer and name pool as they were
before the LLM study changed them (names now carry a leading space, format tokens were revised, and the pool was
filtered). This file restores that earlier code verbatim (recovered from m0_code.zip, 2026-11-01 07:07), so that
pilot1/train.py --serializer pilot reproduces the logged runs. E0b uses the same pilot serializer.
"""

import random

import networkx as nx

SERIALIZATIONS = ["EL", "AL", "NC"]


def _ordered(G: nx.Graph, rng: random.Random):
    nodes = list(G.nodes)
    rng.shuffle(nodes)
    return nodes


def edge_list(G, names, rng):
    nodes = _ordered(G, rng)
    edges = [tuple(rng.sample(e, 2)) for e in G.edges]   # random endpoint order
    rng.shuffle(edges)
    colors = ", ".join(f"{names[u]} {G.nodes[u]['color']}" for u in nodes)
    lines = [f"Colors: {colors}", "Edges:"]
    lines += [f"{names[a]} -- {names[b]}" for a, b in edges]
    return "\n".join(lines)


def adjacency_list(G, names, rng):
    lines = []
    for u in _ordered(G, rng):
        nbrs = list(G[u])
        rng.shuffle(nbrs)
        nb = ", ".join(names[w] for w in nbrs) if nbrs else "none"
        lines.append(f"{names[u]} ({G.nodes[u]['color']}): {nb}")
    return "\n".join(lines)


def node_centric(G, names, rng):
    lines = []
    for u in _ordered(G, rng):
        nbrs = list(G[u])
        rng.shuffle(nbrs)
        nb = ", ".join(f"{names[w]} ({G.nodes[w]['color']})" for w in nbrs) if nbrs else "none"
        lines.append(f"{names[u]} is {G.nodes[u]['color']}. Neighbors: {nb}.")
    return "\n".join(lines)


def serialize(G, names, kind, seed):
    rng = random.Random(seed)
    if kind == "EL":
        return edge_list(G, names, rng)
    if kind == "AL":
        return adjacency_list(G, names, rng)
    if kind == "NC":
        return node_centric(G, names, rng)
    raise ValueError(kind)


# name pool of the pilot (m0/build.py at the time of the pilot)
NAME_POOL = """
apple anchor arrow badge bakery banjo barrel basket beacon bench biscuit blanket
bottle bracket bridge bubble bucket button cabin cactus camel candle canyon carpet
castle cereal chalk cherry chimney circus cliff clock cloud cobalt coffee comet
cookie copper cotton crayon cricket crystal curtain cushion dagger desert diamond
dolphin donkey dragon drum eagle easel elbow engine falcon feather fence ferry
fiddle forest fossil fountain fox galaxy garden garlic giant ginger glacier globe
goblet gravel guitar hammer harbor helmet hermit hollow honey hornet island ivory
jacket jaguar jelly jungle kettle kingdom kitten ladder lantern lemon library
lizard lobster locket magnet mango maple marble meadow melon mirror monkey mosaic
muffin needle nickel noodle oasis oyster paddle palace panda parrot pebble pencil
pepper piano pickle pillow pirate planet pocket potato pumpkin puzzle quarry quilt
rabbit radar raven ribbon rocket saddle salmon sandal satin scarf shadow shovel
silver ski slipper socket spider sponge squirrel statue stove sugar summit sunset
tablet teapot temple ticket tiger timber tomato tower tractor trumpet tulip tunnel
turtle umbrella valley velvet violin volcano wagon walnut walrus whistle willow
window wizard yogurt zebra
""".split()
