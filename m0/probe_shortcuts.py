"""Shortcut ceilings of the matched pairs (CONFIG F-3; paper appendix "Pretrained Language Models").

For every question class, a gradient-boosting classifier is trained on surface statistics that a model could
use without the logic the question requires, and evaluated with 5-fold cross-validation grouped by pair
(both members of a pair are always in the same fold). The reported number is the balanced accuracy of that
classifier: an estimated ceiling of what the probed feature classes give (cross-validated accuracy), not a
bound on every possible shortcut. Features (none determines a label by itself):
  graph  : n, |E|, #red, sum and max of red-vertex degrees, #red-red edges, #red-blue edges, max degree
  query  : deg v, sum and max of neighbor degrees of v (2-hop size), #triangles at v, clustering of v
Usage: python probe_shortcuts.py data/m0_instances.jsonl --out results/shortcut_ceiling.json
"""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from instances import load, to_graph


def features(rec):
    G = to_graph(rec)
    red = {u for u in G if G.nodes[u]["color"] == "red"}
    rdeg = [G.degree(u) for u in red] or [0]
    rr = sum(1 for a, b in G.edges if a in red and b in red)
    rb = sum(1 for a, b in G.edges if (a in red) != (b in red))
    f = [G.number_of_nodes(), G.number_of_edges(), len(red), sum(rdeg), max(rdeg), rr, rb,
         max((d for _, d in G.degree), default=0)]
    v = rec["query"]
    if v is None:
        f += [0, 0, 0, 0, 0.0]
    else:
        nd = [G.degree(u) for u in G[v]] or [0]
        f += [G.degree(v), sum(nd), max(nd), nx.triangles(G, v), nx.clustering(G, v)]
    return f


# features that determine or nearly determine the label of a class are excluded for that class
EXCLUDE = {"global": [2, 3, 4, 5, 6],          # all red-count features (the labels are red counts)
           "control": [5, 7]}                   # X1: #red-red edges; X2: max degree
NAMES = ["n", "E", "n_red", "red_deg_sum", "red_deg_max", "rr_edges", "rb_edges", "max_deg",
         "deg_v", "nbr_deg_sum", "nbr_deg_max", "triangles_v", "clustering_v"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", nargs="+")
    ap.add_argument("--out", default="results/shortcut_ceiling.json")
    a = ap.parse_args()
    recs = [r for p in a.instances for r in load(p)]
    import hashlib
    # the ceilings refer to exactly these files: one SHA-256 per file (RUN_PROTOCOL §3)
    out = {"instances_sha256": {Path(p).name: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in a.instances}}
    for cls in sorted({r["cls"] for r in recs}):
        rs = [r for r in recs if r["cls"] == cls]
        X = np.array([features(r) for r in rs], float)
        keep = [i for i in range(X.shape[1]) if i not in EXCLUDE.get(cls, [])]
        X = X[:, keep]
        y = np.array([r["label"] for r in rs], int)
        g = np.array([r["pair"] for r in rs])
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, g):
            clf = GradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
        out[cls] = {"bal_acc": float(balanced_accuracy_score(y, pred)), "n": int(len(y))}
        print(f"{cls:8s} shortcut ceiling {out[cls]['bal_acc']:.3f}  (n={len(y)})")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
