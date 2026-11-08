"""Instance files (CONFIG.md §6.10).

e11bL  Degree sweep at fixed length (E11b-L): every E11b pair (data/e11b_instances.jsonl) is padded, separately for
       each format, with isolated blue vertices until its serialization is at least as long (word-level tokens) as
       the mean E11b prompt of that format at maximum degree 48, so prompt length no longer grows with the degree.
       Isolated vertices change neither label (they have no neighbors) nor the query key. Positives keep the E11b
       position condition (k_min >= max(3, D/2) for AL and NC under the order seed of the record); a padded graph
       whose order seed breaks it is redrawn with the next seed. One record per (pair member, format), field
       "ser_only" (instances.prompts_for serializes it in that format only), "pad" (number of isolated vertices).
e14    Distance dissociation (E14): the first 300 D1a pairs of E1 (data/m0_instances.jsonl) under EL and ECL with a
       neutral filler of F in {64, 256, 1024} word tokens either between the color line and the edge lines
       ("between": the text distance from an edge line to the endpoint colors grows by F) or before the color line
       ("before": the same prompt length, distances unchanged), plus F = 0. Field "filler" (instances.add_filler).
Usage: python build_r14.py e11bL --e11b data/e11b_instances.jsonl --out data/e11bL_instances.jsonl
       python build_r14.py e14 --e1 data/m0_instances.jsonl --out data/e14_instances.jsonl
Both commands also write the mean prompt length of the positives (results/e11bL_words.json: "<format>|<degree>";
results/e14_words.json: "<format>", no filler), which import_measured.py --r14 reads.
E16 (multi-token names, exploratory) needs no new file: run_llm.py --ids compound | num4 on the E1 D1a pairs.
"""
import argparse
import json
import os
import re
from collections import defaultdict

from instances import SER_OFFSET, assign_names, load, to_graph
from names import POOL
from serialize import kmin_dmin, serialize

F4 = ("EL", "AL", "NC", "ECL")
WORD = re.compile(r"[A-Za-z0-9]+|--|[^\sA-Za-z0-9]")
FILLERS = (64, 256, 1024)


def tiny():
    return bool(os.environ.get("E2E_TINY"))


def nwords(rec, f):
    G = to_graph(rec)
    names = assign_names(rec, POOL)
    return len(WORD.findall(serialize(G, names, f, seed=rec["graph_seed"] + SER_OFFSET[f])))


def padded(rec, pad, f, seed):
    r = dict(rec, n=rec["n"] + pad, colors=rec["colors"] + ["blue"] * pad, graph_seed=seed, pad=pad, ser_only=f)
    return r


def cmd_e11bL(a):
    recs = load(a.e11b)
    target = {}
    for f in F4:
        w48 = [nwords(r, f) for r in recs if r["degree_cell"] == 48]
        target[f] = sum(w48) / len(w48)
    by_pair = defaultdict(list)
    for r in recs:
        by_pair[r["pair"]].append(r)
    pairs = list(by_pair)
    if tiny():
        seen = defaultdict(int)
        keep = []
        for p in pairs:
            key = (by_pair[p][0]["pid"], by_pair[p][0]["degree_cell"])
            if seen[key] < 4:
                seen[key] += 1
                keep.append(p)
        pairs = keep
    out = []
    for p in pairs:
        pos, neg = sorted(by_pair[p], key=lambda r: not r["label"])
        D = pos["degree_cell"]
        for f in F4:
            pad, seed = 0, pos["graph_seed"]
            if D != 48:
                w0 = nwords(pos, f)
                inc = nwords(padded(pos, 1, f, seed), f) - w0          # words per isolated vertex
                pad = max(0, -(-int(target[f] - w0) // inc))
                while nwords(padded(pos, pad, f, seed), f) < target[f]:
                    pad += 1
            for tries in range(200):
                cand = padded(pos, pad, f, seed)
                G = to_graph(cand)
                ok = all(kmin_dmin(G, pos["query"], g, seed + SER_OFFSET[g])[0] >= max(3, D // 2) for g in ("AL", "NC"))
                if ok or D == 48:
                    break
                seed += 1
            else:
                raise RuntimeError(f"no order seed keeps the position condition for {p} under {f}")
            for r in (pos, neg):
                q = padded(r, pad, f, seed)
                q["id"] = f"{r['id']}-L{f}"
                q["pair"] = f"{r['pair']}-L{f}"
                q["src_pair"] = r["pair"]
                q["target_words"] = round(target[f], 1)
                out.append(q)
    return out


def cmd_e14(a):
    recs = [r for r in load(a.e1) if r["pid"] == "D1a"]
    pairs = []
    for r in recs:
        if r["pair"] not in pairs:
            pairs.append(r["pair"])
    pairs = pairs[: (4 if tiny() else a.pairs)]
    keep = set(pairs)
    out = []
    for r in recs:
        if r["pair"] not in keep:
            continue
        conds = [("none", 0)] + [(pos, F) for F in FILLERS for pos in ("between", "before")]
        for pos, F in conds:
            q = dict(r, filler={"pos": pos, "words": F}, id=f"{r['id']}-F{pos}{F}", pair=f"{r['pair']}-F{pos}{F}",
                     src_pair=r["pair"], filler_cell=f"{pos}|{F}")
            out.append(q)
    return out


def words_stats(cmd, out):
    """Mean prompt length in word-level tokens of the positive instances, as the importer uses it
    (analysis/import_measured.py --r14): E11b-L per format and degree, E14 per format without filler."""
    if cmd == "e11bL":
        acc = defaultdict(list)
        for r in out:
            if r["label"]:
                acc[(r["ser_only"], r["degree_cell"])].append(nwords(r, r["ser_only"]))
        return {f"{f}|{D}": round(sum(v) / len(v), 1) for (f, D), v in sorted(acc.items())}
    return {f: round(sum(v) / len(v), 1) for f in ("EL", "ECL")
            for v in [[nwords(r, f) for r in out if r["filler_cell"] == "none|0" and r["label"]]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e11bL", "e14"])
    ap.add_argument("--e11b", default="data/e11b_instances.jsonl")
    ap.add_argument("--e1", default="data/m0_instances.jsonl")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", help="prompt-length summary (default results/<cmd>_words.json)")
    a = ap.parse_args()
    out = cmd_e11bL(a) if a.cmd == "e11bL" else cmd_e14(a)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} records to {a.out}")
    wp = a.words or f"results/{a.cmd}_words.json"
    os.makedirs(os.path.dirname(wp) or ".", exist_ok=True)
    json.dump(words_stats(a.cmd, out), open(wp, "w"), indent=1)
    print(f"wrote {wp}")


if __name__ == "__main__":
    main()
