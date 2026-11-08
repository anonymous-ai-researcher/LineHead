"""C4: d_min in model tokens. For every depth-one positive prompt (D1a, D1c; template T1) under AL and NC, the
distance in tokens of the model's tokenizer from v's entry in the line where v is earliest (k_min) to that line's
head color token. Within a format the word-level distance is a function of k_min (AL 2k+1, NC 5k-1); this script
reports how far that holds for real tokenizers: per tokenizer and format, the mean d_min by k_min, the Spearman
correlation of d_min with k_min, and the share of positives whose d_min equals the modal value for their k_min
(1.0 means d_min is determined by k_min, so the two cannot be separated within a format).

Usage: python dmin_tokens.py --tokenizers Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
           --instances data/m0_instances.jsonl --out results/dmin_tokens.json
       --tokenizers word  uses the word-level tokenization of the paper (offline test)
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

from instances import SER_OFFSET, assign_names, load, prompts_for, to_graph
from names import POOL
from serialize import list_positions

WORD = re.compile(r"[A-Za-z]+|--|\n|[^\sA-Za-z]")


def token_starts(tok, text):
    if tok == "word":
        return [m.start() for m in WORD.finditer(text)]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return [a for a, b in enc["offset_mapping"]]


def index_at(starts, pos):
    """Index of the token that covers character position pos (last token starting at or before it)."""
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo


def distances(rec, tok, pool):
    G = to_graph(rec)
    v = rec["query"]
    names = assign_names(rec, pool)
    out = {}
    for ser in ("AL", "NC"):
        p = next(prompts_for(rec, pool, serializations=(ser,), templates=("T1",)))
        text = p["prompt"]
        lp = list_positions(G, ser, rec["graph_seed"] + SER_OFFSET[ser], rec.get("order"))
        best = min((ws.index(v) + 1, u) for u, ws in lp if G.nodes[u]["color"] == "red" and v in ws)
        k, u = best
        line = " " + names[u] + (" (" if ser == "AL" else " is ")
        a = text.index("\n" + line) + 1                     # start of u's line (lines start with a space)
        col = a + len(line)                                  # head color word
        end = text.index("\n", a)
        seg = text[a:end]
        # character position of the k-th entry name
        head_end = seg.index(":") + 1
        ents = [m.start() + a + head_end for m in re.finditer(r"(?<= )" + re.escape(names[v]) + r"(?=[ ,.()]|$)",
                                                                seg[head_end:])]
        starts = token_starts(tok, text)
        out[ser] = (k, index_at(starts, ents[0]) - index_at(starts, col))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", required=True)
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--labels", default=None, help="comma-separated result keys, one per tokenizer (default: its name)")
    ap.add_argument("--pool-from", "--names-from", dest="pool_from", default=None, help="tokenizers that define the name pool (as run_llm --names-from)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [r for r in load(a.instances) if r["pid"] in ("D1a", "D1c") and r["label"]]
    if os.environ.get("E2E_TINY"):
        recs = recs[:40]
    pool = POOL
    if a.pool_from:
        from names import single_token_names
        pool = single_token_names(a.pool_from.split(","))
    from scipy.stats import spearmanr
    res = {}
    labels = a.labels.split(",") if a.labels else a.tokenizers.split(",")
    for name, label in zip(a.tokenizers.split(","), labels):
        tok = name
        if name != "word":
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name)
        rows = defaultdict(list)
        for r in recs:
            for ser, kd in distances(r, tok, pool).items():
                rows[ser].append(kd)
        res[label] = {}
        for ser, kd in rows.items():
            byk = defaultdict(list)
            for k, d in kd:
                byk[k].append(d)
            modal = sum(Counter(ds).most_common(1)[0][1] for ds in byk.values()) / len(kd)
            rho = spearmanr([k for k, _ in kd], [d for _, d in kd]).correlation if len(set(k for k, _ in kd)) > 1 else 1.0
            res[label][ser] = dict(spearman=float(rho), determined=float(modal), n=len(kd),
                                  mean_by_k={str(k): sum(ds) / len(ds) for k, ds in sorted(byk.items()) if k <= 6})
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res)[:400])


if __name__ == "__main__":
    main()
