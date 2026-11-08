"""Check the position control of the E3 test sets from the TOKENS alone (independently of gen_e3.py): for every
positive of a kset / kid file (AL or NC), no listed construction other than the route through the line head
applies (App. new:prop-rounds):
  AL own-line   needs a red neighbor u in v's own list whose line precedes v's line        -> must not exist
  NC nested     needs a red neighbor u in v's own list with an occurrence before v's line   -> must not exist
  NC inline     needs a red neighbor at position i <= floor((L-4)/5) of v's own list, L <= 12 -> position >= 2
and the stored k_min equals the k_min read from the tokens. For kown files: k_min >= 4 and the first entry of v's
own list is red with its line before v's line. Writes the number of violations per file.
Usage: python ownline_check.py data/dia_AL_kset.jsonl data/dia_NC_kset.jsonl ... --out results/kset_blocking.json
"""
import argparse
import json


def lines_of(tokens, ser):
    """[(head, color, [entries])] from AL / NC token sequences."""
    out, cur = [], []
    for t in tokens[1:]:
        if t == "<NL>":
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    res = []
    for ln in out:
        if not ln or not ln[0].startswith("@"):
            continue
        if ser == "AL" and len(ln) >= 5 and ln[1] == "(" and ln[4] == ":":
            res.append((ln[0], ln[2], [t for t in ln[5:] if t.startswith("@")]))
        elif ser == "NC" and len(ln) >= 6 and ln[1] == "is" and ln[4] == "Neighbors":
            ents = [ln[i] for i in range(6, len(ln)) if ln[i].startswith("@")]
            res.append((ln[0], ln[2], ents))
    return res


def access(tokens, ser):
    v = tokens[-1]
    L = lines_of(tokens, ser)
    color = {h: c for h, c, _ in L}
    pos = {h: i for i, (h, _, _) in enumerate(L)}
    lv = pos[v]
    own = L[lv][2]
    early = {h for h, _, _ in L[:lv]} | {e for _, _, es in L[:lv] for e in es}
    reds = [(i, u) for i, u in enumerate(own, start=1) if color.get(u) == "red"]
    kmin = min((es.index(v) + 1 for h, c, es in L if c == "red" and v in es), default=None)
    return dict(kmin=kmin, j_prec=min((i for i, u in reds if pos[u] < lv), default=None),
                j_occ=min((i for i, u in reds if u in early), default=None), j_inl=reds[0][0] if reds else None)


def check(path):
    ser = "AL" if "_AL_" in path else "NC"
    bad, n = 0, 0
    for line in open(path):
        r = json.loads(line)
        if not r["label"]:
            continue
        n += 1
        a = access(r["tokens"], ser)
        if "kown" in path:
            ok = a["kmin"] is not None and a["kmin"] >= 4 and a["j_prec"] == 1
        else:
            ok = a["j_prec"] is None and a["j_occ"] is None and (a["j_inl"] or 0) >= 2 and a["kmin"] == r["k_min"]
        bad += not ok
    return {"positives": n, "violations": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {p.split("/")[-1]: check(p) for p in a.files}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    if any(v["violations"] for v in res.values()):
        raise SystemExit("position control violated")


if __name__ == "__main__":
    main()
