"""k_min distribution of the E3 test sets (paper Table T_kdist; block kdist of analysis/schema.json), measured on the
generated data (make_data.sh): for the positives of dia under EL, the share of each k_min stratum (k_min in the AL
order, as gen_e3.py stores it) in the ID and OOD test sets, the share of positives in which v is the first
endpoint of at least one red edge line (id_vfirst, ood_vfirst), and on the k-controlled sets the shares with v
first (kc_vfirst) and second (kc_vsecond) endpoint of at least one red edge line, overall and per stratum
(kc_by). Own-line access: the share
of ID / OOD positives beyond the head route of six layers (k_min > 2) that the AL own-line route reaches at six
layers (j_prec <= 1, stored by gen_e3.py), id_own and ood_own; the NC own-line routes need at least 7 layers.
The output also records the SHA-256 of the generator code (gen_e3.py, ../m0/serialize.py, this script) and of the
three test files it read, so the paper's kdist block can be checked against the frozen generator
(e3/results/kdist_measured.json is this output for the make_data.sh settings).
  python kdist.py --data data --out results/kdist.json
"""
import argparse
import hashlib
import json
from pathlib import Path

STRATA = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 32, "13-32")]


def orient(tokens):
    """(v first endpoint of some red edge, v second endpoint of some red edge) from an EL token sequence."""
    v = tokens[-1]
    i = tokens.index(":") + 1
    col = {}
    while tokens[i] != "<NL>":
        if tokens[i].startswith("@"):
            col[tokens[i]] = tokens[i + 1]
        i += 1
    first = second = False
    for j in range(i, len(tokens) - 2):
        if tokens[j] == "--":
            a, b = tokens[j - 1], tokens[j + 1]
            first |= a == v and col.get(b) == "red"
            second |= b == v and col.get(a) == "red"
    return first, second


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/kdist.json")
    a = ap.parse_args()
    out = {}
    for split in ("id", "ood"):
        pos = [json.loads(l) for l in open(Path(a.data) / f"dia_EL_{split}.jsonl")]
        pos = [e for e in pos if e["label"] == 1]
        km = [e["k_min"] for e in pos]
        out[split] = {lab: sum(lo <= k <= hi for k in km) / len(km) for lo, hi, lab in STRATA}
        out[split + "_vfirst"] = sum(orient(e["tokens"])[0] for e in pos) / len(pos)
        out[split + "_own"] = sum(e["k_min"] > 2 and (e.get("j_prec") or 99) <= 1 for e in pos) / len(pos)
    kc = [e for e in map(json.loads, open(Path(a.data) / "dia_EL_kset.jsonl")) if e["label"] == 1]
    o = [orient(e["tokens"]) for e in kc]
    out["kc_vfirst"] = sum(f for f, _ in o) / len(o)
    out["kc_vsecond"] = sum(s for _, s in o) / len(o)
    # per k_min stratum of the k-controlled set (split label kset<i>): the partial-route values of rule V-B8 use them
    out["kc_by"] = {}
    for i, (_, _, lab) in enumerate(STRATA):
        oi = [x for e, x in zip(kc, o) if e["split"] == f"kset{i}"]
        out["kc_by"][lab] = {"vfirst": sum(f for f, _ in oi) / len(oi), "vsecond": sum(s for _, s in oi) / len(oi)}
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    here = Path(__file__).resolve().parent
    out["sha256"] = {"gen_e3.py": sha(here / "gen_e3.py"), "serialize.py": sha(here.parent / "m0" / "serialize.py"),
                     "kdist.py": sha(here / "kdist.py"),
                     **{f: sha(Path(a.data) / f) for f in ("dia_EL_id.jsonl", "dia_EL_ood.jsonl", "dia_EL_kset.jsonl")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
