"""Export every block of results/summary.json as a flat CSV table, one file per block.

  python analysis/tables.py [--out results/tables]

Each row is one cell: the path of keys that leads to it (joined by "/") and its value; list values (per-seed
results, per-template values) are written as one row per element with the index appended to the path. The paper's
tables print these cells rounded half up at the displayed precision (analysis/fmt.py).
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from load_data import load_merged  # noqa: E402


def rows(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from rows(v, path + (str(k),))
    elif isinstance(d, list) and d and not all(isinstance(x, str) for x in d):
        for i, v in enumerate(d):
            yield from rows(v, path + (str(i),))
    else:
        yield "/".join(path), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "results" / "tables"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = load_merged()
    n_missing = 0
    for block, B in P.items():
        with open(out / f"{block}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell", "value"])
            for path, v in rows(B):
                if v is None:
                    n_missing += 1
                w.writerow([path, "" if v is None else v])
    print(f"wrote {len(P)} tables to {out}" + (f"; {n_missing} cells not imported yet" if n_missing else ""))


if __name__ == "__main__":
    main()
