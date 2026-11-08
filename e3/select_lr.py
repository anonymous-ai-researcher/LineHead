"""Select the E3 peak learning rate from the preregistered sweep (run_lr_sweep.sh; CONFIG.md §3.4).

Rule (fixed before the sweep): the lr whose run has the highest ID-validation balanced accuracy at the final
step (the last curve entry); ties go to the smaller lr. The ID and OOD test sets are not used.
  python select_lr.py results/lr_sweep.jsonl --out results/lr_choice.json     # writes the choice
  python select_lr.py --print results/lr_choice.json                          # prints the lr (for run_e3.sh)
"""
import argparse
import json
import sys

GRID = (1e-4, 3e-4, 1e-3)


def choose(runs):
    by = {}
    for r in runs:
        if r.get("pos") == "local" and r.get("ser") == "AL" and r.get("task") == "dia" and r.get("seed") == 0:
            by[float(r["lr"])] = r["curve"][-1]["val_bal_acc"]          # last run per lr wins
    missing = [x for x in GRID if x not in by]
    if missing:
        raise SystemExit(f"lr sweep incomplete: no run for {missing}")
    best = max(GRID, key=lambda x: (by[x], -x))
    return {"lr": best, "val_bal_acc": {str(k): by[k] for k in GRID}, "rule": "max final ID-validation balanced accuracy; ties -> smaller lr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out")
    ap.add_argument("--print", action="store_true", dest="print_")
    a = ap.parse_args()
    if a.print_:
        print(json.load(open(a.path))["lr"])
        return
    c = choose([json.loads(l) for l in open(a.path) if l.strip()])
    json.dump(c, open(a.out, "w"), indent=1)
    print(json.dumps(c), file=sys.stderr)


if __name__ == "__main__":
    main()
