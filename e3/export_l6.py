"""E3d at L = 6 reuses the main E3 runs (CONFIG.md §3.5): copy the 1-local, dia runs with seeds 0-2 (all four
formats, 12 runs) from results/e3.jsonl into results/e3d.jsonl, tagged "reused_from": "e3". Idempotent: a run
already present in e3d.jsonl (same layers, format, seed) is not copied again.
  python export_l6.py --e3 results/e3.jsonl --e3d results/e3d.jsonl
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--e3", default="results/e3.jsonl")
ap.add_argument("--e3d", default="results/e3d.jsonl")
a = ap.parse_args()
have = set()
try:
    have = {(r["layers"], r["ser"], r["seed"]) for r in map(json.loads, open(a.e3d)) if r}
except FileNotFoundError:
    pass
runs = {}
for r in map(json.loads, (l for l in open(a.e3) if l.strip())):
    if r["pos"] == "local" and r["task"] == "dia" and r["seed"] in (0, 1, 2) and r["layers"] == 6:
        runs[(r["layers"], r["ser"], r["seed"])] = r                 # last run of a configuration wins
new = [dict(r, reused_from="e3") for k, r in sorted(runs.items()) if k not in have]
with open(a.e3d, "a") as f:
    for r in new:
        f.write(json.dumps(r) + "\n")
print(f"copied {len(new)} L=6 runs into {a.e3d} ({len(runs)} found)")
if len(runs) != 12:
    raise SystemExit(f"expected 12 L=6 runs (4 formats x seeds 0-2), found {len(runs)}")
