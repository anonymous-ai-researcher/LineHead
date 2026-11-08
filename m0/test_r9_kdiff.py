"""Unit test of r9_stats.py kdiff (V-C8): synthetic records with known per-format changes across k_min.
NC declines by 8 points of TPR from k_min = 1 to >= 5 (Delta = 0.04), EL and ECL are flat, so T = 0.04 > 0.
  python test_r9_kdiff.py"""
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    rng = random.Random(0)
    tmp = Path(tempfile.mkdtemp())
    rows = []
    for m in ("A", "B"):
        for q in ("D1a", "D1c"):
            for i in range(300):
                k = (1, 2, 3, 5, 6)[i % 5]
                for ser in ("EL", "NC", "ECL", "AL"):
                    p = {"EL": 0.7, "ECL": 0.9, "AL": 0.9, "NC": 0.95 - (0.08 if k >= 5 else 0.0)}[ser]
                    for t in ("T1", "T2", "T3"):
                        pr = rng.random() < p
                        rows.append(dict(id=f"{q}-{i}-p", pair=f"{q}-{i}", pid=q, cls="depth1", label=True, ser=ser, tpl=t,
                                         pred=pr, model=m, k_min=k))
                        rows.append(dict(id=f"{q}-{i}-n", pair=f"{q}-{i}", pid=q, cls="depth1", label=False, ser=ser,
                                         tpl=t, pred=False, model=m, k_min=None))
    f = tmp / "e1_x.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp / "kd.json"
    subprocess.run([sys.executable, "r9_stats.py", "kdiff", str(f), "--out", str(out)], check=True,
                   env=dict(os.environ, E2E_TINY="1"), cwd=Path(__file__).parent, stdout=subprocess.DEVNULL)
    r = json.load(open(out))
    assert abs(r["nc_minus_el"] - 0.04) < 0.03 and r["ecl_minus_el_abs"] < 0.03, r
    assert r["stat"] > 0 and r["lo90"] <= r["stat"] <= r["hi90"] and 0 <= r["p_one_sided"] <= 1, r
    assert set(r["by_model"]) == {"A", "B"}
    # the direct NC - ECL contrast, and P4 as non-inferiority (AL = ECL = 0.9; NC 0.95, then 0.87)
    assert abs(r["nc_minus_ecl"] - 0.04) < 0.03 and r["nc_minus_ecl_lo90"] <= r["nc_minus_ecl"], r
    out2 = tmp / "p4.json"
    subprocess.run([sys.executable, "r9_stats.py", "p4ni", str(f), "--out", str(out2)], check=True,
                   env=dict(os.environ, E2E_TINY="1"), cwd=Path(__file__).parent, stdout=subprocess.DEVNULL)
    p4 = json.load(open(out2))
    assert set(p4) == {"1", "2", "3-4", ">=5"} and all(v["shortfall"] <= v["hi95"] + 1e-9 for v in p4.values()), p4
    assert 0.0 < p4["1"]["shortfall"] < 0.06 and abs(p4[">=5"]["shortfall"]) < 0.03, p4
    print("test_r9_kdiff: ok", {k: round(r[k], 4) for k in ("nc_minus_el", "ecl_minus_el_abs", "stat", "lo90", "p_one_sided")})


if __name__ == "__main__":
    main()
