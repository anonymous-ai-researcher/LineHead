"""Unit test of compete.fit_models: both models give the undecidable cells chance with no parameter, the
divergent RMSE comes from the leave-one-condition-out predictions, and the quasi-likelihood dispersion is >= 1.
  python test_compete.py"""
import math
import random

from compete import fit_models


def cells():
    rng = random.Random(0)
    out = []
    conds = [("EL|none", "nested", False, 2.0), ("AL|none", "single", False, 1.0), ("NC|none", "single", False, 0.7),
             ("ECL|none", "single", False, 0.9), ("NCH|none", "transport", False, 1.0), ("AL|entries", "none", False, 1.5),
             ("NC|heads", "single", False, 1.2), ("Q0:AL|heads", "none", True, 3.0), ("Q0:AL|none", "single", True, 0.5),
             ("Friendship|none", "nested", False, 2.2)]
    for m in ("A", "B", "C"):
        for cd, r, q0, d in conds:
            base = {"single": 0.85, "nested": 0.7, "transport": 0.6, "none": 0.5}[r]
            out.append(dict(model=m, cond=cd, q0=q0, ba=min(0.99, max(0.01, base + rng.gauss(0, 0.02))), n=600,
                            log_dist=d, route=r))
    return out


def main():
    cs = cells()
    r = fit_models(cs)
    for kind in ("locality", "route"):
        for c in cs:
            if c["route"] == "none":
                key = f'{c["model"]}|{c["cond"]}'
                assert r[kind]["pred"][key] == 0.5 and r[kind]["heldout_pred"][key] == 0.5, (kind, key)
        div = [c for c in cs if c["cond"] == "NCH|none" or "|heads" in c["cond"] or "|entries" in c["cond"]]
        rm = 100 * math.sqrt(sum((r[kind]["heldout_pred"][f'{c["model"]}|{c["cond"]}'] - c["ba"]) ** 2 for c in div) / len(div))
        assert abs(rm - r[kind]["divergent_rmse_pts"]) < 1e-9, (kind, rm, r[kind]["divergent_rmse_pts"])
        assert r[kind]["qaic"] < r[kind]["aic"] or r["dispersion"] == 1.0
    assert r["dispersion"] >= 1.0 and r["n_divergent"] == len(div)
    assert r["n_free"] == sum(c["route"] != "none" for c in cs)
    for k_ in ("all", "free", "divergent"):
        lo, hi = r["rmse_diff_ci"][k_]
        assert lo <= hi
    lo, hi = r["rmse_diff_ci"]["all"]
    assert lo - 1e-9 <= r["delta_rmse_pts"] + 5 and r["delta_rmse_pts"] - 5 <= hi
    print("test_compete: ok", {k: round(r[k], 3) for k in ("delta_qaic", "delta_rmse_pts", "dispersion")})


if __name__ == "__main__":
    main()
