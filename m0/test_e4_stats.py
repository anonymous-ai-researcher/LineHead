"""Unit test: Holm in e4_stats.py is applied within each model, not pooled (run: python test_e4_stats.py)."""
from statsmodels.stats.multitest import multipletests

from e4_stats import holm_within

tests = [{"model": m, "a": "ECL", "b": b, "p": p} for m, ps in (("A", (0.01, 0.02, 0.04)), ("B", (0.001, 0.5, 0.03)))
         for b, p in zip(("NC", "EL", "ELfar"), ps)]
holm_within(tests, ["model"])
for m in ("A", "B"):
    fam = [t for t in tests if t["model"] == m]
    assert [t["p_holm"] for t in fam] == list(multipletests([t["p"] for t in fam], method="holm")[1])
assert tests[0]["p_holm"] != multipletests([t["p"] for t in tests], method="holm")[1][0]
print("ok: Holm within each model")
