"""E17 unit test (run: python test_r16.py): the flip contrast of e11_heads.py summary (CONFIG §6.11)."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import e11_heads


def test_flip():
    """EL always right; AL right on positives only (balanced accuracy 0.5); NC right on every pair but the first
    ten: EL - AL = 0.5, EL - NC = 10/40, EL - mean = 0.375, and the one-sided bound lies below each mean."""
    recs = []
    for pair in range(40):
        for label in (True, False):
            for ser, ok in (("EL", True), ("AL", label), ("NC", pair >= 10), ("ECL", True)):
                recs.append(dict(model="m", variant="top128", ser=ser, pid="D1a", pair=pair, label=label,
                                 pred=label if ok else not label, id=f"{pair}{label}", tpl="T1", cls="depth1"))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e17_m.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        none = [dict(r, variant="none") for r in recs]
        q = Path(d) / "e11a_m.jsonl"
        q.write_text("".join(json.dumps(r) + "\n" for r in none))
        out = e11_heads.cmd_summary(SimpleNamespace(results=[str(q), str(p)]))
    fl = {f["contrast"]: f for f in out["flip"] if f["h"] == 128}
    assert abs(fl["EL-AL"]["diff"] - 0.5) < 1e-9 and abs(fl["EL-NC"]["diff"] - 0.25) < 1e-9
    assert abs(fl["EL-list"]["diff"] - 0.375) < 1e-9
    assert all(f["lo05"] <= f["diff"] for f in fl.values()) and fl["EL-AL"]["n_pairs"] == 40
    assert not [f for f in out["flip"] if f["h"] != 128]          # no contrast for the unablated condition


if __name__ == "__main__":
    test_flip()
    print("test_r16: ok")
