"""E18 unit tests (run: python test_r18.py): EL structure parsing, entry positions, the summary of the edge-head,
entries-only and degree conditions (CONFIG §6.12)."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import e11_heads


def test_el_structure():
    p = "Graph:\nColors: owl blue, hen red, elk blue\nEdges:\nowl -- hen\nhen -- elk\n\nQuestion: ..."
    colors, edges = e11_heads.el_structure(p)
    assert set(colors) == {"owl", "hen", "elk"} and colors["hen"][1][0] == "red"
    assert [(a[0], b[0]) for a, b in edges] == [("owl", "hen"), ("hen", "elk")]
    q = "Colors: owl blue, hen red\nEdges:\nowl blue -- hen red\n"
    assert [(a[0], b[0]) for a, b in e11_heads.el_structure(q)[1]] == [("owl", "hen")]       # ECL lines


def test_summary_e18():
    """Edge heads: EL wrong on every positive, the lists right; entries only: as E17's flip test; degree: 3 of 4."""
    recs = []
    for pair in range(40):
        for label in (True, False):
            for ser in ("EL", "AL", "NC", "ECL"):
                base = dict(model="m", ser=ser, pid="D1a", pair=pair, label=label, id=f"{pair}{label}{ser}", tpl="T1", cls="depth1")
                recs.append(dict(base, variant="none", pred=label))
                recs.append(dict(base, variant="etop128", pred=(not label) if (ser == "EL" and label) else label))
                for s in range(5):
                    recs.append(dict(base, variant=f"erand128s{s}", pred=label))
                ok = {"EL": True, "AL": label, "NC": pair >= 10, "ECL": True}[ser]
                recs.append(dict(base, variant="ntop128", pred=label if ok else not label))
    for i in range(4):
        for ser in ("AL", "NC"):
            recs.append(dict(model="m", ser=ser, pid="degree", pair=f"g{i}", label=True, id=f"g{i}{ser}", tpl="T1",
                             cls="graphqa", variant="top128", exact=i < 3, pred=i < 3))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e18_m.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        out = e11_heads.cmd_summary(SimpleNamespace(results=[str(p)]))
    cell = {(c["cond"], c["ser"]): c for c in out["cells"] if c["pid"] == "D1a"}
    assert abs(cell[("etop128", "EL")]["drop"] - 0.5) < 1e-9 and abs(cell[("etop128", "AL")]["drop"]) < 1e-9
    er = {(r["ser"]): r for r in out["edge_random"] if r["pid"] == "D1a"}
    assert er["EL"]["n_sets"] == 5 and abs(er["EL"]["mean"] - 1.0) < 1e-9
    fl = {f["contrast"]: f for f in out["flip"] if f["kind"] == "ntop" and f["h"] == 128}
    assert abs(fl["EL-list"]["diff"] - 0.375) < 1e-9
    dg = {c["ser"]: c for c in out["degree"]}
    assert abs(dg["AL"]["acc"] - 0.75) < 1e-9 and dg["NC"]["n"] == 4


if __name__ == "__main__":
    test_el_structure()
    test_summary_e18()
    print("test_r18: ok")
