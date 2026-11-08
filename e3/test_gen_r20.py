"""Tests for gen_r20.py: labels match the task semantics, and each pair differs minimally."""
import gen_r20 as g


def _check(task, split, n=40):
    for Gp, Gn, vp, vn, names, s in g.pairs(task, split, 0, n):
        if split.startswith("pm"):
            assert all(d == 1 for _, d in Gp.degree)
            E = Gp.number_of_nodes() // 2
            assert 8 <= E <= 32 if split == "pmtrain" else E == int(split[2:])
            assert not g.red(Gp, vp) and g.red(Gp, next(iter(Gp[vp]))) and not g.red(Gp, next(iter(Gp[vn])))
        elif task in g.GLOBAL:
            want = 1 if task == "g1" else 3
            assert sum(g.red(Gp, u) for u in Gp) == want and sum(g.red(Gn, u) for u in Gn) == want - 1
            assert set(Gp.edges) == set(Gn.edges)
            tp, tn = g.tokens(Gp, names, "AL", task, s), g.tokens(Gn, names, "AL", task, s)
            assert len(tp) == len(tn) and sum(a != b for a, b in zip(tp, tn)) >= 1 and tp[-1] == g.TASK_TOKEN[task]
        else:
            assert max(d for _, d in Gp.degree) <= 3 and Gp.degree(vp) == Gp.degree(vn)
            assert Gp.number_of_nodes() <= (40 if split == "ood" else 16)
            assert not g.red(Gp, vp) and not g.red(Gp, vn)
            if task == "dd":
                assert g.two_hop(Gp, vp) and not g.two_hop(Gp, vn)
            else:
                assert sum(g.red(Gp, w) for w in Gp[vp]) == 2 and sum(g.red(Gp, w) for w in Gp[vn]) == 1
            tp, tn = g.tokens(Gp, names, "NC", task, s, vp), g.tokens(Gp, names, "NC", task, s, vn)
            assert tp[:-1] == tn[:-1] and tp[-1] != tn[-1]


def test_all():
    for task in ("g1", "c3", "dd", "c2"):
        for split in ("train", "ood"):
            _check(task, split)
    for sp in ("pm8", "pm32", "pmtrain"):
        _check("dia", sp)


if __name__ == "__main__":
    test_all()
    print("ok")
