"""Unit test of match_heads.py: the attention weights it reconstructs reproduce each block's output exactly
(every arm), and the run and summary commands work on a random checkpoint.
  python test_match_heads.py"""
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_e3  # noqa: E402
from match_heads import attn_probs  # noqa: E402
from train_e3 import Model  # noqa: E402


def test_reconstruction():
    torch.manual_seed(0)
    for pos in ("nope", "local", "alibi_only", "local_alibi", "local_hardsince", "rope"):
        m = Model(50, d=32, layers=2, heads=8, pos=pos).eval()
        x = torch.randint(2, 50, (1, 23))
        h = m.emb(x)
        bias, glob = m.bias(23, "cpu")
        for blk in m.blocks:
            p = attn_probs(blk, m, h, bias, glob)
            B, T, D = h.shape
            q, k, v = blk.qkv(blk.ln1(h)).view(B, T, 3, blk.h, D // blk.h).permute(2, 0, 3, 1, 4)
            a = p.to(v.dtype) @ v
            y = h + blk.o(a.transpose(1, 2).reshape(B, T, D))
            y = y + blk.ff(blk.ln2(y))
            ref = blk(h, bias, glob)
            assert torch.allclose(y, ref, atol=1e-4), (pos, float((y - ref).abs().max()))
            h = ref


def test_run():
    tmp = Path(tempfile.mkdtemp())
    data = tmp / "data"
    data.mkdir()
    for split in ("id", "ood"):                  # match_heads.py draws bin 1-16 from id, the others from ood
        with open(data / f"dia_AL_{split}.jsonl", "w") as f:
            for k, (G, vp, vn, names, s, order, lab) in enumerate(gen_e3.pairs("dia", split, 2, 3)):
                gen_e3.write_pair(f, G, vp, vn, names, "AL", "dia", s, k, lab)
    exs = [json.loads(l) for sp in ("id", "ood") for l in open(data / f"dia_AL_{sp}.jsonl")]
    words = sorted({t for e in exs for t in e["tokens"] if not t.startswith("@")} | {"<BOS>", "<NL>", "Q", ":", "RED", "NBR"})
    vocab = {w: i + 2 for i, w in enumerate(words)}
    m = Model(len(vocab) + 2, d=32, layers=2, heads=4, pos="local")
    args = {"d": 32, "layers": 2, "heads": 4, "pos": "local", "task": "dia", "ser": "AL", "seed": 0}
    torch.save({"state": m.state_dict(), "vocab": vocab, "args": args}, tmp / "c.pt")
    env = dict(os.environ, E2E_TINY="1")
    subprocess.run([sys.executable, "match_heads.py", "--ckpt", str(tmp / "c.pt"), "--data", str(data), "--out",
                    str(tmp / "mh.jsonl")], check=True, cwd=HERE, env=env, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "match_heads.py", "summary", str(tmp / "mh.jsonl"), "--out", str(tmp / "s.json")],
                   check=True, cwd=HERE, stdout=subprocess.DEVNULL)
    r = json.loads(open(tmp / "mh.jsonl").readline())
    assert 0 <= r["best_acc"] <= 1 and len(r["best_head"]) == 2
    s = json.load(open(tmp / "s.json"))["error"]
    assert "1-local|AL" in s and all(0 <= v["mean"] <= 1 for k, v in s["1-local|AL"].items() if isinstance(v, dict))
    assert "1-16" in s["1-local|AL"], s["1-local|AL"]              # the in-distribution bin is present


if __name__ == "__main__":
    test_reconstruction()
    test_run()
    print("test_match_heads: ok")
