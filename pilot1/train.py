"""Train a small causal transformer on one (task, serialization, arm, seed) and report test accuracy.

arm = blind : every node name is replaced by the single token <N> (ID-blind, Theorem 1)
arm = ids   : node names are distinct tokens, drawn at random per graph from a 200-word pool

Word-level tokens; the label is read from the last position (the "?" token).
Usage: python train.py --task X1 --ser NC --arm blind --seed 0 --out results.jsonl
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.environ.get("M0_DIR", str(Path(__file__).resolve().parent.parent / "m0")))   # m0 next to pilot1
# The pilot uses its own serializer and name pool (serialize_pilot.py); --serializer current uses the m0
# serializer and pool instead. Default: pilot.
_SER = "current" if "--serializer" in sys.argv and sys.argv[sys.argv.index("--serializer") + 1] == "current" else "pilot"
if _SER == "pilot":
    from serialize_pilot import NAME_POOL, serialize   # noqa: E402
else:
    from build import NAME_POOL           # noqa: E402
    from serialize import serialize        # noqa: E402

from gen import PAIRS, QUESTION        # noqa: E402

TOK = re.compile(r"[A-Za-z]+|--|\n|[^\sA-Za-z]")   # line breaks are kept as a token (format token in the notes)


def make_examples(task, ser, arm, n_pairs, seed):
    rng = random.Random(seed)
    pool = NAME_POOL[:200]
    out = []
    for _ in range(n_pairs):
        for G, v, lab in PAIRS[task](rng):
            names = dict(zip(G.nodes, rng.sample(pool, G.number_of_nodes())))
            text = serialize(G, names, ser, seed=rng.randrange(2**31))
            q = QUESTION[task].format(v=names[v]) if v is not None else QUESTION[task]
            toks = ["<NL>" if t == "\n" else t for t in TOK.findall(text)] + ["<NL>", "Q", ":"] + TOK.findall(q)
            if arm == "blind":
                name_set = set(names.values())
                toks = ["<N>" if t in name_set else t for t in toks]
            out.append((toks, int(lab)))
    return out


class Tiny(nn.Module):
    def __init__(self, vocab, d=64, layers=3, heads=4, max_len=640, alibi=False):
        super().__init__()
        self.heads, self.alibi = heads, alibi
        # ALiBi recency bias (Press et al., 2022): head h adds -m_h * (i - j) to the score of key j,
        # with geometric slopes m_h = 2^(-8h/H). It lets a head attend to the most recent match.
        self.register_buffer("slopes", torch.tensor([2 ** (-8 * (h + 1) / heads) for h in range(heads)]))
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, x, lengths):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None]
        if self.alibi:
            idx = torch.arange(T, device=x.device)
            dist = (idx[:, None] - idx[None, :]).float()                 # i - j
            bias = -self.slopes[:, None, None] * dist[None]              # H x T x T
            bias = bias.masked_fill(dist[None] < 0, float("-inf"))       # causal
            mask = bias.repeat(B, 1, 1)                                  # (B*H) x T x T
            h = self.enc(h, mask=mask)
        else:
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
            h = self.enc(h, mask=causal, is_causal=True)
        last = h[torch.arange(B), lengths - 1]
        return self.head(self.norm(last))


def batchify(exs, vocab):
    L = max(len(t) for t, _ in exs)
    x = torch.zeros(len(exs), L, dtype=torch.long)
    for i, (t, _) in enumerate(exs):
        x[i, :len(t)] = torch.tensor([vocab[w] for w in t])
    lengths = torch.tensor([len(t) for t, _ in exs])
    y = torch.tensor([l for _, l in exs])
    return x, lengths, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ser", required=True)
    ap.add_argument("--arm", choices=["blind", "ids"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-pairs", type=int, default=6000)
    ap.add_argument("--test-pairs", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alibi", action="store_true", help="add ALiBi recency bias to attention")
    ap.add_argument("--serializer", choices=("pilot", "current"), default="pilot",
                    help="pilot: the serializer and name pool used for the measured pilot (default)")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--d", type=int, default=64, help="E0b: model width")
    ap.add_argument("--layers", type=int, default=3, help="E0b: number of layers")
    ap.add_argument("--device", default="cpu", help="cpu (default; the measured pilot) or cuda (E0b on one A100)")
    a = ap.parse_args()
    nominal = {"steps": a.steps, "train_pairs": a.train_pairs, "test_pairs": a.test_pairs}
    if os.environ.get("E2E_TINY"):               # offline end-to-end test only: a few steps on a few pairs
        a.steps, a.train_pairs, a.test_pairs = min(a.steps, 5), min(a.train_pairs, 16), min(a.test_pairs, 8)
    torch.manual_seed(a.seed)
    t0 = time.time()
    # disjoint data seeds for train and test; same test set across model seeds
    train = make_examples(a.task, a.ser, a.arm, a.train_pairs, seed=1000 + a.seed)
    test = make_examples(a.task, a.ser, a.arm, a.test_pairs, seed=99)
    words = sorted({w for t, _ in train + test for w in t} | set(NAME_POOL) | {"<N>"})
    vocab = {w: i + 1 for i, w in enumerate(words)}
    model = Tiny(len(vocab) + 1, d=a.d, layers=a.layers, heads=4, alibi=a.alibi).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    rng = random.Random(a.seed)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for step in range(a.steps):
        x, ln, y = (t.to(a.device) for t in batchify(rng.sample(train, a.bs), vocab))
        loss = lossf(model(x, ln), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    # PyTorch's inference fast path mishandles a float (ALiBi) attention mask; dropout is 0, so
    # disabling the fast path makes evaluation identical to the training-mode forward pass.
    torch.backends.mha.set_fastpath_enabled(False)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(test), 256):
            x, ln, y = (t.to(a.device) for t in batchify(test[i:i + 256], vocab))
            correct += (model(x, ln).argmax(-1) == y).sum().item()
    rec = {"task": a.task, "ser": a.ser, "arm": a.arm, "seed": a.seed, "alibi": a.alibi, "n_test": len(test),
           "d": a.d, "layers": a.layers, "train_pairs": nominal["train_pairs"], "steps": nominal["steps"],
           "acc": correct / len(test), "final_train_loss": loss.item(),
           "mean_len": sum(len(t) for t, _ in test) / len(test), "sec": round(time.time() - t0, 1),
           "device": a.device, "n_gpus": 1 if a.device.startswith("cuda") else 0}
    if os.environ.get("E2E_TINY"):
        rec["e2e_tiny"] = True
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
