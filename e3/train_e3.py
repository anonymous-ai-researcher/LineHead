"""E3 / E3d: causal transformer trained from scratch; only the positional mechanism varies (CONFIG.md §3).

Arms (--pos)
  nope           8 global heads, no positional information                                 (P, M)
  local          heads 1-4 global, heads 5-8 attend only to the preceding token (1-local)   (+Y)
  alibi_only     8 global heads with ALiBi, slopes 2^(-8h/8), h = 1..8                     (recency, no exact Y)
  local_alibi    1-local, plus ALiBi on the 4 global heads, slopes 2^(-8h/4), h = 1..4     (+Y, approximate S)
  local_hardsince 1-local, and head 4 of every layer replaced by rightmost unique hard attention:
                 the rightmost j < i maximizing q_i . k_j / sqrt(d_h) (scores in fp32, exact rightmost tie-break by
                 rightmost_argmax, no additive bias), straight-through gradient  (+Y, exact S)
  rope           8 global heads with rotary embeddings, base 10,000                         (assumption: Y, approx S)
  dlocal         (E25, E27) 4 global heads and max(4, D) local heads with fixed offsets 1..D (--window D;
                 head 4+j reads position i-(j mod D)-1), so carried names advance up to D positions per layer
                 (Proposition ehat:local; Corollary ehat:window). Pass --heads 4+max(4, D) and --d 32*heads.
  lastglobal     (E26) layers 1..L-1 have 8 1-local heads, the last layer 4 global and 4 1-local heads:
                 a single layer of matching, so every computed formula has match-nesting depth <= 1 (Theorem md:el)
Masking: strict (key j < query i); <BOS> at position 0 attends to itself. No position embeddings.
Names (F-2): every token "@name" is embedded by a frozen vector e ~ N(0, I/d) drawn independently per sequence
(and per name within it), in training and in evaluation; there are no learned name embeddings. Other tokens
have learned embeddings initialized N(0, 1/d) so that all inputs have comparable scale.
Optimization (CONFIG §3.4): AdamW (0.9, 0.98), eps 1e-8, weight decay 0.01 except LayerNorm and biases;
peak lr 3e-4, 2,000 linear warm-up steps, cosine to 3e-5; batch 128; 40,000 steps; clip 1.0; bf16 autocast.
Evaluation every 2,000 steps on 2,000 ID validation examples (a separate split, gen_e3.py --split val, disjoint
from the ID test set); final evaluation on the ID test, OOD test and
k-controlled sets if present. Balanced accuracy throughout.

Data files in --data: {task}_{ser}_train.jsonl, _id.jsonl, _ood.jsonl, _kset.jsonl, and if present _kid.jsonl
(in-distribution k_min strata) and _kown.jsonl (own-line set) (gen_e3.py). For the k-controlled set the positives'
(k_min, line length, correct) triples are appended to --kset-examples (line-length control).
Usage:
  python train_e3.py --task dia --ser NC --pos local --seed 0 --data data --out results/e3.jsonl
  python train_e3.py --task dia --ser NC --pos local --layers 12 --seed 0 --data data --out results/e3d.jsonl
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

KSTRATA = ["1", "2", "3", "4-6", "7-12", "13-32"]
ARMS = ["nope", "local", "alibi_only", "local_alibi", "local_hardsince", "rope", "dlocal", "lastglobal"]


def rope(x, base=10000.0):
    T, Dh = x.shape[-2], x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, Dh, 2, device=x.device, dtype=torch.float32) / Dh))
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, pos):
        super().__init__()
        self.h, self.pos = h, pos
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bias, hard_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.pos == "rope":
            q, k = rope(q), rope(k)
        if self.pos in ("alibi_only", "local_alibi"):
            a = alibi_attention(q, k, v, bias)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)       # (B, H, T, Dh)
        if self.pos == "local_hardsince":
            # head index 3 (head 4): rightmost unique hard attention, straight-through estimator
            h = 3
            # scores in fp32 (outside autocast): under bf16 a small positional tie-break would vanish
            with torch.autocast(device_type=x.device.type, enabled=False):
                s = (q[:, h].float() @ k[:, h].float().transpose(-1, -2)) / math.sqrt(q.shape[-1])  # (B, T, T)
                s = s.masked_fill(~hard_mask, float("-inf"))
                soft = torch.softmax(s, -1).to(v.dtype)
            hard = F.one_hot(rightmost_argmax(s), T).to(v.dtype)
            att = hard + soft - soft.detach()
            a = a.clone()
            a[:, h] = att @ v[:, h]
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.ln2(x))


def alibi_attention(q, k, v, bias):
    """Attention with the ALiBi bias in fp32, outside autocast: in bf16 the bias -m*delta loses its resolution
    for long sequences (spacing 2-4 at |bias| > 256, coarser than the slopes' unit steps once delta > 512), so the
    recency order between distant keys would vanish. Scores, bias and softmax in fp32; output in v's dtype."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1]) + bias.float()
        return (torch.softmax(s, -1) @ v.float()).to(v.dtype)


def rightmost_argmax(s):
    """Exact rightmost unique-hard selection: among the positions with the maximal score (exact equality, in the
    dtype of s), the last one. No additive tie-break, so the result does not depend on numeric precision."""
    T = s.shape[-1]
    is_max = s == s.max(-1, keepdim=True).values
    return (T - 1) - torch.flip(is_max, dims=[-1]).int().argmax(-1)


class Model(nn.Module):
    def __init__(self, vocab, d=256, layers=6, heads=8, pos="nope", window=1):
        super().__init__()
        self.pos, self.heads, self.d, self.window, self.layers = pos, heads, d, window, layers
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=d ** -0.5)
        self.blocks = nn.ModuleList(Block(d, heads, pos) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def bias(self, T, device, last=True):
        """(H, T, T) additive bias and the (T, T) strict mask used by the hard head. `last`: the bias of the last
        layer (lastglobal arm: the other layers have local heads only)."""
        i = torch.arange(T, device=device)
        dist = (i[:, None] - i[None, :]).float()
        first = (i[:, None] == 0) & (i[None, :] == 0)
        glob = (dist > 0) | first
        loc = (dist == 1) | first
        H = self.heads
        n_glob = H if self.pos in ("nope", "rope", "alibi_only") else 4 if self.pos == "dlocal" else H // 2
        if self.pos == "lastglobal" and not last:
            n_glob = 0
        if self.pos == "dlocal":       # head n_glob + j reads the fixed offset (j mod D) + 1 (first token: itself)
            offs = [(h - n_glob) % self.window + 1 for h in range(H)]
            allowed = torch.stack([glob if h < n_glob else (dist == offs[h]) | (first | ((i[:, None] < offs[h]) & (dist == 0)))
                                   for h in range(H)])
        else:
            allowed = torch.stack([glob if h < n_glob else loc for h in range(H)])
        b = torch.zeros(H, T, T, device=device)
        if self.pos in ("alibi_only", "local_alibi"):
            slopes = torch.tensor([2 ** (-8 * (h + 1) / n_glob) for h in range(n_glob)], device=device)
            b[:n_glob] = -slopes[:, None, None] * dist.clamp(min=0)[None]
        return b.masked_fill(~allowed, float("-inf")), glob

    def forward(self, x, name_vec, is_name, lengths):
        h = self.emb(x)
        h = torch.where(is_name[..., None], name_vec.to(h.dtype), h)
        bias, glob = self.bias(x.shape[1], x.device)       # fp32 (ALiBi arms use it in fp32, alibi_attention)
        early = self.bias(x.shape[1], x.device, last=False)[0] if self.pos == "lastglobal" else bias
        for li, blk in enumerate(self.blocks):
            h = blk(h, bias if li == len(self.blocks) - 1 else early, glob)
        return self.head(self.ln(h[torch.arange(x.shape[0], device=x.device), lengths - 1]))


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return [json.loads(l) for l in f]


def encode(e, vocab):
    """Cached per example: token ids (names -> 0, overwritten by name vectors) and, per position, the index of
    the position's name among the example's distinct names in order of first occurrence (-1: not a name)."""
    if "_ids" not in e:
        table, ids, nid = {}, [], []
        for tok in e["tokens"]:
            if tok.startswith("@"):
                nid.append(table.setdefault(tok, len(table)))
                ids.append(0)
            else:
                nid.append(-1)
                ids.append(vocab.get(tok, 1))
        e["_ids"], e["_nid"], e["_nn"] = torch.tensor(ids), torch.tensor(nid), len(table)
    return e["_ids"], e["_nid"], e["_nn"]


def batchify(exs, vocab, d, device, gen):
    """Token ids, per-sequence frozen random name vectors (one fresh N(0, 1/d) vector per distinct name and
    sequence, drawn in one call), name mask, lengths, labels. Vectorised: no Python loop over tokens."""
    enc = [encode(e, vocab) for e in exs]
    L = max(len(i) for i, _, _ in enc)
    x = torch.zeros(len(exs), L, dtype=torch.long)
    nid = torch.full((len(exs), L), -1, dtype=torch.long)
    off = 0
    for r, (ids, ni, nn_) in enumerate(enc):
        x[r, :len(ids)] = ids
        nid[r, :len(ni)] = torch.where(ni >= 0, ni + off, ni)
        off += nn_
    vecs = torch.randn(max(off, 1), d, generator=gen) * d ** -0.5
    is_name = nid >= 0
    name_vec = vecs[nid.clamp(min=0)] * is_name[..., None]
    ln = torch.tensor([len(i) for i, _, _ in enc])
    y = torch.tensor([e["label"] for e in exs])
    return x.to(device), name_vec.to(device), is_name.to(device), ln.to(device), y.to(device)


def bal_acc(pred, y):
    accs = [(pred[y == c] == c).float().mean().item() for c in (0, 1) if (y == c).any()]
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["red", "dia", "g1", "c3", "dd", "c2"], required=True,
                    help="g1 some vertex red, c3 at least three red, dd two-hop red, c2 at least two red neighbors (gen_r20.py)")
    ap.add_argument("--ser", choices=["EL", "AL", "NC", "ECL", "NCH", "ELB"], required=True)
    ap.add_argument("--pos", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=2_000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--window", type=int, default=1, help="largest local offset D of the dlocal arm")
    ap.add_argument("--extra-tests", default="", help="comma-separated test splits, e.g. pm8,pm16 (gen_r20.py)")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=2_000)
    ap.add_argument("--val-size", type=int, default=2_000)
    ap.add_argument("--max-train", type=int, default=None, help="smoke test only")
    ap.add_argument("--train-size", type=int, default=None,
                    help="E15: train on the first N examples of the training file only (data efficiency); "
                         "the step budget is unchanged, so smaller sets are repeated more often")
    ap.add_argument("--curriculum-steps", type=int, default=4_000,
                    help="first steps sample only training graphs with at most --curriculum-max-n vertices")
    ap.add_argument("--curriculum-max-n", type=int, default=10)
    ap.add_argument("--out", default="results/e3.jsonl")
    ap.add_argument("--save-dir", default=None, help="save the trained model (state, vocabulary, arguments) for match_heads.py")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads (data preparation, CPU runs)")
    ap.add_argument("--kset-examples", default=None, help="default: kset_examples.jsonl next to --out")
    a = ap.parse_args()
    nominal = {"steps": a.steps}
    if os.environ.get("E2E_TINY"):        # offline end-to-end test only: a few steps of a tiny model
        a.steps, a.d, a.heads, a.bs, a.warmup = 6, 32, 4, 8, 2
        a.eval_every, a.val_size, a.curriculum_steps, a.eval_bs = 3, 16, 2, 16
        a.threads = 1                    # jobs run in parallel on few cores: no thread oversubscription
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.enable_flash_sdp(False)      # CONFIG §0: float bias, identical train/eval kernels
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.bfloat16 if dev == "cuda" else torch.float32
    pre = Path(a.data) / f"{a.task}_{a.ser}"
    tr = load(f"{pre}_train.jsonl")[: a.max_train]
    n_used = None
    if a.train_size:
        n_used = a.train_size
        if os.environ.get("E2E_TINY"):    # offline end-to-end test only: the tiny training files are short
            n_used = min(a.train_size, max(a.bs, len(tr) * a.train_size // 500000))
        if n_used > len(tr):
            raise SystemExit(f"--train-size {n_used} exceeds the {len(tr)} training examples")
        tr = tr[:n_used]
    tests = {k: load(f"{pre}_{k}.jsonl") for k in ("id", "ood")}
    tests["kset"] = load(f"{pre}_kset.jsonl") if Path(f"{pre}_kset.jsonl").exists() else []   # dia only
    for k in ("kid", "kown", "d256", "hard"):
        tests[k] = load(f"{pre}_{k}.jsonl") if Path(f"{pre}_{k}.jsonl").exists() else []
    # instances of the proof's construction for the model's own depth (hard_pairs.py --proof-L)
    hp = Path(f"{pre}_hardp{a.layers}.jsonl")
    tests["hardp"] = load(str(hp)) if hp.exists() else []
    extra = [k for k in a.extra_tests.split(",") if k]
    for k in extra:
        tests[k] = load(f"{pre}_{k}.jsonl")
    words = sorted({t for e in tr for t in e["tokens"] if not t.startswith("@")} | {"<BOS>", "<NL>", "Q", ":", "RED", "NBR"})
    vocab = {w: i + 2 for i, w in enumerate(words)}               # 0 pad, 1 unk
    model = Model(len(vocab) + 2, a.d, a.layers, a.heads, a.pos, a.window).to(dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "emb" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-8)
    floor = a.lr_min / a.lr

    def lr_lambda(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        t = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = random.Random(a.seed)
    gen = torch.Generator().manual_seed(10_000 + a.seed)

    def evaluate(exs, by=None):
        """Balanced accuracy; `by(e)` gives a stratum key -> per-stratum balanced accuracy over pairs."""
        model.eval()
        egen = torch.Generator().manual_seed(777)                  # fixed name vectors for evaluation
        preds, ys, keys = [], [], []
        with torch.no_grad(), torch.autocast(dev, dtype=amp, enabled=dev == "cuda"):
            i = 0
            while i < len(exs):
                # long prompts (d256: up to ~6,000 tokens) are evaluated in smaller chunks, so that the
                # attention memory stays that of eval_bs prompts of 512 tokens (name vectors: same fixed generator)
                j = i + 1
                while j < min(len(exs), i + a.eval_bs) and \
                        (j + 1 - i) * max(len(e["tokens"]) for e in exs[i:j + 1]) ** 2 <= a.eval_bs * 512 ** 2:
                    j += 1
                chunk = exs[i:j]
                i = j
                x, nv, isn, ln, y = batchify(chunk, vocab, a.d, dev, egen)
                preds.append(model(x, nv, isn, ln).argmax(-1).cpu()); ys.append(y.cpu())
                keys += [by(e) if by else None for e in chunk]
        model.train()
        p, y = torch.cat(preds), torch.cat(ys)
        res = {"all": bal_acc(p, y), "_pred": p, "_y": y}
        if by:
            # a pair's stratum is defined by its positive; propagate it to the negative of the same pair
            pk = {}
            for e, k in zip(exs, keys):
                if e["label"] == 1:
                    pk[e["pair"]] = k
            ks = [pk.get(e["pair"]) for e in exs]
            for k in sorted({k for k in ks if k is not None}, key=str):
                m = torch.tensor([kk == k for kk in ks])
                res[str(k)] = bal_acc(p[m], y[m])
        return res

    def strip(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    # validation: a separate in-distribution split (gen_e3.py --split val, own seed), never the ID test set
    val = load(f"{pre}_val.jsonl")[: a.val_size] if Path(f"{pre}_val.jsonl").exists() else None
    if val is None:
        print(f"warning: {pre}_val.jsonl missing; no validation curve (run make_data.sh)")
    curve, t0 = [], time.time()
    model.train()
    # curriculum (CONFIG §3.4): small graphs first; the CPU check showed learning starts on small graphs
    # after about 2,000 steps, while full-size graphs alone can sit at chance for longer
    tr_small = [e for e in tr if e.get("n", 99) <= a.curriculum_max_n] or tr
    for step in range(1, a.steps + 1):
        pool = tr_small if step <= a.curriculum_steps else tr
        x, nv, isn, ln, y = batchify(rng.sample(pool, a.bs), vocab, a.d, dev, gen)
        with torch.autocast(dev, dtype=amp, enabled=dev == "cuda"):
            loss = F.cross_entropy(model(x, nv, isn, ln).float(), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            rec = {"step": step, "train_loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0]}
            if val:
                rec["val_bal_acc"] = round(evaluate(val)["all"], 4)
            curve.append(rec)
            print(json.dumps(rec), flush=True)

    def kstratum(e):
        return e["split"].replace("kset", "").replace("kid", "")

    out = {"task": a.task, "ser": a.ser, "pos": a.pos, "layers": a.layers, "seed": a.seed, "d": a.d, "window": a.window,
           "heads": a.heads, "steps": nominal["steps"], "bs": a.bs, "lr": a.lr, "final_train_loss": loss.item(),
           "train_size": a.train_size, "n_train_used": n_used if n_used is not None else len(tr),   # nominal and used
           "curve": curve, "sec": None, "n_gpus": 1 if dev == "cuda" else 0}
    if tests["id"]:
        out["acc_id"] = evaluate(tests["id"])["all"]
    if tests["ood"]:
        out["acc_ood"] = evaluate(tests["ood"])["all"]
    if tests["kset"]:
        r = evaluate(tests["kset"], by=kstratum)
        out["acc_k"] = {KSTRATA[int(k)]: v for k, v in strip(r).items() if k != "all"}
        trip = [[e["k_min"], e.get("line_len"), int(pp == yy), i] for i, (e, pp, yy) in      # instance index
                enumerate(zip(tests["kset"], r["_pred"].tolist(), r["_y"].tolist())) if e["label"] == 1]
        kx = Path(a.kset_examples or Path(a.out).with_name("kset_examples.jsonl"))
        kx.parent.mkdir(parents=True, exist_ok=True)
        with open(kx, "a") as f:
            f.write(json.dumps({"task": a.task, "ser": a.ser, "pos": a.pos, "layers": a.layers, "seed": a.seed,
                                "pos_k_len_correct": trip}) + "\n")
    for k in extra:                         # further test splits (pm<E>, dd, c2 OOD variants)
        out[f"acc_{k}"] = evaluate(tests[k])["all"]
    if tests["kid"]:
        r = evaluate(tests["kid"], by=kstratum)
        out["acc_kid"] = {KSTRATA[int(k)]: v for k, v in strip(r).items() if k != "all"}
    if tests["kown"]:
        out["acc_kown"] = evaluate(tests["kown"])["all"]
    if tests["d256"]:                       # Delta <= 256 test set (Theorem 8)
        r = evaluate(tests["d256"], by=lambda e: e["maxdeg"])
        out["acc_d256"] = r["all"]
        out["acc_d256_by"] = {k: v for k, v in strip(r).items() if k != "all"}      # by hub degree
    if tests["hard"]:                       # hard pairs of Theorem lb:main (hard_pairs.py)
        out["acc_hard"] = evaluate(tests["hard"])["all"]
    if tests["hardp"]:                      # proof's construction at depth a.layers
        out["acc_hardp"] = evaluate(tests["hardp"])["all"]
    out["sec"] = round(time.time() - t0, 1)
    if a.save_dir:                          # checkpoint for the name-matching analysis (match_heads.py)
        Path(a.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "vocab": vocab, "args": vars(a)},
                   Path(a.save_dir) / f"{a.task}_{a.ser}_{a.pos}_L{a.layers}_s{a.seed}.pt")
    if os.environ.get("E2E_TINY"):
        out["e2e_tiny"] = True
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps({k: out.get(k) for k in ("task", "ser", "pos", "layers", "seed", "acc_id", "acc_ood", "acc_k")}))


if __name__ == "__main__":
    main()
