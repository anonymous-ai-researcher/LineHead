"""E24 (CONFIG.md §6.14; rule C53): LoRA fine-tuning of pretrained base models per format, the sample
efficiency test of the learnability hypothesis behind P3 on pretrained models.

Models  Qwen/Qwen2.5-0.5B and Qwen/Qwen2.5-1.5B (base). LoRA r = 16, alpha = 32, dropout 0.05, on q_proj, k_proj,
        v_proj, o_proj; AdamW, learning rate 2e-4, linear decay, 3 epochs, batch 16; seeds 0-2.
Train   N in {1000, 4000, 16000} D1a prompts (half positive) of the E1 distribution (build.random_graph, 10-40 vertices),
        drawn with seed 24 and rejected if their edge set equals that of any E1 test graph; one format per run
        (EL, AL, NC, ECL; and ELB for E33b), template T1, target " Yes" / " No" after the chat prompt (loss on the target token only).
Test    the E1 D1a test pairs (data/m0_instances.jsonl, 400 pairs, T1) and 200 D1a pairs on graphs with 60-100
        vertices (mean degree U[2, 8], seed 25); scored as run_llm.py (log p Yes vs log p No at the answer token).
Requires peft==0.13.2 (RUN_PROTOCOL.md §2).
Usage
  python lora_r20.py build --out data/e24_train.jsonl            # 16,000 pairs; runs use the first N/2 pairs
  python lora_r20.py build-test --out data/e24_large.jsonl
  python lora_r20.py run --model Qwen/Qwen2.5-0.5B --ser AL --n 4000 --seed 0 --out results/e24.jsonl
"""
import argparse
import json
import random
import time
from pathlib import Path

from build import pairs_one_graph, random_graph, record
from e11_heads import score_prompts, tiny
from instances import load, prompts_for
from names import POOL
from tasks import BY_ID

SIZES = (1000, 4000, 16000)


def _pairs(seed, n_pairs, n_min, n_max, exclude=()):
    prop = BY_ID["D1a"]
    rng = random.Random(seed)
    out, k = [], 0
    while k < n_pairs:
        G = random_graph(rng, n_min=n_min, n_max=n_max)
        if frozenset(map(frozenset, G.edges)) in exclude:
            continue
        by = {}
        for v in G:
            if prop.eligible(G, v):
                by.setdefault(prop.query_key(G, v), {True: [], False: []})[bool(prop.label(G, v))].append(v)
        keys = sorted(q for q, b in by.items() if b[True] and b[False])
        if not keys:
            continue
        q = rng.choice(keys)
        gs = rng.randrange(2**31)
        for v, lab in ((rng.choice(by[q][True]), True), (rng.choice(by[q][False]), False)):
            r = record(prop, G, v, lab, k, 2 * k + (not lab), gs, "one_graph")
            out.append(r)
        k += 1
    return out


def cmd_build(a):
    test = {frozenset(map(frozenset, map(tuple, r["edges"]))) for r in load(a.instances)}
    recs = _pairs(24, 8 if tiny() else 8000, 10, 40, exclude=test)
    _dump(recs, a.out)


def cmd_build_test(a):
    _dump(_pairs(25, 4 if tiny() else 200, 60, 100), a.out)


def _dump(recs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(len(recs), "records ->", path)


def cmd_run(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    t0 = time.time()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"              # the answer logit is read at the last position of every row
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32 if tiny() else torch.bfloat16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev)
    tgt = ["q_proj", "k_proj", "v_proj", "o_proj"]
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=tgt, task_type="CAUSAL_LM"))
    n = min(a.n, 16) if tiny() else a.n
    train = load(a.train)[:n]
    ex = [(next(prompts_for(r, POOL, serializations=(a.ser,), templates=("T1",)))["prompt"], r["label"]) for r in train]
    yes, no = tok.encode(" Yes", add_special_tokens=False)[-1], tok.encode(" No", add_special_tokens=False)[-1]

    def chat_text(p):
        if tok.chat_template:
            return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        return p + "\n"
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    bs, epochs = 16, 3
    steps = epochs * ((len(ex) + bs - 1) // bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    m.train()
    rng = random.Random(a.seed)
    for _ in range(epochs):
        rng.shuffle(ex)
        for i in range(0, len(ex), bs):
            chunk = ex[i:i + bs]
            enc = tok([chat_text(p) for p, _ in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            logits = m(**enc).logits[:, -1].float()
            y = torch.tensor([yes if l else no for _, l in chunk], device=dev)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step(); sch.step(); opt.zero_grad()
    m.eval()
    tests = {"e1": [r for r in load(a.instances) if r["pid"] == "D1a"][: 8 if tiny() else 800],
             "large": load(a.large)[: 8 if tiny() else 400]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        for split, recs in tests.items():
            ps = [p for r in recs for p in prompts_for(r, POOL, serializations=(a.ser,), templates=("T1",))]
            for p, (ly, ln) in zip(ps, score_prompts(tok, m, torch, [p["prompt"] for p in ps])):
                f.write(json.dumps({"pair": p["pair"], "label": p["label"], "ser": a.ser, "pid": "D1a", "tpl": "T1",
                                    "split": split, "lp_yes": ly, "lp_no": ln, "pred": ly > ln, "model": a.model,
                                    "variant": f"lora-n{a.n}-s{a.seed}", "n_train": a.n, "seed": a.seed}) + "\n")
    print(json.dumps({"sec": round(time.time() - t0, 1), "loss": loss.item()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "build-test", "run"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--ser", choices=["EL", "AL", "NC", "ECL", "ELB"], default="AL")
    ap.add_argument("--n", type=int, choices=SIZES, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--train", default="data/e24_train.jsonl")
    ap.add_argument("--large", default="data/e24_large.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "build-test": cmd_build_test, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
