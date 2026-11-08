"""Scoring invariances of run_llm.HF on a tiny local random Llama (make_tiny_model.py; no download).

1. Identical prompts get bit-identical Yes/No scores whatever the batch composition (each distinct token-id
   sequence is scored once and reused), which is what makes masked query pairs exactly 0.500.
2. Left padding is position-correct: a prompt scored alone (no padding) and inside a batch with longer prompts
   (left-padded, explicit position ids) agree to fp32 rounding; without position ids they would not.
3. The cheap name check (names.name_ok / single_token_names) keeps names whose " w" is one stable token and
   reports the mask word.
Run: python test_run_llm_hf.py
"""
import tempfile

import torch

import run_llm
from make_tiny_model import build
from names import single_token_names


def make_hf(path, batch):
    hf = run_llm.HF.__new__(run_llm.HF)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf.torch = torch
    hf.tok = AutoTokenizer.from_pretrained(path)
    hf.tok.padding_side = "left"
    hf.m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").eval()
    hf.batch, hf.cot, hf.max_new, hf.cache = batch, False, 8, {}
    hf.yes, hf.no = hf._ids(["Yes", " Yes", "yes", " yes"]), hf._ids(["No", " No", "no", " no"])
    return hf


def main():
    path = build(tempfile.mkdtemp(), layers=2, hidden=32)
    short = " elk red: jar blue\n jar blue: elk red\n Does elk have a red neighbor? Answer Yes or No."
    long_ = short + " " + " ".join(["fig blue: owl red\n"] * 30)
    # 1. identical prompts across batch compositions
    a = make_hf(path, batch=1)([short, short])
    b = make_hf(path, batch=4)([long_, short, long_ + " hen", short])
    assert a[0]["lp_yes"] == a[1]["lp_yes"] == b[1]["lp_yes"] == b[3]["lp_yes"], (a, b)
    assert a[0]["lp_no"] == b[3]["lp_no"]
    hf = make_hf(path, batch=4)
    hf([short, short, short])
    assert len(hf.cache) == 1, "a repeated prompt must be scored once"
    # 2. position ids under left padding: compare the raw forward of a fresh model (no cache)
    hf = make_hf(path, batch=4)
    q_s, q_l = hf._encode([hf._chat(short), hf._chat(long_)])
    alone = hf.score_ids([q_s])[0]
    hf.cache.clear()
    batched = hf.score_ids([q_l, q_s])[1]
    assert abs(alone[0] - batched[0]) < 1e-4 and abs(alone[1] - batched[1]) < 1e-4, (alone, batched)
    ids, att, pos = hf._pad([q_l, q_s])
    with torch.no_grad():
        right = hf.m(input_ids=ids, attention_mask=att, position_ids=pos).logits[1, -1].float()
        ref = hf.m(input_ids=torch.tensor([q_s])).logits[0, -1].float()
    assert torch.allclose(right, ref, atol=1e-4), "explicit position ids must reproduce the unpadded logits"
    # (RoPE scores depend on relative positions only, so omitting position ids shifts all positions by the pad
    # length; that is exact only in exact arithmetic and changes rounding, hence the explicit ids)
    # 3. cheap name check with the local tokenizer
    from transformers import AutoTokenizer
    keep = single_token_names([path], pool=["elk", "jar", "zzzunknownword"], load=AutoTokenizer.from_pretrained)
    assert keep == ["elk", "jar"], keep
    print("ok: identical prompts bit-identical across batches; left padding position-correct; name check")


if __name__ == "__main__":
    main()
