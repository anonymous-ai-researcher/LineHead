"""A tiny, randomly initialized Llama-architecture causal LM with a word-level tokenizer, built locally (no
download), for offline tests only: test_run_llm_hf.py (scoring invariances) and pipeline tests
(E9 runs on it with E2E_TINY=1). Its outputs are meaningless and must never be reported.

The tokenizer splits words, '--' and single punctuation characters (the word-level tokens of the theory) and
keeps character offsets; it has a chat template, so run_llm.HF and e9_mechanism.py use it unchanged.

  python make_tiny_model.py --out /tmp/tiny-llama [--layers 28]
"""
import argparse
import json
from pathlib import Path

WORD_RE = r"[A-Za-z0-9]+|--|[^\sA-Za-z0-9]"
CHAT = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def vocabulary():
    import re
    from instances import load, prompts_for
    from names import POOL
    from prompts import TEMPLATES
    words = set(POOL) | {"kit", "Yes", "No", "yes", "no", "Let", "s", "think", "step", "by", "Therefore", "the",
                         "answer", "or", "is", "user", "assistant"}
    here = Path(__file__).resolve().parent
    for f in ("m0_instances.jsonl", "e2_controls.jsonl", "e8_colors3.jsonl"):
        for r in load(here / "data" / f)[::200]:
            for p in prompts_for(r, serializations=("EL", "AL", "NC", "ECL", "GraphML", "EL2", "ELfar", "NCH"),
                                 templates=tuple(TEMPLATES)):
                words |= set(re.findall(WORD_RE, p["prompt"]))
    for i in range(64):
        words.add(str(i))
    return sorted(words)


def build(out, layers=28, hidden=32, heads=4, seed=0):
    import torch
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    specials = ["<pad>", "<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>"]
    vocab = {w: i for i, w in enumerate(specials + [w for w in vocabulary() if w not in specials])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"<\|[a-z]+\|>"), behavior="isolated"),
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Split(Regex(WORD_RE), behavior="isolated")])
    tk.decoder = decoders.WordPiece(prefix="##")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", unk_token="<unk>", bos_token="<s>",
                                  eos_token="</s>", additional_special_tokens=["<|user|>", "<|assistant|>"])
    tok.chat_template = CHAT
    tok.save_pretrained(out)
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=heads,
                      max_position_embeddings=8192, pad_token_id=0, bos_token_id=2, eos_token_id=3)
    LlamaForCausalLM(cfg).save_pretrained(out)
    json.dump({"tiny_random_model": True, "layers": layers, "hidden": hidden, "vocab": len(vocab)},
              open(out / "TINY_MODEL.json", "w"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--hidden", type=int, default=32)
    a = ap.parse_args()
    print("wrote", build(a.out, a.layers, a.hidden))
