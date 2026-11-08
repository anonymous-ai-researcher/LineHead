"""Run one model on every (instance, serialization, template) prompt.

Backends
  hf         local Hugging Face model. Scores the answer by comparing the
             log-probabilities of "Yes" and "No" as the first answer token after
             the chat template, so there are no unparseable outputs.
  openai     OpenAI chat endpoint (E10, GPT-4o). Scoring call: temperature 0, max_tokens 1, seed 0, no system
             message, logprobs with top_logprobs 20; the Yes and No scores are the log-sums of the probabilities of
             every variant present among the 20 candidates ("Yes", " Yes", "yes", " yes" and the same for No).
             If neither word is among them, the generated token is parsed (fallback; counted in `fallback`).
  anthropic  Anthropic Messages API, temperature 0, answer parsed from text.
  mock       random answers for testing the pipeline only. Its output is marked
             mock and must never be reported.

Examples
  python run_llm.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --out results/qwen7b.jsonl
  python run_llm.py --backend openai --model gpt-4.1 --out results/gpt41.jsonl
  python run_llm.py --backend hf --model meta-llama/Llama-3.1-8B-Instruct \
      --names-from Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct --out ...

Output: one JSON line per prompt with the prediction, the label, and for the hf
backend the Yes/No log-probabilities. Re-running with the same --out resumes.
"""
import argparse
import json
import random
import os
import re
import time
from pathlib import Path

from instances import load, prompts_for
from names import POOL as NAME_POOL, occurrences_ok, single_token_names, verify_name_tokens

YESNO = re.compile(r"\b(yes|no)\b", re.I)


def parse(text):
    m = YESNO.search(text or "")
    return None if m is None else m.group(1).lower() == "yes"


# ---------------------------------------------------------------- backends
class HF:
    def __init__(self, model, dtype="bfloat16", batch=16, cot=False, max_new=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.m = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map="auto", attn_implementation="eager").eval()
        # B3: record the exact revision (commit hash) of the weights that were loaded
        self.revision = getattr(self.m.config, "_commit_hash", None)
        self.batch = batch
        self.cot, self.max_new = cot, max_new
        self.cache = {}                       # exact token ids -> (log p Yes, log p No)
        # first-token ids for the answer words, with and without a leading space
        self.yes = self._ids(["Yes", " Yes", "yes", " yes"])
        self.no = self._ids(["No", " No", "no", " no"])

    def _ids(self, words):
        ids = set()
        for w in words:
            t = self.tok.encode(w, add_special_tokens=False)
            if t:
                ids.add(t[0])
        return sorted(ids)

    def _chat(self, prompt):
        """prompt is a string, or a list of (user, assistant) demonstration turns followed by the query string
        (E6 few-shot)."""
        msgs = []
        if isinstance(prompt, list):
            for u, a in prompt[:-1]:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            prompt = prompt[-1]
        msgs.append({"role": "user", "content": prompt})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + ("Let's think step by step." if self.cot else "")

    def _cot_expand(self, chunk):
        """E6 zero-shot CoT (Kojima et al., 2022): greedy reasoning, then the answer cue; scoring follows."""
        torch = self.torch
        enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.m.device)
        with torch.no_grad():
            gen = self.m.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        self._cot_len = (new != self.tok.pad_token_id).sum(1).tolist()      # for the truncation rate (E6)
        return [c + t + "\nTherefore, the answer (Yes or No) is" for c, t in zip(chunk, texts)], texts

    def generate_answers(self, prompts, max_new=8):
        """E8: greedy generation of at most 8 tokens, answer parsed with the Yes/No regex."""
        torch = self.torch
        out = []
        for i in range(0, len(prompts), self.batch):
            chunk = [self._chat(p) for p in prompts[i:i + self.batch]]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.m.device)
            with torch.no_grad():
                g = self.m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
            for t in self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                out.append({"gen_text": t, "gen_pred": parse(t)})
        return out

    def _encode(self, texts):
        return [tuple(self.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]

    def _pad(self, seqs):
        """Left padding with explicit position ids (cumulative sum of the attention mask), so that a prompt gets
        the positions 0..len-1 whatever the batch it is in."""
        torch = self.torch
        T = max(len(q) for q in seqs)
        ids = torch.full((len(seqs), T), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for r, q in enumerate(seqs):
            ids[r, T - len(q):] = torch.tensor(q, dtype=torch.long)
            att[r, T - len(q):] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        dev = self.m.device
        return ids.to(dev), att.to(dev), pos.to(dev)

    def score_ids(self, seqs):
        """Yes/No log-probabilities for token-id sequences. Every DISTINCT sequence is scored exactly once and the
        result reused (both members of a masked pair, repeated prompts across formats or calls), so identical
        prompts get bit-identical scores whatever the batch composition; batches are formed by length."""
        torch = self.torch
        todo = sorted({q for q in seqs if q not in self.cache}, key=len)
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            ids, att, pos = self._pad(chunk)
            with torch.no_grad():
                logits = self.m(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, -1)
            ly = torch.logsumexp(lp[:, self.yes], -1).tolist()
            ln = torch.logsumexp(lp[:, self.no], -1).tolist()
            for q, a, b in zip(chunk, ly, ln):
                self.cache[q] = (a, b)
        return [self.cache[q] for q in seqs]

    def __call__(self, prompts):
        texts = [self._chat(p) for p in prompts]
        reasoning = [None] * len(texts)
        if self.cot:
            texts, reasoning = [], []
            for i in range(0, len(prompts), self.batch):
                t, r = self._cot_expand([self._chat(p) for p in prompts[i:i + self.batch]])
                texts += t
                reasoning += [(x, n) for x, n in zip(r, self._cot_len)]
        seqs = self._encode(texts)
        out = []
        for q, (a, b), r in zip(seqs, self.score_ids(seqs), reasoning):
            n_tok = len(q)
            if "gemma" in self.tok.name_or_path.lower() and n_tok >= 4096:
                raise SystemExit(f"prompt of {n_tok} tokens reaches Gemma-2's 4,096-token sliding window")
            rec = {"pred": a > b, "lp_yes": a, "lp_no": b, "n_tokens": int(n_tok)}
            if r is not None:
                rec["reasoning"], rec["reasoning_tokens"] = r[0], int(r[1])
            out.append(rec)
        return out


RETRYABLE = ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError", "ServiceUnavailableError",
             "OverloadedError", "ConnectionError", "TimeoutError")


def with_retry(fn, tries=8, base=2.0, cap=60.0, sleep=time.sleep, rng=None):
    """Call fn(); on a transient API error (rate limit, connection, timeout, server error 5xx) wait base * 2^i
    seconds (capped, with jitter) and try again, up to `tries` attempts; other errors (bad request,
    authentication) and the last failure propagate. The run then stops and a rerun with the same --out resumes
    after the last written record (API backends write every record as soon as it is scored)."""
    import random as _r
    rng = rng or _r.Random(0)
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                          # noqa: BLE001 (the class name decides)
            status = getattr(e, "status_code", None)
            transient = type(e).__name__ in RETRYABLE or (isinstance(status, int) and status >= 500) or status == 429
            if not transient or i == tries - 1:
                raise
            wait = min(cap, base * 2 ** i) * (0.5 + rng.random() / 2)
            print(f"  {type(e).__name__}: retry {i + 1}/{tries - 1} in {wait:.1f} s", flush=True)
            sleep(wait)


class OpenAIChat:
    """E10 (B4). One scoring call per prompt: temperature 0, max_tokens 1, seed 0, no system message, logprobs with
    the top 20 candidates of the first answer token. Score = log of the summed probability of the Yes variants
    present minus that of the No variants present; a variant outside the top 20 contributes nothing. If no
    variant of either word is present, the generated token is parsed instead (`fallback`: True; unparseable
    counts as wrong). Calls are logged with their date (`called`). Transient errors are retried with exponential
    backoff (with_retry); a run that still stops resumes from its output file."""
    YES, NO = ("Yes", " Yes", "yes", " yes"), ("No", " No", "no", " no")

    def __init__(self, model, base_url=None):
        from openai import OpenAI
        self.c, self.model = OpenAI(base_url=base_url), model

    def score(self, top):
        import math
        y = sum(math.exp(t.logprob) for t in top if t.token in self.YES)
        n = sum(math.exp(t.logprob) for t in top if t.token in self.NO)
        return (math.log(y) if y else None), (math.log(n) if n else None)

    def __call__(self, prompts):
        out = []
        for p in prompts:
            r = with_retry(lambda: self.c.chat.completions.create(
                model=self.model, temperature=0, max_tokens=1, seed=0, logprobs=True, top_logprobs=20,
                messages=[{"role": "user", "content": p}]))
            ch = r.choices[0]
            text = ch.message.content
            top = ch.logprobs.content[0].top_logprobs if ch.logprobs and ch.logprobs.content else []
            ly, ln = self.score(top)
            if ly is None and ln is None:
                pred, fb = parse(text), True
            else:
                pred, fb = (ly if ly is not None else -1e9) > (ln if ln is not None else -1e9), False
            out.append({"pred": pred, "lp_yes": ly, "lp_no": ln, "fallback": fb, "raw": text,
                        "system_fingerprint": getattr(r, "system_fingerprint", None),
                        "called": time.strftime("%Y-%m-%d", time.gmtime()),
                        "n_tokens": getattr(r.usage, "prompt_tokens", None)})
        return out


class AnthropicChat:
    def __init__(self, model):
        import anthropic
        self.c, self.model = anthropic.Anthropic(), model

    def __call__(self, prompts):
        out = []
        for p in prompts:
            r = with_retry(lambda: self.c.messages.create(model=self.model, max_tokens=8, temperature=0,
                                                          messages=[{"role": "user", "content": p}]))
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            out.append({"pred": parse(text), "raw": text, "n_tokens": r.usage.input_tokens})
        return out


class Mock:
    """Pseudo-random answers that are a deterministic function of the prompt text (so identical prompts get
    identical answers, as with the hf backend, and masked pairs are exactly at 0.5). Pipeline tests only; every
    record is marked mock and must never be reported."""
    def __init__(self, seed=0, cot=False):
        self.seed, self.cot = seed, cot

    def _h(self, p, salt=""):
        import zlib
        return zlib.crc32(f"{self.seed}{salt}|{json.dumps(p)}".encode())

    def __call__(self, prompts):
        out = []
        for p in prompts:
            h = self._h(p)
            r = {"pred": h % 2 == 0, "lp_yes": -(h % 997) / 100, "lp_no": -((h // 997) % 997) / 100,
                 "n_tokens": len(str(p).split()), "fallback": h % 53 == 0, "mock": True}
            if self.cot:
                r["reasoning"], r["reasoning_tokens"] = "mock reasoning", int(self._h(p, "cot") % 520)
            out.append(r)
        return out

    def generate_answers(self, prompts, max_new=8):
        return [{"gen_text": "Yes" if self._h(p, "gen") % 10 else "No", "gen_pred": bool(self._h(p, "gen") % 10)}
                for p in prompts]


def attach_demos(todo, demo_recs, pool, sers, shots):
    """E6 demonstrations: same question, serialization and template, held-out graphs, labels alternating. The
    first label (2 Yes + 1 No or 1 Yes + 2 No for K=3) is chosen per graph by a hash of the graph id, independently
    of the test label, so that majority-label and recency biases (Zhao et al., 2021) cancel across the balanced test
    set. The demonstrations are drawn by a generator seeded from (id, ser, tpl), so a prompt gets the same
    demonstrations whatever the order of the run and however much was done before a resume."""
    import zlib
    demos = {}
    for p in (q for r in demo_recs for q in prompts_for(r, pool, serializations=sers)):
        demos.setdefault((p["pid"], p["ser"], p["tpl"], p["label"]), []).append(p["prompt"])
    for p in todo:
        first = zlib.crc32(str(p["id"]).encode()) % 2 == 0
        rng_d = random.Random(zlib.crc32(f"{p['id']}|{p['ser']}|{p['tpl']}".encode()))
        turns = []
        for k in range(shots):
            lab = first if k % 2 == 0 else not first
            turns.append((rng_d.choice(demos[(p["pid"], p["ser"], p["tpl"], lab)]), "Yes" if lab else "No"))
        p["prompt"] = turns + [p["prompt"]]
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai", "anthropic", "mock"], required=True)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--instances", default="data/m0_instances.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--batch", type=int, default=16, help="CONFIG §5.9: 16 for 7-9B models, 4 for 72B")
    ap.add_argument("--names-from", default=None,
                    help="comma-separated tokenizers; restrict names to single tokens in all of them")
    ap.add_argument("--limit", type=int, default=None, help="first N instances only (smoke test)")
    ap.add_argument("--serializations", default="EL,AL,NC", help="comma-separated; E4 uses ECL (and EL,AL,NC)")
    ap.add_argument("--templates", default=None, help="comma-separated templates (default: T1,T2,T3); T1 for the reasoning model")
    ap.add_argument("--pids", default=None, help="comma-separated question ids to keep, e.g. D1a (E7)")
    ap.add_argument("--classes", default=None, help="comma-separated classes to keep, e.g. depth1 (E4)")
    ap.add_argument("--shots", type=int, default=0,
                    help="E6: number of worked demonstrations (answers only) drawn from --demo-instances")
    ap.add_argument("--demo-instances", default=None, help="E6: held-out instance file for demonstrations")
    ap.add_argument("--cot", action="store_true", help="E6: zero-shot chain of thought before scoring")
    ap.add_argument("--max-new", type=int, default=512, help="CoT: tokens of reasoning generated before the answer cue "
                    "(4096 for the reasoning model DeepSeek-R1-Distill-Qwen-7B)")
    ap.add_argument("--mask", choices=["all", "heads", "entries", "rand_heads", "rand_entries"], default=None,
                    help="E2: all names -> 'kit'; E7: only line-head or only list-entry names -> 'kit'; E7 control: "
                         "as many random name occurrences as the heads (entries) mask")
    ap.add_argument("--ids", choices=["names", "int", "compound", "num4"], default="names",
                    help="E8: integer identifiers; E16: compound (hyphenated) names or four-digit integers")
    ap.add_argument("--question-first", action="store_true", help="E8")
    ap.add_argument("--sort-edges", action="store_true", help="E8: EL sorted by source vertex")
    ap.add_argument("--generate", action="store_true", help="E8: also generate and parse an answer (agreement)")
    ap.add_argument("--variant", default="default", help="free tag stored in every record (E8: ids_int, ...)")
    ap.add_argument("--name-report", default="results/name_filter.json")
    ap.add_argument("--verify-tokens", type=int, default=200,
                    help="check token identity of every name occurrence on the first N prompts (0 = all)")
    ap.add_argument("--rendered-log", default="results/rendered_prompts.txt")
    ap.add_argument("--revisions", default="results/model_revisions.json", help="B3: weights revision per model")
    ap.add_argument("--n-gpus", type=int, default=None,
                    help="GPUs the run occupies (written to every record for T_compute); default: visible CUDA devices "
                         "for hf (72B: 4), 0 otherwise")
    args = ap.parse_args()

    pool = NAME_POOL
    if args.names_from:
        pool = single_token_names(args.names_from.split(","), report=args.name_report)
        print(f"{len(pool)} names are single tokens (with a leading space) for all of: {args.names_from}")
        if len(pool) < 40:
            raise SystemExit("fewer than 40 single-token names; graphs need up to 40")

    recs = load(args.instances)
    if args.classes:
        keep = set(args.classes.split(","))
        recs = [r for r in recs if r["cls"] in keep]
    if args.pids:          # E7: masks on the E1 D1a instances (--pids D1a), so "none" is the E1 condition itself
        keep = set(args.pids.split(","))
        recs = [r for r in recs if r["pid"] in keep]
    recs = recs[: args.limit]                    # after the filters, so a smoke test sees the kept questions
    if os.environ.get("E2E_TINY"):               # offline end-to-end test only: the first 8 pairs of each question
        first = {}                               # (and of each degree cell of E11b, so every cell is filled)
        for r in recs:
            key = (r["pid"], r.get("degree_cell"), r.get("filler_cell"), r.get("ser_only"))   # every cell
            first.setdefault(key, [])
            if r["pair"] not in first[key] and len(first[key]) < 8:
                first[key].append(r["pair"])
        keep_pairs = {p for v in first.values() for p in v}
        recs = [r for r in recs if r["pair"] in keep_pairs]
    sers = tuple(args.serializations.split(","))
    tpls = {} if not args.templates else {"templates": tuple(args.templates.split(","))}
    todo = [p for r in recs for p in prompts_for(r, pool, serializations=sers, mask=args.mask, ids=args.ids,
                                                  question_first=args.question_first, sort_edges=args.sort_edges, **tpls)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        good = []
        with open(out) as f:
            for line in f:
                try:
                    good.append(json.loads(line))
                except json.JSONDecodeError:          # an interrupted run can leave a truncated last line
                    print("resume: skipping a truncated line")
        with open(out, "w") as f:                     # rewrite without it, so the file stays valid JSON lines
            f.writelines(json.dumps(d) + "\n" for d in good)
        done = {(d["id"], d["ser"], d["tpl"]) for d in good}
    todo = [p for p in todo if (p["id"], p["ser"], p["tpl"]) not in done]
    print(f"{len(todo)} prompts to run ({len(done)} already done)")

    if args.shots:
        attach_demos(todo, load(args.demo_instances), pool, sers, args.shots)
    model = {"hf": lambda: HF(args.model, batch=args.batch, cot=args.cot, max_new=args.max_new),
             "openai": lambda: OpenAIChat(args.model, args.base_url),
             "anthropic": lambda: AnthropicChat(args.model),
             "mock": lambda: Mock(cot=args.cot)}[args.backend]()

    # F-1, every prompt (cheap): names occur only after a space and before a checked context, so with the
    # per-tokenizer name check (names.name_ok, applied in single_token_names) every occurrence is one token id
    if args.ids == "names":
        flat = [p for p in todo if isinstance(p["prompt"], str)] + \
               [{"prompt": t[0], "names": p["names"]} for p in todo if isinstance(p["prompt"], list) for t in p["prompt"][:-1]] + \
               [{"prompt": p["prompt"][-1], "names": p["names"]} for p in todo if isinstance(p["prompt"], list)]
        flat = [p for p in flat if p.get("names")] if args.mask in ("rand_heads", "rand_entries") else flat
        bad_occ = sum(not occurrences_ok(p["prompt"], p["names"]) for p in flat)
        print(f"name occurrence check: {bad_occ}/{len(flat)} prompts with a name outside the checked contexts")
        if bad_occ:
            raise SystemExit("a name occurs without a leading space or before an unchecked character")
    if args.backend == "hf":
        # F-1 token check, full tokenization: every occurrence of a name must be the same single token
        chk = todo if args.verify_tokens == 0 else todo[: args.verify_tokens]
        bad = 0
        for p in chk:
            if isinstance(p["prompt"], str) and p["ids"] == "names" and p["mask"] != "all":
                ok, _ = verify_name_tokens(model.tok, p["prompt"], [w for w in p["names"] if w != "kit"])
                bad += not ok
        print(f"token check: {bad}/{len(chk)} prompts with an inconsistent name tokenization")
        if bad:
            raise SystemExit("name tokenization is inconsistent; see names.verify_name_tokens")
        rv = Path(args.revisions)
        revs = json.loads(rv.read_text()) if rv.exists() else {}
        revs[args.model] = model.revision
        rv.parent.mkdir(parents=True, exist_ok=True)
        rv.write_text(json.dumps(revs, indent=1, sort_keys=True))
        # F-7: log the exact rendered chat input (default system prompt / date header kept) for 3 prompts
        Path(args.rendered_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.rendered_log, "a") as g:
            for p in todo[:3]:
                g.write(f"===== {args.model} {p['id']} {p['ser']} {p['tpl']}\n{model._chat(p['prompt'])}\n")

    mode = "cot" if args.cot else ("few" if args.shots else "zero")       # E6 prompting mode (E1 = zero-shot)
    n_gpus = args.n_gpus
    if n_gpus is None:
        n_gpus = 0
        if args.backend == "hf":
            import torch
            n_gpus = torch.cuda.device_count()
    step = 1 if args.backend in ("openai", "anthropic") else 64     # API calls: every record written at once
    with open(out, "a") as f:
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            t0 = time.time()
            res = model([p["prompt"] for p in chunk])
            if args.generate and args.backend in ("hf", "mock"):
                for r, g in zip(res, model.generate_answers([p["prompt"] for p in chunk])):
                    r.update(g)
            dt = time.time() - t0
            for p, r in zip(chunk, res):
                rec = {k: p[k] for k in ("id", "pid", "cls", "pair", "label", "ser", "tpl", "n", "maxdeg", "deg_v",
                                         "k_min", "d_min", "rel_pos", "mask", "ids", "question_first",
                                         "sort_edges", "masked", "src", "filler_cell", "pad")}   # E14 and E11b-L cells
                rec.update(r, model=args.model, backend=args.backend, variant=args.variant, mode=mode,
                           sec=round(dt / len(chunk), 4), n_gpus=n_gpus)
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  {min(i + step, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
