# Configuration of all experiments

This file is the single source of the hardware, software and hyperparameters of every experiment. The commands
are in RUN_PROTOCOL.md, the rules that check the results in VALIDATION_RULES.md. Any deviation from this file is
recorded in `DEVIATIONS.md` with its reason.

---

## 0. Hardware and software

| Item | Value | Note |
|---|---|---|
| GPU | one node with 8x NVIDIA A100-SXM4 80GB (NVLink) | E3: one GPU per run; 72B inference on 4 GPUs (Accelerate `device_map="auto"`, layer-wise split, not tensor parallel) |
| CPU / memory | 2x AMD EPYC 7763 (128 cores), 1 TB | data generation and bootstrap on CPU |
| Storage | 2 TB NVMe | model weights about 0.5 TB |
| OS | Ubuntu 22.04, NVIDIA driver 550.x | |
| CUDA / cuDNN | 12.4 / 9.1 | |
| Python | 3.11.9 | |
| PyTorch | 2.4.1 | E3 training sets `torch.backends.cuda.enable_flash_sdp(False)` (Section 3) |
| transformers / accelerate | 4.46.3 / 1.1.1 | all LLM experiments (scoring, the CoT of E6, the generation agreement of E8) use HF Transformers: scoring by log-probability, generation by greedy `generate` (`m0/run_llm.py::_cot_expand`) |
| networkx / numpy / scipy / statsmodels / scikit-learn | 3.4.2 / 1.26.4 / 1.14.1 / 0.14.4 / 1.5.2 | |
| tiktoken / openai | 0.8.0 / 1.54.4 | E10: single-token check under o200k_base and top-20 logprob scoring |
| Randomness | `PYTHONHASHSEED=0` in every run; library seeds derived from the run seed | data seeds are given per experiment |

Model revisions are fixed to the Hugging Face commit hash on the download date; `run_llm.py` writes the commit hash
of the loaded weights to `results/model_revisions.json` in every run. Batch size 16 (72B: 4); the code default is 16.

---

## 1. Common settings

### 1.1 Graphs and colors
- Two colors {red, blue} by default; the three-color ablation of E8 adds green.
- Names are common English nouns. The pool excludes every color-related word (cobalt, salmon, ivory, silver, copper,
  cherry, tomato, lemon, ginger, mango, pumpkin, olive, plum, peach, amber, coral, indigo, violet, rose, lime, ...;
  the blacklist is in `m0/names.py`).

### 1.2 Tokens (F-1)
- Every occurrence of every name is preceded by a space, also at the start of a line: every line starts with a space,
  so every line break in a serialization is `"\n "`.
- Name filter: `" "+w` is exactly one token under all four tokenizers.
- Check (`names.py`): per tokenizer, once for the name set: `" "+w` is a single token and stays the same token id
  when followed by any character that occurs in the prompts (space, line break, comma, colon, period, question mark,
  semicolon, closing parenthesis, quote, hyphen or end of text); names that fail are dropped and recorded per
  tokenizer in `results/name_filter.json`. The mask word `" kit"` is checked the same way (the token count of `"kit"`
  without a space is also recorded). Then a cheap string check on **every** prompt: each occurrence of each name is
  preceded by a space and followed by one of these characters; together the two checks guarantee that all
  occurrences of a name map to the same token id. The first 200 prompts are also fully tokenized and compared.
- The number of names that pass is recorded in `results/name_filter.json`.

### 1.3 Matched pairs (F-3)
The positive and the negative instance of every question are matched exactly on:

| Question class | Key |
|---|---|
| Global | (\|V\|, \|E\|) |
| Depth one / counting | (\|V\|, \|E\|, deg v, #red, sum of deg u over red u); for query-vertex questions both instances are two query vertices of the same graph, additionally matched on the sum of deg u over u in N(v) |
| Depth two | the depth-one key plus the sum of deg u over u in N(v) and the number of red neighbors of v |

The rich-feature probe reported in the appendix (gradient-boosted trees, 5-fold cross-validation grouped by pair)
must stay at or below 0.53.

---

## 2. Experiments

The scripts are listed in RUN_PROTOCOL.md; the block of `analysis/schema.json` that holds each experiment's results
is given here.

| Id | Content | Main prediction | Block |
|---|---|---|---|
| E0 | Pilot | the bound of Theorem 1 is reached | `alibi_pilot_X1_NC` |
| E0b | Pilot scaling grid for X1 under NC | the X1-NC cell is limited by capacity or not | `pilot_scaling`, `pilot_chance3` |
| E3 | Positional mechanisms | P1, P2, P3, P4 (controlled part) | `synthetic` |
| E3d | Depth sweep | the failure frontier k*(L) | `depth_sweep` |
| E1 | Main LLM study (4 models, 12 questions) | P3, interaction, planned contrasts | `llm_main` |
| E2 | Sigma-matched name-blind controls and positive controls | Theorem 1, Corollary 2 | `llm_blind` |
| E4 | ECL and format ablations (4 models) | P4, redundancy and locality | `llm_formats` |
| E7 | Partial name masking (route test) | the name-transport and since routes | `llm_partial` |
| E5 | Real topologies and GraphML | effects only through k and d | `real_graphs`, `graphml` |
| E6 | Prompting | the order is kept | `prompting` |
| E8 | Robustness ablations | conclusions do not depend on implementation details | `robustness` |
| E9 | Mechanism evidence (Qwen2.5-7B) | line-head color and line-head name are decodable | `mechanism` |
| E10 | API model (GPT-4o) | the order holds for a closed model | `api_model` |
| E3x | Learning-rate check of NoPE and RoPE, NC-H arm; position-controlled k sets, ID strata, own-line set (P2b) | only the line-head route remains among our constructions; own-line route direction; transport | `e3_extra`, `kset_blocking` |
| A | Additional analyses: paired metrics, position covariates, question random effects, prompt length, TOST, d_min under the real tokenizers, competing models, random masks, twin pairs, non-word NC and the encodings of Fatemi et al. | rules B14-B16, C16-C21, D10-D12 | `llm_stats`, `competing`, `mask_controls`, `nl_formats` |

E11 to E36 are described in Section 6.

---

## 3. E3: positional mechanisms (controlled models)

### 3.1 Model
- Causal transformer, pre-LN, 6 layers, d_model = 256, 8 heads (head dimension 32), FFN 1,024, GELU, dropout 0,
  about 4.75M parameters (the Model of `e3/train_e3.py`: 4,739,586 + 256 x vocabulary size; 4,744,706 at vocabulary
  20, 4,749,826 at 40).
- Every sequence starts with `<bos>` (E-4, all formats). Strict causal mask: position i sees j < i; `<bos>` sees itself.
- **Name embeddings (F-2)**: for every sequence, each name that occurs gets an independent frozen vector
  e ~ N(0, I/d), in training and test; names have no learned vocabulary rows. Format tokens and color words have
  learned embeddings.
- Question suffix (so that the query name is the last token, template offset k_t = 0): `<NL> Q : RED v` or
  `<NL> Q : NBR v` (RED: "is v red", NBR: "does v have a red neighbor"), read out at the last token (the query name).
- Readout: linear layer on the last token's representation, binary cross-entropy.
- ECL uses the syntax `a c(a) -- b c(b)` (as the color line of EL, `name color`), maximum offset 4 (appendix lemma);
  the LLM experiments use the same syntax.

### 3.2 Mechanisms
Names in the paper (one spelling per mechanism): NoPE, 1-local, ALiBi only, 1-local+ALiBi, 1-local+hard since, RoPE;
the first column below is the id in the code and in the summary. Format ablations: EL-2 (both directions), EL-far
(distant color block), NC-H (no line-head color).

| Mechanism | Definition | Logic |
|---|---|---|
| NoPE | 8 global heads, no position information | P, M |
| 1-local | heads 1-4 global, heads 5-8 see only the previous token | + Y |
| ALiBi-only | 8 global heads with ALiBi bias, slopes 2^{-8h/8} | recency (no exact Y) |
| 1-local+ALiBi | 1-local, the 4 global heads with ALiBi, slopes 2^{-8h/4}, h = 1..4 | + Y + approximate S |
| 1-local+hard-since | 1-local, with head 4 of every layer replaced by rightmost unique hard attention (over j < i, the rightmost maximizer of q_i k_j / sqrt(d_h); scores in fp32 with autocast off, exact rightmost tie rule `rightmost_argmax`, no position bias; d_h = 32, straight-through forward; unit test `e3/test_train_e3.py` under bf16 autocast) | + Y + exact S |
| RoPE | 8 global heads, RoPE base 10,000 | assumption: Y and approximate S |

### 3.3 Tasks, formats and data
- Tasks: `red` (v is red), `dia` (v has a red neighbor).
- Formats: EL, AL, NC, ECL; the NC-H arm adds NCH (NC without line-head colors).
- Size: per (task, format) 500k training examples, 10k ID test, 10k OOD test.
- Training and ID distribution: n ~ U{8..16}; maximum degree Delta <= 6 (rejection sampling); p_red ~ U[0.2, 0.5];
  query vertex deg v >= 1. Each pair is two query vertices of equal degree in one graph, one positive and one
  negative: for dia both query vertices are blue (one with a red neighbor, one without); for red one red, one blue.
- OOD distribution: n ~ U{17..64}, Delta ~ U{8..32} (a star plus random edges), otherwise as above.
- **k-controlled test sets (E-5)**: for positives, k_min = the smallest position of v in the lists of its red
  neighbors, in 6 strata 1, 2, 3, 4-6, 7-12, 13-32, 2,000 pairs per stratum; d_min = the token distance of that entry
  to its line-head color token is recorded. EL and ECL have no line heads; their k_min is recorded for comparison.
- **Position control**: the k-controlled sets (`kset`) and the ID strata (`kid`, strata 1, 2, 3, 4-6, 2,000
  pairs each, seed 5) keep the pair seed's list order, but the lines of the two query vertices move forward: the
  positive v comes before every line of a red neighbor that names v; all lines that do not name a red neighbor of v+
  are placed before the query lines in random order, with 0 or 1 such line between the two query lines (Prop.
  lb:top). `e3/shortcut_check.py` checks with the exact TL evaluator that the line-index shortcut Psi_{<=M} (round
  depth <= 6) stays at or below 0.55 on kset and kid (`results/kset_shortcut.json`, `kid_shortcut.json`); the generator places
  enough lines before the first query line that these formulas give both prompts of a pair the same answer.
  The red neighbor in the first position of v's own list is swapped with the first blue neighbor (no blue neighbor:
  rejected), which blocks the own-line routes of AL and NC for L <= 12; `e3/ownline_check.py` verifies this on the
  tokens alone (`results/kset_blocking.json`). Own-line set (`kown`, AL, 2,000 pairs, seed 6): positives with
  k_min >= 4 whose own list has a red neighbor in first position and that neighbor's line before v's line. Positive
  records also store `line_len` (the degree of the red neighbor that attains k_min) and the own-line positions
  `j_prec`, `j_occ`, `j_inl`.

### 3.4 Optimization
- AdamW, beta = (0.9, 0.98), eps = 1e-8, weight decay 0.01 (not on embeddings, LayerNorm and biases; in the code
  `no_decay` = one-dimensional parameters and parameters whose name contains emb).
- Peak learning rate 3e-4, 2,000 linear warm-up steps, cosine decay to 3e-5.
- Batch size 128, 40,000 steps (about 10 epochs), gradient clipping 1.0, bf16 autocast (fp32 master weights).
- Curriculum: the first 4,000 steps sample only training graphs with at most 10 vertices, afterwards all training
  graphs (learning starts on small graphs, at about 2,000 steps). The same for every mechanism.
- Evaluation every 2,000 steps on 2,000 ID validation examples (a separate split, `gen_e3.py --split val --seed 4`,
  disjoint from the ID test set); the last step is reported, no early stopping, the validation set selects nothing.
- Seeds 0-4.
- No per-mechanism tuning. The learning rate comes from a preregistered three-point sweep (E-18): before the main
  grid, one run per peak learning rate {1e-4, 3e-4, 1e-3} on 1-local, AL, dia, seed 0 (`e3/run_lr_sweep.sh`),
  selected by the last-step balanced accuracy on the ID validation set (ties: the smaller value; the test sets play
  no part); `e3/select_lr.py` writes `results/lr_choice.json`, which `run_e3.sh` and `run_e3d.sh` read. If a value
  other than 3e-4 is selected, the main grid uses it and the change is recorded in DEVIATIONS.md.

### 3.5 Resources
- Learning-rate sweep: 3 runs of about 0.7 h, 2.1 GPU hours.
- 6 mechanisms x 2 tasks x 4 formats x 5 seeds = 240 runs of about 0.7 h on one A100, 168.0 GPU hours. Data
  preparation takes about 12 ms per step (128 sequences, 2 CPU threads), about 0.14 h for 40,000 steps, included in the
  0.7 h. `train_e3.py --threads` sets the torch CPU threads (default 8).
- E3d: 1-local, L in {2, 3, 4, 6, 8, 12}, dia, 4 formats, 3 seeds. L = 6 is the E3 configuration and reuses its seeds
  0-2; 5 x 4 x 3 = 60 new runs of about 0.1 + 0.1 L hours (0.7 h at L = 6, as E3), about 40.8 GPU hours.

---

## 4. E0b: pilot scaling grid
- Task X1, format NC, name-blind.
- Grid: width {64, 128, 256} x layers {3, 6} x data {4k, 16k pairs} x steps {8k, 32k}, 3 seeds per cell, on CPU or one
  A100.
- The chance cells (X2-EL, X1-EL/AL, D1a in three formats) are rerun with seeds 0-2 under new step budgets: X2 4,000
  steps, X1 and D1a 8,000 steps (`pilot1/run_e0b.sh`); in the pilot they were one seed at 1,000 steps. Output
  `results_chance3.jsonl`, block `pilot_chance3`, rule V-B13 (every seed within 0.5 +- 0.029), appendix table
  T_pilotchance. `train.py --device cuda` runs on one A100 (`E0B_DEVICE=cuda bash run_e0b.sh`); the default is CPU.
- Otherwise as the pilot: learned absolute positions, AdamW 1e-3, batch size 32.

---

## 5. LLM experiments (E1, E2, E4, E5, E7, E8)

### 5.1 Models
- Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen2.5-72B-Instruct.
- bf16, HF forward pass, `attn_implementation="eager"` (Gemma-2 needs soft-capping), batch size 16 for 7-9B, 4 for 72B.

### 5.2 Chat templates
- One user turn.
- Without a system message, the Qwen2.5 template inserts a default system prompt and Llama-3.1 inserts a date header;
  both defaults are kept and reported in the appendix (F-7).

### 5.3 Scoring
- The next-token distribution at the answer position; the summed probability of the first tokens {"Yes", " Yes",
  "yes", " yes"} is compared with that of {"No", " No", "no", " no"}.
- No contextual calibration: the fixed rule above decides. A constant shift of the Yes/No scores moves this threshold
  and can raise or lower balanced accuracy, but it does not change which prompt of a matched pair scores higher, so
  the threshold-free paired metric (share of pairs and templates in which the positive prompt has the larger
  log p(Yes) - log p(No)) is reported alongside (`m0/r9_stats.py paired`).

### 5.4 Questions (E-12, three per class)

| Class | Id | Formula | Wording |
|---|---|---|---|
| Global | G1 | exists red | at least one red node |
| | G2 | exists^{>=3} red | at least three red nodes |
| | G3 | exists^{>=3} red | three or more nodes colored red (paraphrase of G2) |
| Depth one | D1a | diamond red | x has at least one red neighbor |
| | D1b | box red | all neighbors of x red |
| | D1c | diamond red (paraphrase) | x is adjacent to a red node |
| Depth two | D2a | diamond diamond red | |
| | D2b | box diamond red | |
| | D2c | diamond diamond red (paraphrase) | x connected through one intermediate node to a red node |
| Counting | C2 | diamond_{>=2} red | |
| | C3 | diamond_{>=3} red | |
| | C2' | diamond_{>=2} red (paraphrase) | two or more red neighbors |

- Diagnostic question Q0: "is v red", used in E7 only, query vertex of either color.
- 400 pairs per question, 3 templates, 3 formats. A global pair is two graphs (800 graphs); a query-vertex pair is two
  query vertices of one graph (400 graphs).

### 5.5 Graphs
- Random graphs G(n, p), n ~ U{10..40}, expected degree ~ U[2, 8], p_red ~ U[0.1, 0.4].
- Depth-one questions are also stratified by k_min: 1, 2, 3-4, >=5 (positives with k_min >= 8 are about 1% of E1).
- k_min (all formats): AL, NC and NC-H use the list order of the prompt itself; formats without line heads (EL, ECL,
  EL-2, EL-far, GraphML) use the k_min of the NC prompt of the same graph and naming (the NC order seed). The EL-type
  formats therefore have the stratum shares of NC; the shares of AL differ only by sampling of the order seed
  (`m0/instances.py`).

### 5.6 E4 format ablations
| Format | Definition |
|---|---|
| ECL | edge lines carry the colors of both endpoints; all 12 questions (four classes); the other ablations depth one only |
| EL-2 | every edge written once in each direction (redundancy) |
| EL-far | after the edge list, a color block whose line j gives the colors of the endpoints of edge j (the information of ECL, colors far from the names) |
| NC-H | NC without line-head colors: "u . Neighbors : w (red) ..." |
| NC-nw | NC with "is" and "Neighbors" replaced by the non-words "dax" and "Wug" at the same positions (familiarity control) |
| Incident | the incident encoding of Fatemi et al. with colors: "Node u (red) is connected to nodes w1, w2." (AL type, AL's order) |
| Friendship | the friendship encoding of Fatemi et al. with colors: "G describes a friendship graph among a (red), ...", then one line "a and b are friends." per edge (EL type, EL's order) |

### 5.7 E7 partial name masking
Under AL and NC, only the line-head names or only the entry names are masked. The mask word `kit` satisfies the token
conditions of 1.2. Questions D1a and Q0. Controls: random masks of as many name occurrences as the line-head mask (n)
or the entry mask (2|E|), drawn per graph and shared within a pair; twin-graph Q0 pairs (`build_twin.py`, 400 pairs,
seed 11), for which the two prompts of a pair are token-identical under AL with masked line heads.

### 5.8 E8 robustness
On Qwen2.5-7B and Llama-3.1-8B, depth-one and global questions:
- integer ids as names (0..39);
- EL sorted by source vertex;
- question before the graph;
- three colors;
- every template reported separately;
- agreement of generated answers (greedy, at most 8 tokens, parsed by regular expression) with log-probability scoring.

### 5.9 Resources
- E1: about 4.4 (Qwen2.5-7B), 4.6 (Llama-3.1-8B), 5.2 (Gemma-2-9B; eager attention for logit soft-capping and longer
  token sequences) and 30 (72B, 4 GPUs) GPU hours.
- Other LLM experiments, from the E1 throughput (86,400 scored prompts per model in the hours above) and their prompt
  counts (3-shot prompts weighted 2.5, generated answers 5): E2 about 51.6, E4 about 36.8 (ECL on all 12 questions;
  EL-2, EL-far, NC-H depth one only), E7 about 27.0 (including NC-H on Q0), E8 about 24.0, E5 about 2.6, E6 about 64.5
  GPU hours.
- E0b on CPU: 90 runs (72 grid runs and 18 chance-cell reruns), about 493.7 single-process CPU hours (from a 936 s
  reference run, scaled by steps x layers x width^2).

---

## 6. E5, E6, E9, E10 and the preregistered sets

### E5: real topologies and GraphML
- ogbg-molhiv, 10-40 heavy atoms (largest connected component for salts and mixtures, `build_real.py`), carbon blue,
  other atoms red.
- ogbn-arxiv, BFS subgraphs of 10-40 vertices, **colors from the true labels, binarized**: red if the category is one
  of cs.LG, cs.AI, cs.CV, cs.CL, otherwise blue.
- Models Qwen2.5-7B and Llama-3.1-8B.
- 300 D1a pairs per source (`m0/build_real.py --pairs 300`), 3 templates, formats EL, AL, NC, ECL and GraphML.
- n, Delta, mean degree and the k_min distribution of the subgraphs are reported.

### E6: prompting
- 3-shot: alternating labels, the first label decided by a hash.
- Zero-shot CoT: HF `generate`, greedy, at most 512 tokens; the truncation rate is reported.
- Models Qwen2.5-7B and Qwen2.5-72B.

### E9: mechanism evidence (Qwen2.5-7B, 28 layers)
- **Linear probes**: logistic regression on the residual stream at list-entry name tokens, decoding the color of the
  entry's line head, per layer and per distance d (tokens to the line-head color: 1-8, 9-24, >24; the E1 degree is at
  most 17, so AL has d <= 35); classes reweighted by frequency, balanced accuracy (chance 0.5; about one line in five
  has a red head); 8 random entries per graph plus every entry with d > 24 (147 under AL, 8,277 under NC); the 400
  graphs of the E1 D1a pairs (one prompt per pair, the E1 name pool, `--names-from`); 5-fold cross-validation grouped
  by graph. Line-head name retrieval is reported in the same distance groups, with the layer used (the best layer of
  the near group). Residual index 28 is the output of block 28 (captured by a hook; HF's hidden_states[28] is after
  the final norm); the Yes/No token sets of patching are those of the main scoring (including lower case).
- **Line-head name retrieval**: the residual at the entry position as query, cosine similarity to every line-head name
  position of the graph, top-1 accuracy.
- **Patching**: in the lines of red neighbors the line-head color is replaced by blue, and the clean run's residual at
  that color token is written back at residual index i = 0..28 (0: embedding output, i >= 1: output of block i, as
  hidden_states[i]), 29 values; reported is the recovered share of the logit difference; i = 0 recovers 1, i = 28
  recovers 0 (`m0/e9_mechanism.py::patch_pass`). The share is a ratio of sums: sum(patched - corrupted) /
  sum(clean - corrupted) over graphs.
- **Inner match under EL**: at the second endpoint of an edge line, the probe decodes that endpoint's own color
  (reweighted, balanced accuracy).
- **Controls**: a control task for the probe (labels a fixed random function of the entry name; selectivity = probe
  accuracy - control accuracy); a random baseline for retrieval (mean of 1 / number of lines). About 6.5 GPU hours.

### E10: API model
- gpt-4o-2024-08-06, one call per prompt: temperature 0, max_tokens 1, seed 0, no system message, top-20 logprobs; the
  Yes/No scores are the sums over the case and space variants among the top 20; if none occurs, the generated token is
  parsed (fallback). Names are checked to be single tokens under tiktoken o200k_base.
- Global and depth-one questions; formats EL, AL, NC, ECL; 200 pairs per question.
- 28,800 calls (200 pairs x 2 instances x 3 templates x 4 formats x 6 questions), retries not counted.

### 6.2 E11a, E11b, E11c
- **E11a since-head ablation (Qwen2.5-7B)**: on the E1 D1a prompts (AL, NC, T1, the last 100 pairs), every attention
  head is scored by the attention mass of list-entry name tokens on their own line head (name and color) minus the
  mean mass on other line heads (`m0/e11_heads.py score`, eager attention, the E9 tooling). Mean ablation of the top h
  heads (h in {4, 8, 16}): the head's slice of the o_proj input is replaced by its mean over 64 calibration prompts
  (16 per format), computed separately per format and token role (names, color words, other tokens by id); a test
  prompt takes at each position the mean of its format and role (unseen roles: the mean over all of the format's
  tokens), as in the template-wise means of Wang et al. (2023). The pooled mean over all formats and positions is kept
  as a sensitivity check (pooled-top16 and 5 pooled-rand16 sets). Controls: 5 random head sets per h with the per-layer
  counts of the top h. Test: the first 300 D1a pairs and the first 300 Q0 pairs of E7, formats EL, AL, NC, ECL, T1.
  About 6.6 GPU hours.
- **E11b degree sweep**: `m0/build_r12.py e11b`, maximum degree Delta in {6, 12, 24, 48}, two hubs of degree Delta per
  graph (one red, one blue), positives are neighbors of the red hub, negatives neighbors of the blue hub, pairs matched
  on degree and the neighbors' degree sum; the positive's position in the red hub's list is k_min >= max(3, Delta/2);
  150 pairs each of D1a and D1c, 3 templates, four formats, four models; prompts shorter than 4,096 tokens (the window
  of Gemma-2). The format x Delta interaction is reported; whether EL reaches NC at Delta = 48 is reported, not
  predicted. About 29.5 GPU hours.
- **E11c (exploratory)**: BLOOM-7B1 and MPT-7B (ALiBi) and Qwen2.5-7B (RoPE) on the same name pool, D1a and D1c, four
  formats, 3 templates; names that are single tokens under all six tokenizers. About 2.9 GPU hours. No prediction.

### 6.3 E12: since probe and zero-parameter prediction
- Probe prompts (`build_r12.py e12`): lines in AL or NC form; the query name occurs once, as the k-th entry of a line,
  k in {1, 2, 3, 4, 6, 8, 12, 16, 24}; question: "is the line-head color of the line that lists x red?". A pair is one
  prompt with only that line-head color changed. 60 pairs per (format, k), four models.
- r(k) is the balanced accuracy at k; the zero-parameter prediction (`e12_probe.py predict`) assigns each positive of
  E1 D1a and D1c the probe's TPR at its k_min (in the list order of the E1 prompt) and each negative the probe's TNR,
  giving a predicted balanced accuracy per k_min stratum for AL and NC with no fitted parameter; the mean absolute
  error against E1 is reported. About 1.1 GPU hours.

### 6.4 E13: GraphQA node tasks
- Node degree, edge existence and node neighbors of Fatemi et al. (2024) (`build_r12.py e13`: G(n, p),
  n ~ U{5..20}, p ~ U[0.1, 0.5], 200 questions per task; edge existence as 200 matched edge / non-edge pairs), formats
  EL, AL, NC, ECL, Incident, Friendship, four models. Edge existence is scored as E1; degree and neighbors are
  generated (greedy, at most 48 tokens) and compared exactly with the truth; the two kinds are reported separately.
  About 7.4 GPU hours.

### 6.5 Reasoning model (outside the theory)
- DeepSeek-R1-Distill-Qwen-7B, D1a, T1, four formats, at most 4,096 reasoning tokens before the answer prompt (scored as
  the CoT of E6). About 24.4 GPU hours. No prediction.

### 6.6 E3: the Delta <= 256 test set, hard pairs and token cost
- `e3/gen_e3.py --split d256`: hub degree Delta in {64, 128, 192, 256} (250 pairs each), n ~ U{Delta+1..Delta+64},
  the red hub is the positive's only red neighbor, random order, AL and NC, 1,000 pairs (seed 7); evaluated in every E3
  run (`acc_d256`) and reported by hub degree.
- Hard pairs (Theorem lb:main): `e3/hard_pairs.py --ser AL|NC --pairs 1000 --seed 8`, the smallest blocking
  construction for L = 6 (AL: N = 10 hubs, k = 3, J = 2, one buffer line per unit, 465 tokens, Delta = 10; NC: N = 12,
  k = 2, J = 2, no buffer, 624 tokens, Delta = 12), fresh names per pair; all pairs of a format are isomorphic. No
  formula of round depth <= 6 separates the two prompts of a pair (`e3/test_hard_pairs.py` checks this exactly with
  `rtypes.py`, and that k = k*(6) separates). Evaluated in every E3 run (`acc_hard`), no extra training. Rule C26:
  1-local within 0.5 +- 0.03, hard since >= 0.93.
- Token cost per format: `m0/token_cost.py` counts the word-level tokens (the tokens of the theory) of the graph text
  on the E1 instances, no model needed; model-token lengths come from the E1 run records.

### 6.7 E11a replication and activation patching
- Replication: Llama-3.1-8B-Instruct, the Qwen2.5-7B procedure (heads ranked on its own scoring prompts, mean ablation
  h in {4, 8, 16}, 5 random sets with the same layer distribution), 300 D1a and 300 Q0 pairs, template T1. Rule C28
  (wider than C22: the top-16 drop >= 0.03 and >= twice the random heads). About 9.6 GPU hours including patching.
- Activation patching (both models, `e11_heads.py patch`): the patched heads' outputs come from a corrupted prompt
  (recolor: all line-head colors swapped red/blue; rename: all line-head names replaced by pool names not in the
  prompt), aligned token by token into the clean forward pass; conditions: the top h in {4, 8, 16} heads and 16 random
  heads (5 sets), AL and NC only. Rule C29. Qwen2.5-7B about 4.2 GPU hours.

### 6.8 Name matching inside the E3 models
- The dia runs of the main grid save their models (`train_e3.py --save-dir ckpt`). `e3/match_heads.py` checks, for
  every head and every name position whose name occurred earlier, whether the largest attention weight falls on an
  earlier occurrence of that name; reported are the error of the most accurate head at 1-16, 17-32, 33-48, 49-64 names
  and the number of heads with accuracy > 0.9. 500 prompts per model, 125 per name-count bin (1-16 from the ID test
  set, the others from the OOD test set), about 0.5 h on one GPU; the time is stored in the sec field of
  `match_heads.jsonl` (e3_match_hours of T_compute). Rule C30.

### 6.9 E3: hard pairs with the proof's parameters
- `hard_pairs.py --proof-L L`, L in {4, 6, 8}: the parameters of the proof of Theorem lb:main (kappa = 1,
  N = 2 + 2(L-1)(kappa+1), buffers >= L^2 + 2L tokens), AL 1,172 / 2,853 / 6,110 tokens, NC 1,335 / 3,510 / 6,402
  tokens; 500 namings per format (seed 9); evaluated by the E3 models of the same depth (L = 4, 8: the 1-local depth
  sweep; L = 6: every arm of the main grid). L = 12 would need about 17,500 tokens and is not generated. Rule C27.
- The finite-depth test of Theorem 8 is infeasible: its counterexample needs m >= 2I + 2 pieces, and I has only a
  tower-exponential upper bound, so no instance fits any context.
- Length: these prompts are 2 to 7 times longer than the longest OOD prompt; the hard-since threshold of rule C25 is
  0.80, fixed before any run, to allow for the loss of length generalization at these lengths.

### 6.10 E11a on Gemma-2-9B, E11b-L, E14, E15, E16 (added before any of their runs)
- **E11a on Gemma-2-9B (third replication)**: the Llama-3.1-8B procedure (own scoring prompts, role-wise mean ablation
  h in {4, 8, 16}, 5 random sets with the same layer distribution), 300 D1a and 300 Q0 pairs, T1, 19 conditions. No
  activation patching. Rule C31 (the criterion of C28). About 5.9 GPU hours.
- **E11b-L degree sweep at fixed length**: `m0/build_r14.py e11bL` pads every E11b pair, per format, with isolated blue
  vertices until the format's graph text is no shorter than the mean length of that format in E11b at Delta = 48
  (word-level tokens: EL about 754, AL about 1,003, NC about 2,010, ECL about 1,018); isolated vertices change neither
  the label nor the matching key; positives keep the position condition of E11b (k_min >= max(3, Delta/2)), otherwise
  the next order seed is used. One record per (pair member, format) (fields ser_only, pad). Delta = 48 is not padded,
  so its prompts are those of E11b. Four models, templates T1-T3. Rule C32: in every model each list format declines
  more than each edge format, the interaction is >= 0.03, and the edge formats decline <= 0.03. About 38.3 GPU hours.
- **E14 text distance versus route**: `m0/build_r14.py e14` takes the first 300 E1 D1a pairs and inserts, in EL and
  ECL, F in {64, 256, 1024} word-level tokens of neutral lines ("Note: this line is filler text.", no names, color
  words or answer words), between the color line and the edge lines (between: the distance from an edge line to its
  endpoints' colors grows by F) or before the color line (before: the prompt grows equally, the distance does not),
  plus F = 0. Field filler (`instances.add_filler`). Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, T1. Rule C33: at F = 1024
  the absolute difference between and before is <= 0.03 (every model; also for ECL). About 2.1 GPU hours.
- **E15 data efficiency**: `e3/run_e15.sh`, the 1-local+hard since arm, task dia, four formats, the first
  N in {25k, 50k, 125k, 250k} training examples (`train_e3.py --train-size`), still 40,000 steps, seeds 0-2; the 500k
  column is E3 seeds 0-2. Reported: ID and OOD balanced accuracy and the examples needed to reach 0.95 (log-linear
  interpolation between neighboring sizes). Rule C34: EL needs >= 1.3 times the examples of each single-match format,
  and all four formats reach >= 0.95 at 500k. 48 runs, about 33.6 GPU hours.
- **E16 multi-token names (exploratory, outside the theory)**: `run_llm.py --ids compound` (two pool names joined by a
  hyphen) and `--ids num4` (four-digit integers, different per graph), the E1 D1a and D1c pairs, four formats, T1-T3,
  Qwen2.5-7B and Llama-3.1-8B. The single-token control is E1 itself. No preregistered prediction. About 5.2 GPU hours.
- Cells that repeat existing prompts are copied: E11b-L at Delta = 48 is E11b, E15 at 500k is E3 seeds 0-2, the
  single-token cells of E16 are E1.

### 6.11 E17: dose of the since-head ablation (added before any of its runs)
- **Purpose**: change only what attention can do, inside one pretrained model. The theory (Section 4, Propositions
  prop:shift and prop:since) says that with since the list formats need one match and therefore lead; without since,
  the list formats keep only their own-line routes (AL needs a nested match), while the two nested matches of EL are
  unaffected, so the order should reverse. E11a ablates 16 heads; E17 raises the dose to 32, 64 and 128 heads to test
  whether the reversal appears inside an LLM.
- **Design**: `m0/e11_heads.py ablate --hs 32,64,128 --no-base`, three models (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B),
  the E11a ranking (same score file), calibration and role-wise means; per h one top-h set and 5 random sets with the
  same layer distribution (seeds 0-4; when a layer has too few other heads, the missing ones come from the nearest
  layers, so every set has h heads); the E11a prompts (the first 300 E1 D1a pairs and the first 300 E7 Q0 pairs, four
  formats, T1). The unablated and top-16 cells are the E11a cells. Summary:
  `e11_heads.py summary results/e11a_*.jsonl results/e17_*.jsonl`, with the flip contrasts (EL minus AL, EL minus NC,
  EL minus the mean of AL and NC, per pair, one-sided 95% pair-bootstrap lower bound). About 14.8 GPU hours.
- **Predictions** (rules C35-C37; P8):
  - **C35 dose monotone**: the balanced accuracy of AL and NC on diamond red does not rise over h in {16, 32, 64, 128}
    (tolerance 0.015).
  - **C36 flip**: at h = 128, in at least two models the one-sided 95% lower bound of EL minus the mean of AL and NC is
    above 0; and in every model the top-128 heads cost the list formats at least 3 times the cost of 128 random heads
    of the same layers.
  - **C37 specificity**: under the top 128, EL and ECL lose at most 0.04 each on diamond red and "v is red" loses at
    most 0.03 in each format; 128 random heads cost each format at most 0.04.
- **Readings fixed before the runs**:
  - C35 fails: the ranking beyond 16 heads is unstable; the dose curve is not interpreted and the flip is not claimed.
  - C37 holds, C36 fails: with most since heads removed the list formats stay at or above EL; since may be spread over
    more than 128 heads, or LLMs have routes outside the theory. The flip inside LLMs is not claimed; the title and
    conclusion state the flip in the controlled models only.
  - C37 fails: ablating 128 heads damages general ability; a flip, if any, is not attributable to since and is
    reported as uninterpretable.
  - Only if C35, C37 and C36 all hold does the paper state that ablating since heads in LLMs reverses the order, and
    use the title "Which Graph Serialization Is Easiest Depends on What Attention Can Do"; otherwise the title is
    "Graph Serializations Differ in What Attention Must Do".

### 6.12 E18: controls of the since-head ablation and the stratified test of P4 (added after E17, before any of their runs)
- **Purpose**: the since heads are selected on list-format prompts, and mean ablation replaces the heads' output at
  every position, so the asymmetry (only the list formats lose) could come from the selection itself or from the
  heads' role elsewhere (for example at the answer position). E18 tests this with three controls; P4's two point
  checks become tests.
- **Design** (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B; the E11a calibration, role-wise means and prompts: the first 300
  E1 D1a pairs and the first 300 E7 Q0 pairs, four formats, T1):
  - **(a) Edge-format heads**: `m0/e11_heads.py score-edge` scores every head on the EL prompts of the same 100
    scoring pairs by the attention of an edge line's second endpoint to its own entry in the color line (name and
    color) minus the mean attention to other color-line entries (the inner match of Proposition prop:shift(b)).
    Ablation of the top 16 and top 128, and of 5 random sets of 128 heads in the same layers
    (`ablate --scores results/e18_edge_scores.json --prefix e --hs 16,128 --no-base`; variants etop<h>,
    erand<h>s<seed>).
  - **(b) Ablation at the entries only**: the top-16 and top-128 since heads of the E11a ranking, replaced only at the
    list-entry name tokens (under EL and ECL at the names in edge lines; `ablate --prefix n --positions entries
    --hs 16,128 --no-base --no-random`; variants ntop<h>), with the flip contrasts (EL minus AL, NC and their mean, per
    pair, one-sided 95% lower bound).
  - **(c) A task that needs no since**: GraphQA node degree (the 200 E13 questions per format; under the list formats
    it suffices to count v's own line), greedy generation of at most 8 tokens under the top-128 since heads and 5
    random sets of 128 heads in the same layers (`ablate --task degree --hs 128 --no-base`); the unablated cells are
    those of E13.
  - **P4 test**: `m0/r9_stats.py p4ni` computes, per k_min stratum (1, 2, 3-4, >=5), the four-model mean shortfall of
    ECL against each model's better list format and its upper 95% pair-bootstrap bound; `r9_stats.py kdiff`
    additionally gives NC's decline from k_min = 1 to >= 5 minus ECL's decline (nc_minus_ecl) and its one-sided 95%
    lower bound.
  - About 16.4 GPU hours (three models).
- **Predictions** (rules C38-C42):
  - **C38 edge-format heads**: at h = 128, in at least two models EL loses >= 0.05 on diamond red and >= 3 times the
    mean loss of AL and NC; "v is red" under EL loses >= 0.05; ECL loses less than EL; AL and NC are within 0.015 of
    the random heads of the same layers.
  - **C39 entries only**: at h = 128, in at least two models the one-sided 95% lower bound of EL minus the list mean is
    above 0; AL and NC keep >= 70% of the all-position loss.
  - **C40 degree**: under the top 128, GraphQA degree loses <= 0.05 under AL and under NC, and at most a third of the
    same model's diamond-red loss.
  - **C41 P4 non-inferiority**: in every k_min stratum, the upper 95% bound of the four-model mean shortfall of ECL
    against the better list format is <= 0.02.
  - **C42 direct contrast**: NC's decline minus ECL's decline has a one-sided 95% lower bound above 0.
- **Readings fixed before the runs**:
  - C38 fails: the list formats lose under any 128 heads of these layers; the since-head result is read as
    correlational only. If C38 holds, the list formats are not generally more fragile under this ablation (heads
    selected for one format's route hurt that format); whether the since heads matter for the list formats only
    because they were selected on list prompts is tested by C39 and C40.
  - C39 fails: the heads act elsewhere (for example at the answer position); the route is not localized at the entries.
  - C40 fails: the ablation damages general reading of the list formats; the asymmetry is not attributable to since.
  - C41 holds in some strata only: P4 holds in the strata that pass, as stated; C42 fails: the k_min strata do not
    separate ECL from NC.

### 6.13 E19 attention knockout and entry patching, E20 within-line filler, E21 relational tables (added before any of their runs)
- **Purpose**: (1) cut the route directly, without selecting heads; (2) change the distance to the line head at a
  fixed list position, separating a distance decay of approximate since from a general locality preference; (3) apply
  the derivation of formats from attention capabilities to a second surface domain.
- **E19 design** (`m0/r19.py knockout`, `patch`; three models, the E11a prompts: the first 300 E1 D1a pairs and the
  first 300 E7 Q0 pairs, AL and NC, T1):
  - Knockout: in every layer and head, the attention of every list-entry name to its own line-head segment (from the
    line head to just before the first entry: line-head name, color and separators) is removed (a custom 4D attention
    mask adding -inf before the softmax; the attention knockout of Geva et al. 2023); control: the same entries lose
    their attention to the previous line's head segment instead (entries of the first line unchanged). Variants
    ko-own, ko-other.
  - Entry patching: the top-128 since heads of the E11a ranking, recorded on the prompt with all line-head colors
    flipped and written back only at the entry-name positions of the original prompt (variant
    patch-recolor-entries128, D1a only).
- **E20 design** (`m0/r19.py filler`; the same three models and E11a D1a prompts, AL and NC): every list line gets
  F in {8, 24} neutral words (a cycle of "this filler text carries no content"; no names, color words or answer
  words), between the line head and the first entry (inside: every entry moves F words away from its line head) or
  after the last entry (outside: the prompt grows equally, the distance does not). The edge-format counterpart is E14.
- **E21 design** (`m0/r19.py tables-build`, `tables-run`; four models, `data/e21_instances.jsonl`, seed 21): projects,
  people, a membership relation and a binary level of people (senior or junior, P(senior) = 0.25; 6-9 projects and
  10-14 people per graph, 2-5 members per project). Four serializations: rows (a level line and one line per
  membership, as EL), person (one line per person with level and projects, as AL), project (one line per project with
  its members and their levels, as NC), inline (rows with each member's level, as ECL). Questions T1 "Does project p
  have at least one senior member?" (depth one) and G1 "Are at least four people senior?" (global); 200 pairs each
  (T1: two query projects with the same number of members in one graph; G1: one graph with one person's level
  flipped, 4 versus 3 seniors), templates T1, T2.
- About 7.3 GPU hours (E19 1.6, E20 2.4, E21 3.3).
- **Predictions** (rules C43-C49):
  - **C43**: in at least two models, ko-own costs AL and NC each >= 0.05 on diamond red and >= 3 times the control
    (ko-other); "v is red" loses <= 0.02 each.
  - **C44**: in at least two models, entry patching costs AL and NC at least the loss of ablating the same heads at the
    entries only (E18).
  - **C45**: in at least two models, at F = 24 inside minus outside is <= -0.03 with a one-sided 95% upper bound < 0
    (AL and NC); the cost at F = 24 is not smaller than at F = 8 (tolerance 0.01).
  - **C46**: the mean contrast of the list formats at F = 24 is more than 0.02 below the mean contrast of EL and ECL at
    F = 64 in E14 (where the distance grows more).
  - **C47**: on depth one, project > person > rows (four-model mean and at least three models).
  - **C48**: the four-model mean of inline is not more than 0.02 below the better of project and person.
  - **C49**: on the global question the four-model means of the four formats differ by <= 0.02.
- **Readings fixed before the runs**:
  - C43 fails: the color reaches the entries by another path (relay), or the knockout breaks the prompt in general;
    the paper does not state that the lead of the list formats depends on direct attention from entries to their own
    line head.
  - C44 fails: the heads matter but do not carry the line-head color.
  - C45 fails: the list-position effect of E11b-L does not come from distance; C46 fails: a general locality preference
    explains both kinds of format.
  - C47-C49 fail: the graph results depend on graph vocabulary, or the format differences come from general difficulty;
    the transfer is not claimed and only the numbers are reported.
- **Scope**: a relational instance is a colored bipartite graph whose project vertices are uncolored; the constructions
  of Propositions prop:shift, prop:since and Lemma lem:ecl use only the line structure and apply unchanged; the lower
  bounds are not re-proved for bipartite instances. E21 tests a second surface domain of the same logical structure,
  not questions that need joins over several tables.

### 6.14 Sixth preregistered set: E22-E33 (added before any of their runs; E33 and C67 last, still before any E33 run)
- **Purpose**: (1) a positive control for P1; (2) whether small soft-attention models show the format order under
  limited data (the bridge from the theory to LLMs), and the sample efficiency of fine-tuning pretrained models per
  format; (3) how a local window w moves the frontier; (4) a direct test of Theorem md:el and Proposition leap:budget
  with one layer of global heads (match-nesting depth <= 1); (5) depth two and counting in the controlled models;
  (6) the mechanism evidence across templates, ranking stability, the knockout at 72B and sufficiency patching; (7) the
  NC routes at high degree and a within-line versus between-line filler of equal length; (8) coverage of LLM settings:
  ECL under CoT and a reasoning model, all question classes on GPT-4o, integer and multi-token names, large graphs,
  few-shot prompts, system prompts and edge orders, a scale series of one family (base and instruct), ALiBi models,
  algorithmic tasks, real topologies on four models, oversampled late positions; (9) seeds and width of the controlled
  experiments; (10) whether the learning cost comes from nesting or from the distance between colors and edges (E33).
- **E22 NoPE positive control** (`e3/r20_e3.sh pos-control`; NoPE arm, L = 6, 5 seeds, the E3 generator and training
  setup): tasks G1 ("some vertex is red") and G2 ("at least three vertices are red", the E1 question; task name c3 in
  the data code), a pair is one graph with one vertex's color flipped; four formats; ID and OOD test sets as E3.
  Theory: G1 is definable with P alone under all four formats, G2 with P alone under EL and AL; under NC and ECL the
  definition of Proposition new:prop-global uses Y, and definability with P alone is not proved (the remark after the
  proposition gives NC graphs with equal token-class counts).
- **E23 soft attention with limited data** (`e3/r20_e3.sh data-limited`; arms RoPE, ALiBi-only, 1-local+ALiBi and
  1-local+hard since, diamond red, N in {25k, 50k, 100k, 200k, 500k} training examples, 5 seeds, 40,000 steps,
  evaluation on 2,000 ID validation examples every 250 steps): reported are the examples needed to reach 0.95 (N95,
  log-linear interpolation between the final ID accuracies at each N) and the steps to reach 0.95 at 500k.
- **E24 fine-tuning pretrained models per format** (`m0/lora_r20.py`; the base models Qwen2.5-0.5B and Qwen2.5-1.5B,
  LoRA r = 16, alpha = 32 on the q, k, v, o projections, learning rate 2e-4, 3 epochs, 3 seeds): training sets of 1k,
  4k, 16k D1a instances from the E1 distribution (seed 24, disjoint from the E1 test graphs), four formats; test: the
  E1 D1a test pairs (400 pairs, T1) and larger graphs of 60-100 vertices (200 pairs).
- **E25 local window** (`e3/r20_e3.sh window`; w-local arm: 4 global heads and max(4, w) local heads with fixed offsets
  1..w (for w = 2: 1, 2, 1, 2), 32 dimensions per head, width 384 at w = 8; w in {2, 4, 8}, L in {3, 4, 6}, AL and NC,
  diamond red, 3 seeds; on the k-controlled sets by k_min). Theory: upper bounds
  k*_{AL,w}(L) = max(0, floor(((L-1)w-1)/2)), k*_{NC,w}(L) = floor(((L-1)w+1)/5) (Corollary ehat:window); lower bound
  k*_F(wL) (Proposition ehat:local).
- **E26 one layer of global heads** (`e3/r20_e3.sh nest`; L = 6, 8 1-local heads in the first five layers, 4 global and
  4 1-local heads in the last layer; control: the 1-local arm of E3; EL and ECL, diamond red, 3 seeds): test graphs of
  maximum degree 1 (perfect matchings), blue query vertices, E in {8, 16, 32, 64, 128} edges, 2,000 pairs per E (two
  blue query vertices of one graph whose partners are one red and one blue); training: half graphs of this kind with
  E <= 32, half the E3 training set.
- **E27 depth two and counting** (`e3/r20_e3.sh depth2`; the 8-local arm (w = 8, L = 6), 1-local+hard since, RoPE and
  1-local (the control of C57); tasks DD (diamond diamond red, blue query vertex) and C2 (diamond_{>=2} red), graphs of
  maximum degree <= 3, ID 8-16 vertices, OOD 17-40 vertices; four formats, 3 seeds).
- **E28 mechanism replication** (`m0/r20.py`; Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B; the E11a prompts): (a) the top-16
  since-head ablation of E11a and the E19 knockout under templates T2 and T3; (b) ranking stability: the 100 scoring
  prompts split randomly in two halves (10 times), Jaccard overlap of the two halves' top-16 and top-128 head sets;
  (c) the E19 knockout on Qwen2.5-72B; (d) sufficiency patching: run on the prompt with all line-head colors flipped
  and write the clean prompt's top-128 since-head outputs back at the entry-name positions only (the reverse of E19),
  control 5 random sets of 128 heads.
- **E29 NC routes at high degree** (`m0/r20.py routes-degree`; the fixed-length degree sweep of E11b-L,
  Delta in {6, 12, 24, 48}, three models): variants NC, NC-H, NC with masked line heads, NC with masked entries.
- **E30 within-line versus between-line filler** (`m0/r20.py filler-between`; the E20 prompts and models): F = 24
  neutral words per list line, between the line head and the first entry (inside, as E20) or as a separate line after
  the line (between: the within-line distance is unchanged, the prompt equally long).
- **E31 coverage of LLM settings** (`m0/r20.py`; unless stated otherwise four open models, 200 pairs, T1):
  - (a) CoT (as E6) on the four question classes and five formats (including ECL); the reasoning model
    DeepSeek-R1-Distill-Qwen-7B on the four classes and five formats;
  - (b) GPT-4o on the four classes and five formats (the E10 setup, T1 only): the 6 new depth-two and counting
    questions in five formats and the 6 global and depth-one questions under NC-H, 200 pairs each, 14,400 calls (the
    global and depth-one cells of the four formats come from E10);
  - (c) integer ids (four digits) and compound names (two names joined by a hyphen) under the main protocol, four
    classes;
  - (d) large graphs: 100, 200, 500 vertices (mean degree 4), the position of v's line controlled to the first, middle
    or last third; Gemma-2-9B at 100 vertices only (8k context);
  - (e) few-shot K in {1, 5, 10}, two system prompts, three edge orders under EL (random, by source vertex, BFS),
    Qwen2.5-7B and Llama-3.1-8B, depth-one and global questions;
  - (f) scale series: Qwen2.5 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B instruct and 0.5B, 1.5B, 7B base, depth-one and global
    questions;
  - (g) the ALiBi models BLOOM-7B1 and MPT-7B on the E11c name pool, D1a by k_min;
  - (h) algorithmic tasks: connectivity (are u and w connected), cycle (does the graph have a cycle), shortest path (is
    the distance of u and w at most 2), 200 pairs each;
  - (i) real topologies (the molecule and citation subgraphs of E5) on four models, D1a and D2a;
  - (j) late positions oversampled: 1,000 D1a pairs with k_min >= 5 in the AL order (power 0.8 at alpha = 0.05 for a
    3-point difference between strata).
- **E32 seeds and width of the controlled experiments** (`e3/r20_e3.sh robust`): the depth sweep extended to 5 seeds;
  E3 repeated for the 1-local and 1-local+hard since arms at d in {128, 256, 512}.
- **E33 nesting versus distance** (`e3/r20_e3.sh nest-dist`, `m0/lora_r20.py --ser ELB`,
  `m0/run_llm.py --serializations ELB`): format ELB (`m0/serialize.py`, `edge_blocks`) splits the edges of EL, in the
  same random order, into blocks of 3, each preceded by a line `Colors:` with the names and colors of the block's
  endpoints (each endpoint once, random order), and a last line with the isolated vertices; colors are therefore a few
  lines from the edges, but reading a color still requires matching by endpoint name: the formula of Proposition
  prop:shift(b) holds unchanged under ELB (two nested matches), no single-match formula is known, and the lower bound of
  Theorem md:el is not re-proved for ELB. (a) The four arms of E23 under ELB, with the E3 diamond-red graphs, splits
  and seeds, and the N, steps, seeds and validation schedule of E23 (100 runs); EL, AL, NC, ECL come from E23. (b) The
  two base models of E24 fine-tuned under ELB, with E24's data, setup and seeds (18 runs). (c) The four open models
  under ELB on G1-G3 and D1a-D1c, with the E1 protocol (the E1 pairs, 400 pairs per question, three templates, E1
  scoring); the other formats come from E1.
- About 314 GPU hours and 14,400 API calls (E22 6, E23 64, E24 18, E25 27, E26 9, E27 16, E28 21, E29 8, E30 3, E31 98,
  E32 20, E33 24: controlled and LoRA 20, LLM 4).
- **Predictions** (rules C50-C67; tolerances from sampling error: one cell of 200 pairs has a standard error of about
  0.02, so comparisons between cells mostly use model means):
  - **C50** (P1 control): NoPE on G1 reaches OOD >= 0.95 in all four formats; on G2 EL and AL reach OOD >= 0.95, NC and
    ECL are at least 0.10 lower.
  - **C51** (P3): in all four arms, EL's N95 is at least 1.3 times the smaller of AL's and NC's, and ECL's N95 is at
    most 1.1 times it; at 500k the hard-since arm needs >= 1,000 more steps to reach 0.95 under EL than under AL and NC
    (95% interval over seeds excludes 0).
  - **C52** (P3): in every soft-attention arm, at the smallest N where any format first exceeds 0.90, EL is >= 0.03
    below the best list format; at 500k the four formats are within 0.03 OOD (as in E3).
  - **C53** (P3, pretrained models): in both models, at 1k and 4k examples EL is >= 0.03 below the better of AL and
    NC, and ECL is not more than 0.02 below it; at 16k EL's gap is smaller than at 1k.
  - **C54** (P2 generalized): the w-local arm reaches >= 0.90 in the strata with k_min <= k*_{F,w}(L) and <= 0.56 in
    the strata with k_min > k*_F(wL); strata in between are not predicted.
  - **C55** (Theorem md:el, Proposition leap:budget): with one layer of global heads, ECL reaches >= 0.95 at every E;
    EL at E = 128 is at least 0.15 below E = 8 and at least 0.15 below ECL; the control arm reaches >= 0.95 under EL at
    every E.
  - **C56** (Proposition new:prop-d2-nc, Theorem new:thm-d2-lower): on the 8-local arm, DD under NC reaches >= 0.90 ID
    and OOD, DD under AL stays <= 0.56 OOD; EL is not predicted.
  - **C57** (Corollary cor:count): C2 on the hard-since arm reaches OOD >= 0.95 in all four formats; on the 1-local arm
    C2 reaches >= 0.95 under EL, and AL and NC are at least 0.10 below EL.
  - **C58** (P6): under T2 and T3 the top-16 ablation and the knockout cost the list formats >= 0.05 each, within 0.03
    of the loss under T1; "v is red" <= 0.02.
  - **C59**: mean Jaccard overlap of the halves' top-16 >= 0.6, top-128 >= 0.7 (three models).
  - **C60** (P6): on Qwen2.5-72B the knockout costs AL and NC >= 0.05 each and >= 3 times the control.
  - **C61**: in at least two models sufficiency patching restores >= 0.30 of the clean-minus-flipped difference of AL
    and NC, 128 random heads <= 0.05.
  - **C62** (P7; mean of three models, 150 pairs per cell): at Delta = 48 the decline of NC with masked line heads
    relative to Delta = 6 is within 0.04 of AL's decline; the declines of NC-H and of NC with masked entries are both
    >= 0.04 smaller than that of NC with masked line heads.
  - **C63** (P7): inside minus between <= -0.03 (AL, NC, at least two models, one-sided 95% upper bound < 0); the loss of
    between relative to no filler is at most the loss given by the E11b length factor plus 0.02.
  - **C64** (recommendation): under CoT, depth-one ECL >= max(AL, NC) - 0.02 (four-model mean) and the range of the
    five formats on global questions <= 0.03 (four-model mean); for the reasoning model depth-one ECL >= max(AL, NC) -
    0.05; on GPT-4o NC >= AL > EL (depth one and depth two), ECL >= NC - 0.01 (depth one), range on global
    questions <= 0.03.
  - **C65** (scope of the recommendation): with integer ids and compound names at least three models keep depth-one
    EL < AL < NC and the four-model mean has ECL >= max(AL, NC) - 0.03; on large graphs of 100 and 200 vertices the
    four-model mean keeps NC > AL > EL, ECL >= max(AL, NC) - 0.03 at each size, and the range over the three positions
    of v's line (mean of list formats and models) is <= 0.03; at least 80% of the cells of the few-shot, system-prompt
    and edge-order variants keep the order and the recommendation; in the scale series every model >= 3B has a
    depth-one list lead (max(AL, NC) - EL) >= 0.03 and a range of the four formats <= 0.05 on global questions, and
    base and instruct models have the same depth-one order; the ALiBi models decline from k_min = 1 to >= 5 by >= 0.02
    more in the list formats than Qwen2.5-7B on the same name pool; on the algorithmic tasks ECL >= EL - 0.02
    (four-model mean); on real topologies at least three models keep the depth-one order, within 0.05 of the
    prediction from the k_min shares; on the oversampled k_min >= 5 set, NC - EL >= 0.05 in every model (95% interval
    lower end > 0), and the four-model mean of ECL is within 0.03 of its E1 k_min = 1 stratum.
  - **C66** (P2 robustness): over the 5 seeds of the depth sweep and widths 128, 256, 512, >= 0.90 inside the frontier
    and <= 0.56 outside it.
  - **C67** (P3, nesting versus distance): (a) in every arm ELB's N95 >= the geometric mean of the N95 of EL and ECL;
    (b) in both models at 1k and 4k examples ELB is >= 0.03 below the better of AL and NC; (c) on the four-model mean,
    depth one (D1a-D1c) ELB <= EL + 0.04 and <= ECL - 0.05, and the range of EL, ELB, ECL on global questions (G1-G3)
    <= 0.03.
- **Reporting of C65** (fixed before the runs): each of its eight parts (c)-(j) (names, large graphs, prompts, scale,
  ALiBi, algorithmic tasks, real topologies, late positions) keeps its criterion; table T_prereg_r20b gives the outcome
  per part ("consistent in all eight parts", or the parts that fail); the checker decides the rule as a whole.
- **Reporting of C51** (fixed before the runs): C51 has an EL clause (EL's N95 >= 1.3 times, and the hard-since step
  difference) and an ECL clause (ECL's N95 <= 1.1 times). The checker decides the rule as a whole; if only the ECL
  clause fails, table T_prereg_r20 states "violated (ECL clause)" and the appendix gives ECL's largest ratio against
  the 1.1 the rule allows.
- **Readings fixed before the runs**:
  - C50 fails: NoPE's failure may be a general length-extrapolation problem; P1 cannot be attributed to missing shifts
    alone.
  - C51, C52 fail: the controlled soft-attention models show no format order under limited data, so the LLM order
    cannot be attributed to nested matches being harder to learn; C53 fails: the sample efficiency of pretrained
    models does not support the hypothesis.
  - C54 fails: the frontier does not move with the local window as the theory says, and the 1-local step cannot be
    read as a shift budget.
  - C55 fails: a model with one layer of global heads solves EL on large graphs without nested matches, or fails even
    ECL; the lower bound is unrelated to trained models.
  - C56, C57 fail: the constructions or lower bounds of depth two and counting do not act in trained models.
  - C58-C61 fail: the mechanism evidence depends on the template, on chance in head selection or on model scale, or
    the heads are necessary but not sufficient.
  - C62, C63 fail: the difference between AL and NC at high degree does not come from the routes, or the distance cost
    comes from general locality.
  - C64, C65 fail: the recommendation is limited to the settings of the original protocol (direct answers,
    single-token names, at most 40 vertices); the scope sentence in the paper says so.
  - C66 fails: the frontier depends on seeds or width.
  - C67 fails: EL's learning cost comes from the distance between colors and edges, not from nesting depth; the
    learnability hypothesis cannot be tied to the nesting depth of Theorem md:el.
- **Scope**: E22, E25-E27, E32 test consequences on finite graphs of proved results (Proposition new:prop-global,
  Corollary ehat:window, Theorem md:el and Proposition leap:budget, Proposition new:prop-d2-nc and Theorem
  new:thm-d2-lower, Corollary cor:count, Theorem lb:main); E23, E24, E33 test the hypothesis that nested matches are
  harder to learn (the premise of P3 and P4), not a theorem; E28-E31 test the predictions and the recommendation for
  LLMs; the algorithmic tasks (E31h) are outside the theory and only check that the recommendation costs nothing.

### 6.15 Seventh preregistered set: E34-E36 (added before any of their runs)

Three tests of the since-head evidence: whether the heads carry the line-head color or, more generally, which line an
entry belongs to (binding); whether their specificity only reflects a large output norm; whether the flip at 128 heads
depends on template T1, the wording of D1a or model scale. Rules V-C68 to V-C74 (paper table T_prereg_r21,
VALIDATION_RULES.md, seventh set), each with the reading of both outcomes. Code `m0/r21.py`, summary
`results/blocks_r21.json`, import `analysis/import_measured.py --r21`.

- **E34 binding control** (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B): GraphQA edge existence (the 200 pairs per format of
  E13, T1 layout, scored by the Yes/No log-probabilities of E1). Under AL and NC the answer requires knowing which line
  lists the second name, not the line-head color. Conditions: the top-128 since heads of the E11a ranking (all
  positions, role-wise mean ablation), 5 random sets of 128 heads with the same per-layer counts; the E19 knockout
  (own: entry to its own line-head segment; other: to the previous line's head segment; list entries only). The
  unablated cells are those of E13.
- **E35 norm-matched null** (the same three models; the E11a/E17 prompts: the first 300 E1 D1a pairs and E7 Q0 pairs,
  T1, four formats): 100 random sets of 128 heads, each head drawn from the heads outside the top 128 in the same layer
  and the same decile of head-output norm (the mean L2 norm of the head's slice of the o_proj input over all positions
  of the 64 E11a calibration prompts; deciles over all heads of the model; if no head is available, the same layer,
  then the nearest layer with the same decile). The top-128 cells are those of E17; the exceedance fraction is
  e = (1 + number of sets that lose at least as much as the top 128) / 101.
- **E36 replication at h = 128**: (a) templates T2 and T3 (the E28 prompts), three models, top 128 and 5 random sets of
  128 heads; the unablated cells are those of E28a. (b) The paraphrase D1c (first 300 pairs, T1), three models,
  unablated and top 128. (c) Qwen2.5-72B-Instruct (D1a, T1, first 300 pairs, four GPUs), heads ranked by its own
  attention as in E11a; h = 128 (2.5% of its 5,120 heads) and h = 640 (12.5%, within the 12 to 19% that 128 heads make
  up in the smaller models), each with 5 random sets.
- Compute (A100-80GB GPU hours, at the rates of E17): E34 about 2.1, E35 about 78.9, E36 about 20.3.

## 7. Statistics
- Balanced accuracy; 95% intervals from 2,000 bootstrap resamples **of pairs, stratified by question** (S-4; every
  question keeps its number of pairs); all prompts of a pair enter together.
- Format comparisons: graph-level paired permutation tests (10,000 permutations, the 3 templates aggregated per graph,
  S-3).
- Interaction tests: GEE (exchangeable correlation, clustered by graph: a query-vertex pair is one graph, the two
  graphs of a global pair are two clusters); the robustness version adds question fixed effects (class x question
  index within class, `C(cls) + C(qi) + C(cls):C(qi)`), Wald test with 6 degrees of freedom.
- **Planned contrasts (S-1, preregistered, primary test)**: Delta_c = (NC - EL)_c - (NC - EL)_global for c in depth
  one, depth two, counting; the same for AL - EL. One-sided tests: z = Delta / se, se the standard deviation of the
  pair-bootstrap distribution, normal-approximation p, Holm over the 6 contrasts within a model; the empirical
  bootstrap p = (1 + b) / 2,001, with b the number of the 2,000 resamples with Delta <= 0, is also reported. Implementation:
  `m0/stats.py::planned_contrasts`.
- Pseudo-R^2 is reported only as the increment of the interaction terms (S-2).
- Degree and distance effects (S-5): per format, logistic regression of correctness on log d_min, k_min, log prompt
  length, n and deg v.
- Multiple comparisons: Holm, per family (the paper's appendix, statistics section).
