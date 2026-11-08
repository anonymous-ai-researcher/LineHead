# Run protocol

The commands that produce every result of the paper, the format of each result file, and the import into
`results/summary.json`. The configuration (hardware, software, hyperparameters, data distributions) is CONFIG.md;
the rules that check the results are VALIDATION_RULES.md.

- Every seed and every run is reported. A run that crashes or produces NaN may be rerun; the number of reruns and the
  reason go to `DEVIATIONS.md`.
- No tuning on test sets: the hyperparameters are fixed in CONFIG §3.4 (the only tuning step is the preregistered
  learning-rate sweep described there).
- Deterministic scoring: LLM scoring compares the log-probabilities of Yes and No, without sampling. Only the CoT of E6,
  the generation check of E8 and the generated answers of E13, E18c and E31 produce text, all with greedy decoding.
- One name list: all HF models use the same `--names-from` list (the intersection over the four tokenizers), so the
  four models see identical prompts.
- Tokens (F-1): every line of a serialization starts with a space, so every name is preceded by a space. At start-up
  `run_llm.py` checks on the first 200 prompts that all occurrences of a name map to the same token id and exits
  otherwise.
- The predictions and their falsification criteria are fixed in PREREGISTRATION.md, CONFIG §6 and VALIDATION_RULES.md
  before the runs; results are reported as they come out.

---

## 0. Overview

| Id | Content | Code | Compute | Result files |
|---|---|---|---|---|
| E0 | pilot, name-blind and names-kept grid, preregistered ALiBi test | `pilot1/run_grid.sh`, `pilot1/run_alibi.sh` | CPU | `pilot1/results.jsonl`, `results_alibi.jsonl` |
| E0b | pilot scaling grid for X1-NC (72 runs); chance cells rerun with seeds 0-2 (X2 4,000 steps; X1, D1a 8,000 steps) | `pilot1/run_e0b.sh` | about 493.7 single-process CPU hours, or one A100 | `pilot1/results_e0b.jsonl`, `results_chance3.jsonl` |
| E3 | preregistered learning-rate sweep (3 runs); 6 mechanisms x 2 tasks x 4 formats x 5 seeds (240 runs) | `e3/make_data.sh`, `e3/run_lr_sweep.sh`, `e3/run_e3.sh` | 2.1 + 168.0 GPU hours | `e3/results/lr_sweep.jsonl`, `lr_choice.json`, `e3.jsonl` |
| E3d | depth sweep (1-local, L in {2, 3, 4, 6, 8, 12}; L = 6 reuses E3 seeds 0-2, 60 new runs) | `e3/run_e3d.sh` | about 40.8 GPU hours | `e3/results/e3d.jsonl` |
| E1 | 4 LLMs, 12 questions x 3 formats x 3 templates | `m0/run_llm.py`, `m0/stats.py` | 4.4 / 4.6 / 5.2 / 30 GPU hours | `m0/results/e1_<model>.jsonl`, `summary_e1.json` |
| E2 | name-blind controls (query questions; positive controls X1, X2), 4 models | `run_llm.py --mask all` | about 51.6 GPU hours | `e2_<model>*.jsonl`, `summary_e2.json` |
| E4 | format ablations ECL, EL2, ELfar, NCH, NCnw, Incident, Friendship, 4 models | `run_llm.py`, `m0/e4_stats.py` | about 36.8 GPU hours | `e4_<model>.jsonl`, `summary_e4.json` |
| E7 | partial name masking (AL, NC; D1a, Q0), random masks, twin Q0 pairs, 4 models | `run_llm.py --mask ...` | about 27.0 GPU hours | `e7_<model>_<mask>.jsonl`, `summary_e7.json` |
| E5 | real topologies (molhiv, arxiv) and GraphML | `m0/build_real.py` | about 2.6 GPU hours, plus the OGB download | `e5_*.jsonl`, `summary_e5_*.json` |
| E6 | 3-shot and zero-shot CoT (Qwen2.5-7B, 72B, depth one) | `run_llm.py --shots/--cot` | about 64.5 GPU hours | `e6_*.jsonl`, `summary_e6.json` |
| E8 | robustness (Qwen2.5-7B, Llama-3.1-8B) | `run_llm.py` options | about 24.0 GPU hours | `e8_<model>_<variant>.jsonl`, `summary_e8.json` |
| E9 | mechanism evidence (Qwen2.5-7B) | `m0/e9_mechanism.py` | about 6.5 GPU hours | `e9_*.json` |
| E3x | learning-rate check of NoPE and RoPE (6 runs); NC-H arm (20 runs); position-controlled sets evaluated in every run | `e3/run_e3_extra.sh`, `e3/r9_e3.py` | about 18.2 GPU hours | `lr_check.jsonl`, `e3_nch.jsonl`, `summary_r9_e3.json` |
| E10 | gpt-4o-2024-08-06 (top-20 logprobs, max_tokens 1) | `run_llm.py --backend openai` | 28,800 calls | `e10_gpt4o.jsonl`, `summary_e10.json` |
| E11-E21 | Section 4.13 to 4.18 | `m0/e11_heads.py`, `m0/r19.py`, `e3/` | see CONFIG §6.2-§6.13 | see the sections |
| E22-E33 | Section 4.19 | `e3/r20_e3.sh`, `e3/gen_r20.py`, `m0/r20.py`, `m0/lora_r20.py`, `m0/r20_blocks.py` | about 314 GPU hours and 14,400 API calls | `e3/results/r20_*.jsonl`, `m0/results/e2[4-9]*`, `e3[01]*`, `blocks_r20.json` |
| E34-E36 | Section 4.20 | `m0/r21.py` | about 101 GPU hours | `m0/results/e34*`, `e35*`, `e36*`, `blocks_r21.json` |

Suggested order: data (Section 3); E1 on Qwen2.5-7B first, checking the token checks; E2, E4, E7 (they reuse the E1
data); the other three models; E5, E6, E8, E9, E10; Sections 4.13 to 4.20. E3, E3d, E0 and E0b run in parallel.

---

## 1. Code

| Folder | Content |
|---|---|
| `m0/` | instances, LLM runs and statistics (see `m0/README.md`) |
| `e3/` | data generator, training and evaluation of the controlled models |
| `pilot1/` | the pilot and its scaling grid |
| `theory_checks/` | numeric checks of the constructions and lower bounds (see `theory_checks/README.md`) |
| `analysis/` | import of the summaries, validation rules, table export, figure scripts |

---

## 2. Environment (CONFIG §0)

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip freeze > env.txt; nvidia-smi > gpu.txt
export PYTHONHASHSEED=0
```

- Accept the licenses of Llama and Gemma on Hugging Face first.
- `run_llm.py` writes the commit hash of the loaded weights to `m0/results/model_revisions.json`.

---

## 3. Data (CONFIG §1, §3.3, §5)

The fenced blocks of Sections 3, 4 and 7 are all the commands, to be run in the order in which they appear. The first
block defines the variables used later.

```bash
# 0. variables used by every block below
set -e
: "${ROOT:=$(pwd)}"                      # repository root: the folders m0/ e3/ pilot1/ analysis/
: "${BACKEND:=hf}"                       # LLM backend of E1-E8
: "${API_BACKEND:=openai}"               # E10
: "${E9_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${E11_REPL:=meta-llama/Llama-3.1-8B-Instruct}"   # E11a replication
: "${E11_REPL2:=google/gemma-2-9b-it}"             # E11a on a third model
: "${NAMES=Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,google/gemma-2-9b-it,Qwen/Qwen2.5-72B-Instruct}"
MODELS="Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b-it Qwen/Qwen2.5-72B-Instruct"
PAIR_MODELS="Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct"     # E5, E8
E6_MODELS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-72B-Instruct"
E14_MODELS="Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b-it"
: "${R20_BASE:=Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B}"             # E24 base models
: "${R20_SCALE:=Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct Qwen/Qwen2.5-72B-Instruct}"
: "${R20_SCALE_BASE:=Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-7B}"
: "${R20_ALIBI:=bigscience/bloom-7b1 mosaicml/mpt-7b}"           # E31g (with Qwen2.5-7B on the same pool)
: "${R20_R1:=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"           # E31a reasoning model
: "${R20_BIG:=Qwen/Qwen2.5-72B-Instruct}"                         # E28c
: "${R20_HF4:=$MODELS}"                                           # r20.py runs on the four LLMs
: "${R20_HF2:=$PAIR_MODELS}"                                      # r20.py prompting
NAMES_ARG=${NAMES:+--names-from $NAMES}  # one name list for all models: the 4-tokenizer intersection
batch() { case "$1" in *72B*) echo 4;; *) echo 16;; esac; }
B() { echo "--backend $BACKEND --model $1 $NAMES_ARG --batch $(batch $1)"; }   # common run_llm options
```

```bash
# 3. data (m0/data ships a copy generated by exactly these commands; regenerating gives the same bytes)
cd "$ROOT/m0"
python build.py --pairs 400 --seed 0 --out data/m0_instances.jsonl            # E1: 12 questions x 400 pairs = 9,600
python build.py --pairs 50  --seed 7 --pids D1a,D1b,D1c --out data/demo_instances.jsonl   # E6 demonstrations
python build.py --pairs 400 --seed 5 --pids Q0 --out data/e7_instances.jsonl              # E7: Q0 only; D1a from E1
python build.py --pairs 200 --seed 3 --colors 3 --pids G1,G2,G3,D1a,D1b,D1c --out data/e8_colors3.jsonl
python build.py --pairs 200 --seed 9 --pids G1,G2,G3,D1a,D1b,D1c --out data/e10_instances.jsonl
python build_blind.py --pairs 400 --seed 11 --out data/e2_controls.jsonl      # X2 (shared sigma_EL), X1 (shared sigma_AL)
python build_twin.py --pairs 400 --seed 11 --out data/e7_twin.jsonl             # E7 twin Q0 pairs (heads-masked AL prompts identical)
python compete.py static --instances data/m0_instances.jsonl --e7 data/e7_instances.jsonl --out results/compete_static.json
python prompt_box.py --out results/prompt_box.tex                             # appendix listing of verbatim prompts
python verify_data.py data/m0_instances.jsonl                                 # labels, balance, pairs, leading spaces
python probe_shortcuts.py data/m0_instances.jsonl --out results/shortcut_ceiling.json
python probe_shortcuts.py data/e2_controls.jsonl --out results/shortcut_ceiling_controls.json
python kmin_shares.py data/m0_instances.jsonl --out results/kmin_shares.json
cd "$ROOT/e3" && bash make_data.sh && python3 kdist.py --data data --out results/kdist.json   # about 1.5 h CPU
```

### Matched pairs
- **Global questions**: two different graphs with the same (|V|, |E|).
- **Query questions**: one graph, one order, two query vertices. The two prompts differ only in the query name, so every
  graph-level statistic is identical, including #red and the degree sum of the red vertices.
  - Depth one, counting and Q0 match deg v and the sum of deg u over u in N(v).
  - Depth two additionally matches the number of red neighbors of v.
- **E2 positive controls**: X2 (some vertex has degree >= 3): positive and negative share sigma_EL. X1 (some red vertex
  has a red neighbor): they share sigma_AL (degree-preserving edge swap).
- With names removed, the two prompts of a query pair are token-identical, so a name-blind model has balanced accuracy
  **exactly** 0.500. `run_llm.py` scores every distinct token sequence once and reuses the result (both prompts of a
  pair, any batch composition), and passes explicit position ids under left padding, so this holds bit for bit for the
  HF backend (unit test `m0/test_run_llm_hf.py`). A deviation means an implementation error.

### Shortcut ceilings of the instance files
`probe_shortcuts.py` reports the cross-validated accuracy of gradient-boosted trees on the probed feature classes (5
folds grouped by pair; not a bound on all shortcuts):

| Class | global | depth1 | depth2 | count | control (decidable by design) |
|---|---|---|---|---|---|
| Ceiling | 0.483 | 0.501 | 0.504 | 0.507 | 0.827 |

Acceptance: all four question classes <= 0.53. The result files record the SHA-256 of the instance files they were
computed on (`instances_sha256`); `kmin_shares.json` (k_min shares, no model needed) does the same. The k_min
distribution of the E3 test sets is computed directly from the generated data (`e3/kdist.py`; appendix table
T_kdist); `e3/results/kdist_measured.json` is its output for the `make_data.sh` settings, with the SHA-256 of the
generator code and of the three test files, and a rerun gives an identical `kdist.json`. The same holds for
`m0/results/token_cost_measured.json` (`token_cost.py`) and `e3/results/kset_shortcut.json`, `kid_shortcut.json`
(`shortcut_check.py`).

---

## 4. Commands per experiment

Model loops are written out; summaries over all models follow the loop. The batch size is given by `batch` (16 for
7-9B, 4 for 72B).

### 4.0 E0 (`pilot1/`)
```bash
cd "$ROOT/pilot1" && bash run_grid.sh 1 && bash run_grid.sh 2 && bash run_alibi.sh
python analyze.py results.jsonl --tex results_table.tex
```
The ALiBi test is preregistered in PREREGISTRATION.md: X1 under NC, name-blind, the pilot's model, data, seeds 0-2,
8,000 steps, learning rate 1e-3, batch size 32, with ALiBi as the only change; prediction: mean test accuracy
>= 0.95; falsified if the mean stays below 0.80.

### 4.1 E0b (`pilot1/`)
```bash
cd "$ROOT/pilot1" && bash run_e0b.sh
```
The grid is given in the script's comments. The chance cells are brought to 3 seeds with X2 4,000 steps and X1
and D1a 8,000 steps.

### 4.2 E3 and E3d (`e3/`)
The learning-rate sweep is the first, preregistered step (CONFIG §3.4): 1-local, AL, dia, seed 0, peak learning rates
{1e-4, 3e-4, 1e-3}, selected by the last-step balanced accuracy on the ID validation set (ties: the smaller value;
the test sets play no part). `run_e3.sh` and `run_e3d.sh` read the selected rate from `results/lr_choice.json`.
```bash
cd "$ROOT/e3" && bash run_lr_sweep.sh && bash run_e3.sh && bash run_e3d.sh && bash run_e3_extra.sh
python3 r9_e3.py --e3 results/e3.jsonl --e3d results/e3d.jsonl --lr results/lr_check.jsonl --nch results/e3_nch.jsonl \
    --kx results/kset_examples.jsonl --out results/summary_r9_e3.json
```
- 6 mechanisms: `nope`, `local`, `alibi_only`, `local_alibi`, `local_hardsince`, `rope`.
- Every run is evaluated once on the ID test set, the OOD test set and the k-controlled sets, and every 2,000 steps on
  2,000 ID validation examples.
- `run_e3d.sh` trains the 60 runs with L != 6 and then copies the 12 E3 runs of 1-local, dia, seeds 0-2 (L = 6) into
  `e3d.jsonl` with `export_l6.py` (marked `reused_from: e3`), so `e3d.jsonl` has 72 lines.
- The hard heads of `local_hardsince` use a straight-through gradient estimate; if a seed's training loss is still
  above 0.6 after 10k steps, see Section 6 and do not change hyperparameters. The attention scores of the two ALiBi
  mechanisms (bias included) are computed in fp32 (in bf16 the bias loses resolution beyond distance 512).
- Memory: OOD sequences reach about 1,700 tokens; `--eval-bs 32` is safe on 80 GB; on smaller GPUs lower `--eval-bs`
  and record it in DEVIATIONS.md.

### 4.3 E1 (every model) and summaries
```bash
cd "$ROOT/m0"
for M in $MODELS; do
  python run_llm.py $(B $M) --instances data/m0_instances.jsonl --out results/e1_$(basename $M).jsonl
done
python stats.py results/e1_*.jsonl --out results/summary_e1.json
# k_min strata: Table T_kstrata and Figure 4b use the positives of D1a+D1c; the E5 predictions use D1a alone
python e4_stats.py results/e1_*.jsonl --by model,ser --pids D1a,D1c --out results/summary_e1_strata.json
python e4_stats.py results/e1_*.jsonl --by model,ser --pids D1a --out results/summary_e1_strata_d1a.json
# Figure 1b: four-model means with a joint pair bootstrap (one resample for all models, stratified by question)
python joint_stats.py mean-ci results/e1_*.jsonl --by cls,ser --out results/summary_e1_meanci.json
```
The start-up output must show `name occurrence check: 0/...` and `token check: 0/200 ...`;
`results/name_filter.json` records the names each tokenizer drops.

### 4.4 E2
```bash
cd "$ROOT/m0"
for M in $MODELS; do
  python run_llm.py $(B $M) --mask all --instances data/m0_instances.jsonl --out results/e2_$(basename $M)_query.jsonl
  python run_llm.py $(B $M) --mask all --instances data/e2_controls.jsonl --out results/e2_$(basename $M)_controls.jsonl
done
python e4_stats.py results/e2_*_query.jsonl --by model,ser,cls --out results/summary_e2.json
python e4_stats.py results/e2_*_controls.jsonl --by model,ser,cls,pid --out results/summary_e2_controls.json   # X1/X2 by pid
python joint_stats.py blind results/e2_*.jsonl --out results/summary_e2_holm.json
```
`summary_e2_holm.json`: tests against 0.5 of the 12 positive-control cells that the theorem calls undecidable
(Holm); the 36 query-vertex cells are exactly 0.5 by construction and are only checked (`query_cells_exactly_half`).

### 4.5 E4
```bash
cd "$ROOT/m0"
for M in $MODELS; do
  # ECL on all twelve questions (all four classes)
  python run_llm.py $(B $M) --serializations ECL --instances data/m0_instances.jsonl --out results/e4_$(basename $M)_ecl.jsonl
  # depth-one ablations; EL, AL, NC come from E1 (same instances, names and order seeds)
  python run_llm.py $(B $M) --classes depth1 --serializations EL2,ELfar,NCH,NCnw,Incident,Friendship --instances data/m0_instances.jsonl \
      --out results/e4_$(basename $M).jsonl
done
python e4_stats.py results/e4_*.jsonl results/e1_*.jsonl --by model,ser --cls depth1 \
    --tests ECL:NC,ECL:EL,EL2:EL,ELfar:EL,ECL:ELfar,NCH:NC,NCH:AL,NCnw:NC,Incident:AL,Friendship:EL --out results/summary_e4.json
python e4_stats.py results/e4_*_ecl.jsonl --by model,ser,cls --out results/summary_e4_ecl.json
# k_min strata of the ablations and ECL: D1a+D1c positives (as E1), and D1a alone (for E5)
python e4_stats.py results/e4_*.jsonl --by model,ser --pids D1a,D1c --out results/summary_e4_strata.json
python e4_stats.py results/e4_*.jsonl --by model,ser --pids D1a --out results/summary_e4_strata_d1a.json
```
NCH and NC share the order seed; their prompts differ only in the line-head colors.

### 4.6 E7
```bash
cd "$ROOT/m0"
for M in $MODELS; do
  for MK in heads entries; do      # D1a: masks on the E1 instances; the unmasked condition is E1 itself
    python run_llm.py $(B $M) --mask $MK --serializations AL,NC --instances data/m0_instances.jsonl --pids D1a \
        --out results/e7_$(basename $M)_$MK.jsonl
    python run_llm.py $(B $M) --mask $MK --serializations AL,NC --instances data/e7_instances.jsonl --pids Q0 \
        --out results/e7q0_$(basename $M)_$MK.jsonl
  done
  for MK in rand_heads rand_entries; do      # as many random name occurrences as the targeted masks
    python run_llm.py $(B $M) --mask $MK --serializations AL,NC --instances data/m0_instances.jsonl --pids D1a \
        --out results/e7_$(basename $M)_$MK.jsonl
    python run_llm.py $(B $M) --mask $MK --serializations AL,NC --instances data/e7_instances.jsonl --pids Q0 \
        --out results/e7q0_$(basename $M)_$MK.jsonl
  done
  for MK in none heads entries; do           # twin Q0 pairs (AL); the heads-masked prompts of a pair coincide
    MA=$([ $MK = none ] || echo "--mask $MK")
    python run_llm.py $(B $M) $MA --serializations AL --instances data/e7_twin.jsonl --out results/e7twin_$(basename $M)_$MK.jsonl
  done
  # Q0 is not in E1: run its unmasked condition, and NC without head colors (NC-H) on Q0 (NC-H on D1a is E4)
  python run_llm.py $(B $M) --serializations AL,NC,NCH --instances data/e7_instances.jsonl --pids Q0 \
      --out results/e7q0_$(basename $M)_none.jsonl
done
python e4_stats.py results/e7_*.jsonl results/e7q0_*.jsonl results/e1_*.jsonl results/e4_*.jsonl --pids D1a,Q0 \
    --by model,ser,mask,pid --out results/summary_e7.json
# Figure 4a: four-model means, joint pair bootstrap (as Figure 1b), including NC-H on D1a (from E4)
python joint_stats.py mean-ci results/e7_*.jsonl results/e7q0_*.jsonl results/e1_*.jsonl results/e4_*.jsonl \
    --pids D1a,Q0 --by ser,mask,pid --out results/summary_e7_meanci.json
python e4_stats.py results/e7twin_*.jsonl --by model,ser,mask --out results/summary_e7_twin.json
```
`summary_e7.json` is computed after the model loop over all `e1_*.jsonl` (the unmasked D1a condition is E1) and also
carries the NC-H cells on D1a (E4) and Q0 (table T_partial). The mask word `kit` is on the name blacklist and must be
a stable single token itself (`mask_word` in `name_filter.json`).

### 4.7 E5 (Qwen2.5-7B, Llama-3.1-8B)
```bash
cd "$ROOT/m0"
python build_real.py --source molhiv --pairs 300 --seed 0 --out data/e5_molhiv.jsonl   # also writes .stats.json
python build_real.py --source arxiv  --pairs 300 --seed 0 --out data/e5_arxiv.jsonl
for M in $PAIR_MODELS; do
  for SRC in molhiv arxiv; do
    python run_llm.py $(B $M) --serializations EL,AL,NC,ECL,GraphML --instances data/e5_$SRC.jsonl \
        --out results/e5_$(basename $M)_$SRC.jsonl
  done
  python run_llm.py $(B $M) --classes depth1 --serializations GraphML --instances data/m0_instances.jsonl \
      --out results/e5_$(basename $M)_graphml.jsonl
done
python e4_stats.py results/e5_*_molhiv.jsonl results/e5_*_arxiv.jsonl --by model,src,ser --out results/summary_e5.json
python e4_stats.py results/e5_*_graphml.jsonl results/e1_*.jsonl --by model,ser --cls depth1 --tests GraphML:EL \
    --out results/summary_e5_graphml.json
# GraphML k_min strata on the same question sets as the other formats: D1a+D1c (V-C10) and D1a alone (E5)
python e4_stats.py results/e5_*_graphml.jsonl --by model,ser --pids D1a,D1c --out results/summary_e5_graphml_strata.json
python e4_stats.py results/e5_*_graphml.jsonl --by model,ser --pids D1a --out results/summary_e5_graphml_strata_d1a.json
```
- `build_real.py` writes the field `src` (molhiv, arxiv) into the instances and `run_llm.py` carries it into every
  record, so both sources can share one summary.
- arxiv colors come from the true categories: cs.LG, cs.AI, cs.CV, cs.CL red, the rest blue.
- GraphML ids are written `id=" elk"` (a leading space inside the attribute value) to satisfy the token rule; this
  deviates on purpose from the NMTOKEN type of the GraphML schema.

### 4.8 E6 (Qwen2.5-7B, Qwen2.5-72B)
```bash
cd "$ROOT/m0"
for M in $E6_MODELS; do
  python run_llm.py $(B $M) --classes depth1 --shots 3 --demo-instances data/demo_instances.jsonl \
      --out results/e6_$(basename $M)_fs.jsonl
  python run_llm.py --backend $BACKEND --model $M $NAMES_ARG --batch 8 --classes depth1 --cot \
      --out results/e6_$(basename $M)_cot.jsonl
done
# the zero-shot cells are the E1 condition itself (same instances, names and templates): E1 records, mode "zero"
python e4_stats.py results/e6_*.jsonl results/e1_Qwen2.5-7B-Instruct.jsonl results/e1_Qwen2.5-72B-Instruct.jsonl \
    --by model,mode,ser --cls depth1 --out results/summary_e6.json
```
- Every record carries `mode` (zero, few, cot). The zero-shot cells are the depth-one condition of E1 and are not rerun.
- The first label of the 3-shot demonstrations is decided by a hash of the test instance id, independent of the test
  label.
- CoT generates at most 512 tokens; the truncation rate (reasoning reaching 512 tokens) is the `truncated` field of the
  summary. Generation always uses HF `generate` (greedy).

### 4.9 E8 (Qwen2.5-7B, Llama-3.1-8B)
```bash
cd "$ROOT/m0"
for M in $PAIR_MODELS; do
  R="$(B $M) --classes global,depth1"
  python run_llm.py $R --ids int --variant int_ids --out results/e8_$(basename $M)_int_ids.jsonl
  python run_llm.py $R --sort-edges --variant sorted_el --out results/e8_$(basename $M)_sorted_el.jsonl
  python run_llm.py $R --question-first --variant question_first --out results/e8_$(basename $M)_question_first.jsonl
  python run_llm.py $R --instances data/e8_colors3.jsonl --variant three_colors --out results/e8_$(basename $M)_three_colors.jsonl
  python run_llm.py $R --generate --pids D1a,D1b,D1c --variant generate --out results/e8_$(basename $M)_generate.jsonl
  # generated answer = scored answer (generation_agreement)
  python stats.py results/e8_$(basename $M)_generate.jsonl --perm-reps 100 --out results/summary_e8_generate_$(basename $M).json
done
# variants by class; "default" is the E1 condition (variant tag "default" in the E1 records)
python e4_stats.py results/e8_*.jsonl results/e1_Qwen2.5-7B-Instruct.jsonl results/e1_Llama-3.1-8B-Instruct.jsonl \
    --by model,variant,ser,cls --out results/summary_e8.json
```
- One set of variant tags: `default`, `int_ids`, `sorted_el`, `question_first`, `three_colors`, `generate` (the keys of
  `analysis/schema.json`). Table T_robust reports the depth-one cells; the global cells are in the summary only.
- Sorting changes EL only; AL and NC prompts under `sorted_el` equal the default.
- Integer ids do not satisfy the single-token rule (Qwen and Gemma split digits); this is what the variant tests, so
  the token check is skipped under `--ids int`.
- The per-template results come from the `per_template` field of `summary_e1.json`.

### 4.10 E9 (Qwen2.5-7B)
```bash
cd "$ROOT/m0"
python e9_mechanism.py --model $E9_MODEL $NAMES_ARG --out-dir results   # 400 graphs (D1a pairs, one prompt per pair)
```

### 4.11 E10 (API)
```bash
cd "$ROOT/m0"      # needs OPENAI_API_KEY for the openai backend
python run_llm.py --backend $API_BACKEND --model gpt-4o-2024-08-06 --serializations EL,AL,NC,ECL \
    --instances data/e10_instances.jsonl --out results/e10_gpt4o.jsonl
python e4_stats.py results/e10_gpt4o.jsonl --by model,ser,cls --out results/summary_e10.json
```
- Rate limits, connection errors, timeouts and 5xx errors are retried with exponential backoff (from 2 s, doubling, at
  most 60 s, with jitter, at most 8 times, `run_llm.with_retry`); other errors (bad request, authentication) stop the
  run. The API backend writes a record after every prompt, so rerunning with the same `--out` continues after the last
  record without repeating finished prompts (unit test `m0/test_run_llm_api.py`).
- One call per prompt: temperature 0, `max_tokens` 1, `seed` 0, no system message, top-20 `logprobs`. The Yes score is
  the log of the summed probability of "Yes", " Yes", "yes", " yes" among the top 20, No likewise; if neither occurs,
  the generated token is parsed (`fallback`, its rate is the `fallback` field of the summary); an unparsable answer
  counts as wrong.
- Names are checked to be single tokens under tiktoken o200k_base.
- `seed=0` does not guarantee full determinism: every record stores the call date (`called`) and the
  `system_fingerprint`.

### 4.12 Additional analyses (read existing records only; `dmin_tokens.py` loads tokenizers only)
```bash
cd "$ROOT/m0"
python r9_stats.py paired   results/e1_*.jsonl results/e4_*_ecl.jsonl --out results/summary_paired.json
python r9_stats.py position results/e1_*.jsonl --out results/summary_position.json
python r9_stats.py tost     --results results --out results/summary_tost.json
python r9_stats.py lengths  results/e1_*.jsonl results/e4_*.jsonl results/e5_*_graphml.jsonl --out results/summary_lengths.json
python r9_stats.py qre      results/e1_*.jsonl --out results/summary_qre.json
python r9_stats.py kdiff    results/e1_*.jsonl results/e4_*_ecl.jsonl --out results/summary_kdiff.json
python dmin_tokens.py --tokenizers ${DMIN_TOKENIZERS:-${MODELS// /,}} --labels Qwen2.5-7B,Llama-3.1-8B,Gemma-2-9B,Qwen2.5-72B \
    $NAMES_ARG --instances data/m0_instances.jsonl --out results/dmin_tokens.json
python compete.py fit --static results/compete_static.json --e4 results/summary_e4.json --e7 results/summary_e7.json \
    --out results/summary_compete.json
```

### 4.13 E11a, E11b, E11c, the reasoning model, E12, E13; E3 hard pairs and the Delta <= 256 set
The E3 part needs no new training: `make_data.sh` also generates `dia_AL_hard.jsonl` and `dia_NC_hard.jsonl`
(`hard_pairs.py`, the hard pairs of Theorem lb:main, 1,000 pairs each, seed 8) and runs `shortcut_check.py` (the
line-index shortcut on the k-controlled sets, at most 0.55; the outputs equal the released `results/kset_shortcut.json`,
`kid_shortcut.json`); `train_e3.py` records `acc_hard` when these files exist, `r9_e3.py` collects them in the block
`hard`, imported with `--r9-e3` (rule C26). `make_data.sh` must therefore run before training.
Data (the instance files are released; regenerating gives the same bytes) and the token cost per format (no model):
```bash
cd "$ROOT/m0"
python build_r12.py e11b --out data/e11b_instances.jsonl     # 4 degrees x 2 questions x 150 pairs, prompts < 4,096 tokens
python build_r12.py e12  --out data/e12_probe.jsonl          # 2 formats x 9 positions x 60 pairs
python build_r12.py e13  --out data/e13_graphqa.jsonl        # 3 tasks x 200 (edge existence: 200 matched pairs)
python token_cost.py --instances data/m0_instances.jsonl --out results/token_cost.json   # = results/token_cost_measured.json
```
E11a (Qwen2.5-7B, about 6.6 GPU hours: 19 conditions plus 6 pooled-mean sensitivity conditions, 25 x 4,800 prompts,
plus about 0.4 h of scoring and calibration):
```bash
cd "$ROOT/m0"
python e11_heads.py score  --model $E9_MODEL $NAMES_ARG --out results/e11_head_scores.json
python e11_heads.py ablate --model $E9_MODEL $NAMES_ARG --scores results/e11_head_scores.json --out results/e11a_qwen7b.jsonl
python e11_heads.py summary results/e11a_qwen7b.jsonl --out results/summary_e11a.json
```
E11b (four models, about 29.5 GPU hours; prompts about twice as long as E1), E11c (exploratory: BLOOM-7B1, MPT-7B
and Qwen2.5-7B on the same name pool, about 2.9 GPU hours) and the reasoning model (DeepSeek-R1-Distill-Qwen-7B,
outside the theory, about 24.4 GPU hours):
```bash
cd "$ROOT/m0"
for M in $MODELS; do
  python run_llm.py $(B $M) --serializations EL,AL,NC,ECL --instances data/e11b_instances.jsonl \
      --out results/e11b_$(basename $M).jsonl
done
python e4_stats.py results/e11b_*.jsonl --by model,ser,maxdeg --out results/summary_e11b.json
E11C_NAMES=${NAMES:+$NAMES,bigscience/bloom-7b1,mosaicml/mpt-7b}
for M in ${E11C_MODELS:-bigscience/bloom-7b1 mosaicml/mpt-7b Qwen/Qwen2.5-7B-Instruct}; do
  python run_llm.py --backend $BACKEND --model $M ${E11C_NAMES:+--names-from $E11C_NAMES} --batch 16 \
      --serializations EL,AL,NC,ECL --pids D1a,D1c --instances data/m0_instances.jsonl --out results/e11c_$(basename $M).jsonl
done
python e4_stats.py results/e11c_*.jsonl --by model,ser --out results/summary_e11c.json
python run_llm.py --backend $BACKEND --model ${REASONING_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B} $NAMES_ARG --batch 4 \
    --cot --max-new 4096 --serializations EL,AL,NC,ECL --pids D1a --templates T1 --instances data/m0_instances.jsonl \
    --out results/reasoning_r1distill.jsonl
python e4_stats.py results/reasoning_r1distill.jsonl --by model,ser --out results/summary_reasoning.json
```
E12 (four models, about 1.1 GPU hours) and E13 (four models, about 7.4 GPU hours):
```bash
cd "$ROOT/m0"
for M in $MODELS; do
  python e12_probe.py run --backend $BACKEND --model $M --batch $(batch $M) --probe data/e12_probe.jsonl \
      --out results/e12_$(basename $M).jsonl
  python e13_graphqa.py run --backend $BACKEND --model $M $NAMES_ARG --batch $(batch $M) --instances data/e13_graphqa.jsonl \
      --out results/e13_$(basename $M).jsonl
done
python e12_probe.py summary results/e12_*.jsonl --out results/summary_e12.json
python e12_probe.py predict --probe-summary results/summary_e12.json --strata results/summary_e1_strata.json \
    --instances data/m0_instances.jsonl --out results/summary_e12_predict.json
python e13_graphqa.py summary results/e13_*.jsonl --out results/summary_e13.json
```
- E11a controls: 5 random head sets per h with the per-layer counts of the top h (excluding the top h); mean ablation
  uses 64 calibration prompts (16 per format, from the last part of the E1 depth-one instances), with means per format
  and token role (the pooled mean is the sensitivity check pooled-*); the test uses the first 300 D1a and Q0 pairs and
  the scoring the last 100 D1a pairs; the three sets do not overlap.
- E11b pairs are in one graph: two hubs of degree Delta (one red, one blue); the positive is a neighbor of the red hub,
  the negative a neighbor of the blue hub, with equal degree and neighbor degree sum (the E1 pairing); the positive's
  position in the red hub's list is k_min >= max(3, Delta/2). `run_llm.py` stops if a Gemma-2 prompt reaches 4,096
  tokens.
- An E12 pair is one prompt with the line-head color of the query's line changed from red to blue; `predict` fits no
  parameter; its error is reported without a threshold other than rule D15.
- E13 edge existence is scored by the Yes/No log-probabilities as E1; degree and neighbors generate at most 48 tokens,
  take the first integer or the set of names in the answer, and compare exactly with the truth; the two kinds are
  reported separately.
- The reasoning model gets the E1 prompts, generates at most 4,096 reasoning tokens, then the answer prompt, scored as
  E6; outside the theory, no prediction.
- The E11c name pool is the intersection of the four E1 tokenizers with those of BLOOM and MPT, so its prompts differ
  from E1; Qwen2.5-7B is rerun on the same pool as the control.

### 4.14 E11a replication and activation patching, name matching inside E3; E3 hard pairs with the proof's parameters
E3 part: `make_data.sh` also generates `dia_{AL,NC}_hardp{4,6,8}.jsonl` (`hard_pairs.py --proof-L`, the proof's
parameters of Theorem lb:main, 500 pairs each, seed 9); `train_e3.py` evaluates `hardp{L}` for models of depth L
(main grid L = 6, depth sweep L = 4, 8) and records `acc_hardp`; `r9_e3.py` collects the block `hardp` (rule C27). The
dia runs of the main grid save their models to `e3/ckpt/` (`--save-dir`, about 20 MB each, 120 in total). L = 12 would
need about 17,500 tokens and is not generated.
Activation patching on Qwen2.5-7B (about 4.2 GPU hours: 16 conditions x 2,400 prompts, two forward passes each) and the
replication on Llama-3.1-8B (scoring, mean ablation and patching, about 9.6 GPU hours):
```bash
cd "$ROOT/m0"
python e11_heads.py patch  --model $E9_MODEL $NAMES_ARG --scores results/e11_head_scores.json --out results/e11a_patch_qwen7b.jsonl
python e11_heads.py summary results/e11a_qwen7b.jsonl results/e11a_patch_qwen7b.jsonl --out results/summary_e11a.json
python e11_heads.py score  --model $E11_REPL $NAMES_ARG --out results/e11_head_scores_llama.json
python e11_heads.py ablate --model $E11_REPL $NAMES_ARG --scores results/e11_head_scores_llama.json --out results/e11a_llama8b.jsonl
python e11_heads.py patch  --model $E11_REPL $NAMES_ARG --scores results/e11_head_scores_llama.json --out results/e11a_patch_llama8b.jsonl
python e11_heads.py summary results/e11a_llama8b.jsonl results/e11a_patch_llama8b.jsonl --out results/summary_e11a_llama.json
```
Name matching inside the E3 models (the six-layer models of the main grid, dia, four formats, six arms, 5 seeds; 500
prompts per model: 125 per name-count bin, 1-16 from the ID test set, the rest from the OOD test set; about 0.5 h on one
GPU, or CPU):
```bash
cd "$ROOT/e3"
rm -f results/match_heads.jsonl
for c in ckpt/dia_*_L6_s*.pt; do python3 match_heads.py --ckpt $c --data data --n 500 --out results/match_heads.jsonl; done
python3 match_heads.py summary results/match_heads.jsonl --out "$ROOT/m0/results/summary_match.json"
```
- Patching: the patched heads' outputs (their slices of the o_proj input) are recorded on a corrupted prompt and written
  into the clean forward pass; two corruptions: all line-head colors swapped red/blue (recolor), all line-head names
  replaced by pool names not in the prompt (rename), entries unchanged; pairs whose token sequences differ in length
  are skipped (counted in `*_timing.json`). Random control: 16 random heads with the layer distribution of the top 16,
  5 sets. AL and NC only (EL and ECL have no line heads).
- Name matching: for every head and every name position whose name occurred earlier, whether the head's largest
  attention weight falls on an earlier occurrence of that name (M_{0,0}); reported are the error of the head with the
  best overall accuracy per name count (1-16, 17-32, 33-48, 49-64) and the number of heads above 0.9.

### 4.15 E11a on Gemma-2-9B, E11b-L, E14, E15, E16 (CONFIG §6.10)
Data (the instance files are released; regenerating gives the same bytes):
```bash
cd "$ROOT/m0"
python build_r14.py e11bL --e11b data/e11b_instances.jsonl --out data/e11bL_instances.jsonl   # E11b pairs padded per format
python build_r14.py e14 --e1 data/m0_instances.jsonl --out data/e14_instances.jsonl           # 300 D1a pairs x 7 filler conditions
```
E11a on Gemma-2-9B (scoring and mean ablation, no patching, about 5.9 GPU hours), E11b-L (four models, about 38.3 GPU
hours; every prompt as long as the E11b prompt at Delta = 48), E14 (three 7-9B models, T1, about 2.1 GPU hours) and E16
(exploratory, Qwen2.5-7B and Llama-3.1-8B, about 5.2 GPU hours):
```bash
cd "$ROOT/m0"
python e11_heads.py score  --model $E11_REPL2 $NAMES_ARG --out results/e11_head_scores_gemma.json
python e11_heads.py ablate --model $E11_REPL2 $NAMES_ARG --scores results/e11_head_scores_gemma.json --out results/e11a_gemma9b.jsonl
python e11_heads.py summary results/e11a_gemma9b.jsonl --out results/summary_e11a_gemma.json
for M in $MODELS; do
  python run_llm.py $(B $M) --serializations EL,AL,NC,ECL --instances data/e11bL_instances.jsonl \
      --out results/e11bL_$(basename $M).jsonl
done
python e4_stats.py results/e11bL_*.jsonl --by model,ser,maxdeg --out results/summary_e11bL.json
for M in $E14_MODELS; do
  python run_llm.py $(B $M) --serializations EL,ECL --templates T1 --instances data/e14_instances.jsonl \
      --out results/e14_$(basename $M).jsonl
done
python e4_stats.py results/e14_*.jsonl --by model,ser,filler_cell --out results/summary_e14.json
for M in $PAIR_MODELS; do for I in compound num4; do
  python run_llm.py $(B $M) --serializations EL,AL,NC,ECL --pids D1a,D1c --ids $I --variant ids_$I \
      --instances data/m0_instances.jsonl --out results/e16_$(basename $M)_$I.jsonl
done; done
python e4_stats.py results/e16_*.jsonl --by model,ids,ser --out results/summary_e16.json
```
E15 (e3, 48 runs of about 0.7 GPU hours; needs the data and learning-rate choice of Section 4.2; the 500k column comes
from the E3 main grid):
```bash
cd "$ROOT/e3"
E3_JOBS=${E3_JOBS:-8} bash run_e15.sh
python3 e15_summary.py results/e15.jsonl results/e3.jsonl --out "$ROOT/m0/results/summary_e15.json"
```
- E11b-L: every pair is padded per format with isolated vertices (fields ser_only, pad), so a record generates a
  prompt in one format only; Delta = 48 is not padded and its prompts equal E11b (rule R9 checks that the cells
  agree).
- E14: both placements share graphs, names and orders; the neutral lines contain no names, color words or answer
  words. Rule C33 compares between and before at F = 1024.
- E16: `--ids compound` joins two pool names by a hyphen, `--ids num4` uses four-digit integers, different per graph;
  the name token check does not apply (multi-token names); the single-token control is E1 itself. No preregistered
  prediction.
- E15: `train_e3.py --train-size N` takes the first N examples of the training file, with the same number of steps; the
  records carry train_size (the nominal N) and n_train_used (the examples actually used).

### 4.16 E17 dose of the since-head ablation (CONFIG §6.11)
Run after the E11a runs of Sections 4.13 to 4.15: the same score files, calibration and prompts, only more heads
ablated (h = 32, 64, 128; per h one top-h set and 5 random sets in the same layers; the unablated and top-16 cells are
the E11a cells). About 14.8 GPU hours for the three models.
```bash
cd "$ROOT/m0"
python e11_heads.py ablate --model $E9_MODEL $NAMES_ARG --scores results/e11_head_scores.json \
    --hs 32,64,128 --no-base --out results/e17_qwen7b.jsonl
python e11_heads.py ablate --model $E11_REPL $NAMES_ARG --scores results/e11_head_scores_llama.json \
    --hs 32,64,128 --no-base --out results/e17_llama8b.jsonl
python e11_heads.py ablate --model $E11_REPL2 $NAMES_ARG --scores results/e11_head_scores_gemma.json \
    --hs 32,64,128 --no-base --out results/e17_gemma9b.jsonl
python e11_heads.py summary results/e11a_qwen7b.jsonl results/e11a_llama8b.jsonl results/e11a_gemma9b.jsonl \
    results/e17_*.jsonl --out results/summary_e17.json
python test_r16.py
```
- The summary combines the E11a and E17 records: per model, none and top16 come from E11a; the flip contrasts (field
  flip: EL minus AL, EL minus NC, EL minus their mean, per pair, one-sided 95% lower bound lo05) are also written for
  the top h of E11a; the import takes h = 128.
- Criteria C35-C37 and the readings of both outcomes: CONFIG §6.11 and the paper's appendix table tab:t-prereg-r16.

### 4.17 E18 controls of the ablation, stratified test of P4 (CONFIG §6.12)
Run after E17: the same calibration, role-wise means and prompts (GraphQA degree uses the E13 questions). About 16.4
GPU hours for the three models.
```bash
cd "$ROOT/m0"
for M in "$E9_MODEL qwen7b" "$E11_REPL llama8b" "$E11_REPL2 gemma9b"; do set -- $M
  python e11_heads.py score-edge --model $1 $NAMES_ARG --out results/e18_edge_scores_$2.json
  python e11_heads.py ablate --model $1 $NAMES_ARG --scores results/e18_edge_scores_$2.json \
      --prefix e --hs 16,128 --no-base --out results/e18_edge_$2.jsonl
done
python e11_heads.py ablate --model $E9_MODEL $NAMES_ARG --scores results/e11_head_scores.json \
    --prefix n --positions entries --hs 16,128 --no-base --no-random --out results/e18_entry_qwen7b.jsonl
python e11_heads.py ablate --model $E11_REPL $NAMES_ARG --scores results/e11_head_scores_llama.json \
    --prefix n --positions entries --hs 16,128 --no-base --no-random --out results/e18_entry_llama8b.jsonl
python e11_heads.py ablate --model $E11_REPL2 $NAMES_ARG --scores results/e11_head_scores_gemma.json \
    --prefix n --positions entries --hs 16,128 --no-base --no-random --out results/e18_entry_gemma9b.jsonl
python e11_heads.py ablate --model $E9_MODEL $NAMES_ARG --scores results/e11_head_scores.json \
    --task degree --hs 128 --no-base --out results/e18_degree_qwen7b.jsonl
python e11_heads.py ablate --model $E11_REPL $NAMES_ARG --scores results/e11_head_scores_llama.json \
    --task degree --hs 128 --no-base --out results/e18_degree_llama8b.jsonl
python e11_heads.py ablate --model $E11_REPL2 $NAMES_ARG --scores results/e11_head_scores_gemma.json \
    --task degree --hs 128 --no-base --out results/e18_degree_gemma9b.jsonl
python e11_heads.py summary results/e11a_qwen7b.jsonl results/e11a_llama8b.jsonl results/e11a_gemma9b.jsonl \
    results/e17_*.jsonl results/e18_*.jsonl --out results/summary_e18.json
python r9_stats.py p4ni results/e1_*.jsonl results/e4_*_ecl.jsonl --out results/summary_p4ni.json
python test_r18.py
```
- Summary fields: edge (etop<h>), edge_random (erand<h>, mean/min/max of the 5 sets), entry and entry_flip (ntop<h>;
  records carry a kind field, only kind = top is used), degree and degree_random; times in `e18_*_timing.json`, which
  the import turns into the row "E18 ablation controls" of T_compute.
- Import: `--r18 $R` (reads summary_e18.json; summary_p4ni.json is read by `--r9 $R`). Criteria C38-C42 and their
  readings: CONFIG §6.12 and table tab:t-prereg-r18.

### 4.18 E19 knockout and entry patching, E20 within-line filler, E21 relational tables (CONFIG §6.13)
Run after Section 4.17 (entry patching uses the E11a score files; the E21 instance file is released as
`m0/data/e21_instances.jsonl` and `r19.py tables-build` rebuilds it byte for byte). The knockout uses a custom 4D
attention mask and needs transformers >= 4.43 with eager attention. About 7.3 GPU hours.
```bash
cd "$ROOT/m0"
for M in "$E9_MODEL qwen7b results/e11_head_scores.json" "$E11_REPL llama8b results/e11_head_scores_llama.json" \
         "$E11_REPL2 gemma9b results/e11_head_scores_gemma.json"; do set -- $M
  python r19.py knockout --model $1 $NAMES_ARG --out results/e19_ko_$2.jsonl
  python r19.py patch    --model $1 $NAMES_ARG --scores $3 --out results/e19_patch_$2.jsonl
  python r19.py filler   --model $1 $NAMES_ARG --out results/e20_$2.jsonl
  python r19.py tables-run --model $1 $NAMES_ARG --out results/e21_$2.jsonl
done
python r19.py tables-run --model Qwen/Qwen2.5-72B-Instruct $NAMES_ARG --out results/e21_qwen72b.jsonl
python r19.py summary results/e19_*.jsonl results/e20_*.jsonl results/e21_*.jsonl --out results/summary_r19.json
python test_r19.py
```
- Summary: summary_r19.json holds cells (balanced accuracy and 95% pair-bootstrap interval per model, variant, format
  and question) and diffs (ko-own minus ko-other, inside minus outside: paired differences with one-sided 95% upper
  bounds); times in `e19_*`, `e20_*`, `e21_*_timing.json`.
- Import: `--r19 $R`. Criteria C43-C49 and their readings: CONFIG §6.13 and table tab:t-prereg-r19.

### 4.19 Sixth preregistered set E22-E33 (CONFIG §6.14)
Run after Section 4.18. The controlled experiments use `e3/r20_e3.sh` (one mode per line; `data` first generates their
data and needs the `make_data.sh` of Section 4.2), the LLM experiments `m0/r20.py`, `m0/lora_r20.py` and `run_llm.py`
(variant tags in `--variant`). E24 needs peft==0.13.2. About 314 GPU hours and 14,400 API calls.
```bash
cd "$ROOT/e3"
bash r20_e3.sh data
for MODE in pos-control data-limited window nest nest-dist depth2 robust; do bash r20_e3.sh $MODE; done   # nest-dist: E33a
python3 test_gen_r20.py && python3 test_train_e3.py
python3 r20_summary.py --results results --e3d results/e3d.jsonl --out ../m0/results/summary_r20_e3.json
cd "$ROOT/m0"
python lora_r20.py build --out data/e24_train.jsonl
python lora_r20.py build-test --out data/e24_large.jsonl
for M in $R20_BASE; do for F in EL AL NC ECL ELB; do for N in 1000 4000 16000; do for S in 0 1 2; do   # ELB: E33b
  python lora_r20.py run --model $M --ser $F --n $N --seed $S --out results/e24.jsonl
done; done; done; done
python r20.py build-large --out data/e31d_instances.jsonl
python r20.py build-algo  --out data/e31h_instances.jsonl
python r20.py build-latek --out data/e31j_instances.jsonl
for M in "$E9_MODEL qwen7b results/e11_head_scores.json" "$E11_REPL llama8b results/e11_head_scores_llama.json" \
         "$E11_REPL2 gemma9b results/e11_head_scores_gemma.json"; do set -- $M
  python r20.py replicate --model $1 $NAMES_ARG --scores $3 --out results/e28a_$2.jsonl
  python r20.py stability --model $1 $NAMES_ARG --out results/e28b_$2.json
  python r20.py restore   --model $1 $NAMES_ARG --scores $3 --out results/e28d_$2.jsonl
  python r20.py routes-degree  --model $1 $NAMES_ARG --out results/e29_$2.jsonl
  python r20.py filler-between --model $1 $NAMES_ARG --out results/e30_$2.jsonl
done
python r20.py knockout-big --model $R20_BIG $NAMES_ARG --out results/e28c_qwen72b.jsonl
for M in $MODELS; do T=$(basename $M)
  python run_llm.py --backend $BACKEND --model $M $NAMES_ARG --batch 8 --cot --templates T1 --serializations EL,AL,NC,ECL,NCH \
      --instances data/m0_instances.jsonl --variant cot --out results/e31a_$T.jsonl
  for V in num4 compound; do
    python run_llm.py $(B $M) --ids $V --templates T1 --serializations EL,AL,NC,ECL --variant $V --out results/e31c_${T}_$V.jsonl
  done
  python run_llm.py $(B $M) --templates T1 --serializations EL,AL,NC,ECL --instances data/e31d_instances.jsonl --out results/e31d_$T.jsonl
  for SRC in molhiv arxiv; do N=$([ $SRC = molhiv ] && echo molecules || echo citation)
    python run_llm.py $(B $M) --templates T1 --serializations EL,AL,NC,ECL --pids D1a,D2a --instances data/e5_$SRC.jsonl \
        --out results/e31i_${N}_$T.jsonl
  done
  python run_llm.py $(B $M) --templates T1 --serializations EL,AL,NC,ECL --instances data/e31j_instances.jsonl --out results/e31j_$T.jsonl
  python run_llm.py $(B $M) --serializations ELB --classes global,depth1 --out results/e33c_$T.jsonl   # E33c: E1 protocol, T1-T3
done
python run_llm.py --backend $BACKEND --model $R20_R1 $NAMES_ARG --batch 4 --cot --max-new 4096 --templates T1 \
    --serializations EL,AL,NC,ECL,NCH --instances data/m0_instances.jsonl --variant r1 --out results/e31a_r1.jsonl
python run_llm.py --backend $API_BACKEND --model gpt-4o-2024-08-06 --templates T1 --serializations EL,AL,NC,ECL,NCH \
    --classes depth2,count --instances data/m0_instances.jsonl --out results/e31b_gpt4o.jsonl
python run_llm.py --backend $API_BACKEND --model gpt-4o-2024-08-06 --templates T1 --serializations NCH \
    --classes global,depth1 --instances data/m0_instances.jsonl --out results/e31b_gpt4o_nch.jsonl
for M in $R20_HF4; do python r20.py algo --model $M $NAMES_ARG --out results/e31h_$(basename $M).jsonl; done
for M in $R20_HF2; do python r20.py prompting --model $M $NAMES_ARG --out results/e31e_$(basename $M)_sys.jsonl; done
for M in $PAIR_MODELS; do T=$(basename $M)
  for K in 1 5 10; do
    python run_llm.py $(B $M) --shots $K --demo-instances data/demo_instances.jsonl --templates T1 --pids D1a,G1 \
        --serializations EL,AL,NC,ECL,NCH --variant shots$K --out results/e31e_${T}_shots$K.jsonl
  done
  python run_llm.py $(B $M) --sort-edges --templates T1 --pids D1a,G1 --serializations EL --variant el-sorted \
      --out results/e31e_${T}_sorted.jsonl
done
for M in $R20_SCALE; do S=$(basename $M | sed 's/Qwen2.5-//; s/-Instruct//')
  python run_llm.py $(B $M) --templates T1 --classes global,depth1 --serializations EL,AL,NC,ECL --variant "$S|instruct" \
      --out results/e31f_${S}_instruct.jsonl
done
for M in $R20_SCALE_BASE; do S=$(basename $M | sed 's/Qwen2.5-//')
  python run_llm.py $(B $M) --templates T1 --classes global,depth1 --serializations EL,AL,NC,ECL --variant "$S|base" \
      --out results/e31f_${S}_base.jsonl
done
for M in $R20_ALIBI Qwen/Qwen2.5-7B-Instruct; do
  python run_llm.py $(B $M) --names-from ${NAMES:-Qwen/Qwen2.5-7B-Instruct},bigscience/bloom-7b1,mosaicml/mpt-7b --pids D1a \
      --templates T1 --serializations EL,AL,NC,ECL --out results/e31g_$(basename $M).jsonl
done
python r20.py summary results/e28a_*.jsonl results/e28d_*.jsonl results/e29_*.jsonl results/e30_*.jsonl results/e31e_*.jsonl \
    results/e31h_*.jsonl --stability results/e28b_*.json --out results/summary_r20.json
python r20_blocks.py --out results/blocks_r20.json
python test_r20.py
```
- Every E23 run is evaluated every 250 steps on 2,000 ID validation pairs (field `curve`); N95 and the steps to reach 0.95
  are computed by `r20_summary.py` and `r20_blocks.py` with the same interpolation (as E15).
- `blocks_r20.json` is the import file, in the structure of the blocks e22-e33 of `analysis/schema.json`, with the
  cells of the result files that exist; cells that restate other blocks (the T1 losses of E11a, the AL cells of
  E11b-L, the outside cells of E20, the GPT-4o cells of E10, the k_min-share predictions, the k_min = 1 stratum of E1)
  are filled by the importer from the imported data.
- E33: `r20_e3.sh data` also generates `data/dia_ELB_{train,val,id,ood}.jsonl` (`gen_e3.py --ser ELB`, the E3 graphs and
  seeds); `nest-dist` writes `results/r20_nest-dist.jsonl`; the ELB runs of E33b share `results/e24.jsonl` with E24 and
  `r20_blocks.py` puts them into the e33 block (`lora`) by format; `results/e33c_<model>.jsonl` of E33c goes to `llm`
  of the e33 block. The EL, AL, NC, ECL cells come from E23, E24 and E1.
- Import: `--r20 $R/blocks_r20.json`. Criteria C50-C67 and their readings: CONFIG §6.14 and tables tab:t-prereg-r20,
  tab:t-prereg-r20b.

### 4.20 Seventh preregistered set E34-E36 (CONFIG §6.15)
Run in `m0/`, one call per model (TAG = qwen7b, llama8b, gemma9b; 72B on four GPUs). The since-head rankings are the
E11a score files (`results/e11_head_scores<_tag>.json`); 72B is ranked first.

```bash
python r21.py edge    --model $M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e34_<tag>.jsonl
python r21.py edge-ko --model $M $NAMES_ARG --out results/e34ko_<tag>.jsonl
python r21.py null    --model $M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e35_<tag>.jsonl
python r21.py tpl     --model $M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e36tpl_<tag>.jsonl
python r21.py d1c     --model $M $NAMES_ARG --scores results/e11_head_scores<_tag>.json --out results/e36d1c_<tag>.jsonl
python e11_heads.py score --model Qwen/Qwen2.5-72B-Instruct $NAMES_ARG --out results/e11_head_scores_qwen72b.json
python r21.py q72     --model Qwen/Qwen2.5-72B-Instruct $NAMES_ARG --scores results/e11_head_scores_qwen72b.json --out results/e36q72.jsonl
python r21.py summary results/e34_*.jsonl results/e34ko_*.jsonl results/e35_*.jsonl results/e36*.jsonl --out results/blocks_r21.json
```

Import: `--r21 $R/blocks_r21.json`. Criteria C68-C74 and their readings: CONFIG §6.15 and table tab:t-prereg-r21.

`null` also writes `e35_<tag>_sets.json` (the output norm of every head and the 100 head sets). Checks: the none cells of
E34 (when rerun with `--with-base`) equal the edge-existence cells of E13 bit for bit; every E35 set has exactly 128
heads and none of the top 128; the 72B ranking uses only 72B's own attention.

## 5. Result file formats

### 5.1 LLM records (`run_llm.py`, one JSON line per prompt)
```
id, pid, cls, pair, label, ser, tpl, n, maxdeg, deg_v, k_min, d_min, rel_pos, mask, ids, question_first, sort_edges,
masked, src, [filler_cell, pad], pred, lp_yes, lp_no, n_tokens, [reasoning, reasoning_tokens], [gen_text, gen_pred], [raw, fallback,
called, system_fingerprint], model, backend, variant, mode, sec
```
- `src`: source of E5 instances (molhiv, arxiv), null elsewhere. `mode`: zero, few (`--shots`), cot (`--cot`). `sec`:
  wall-clock seconds per prompt (for T_compute).
- `k_min`: the smallest position (from 1) of the query vertex in its red neighbors' lists, defined for depth-one
  positives (null for negatives). AL, NC, NCH use their own order; EL, ECL, GraphML and the other edge formats have no
  neighbor lists and take the k_min of the matching NC prompt (same instance, NC order seed, CONFIG §5.5).
- `d_min`: the word-level distance of that entry to its line-head color (AL 2k+1, NC 5k-1).
- `rel_pos`: the relative position of the needed line of a depth-one positive in the serialization (0 first, 1
  last); null for negatives, masked prompts and integer ids. `fallback` (E10): the generated token was parsed because
  no Yes/No variant was among the top 20.

### 5.2 `summary_e1.json` (`stats.py`, per model)
```
{model: {cells: [{cls, ser, bal_acc, ci_low, ci_high, n_pairs, n_prompts}],
         per_question: [{pid, ser, bal_acc}], per_template: [{tpl, cls, ser, bal_acc}],
         interaction / interaction_length_adjusted / interaction_qfe: {wald_chi2, df, p, p_holm_across_models},
         pairwise: [{cls, a, b, mean_diff, p, p_holm, n_pairs}],
         planned_contrasts: [{format, cls, delta, ci_low, ci_high, p_one_sided, p_holm}],
         distance_regression: {ser: {term: {coef, se, p}}}, [generation_agreement]},
 _pooled: {pseudo_r2_base, pseudo_r2_full, gain_interaction}}
```

### 5.3 `summary_e2/e4/e5/e6/e7/e8/e10.json` (`e4_stats.py`)
```
{cells: [{<by columns>, bal_acc, ci_low, ci_high, n_pairs, [truncated],
          strata: [{stratum ∈ k1,k2,k3-4,k5+,le2,3to7,ge8, bal_acc, ci_low, ci_high, n_pairs}]}],
 tests: [{a, b, mean_diff, p, p_holm, n_pairs, <分组列>}]}
```
`<by columns>` are the columns given by `--by` (model, ser, cls, src, mode, variant, mask, pid). `truncated` occurs
only in CoT cells. `summary_e2_holm.json` (`joint_stats.py blind`): `{controls: [{model, pid, ser, bal_acc, se, p,
p_holm, off_chance}], n_cells, n_off_chance, query_cells_exactly_half, n_query_cells}`; `summary_*_meanci.json`
(`joint_stats.py mean-ci`): `[{<by columns>, mean_bal_acc, ci_low, ci_high, n_models, n_pairs}]`.
`strata` exist for depth-one cells only; every format has k strata (edge formats use the k_min of the matching NC
prompt). The balanced accuracy of a k stratum is the mean of the TPR of the stratum's positives and the TNR of all
negatives of the condition (negatives have no k_min).

### 5.4 E3 / E3d (`train_e3.py`, one line per run)
```
{task, ser, pos, layers, seed, d, heads, steps, bs, lr, final_train_loss, sec,
 acc_id, acc_ood, acc_k: {"1","2","3","4-6","7-12","13-32": bal_acc},
 curve: [{step, train_loss, lr, val_bal_acc}], [reused_from], [train_size, n_train_used]}
```
The k-controlled sets exist for `dia` only, so `red` runs have no `acc_k`. If a configuration (mechanism, task, format,
layers, seed) occurs more than once (a rerun), the import uses the last line and warns. `results/lr_sweep.jsonl` has the
same format; `results/lr_choice.json`: `{lr, val_bal_acc: {lr: acc}, rule}`. `results/kdist.json`:
`{id: {stratum: share}, id_vfirst, ood: {...}, ood_vfirst, kc_vfirst, kc_vsecond}`.

### 5.5 E0 / E0b (`pilot1/train.py`)
```
{task, ser, arm, seed, alibi, n_test, d, layers, train_pairs, steps, acc, final_train_loss, mean_len, sec}
```
If a configuration (d, layers, train_pairs, steps, seed) occurs more than once, the import uses the last line and warns.

### 5.6 E9
| File | Format |
|---|---|
| `e9_probe_{AL,NC}.json` | `{layer: {"1-8", "9-24", ">24", "all": balanced accuracy, "n": {bin: entries}, "control", "selectivity"}}` (classes reweighted, chance 0.5; every entry with distance > 24 kept) |
| `e9_retrieve_{AL,NC}.json` | `{top1_by_layer: [29 values], layer, by_bin: {"1-8", "9-24", ">24": top-1}, n_entries}` (`by_bin` at layer `layer`, the layer with the best top-1 in the nearest bin) |
| `e9_patch_{AL,NC}.json` | `{restored_by_layer: [29 values, index i = 0..28: 0 the embedding output, i >= 1 the output of block i (as hidden_states[i])], n}` |
| `e9_el_inner.json` | `{acc_by_layer: {layer: acc}, n}` |
| `e9_timing.json` | `{sec, graphs}` |

### 5.7 E5 subgraph statistics
`data/e5_<src>.jsonl.stats.json`: `{src, n: [...], edges: [...], maxdeg: [...], meandeg: [...], cycles: [...], kmin_AL: [...],
kmin_NC: [...]}`, one value per pair (kmin of the positive; cycles = cycle rank m - n + c). The paper's k_min shares use the
NC order (as the LLM strata).

---

## 6. Troubleshooting

1. **Tokens**: `token check` is not 0, or fewer than 40 names remain in `name_filter.json`.
2. **Pairs**: all 4 checks of `verify_data.py` must pass.
3. **E3 plateau and curriculum**: before the matching circuit forms there is a plateau at loss about 0.69 and accuracy
   0.5; on small graphs it ends after about 2,000 steps. `train_e3.py` therefore uses the curriculum by default (the
   first 4,000 steps sample only training graphs with at most 10 vertices, `--curriculum-steps 4000
   --curriculum-max-n 10`), then all training graphs. In the learning-curve logs `red` typically leaves 0.5 between 4k
   and 8k steps, `dia` between 6k and 12k.
4. **E3 name embeddings**: training and evaluation must use the random frozen vectors of `batchify`; evaluation uses
   the fixed seed 777. If a mechanism is at chance even in distribution, check that no `@`-prefixed token entered the
   vocabulary.
5. **E3 mask**: under the strict mask only `<BOS>` sees itself; print the allowed matrix of `Model.bias` at T = 6.
6. **Hard since heads**: for all-zero scores the argmax must pick the rightmost visible position (unit test
   `e3/test_train_e3.py`).
7. **E2 query questions**: with names removed the positive and negative prompts are identical, so the result must be
   **exactly** 0.500; otherwise the mask misses the query name or the pairs are shuffled.
8. **LLM memory**: 72B with `--batch 4`; Gemma-2 stops with an error if a prompt reaches 4,096 tokens. The longest E1
   prompt (NC, 40 vertices) has about 2,000 word-level tokens, below 4,096 model tokens under all four tokenizers (the
   recorded `n_tokens` are authoritative).

---

## 7. Import, validation, tables and figures

1. Import every summary into `results/summary.json` (one option per summary file; the order is fixed, the strata last,
   so that the D1a+D1c strata are not overwritten). Every path is checked against the structure of
   `analysis/schema.json`; an unknown key (for example a misspelt variant tag) stops the import.
```bash
cd "$ROOT"
R=m0/results
python analysis/import_measured.py --e1 $R/summary_e1.json --e1-strata $R/summary_e1_strata.json \
    --e1-strata-d1a $R/summary_e1_strata_d1a.json --e1-meanci $R/summary_e1_meanci.json \
    --e2 $R/summary_e2.json --e2-controls $R/summary_e2_controls.json --e2-holm $R/summary_e2_holm.json \
    --e4 $R/summary_e4.json --e4-ecl $R/summary_e4_ecl.json \
    --e4-strata $R/summary_e4_strata.json --e4-strata-d1a $R/summary_e4_strata_d1a.json \
    --e7 $R/summary_e7.json --e7-meanci $R/summary_e7_meanci.json \
    --e5 $R/summary_e5.json --e5-stats m0/data/e5_molhiv.jsonl.stats.json m0/data/e5_arxiv.jsonl.stats.json \
    --graphml $R/summary_e5_graphml.json --graphml-strata $R/summary_e5_graphml_strata.json \
    --graphml-strata-d1a $R/summary_e5_graphml_strata_d1a.json \
    --e6 $R/summary_e6.json --e8 $R/summary_e8.json \
    --e8-generate $R/summary_e8_generate_Qwen2.5-7B-Instruct.json $R/summary_e8_generate_Llama-3.1-8B-Instruct.json \
    --e9 $R --e10 $R/summary_e10.json \
    --e7-twin $R/summary_e7_twin.json --r9 $R --r9-e3 e3/results/summary_r9_e3.json --r12 $R --r14 $R --r16 $R --r18 $R \
    --r19 $R --r20 $R/blocks_r20.json --r21 $R/blocks_r21.json \
    --kset-blocking e3/results/kset_blocking.json \
    --e3 e3/results/e3.jsonl --e3d e3/results/e3d.jsonl --kdist e3/results/kdist.json \
    --e0b pilot1/results_e0b.jsonl --e0b-chance pilot1/results_chance3.jsonl \
    --compute m0/results e3/results pilot1
```
   `summary_e1.json` brings the planned contrasts (normal p of z = Delta/se and the empirical bootstrap p), the 12
   permutation tests, the distance regressions, the three GEE interaction tests and the per-template results
   (T_robust). `--compute` computes T_compute from the `sec` fields of the run records.
2. Validate, export the tables and draw the figures:
```bash
cd "$ROOT"
python analysis/validate.py > validate_log.txt 2>&1 || true      # keep the log: failed predictions are findings
python analysis/tables.py
for f in analysis/figures/fig*.py; do (cd analysis/figures && python $(basename $f)); done
```
3. A failed rule is never a reason to change data. A failed exact rule (VALIDATION_RULES.md) points to a bug in a run,
   a summary script or the import; a failed directional prediction is a finding and is reported with the reading fixed
   in CONFIG §6; a warning is checked against the configuration, the seeds and the code, and reported as found.

## 8. Outputs to keep
1. `m0/results/`: every `.jsonl` and `summary_*.json`, `name_filter.json`, `rendered_prompts.txt`,
   `model_revisions.json`, `shortcut_ceiling.json`.
2. `e3/results/e3.jsonl` (240 lines), `e3d.jsonl` (60 new runs plus the 12 L = 6 lines exported from E3 seeds 0-2)
   and all logs.
3. `pilot1/results.jsonl`, `results_alibi.jsonl`, `results_e0b.jsonl` (72 lines), `results_chance3.jsonl`.
4. `env.txt`, `gpu.txt`, `DEVIATIONS.md`, `validate_log.txt`, `results/summary.json`.
