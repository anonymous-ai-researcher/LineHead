<div align="center">

# LineHead

### Which Graph Serialization Is Easiest Depends on What Attention Can Do

*Code, frozen instance files, protocol and preregistration of the paper*

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Transformers 4.46.3](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.46.3-FFD21E)](https://github.com/huggingface/transformers)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Instances: CC BY 4.0](https://img.shields.io/badge/instance%20files-CC%20BY%204.0-lightgrey)](#license)
[![Preregistered](https://img.shields.io/badge/protocol-preregistered-success)](PREREGISTRATION.md)

[Overview](#overview) •
[Repository map](#repository-map) •
[Installation](#installation) •
[Verify the release](#verify-the-release) •
[Reproduce](#reproducing-the-experiments) •
[From runs to paper](#from-runs-to-tables-and-figures) •
[Reproducibility](#reproducibility-details) •
[License](#license)

</div>

---

## Overview

Language models read graphs as text, and the community disagrees about which text format works best. This repository
accompanies a paper that answers the question with a **logic of attention**: it proves which attention operations
each serialization requires, and then tests these predictions in transformers trained from scratch and in pretrained
LLMs.

<table>
<tr>
<th width="33%">Theory</th>
<th width="33%">Controlled transformers</th>
<th width="33%">Pretrained LLMs</th>
</tr>
<tr valign="top">
<td>

A logic with three attention capabilities: name **matching** <code>M</code>, local **shifts** <code>Y</code> and
**since** <code>S</code> (attending to the latest position of a kind). Exact characterizations of what each format
conveys to a name-blind model, and optimal match counts with names.

</td>
<td>

Causal transformers trained from scratch with six positional mechanisms (NoPE, 1-local heads, ALiBi only,
1-local+ALiBi, 1-local+hard since, RoPE), four formats and five seeds, on in-distribution and larger test graphs.

</td>
<td>

Four open LLMs (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, Qwen2.5-72B) and GPT-4o: masking, degree sweeps, head ablation,
attention knockout, patching and probes that locate the operation each format needs.

</td>
</tr>
</table>

**The formats.** Edge list (`EL`), adjacency list (`AL`), node-centric (`NC`), and the theory-derived edge list with
inline endpoint colors (`ECL`).

**The central idea.** To read a neighbor's color, an edge list needs two *nested* name matches, while a list format
needs one match plus its **line head**, the vertex whose line it is. Local shifts cannot find the line head at high
degree, but since can. Which format is easiest therefore depends on what attention can do.

```
            capabilities                    what the list formats need          what EL needs
  ┌───────────────────────────┐        ┌──────────────────────────────┐   ┌───────────────────────┐
  │ M   name matching         │ ─────▶ │ one match + find line head   │   │ two nested matches    │
  │ Y   local shifts          │ ─────▶ │ line head reachable only at  │   │                       │
  │                           │        │ bounded degree               │   │                       │
  │ S   since                 │ ─────▶ │ line head at any degree      │   │                       │
  └───────────────────────────┘        └──────────────────────────────┘   └───────────────────────┘
```

---

## Repository map

```
LineHead/
├── m0/                  LLM experiments
│   ├── data/            frozen instance files (prompts are built from them)
│   ├── results/         instance-derived files (k_min shares, shortcut ceilings); run outputs are written here
│   ├── build*.py        instance builders            run_llm.py   scoring and generation (HF, OpenAI, mock)
│   ├── stats.py         E1 statistics                e4_stats.py  E2, E4 to E8, E10 statistics
│   └── e9_*, e11_*, r19.py, r20*.py, r21.py, lora_r20.py         mechanism experiments E9, E11 to E36
├── e3/                  controlled transformers: data generator, training, evaluation, test-set checks
├── pilot1/              pilot E0 and its scaling grid E0b
├── theory_checks/       machine checks of the appendix, with their outputs (results_*.txt)
├── analysis/            import, validation rules, table export, figure scripts
│   └── figures/         one script per data figure of the paper; _record.py lists every drawn value
├── results/             summary.json and tables/*.csv are written here by the analysis
├── tools/               compare_release.py (see RELEASE_NOTES.md)
├── CONFIG.md            hardware, software, hyperparameters, data distributions of every experiment
├── RUN_PROTOCOL.md      every command, in order, and the format of every output file
├── VALIDATION_RULES.md  rules applied to the results: theory constraints, predictions, consistency
├── PREREGISTRATION.md   what is frozen, how to verify it, where each prediction is stated
├── MANIFEST.sha256      SHA-256 of every frozen file
└── RELEASE_NOTES.md     file-by-file differences between this release and the files used for the runs
```

> **What is included.** The complete code, the frozen LLM instance files, the instance-derived files, the outputs of
> the theory checks, and the protocol documents. **Run outputs are not included**: every run writes its records and
> summaries into the `results/` folders when executed with the commands of `RUN_PROTOCOL.md`.

---

## Installation

```bash
git clone <this repository> LineHead && cd LineHead

python3.11 -m venv venv && source venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export PYTHONHASHSEED=0
```

<details>
<summary><b>Additional requirements</b></summary>

- **Gated models.** Accept the licenses of Llama 3.1 and Gemma 2 on Hugging Face and log in (`huggingface-cli login`)
  before the LLM runs.
- **GPT-4o (E10, E31).** Set `OPENAI_API_KEY`; the model version is pinned in `CONFIG.md`.
- **Figures.** The figure scripts render text with LaTeX: a TeX installation with `times`, `newtxmath` and
  `inconsolata` is needed.
- **Real topologies (E5).** `m0/build_real.py` downloads ogbg-molhiv and ogbn-arxiv through the `ogb` package.

</details>

---

## Verify the release

All of the following run on a CPU in minutes and need no model download.

```bash
sha256sum MANIFEST.sha256              # prints the value quoted in the paper
sha256sum -c MANIFEST.sha256           # every frozen file is unchanged

(cd m0 && python -m pytest -q)         # serializers, instance builders, scoring, statistics
(cd e3 && python -m pytest -q)         # data generator, training code, test-set checks
(cd theory_checks && bash run_all.sh)  # machine checks of the appendix; compare with results_*.txt
```

`PREREGISTRATION.md` also gives the one-line check that no unlisted file was added to the frozen set.

<details>
<summary><b>End-to-end smoke test without a GPU</b></summary>

With `E2E_TINY=1`, every script shrinks to a few pairs and runs on a CPU with the `mock` backend and a tiny random
local model (`m0/make_tiny_model.py`). This exercises the full pipeline, from instances to tables. The mock backend
answers at random and marks every record it writes as `mock`; the statistics label any summary built from such
records as a pipeline test, so these outputs serve only to test the code.

</details>

---

## Reproducing the experiments

`RUN_PROTOCOL.md` holds every command in the order in which it runs. Its Section 3 defines the shared shell variables,
Section 4 has one block per experiment, and Section 5 documents each output format. The steps below summarize it.

### 1  Data

```bash
cd m0
python build.py --pairs 400 --seed 0 --out data/m0_instances.jsonl     # E1: 12 questions x 400 matched pairs
python verify_data.py data/m0_instances.jsonl                          # labels, balance, pairs, token checks
cd ../e3 && bash make_data.sh                                          # E3 training and test sets, fixed seeds
```

The instance files in `m0/data/` are frozen; rebuilding them with the commands of `RUN_PROTOCOL.md` §3 reproduces them
byte for byte. Write rebuilt files to another path and compare with `cmp` rather than overwriting the frozen copies.

### 2  Controlled transformers (E3, E3d)

```bash
cd e3
bash run_lr_sweep.sh     # preregistered learning-rate sweep, selected on the validation set only
bash run_e3.sh           # 6 mechanisms x 2 tasks x 4 formats x 5 seeds
bash run_e3d.sh          # depth sweep of the 1-local arm
bash run_e3_extra.sh     # learning-rate check of NoPE and RoPE, NC-H arm
```

Every run is evaluated on the in-distribution test set, the larger out-of-distribution graphs and the
position-controlled sets, and writes one JSON line with its configuration, seed, accuracies and wall-clock time.

### 3  LLM experiments (E1 to E10)

```bash
# shell variables and helpers ($MODELS, $(B ...)): first block of RUN_PROTOCOL.md §3
cd "$ROOT/m0"
for M in $MODELS; do
  python run_llm.py $(B $M) --instances data/m0_instances.jsonl --out results/e1_$(basename $M).jsonl
done
python stats.py results/e1_*.jsonl --out results/summary_e1.json
```

Scoring compares the log-probabilities of "Yes" and "No" without sampling. Every distinct token sequence is scored once,
so the two prompts of a name-masked pair receive bit-identical scores. All models see the same prompts, because the
name list is the intersection over the four tokenizers.

### 4  Mechanism experiments and preregistered sets (E11 to E36)

| Experiments | Content | Protocol | Main scripts |
|---|---|---|---|
| E0, E0b | pilot, preregistered ALiBi test, scaling grid | §4.0–4.1 | `pilot1/run_grid.sh`, `run_alibi.sh`, `run_e0b.sh` |
| E3, E3d | positional mechanisms, depth sweep | §4.2 | `e3/run_e3.sh`, `run_e3d.sh` |
| E1, E2, E4, E7 | main grid, name-blind and positive controls, format ablations, partial masking | §4.3–4.6 | `m0/run_llm.py`, `stats.py`, `e4_stats.py` |
| E5 | real topologies and GraphML | §4.7 | `m0/build_real.py` |
| E6, E8, E10 | prompting, robustness, GPT-4o | §4.8, 4.9, 4.11 | `m0/run_llm.py` options |
| E9 | probes, retrieval and patching | §4.10 | `m0/e9_mechanism.py` |
| E11 to E16 | since-head ablation and replications, degree sweeps, since probe, GraphQA, filler, data efficiency | §4.13–4.15 | `m0/e11_heads.py`, `e12_probe.py`, `e13_graphqa.py` |
| E17 to E21 | ablation dose and controls, attention knockout, entry patching, within-line filler, relational tables | §4.16–4.18 | `m0/r19.py`, `e11_heads.py` |
| E22 to E33 | sixth preregistered set | §4.19 | `e3/r20_e3.sh`, `m0/r20.py`, `lora_r20.py`, `r20_blocks.py` |
| E34 to E36 | seventh preregistered set | §4.20 | `m0/r21.py` |

`CONFIG.md` §6 states the design and the preregistered predictions of each set. `RUN_PROTOCOL.md` §0 recommends an
order of the runs and gives the expected compute of each experiment.

---

## From runs to tables and figures

```
  run records (.jsonl)  ──▶  per-experiment summaries  ──▶  results/summary.json  ──┬──▶  validate.py   rules of VALIDATION_RULES.md
  m0/  e3/  pilot1/          stats.py, e4_stats.py, ...     import_measured.py       ├──▶  tables.py     results/tables/<block>.csv
                                                                                      └──▶  figures/      Figures 1 and 3 to 12
```

```bash
python analysis/import_measured.py ...                    # full argument list: RUN_PROTOCOL.md §7
python analysis/validate.py > validate_log.txt 2>&1 || true
python analysis/tables.py
for f in analysis/figures/fig*.py; do (cd analysis/figures && python $(basename $f)); done
```

- **Import.** Every path is checked against `analysis/schema.json`, so an unknown key, for example a misspelt variant
  tag, stops the import. `--compute` derives the compute table from the wall-clock fields of the run records.
- **Validation.** `validate.py` applies every rule of `VALIDATION_RULES.md`: exact theory constraints, the preregistered
  predictions with their falsification criteria, and cross-experiment consistency. A failed prediction is a finding
  and is reported as such; a failed exact rule points to a bug in a run, a summary or the import.
- **Rounding.** Every displayed number follows one rule (`analysis/fmt.py`): half up at the displayed precision.
- **Figures.** `analysis/figures/_record.py` runs any figure script and lists each value it draws and each number it
  prints, so a figure can be checked against `results/summary.json`.

| Paper figure | Script | Paper figure | Script |
|---|---|---|---|
| Figure 1 | `fig1.py` | Figure 7 | `figA_real.py` |
| Figure 2 | `fig2.py` (diagram, no data) | Figure 8 | `figA_mech.py` |
| Figure 3 | `fig3.py` | Figure 9 | `fig5.py` |
| Figure 4 | `fig4.py` | Figure 10 | `figA_ctl.py` |
| Figure 5 | `figA_pilot.py` | Figure 11 | `figA_r19.py` |
| Figure 6 | `figA_depth.py` | Figure 12 | `figA_r20.py` |

---

## Reproducibility details

| Aspect | Setting |
|---|---|
| Hardware | one node with 8x NVIDIA A100-SXM4 80GB (NVLink), 2x AMD EPYC 7763, 1 TB RAM; Qwen2.5-72B spread over four GPUs by layer; pilot, data generation, statistics and theory checks on CPU |
| Software | Python 3.11.9, CUDA 12.4, PyTorch 2.4.1, Transformers 4.46.3, Accelerate 1.1.1, PEFT 0.13.2, NetworkX 3.4.2, NumPy 1.26.4, SciPy 1.14.1, statsmodels 0.14.4, scikit-learn 1.5.2 (`requirements.txt`) |
| Models | weights pinned to a Hugging Face commit hash; `run_llm.py` logs the hash of the loaded weights in every run (`m0/results/model_revisions.json`) |
| Randomness | fixed seeds for every generator, split and training run (`CONFIG.md`); `PYTHONHASHSEED=0` |
| Decoding | log-probability scoring of Yes and No; greedy decoding wherever text is generated (CoT, generation checks) |
| Tokenization | every line of a serialization starts with a space; `run_llm.py` checks at start-up that each name maps to the same token id everywhere and exits otherwise |
| Hyperparameters | fixed in `CONFIG.md` §3.4; the only tuning step is the preregistered learning-rate sweep, selected on validation data |
| Deviations | any rerun of a crashed run, and its reason, is recorded in `DEVIATIONS.md` |
| Environment record | `pip freeze > env.txt` and `nvidia-smi > gpu.txt` before the runs |

---

## Preregistration

The hypotheses, measured variables, conditions and analyses were fixed before the runs they govern; later sets were
added before their own runs. The frozen files are listed in `MANIFEST.sha256`, whose own SHA-256 is quoted in the
paper, so anyone can verify that the protocol, checker, analysis code, instance files and experiment code are those
the runs used. `PREREGISTRATION.md` explains what is frozen and how to check it, and `RELEASE_NOTES.md` lists every
difference between this release and the files used for the runs.

---

## Data

- **Random graphs.** Generated by the code with fixed seeds (`m0/build*.py`, `e3/make_data.sh`).
- **Real topologies.** ogbg-molhiv (MIT) and ogbn-arxiv (ODC-BY) from the Open Graph Benchmark, downloaded by
  `m0/build_real.py`.
- **GraphQA node tasks.** Regenerated by the code from the task definitions and graph distribution of Fatemi et al.
  (2024).
- No personal data are used, and no human subjects were involved.

---

## License

- **Code**: MIT License, see [`LICENSE`](LICENSE).
- **Instance files** (`m0/data/` and the instance-derived files in `m0/results/` and `e3/results/`): Creative Commons
  Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Models** are used under their own licenses: Apache 2.0 (Qwen2.5 except 3B and 72B, MPT-7B), Qwen Research License
  (Qwen2.5-3B), Qwen License (Qwen2.5-72B), Llama 3.1 Community License, Gemma Terms of Use, BigScience RAIL
  (BLOOM-7B1), MIT (DeepSeek-R1-Distill-Qwen-7B), and the OpenAI terms of use (GPT-4o).
