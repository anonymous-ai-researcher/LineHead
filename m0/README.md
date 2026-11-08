# m0: instances, LLM runs and statistics of the LLM experiments

The commands are in `../RUN_PROTOCOL.md`, the configuration in `../CONFIG.md`.

## Frozen inputs (do not regenerate over them)

`data/` holds the frozen instance files listed in `../MANIFEST.sha256`. The build commands of RUN_PROTOCOL §3, §4.13
and §4.15 reproduce them byte for byte; if you rebuild them, write to another path and compare (`cmp`).

| File | Content | Lines |
|---|---|---|
| `data/m0_instances.jsonl` | E1 (and the E2 query questions, E4, E7 D1a, E8, E9): 12 questions (G1-G3, D1a-D1c, D2a-D2c, C2, C2p, C3) x 400 matched pairs | 9,600 |
| `data/e2_controls.jsonl` | E2 positive controls X1, X2, 400 pairs each | 1,600 |
| `data/e7_instances.jsonl` | E7 Q0 ("v is red"), 400 pairs | 800 |
| `data/e7_twin.jsonl` | E7 twin-graph Q0 pairs | 800 |
| `data/demo_instances.jsonl` | E6 demonstrations, D1a-D1c, 50 pairs each (held out) | 300 |
| `data/e8_colors3.jsonl` | E8 three colors, G1-G3 and D1a-D1c, 200 pairs each | 2,400 |
| `data/e10_instances.jsonl` | E10 GPT-4o, G1-G3 and D1a-D1c, 200 pairs each | 2,400 |
| `data/e11b_instances.jsonl` | E11b degree sweep, 4 degrees x 2 questions x 150 pairs | 2,400 |
| `data/e11bL_instances.jsonl` | E11b-L, the E11b pairs padded per format | 9,600 |
| `data/e12_probe.jsonl` | E12 since probe, 2 formats x 9 positions x 60 pairs | 2,160 |
| `data/e13_graphqa.jsonl` | E13 GraphQA node tasks | 800 |
| `data/e14_instances.jsonl` | E14, 300 D1a pairs x 7 filler conditions | 4,200 |
| `data/e21_instances.jsonl` | E21 relational tables | 800 |
| `results/kmin_shares.json`, `results/shortcut_ceiling*.json`, `results/compete_static.json`, `results/token_cost_measured.json` | computed on the frozen instances (no model), with the instances' SHA-256 | |

Prompts per model in E1: 9,600 instances x 3 serializations (EL, AL, NC) x 3 templates = 86,400 scored prompts.

## Code

| File | Role |
|---|---|
| `tasks.py` | the 15 properties (12 questions, Q0, X1, X2) with exact labels and their logic class |
| `serialize.py` | EL, AL, NC, ECL, GraphML, EL-2, EL-far, NC-H, NC-nw, Incident, Friendship, ELB; every line starts with a space (CONFIG §1.2) |
| `prompts.py`, `instances.py` | 3 templates; instance -> prompts (masks for E2/E7, integer ids, sorted edges, question first, filler) |
| `names.py` | name pool, color and task blacklists; per-tokenizer name check `name_ok` (one stable token id before every following character) and per-prompt check `occurrences_ok` |
| `build.py`, `build_blind.py`, `build_twin.py`, `build_real.py`, `build_r12.py`, `build_r14.py` | instance builders (E1/E6/E7/E8/E10; E2 controls; E7 twin pairs; E5 real topologies; E11b, E12, E13; E11b-L, E14) |
| `verify_data.py`, `probe_shortcuts.py`, `kmin_shares.py`, `token_cost.py`, `prompt_box.py` | label and pair checks; estimated shortcut ceiling; k_min shares; token cost per format; verbatim prompt listing |
| `run_llm.py` | backends `hf` (log-probability of Yes vs No; every distinct token sequence scored once, left padding with explicit position ids), `openai`, `anthropic`, `mock` (random answers for pipeline tests only; its records are marked `mock`) |
| `stats.py` | E1 statistics: cells, per question and template, permutation tests, planned contrasts, GEE interaction tests, distance regressions, generation agreement |
| `e4_stats.py`, `joint_stats.py` | cell summaries with k_min strata; four-model joint bootstrap means; E2 Holm family |
| `r9_stats.py`, `dmin_tokens.py`, `compete.py` | additional analyses: paired metrics, position, TOST, lengths, question random effects, k_min differences, P4 non-inferiority; d_min under the model tokenizers; competing cell models |
| `e9_mechanism.py` | E9 probes, retrieval, patching, inner match |
| `e11_heads.py` | E11a, E17, E18: head scoring, mean ablation, activation patching, summaries |
| `e12_probe.py`, `e13_graphqa.py` | E12 since probe and zero-parameter prediction; E13 GraphQA |
| `r19.py` | E19 knockout and entry patching, E20 within-line filler, E21 relational tables |
| `r20.py`, `lora_r20.py`, `r20_blocks.py` | E28 to E31 and E33c; E24 and E33b LoRA fine-tuning; the import file of E22 to E33 |
| `r21.py` | E34 to E36 |
| `fig1.py` | Figure 1 from a summary file |
| `make_tiny_model.py` | a tiny random local Llama with a word-level tokenizer, for tests only |

## Tests

```bash
python -m pytest -q            # unit tests (no GPU, no model download)
python test_e4_stats.py        # strata: TPR of the stratum's positives with the TNR of all negatives; Holm within model
python test_run_llm_hf.py      # identical prompts bit-identical across batch compositions; position-correct left padding; name check
```
`E2E_TINY=1` shrinks every script to a few pairs and resamples so that a full pass runs on CPU with the `mock` backend
and the tiny model; its outputs are meaningless and are never analysed.
