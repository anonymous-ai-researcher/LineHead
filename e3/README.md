# e3: controlled models (positional mechanisms)

Causal transformers trained from scratch, with only the positional mechanism varied (CONFIG §3). The commands are
in RUN_PROTOCOL §4.2, §4.13 to §4.15 and §4.19.

| File | Role |
|---|---|
| `gen_e3.py`, `make_data.sh` | matched pairs for "v is red" and "v has a red neighbor"; ID, OOD, k-controlled, ID-strata, own-line, Delta <= 256 sets |
| `hard_pairs.py` | the hard pairs of Theorem lb:main, smallest construction and the proof's parameters |
| `ownline_check.py`, `shortcut_check.py`, `kdist.py` | position control checked on the tokens; line-index shortcut with the exact TL evaluator; k_min distribution of the test sets |
| `train_e3.py` | training and evaluation of one run (mechanisms NoPE, 1-local, ALiBi-only, 1-local+ALiBi, 1-local+hard-since, RoPE, w-local, the one-global-layer arm) |
| `run_lr_sweep.sh`, `select_lr.py` | the preregistered learning-rate sweep and its selection |
| `run_e3.sh`, `run_e3d.sh`, `export_l6.py`, `run_e3_extra.sh`, `run_e15.sh` | main grid, depth sweep (L = 6 reused from the main grid), learning-rate check and NC-H arm, E15 |
| `gen_r20.py`, `r20_e3.sh`, `r20_summary.py` | data, runs and summary of E22, E23, E25 to E27, E32, E33a |
| `match_heads.py` | name matching measured inside the trained models |
| `r9_e3.py`, `e15_summary.py` | summaries of the E3 runs |
| `tl_eval.py`, `rtypes.py` | evaluator of the logic and round types on token sequences |

`results/kdist_measured.json`, `results/kset_shortcut.json`, `results/kid_shortcut.json` and
`results/kset_blocking_check.json` are computed from the generated test sets (no model) and record the SHA-256 of the
generator code and of the files they read; `make_data.sh` followed by `kdist.py` reproduces them.

Tests: `python -m pytest -q` (no GPU).
