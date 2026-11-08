# Differences between this release and the files used for the runs

This repository is the released copy of the files that the experiments ran with. This file lists, file by file, how
the two differ. It is listed in `MANIFEST.sha256`, so its content is covered by the manifest's SHA-256.

## Experiment code (`m0/`, `e3/`, `pilot1/`, `theory_checks/`)

`tools/compare_release.py OTHER_ROOT` produces the categories below for any other copy of these folders; its
docstring states the comparison rules. Python files are compared on the syntax tree (`ast.dump`) with docstrings
removed, shell scripts on the text with comments and blank lines removed.

| Category | Count | Files |
|---|---|---|
| Identical bytes | 89 | among them every instance file in `m0/data/` and every theory-check output |
| Comments and docstrings only | 51 | 44 Python files and the shell scripts `e3/make_data.sh`, `e3/run_e3.sh`, `e3/run_e3_extra.sh`, `e3/run_e15.sh`, `e3/r20_e3.sh`, `pilot1/run_grid.sh`, `theory_checks/part2/run_mytypes.sh` |
| String literals only | 3 | `m0/run_llm.py`, `e3/train_e3.py` (argparse help texts); `m0/prompt_box.py` (default output path, now `prompt_box.tex`) |
| Code | 2 | tests only: `m0/test_r12.py` (a pytest fixture, so that `python -m pytest` collects every test); `m0/test_run_llm_misc.py` (imports `analysis/import_measured.py` from its location in this repository) |
| Other files with different bytes | 7 | README files of `m0/` and `theory_checks/`; the four instance-derived files below |
| New | 3 | `pilot1/run_alibi.sh` (the E0 ALiBi runs as preregistered: one `train.py` call per seed); README files |

Effects of the non-identical categories:
- Docstrings reach a program only through `__doc__`. Here that is the `--help` text of `m0/r19.py`, `m0/r20.py`,
  `m0/r20_blocks.py`, `m0/r21.py` and `m0/lora_r20.py`, which pass `description=__doc__` to argparse.
- In `m0/run_llm.py` and `e3/train_e3.py` the changed strings are argparse help texts only; no default value
  changed, so they change only `--help`. The default output path of `m0/prompt_box.py` is a listing of verbatim
  prompts that no other script reads; `analysis/figures/fig1.py` imports its `prompt` function, which is unchanged.
- The instance-derived files (`m0/results/token_cost_measured.json`, `e3/results/kdist_measured.json`,
  `e3/results/kset_shortcut.json`, `e3/results/kid_shortcut.json`) were recomputed with the released code. Every value
  is unchanged. Only the SHA-256 values they record for code files changed, because those files' comments changed,
  and each recorded SHA-256 equals the released file.

Not released: the pilot's run logs, mock outputs of the pipeline tests, smoke-test data, and a test that compares a
rerun with the pilot's log.

## Analysis (`analysis/`)

`analysis/import_measured.py` reads the experiment summaries into `results/summary.json`, whose structure is
`analysis/schema.json`. `analysis/check_rules.py` and `analysis/real_rules.py` apply the validation rules.

Compared with the checker used for the paper:

1. **Rules not included.** A3, A4, A5, D6b, D8, S12, S13, S14 and F1 are left out.
   - They checked the expected values themselves (interval widths, seed spread and similar properties, and the
     compute plan). The checker used for the paper already skipped them automatically for every block holding
     measured results.
   - On a summary of measured results, both checkers therefore evaluate the same rules.
   - None of these rules is cited in the paper. The paper cites V-B1 to V-B16, V-C1 to V-C74, V-D12 and V-D13, and
     all of them are in `analysis/check_rules.py` with unchanged criteria.
2. **Rule C15.** Its base-cell clause compares the base cell of E0b with the X1-NC cell of E0 read from
   `pilot1/results.jsonl`, instead of a fixed number. Without that file the clause is reported as skipped, with
   its reason, and is not counted as passed. The other clauses of C15 are unchanged.
3. **Rule D1 for the declines of E11b and E11b-L.** The checker used for the paper compared a stored decline with
   its cells only when all three came from the same source, which matters only while a block is partly imported.
   The released checker compares every decline. On a complete summary both give the same result.
4. **Constants.** The constants that rules B6, B8 and C1 use (6 layers, the round counts 2k+2 and 5k, and the
   preregistered route parameters c_pos and tnr) are written directly in `analysis/check_rules.py`, with the values
   the checker used for the paper reads.

`analysis/tables.py` (CSV export) and `analysis/validate.py` (runs both checkers; R7 as warnings) are new. Both
checkers report every check as passed, failed or skipped (R7: warned), with counts per rule under `--verbose`;
running both versions of the checker on the same results and comparing their passed and failed checks shows
whether they apply the same rules. The
figure scripts are those of the paper and read `results/summary.json`.
