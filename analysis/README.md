# analysis: import, validation, tables and figures

| File | Role |
|---|---|
| `schema.json` | structure of `results/summary.json`: every block and key, the design constants, and the blocks computed from the instance files |
| `import_measured.py` | reads the experiment summaries and writes `results/summary.json` (RUN_PROTOCOL §7) |
| `load_data.py` | loads the schema with the imported values |
| `check_rules.py`, `real_rules.py`, `validate.py` | the rules of VALIDATION_RULES.md |
| `tables.py`, `fmt.py` | CSV export of every block; the rounding rule of every displayed number (half up at the displayed precision) |
| `figures/` | the paper's figures from `results/summary.json` (`style.py` holds the shared style; `_record.py` records the values a figure draws) |

The figure scripts render text with LaTeX (`text.usetex`), so a TeX installation with the `times`, `newtxmath` and
`inconsolata` packages is needed.
