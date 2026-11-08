# results/

`analysis/import_measured.py` writes `summary.json` here: one block per experiment, in the structure of
`analysis/schema.json`, with the values read from the summary files of `m0/results/`, `e3/results/` and `pilot1/`.
`analysis/tables.py` exports every block to `tables/<block>.csv`. The commands are in RUN_PROTOCOL.md §7.
