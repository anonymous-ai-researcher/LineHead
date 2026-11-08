# pilot1: the pilot (E0) and its scaling grid (E0b)

Small causal transformers with learned absolute positions on graphs with 5 to 9 vertices (paper Section 4 and the
appendix on the pilot). The commands are in RUN_PROTOCOL §4.0 and §4.1.

| File | Role |
|---|---|
| `train.py` | one run: task, serialization, arm (name-blind or names kept), seed, width, layers, steps, optional ALiBi |
| `serialize_pilot.py` | the pilot's serializer and name pool |
| `gen.py` | graph and pair generation of the pilot |
| `run_grid.sh` | E0 grid in two phases (name-blind cells; names kept on D1a) |
| `run_alibi.sh` | the preregistered ALiBi test of E0 (PREREGISTRATION.md) |
| `run_e0b.sh` | E0b scaling grid and the chance-cell reruns |
| `analyze.py` | per-cell accuracy, exact binomial tests against 0.5 with Holm correction, Wilson intervals |
