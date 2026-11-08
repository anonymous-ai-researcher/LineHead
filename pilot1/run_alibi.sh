#!/bin/bash
# E0, preregistered ALiBi test (PREREGISTRATION.md): X1 under NC, name-blind, the pilot's model, data and schedule,
# seeds 0-2; the only change is the ALiBi bias.
cd "$(dirname "$0")"
OUT=results_alibi.jsonl
for s in 0 1 2; do
  python3 train.py --task X1 --ser NC --arm blind --seed $s --steps 8000 --bs 32 --lr 1e-3 --train-pairs 4000 \
      --test-pairs 1000 --alibi --out $OUT
done
