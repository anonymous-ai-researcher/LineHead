#!/bin/bash
# E0b (CONFIG.md §4): scaling sweep for X1 under NC, name-blind, learned absolute positions, AdamW 1e-3, batch 32.
# Grid: width {64,128,256} x layers {3,6} x train pairs {4000,16000} x steps {8000,32000}, 3 seeds = 72 runs.
# Also completes the pilot chance cells with 3 seeds at the budgets of the decidable cells (F-7).
cd "$(dirname "$0")"
DEV=${E0B_DEVICE:-cpu}                 # cpu (default) or cuda: E0B_DEVICE=cuda bash run_e0b.sh on one A100
OUT=results_e0b.jsonl
for d in 64 128 256; do for L in 3 6; do for P in 4000 16000; do for S in 8000 32000; do for seed in 0 1 2; do
  python3 train.py --task X1 --ser NC --arm blind --seed $seed --steps $S --bs 32 --lr 1e-3 \
    --train-pairs $P --test-pairs 1000 --d $d --layers $L --device $DEV --out $OUT
done; done; done; done; done
# chance cells, 3 seeds, budgets 4000 (X2) and 8000 (X1, D1a)
OUT2=results_chance3.jsonl
for seed in 0 1 2; do
  python3 train.py --task X2 --ser EL --arm blind --seed $seed --steps 4000 --bs 32 --lr 1e-3 --train-pairs 4000 --test-pairs 1000 --device $DEV --out $OUT2
  python3 train.py --task X1 --ser EL --arm blind --seed $seed --steps 8000 --bs 32 --lr 1e-3 --train-pairs 4000 --test-pairs 1000 --device $DEV --out $OUT2
  python3 train.py --task X1 --ser AL --arm blind --seed $seed --steps 8000 --bs 32 --lr 1e-3 --train-pairs 4000 --test-pairs 1000 --device $DEV --out $OUT2
  for ser in EL AL NC; do
    python3 train.py --task D1a --ser $ser --arm blind --seed $seed --steps 8000 --bs 32 --lr 1e-3 --train-pairs 4000 --test-pairs 1000 --device $DEV --out $OUT2
  done
done
