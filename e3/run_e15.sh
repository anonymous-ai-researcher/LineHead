#!/bin/bash
# E15 (CONFIG.md §6.10): data efficiency under exact since. Arm 1-local+hard since, task dia, formats EL,
# AL, NC, ECL, training sets of N in {25k, 50k, 125k, 250k} examples (the full set is the E3 run itself), seeds 0-2:
# 4 x 4 x 3 = 48 runs of 40,000 steps (~0.7 h each). Results: results/e15.jsonl (field train_size).
cd "$(dirname "$0")"; mkdir -p results logs
LR=$(python3 select_lr.py --print results/lr_choice.json) || { echo "run run_lr_sweep.sh first"; exit 1; }
J=${E3_JOBS:-8}
i=0
for seed in 0 1 2; do for N in 25000 50000 125000 250000; do for ser in EL AL NC ECL; do
  CUDA_VISIBLE_DEVICES=$((i % J)) python3 train_e3.py --task dia --ser $ser --pos local_hardsince --seed $seed \
      --lr $LR --data data --train-size $N --out results/e15.jsonl > logs/e15_${ser}_${N}_${seed}.log 2>&1 &
  i=$((i + 1))
  if [ $((i % J)) -eq 0 ]; then wait; fi
done; done; done
wait
