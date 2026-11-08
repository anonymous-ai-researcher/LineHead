#!/bin/bash
# Pre-registered learning-rate sweep (CONFIG.md §3.4), run BEFORE the main grid: 1-local, AL, dia, seed 0,
# peak lr in {1e-4, 3e-4, 1e-3}, everything else as in E3. select_lr.py picks the lr with the highest final
# ID-validation balanced accuracy (ties: the smaller lr) and writes results/lr_choice.json, which run_e3.sh and
# run_e3d.sh read. 3 runs, about 2.1 GPU hours.
cd "$(dirname "$0")"; mkdir -p results logs
i=0
for LR in 1e-4 3e-4 1e-3; do
  CUDA_VISIBLE_DEVICES=$i python3 train_e3.py --task dia --ser AL --pos local --seed 0 --lr $LR --data data \
      --out results/lr_sweep.jsonl > logs/lr_sweep_$LR.log 2>&1 &
  i=$((i + 1))
done
wait
python3 select_lr.py results/lr_sweep.jsonl --out results/lr_choice.json
