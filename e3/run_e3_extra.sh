#!/bin/bash
# additions to E3 (CONFIG.md §3.6), after run_e3.sh:
#  (1) learning-rate check of NoPE and RoPE : AL, dia, seed 0, peak lr in {1e-4, 3e-4, 1e-3} -> 6 runs;
#      everything else as in E3 (the main grid keeps the lr chosen by run_lr_sweep.sh).
#  (2) NC-H arm (name transport): format NCH (NC without head colors), arms local and local_hardsince,
#      tasks red and dia, seeds 0-4 -> 20 runs.
# The in-distribution k_min strata (kid) and the own-line set (kown) are evaluated by every run whose data folder
# holds them (make_data.sh), including the E3 and E3d runs. About 18.2 GPU hours (26 runs x 0.7 h).
cd "$(dirname "$0")"; mkdir -p results logs
LR=$(python3 select_lr.py --print results/lr_choice.json) || { echo "run run_lr_sweep.sh first"; exit 1; }
J=${E3_JOBS:-8}
jobs=()
for pos in nope rope; do for lr in 1e-4 3e-4 1e-3; do jobs+=("dia AL $pos 0 $lr results/lr_check.jsonl"); done; done
for seed in 0 1 2 3 4; do for pos in local local_hardsince; do for task in red dia; do
  jobs+=("$task NCH $pos $seed $LR results/e3_nch.jsonl")
done; done; done
i=0
for j in "${jobs[@]}"; do
  set -- $j
  CUDA_VISIBLE_DEVICES=$((i % J)) python3 train_e3.py --task $1 --ser $2 --pos $3 --seed $4 --lr $5 --data data \
      --out $6 --kset-examples results/kset_examples_extra.jsonl > logs/extra_$1_$2_$3_$4_$5.log 2>&1 &
  i=$((i + 1))
  if [ $((i % J)) -eq 0 ]; then wait; fi
done
wait
