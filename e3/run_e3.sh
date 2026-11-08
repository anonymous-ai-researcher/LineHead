#!/bin/bash
# E3 main grid (CONFIG.md §3): 6 arms x 2 tasks x 4 formats x 5 seeds = 240 runs, one A100 each (~0.7 h).
# Runs 8 jobs in parallel on an 8-GPU node. Results are appended to results/e3.jsonl.
cd "$(dirname "$0")"; mkdir -p results logs
LR=$(python3 select_lr.py --print results/lr_choice.json) || { echo "run run_lr_sweep.sh first"; exit 1; }
J=${E3_JOBS:-8}                                   # parallel jobs (GPUs)
jobs=()
for seed in 0 1 2 3 4; do for pos in nope local alibi_only local_alibi local_hardsince rope; do
  for task in red dia; do for ser in EL AL NC ECL; do jobs+=("$task $ser $pos $seed"); done; done
done; done
i=0
for j in "${jobs[@]}"; do
  set -- $j
  CUDA_VISIBLE_DEVICES=$((i % J)) python3 train_e3.py --task $1 --ser $2 --pos $3 --seed $4 --lr $LR --data data \
      --out results/e3.jsonl $([ $1 = dia ] && echo --save-dir ckpt) > logs/e3_$1_$2_$3_$4.log 2>&1 &   # dia checkpoints
  i=$((i + 1))
  if [ $((i % J)) -eq 0 ]; then wait; fi
done
wait
