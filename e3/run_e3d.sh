#!/bin/bash
# E3d depth sweep (CONFIG.md §3.5): 1-local arm, task dia, L in {2,3,4,6,8,12}, 4 formats, seeds 0-2; L=6 reuses the main E3 runs (1-local, dia, seeds 0-2), so 60 new runs.
cd "$(dirname "$0")"; mkdir -p results logs
LR=$(python3 select_lr.py --print results/lr_choice.json) || { echo "run run_lr_sweep.sh first"; exit 1; }
J=${E3_JOBS:-8}
i=0
for seed in 0 1 2; do for L in 2 3 4 8 12; do for ser in EL AL NC ECL; do
  CUDA_VISIBLE_DEVICES=$((i % J)) python3 train_e3.py --task dia --ser $ser --pos local --layers $L --seed $seed \
      --lr $LR --data data --out results/e3d.jsonl > logs/e3d_${ser}_L${L}_$seed.log 2>&1 &
  i=$((i + 1)); if [ $((i % J)) -eq 0 ]; then wait; fi
done; done; done
wait
python3 export_l6.py --e3 results/e3.jsonl --e3d results/e3d.jsonl     # L = 6: the 12 main-grid runs (seeds 0-2)
