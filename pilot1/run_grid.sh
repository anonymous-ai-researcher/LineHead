#!/bin/bash
# Pilot grid (E0), run in two phases: bash run_grid.sh 1; bash run_grid.sh 2.
#   phase 1: ID-blind cells. Solvable cells 3 seeds; chance cells (guaranteed by Theorem 1 (thm:blind) of the paper, because
#            positive and negative masked inputs have identical distributions) 1 seed each as an
#            implementation check. Step budgets: G1 1000, X2 4000, X1 8000, chance cells 1000.
#   phase 2: names kept (ID tokens distinct), D1a, 3 serializations x 3 seeds, 4000 steps.
cd "$(dirname "$0")"
OUT=results.jsonl
run() { python3 train.py --task $1 --ser $2 --arm $3 --seed $4 --steps $5 --bs 32 --lr 1e-3 --train-pairs 4000 --test-pairs 1000 --out $OUT; }
if [ "$1" = "1" ]; then
  for s in 0 1 2; do for ser in EL AL NC; do run G1 $ser blind $s 1000; done; done
  for s in 0 1 2; do run X2 AL blind $s 4000; run X2 NC blind $s 4000; done
  for s in 0 1 2; do run X1 NC blind $s 8000; done
  run X2 EL blind 0 1000; run X1 EL blind 0 1000; run X1 AL blind 0 1000
  for ser in EL AL NC; do run D1a $ser blind 0 1000; done
  echo PHASE1_DONE
fi
if [ "$1" = "2" ]; then
  for s in 0 1 2; do for ser in EL AL NC; do run D1a $ser ids $s 4000; done; done
  echo PHASE2_DONE
fi
