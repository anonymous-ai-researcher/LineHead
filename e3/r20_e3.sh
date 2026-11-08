#!/bin/bash
# (CONFIG.md §6.14): the controlled experiments E22, E23, E25, E26, E27, E32 and E33, after make_data.sh.
# Usage: bash r20_e3.sh data | pos-control | data-limited | window | nest | nest-dist | depth2 | robust | all
#   data          generate the data (gen_r20.py; E26 training mix) into data/ and data_nest/
#   pos-control   E22: NoPE, L = 6, tasks g1 and c3, 4 formats, seeds 0-4                         (40 runs, ~6 GPU h)
#   data-limited  E23: rope, alibi_only, local_alibi, local_hardsince; dia; N in {25k..500k}; 4 formats; seeds 0-4;
#                 evaluated every 250 steps                                                     (400 runs, ~64 GPU h)
#   window        E25: dlocal, w in {2,4,8}, L in {3,4,6}, AL and NC, dia, seeds 0-2           (54 runs, ~27 GPU h)
#   nest          E26: lastglobal and local, L = 6, EL and ECL, dia, test pm8..pm128, seeds 0-2 (12 runs, ~9 GPU h)
#   depth2        E27: dlocal (w = 8), local_hardsince, rope, local; tasks dd and c2; 4 formats; seeds 0-2 (96 runs, ~16 GPU h)
#   nest-dist     E33: the four arms of E23 under ELB (EL with block colors: two nested matches, colors next to
#                 the edges), N in {25k..500k}, seeds 0-4; EL, AL, NC, ECL come from E23              (100 runs, ~16 GPU h)
#   robust        E32: depth sweep seeds 3-4 (L in {2,3,4,8,12}), local and local_hardsince at d in {128, 512}
#                 (d = 256 is the main grid), dia, 4 formats, seeds 0-2                         (88 runs, ~20 GPU h)
# Results: results/r20_<mode>.jsonl (one JSON line per run, train_e3.py format); summary: r20_summary.py.
cd "$(dirname "$0")"; mkdir -p results logs
LR=$(python3 select_lr.py --print results/lr_choice.json 2>/dev/null) || LR=3e-4
J=${E3_JOBS:-8}
SERS="EL AL NC ECL"
# offline end-to-end test (E2E_TINY=1): one seed and the smallest grid of every design, so every code path runs
if [ -n "$E2E_TINY" ]; then S5="0"; S3="0"; S34="3"; NS="25000 500000"; WS="2 8"; LS="3"; DS="2 4"; DW="128";
else S5="0 1 2 3 4"; S3="0 1 2"; S34="3 4"; NS="25000 50000 100000 200000 500000"; WS="2 4 8"; LS="3 4 6"; DS="2 3 4 8 12"; DW="128 512"; fi
i=0
run() {   # run <log name> <out> <train_e3.py arguments...>
  local log=$1 out=$2; shift 2
  CUDA_VISIBLE_DEVICES=$((i % J)) python3 train_e3.py --lr $LR --out $out "$@" > logs/r20_$log.log 2>&1 &
  i=$((i + 1)); if [ $((i % J)) -eq 0 ]; then wait; fi
}

mode_data() {
  for task in g1 c3 dd c2; do for ser in $SERS; do
    python3 gen_r20.py --task $task --ser $ser --split train --pairs 250000 --seed 0 --out data/${task}_${ser}_train.jsonl &
    python3 gen_r20.py --task $task --ser $ser --split val   --pairs 1000   --seed 4 --out data/${task}_${ser}_val.jsonl &
    python3 gen_r20.py --task $task --ser $ser --split id    --pairs 5000   --seed 1 --out data/${task}_${ser}_id.jsonl &
    python3 gen_r20.py --task $task --ser $ser --split ood   --pairs 5000   --seed 2 --out data/${task}_${ser}_ood.jsonl &
  done; wait; done
  for sp in train val id ood; do                        # E33: dia under ELB (E3 graphs and seeds)
    case $sp in train) n=250000; sd=0;; val) n=1000; sd=4;; id) n=5000; sd=1;; ood) n=5000; sd=2;; esac
    python3 gen_e3.py --task dia --ser ELB --split $sp --pairs $n --seed $sd --out data/dia_ELB_$sp.jsonl &
  done; wait
  mkdir -p data_nest
  for ser in EL ECL; do
    for E in 8 16 32 64 128; do
      python3 gen_r20.py --task dia --ser $ser --split pm$E --pairs 2000 --seed 7 --out data_nest/dia_${ser}_pm$E.jsonl &
    done
    python3 gen_r20.py --task dia --ser $ser --split pmtrain --pairs 125000 --seed 0 --out data_nest/pm_${ser}.jsonl &
    wait
    # training mix: half E3 dia training pairs, half matchings (pairs stay adjacent; train_e3.py shuffles examples)
    head -n 250000 data/dia_${ser}_train.jsonl | cat - data_nest/pm_${ser}.jsonl > data_nest/dia_${ser}_train.jsonl
    rm -f "data_nest/pm_${ser}.jsonl"
    for k in val id ood kset; do ln -sf ../data/dia_${ser}_$k.jsonl data_nest/dia_${ser}_$k.jsonl; done
  done
}

mode_pos_control() {
  for seed in $S5; do for task in g1 c3; do for ser in $SERS; do
    run e22_${task}_${ser}_$seed results/r20_pos-control.jsonl --task $task --ser $ser --pos nope --seed $seed --data data
  done; done; done
}

mode_data_limited() {
  for seed in $S5; do for pos in rope alibi_only local_alibi local_hardsince; do
    for N in $NS; do for ser in $SERS; do
      run e23_${pos}_${ser}_${N}_$seed results/r20_data-limited.jsonl --task dia --ser $ser --pos $pos --seed $seed \
          --train-size $N --eval-every 250 --data data
    done; done
  done; done
}

mode_nest_dist() {
  for seed in $S5; do for pos in rope alibi_only local_alibi local_hardsince; do for N in $NS; do
    run e33_${pos}_${N}_$seed results/r20_nest-dist.jsonl --task dia --ser ELB --pos $pos --seed $seed \
        --train-size $N --eval-every 250 --data data
  done; done; done
}

mode_window() {
  for seed in $S3; do for w in $WS; do for L in $LS; do for ser in AL NC; do
    H=$((4 + (w > 4 ? w : 4)))
    run e25_w${w}_L${L}_${ser}_$seed results/r20_window.jsonl --task dia --ser $ser --pos dlocal --window $w \
        --heads $H --d $((32 * H)) --layers $L --seed $seed --data data
  done; done; done; done
}

mode_nest() {
  for seed in $S3; do for pos in lastglobal local; do for ser in EL ECL; do
    run e26_${pos}_${ser}_$seed results/r20_nest.jsonl --task dia --ser $ser --pos $pos --layers 6 --seed $seed \
        --data data_nest --extra-tests pm8,pm16,pm32,pm64,pm128
  done; done; done
}

mode_depth2() {
  for seed in $S3; do for task in dd c2; do for ser in $SERS; do
    run e27_dlocal_${task}_${ser}_$seed results/r20_depth2.jsonl --task $task --ser $ser --pos dlocal --window 8 \
        --heads 12 --d 384 --layers 6 --seed $seed --data data
    for pos in local_hardsince rope; do
      run e27_${pos}_${task}_${ser}_$seed results/r20_depth2.jsonl --task $task --ser $ser --pos $pos --seed $seed --data data
    done
    run e27_local_${task}_${ser}_$seed results/r20_depth2.jsonl --task $task --ser $ser --pos local --seed $seed --data data
  done; done; done
}

mode_robust() {
  for seed in $S34; do for L in $DS; do for ser in $SERS; do
    run e32_depth_${ser}_L${L}_$seed results/r20_robust.jsonl --task dia --ser $ser --pos local --layers $L --seed $seed --data data
  done; done; done
  for seed in $S3; do for d in $DW; do for pos in local local_hardsince; do for ser in $SERS; do
    run e32_d${d}_${pos}_${ser}_$seed results/r20_robust.jsonl --task dia --ser $ser --pos $pos --d $d --seed $seed --data data
  done; done; done; done
}

case "$1" in
  data) mode_data ;;
  pos-control) mode_pos_control ;;
  data-limited) mode_data_limited ;;
  window) mode_window ;;
  nest) mode_nest ;;
  nest-dist) mode_nest_dist ;;
  depth2) mode_depth2 ;;
  robust) mode_robust ;;
  all) mode_data; mode_pos_control; mode_data_limited; mode_window; mode_nest; mode_nest_dist; mode_depth2; mode_robust ;;
  *) echo "usage: $0 data|pos-control|data-limited|window|nest|nest-dist|depth2|robust|all"; exit 1 ;;
esac
wait
