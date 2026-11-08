#!/bin/bash
# E3 data (CONFIG.md §3.3). Graphs depend on (task, split, seed) only, so all four formats share graphs and orders.
set -e
cd "$(dirname "$0")"; mkdir -p data
for task in red dia; do for ser in EL AL NC ECL NCH; do
  python3 gen_e3.py --task $task --ser $ser --split train --pairs 250000 --seed 0 --out data/${task}_${ser}_train.jsonl &
  python3 gen_e3.py --task $task --ser $ser --split val   --pairs 1000   --seed 4 --out data/${task}_${ser}_val.jsonl &
  python3 gen_e3.py --task $task --ser $ser --split id    --pairs 5000   --seed 1 --out data/${task}_${ser}_id.jsonl &
  python3 gen_e3.py --task $task --ser $ser --split ood   --pairs 5000   --seed 2 --out data/${task}_${ser}_ood.jsonl &
  if [ $task = dia ] && [ $ser != NCH ]; then
    # position-controlled k_min sets (own-line routes blocked; ownline_check.py verifies), ID strata, own-line set;
    # the NC-H arm is evaluated on ID and OOD only, so no NCH k-sets are generated
    python3 gen_e3.py --task dia --ser $ser --split kset --pairs 2000 --seed 3 --out data/dia_${ser}_kset.jsonl &
    python3 gen_e3.py --task dia --ser $ser --split kid  --pairs 2000 --seed 5 --out data/dia_${ser}_kid.jsonl &
    if [ $ser = AL ] || [ $ser = NC ]; then         # Delta <= 256 test set
      python3 gen_e3.py --task dia --ser $ser --split d256 --pairs 1000 --seed 7 --out data/dia_${ser}_d256.jsonl &
      python3 hard_pairs.py --ser $ser --pairs 1000 --seed 8 --out data/dia_${ser}_hard.jsonl &   # Theorem lb:main
      for L in 4 6 8; do                     # instances of the proof's construction, depth L
        python3 hard_pairs.py --ser $ser --pairs 500 --seed 9 --proof-L $L --out data/dia_${ser}_hardp$L.jsonl &
      done
    fi
    if [ $ser = AL ]; then
      python3 gen_e3.py --task dia --ser AL --split kown --pairs 2000 --seed 6 --out data/dia_AL_kown.jsonl &
    fi
  fi
  wait
done; done
mkdir -p results
python3 ownline_check.py data/dia_AL_kset.jsonl data/dia_NC_kset.jsonl data/dia_AL_kid.jsonl data/dia_NC_kid.jsonl \
    data/dia_AL_kown.jsonl --out results/kset_blocking.json
# line-index shortcuts of the position-controlled sets at chance (at most 0.55; Prop. lb:top)
python3 shortcut_check.py --split kset --seed 3 --pairs 300 --out results/kset_shortcut.json
python3 shortcut_check.py --split kid --seed 5 --pairs 300 --out results/kid_shortcut.json
