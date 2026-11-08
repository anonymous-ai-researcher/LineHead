#!/bin/sh
# Regenerates every results file of proofs13 (run from any directory; reads ../../e3/ownline_check.py).
cd "$(dirname "$0")"
./run_A.sh                                          # item (A): results_A.txt
python3 -I el_depth1.py      > results_B.txt 2>&1   # item (B)
python3 -I latest_match.py   > results_C_latest.txt # item (C)
python3 -I dims_ln.py        > results_D.txt 2>&1   # item (D)
./run_mytypes.sh                                    # independent check: results_mytypes.txt
python3 -I dims13.py         > /dev/null            # results_dims13.json
