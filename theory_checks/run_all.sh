#!/bin/bash
# Reproduces results_pairs.txt. Exact (frozenset) types for kappa=1; Zobrist-hashed for the larger sweeps.
cd "$(dirname "$0")"
{
echo "== exact, kappa=1 (the paper's rho: threshold-1 formulas)"
python3 -I check_pairs.py AL 2 12
python3 -I check_pairs.py NC 2 12
echo "== hashed, kappa=1, larger L"
python3 -I check_pairs.py AL 13 16 --hash
python3 -I check_pairs.py NC 13 16 --hash
echo "== sanity: k = k*(L) (head route reachable) must be distinguishable"
python3 -I check_pairs.py AL 4 12 --k -1 --hash
python3 -I check_pairs.py NC 5 16 --k -1 --hash
echo "== fresh (unshared) hub entries at k*(L)+1: self-matching shortcut"
python3 -I check_pairs.py AL 2 12 --fresh --hash
python3 -I check_pairs.py NC 2 12 --fresh --hash
echo "== counting thresholds for free (kappa=2,3), N = 2L(kappa+1)+6"
for kap in 2 3; do for L in 4 5 6 7 8 9 10; do N=$((2*L*(kap+1)+6))
python3 -I check_pairs.py AL $L $L --kappa $kap --N $N --hash; python3 -I check_pairs.py NC $L $L --kappa $kap --N $N --hash; done; done
} 2>&1 | tee results_pairs.txt
