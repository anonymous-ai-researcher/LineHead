#!/bin/sh
# Item (A) machine checks; writes results_A.txt
cd "$(dirname "$0")"
{
python3 -I ownline_pairs.py AL_Phi 1 5
python3 -I ownline_pairs.py NC_Phi 1 2
python3 -I ownline_pairs.py NC_Psi 1 2
python3 -I ownline_pairs.py AL_Phi 1 4 --proofparams --hash
python3 -I ownline_pairs.py NC_Phi 1 2 --proofparams --hash
python3 -I ownline_pairs.py NC_Psi 1 1 --proofparams --hash
python3 -I ownline_pairs.py AL_Phi 1 3 --kappa 2 --hash --proofparams
python3 -I ownline_pairs.py NC_Phi 1 1 --kappa 2 --hash --proofparams
python3 -I ownline_pairs.py NC_Psi 1 1 --kappa 2 --hash --proofparams
} > results_A.txt 2>&1
