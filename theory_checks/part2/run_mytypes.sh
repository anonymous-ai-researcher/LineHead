#!/bin/sh
# Independent check (mytypes.py, exact types, exactly the proof's L, N and buffers); writes results_mytypes.txt
cd "$(dirname "$0")"
{
for c in "a 1 1" "a 2 1" "a 3 1" "b 1 1" "b 2 1" "c 1 1" "c 2 1" "a 1 2" "a 2 2" "b 1 2" "c 1 2" "a 1 3"; do
python3 -I mytypes.py $c
done
} > results_mytypes.txt 2>&1
