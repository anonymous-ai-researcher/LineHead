# Theory checks

Scripts behind the machine checks of Appendix D (Theorem 30, Propositions 28 and 29, the parameters of
Theorem 46, and the realizability of M in Appendix D.17), with their outputs (results_*.txt). They import e3/ and m0/
of this repository and need no GPU. Table 4 of the paper lists every check.

| Script | What it checks | Output |
|---|---|---|
| tl_eval.py | TL[P,Y,S,M] evaluator with round depth | used by the others |
| rtypes.py, test_rtypes.py | round types (exact or hashed) and their soundness against tl_eval.py | test output |
| construct.py, check_pairs.py, run_all.sh | the pairs of Theorem 30; equal round types at the final position | results_pairs.txt, results_proofparams.txt |
| small_pairs.py | smallest hard pairs found in a bounded grid of N and buffer sizes | results_small.txt |
| top_placement_check.py | Proposition 28 on the top placement of the line heads | results_top.txt |
| selfmatch_formula.py | Proposition 29 | results_selfmatch.txt |
| fresh_and_dr.py | unshared entries; decoupled depth and offsets | results_fresh_dr.txt |
| lower_bound_numbers.py | parameters of the counterexample of Theorem 46 | results_lower_bound.txt |
| m_attention_sim.py | simulation of the soft-attention construction for M | results_msim.txt |
| diag.py | first separating component of a pair (diagnostics) | - |

The checks of part2/ and part3/ are described in their own README files.

Run each with `python3 -I <script>` from this directory. top_placement_check.py builds its pairs with the top
placement of the line heads (an option of gen_e3.pairs); with the default placement it does not reproduce
results_top.txt.
