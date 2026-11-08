# Theory checks

Scripts and outputs behind the results of Appendix D: Theorem own:main (own-line tightness), Theorem md:el
and Proposition md:bounded (match nesting under EL), the latest-match remark of Section ehat:sec, and Lemma ln:conc2,
Theorem ln:thm and the dimension numbers of Section mr:sec. They import ../../e3/ownline_check.py read-only and need
no GPU. Run `./run_all.sh` from any directory (about 2 minutes, plus about 10 minutes for run_mytypes.sh).

| Script | What it checks | Output |
|---|---|---|
| construct13.py, ownline_pairs.py, run_A.sh | pairs of Theorem own:main; round types equal through L, first difference at the threshold. The script sets L to the threshold before computing the number of hubs, so its hashed runs use at least the proof's N | results_A.txt |
| mytypes.py, run_mytypes.sh | an independent implementation written from the text: its own round types and pair builder with exactly the proof's L, N and buffers | results_mytypes.txt |
| el_depth1.py | sanity check of Theorem md:el (random depth-1 formulas) and of Proposition md:bounded | results_B.txt |
| latest_match.py, construct12.py | name-only unique hard attention ("latest match") on the pairs of Theorem lb:main | results_C_latest.txt |
| dims_ln.py | dimension bounds at eps = 1/4, Monte Carlo, 32-dimensional heads, pre-LayerNorm simulation | results_D.txt |
| dims13.py | the numbers quoted for Theorem mr:thm (eps = 1/4) and Theorem ln:thm (eps = 1/8) | results_dims13.json |
| mydims.py | the independent recomputation of the eps = 1/4 numbers | stdout |
| rtypes.py, tl_eval.py | round types and TL evaluator (copies of ../rtypes.py and ../tl_eval.py) | used by the others |

The numbers quoted in the paper's appendix are these outputs.
