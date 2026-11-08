# part3: signal per composition step (Appendix, "Signal per Composition Step")

- `check_leap.py`: numeric checks [A]-[D] of Lemma leap:fool, Proposition leap:budget and Corollary leap:signal.
  `check_leap_output.txt` was produced with `NGRAPH=150` (the default in the script is 60); part [D] with the
  Theta rows is in `check_leap_output_D.txt`.
- `independent/type_check.py`: exact (no sampling) position-type check of Lemma leap:fool for s = 1..16; 0 violations.
- `independent/cross.py`: cross-check of Theta_{E0} on H_s and of the AL/NC/ECL constant-budget formulas.
- `independent/rerun_ngraph150.txt`: rerun of part [C] with NGRAPH=150.

Run with `python3 -I <script>`; the scripts import `../part2/tl_eval.py` and `el_depth1.py`.
