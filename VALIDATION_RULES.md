# Validation rules (V-rules)

`analysis/check_rules.py` applies the rules of groups A to F and of the preregistered sets to the imported summary
(`results/summary.json`); `analysis/real_rules.py` applies group R. `analysis/validate.py` runs both. The paper's
appendix calls the rules V-A1, V-B7, V-C35 and so on.

Notation: BA = balanced accuracy; EL, AL, NC, ECL are formats; a cell is (model or mechanism, class or task, format).
All tolerances are absolute.

Every check is reported as passed, failed or skipped. A check is skipped only when its input file is missing (at
present: the E0 clause of C15 without `pilot1/results.jsonl`); it is listed with its reason and never counted as
passed. `--verbose` prints the counts per rule.

## Three kinds of rule

- **Exact rules** hold for any correct data under their stated premises: the compared values come from the same run,
  or displayed values are compared within the stated rounding. A failure means a bug in a run, a summary script or
  the import. These are A1, B1, B14, B15, the D rules (D1-D3, D5-D7, D9-D11, D13), F2 and R1-R6, R8-R11. Fix the code,
  never the data.
- **Predictions** are the remaining B rules (theory constraints), the C rules (preregistered directions), the E rules
  (scale), D12, D14, D15 and M1. Once the implementation is confirmed against CONFIG.md, a failure is a falsified
  prediction and is reported with the reading fixed in CONFIG §6; thresholds are not changed.
- **Expectations** (R7) state what a measurement normally gives. `validate.py` prints a violation as a warning:
  check the configuration, the seeds and the code, then report it as found.

| # | Dimension | Rules | Requirement (example) | Typical cause of a failure |
|---|---|---|---|---|
| 1 | Range | A1 | accuracies in [0, 1]; chance cells in [0.47, 0.53] | Yes and No swapped in scoring; unnormalized probabilities |
| 2 | Count granularity | R1 | a BA over 300 pairs and one template is k/600; an exact-answer accuracy over 200 items is k/200 | pair or template count differs from CONFIG; lost or double-counted prompts |
| 3 | Rate identities | R2 | BA = (TPR + TNR)/2, each rate a count over its side | TPR and TNR from different samples; misaligned labels |
| 4 | Intervals | R3 | the point estimate lies in its interval; pair-bootstrap intervals are not systematically symmetric | normal approximation instead of the pair bootstrap; resampling prompts instead of pairs |
| 5 | Summaries equal their parts | R4, D5-D7 | a class cell is the mean of its three questions; a four-model mean is the mean of the models; strata weighted by their shares give the overall value; seed means and s.d. recomputed from the seeds | a missing question or model; wrong weights; means of rounded values |
| 6 | Differences | R5 | a stored decline or contrast equals the difference of its cells | the two cells come from different result files |
| 7 | Copies | R9, D1-D3 | cells of the same prompts are identical (E14 without filler = the unablated E11a cell of the model; E11b-L at Delta = 48 = E11b; E15 at 500k = E3 seeds 0-2) | a condition rerun instead of reused; a wrong file |
| 8 | Repeated runs | R6, R11 | a mean lies in [min, max] of its runs; random head sets are not identical | seeds not varied; a row overwritten |
| 9 | Common sense | R7a-R7j | a trained model is not better on larger test graphs; more data is not worse; ablating more heads is not milder; harder questions are not easier; longer filler is not better | ID and OOD files swapped; a size argument ignored; a wrong head set |
| 10 | Theory and preregistration | B, C, E, M1 | the constraints of the theorems and the preregistered directions | first exclude implementation errors; then report as falsified |
| 11 | Tests | R8, D10 | p in [0, 1]; Holm-adjusted p >= raw p and monotone in it; "equivalent" agrees with the TOST interval | adjustment reversed; families differ from CONFIG |
| 12 | Bookkeeping | F2 | every block names its CONFIG section | a block added without its configuration |

## A  Numeric form
- **A1** Every accuracy lies in [0, 1] (signed differences and test statistics excepted); chance cells lie in
  [0.47, 0.53].

## B  Theory constraints
- **B1** (Theorem 1) In the name-blind controls, a query pair is two query vertices of one graph, so the two masked
  prompts are token-identical; every name-blind model has TPR + TNR = 1 on every pair, BA **exactly 0.500** and the
  interval [0.5, 0.5]. A deviation is a bug (incomplete masking, different prompts).
- **B2** (Theorem 1) Decidable positive-control cells (X2 under AL, NC; X1 under NC) BA >= 0.70; undecidable cells (X2
  under EL; X1 under EL, AL) in [0.47, 0.53].
- **B3** (Theorem 1) The name-blind global questions are within 0.015 of the global questions with names.
- **B4** (Theorem 5, P1) E3 NoPE on the OOD sets is in [0.47, 0.53] for both tasks and all formats; ID <= 0.62.
- **B5** (Proposition 6(a)) Every mechanism other than NoPE whose round budget meets L_min(red) <= 6 reaches red ID >= 0.97.
- **B6** (round bound) 1-local on the k-controlled sets: strata with k <= k*(L) >= 0.93, strata with k > k*(L) in
  [0.47, 0.53]; k*(L) from the appendix proposition (entry k needs 2k+2 rounds under AL and 5k under NC, so
  k*_AL(6) = 2, k*_NC(6) = 1). The k-controlled sets are position-controlled (CONFIG §3.3), so of the listed
  constructions only the line-head route applies (B15).
- **B7** (round bound) Depth sweep: at fixed k, more layers never lower BA by more than 0.02 (3 seeds; not required when
  both ends are at chance); EL >= 0.95 from L = 4; ECL >= 0.95 from L = 5; every format in [0.47, 0.53] at L <= 2.
- **B8** (single-disjunct coverage at L = 3, 4) EL at L = 3 solves only the instances in which v is the first endpoint
  of some red edge, ECL at L in {3, 4} only those in which v is the second endpoint. Edge directions are uniform, so
  `kdist.kc_vfirst` and `kc_vsecond` differ by <= 0.01. Per stratum, with the measured shares `kdist.kc_by` and the
  preregistered route parameters c (accuracy on reached positives) and t (TNR), EL(L=3), ECL(L=3) and ECL(L=4) are each
  within 0.03 of 0.5 (c q + (1 - c q)(1 - t) + t); ECL at L = 3 and 4 differ by <= 0.02; EL minus ECL is within 0.02 of
  the difference of their predictions.
- **B9** (Proposition 7) Under hard since, dia under AL and NC >= 0.95 in every k stratum, largest minus smallest
  stratum <= 0.01.
- **B10** 1-local+ALiBi: dia under AL and NC does not increase with k (at the displayed three decimals) and is >= 0.88 in
  the 13-32 stratum.
- **B11** (P5) Every LLM has, on depth two, NC - max(EL, AL) >= 0.06 and |EL - AL| <= 0.05.
- **B12** (name transport) Partial masking in the LLMs: AL with masked line heads: Q0 in [0.47, 0.53] and D1a 0.01 to
  0.06 below unmasked AL; AL with masked entries: D1a in [0.47, 0.53]; NC with masked line heads: D1a below unmasked
  NC and Q0 >= 0.90; NC with masked entries: D1a within 0.045 of NC-H.
- **B13** (Theorem 1, E0b chance cells) The 6 pilot cells that the theorem calls undecidable (X2-EL, X1-EL/AL, D1a in
  three formats), rerun with seeds 0-2: every seed is within the 99% binomial range of 0.5 on 2,000 balanced test
  examples, |acc - 0.5| <= 2.576 sqrt(0.25/2000), about 0.029.
- **B14** E7 twin Q0 pairs under AL with masked line heads: the two masked prompts of a pair are token-identical
  (`build_twin.py` checks every pair), so BA and both interval ends are exactly 0.500.
- **B15** Position control: `e3/ownline_check.py` checks on the token sequences that no positive of kset and kid (AL,
  NC) has an own-line route (neither AL's j_prec nor NC's j_occ exists, and the first red neighbor in v's own list is at
  position >= 2), that the stored k_min equals the one read from the tokens, and that kown positives have k_min >= 4 and
  j_prec = 1. The number of violations must be 0.
- **B16** (P2b, direction only) On the own-line set, 1-local under AL at L = 6 (5 seeds of E3) and L = 8 (3 seeds of
  E3d) is above chance (> 0.53) and below the positives the line-head route reaches (k_min = 1 stratum of the
  k-controlled set at L = 6, of the depth sweep at L = 8); the routes available at L = 6 remain at L = 8, so the L = 8
  mean is not below the L = 6 mean minus 2 standard errors of their difference.

## C  Preregistered directions (paper appendix, "Preregistered Predictions")
- **C1** (P2) 1-local: EL and ECL dia >= 0.95 ID and OOD; AL and NC dia lie in the band between
  0.5 [c P + (1 - c P)(1 - t) + t] with P = P(k_min <= k*) (the line-head route alone) and the same expression with the
  own-line share of AL added (kdist `*_own`), 0.03 tolerance at both ends; plausible ranges AL [0.75, 0.92] (k* = 2), NC
  [0.65, 0.80] (k* = 1).
- **C2** (P3) 1-local+ALiBi and hard since: dia >= 0.95 ID in all four formats.
- **C3** (RoPE assumption) RoPE within 0.015 of ALiBi-only ID; dia OOD at least 0.05 below ID (length
  extrapolation); red OOD >= 0.93.
- **C4** (P3, LLMs) Every model has on depth one NC > AL > EL with NC - EL >= 0.05; the same direction on counting.
- **C5** (global questions) Every model's global questions differ by <= 0.03 across the three formats.
- **C6** (planned contrasts, S-1) For every model and query class, Delta = (NC - EL)_c - (NC - EL)_global > 0 with a 95%
  interval excluding 0. (AL - EL) is required positive on depth one and counting only; on depth two neither format has a
  route for every order and AL needs two favorable orders (EL one), so -0.06 <= Delta <= 0.005.
- **C7** (interaction) For every model the chi^2_6 before and after length adjustment has p < 0.001; the adjusted
  chi^2 is at most 1.02 times the unadjusted one.
- **C8** (P4) On depth one ECL >= NC - 0.01 and ECL - NC <= 0.08; in every k_min stratum ECL >= max(AL, NC) - 0.02;
  ECL's range over strata <= 0.03. EL and ECL do not depend on list position, so their change from k_min = 1 to >= 5 is
  the same (|Delta_ECL - Delta_EL| <= 0.03); NC adds the line-head decay: Delta_NC - Delta_EL >= 0.02 for 7-9B, >= 0 for
  72B; on the four-model mean |Delta_ECL - Delta_EL| < Delta_NC - Delta_EL. `r9_stats.py kdiff` gives the one-sided 95%
  lower bound of T = (Delta_NC - Delta_EL) - |Delta_ECL - Delta_EL| (reported, not a pass condition); T agrees with the
  strata within 0.003.
- **C9** (redundancy and locality) EL-2 - EL in [0, 0.05]; EL-far <= EL + 0.005; ECL - EL-far >= ECL - EL - 0.005.
- **C10** (GraphML) Within a model |GraphML - EL| <= 0.03; GraphML's decline over k strata is at most EL's + 0.03.
- **C11** (real graphs) For each source and format, the measured D1a BA is within 0.05 of the same model's and format's
  random-graph D1a strata weighted by the source's k_min distribution.
- **C12** (prompting) 3-shot keeps NC >= AL > EL; 3-shot >= zero-shot - 0.01; NC - EL under CoT is smaller than
  zero-shot.
- **C13** (robustness) Every variant keeps NC > AL > EL; the drop under integer ids on Qwen >= the drop on Llama; sorting
  EL gains <= 0.03; question first changes <= 0.02.
- **C14** (API model) GPT-4o on depth one: NC >= AL > EL, ECL >= NC - 0.01; global questions >= 0.97.
- **C15** (pilot scaling) X1-NC is monotone in width, layers, data and steps (tolerance twice the seed s.d.); the base
  cell agrees with the E0 X1-NC cell within 0.005; the largest configuration >= 0.95.
- **C16** NC-nw (non-words "dax", "Wug" for "is", "Neighbors") is within 0.03 of NC on depth one.
- **C17** (competing models, `m0/compete.py fit`) Both models set the cells the theory proves undecidable to 0.5; on the
  remaining cells the route model (route classes, Q0, distance, model effects) has a lower QAIC (dispersion from the
  route model, which contains every term of the locality model), a higher leave-one-condition-out log-likelihood and a
  smaller leave-one-condition-out error than the locality model (distance, Q0, model effects). Only the direction of
  the point estimates is required; the 95% bootstrap interval of the error difference is reported.
- **C18** (learning hypothesis) Under hard since, the log of the steps to 0.9 validation accuracy on dia, regressed on
  the number of nested matches of the construction (EL 2; AL, NC, ECL 1), has a slope whose seed-bootstrap 95% lower
  bound is > 0.
- **C19** (random masks) AL D1a under random masks of entry size is at least 0.03 above the entry mask; AL Q0 under
  random masks of line-head size is at least 0.03 above the line-head mask.
- **C20** On the positives of the k-controlled sets, for the approximate-since mechanisms (1-local+ALiBi, ALiBi-only,
  RoPE): wherever |z(log k)| >= 2 (two-variable logistic regression on log k_min and log line length),
  |z(log line length)| < 2 in the regression with one indicator per k_min value plus log line length.
- **C21** The incident encoding (AL type) is within 0.05 of AL and above EL; the friendship encoding (EL type) is within
  0.05 of EL and below AL.
- **C22** (P6, E11a) Mean ablation of the top 16 line-head heads of Qwen2.5-7B: AL and NC lose >= 0.05 each on D1a and
  >= 3 times the loss of 16 random heads (same layer distribution, mean of 5 sets); EL and ECL lose <= 0.03; Q0 loses
  <= 0.02 in all formats; the top-4 loss is at most the top-16 loss + 0.01.
- **C23** (P7, E11b) From Delta = 6 to 48, AL and NC decline more than EL and ECL in every model; the four-model
  interaction (mean decline of AL and NC minus that of EL and ECL) >= 0.03.
- **C24** (E13) Every model: on degree and neighbors max(AL, NC) >= EL + 0.05; on edge existence max(AL, NC) >= EL.
- **C25** (E3, Delta <= 256) 1-local+hard since >= 0.80 under AL and NC; 1-local at least 0.03 below its OOD value.
- **C26** (E3 hard pairs, Theorem lb:main) The six-layer 1-local arm within 0.03 of 0.5 under AL and NC;
  1-local+hard since >= 0.93.
- **C27** (instances with the proof's parameters, L = 4, 6, 8) 1-local of depth L within 0.03 of 0.5 on the instances
  of depth L (AL, NC); at L = 6 hard since >= 0.75 and >= 1-local + 0.2.
- **C28** (E11a on Llama-3.1-8B) The top-16 ablation costs AL and NC >= 0.03 on D1a and >= twice 16 random heads; EL,
  ECL <= 0.03; Q0 <= 0.02 in all formats.
- **C29** (activation patching, two models) Recolor patching of the top 16 lowers AL's D1a by at least the mean-ablation
  loss - 0.01; rename patching lowers AL by <= 0.03; 16 random heads lower AL and NC by <= 0.02 under both corruptions.
- **C30** (name matching in E3) For every arm and format the best head's error does not fall from 1-16 to 49-64 names
  (tolerance 0.005) and is <= 0.15 at 49-64 names.
- **C31** (E11a on Gemma-2-9B) The criteria of C28.
- **C32** (E11b-L, fixed length) In every model each list format declines from Delta = 6 to 48 more than each edge
  format; the interaction (mean over models) >= 0.03; each edge format declines <= 0.03.
- **C33** (E14) At F = 1024 |between - before| <= 0.03 for each of the three models, under EL and ECL.
- **C34** (E15) EL needs >= 1.3 times the examples of each of AL, NC and ECL to reach 0.95; all four formats >= 0.95 at
  500k.
- **C35-C37** (E17), **C38-C42** (E18), **C43-C49** (E19-E21), **C50-C67** (E22-E33): the criteria are given in
  CONFIG §6.11 to §6.14 with the readings of both outcomes; stored differences and means equal their cells (D1).
- **C68** (P6, E34) Under the top-128 since heads, AL and NC lose <= 0.05 on edge existence and less than a third of the
  same heads' loss on diamond red (E17), in at least 2 models. Holds: the heads carry the line-head color rather than
  which line an entry belongs to. Fails: the heads bind entries to their line in general, and since is read as binding.
- **C69** (E34) The own-line knockout (ko-own) costs AL and NC >= 0.05 on edge existence and >= 3 times the
  previous-line control (ko-other), in at least 2 models. Holds: direct attention from entries to their line head also
  carries line identity, beyond the selected heads. Fails: line identity reaches the entries by another path.
- **C70** (E34) 128 random heads cost every format <= 0.03 on edge existence; EL and ECL lose <= 0.04 under the since
  heads. If it fails, C68 and C69 are not interpreted.
- **C71** (P6, E35) The top-128 loss of AL and NC on diamond red exceeds all 100 norm-matched random sets (in all three
  models); the top-128 losses of EL, ECL and "v is red" are at most the null's 95th percentile + 0.01.
- **C72** (P8, E36) Under T2 and T3 at 128 heads, EL is above the mean of AL and NC (one-sided lower bound > 0) in at
  least 2 models per template; the losses of AL and NC are within 0.04 of T1.
- **C73** (P8, E36) The same flip on the paraphrase D1c, in at least 2 models.
- **C74** (P8, E36) Qwen2.5-72B at 640 heads (12.5%, within the 12 to 19% that 128 heads make up in the smaller models):
  EL above the list mean (lower bound > 0) and a list loss >= 3 times that of 640 random heads; 128 heads are reported
  only.

## D  Cross-consistency
- **D1** The "AL-none" and "NC-none" D1a cells of E7 equal the model's E1 D1a cells; more generally every stored
  difference, decline or contrast equals the difference of its cells, and every stored mean the mean of its parts.
- **D2** The zero-shot row of E6 and the default row of E8 equal the depth-one cells of E1.
- **D5** The share-weighted mean of the strata is within 0.01 of the overall value.
- **D6** The mean of the per-template BAs is within 0.005 of the pooled BA (equal positives and negatives per template).
- **D7** The mean and s.d. of every E3 cell are recomputed from its 5 seed values.
- **D9** The end points of the E9 patching curve: patching the embedding output restores 1, patching after the last
  block restores 0 (tolerance 0.001).
- **D10** Every TOST result is "equivalent" exactly when its 90% interval lies within the preregistered margin (format
  parity 0.03: C5, C10, C16; chance 0.03: B2; masked versus unmasked global questions 0.015: B3).
- **D11** Threshold-free paired metrics agree with BA: TPR + TNR = 2 BA (tolerance 0.002); where |BA - 0.5| > 0.05 the
  paired metric is on the same side of chance as BA.
- **D12** (prediction) On the position-controlled ID k_min strata, 1-local shows the OOD steps: AL at k <= 2 and NC at
  k = 1 >= 0.93, the other strata in [0.47, 0.55].
- **D13** On depth one, wherever two formats differ by > 0.01 in BA, the threshold-free paired metric orders them the
  same way.
- **D14** (prediction, E12) For every model and format the probe's BA at k = 1 is at least its BA at k = 24 - 0.02.
- **D15** (prediction, E12, preregistered threshold) The zero-parameter prediction's mean absolute error over the E1
  strata is <= 8 points.
- **M1** (prediction, E9) The probe's peak accuracy decreases over the distance bins, AL not above NC; the patching
  recovery decreases strictly, >= 0.85 in the first 13 layers and <= 0.2 after layer 19; the peak layer of the EL inner
  match precedes the peak layer of the since probe.

## E  Scale
- **E1** In every class and format Qwen2.5-72B >= Qwen2.5-7B (tolerance 0.005).
- **E2** The depth-one gap NC - EL is smallest for 72B.
- **E3** Mean BA orders the models 72B > Qwen-7B ~ Gemma-9B > Llama-8B, with Qwen-7B and Gemma-9B within 0.02.

## F  Bookkeeping
- **F2** Every block of `analysis/schema.json` names its CONFIG section in `config_ref`.

## R  Computability and common sense (`analysis/real_rules.py`)
The definitions are in the module docstring: R1 count granularity, R2 rates, R3 interval method, R4 pooled = mean of
parts, R5 stored differences, R6 summaries of repeated runs, R7 common sense of the measurement (a-j), R8 p-values and
Holm, R9 the same prompts give the same cell, R10 one evaluation with two summaries, R11 separate runs are not copies.
R7 is reported as warnings; the others are exact.
