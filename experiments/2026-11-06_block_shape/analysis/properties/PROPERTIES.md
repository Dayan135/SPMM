# Structural predictors of block-shape gain

CPU-only analysis of saved files. Run `conda run -n fem python scripts/research_panel/shape_properties.py`.
Only this script and this output directory are authored. No CUDA, Torch, SSH or cluster execution.
Outputs are write-once: an identical rerun verifies bytes; changed sources require `--out` to a new directory.
`FREEZE.json` records source/script hashes, the exact fitted equation and prediction hash.

## Properties and target

| Property | Exact computation |
|---|---|
| read_fraction | x_read / x_staged: fraction of staged column slots used by at least one stored entry |
| reuse_staged | compressed_nnz / x_staged: stored-entry MAC opportunities per staged X scalar |
| reuse_used | compressed_nnz / x_read: stored-entry MAC opportunities per used X scalar |
| reuse_gain | reuse_staged(shape) / reuse_staged(32x32), same matrix and method |
| x_staged_rel | x_staged(shape) / x_staged(32x32), same matrix and method |
| y_commits_rel | active block-row count(shape) / active block-row count(32x32) |
| output_fanout | y_commits / rows: active block-row incidences per global row, including empty global rows in denominator |
| imbalance | balance_gain_t256_p64_f2 = sum busiest row-residue workload / sum ceil(block_nnz/8); >=1 is worse |
| busiest_f2_rel | sum busiest row-residue workload(shape) / same at 32x32; T256/P64/FPT2 fixed |
| capacity_p64 | best_threads_sm_p64 / 1536; theoretical best over T128/256/512, not achieved occupancy |
| cta_p64_t256 | threads_sm_p64_t256 / 256: estimated CTAs/SM at fixed T256 |
| arena_p64_rel | arena_p64(shape) / arena_p64(32x32) |
| max_p99 | max_nnz_blk / nz_p99; outlier pressure from the largest block |
| remainder_fraction | remainder_nnz / total_nnz |
| remainder_delta | remainder_fraction(shape) - remainder_fraction(32x32) |

All counts concern retained BOO blocks, exclude CSR work, count stored entries (including explicit zeros), and are per feature.
`y_commits` is a potential update count: cancellation/zero sums can suppress actual atomics. Read fraction measures utilization, not overlap itself;
reuse_used and reuse_staged capture the multiplicity of nonzeros sharing X. Neither is measured DRAM traffic.
Arena capacity is an SM86 model ignoring register limits. The fixed T256/P64/FPT2 imbalance proxy does not track each tuned winner's mapping.
Zero-denominator properties remain undefined; the training/prediction path asserts finite model inputs rather than imputing them.

Join: screen_best maps input to one matrix/method; screen_table joins gate_relative on matrix/method/shape.
32x32 identity-ratio anchors are excluded. There are 98 rows, 8 inputs and two F values; all allocations are aggregated, never treated as independent data points.
Target is published best_med (median across allocation-specific best-vs-best ratios, rounded to 3 decimals), checked against screen_best within rounding.
These are filtered, post-hoc tuned screening times, not confirmation results. The untimed set is defined by matrix key absent from this shape screen,
not absence of historical square-block timings. Neither F256 nor new confirmation timing is read.

## Gate benchmark and explanations

| passes | rows | median_speedup | wins | win_fraction |
|---|---|---|---|---|
| False | 30 | 0.9125 | 2 | 0.0667 |
| True | 68 | 1.0550 | 36 | 0.5294 |

Pooled Spearman(speedup, x_staged_rel) = -0.337. This reproduces the supplied benchmark.
All correlations, their within-input medians and leave-one-input-out ranges are in correlations.csv; no row-independent p-values are reported.

## Models and validation

Each candidate is a log-speedup linear model with one or two structural properties, plus a shared F512 indicator.
Positive ratio/count properties are log-transformed; bounded fractions, remainder_delta and the gate indicator remain linear.
Each input has equal total fit/score weight. Centering/scaling use training folds only. Fixed ridge penalty 0.02 on standardized coefficients,
intercept unpenalized; no penalty tuning, threshold search, trees or matrix-name features. Gain threshold is fixed at predicted speedup >1.05.
Candidates are the 11 single properties and four X-staging pairs listed in FREEZE.json, plus null/F-only and original-gate/F baselines.
The gate regression learns effect size in training folds; original_gate_boolean preserves the published pass/fail decision.

Every outer fold leaves out all shapes/F/allocations of one input. model_loio.csv compares fixed candidates; its best ranking is exploratory
because the winner was selected on those scores. Nested selection repeats candidate selection using inner LOIO among the seven remaining inputs;
its outer predictions are the fairer estimate of this whole selection procedure. model_family_holdout.csv additionally holds natural/permuted FEM together
and all three diagonal inputs together; these sensitivity groups leave only five units, so estimates remain fragile.

| model | properties | log_mae | log_rmse | accuracy | balanced_accuracy | precision | recall | predicted_gains | actual_gains | called_gain_actual_median | called_no_gain_actual_median | pooled_x_staged_spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| capacity_p64 | capacity_p64 | 0.0964 | 0.1410 | 0.5121 | 0.4120 | 0.0000 | 0.0000 | 14 | 38 | 0.8585 | 1.0460 | -0.3369 |
| gate | passes | 0.1033 | 0.1493 | 0.4933 | 0.4034 | 0.0816 | 0.0330 | 16 | 38 | 0.9040 | 1.0460 | -0.3369 |
| x+busiest_f2_rel | x_staged_rel,busiest_f2_rel | 0.1085 | 0.1430 | 0.7321 | 0.6776 | 0.7385 | 0.4528 | 24 | 38 | 1.0915 | 0.9870 | -0.3369 |
| null |  | 0.1258 | 0.1730 | 0.6214 | 0.5000 | nan | 0.0000 | 0 | 38 | nan | 1.0285 | -0.3369 |
| nested_selection | inner LOIO selection | 0.1107 | 0.1519 | 0.4960 | 0.4166 | 0.1755 | 0.0896 | 21 | 38 | 0.9570 | 1.0470 | -0.3369 |
| nested_gain_selection | inner LOIO balanced-accuracy selection | 0.1206 | 0.1632 | 0.5826 | 0.5065 | 0.3952 | 0.1934 | 20 | 38 | 0.9725 | 1.0335 | -0.3369 |
| original_gate_boolean | published pass rule | nan | nan | 0.6540 | 0.7124 | 0.5237 | 0.9528 | 68 | 38 | 1.0550 | 0.9125 | -0.3369 |

Lowest fixed-candidate LOIO error: **capacity_p64**. Nested MAE selection changes gate log-MAE by **-7.2% reduction**.
Best gain detector among the fixed candidates: **x+busiest_f2_rel**, chosen by input-weighted balanced accuracy at the fixed >5% threshold.
Both selection objectives are reported because low prediction error need not imply useful gain detection. The classification objective was added after
the initial regression comparison exposed that distinction; this is exploratory, not a preregistered model competition.
Nested gain selection repeats balanced-accuracy selection inside each training fold. It does not remove analyst-choice or model-class selection bias.
The **published gate remains the primary rule**: its balanced accuracy exceeds the fixed two-property challenger, and the nested search is near chance.
The two-property challenger improves precision by sacrificing recall; it has not demonstrated an overall improvement over the gate.
Its full-fit equation is `log(speedup) = -0.197531` plus the transformed terms in FREEZE.json.
For the selected X/busiest pair, a 20% reduction in staging with other predictors fixed corresponds to about 6.7% higher predicted speedup;
a 10% reduction in busiest work corresponds to about 6.1%. These are fitted associations, not isolated timing effects.
Natural and permuted FEM have nearly identical structural predictors but materially different shape gains; these counts omit global address/cache locality.
Webbase's max/p99 ratio is approximately 6-12, versus about 1-2.5 elsewhere. That distinguishes it descriptively, but there is no second such input
from which a held-out predictor can learn the regime. Do not fit a special webbase exception and call it validation.
Errors are in natural-log units; exp(log-MAE) is an interpretable multiplicative error factor, not a confidence interval.
Fixed selected-model LOIO webbase forecasts call 0/16 shapes/F gains;
nested gain selection calls 10/16; observed gains are 0/16.
Per-input errors and selected models are in nested_folds.csv. A model's ability to rank the common FEM shapes is not evidence of generalizing to a power-law graph.

## Frozen out-of-sample forecasts

Exploratory challenger `x+busiest_f2_rel` is refit to the 98 rows; FREEZE.json contains a directly evaluable equation.
Every row also freezes primary_gate_prediction. A primary non-pass is abstention/no gain claim, not a forecast of a slowdown.
frozen_predictions.csv fixes all 792 available tall-shape/method/F predictions on 18 untimed matrix keys.
frozen_best_by_method.csv selects a shape separately for each method/F; frozen_matrix_shortlist.csv additionally selects a method by predicted performance.
No method was selected using future timings. Predictions compare against the SAME method's tuned 32x32, not the fastest 32x32 across methods or cuSPARSE.
Methods and shapes were not all observed in training (especially Graclus); extrapolated marginal properties and unseen methods are flagged.
The multiplicative sensitivity range uses the largest input-wise 90th percentile absolute nested residual; it is descriptive and NOT calibrated coverage.
Selection of the largest forecast adds optimism. Treat these as frozen falsifiable test candidates, not established expected speedups.

| matrix | features | method | shape | predicted_speedup | primary_gate_prediction |
|---|---|---|---|---|---|
| AMD/G2_circuit | 128 | metis | 128x64 | 1.0643 | gain candidate |
| AMD/G2_circuit | 512 | metis | 128x64 | 1.0724 | gain candidate |
| LPnetlib/lp_cre_b | 128 | hem | 256x16 | 1.0797 | no gain claim |
| LPnetlib/lp_cre_b | 512 | hem | 256x16 | 1.0880 | no gain claim |
| Schmid/thermal2 | 128 | metis | 128x64 | 1.1514 | gain candidate |
| Schmid/thermal2 | 512 | metis | 128x64 | 1.1602 | gain candidate |
| generated/diag_b0.1_n1048576 | 128 | graclus | 128x64 | 1.2854 | gain candidate |
| generated/diag_b0.1_n1048576 | 512 | graclus | 128x64 | 1.2952 | gain candidate |
| generated/diag_b0.1_n262144 | 128 | graclus | 128x64 | 1.2854 | gain candidate |
| generated/diag_b0.1_n262144 | 512 | graclus | 128x64 | 1.2952 | gain candidate |
| generated/diag_b0.3_n1048576 | 128 | graclus | 128x64 | 1.1622 | gain candidate |
| generated/diag_b0.3_n1048576 | 512 | graclus | 128x64 | 1.1710 | gain candidate |
| generated/diag_b0.3_n262144 | 128 | graclus | 128x64 | 1.1616 | gain candidate |
| generated/diag_b0.3_n262144 | 512 | graclus | 128x64 | 1.1705 | gain candidate |
| generated/diag_b1.0_n1048576 | 128 | hem | 128x16 | 1.2148 | no gain claim |
| generated/diag_b1.0_n1048576 | 512 | hem | 128x16 | 1.2240 | no gain claim |
| generated/diag_dd0.017_n1048576 | 128 | identity | 128x64 | 1.1890 | gain candidate |
| generated/diag_dd0.017_n1048576 | 512 | identity | 128x64 | 1.1981 | gain candidate |
| generated/diag_dd0.017_n262144 | 128 | identity | 128x64 | 1.1903 | gain candidate |
| generated/diag_dd0.017_n262144 | 512 | identity | 128x64 | 1.1993 | gain candidate |
| generated/diag_dd0.033_n1048576 | 128 | identity | 128x64 | 1.1488 | gain candidate |
| generated/diag_dd0.033_n1048576 | 512 | identity | 128x64 | 1.1575 | gain candidate |
| generated/diag_dd0.033_n262144 | 128 | identity | 128x64 | 1.1495 | gain candidate |
| generated/diag_dd0.033_n262144 | 512 | identity | 128x64 | 1.1583 | gain candidate |
| generated/diag_dd0.05_n1048576 | 128 | metis | 128x64 | 1.4487 | gain candidate |
| generated/diag_dd0.05_n1048576 | 512 | metis | 128x64 | 1.4597 | gain candidate |
| generated/diag_skewed_n1048576 | 128 | hem | 256x16 | 1.2359 | no gain claim |
| generated/diag_skewed_n1048576 | 512 | hem | 256x16 | 1.2453 | no gain claim |
| generated/fem2d_n1048576 | 128 | hem | 256x16 | 1.1832 | no gain claim |
| generated/fem2d_n1048576 | 512 | hem | 256x16 | 1.1922 | no gain claim |
| generated/fem2d_permuted_n1048576 | 128 | metis | 128x32 | 1.1456 | gain candidate |
| generated/fem2d_permuted_n1048576 | 512 | metis | 128x32 | 1.1543 | gain candidate |
| generated/pattern_diag_stripe_n1048576 | 128 | identity | 256x16 | 1.0032 | gain candidate |
| generated/pattern_diag_stripe_n1048576 | 512 | identity | 256x16 | 1.0108 | gain candidate |
| generated/pattern_diag_stripe_n262144 | 128 | identity | 256x16 | 1.0032 | gain candidate |
| generated/pattern_diag_stripe_n262144 | 512 | identity | 256x16 | 1.0109 | gain candidate |

## Limits and next evaluation

Only eight inputs, with related FEM/diagonal families and one labeling per input; method effects are confounded with input effects.
Gate-based selection determined much of which tall shapes were timed, so the training sample is not an unbiased shape sweep.
Two F measurements share the same structure; equal-input weighting and grouped folds reduce pseudoreplication but do not create more independent matrices.
These regressions describe best-vs-best tuning, not causal attribution. X staging, useful reuse, output fanout and imbalance can move together.
The pre-existing control exclusions are inherited; raw excluded observations are not reconstructed from aggregate tables.
A 1M sibling is a held-out scale, not a new family; noisy diagonal density changes nnz scaling with N.
For an honest future test, measure the frozen shape and frozen-method 32x32 comparator with a predeclared launch-selection protocol;
score >5% labels and log errors without reselecting shapes/methods from test timings. This forecast freezes shapes, not GPU P/T/FPT settings.
Failure to beat the gate on grouped validation is a negative result, not a reason to expand the feature search on eight inputs.
