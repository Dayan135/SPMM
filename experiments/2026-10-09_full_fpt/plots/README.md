# Full FPT sweep analysis

Complete-call timings. All 104 GPU tasks completed. 1,836 measured BOO combinations plus 36 explicitly reused duplicate-label combinations. No extra measurements are inferred for those duplicates. Selection is post-hoc; allocations were not independently replicated.

## Plots

- [01_speedup_overview](01_speedup_overview.png) (also PDF)
- [02_paired_fpt](02_paired_fpt.png) (also PDF)
- [03_clustering](03_clustering.png) (also PDF)
- [04_group_size](04_group_size.png) (also PDF)
- [05_structure](05_structure.png) (also PDF)
- [06_size_scaling](06_size_scaling.png) (also PDF)

## Baseline sensitivity

| Comparison | Winning matrix / F cells | Total |
|---|---:|---:|
| Best BOO vs same-cell direct | 30 | 78 |
| Same-cell direct, all four algorithms available | 28 | 71 |
| Best BOO vs fastest direct observed across all label shards | 29 | 78 |

The pooled comparison is a sensitivity check across allocations, not a paired measurement. Missing algorithms can make apparent wins optimistic. See best_configs.csv for both ratios.

## Paired FPT behavior

| Family | F | FPT | Cells | Geomean speedup | Worst speedup | >5% regressions |
|---|---:|---:|---:|---:|---:|---:|
| real | 32 | 2 | 48 | 1.073 | 1.015 | 0 |
| real | 32 | 4 | 48 | 1.107 | 0.934 | 1 |
| real | 128 | 2 | 48 | 1.103 | 1.019 | 0 |
| real | 128 | 4 | 48 | 1.135 | 0.899 | 2 |
| real | 512 | 2 | 48 | 1.122 | 1.017 | 0 |
| real | 512 | 4 | 48 | 1.141 | 0.884 | 3 |
| generated | 32 | 2 | 156 | 1.149 | 1.005 | 0 |
| generated | 32 | 4 | 156 | 1.201 | 0.955 | 0 |
| generated | 128 | 2 | 156 | 1.212 | 0.997 | 0 |
| generated | 128 | 4 | 156 | 1.264 | 0.871 | 8 |
| generated | 512 | 2 | 156 | 1.232 | 1.000 | 0 |
| generated | 512 | 4 | 156 | 1.280 | 0.843 | 15 |

Ratios below one are regressions. Geomeans weight configurations equally, not by runtime or application frequency.

## Best configuration at F=512

| Matrix | Method | k | FPT | Same-cell speedup | Pooled-baseline speedup |
|---|---|---:|---:|---:|---:|
| AMD/G2_circuit | identity | 32 | 2 | 0.564 | 0.560 |
| Boeing/pwtk | hem | 32 | 4 | 1.223 | 1.205 |
| LPnetlib/lp_cre_b | identity | 32 | 4 | 0.576 | 0.575 |
| SNAP/roadNet-CA | metis | 64 | 2 | 0.823 | 0.822 |
| Schmid/thermal2 | hem | 32 | 2 | 0.688 | 0.628 |
| Williams/webbase-1M | identity | 32 | 2 | 0.506 | 0.492 |
| diag_b0.1 · 1M | identity | 64 | 2 | 1.286 | 1.285 |
| diag_b0.1 · 262k | identity | 64 | 2 | 1.258 | 1.256 |
| diag_b0.3 · 1M | graclus | 64 | 4 | 1.293 | 1.293 |
| diag_b0.3 · 262k | identity | 64 | 4 | 1.262 | 1.262 |
| diag_b1.0 · 1M | metis | 32 | 4 | 2.701 | 2.574 |
| diag_b1.0 · 262k | identity | 32 | 4 | 2.614 | 2.594 |
| diag_dd0.017 · 1M | graclus | 32 | 2 | 0.808 | 0.807 |
| diag_dd0.017 · 262k | graclus | 64 | 2 | 0.781 | 0.775 |
| diag_dd0.033 · 1M | identity | 64 | 2 | 0.909 | 0.909 |
| diag_dd0.033 · 262k | identity | 64 | 2 | 0.825 | 0.822 |
| diag_dd0.05 · 1M | identity | 64 | 2 | 1.027 | 1.026 |
| diag_dd0.05 · 262k | identity | 64 | 2 | 0.970 | 0.967 |
| diag_skewed · 1M | identity | 32 | 4 | 1.032 | 1.027 |
| diag_skewed · 262k | identity | 32 | 4 | 1.279 | 1.271 |
| fem2d · 1M | graclus | 32 | 4 | 0.872 | 0.828 |
| fem2d · 262k | hem | 32 | 4 | 0.820 | 0.787 |
| fem2d_permuted · 1M | hem | 32 | 4 | 1.620 | 1.620 |
| fem2d_permuted · 262k | hem | 32 | 4 | 1.555 | 1.526 |
| pattern_diag_stripe · 1M | hem | 32 | 2 | 0.835 | 0.834 |
| pattern_diag_stripe · 262k | graclus | 32 | 2 | 0.793 | 0.790 |
