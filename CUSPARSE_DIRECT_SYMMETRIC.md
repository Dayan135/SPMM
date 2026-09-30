# Symmetric direct-cuSPARSE comparison

Follow-up to [CUSPARSE_DIRECT.md](CUSPARSE_DIRECT.md), which compared direct cuSPARSE's C++-only cusparseSpMM timing against PBR/BOO/torch.sparse.mm's Python-side full-call timing -- two different measurement scopes, not an apples-to-apples comparison. This run gives every method both scopes.

Measured source commit: `5dbe7912e329dbd680d96240d60fa518f1988b25`. Source SHA256: `eb957eeb2d96bf475b68c0d37fb6f3aa1611627f1ee192187028d38a83d5665d`.

## API level (every method's Python-callable full call)

| N | Method | Speedup vs direct, median [range] |
|---:|---|---:|
| 16384 | torch | 0.606 [0.529, 0.652]x |
| 16384 | pbr | 0.703 [0.703, 0.712]x |
| 16384 | boo | 0.528 [0.511, 0.554]x |
| 65536 | torch | 0.738 [0.688, 0.745]x |
| 65536 | pbr | 0.856 [0.854, 0.858]x |
| 65536 | boo | 0.752 [0.725, 0.765]x |
| 262144 | torch | 0.738 [0.737, 0.742]x |
| 262144 | pbr | 0.875 [0.875, 0.876]x |
| 262144 | boo | 0.837 [0.834, 0.843]x |

## C++ level (direct cusparseSpMM_only vs PBRTimedSpMM/BOOTimedSpMM cpp_only -- no torch entry, that code belongs to PyTorch)

| N | Method | Speedup vs direct, median [range] |
|---:|---|---:|
| 16384 | pbr | 0.911 [0.906, 0.921]x |
| 16384 | boo | 0.817 [0.814, 0.827]x |
| 65536 | pbr | 0.911 [0.908, 0.916]x |
| 65536 | boo | 0.872 [0.868, 0.873]x |
| 262144 | pbr | 0.888 [0.887, 0.889]x |
| 262144 | boo | 0.872 [0.870, 0.872]x |

## Diagnostic: atomic/zero upper bound

**Output is incorrect by construction (Y left unzeroed, Step-C atomicAdd replaced with a plain store) -- this is not a real timing, it bounds what removing the zero+atomic cost could gain if a future redesign eliminated the need for them (e.g. one-CTA-per-row-group ownership). For BOO specifically this is a stronger claim than for PBR: BOO's rowseg atomicAdd is not purely a synchronization tax -- a row's contribution is genuinely split across multiple blocks in its row-group, so the diagnostic silently drops real cross-block accumulation, not just atomic overhead. Never compare these numbers as if they were achievable today.**

Geomean diagnostic speedup vs direct(cpp) across all 84 workloads: **PBR 14.7x** (range 1.07x-656x) --
PBR's per-call cost is overwhelmingly the zero+atomic+join tax, not compute, so a redesign that
eliminated it could plausibly beat direct cuSPARSE by a wide margin. **BOO 1.41x** (range 0.53x-5.0x,
sometimes *below* 1x) -- consistent with the atomic not being mostly removable overhead for BOO; the
upper bound is real but modest.

| Profile | N | F | PBR diag ms | PBR diag speedup vs direct(cpp) | BOO diag ms | BOO diag speedup vs direct(cpp) |
|---|---:|---:|---:|---:|---:|---:|
| diag_b0.1 | 16384 | 128 | 0.00819 | 7.625 | 0.05427 | 1.151 |
| diag_b0.3 | 16384 | 128 | 0.02355 | 3.087 | 0.10957 | 0.689 |
| diag_b1.0 | 16384 | 128 | 0.09411 | 2.192 | 0.21811 | 0.937 |
| diag_skewed | 16384 | 128 | 0.09421 | 2.214 | 0.22016 | 0.949 |
| fem2d | 16384 | 128 | 0.08550 | 1.222 | 0.19763 | 0.528 |
| fem2d_permuted | 16384 | 128 | 0.11674 | 1.066 | 0.19661 | 0.634 |
| pattern_checkerboard | 16384 | 128 | 0.00307 | 11.500 | 0.03072 | 1.133 |
| pattern_corner_cluster | 16384 | 128 | 0.01178 | 2.609 | 0.01946 | 1.572 |
| pattern_diag_stripe | 16384 | 128 | 0.00307 | 10.667 | 0.01638 | 2.000 |
| pattern_row_skewed | 16384 | 128 | 0.00307 | 11.333 | 0.02662 | 1.308 |
| scattered_1x8 | 16384 | 128 | 0.00307 | 9.667 | 0.01843 | 1.601 |
| scattered_2x2 | 16384 | 128 | 0.00307 | 9.667 | 0.01536 | 1.933 |
| scattered_2x4 | 16384 | 128 | 0.00307 | 9.667 | 0.01722 | 1.718 |
| scattered_8x1 | 16384 | 128 | 0.00307 | 10.667 | 0.01741 | 1.882 |
| diag_b0.1 | 16384 | 512 | 0.00922 | 24.222 | 0.16845 | 1.331 |
| diag_b0.3 | 16384 | 512 | 0.06605 | 3.828 | 0.35942 | 0.698 |
| diag_b1.0 | 16384 | 512 | 0.31846 | 2.514 | 0.83046 | 0.961 |
| diag_skewed | 16384 | 512 | 0.31846 | 2.511 | 0.83808 | 0.953 |
| fem2d | 16384 | 512 | 0.22528 | 1.561 | 0.65075 | 0.540 |
| fem2d_permuted | 16384 | 512 | 0.34816 | 1.237 | 0.68198 | 0.635 |
| pattern_checkerboard | 16384 | 512 | 0.00307 | 31.000 | 0.08192 | 1.181 |
| pattern_corner_cluster | 16384 | 512 | 0.02867 | 2.911 | 0.05018 | 1.633 |
| pattern_diag_stripe | 16384 | 512 | 0.00205 | 45.000 | 0.03277 | 2.825 |
| pattern_row_skewed | 16384 | 512 | 0.00307 | 36.333 | 0.06246 | 1.754 |
| scattered_1x8 | 16384 | 512 | 0.00307 | 26.667 | 0.04198 | 1.951 |
| scattered_2x2 | 16384 | 512 | 0.00307 | 24.667 | 0.03482 | 2.176 |
| scattered_2x4 | 16384 | 512 | 0.00307 | 25.833 | 0.04096 | 1.921 |
| scattered_8x1 | 16384 | 512 | 0.00307 | 27.000 | 0.04240 | 1.951 |
| diag_b0.1 | 65536 | 128 | 0.00819 | 27.125 | 0.15053 | 1.483 |
| diag_b0.3 | 65536 | 128 | 0.06963 | 3.603 | 0.28723 | 0.877 |
| diag_b1.0 | 65536 | 128 | 0.35830 | 2.233 | 0.79906 | 0.998 |
| diag_skewed | 65536 | 128 | 0.35840 | 2.255 | 0.80947 | 0.991 |
| fem2d | 65536 | 128 | 0.30202 | 1.193 | 0.66867 | 0.537 |
| fem2d_permuted | 65536 | 128 | 0.44237 | 1.233 | 0.71570 | 0.757 |
| pattern_checkerboard | 65536 | 128 | 0.00307 | 37.000 | 0.09114 | 1.233 |
| pattern_corner_cluster | 65536 | 128 | 0.02805 | 3.276 | 0.05222 | 1.755 |
| pattern_diag_stripe | 65536 | 128 | 0.00256 | 39.600 | 0.03584 | 2.829 |
| pattern_row_skewed | 65536 | 128 | 0.00307 | 39.833 | 0.07219 | 1.703 |
| scattered_1x8 | 65536 | 128 | 0.00307 | 27.667 | 0.05530 | 1.528 |
| scattered_2x2 | 65536 | 128 | 0.00307 | 24.667 | 0.04506 | 1.682 |
| scattered_2x4 | 65536 | 128 | 0.00307 | 27.333 | 0.04813 | 1.702 |
| scattered_8x1 | 65536 | 128 | 0.00256 | 32.800 | 0.05325 | 1.577 |
| diag_b0.1 | 65536 | 512 | 0.00922 | 92.556 | 0.52634 | 1.622 |
| diag_b0.3 | 65536 | 512 | 0.23347 | 4.219 | 1.08902 | 0.905 |
| diag_b1.0 | 65536 | 512 | 1.27590 | 2.476 | 3.14470 | 1.002 |
| diag_skewed | 65536 | 512 | 1.27693 | 2.389 | 3.16109 | 0.968 |
| fem2d | 65536 | 512 | 0.90317 | 1.579 | 2.51392 | 0.565 |
| fem2d_permuted | 65536 | 512 | 1.51347 | 1.795 | 2.63766 | 1.031 |
| pattern_checkerboard | 65536 | 512 | 0.00307 | 113.333 | 0.23962 | 1.469 |
| pattern_corner_cluster | 65536 | 512 | 0.09830 | 3.043 | 0.13210 | 2.279 |
| pattern_diag_stripe | 65536 | 512 | 0.00307 | 113.167 | 0.07466 | 4.569 |
| pattern_row_skewed | 65536 | 512 | 0.00307 | 137.333 | 0.19763 | 2.152 |
| scattered_1x8 | 65536 | 512 | 0.00256 | 112.200 | 0.13517 | 2.144 |
| scattered_2x2 | 65536 | 512 | 0.00307 | 79.000 | 0.09933 | 2.474 |
| scattered_2x4 | 65536 | 512 | 0.00307 | 93.333 | 0.11878 | 2.414 |
| scattered_8x1 | 65536 | 512 | 0.00307 | 91.000 | 0.12493 | 2.238 |
| diag_b0.1 | 262144 | 128 | 0.00819 | 103.625 | 0.50278 | 1.692 |
| diag_b0.3 | 262144 | 128 | 0.25704 | 3.844 | 1.05370 | 0.944 |
| diag_b1.0 | 262144 | 128 | 1.42848 | 2.174 | 3.08634 | 1.002 |
| diag_skewed | 262144 | 128 | 1.41824 | 2.071 | 3.12474 | 0.942 |
| fem2d | 262144 | 128 | 1.20157 | 1.199 | 2.63933 | 0.545 |
| fem2d_permuted | 262144 | 128 | 1.79098 | 1.162 | 2.87898 | 0.723 |
| pattern_checkerboard | 262144 | 128 | 0.00307 | 135.000 | 0.30515 | 1.341 |
| pattern_corner_cluster | 262144 | 128 | 0.09819 | 3.447 | 0.16282 | 2.079 |
| pattern_diag_stripe | 262144 | 128 | 0.00307 | 118.000 | 0.11213 | 3.233 |
| pattern_row_skewed | 262144 | 128 | 0.00307 | 146.333 | 0.23296 | 1.934 |
| scattered_1x8 | 262144 | 128 | 0.00205 | 144.500 | 0.18074 | 1.637 |
| scattered_2x2 | 262144 | 128 | 0.00256 | 96.400 | 0.14379 | 1.721 |
| scattered_2x4 | 262144 | 128 | 0.00307 | 95.333 | 0.16026 | 1.827 |
| scattered_8x1 | 262144 | 128 | 0.00307 | 91.333 | 0.17101 | 1.641 |
| diag_b0.1 | 262144 | 512 | 0.01126 | 299.227 | 1.91130 | 1.764 |
| diag_b0.3 | 262144 | 512 | 0.91341 | 4.279 | 4.20864 | 0.929 |
| diag_b1.0 | 262144 | 512 | 5.10003 | 2.541 | 12.36006 | 1.056 |
| diag_skewed | 262144 | 512 | 5.12512 | 2.341 | 12.44816 | 0.967 |
| fem2d | 262144 | 512 | 3.71251 | 1.571 | 10.23539 | 0.567 |
| fem2d_permuted | 262144 | 512 | 6.18598 | 2.017 | 10.74842 | 1.160 |
| pattern_checkerboard | 262144 | 512 | 0.00307 | 454.167 | 0.90160 | 1.537 |
| pattern_corner_cluster | 262144 | 512 | 0.36966 | 3.135 | 0.45670 | 2.539 |
| pattern_diag_stripe | 262144 | 512 | 0.00205 | 656.000 | 0.26624 | 5.023 |
| pattern_row_skewed | 262144 | 512 | 0.00307 | 550.500 | 0.73216 | 2.297 |
| scattered_1x8 | 262144 | 512 | 0.00256 | 429.200 | 0.46490 | 2.363 |
| scattered_2x2 | 262144 | 512 | 0.00205 | 447.750 | 0.32352 | 2.845 |
| scattered_2x4 | 262144 | 512 | 0.00307 | 362.000 | 0.38502 | 2.904 |
| scattered_8x1 | 262144 | 512 | 0.00205 | 518.000 | 0.45056 | 2.355 |

## GPU-busy time from nsys traces (one call per workload, per method)

Median "outside SpMM" fraction across all 84 workloads: **torch 68.5%** (range 10.0%-94.4%), **PBR
47.1%** (range 2.6%-95.4%), **BOO 55.7%** (range 4.2%-95.9%). PBR and BOO's own Python-callable API
carries substantial non-kernel overhead too, not just torch's -- consistent with the API-level vs
C++-level gap above.

| Method | Profile | N | F | Profiled event ms | SpMM kernel ms | Outside SpMM |
|---|---|---:|---:|---:|---:|---:|
| torch | diag_b0.1 | 16384 | 128 | 0.22054 | 0.04317 | 80.4% |
| torch | diag_b0.3 | 16384 | 128 | 0.23504 | 0.05504 | 76.6% |
| torch | diag_b1.0 | 16384 | 128 | 0.38490 | 0.20748 | 46.1% |
| torch | diag_skewed | 16384 | 128 | 0.39373 | 0.21248 | 46.0% |
| torch | fem2d | 16384 | 128 | 0.33910 | 0.08480 | 75.0% |
| torch | fem2d_permuted | 16384 | 128 | 0.29363 | 0.10573 | 64.0% |
| torch | pattern_checkerboard | 16384 | 128 | 0.22947 | 0.01757 | 92.3% |
| torch | pattern_corner_cluster | 16384 | 128 | 0.22886 | 0.01475 | 93.6% |
| torch | pattern_diag_stripe | 16384 | 128 | 0.23190 | 0.02202 | 90.5% |
| torch | pattern_row_skewed | 16384 | 128 | 0.21770 | 0.01872 | 91.4% |
| torch | scattered_1x8 | 16384 | 128 | 0.22358 | 0.01757 | 92.1% |
| torch | scattered_2x2 | 16384 | 128 | 0.22070 | 0.02253 | 89.8% |
| torch | scattered_2x4 | 16384 | 128 | 0.28762 | 0.01610 | 94.4% |
| torch | scattered_8x1 | 16384 | 128 | 0.22877 | 0.01616 | 92.9% |
| torch | diag_b0.1 | 16384 | 512 | 0.43603 | 0.17248 | 60.4% |
| torch | diag_b0.3 | 16384 | 512 | 0.48739 | 0.22028 | 54.8% |
| torch | diag_b1.0 | 16384 | 512 | 1.03923 | 0.83215 | 19.9% |
| torch | diag_skewed | 16384 | 512 | 1.01360 | 0.80751 | 20.3% |
| torch | fem2d | 16384 | 512 | 0.54000 | 0.33222 | 38.5% |
| torch | fem2d_permuted | 16384 | 512 | 0.61514 | 0.40751 | 33.8% |
| torch | pattern_checkerboard | 16384 | 512 | 0.27651 | 0.07257 | 73.8% |
| torch | pattern_corner_cluster | 16384 | 512 | 0.24624 | 0.03952 | 84.0% |
| torch | pattern_diag_stripe | 16384 | 512 | 0.25398 | 0.04915 | 80.6% |
| torch | pattern_row_skewed | 16384 | 512 | 0.34982 | 0.07001 | 80.0% |
| torch | scattered_1x8 | 16384 | 512 | 0.24208 | 0.03232 | 86.6% |
| torch | scattered_2x2 | 16384 | 512 | 0.23011 | 0.02541 | 89.0% |
| torch | scattered_2x4 | 16384 | 512 | 0.23901 | 0.02787 | 88.3% |
| torch | scattered_8x1 | 16384 | 512 | 0.31744 | 0.03037 | 90.4% |
| torch | diag_b0.1 | 65536 | 128 | 0.39725 | 0.17001 | 57.2% |
| torch | diag_b0.3 | 65536 | 128 | 0.42877 | 0.22025 | 48.6% |
| torch | diag_b1.0 | 65536 | 128 | 1.03507 | 0.82203 | 20.6% |
| torch | diag_skewed | 65536 | 128 | 1.04461 | 0.83240 | 20.3% |
| torch | fem2d | 65536 | 128 | 0.53910 | 0.33702 | 37.5% |
| torch | fem2d_permuted | 65536 | 128 | 0.83904 | 0.63324 | 24.5% |
| torch | pattern_checkerboard | 65536 | 128 | 0.28851 | 0.07888 | 72.7% |
| torch | pattern_corner_cluster | 65536 | 128 | 0.25466 | 0.04349 | 82.9% |
| torch | pattern_diag_stripe | 65536 | 128 | 0.32208 | 0.04966 | 84.6% |
| torch | pattern_row_skewed | 65536 | 128 | 0.28259 | 0.07037 | 75.1% |
| torch | scattered_1x8 | 65536 | 128 | 0.31725 | 0.03350 | 89.4% |
| torch | scattered_2x2 | 65536 | 128 | 0.23760 | 0.02426 | 89.8% |
| torch | scattered_2x4 | 65536 | 128 | 0.23715 | 0.02957 | 87.5% |
| torch | scattered_8x1 | 65536 | 128 | 0.24134 | 0.03136 | 87.0% |
| torch | diag_b0.1 | 65536 | 512 | 1.13094 | 0.68927 | 39.1% |
| torch | diag_b0.3 | 65536 | 512 | 1.28314 | 0.87947 | 31.5% |
| torch | diag_b1.0 | 65536 | 512 | 3.76166 | 3.34877 | 11.0% |
| torch | diag_skewed | 65536 | 512 | 3.61274 | 3.20090 | 11.4% |
| torch | fem2d | 65536 | 512 | 1.77446 | 1.36855 | 22.9% |
| torch | fem2d_permuted | 65536 | 512 | 2.96256 | 2.55403 | 13.8% |
| torch | pattern_checkerboard | 65536 | 512 | 0.71034 | 0.30403 | 57.2% |
| torch | pattern_corner_cluster | 65536 | 512 | 0.57149 | 0.15750 | 72.4% |
| torch | pattern_diag_stripe | 65536 | 512 | 0.60237 | 0.19526 | 67.6% |
| torch | pattern_row_skewed | 65536 | 512 | 0.67926 | 0.27315 | 59.8% |
| torch | scattered_1x8 | 65536 | 512 | 0.54406 | 0.12707 | 76.6% |
| torch | scattered_2x2 | 65536 | 512 | 0.48618 | 0.08016 | 83.5% |
| torch | scattered_2x4 | 65536 | 512 | 0.53334 | 0.12537 | 76.5% |
| torch | scattered_8x1 | 65536 | 512 | 0.52346 | 0.11347 | 78.3% |
| torch | diag_b0.1 | 262144 | 128 | 1.08998 | 0.68335 | 37.3% |
| torch | diag_b0.3 | 262144 | 128 | 1.29830 | 0.88149 | 32.1% |
| torch | diag_b1.0 | 262144 | 128 | 3.72064 | 3.29306 | 11.5% |
| torch | diag_skewed | 262144 | 128 | 3.51530 | 3.08616 | 12.2% |
| torch | fem2d | 262144 | 128 | 1.79398 | 1.37796 | 23.2% |
| torch | fem2d_permuted | 262144 | 128 | 3.36390 | 2.88510 | 14.2% |
| torch | pattern_checkerboard | 262144 | 128 | 0.73670 | 0.32975 | 55.2% |
| torch | pattern_corner_cluster | 262144 | 128 | 0.58205 | 0.17833 | 69.4% |
| torch | pattern_diag_stripe | 262144 | 128 | 0.65242 | 0.19843 | 69.6% |
| torch | pattern_row_skewed | 262144 | 128 | 0.69725 | 0.28640 | 58.9% |
| torch | scattered_1x8 | 262144 | 128 | 0.54160 | 0.13286 | 75.5% |
| torch | scattered_2x2 | 262144 | 128 | 0.52627 | 0.08198 | 84.4% |
| torch | scattered_2x4 | 262144 | 128 | 0.55872 | 0.13101 | 76.6% |
| torch | scattered_8x1 | 262144 | 128 | 0.52490 | 0.11657 | 77.8% |
| torch | diag_b0.1 | 262144 | 512 | 4.06768 | 2.75866 | 32.2% |
| torch | diag_b0.3 | 262144 | 512 | 4.83997 | 3.47420 | 28.2% |
| torch | diag_b1.0 | 262144 | 512 | 13.07811 | 11.69931 | 10.5% |
| torch | diag_skewed | 262144 | 512 | 12.20554 | 10.80305 | 11.5% |
| torch | fem2d | 262144 | 512 | 6.87894 | 5.55586 | 19.2% |
| torch | fem2d_permuted | 262144 | 512 | 13.17325 | 11.85202 | 10.0% |
| torch | pattern_checkerboard | 262144 | 512 | 2.56512 | 1.25217 | 51.2% |
| torch | pattern_corner_cluster | 262144 | 512 | 1.93744 | 0.62972 | 67.5% |
| torch | pattern_diag_stripe | 262144 | 512 | 2.13914 | 0.76674 | 64.2% |
| torch | pattern_row_skewed | 262144 | 512 | 2.43174 | 1.12142 | 53.9% |
| torch | scattered_1x8 | 262144 | 512 | 1.79155 | 0.48226 | 73.1% |
| torch | scattered_2x2 | 262144 | 512 | 1.61222 | 0.30521 | 81.1% |
| torch | scattered_2x4 | 262144 | 512 | 1.80310 | 0.48572 | 73.1% |
| torch | scattered_8x1 | 262144 | 512 | 1.75760 | 0.44681 | 74.6% |
| pbr | diag_b0.1 | 16384 | 128 | 0.36867 | 0.06825 | 83.8% |
| pbr | diag_b0.3 | 16384 | 128 | 0.46317 | 0.29929 | 37.3% |
| pbr | diag_b1.0 | 16384 | 128 | 0.64941 | 0.42022 | 35.3% |
| pbr | diag_skewed | 16384 | 128 | 0.60512 | 0.43558 | 29.4% |
| pbr | fem2d | 16384 | 128 | 0.39398 | 0.21475 | 45.5% |
| pbr | fem2d_permuted | 16384 | 128 | 0.38691 | 0.21596 | 44.2% |
| pbr | pattern_checkerboard | 16384 | 128 | 0.24384 | 0.03664 | 85.0% |
| pbr | pattern_corner_cluster | 16384 | 128 | 0.23232 | 0.01261 | 94.6% |
| pbr | pattern_diag_stripe | 16384 | 128 | 0.31642 | 0.02237 | 92.9% |
| pbr | pattern_row_skewed | 16384 | 128 | 0.24810 | 0.02352 | 90.5% |
| pbr | scattered_1x8 | 16384 | 128 | 0.24317 | 0.01162 | 95.2% |
| pbr | scattered_2x2 | 16384 | 128 | 0.24736 | 0.01149 | 95.4% |
| pbr | scattered_2x4 | 16384 | 128 | 0.24266 | 0.01389 | 94.3% |
| pbr | scattered_8x1 | 16384 | 128 | 0.32829 | 0.01939 | 94.1% |
| pbr | diag_b0.1 | 16384 | 512 | 0.41757 | 0.27129 | 42.3% |
| pbr | diag_b0.3 | 16384 | 512 | 1.32477 | 1.18205 | 13.2% |
| pbr | diag_b1.0 | 16384 | 512 | 1.87978 | 1.70227 | 9.4% |
| pbr | diag_skewed | 16384 | 512 | 1.90547 | 1.76314 | 9.2% |
| pbr | fem2d | 16384 | 512 | 1.07584 | 0.88968 | 17.3% |
| pbr | fem2d_permuted | 16384 | 512 | 1.06909 | 0.89505 | 16.3% |
| pbr | pattern_checkerboard | 16384 | 512 | 0.35354 | 0.13408 | 62.1% |
| pbr | pattern_corner_cluster | 16384 | 512 | 0.23453 | 0.04502 | 80.8% |
| pbr | pattern_diag_stripe | 16384 | 512 | 0.25968 | 0.08131 | 68.7% |
| pbr | pattern_row_skewed | 16384 | 512 | 0.27062 | 0.08617 | 68.2% |
| pbr | scattered_1x8 | 16384 | 512 | 0.24634 | 0.03574 | 85.5% |
| pbr | scattered_2x2 | 16384 | 512 | 0.37104 | 0.02947 | 92.1% |
| pbr | scattered_2x4 | 16384 | 512 | 0.34330 | 0.03978 | 88.4% |
| pbr | scattered_8x1 | 16384 | 512 | 0.24422 | 0.07069 | 71.1% |
| pbr | diag_b0.1 | 65536 | 128 | 0.40336 | 0.23404 | 44.2% |
| pbr | diag_b0.3 | 65536 | 128 | 1.30931 | 1.13832 | 13.6% |
| pbr | diag_b1.0 | 65536 | 128 | 1.83584 | 1.66682 | 9.2% |
| pbr | diag_skewed | 65536 | 128 | 1.94768 | 1.78467 | 8.9% |
| pbr | fem2d | 65536 | 128 | 1.03110 | 0.85736 | 16.9% |
| pbr | fem2d_permuted | 65536 | 128 | 1.04301 | 0.85826 | 17.7% |
| pbr | pattern_checkerboard | 65536 | 128 | 0.30419 | 0.13161 | 56.7% |
| pbr | pattern_corner_cluster | 65536 | 128 | 0.23309 | 0.04355 | 81.3% |
| pbr | pattern_diag_stripe | 65536 | 128 | 0.25430 | 0.07862 | 69.1% |
| pbr | pattern_row_skewed | 65536 | 128 | 0.25782 | 0.08563 | 66.8% |
| pbr | scattered_1x8 | 65536 | 128 | 0.33027 | 0.03744 | 88.7% |
| pbr | scattered_2x2 | 65536 | 128 | 0.24195 | 0.03005 | 88.7% |
| pbr | scattered_2x4 | 65536 | 128 | 0.24630 | 0.04006 | 85.6% |
| pbr | scattered_8x1 | 65536 | 128 | 0.24387 | 0.06787 | 72.2% |
| pbr | diag_b0.1 | 65536 | 512 | 1.23066 | 0.96798 | 23.7% |
| pbr | diag_b0.3 | 65536 | 512 | 4.87734 | 4.62497 | 5.9% |
| pbr | diag_b1.0 | 65536 | 512 | 7.05827 | 6.77213 | 4.1% |
| pbr | diag_skewed | 65536 | 512 | 7.51677 | 7.25705 | 3.9% |
| pbr | fem2d | 65536 | 512 | 3.86896 | 3.58093 | 7.4% |
| pbr | fem2d_permuted | 65536 | 512 | 3.86931 | 3.57981 | 7.5% |
| pbr | pattern_checkerboard | 65536 | 512 | 0.81312 | 0.52415 | 35.5% |
| pbr | pattern_corner_cluster | 65536 | 512 | 0.46189 | 0.16253 | 64.8% |
| pbr | pattern_diag_stripe | 65536 | 512 | 0.60045 | 0.31238 | 48.0% |
| pbr | pattern_row_skewed | 65536 | 512 | 0.62051 | 0.33369 | 46.2% |
| pbr | scattered_1x8 | 65536 | 512 | 0.41917 | 0.12966 | 69.1% |
| pbr | scattered_2x2 | 65536 | 512 | 0.41478 | 0.10922 | 78.4% |
| pbr | scattered_2x4 | 65536 | 512 | 0.41827 | 0.14611 | 69.3% |
| pbr | scattered_8x1 | 65536 | 512 | 0.55949 | 0.26707 | 52.3% |
| pbr | diag_b0.1 | 262144 | 128 | 1.16752 | 0.89022 | 24.5% |
| pbr | diag_b0.3 | 262144 | 128 | 4.78973 | 4.51489 | 5.9% |
| pbr | diag_b1.0 | 262144 | 128 | 6.97318 | 6.64424 | 4.7% |
| pbr | diag_skewed | 262144 | 128 | 8.04851 | 7.76584 | 3.7% |
| pbr | fem2d | 262144 | 128 | 3.71907 | 3.43328 | 7.7% |
| pbr | fem2d_permuted | 262144 | 128 | 3.74038 | 3.44112 | 8.0% |
| pbr | pattern_checkerboard | 262144 | 128 | 0.80813 | 0.51308 | 36.5% |
| pbr | pattern_corner_cluster | 262144 | 128 | 0.45222 | 0.16038 | 64.5% |
| pbr | pattern_diag_stripe | 262144 | 128 | 0.60032 | 0.30774 | 48.7% |
| pbr | pattern_row_skewed | 262144 | 128 | 0.66547 | 0.32732 | 50.8% |
| pbr | scattered_1x8 | 262144 | 128 | 0.43699 | 0.14035 | 67.9% |
| pbr | scattered_2x2 | 262144 | 128 | 0.38390 | 0.10000 | 75.2% |
| pbr | scattered_2x4 | 262144 | 128 | 0.42870 | 0.13753 | 69.1% |
| pbr | scattered_8x1 | 262144 | 128 | 0.54333 | 0.25955 | 52.2% |
| pbr | diag_b0.1 | 262144 | 512 | 4.45725 | 3.71712 | 16.8% |
| pbr | diag_b0.3 | 262144 | 512 | 17.03389 | 16.29673 | 4.5% |
| pbr | diag_b1.0 | 262144 | 512 | 24.08058 | 23.32543 | 3.1% |
| pbr | diag_skewed | 262144 | 512 | 28.87376 | 28.15816 | 2.6% |
| pbr | fem2d | 262144 | 512 | 15.07776 | 14.33202 | 4.9% |
| pbr | fem2d_permuted | 262144 | 512 | 15.13622 | 14.34328 | 5.2% |
| pbr | pattern_checkerboard | 262144 | 512 | 2.81686 | 2.06895 | 26.6% |
| pbr | pattern_corner_cluster | 262144 | 512 | 1.44365 | 0.63205 | 56.2% |
| pbr | pattern_diag_stripe | 262144 | 512 | 1.98154 | 1.23390 | 37.7% |
| pbr | pattern_row_skewed | 262144 | 512 | 2.07338 | 1.31700 | 36.5% |
| pbr | scattered_1x8 | 262144 | 512 | 1.26272 | 0.50473 | 60.0% |
| pbr | scattered_2x2 | 262144 | 512 | 1.09792 | 0.36447 | 68.5% |
| pbr | scattered_2x4 | 262144 | 512 | 1.25526 | 0.51961 | 60.4% |
| pbr | scattered_8x1 | 262144 | 512 | 1.83978 | 1.05349 | 42.7% |
| boo | diag_b0.1 | 16384 | 128 | 0.30550 | 0.06611 | 81.4% |
| boo | diag_b0.3 | 16384 | 128 | 0.41094 | 0.08973 | 78.2% |
| boo | diag_b1.0 | 16384 | 128 | 0.45462 | 0.24278 | 46.6% |
| boo | diag_skewed | 16384 | 128 | 0.45514 | 0.28643 | 46.4% |
| boo | fem2d | 16384 | 128 | 0.40874 | 0.19516 | 52.3% |
| boo | fem2d_permuted | 16384 | 128 | 0.40989 | 0.20278 | 50.5% |
| boo | pattern_checkerboard | 16384 | 128 | 0.30022 | 0.02774 | 90.8% |
| boo | pattern_corner_cluster | 16384 | 128 | 0.28387 | 0.01840 | 93.5% |
| boo | pattern_diag_stripe | 16384 | 128 | 0.29347 | 0.02662 | 90.9% |
| boo | pattern_row_skewed | 16384 | 128 | 0.30400 | 0.03392 | 91.4% |
| boo | scattered_1x8 | 16384 | 128 | 0.28387 | 0.01702 | 94.0% |
| boo | scattered_2x2 | 16384 | 128 | 0.29344 | 0.01338 | 95.4% |
| boo | scattered_2x4 | 16384 | 128 | 0.39331 | 0.01603 | 95.9% |
| boo | scattered_8x1 | 16384 | 128 | 0.27690 | 0.01574 | 94.3% |
| boo | diag_b0.1 | 16384 | 512 | 0.41507 | 0.21331 | 52.8% |
| boo | diag_b0.3 | 16384 | 512 | 0.55731 | 0.34332 | 38.4% |
| boo | diag_b1.0 | 16384 | 512 | 1.15741 | 0.94635 | 18.2% |
| boo | diag_skewed | 16384 | 512 | 1.14554 | 1.05809 | 18.7% |
| boo | fem2d | 16384 | 512 | 0.94813 | 0.73659 | 22.3% |
| boo | fem2d_permuted | 16384 | 512 | 0.94592 | 0.73445 | 22.4% |
| boo | pattern_checkerboard | 16384 | 512 | 0.41869 | 0.08665 | 79.3% |
| boo | pattern_corner_cluster | 16384 | 512 | 0.34672 | 0.06105 | 82.4% |
| boo | pattern_diag_stripe | 16384 | 512 | 0.30067 | 0.12304 | 76.6% |
| boo | pattern_row_skewed | 16384 | 512 | 0.30918 | 0.11526 | 70.3% |
| boo | scattered_1x8 | 16384 | 512 | 0.28291 | 0.04496 | 84.1% |
| boo | scattered_2x2 | 16384 | 512 | 0.39504 | 0.03290 | 91.7% |
| boo | scattered_2x4 | 16384 | 512 | 0.27962 | 0.04470 | 84.0% |
| boo | scattered_8x1 | 16384 | 512 | 0.28080 | 0.05088 | 81.9% |
| boo | diag_b0.1 | 65536 | 128 | 0.46986 | 0.20457 | 60.8% |
| boo | diag_b0.3 | 65536 | 128 | 0.60714 | 0.31478 | 48.2% |
| boo | diag_b1.0 | 65536 | 128 | 1.16877 | 0.89029 | 23.8% |
| boo | diag_skewed | 65536 | 128 | 1.18326 | 1.04350 | 17.9% |
| boo | fem2d | 65536 | 128 | 0.94973 | 0.72863 | 23.3% |
| boo | fem2d_permuted | 65536 | 128 | 0.98346 | 0.76619 | 22.1% |
| boo | pattern_checkerboard | 65536 | 128 | 0.30323 | 0.08963 | 70.4% |
| boo | pattern_corner_cluster | 65536 | 128 | 0.39418 | 0.04777 | 87.9% |
| boo | pattern_diag_stripe | 65536 | 128 | 0.30720 | 0.09011 | 74.6% |
| boo | pattern_row_skewed | 65536 | 128 | 0.29334 | 0.09110 | 73.2% |
| boo | scattered_1x8 | 65536 | 128 | 0.30643 | 0.05312 | 82.7% |
| boo | scattered_2x2 | 65536 | 128 | 0.32582 | 0.04195 | 87.1% |
| boo | scattered_2x4 | 65536 | 128 | 0.27334 | 0.04496 | 83.6% |
| boo | scattered_8x1 | 65536 | 128 | 0.27821 | 0.05968 | 78.5% |
| boo | diag_b0.1 | 65536 | 512 | 0.98518 | 0.70972 | 33.2% |
| boo | diag_b0.3 | 65536 | 512 | 1.52790 | 1.20004 | 21.5% |
| boo | diag_b1.0 | 65536 | 512 | 3.77206 | 3.43725 | 8.9% |
| boo | diag_skewed | 65536 | 512 | 4.10966 | 4.19343 | 7.9% |
| boo | fem2d | 65536 | 512 | 3.11418 | 2.71304 | 12.9% |
| boo | fem2d_permuted | 65536 | 512 | 3.19030 | 2.80418 | 12.1% |
| boo | pattern_checkerboard | 65536 | 512 | 0.59882 | 0.26623 | 55.5% |
| boo | pattern_corner_cluster | 65536 | 512 | 0.48086 | 0.14317 | 70.2% |
| boo | pattern_diag_stripe | 65536 | 512 | 0.59939 | 0.31711 | 55.7% |
| boo | pattern_row_skewed | 65536 | 512 | 0.58944 | 0.29139 | 56.5% |
| boo | scattered_1x8 | 65536 | 512 | 0.47264 | 0.14320 | 69.7% |
| boo | scattered_2x2 | 65536 | 512 | 0.43549 | 0.10013 | 77.0% |
| boo | scattered_2x4 | 65536 | 512 | 0.52333 | 0.12243 | 76.6% |
| boo | scattered_8x1 | 65536 | 512 | 0.56125 | 0.17795 | 68.3% |
| boo | diag_b0.1 | 262144 | 128 | 1.05190 | 0.74437 | 31.2% |
| boo | diag_b0.3 | 262144 | 128 | 1.53805 | 1.20817 | 21.4% |
| boo | diag_b1.0 | 262144 | 128 | 3.87478 | 3.47715 | 10.3% |
| boo | diag_skewed | 262144 | 128 | 4.88998 | 4.63859 | 6.7% |
| boo | fem2d | 262144 | 128 | 3.25926 | 2.86379 | 12.1% |
| boo | fem2d_permuted | 262144 | 128 | 3.37149 | 3.03633 | 9.9% |
| boo | pattern_checkerboard | 262144 | 128 | 0.66237 | 0.33740 | 49.1% |
| boo | pattern_corner_cluster | 262144 | 128 | 0.50496 | 0.17344 | 65.7% |
| boo | pattern_diag_stripe | 262144 | 128 | 0.63923 | 0.31311 | 52.9% |
| boo | pattern_row_skewed | 262144 | 128 | 0.63373 | 0.31862 | 51.7% |
| boo | scattered_1x8 | 262144 | 128 | 0.52586 | 0.19123 | 63.6% |
| boo | scattered_2x2 | 262144 | 128 | 0.48221 | 0.15737 | 67.4% |
| boo | scattered_2x4 | 262144 | 128 | 0.56624 | 0.16396 | 71.0% |
| boo | scattered_8x1 | 262144 | 128 | 0.58704 | 0.26051 | 55.7% |
| boo | diag_b0.1 | 262144 | 512 | 3.36557 | 2.63566 | 23.8% |
| boo | diag_b0.3 | 262144 | 512 | 5.46941 | 4.67815 | 14.5% |
| boo | diag_b1.0 | 262144 | 512 | 14.33613 | 13.48188 | 6.0% |
| boo | diag_skewed | 262144 | 512 | 18.75786 | 18.40511 | 4.2% |
| boo | fem2d | 262144 | 512 | 11.57629 | 10.79257 | 6.8% |
| boo | fem2d_permuted | 262144 | 512 | 11.84080 | 11.04210 | 6.7% |
| boo | pattern_checkerboard | 262144 | 512 | 1.83242 | 1.02145 | 44.3% |
| boo | pattern_corner_cluster | 262144 | 512 | 1.38243 | 0.52844 | 61.8% |
| boo | pattern_diag_stripe | 262144 | 512 | 1.83146 | 1.08801 | 43.2% |
| boo | pattern_row_skewed | 262144 | 512 | 1.79216 | 1.04779 | 43.9% |
| boo | scattered_1x8 | 262144 | 512 | 1.36432 | 0.51807 | 62.0% |
| boo | scattered_2x2 | 262144 | 512 | 1.16202 | 0.37510 | 67.7% |
| boo | scattered_2x4 | 262144 | 512 | 1.23168 | 0.44287 | 64.0% |
| boo | scattered_8x1 | 262144 | 512 | 1.58141 | 0.75176 | 53.7% |

## Artifacts

- [All configurations, all three jobs](CUSPARSE_DIRECT_SYMMETRIC_all_configs.csv).
- [API-level workloads](CUSPARSE_DIRECT_SYMMETRIC_api_level_workloads.csv), [C++-level workloads](CUSPARSE_DIRECT_SYMMETRIC_cpp_level_workloads.csv), [diagnostic](CUSPARSE_DIRECT_SYMMETRIC_diagnostic_workloads.csv).
- [API-level geomeans](CUSPARSE_DIRECT_SYMMETRIC_api_level_geomeans.csv), [C++-level geomeans](CUSPARSE_DIRECT_SYMMETRIC_cpp_level_geomeans.csv).
- Per-method GPU-busy traces: [torch](CUSPARSE_DIRECT_SYMMETRIC_profiles_torch.csv), [pbr](CUSPARSE_DIRECT_SYMMETRIC_profiles_pbr.csv), [boo](CUSPARSE_DIRECT_SYMMETRIC_profiles_boo.csv).
- [Job metadata](CUSPARSE_DIRECT_SYMMETRIC_metadata.json).
