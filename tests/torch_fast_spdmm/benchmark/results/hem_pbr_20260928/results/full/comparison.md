# HEM ordering → existing PBR: symmetric FEM experiment

GPU: NVIDIA GeForce RTX 3090; job 21719128; status: complete.
Warmups: 5; interleaved repeats: 20. Times below are median milliseconds.

All methods multiply the same FEM operator and dense input, expressed in the appropriate coordinates. Every output is scattered back to the original order and checked against float64 SciPy SpMM (rtol=atol=1e-4). Random permutation shuffles four-DOF mesh nodes, matching `fem2d_permuted`; it does not shuffle individual DOFs.

HEM is computed once per target on the randomly permuted matrix, with feasible-first matching and equal row/column labels and caps. Stable label sorting gives `p[new]=old`, `P[i,p[i]]=1`, and `A_prime=A_permuted[p][:,p]`. The identical labels construct BOO. For a reordered multiplication, `X_prime=X_permuted[p]` and `Y_permuted[p]=Y_prime`. Group boundaries are not padded to PBR tile boundaries.

The timed region is the complete resident-input matmul call, including output allocation/zeroing and stream joins. HEM, sparse reordering, format construction, staging and dense input/output permutations are excluded. This is steady-state execution in the chosen coordinates, not end-to-end latency for a one-off multiplication.

## Fixed parameters: PBR 8×8 / min_nnz=2, HEM target=32

| N | features | seed | original PBR | random PBR | random BOO/HEM | HEM→PBR | random PBR / HEM→PBR | BOO / HEM→PBR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 128 | 0 | 0.1613 | 0.2263 | 0.2298 | 0.2241 | 1.01× | 1.03× |
| 16384 | 512 | 0 | 0.5786 | 0.8801 | 0.7567 | 0.8264 | 1.07× | 0.92× |
| 65536 | 128 | 0 | 0.5796 | 0.8556 | 0.7875 | 0.8366 | 1.02× | 0.94× |
| 65536 | 512 | 0 | 2.2753 | 3.4600 | 2.7955 | 3.2200 | 1.07× | 0.87× |
| 262144 | 128 | 0 | 2.2968 | 3.3724 | 3.0387 | 3.4539 | 0.98× | 0.88× |
| 262144 | 512 | 0 | 9.0787 | 13.8158 | 10.9286 | 13.7173 | 1.01× | 0.80× |

Ratios above 1 favor HEM→PBR.

## Best measured configurations (post-hoc tuning)

All PBR orders receive the same block/minimum sweep. Reordered PBR additionally sweeps HEM targets; BOO sweeps the same targets. These minima are tuning results, not held-out configuration selections. The full grid and IQRs remain in `summary.csv`.

| N | f | seed | original PBR | random PBR | random BOO/HEM | HEM→PBR | winning reordered config |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 16384 | 128 | 0 | 0.1603 | 0.2263 | 0.2298 | 0.2089 | hem_reordered_pbr_k32_b8_min12_block |
| 16384 | 512 | 0 | 0.5786 | 0.8786 | 0.7567 | 0.7711 | hem_reordered_pbr_k32_b8_min12_block |
| 65536 | 128 | 0 | 0.5796 | 0.8524 | 0.7875 | 0.7823 | hem_reordered_pbr_k16_b8_min12_block |
| 65536 | 512 | 0 | 2.2682 | 3.4596 | 2.7955 | 3.1042 | hem_reordered_pbr_k32_b8_min12_block |
| 262144 | 128 | 0 | 2.2815 | 3.3684 | 3.0387 | 3.1606 | hem_reordered_pbr_k32_b8_min12_block |
| 262144 | 512 | 0 | 9.0787 | 13.8158 | 10.9286 | 12.4732 | hem_reordered_pbr_k16_b8_min12_block |

## Structure at fixed PBR parameters, all HEM targets

| N | seed | case | k | blocks | compressed nnz % | CSR remainder nnz | build ms | HEM ms | sparse reorder ms |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 0 | original_pbr | 0 | 17860 | 100.00 | 0 | 22.4 | 0.0 | 0.0 |
| 16384 | 0 | permuted_pbr | 0 | 33954 | 100.00 | 0 | 27.2 | 0.0 | 0.0 |
| 16384 | 0 | permuted_boo | 8 | 16746 | 100.00 | 0 | 16.1 | 588.8 | 0.0 |
| 16384 | 0 | hem_reordered_pbr | 8 | 39112 | 99.84 | 940 | 28.9 | 588.8 | 14.9 |
| 16384 | 0 | permuted_boo | 16 | 8149 | 100.00 | 0 | 15.3 | 589.0 | 0.0 |
| 16384 | 0 | hem_reordered_pbr | 16 | 34768 | 99.90 | 572 | 27.5 | 589.0 | 14.2 |
| 16384 | 0 | permuted_boo | 32 | 4117 | 100.00 | 0 | 15.3 | 585.2 | 0.0 |
| 16384 | 0 | hem_reordered_pbr | 32 | 31482 | 99.93 | 398 | 26.2 | 585.2 | 14.3 |
| 16384 | 0 | permuted_boo | 64 | 2099 | 100.00 | 0 | 16.5 | 590.9 | 0.0 |
| 16384 | 0 | hem_reordered_pbr | 64 | 33706 | 99.99 | 62 | 27.6 | 590.9 | 14.8 |
| 65536 | 0 | original_pbr | 0 | 72580 | 100.00 | 0 | 89.0 | 0.0 | 0.0 |
| 65536 | 0 | permuted_pbr | 0 | 137602 | 100.00 | 0 | 105.6 | 0.0 | 0.0 |
| 65536 | 0 | permuted_boo | 8 | 67450 | 100.00 | 4 | 69.7 | 2245.7 | 0.0 |
| 65536 | 0 | hem_reordered_pbr | 8 | 153606 | 99.83 | 3858 | 110.7 | 2245.7 | 57.8 |
| 65536 | 0 | permuted_boo | 16 | 33313 | 100.00 | 6 | 65.8 | 2272.8 | 0.0 |
| 65536 | 0 | hem_reordered_pbr | 16 | 130658 | 99.93 | 1566 | 106.5 | 2272.8 | 60.2 |
| 65536 | 0 | permuted_boo | 32 | 16767 | 100.00 | 0 | 64.6 | 2294.5 | 0.0 |
| 65536 | 0 | hem_reordered_pbr | 32 | 124836 | 99.94 | 1362 | 105.1 | 2294.5 | 66.9 |
| 65536 | 0 | permuted_boo | 64 | 8640 | 100.00 | 0 | 63.6 | 2305.5 | 0.0 |
| 65536 | 0 | hem_reordered_pbr | 64 | 135278 | 99.89 | 2670 | 108.4 | 2305.5 | 60.1 |
| 262144 | 0 | original_pbr | 0 | 292612 | 100.00 | 0 | 307.2 | 0.0 | 0.0 |
| 262144 | 0 | permuted_pbr | 0 | 553908 | 100.00 | 0 | 347.8 | 0.0 | 0.0 |
| 262144 | 0 | permuted_boo | 8 | 271173 | 100.00 | 4 | 331.3 | 10274.2 | 0.0 |
| 262144 | 0 | hem_reordered_pbr | 8 | 664998 | 99.86 | 13544 | 378.3 | 10274.2 | 263.4 |
| 262144 | 0 | permuted_boo | 16 | 133286 | 100.00 | 0 | 301.4 | 10575.1 | 0.0 |
| 262144 | 0 | hem_reordered_pbr | 16 | 563222 | 99.91 | 8690 | 360.6 | 10575.1 | 270.3 |
| 262144 | 0 | permuted_boo | 32 | 67961 | 100.00 | 0 | 293.8 | 10647.5 | 0.0 |
| 262144 | 0 | hem_reordered_pbr | 32 | 537058 | 99.93 | 6704 | 355.6 | 10647.5 | 269.6 |
| 262144 | 0 | permuted_boo | 64 | 34789 | 100.00 | 0 | 279.8 | 10647.5 | 0.0 |
| 262144 | 0 | hem_reordered_pbr | 64 | 576822 | 99.86 | 13086 | 364.7 | 10647.5 | 270.9 |

## Controls using exactly the same HEM groups

Shared-atomic / rowseg holds the BOO object fixed and changes only the existing kernel selector. Random / reordered BOO holds group membership and the rowseg kernel fixed and changes coordinate locality. Ratios above 1 favor rowseg and reordered coordinates, respectively.

| N | f | seed | HEM k | shared-atomic / rowseg | random / reordered BOO |
|---:|---:|---:|---:|---:|---:|
| 16384 | 128 | 0 | 8 | 1.20× | 1.00× |
| 16384 | 128 | 0 | 16 | 1.31× | 1.00× |
| 16384 | 128 | 0 | 32 | 1.55× | 1.00× |
| 16384 | 128 | 0 | 64 | 2.04× | 0.99× |
| 16384 | 512 | 0 | 8 | 1.38× | 1.00× |
| 16384 | 512 | 0 | 16 | 1.46× | 1.00× |
| 16384 | 512 | 0 | 32 | 1.73× | 1.00× |
| 16384 | 512 | 0 | 64 | 2.26× | 0.99× |
| 65536 | 128 | 0 | 8 | 1.21× | 1.00× |
| 65536 | 128 | 0 | 16 | 1.32× | 1.00× |
| 65536 | 128 | 0 | 32 | 1.64× | 1.00× |
| 65536 | 128 | 0 | 64 | 2.08× | 1.00× |
| 65536 | 512 | 0 | 8 | 1.38× | 1.00× |
| 65536 | 512 | 0 | 16 | 1.45× | 1.00× |
| 65536 | 512 | 0 | 32 | 1.73× | 1.00× |
| 65536 | 512 | 0 | 64 | 2.16× | 1.02× |
| 262144 | 128 | 0 | 8 | 1.22× | 1.00× |
| 262144 | 128 | 0 | 16 | 1.32× | 1.01× |
| 262144 | 128 | 0 | 32 | 1.67× | 0.99× |
| 262144 | 128 | 0 | 64 | 2.06× | 1.03× |
| 262144 | 512 | 0 | 8 | 1.41× | 0.99× |
| 262144 | 512 | 0 | 16 | 1.47× | 1.01× |
| 262144 | 512 | 0 | 32 | 1.74× | 0.98× |
| 262144 | 512 | 0 | 64 | 2.24× | 0.95× |

## What the comparisons establish

- Original versus random PBR measures sensitivity to node order with the PBR settings held fixed.
- Random versus HEM-reordered PBR measures the benefit of this HEM-derived order using the unchanged PBR representation and kernel.
- BOO/HEM versus HEM→PBR compares two execution stacks after sharing the clustering input. Their block partitions, metadata, remainder policy and kernels still differ; this is not a pure format-only speed comparison.
- The BOO kernel control isolates the existing shared-atomic versus rowseg implementation effect on identical data. It does not isolate every difference between the PBR and BOO kernels.
- HEM groups are variable sized; stable concatenation need not recover the original mesh order or align groups with fixed PBR tiles. Failure to accelerate PBR alone does not prove the grouping is poor.
- Raw samples, all configurations, group/permutation arrays, preprocessing costs, source/extension hashes and environment metadata are retained beside this report.
