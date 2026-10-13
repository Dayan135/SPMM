# Reordering versus BOO: E1 and replicated controls

The report tables are in the
[dated result directory](../../tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928/derived).
Per-call samples and saved hierarchy/permutation arrays are in the separate
[result archives](../../RESULTS_ARCHIVE.md).

## Provenance and protocol

Historical E1: Slurm job `21719128`, GPU node `cs-3090-06` (verified with `sacct`), NVIDIA RTX 3090, float32 values, int32 sparse indices, seed 0. N = 16,384 / 65,536 / 262,144; F = 128 / 512. Source base commit: `4492e8c4a5893a776884ca97f0d08a8d45210233`, plus the then-uncommitted benchmark and tests. Measured source SHA-256: `13a7fae70a1c0b1f744e08ef38164c17ac071ffd70457911da0328ecff877d3a`; extension SHA-256: `c10ff765360004030f1f00a2bcaff954a2c4b4b396aebfbb920c4efcfec756da`. Historical metadata retained `unknown` Git fields because the uploaded snapshot had no `.git`; the base commit alone does not identify that benchmark snapshot. This branch commits the harness and evidence.

Raw per-configuration numbers: [REORDER_VS_BOO_E1.csv](../../tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928/derived/REORDER_VS_BOO_E1.csv). All 7,200 per-call samples are in the [root sample archive](../../RESULTS_ARCHIVE.md). Original metadata, saved permutations and logs: [run directory](../../tests/torch_fast_spdmm/benchmark/results/hem_pbr_20260928).

The [exact uploaded E1 source archive](../../tests/torch_fast_spdmm/benchmark/results/hem_pbr_20260928/measured_source.tar.gz) is retained. Recomputing the source fingerprint from its contents exactly matches the measured `13a7fa…` hash above; this preserves the uncommitted source actually used, rather than relying on the base commit alone.

The historical metadata helper's inherited baseline description mentions cuSPARSE; no cuSPARSE arm was measured in E1. The raw evidence is retained unchanged. HEM uses topk=32, hub_degree=1024, exact support recomputation, three residual matching rounds per level, and no max_nnz cap.

Timing T: median of 20 individually CUDA-event-timed complete matmul calls after five warmups; seeded rotating interleaving in one process. Includes output initialization, dispatch gaps and stream joins. Inputs are GPU-resident. Excludes clustering, sparse construction, staging, and dense permutations. A–D use the same underlying A and X, transformed consistently.

PBR sweep: kernel `block`, square block sizes/min_nnz pairs `2:2 2:3 4:2 4:4 4:8 8:2 8:8 8:12`, smem tier 2048, csr_mode=0. BOO: method HEM, target=max size k in `8 16 32 64`, equal row/column labels, feasible_first=True, threads=256, p=64, rowseg=True, rowmajor, legacy launch, remainder threshold=4, csr_mode=0. D sweeps the same k values and PBR configurations. The table selects the best median per arm after the sweep; all configurations remain in CSV. Such post-hoc minima are not held-out results.

## Permutations

`fem2d_permuted` shuffles mesh nodes with `default_rng(seed).permutation(N//4)` and expands each node to its four consecutive DOFs: `q=(4*node_perm[:,None]+arange(4)).ravel()`. It is not a DOF-level random permutation.

E1 used **group-label-only ordering**, with stable original vertex ID order inside groups: `p=argsort(labels, kind="stable")`. It did not record or use the matching hierarchy. `P[i,p[i]]=1`, `A_prime=A_permuted[p][:,p]`, `X_prime=X_permuted[p]`. Results are restored with `Y0[q[p]]=Y_prime` before comparison against float64 SciPy `A0 @ X0` (rtol=atol=1e-4). BOO receives exactly those HEM labels. No group padding or tile alignment is applied.

Group sizes (min / median / max), before any coordinate permutation:

| N | k=8 | k=16 | k=32 | k=64 |
|---:|:---|:---|:---|:---|
| 16384 | 2 / 8 / 8 | 3 / 16 / 16 | 4 / 28 / 32 | 4 / 56 / 64 |
| 65536 | 2 / 8 / 8 | 4 / 16 / 16 | 4 / 28 / 32 | 12 / 56 / 64 |
| 262144 | 3 / 8 / 8 | 4 / 16 / 16 | 4 / 28 / 32 | 4 / 56 / 64 |

## E1 best per arm

Arm A = natural FEM + PBR; B = node-permuted FEM + PBR; C = node-permuted FEM + BOO/HEM; D = HEM group-label-reordered B + PBR. In the config column `b/min` means square PBR tile size/min_nnz; k denotes HEM target. All other exact settings are fixed above. `—` is inapplicable; `NR` means not recorded in E1, not zero. BOO x_loads and occupancy were subsequently reconstructed from the exact saved labels (matrix hashes and block counts checked); dense permutation timings require new measurements and cannot be recovered from old timings.

| N | F | arm | config | time ms | timing | PBR fraction / blocks | BOO blocks / x_loads / occupancy | reference check / max abs error |
|---:|---:|:---|:---|---:|:---|:---|:---|:---|
| 16384 | 128 | A | k=0, b/min=8/12 | 0.160256 | T | 1.000000 / 17860 | — | passed / 4.05e-06 |
| 16384 | 128 | B | k=0, b/min=8/2 | 0.226304 | T | 1.000000 / 33954 | — | passed / 2.3e-06 |
| 16384 | 128 | C | k=32, HEM, 256 threads, p=64 | 0.229824 | T | — | 4117 / 115244 / 100.0% est. | passed / 4.93e-06 |
| 16384 | 128 | D | k=32, b/min=8/12 | 0.208880 | T | 0.888923 / 20338 | — | passed / 3.73e-06 |
| 16384 | 512 | A | k=0, b/min=8/2 | 0.578560 | T | 1.000000 / 17860 | — | passed / 4.49e-06 |
| 16384 | 512 | B | k=0, b/min=8/12 | 0.878560 | T | 1.000000 / 33954 | — | passed / 4.91e-06 |
| 16384 | 512 | C | k=32, HEM, 256 threads, p=64 | 0.756736 | T | — | 4117 / 115244 / 100.0% est. | passed / 4.93e-06 |
| 16384 | 512 | D | k=32, b/min=8/12 | 0.771072 | T | 0.888923 / 20338 | — | passed / 4.58e-06 |
| 65536 | 128 | A | k=0, b/min=8/2 | 0.579584 | T | 1.000000 / 72580 | — | passed / 3.77e-06 |
| 65536 | 128 | B | k=0, b/min=8/12 | 0.852352 | T | 1.000000 / 137602 | — | passed / 2.91e-06 |
| 65536 | 128 | C | k=32, HEM, 256 threads, p=64 | 0.787456 | T | — | 16767 / 473586 / 100.0% est. | passed / 5.8e-06 |
| 65536 | 128 | D | k=16, b/min=8/12 | 0.782336 | T | 0.887448 / 85056 | — | passed / 4.51e-06 |
| 65536 | 512 | A | k=0, b/min=8/8 | 2.268160 | T | 1.000000 / 72580 | — | passed / 5.71e-06 |
| 65536 | 512 | B | k=0, b/min=8/12 | 3.459584 | T | 1.000000 / 137602 | — | passed / 3.87e-06 |
| 65536 | 512 | C | k=32, HEM, 256 threads, p=64 | 2.795520 | T | — | 16767 / 473586 / 100.0% est. | passed / 7.77e-06 |
| 65536 | 512 | D | k=32, b/min=8/12 | 3.104192 | T | 0.901605 / 84324 | — | passed / 5.21e-06 |
| 262144 | 128 | A | k=0, b/min=8/8 | 2.281472 | T | 1.000000 / 292612 | — | passed / 4.32e-06 |
| 262144 | 128 | B | k=0, b/min=8/12 | 3.368448 | T | 1.000000 / 553908 | — | passed / 3.38e-06 |
| 262144 | 128 | C | k=32, HEM, 256 threads, p=64 | 3.038720 | T | — | 67961 / 1916328 / 100.0% est. | passed / 6.41e-06 |
| 262144 | 128 | D | k=32, b/min=8/12 | 3.160576 | T | 0.873301 / 329328 | — | passed / 5.16e-06 |
| 262144 | 512 | A | k=0, b/min=8/2 | 9.078736 | T | 1.000000 / 292612 | — | passed / 4.66e-06 |
| 262144 | 512 | B | k=0, b/min=8/2 | 13.815808 | T | 1.000000 / 553908 | — | passed / 5.15e-06 |
| 262144 | 512 | C | k=32, HEM, 256 threads, p=64 | 10.928592 | T | — | 67961 / 1916328 / 100.0% est. | passed / 8.46e-06 |
| 262144 | 512 | D | k=16, b/min=8/12 | 12.473200 | T | 0.858808 / 330874 | — | passed / 5.6e-06 |

## Preprocessing, separate from matmul

CPU timings are single wall-clock measurements, not CUDA-event timings. HEM is shared by C and D for a given k. The D sparse-reorder measurement includes symmetry validation and CSR canonicalization. `NR` marks work that E1 did not time. Conversion excludes HEM. Staging is separate in the raw CSV.

| N | F | arm | cluster ms | build PAPᵀ ms | convert PBR/BOO ms | permute X once ms | restore Y once ms |
|---:|---:|:---|---:|---:|---:|:---|:---|
| 16384 | 128 | A | 0.000 | 0.000 | 22.639 | — | — |
| 16384 | 128 | B | 0.000 | 0.000 | 27.154 | NR | NR |
| 16384 | 128 | C | 585.196 | 0.000 | 15.256 | NR | NR |
| 16384 | 128 | D | 585.196 | 14.318 | 30.594 | NR | NR |
| 16384 | 512 | A | 0.000 | 0.000 | 22.397 | — | — |
| 16384 | 512 | B | 0.000 | 0.000 | 27.312 | NR | NR |
| 16384 | 512 | C | 585.196 | 0.000 | 15.256 | NR | NR |
| 16384 | 512 | D | 585.196 | 14.318 | 30.594 | NR | NR |
| 65536 | 128 | A | 0.000 | 0.000 | 88.993 | — | — |
| 65536 | 128 | B | 0.000 | 0.000 | 105.280 | NR | NR |
| 65536 | 128 | C | 2294.515 | 0.000 | 64.568 | NR | NR |
| 65536 | 128 | D | 2272.829 | 60.162 | 122.099 | NR | NR |
| 65536 | 512 | A | 0.000 | 0.000 | 87.857 | — | — |
| 65536 | 512 | B | 0.000 | 0.000 | 105.280 | NR | NR |
| 65536 | 512 | C | 2294.515 | 0.000 | 64.568 | NR | NR |
| 65536 | 512 | D | 2294.515 | 66.871 | 120.003 | NR | NR |
| 262144 | 128 | A | 0.000 | 0.000 | 303.991 | — | — |
| 262144 | 128 | B | 0.000 | 0.000 | 346.225 | NR | NR |
| 262144 | 128 | C | 10647.537 | 0.000 | 293.810 | NR | NR |
| 262144 | 128 | D | 10647.537 | 269.649 | 414.828 | NR | NR |
| 262144 | 512 | A | 0.000 | 0.000 | 307.201 | — | — |
| 262144 | 512 | B | 0.000 | 0.000 | 347.761 | NR | NR |
| 262144 | 512 | C | 10647.537 | 0.000 | 293.810 | NR | NR |
| 262144 | 512 | D | 10575.098 | 270.293 | 424.499 | NR | NR |

## Ratios and restrained reading

Ratios are time ratios, not automatically speedups. B/A and D/A > 1 mean slower than natural PBR; C/D < 1 means C has the lower measured time.

| N | F | B/A | D/A | C/D |
|---:|---:|---:|---:|---:|
| 16384 | 128 | 1.412 | 1.303 | 1.100 |
| 16384 | 512 | 1.519 | 1.333 | 0.981 |
| 65536 | 128 | 1.471 | 1.350 | 1.007 |
| 65536 | 512 | 1.525 | 1.369 | 0.901 |
| 262144 | 128 | 1.476 | 1.385 | 0.961 |
| 262144 | 512 | 1.522 | 1.374 | 0.876 |

This one session shows a ≥1.3× ordering penalty relative to A for most B/D cells. C and D are within 1.3× in these selected results; **E1 does not establish a performance difference between them**. No difference below roughly 1.3× is claimed from this single session. Representation and kernel effects remain coupled in C versus D. The three-job replication follows below.


E1 structural reconstruction is in [REORDER_VS_BOO_E1_STRUCTURE.csv](../../tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928/derived/REORDER_VS_BOO_E1_STRUCTURE.csv). It adds deterministic counters, not new E1 timing measurements. Historical X/Y permutation timings remain NR.



## Expanded experiment: three separate Slurm jobs

All arms below use N=65,536 and 262,144, F=128 and 512, float32 values, int32 sparse metadata, seed 0. The jobs use separate GPU allocations; the remaining jobs were released to run in parallel after the first job completed the shared build and tests, with its node excluded. Each job uses protocol T (five warmups, 20 interleaved complete-call CUDA-event timings), the same configuration sweep as E1, and the same A/X hashes. This repeats execution across allocations, not across random input seeds.

Measured source commit: `84fea8f3775cf4f1411bd085d4a4c4fc995a9e9c`. Source SHA-256: `832e0d25043c14d1c728c0df8bca88574adeef8e7b2257fa4f5d3577ce61f528`; extension SHA-256: `cb25a650873546fab37098731915eae105d1a424a4b63ca2f3074ff1f8b63f13`. All three match. E1 evidence commit: `cd820ad`.

| replicate | Slurm job | node | GPU | completed workload cells |
|---:|:---|:---|:---|---:|
| 1 | 21725373 | cs-pheno-09.auth.ad.bgu.ac.il | NVIDIA GeForce RTX 3090 | 4 |
| 2 | 21725374 | cs-3090-05.auth.ad.bgu.ac.il | NVIDIA GeForce RTX 3090 | 4 |
| 3 | 21725417 | cs-3090-02.auth.ad.bgu.ac.il | NVIDIA GeForce RTX 3090 | 4 |

**Arms:** A natural PBR; B node-random PBR; C node-random BOO/HEM; D group-label-only HEM order + PBR; D′ (`Dprime` in CSV) hierarchy HEM order + PBR; E BOO on D’s coordinates with `row_labels=col_labels=labels[p]`; F_PBR/F_BOO use a full DOF permutation `default_rng(0).permutation(N)`; NATURAL_BOO is natural FEM + BOO/HEM.

D′ records identity and each successful contraction’s labels in `_hem_labels`, in original vertex coordinates. With levels stored finest first, `np.lexsort((vertex_id, *levels, final_labels))` prioritizes final groups, then the coarsest matching level, through to the finest. The original vertex ID only breaks remaining ties. It changes within-group order without reclustering. Matching descendants become contiguous; variable group offsets can still prevent global 8×8 alignment. Saved per-level labels, permutations and alignment counts are in the run directory.

For E, the harness asserts exact equality of C/E group pointers, block group IDs, nonzero pointers, packed local indices and values, and asserts E’s row/column IDs are contiguous. Thus the stored blocks are identical; gather/scatter addresses differ. Where a CSR remainder exists, its global coordinates also change.

### Best per arm: one shared configuration selected across the three jobs

For each arm/N/F, choose the configuration with the lowest median of its three job medians; the range uses that **same configuration** in every job. This is post-hoc tuning. It does not silently switch the configuration between jobs. All configurations and per-job times are in the adjacent CSVs. PBR uses kernel=block, smem=2048; BOO uses HEM, rowseg, 256 threads, p=64, rowmajor, legacy launch, remainder threshold=4; caps equal target. The config column supplies the winning block/min_nnz and target.

`x_loads` is the sum of column-group sizes over BOO blocks **per feature**, excluding CSR remainder. Occupancy is `boo_info`’s SM86 shared-memory/thread-limit estimate, not achieved occupancy or a profiler measurement. All checks restore Y to original FEM coordinates first, then compare to float64 SciPy at rtol=atol=1e-4.

| N | F | arm | exact variable config | median [min,max] ms | timing | PBR fraction / blocks | BOO blocks / x_loads / occupancy | check / worst abs error |
|---:|---:|:---|:---|:---|:---|:---|:---|:---|
| 65536 | 128 | A | block 8×8/min8/sm2048; k=0 | 0.577 [0.576, 0.584] | T | 1.000000 / 72580 | — | passed / 5.06e-06 |
| 65536 | 128 | B | block 8×8/min2/sm2048; k=0 | 0.848 [0.845, 0.861] | T | 1.000000 / 137602 | — | passed / 3.22e-06 |
| 65536 | 128 | C | HEM k=32/t256/p64/rowseg | 0.787 [0.783, 0.791] | T | — | 16767 / 473586 / 100.0% est. | passed / 6.01e-06 |
| 65536 | 128 | D | block 8×8/min12/sm2048; k=32 | 0.780 [0.780, 0.789] | T | 0.901605 / 84324 | — | passed / 3.69e-06 |
| 65536 | 128 | Dprime | block 8×8/min12/sm2048; k=32 | 0.745 [0.744, 0.752] | T | 0.919417 / 77666 | — | passed / 4.11e-06 |
| 65536 | 128 | E | HEM k=32/t256/p64/rowseg | 0.795 [0.786, 0.797] | T | — | 16767 / 473586 / 100.0% est. | passed / 6.01e-06 |
| 65536 | 128 | F_PBR | block 4×4/min8/sm2048; k=0 | 1.275 [1.270, 1.278] | T | 0.000000 / 0 | — | passed / 6.71e-06 |
| 65536 | 128 | F_BOO | HEM k=32/t256/p64/rowseg | 0.778 [0.770, 0.781] | T | — | 16701 / 471616 / 100.0% est. | passed / 5.3e-06 |
| 65536 | 128 | NATURAL_BOO | HEM k=32/t256/p64/rowseg | 0.753 [0.744, 0.754] | T | — | 16804 / 475250 / 100.0% est. | passed / 5.36e-06 |
| 65536 | 512 | A | block 8×8/min2/sm2048; k=0 | 2.236 [2.226, 2.298] | T | 1.000000 / 72580 | — | passed / 5.71e-06 |
| 65536 | 512 | B | block 8×8/min8/sm2048; k=0 | 3.409 [3.406, 3.512] | T | 1.000000 / 137602 | — | passed / 4.84e-06 |
| 65536 | 512 | C | HEM k=32/t256/p64/rowseg | 2.793 [2.770, 2.851] | T | — | 16767 / 473586 / 100.0% est. | passed / 7.77e-06 |
| 65536 | 512 | D | block 8×8/min12/sm2048; k=32 | 3.033 [3.014, 3.103] | T | 0.901605 / 84324 | — | passed / 6.2e-06 |
| 65536 | 512 | Dprime | block 8×8/min12/sm2048; k=64 | 2.862 [2.848, 2.941] | T | 0.923474 / 76910 | — | passed / 5.53e-06 |
| 65536 | 512 | E | HEM k=32/t256/p64/rowseg | 2.768 [2.751, 2.826] | T | — | 16767 / 473586 / 100.0% est. | passed / 7.77e-06 |
| 65536 | 512 | F_PBR | block 4×4/min8/sm2048; k=0 | 5.677 [5.670, 5.678] | T | 0.000000 / 0 | — | passed / 8.46e-06 |
| 65536 | 512 | F_BOO | HEM k=32/t256/p64/rowseg | 2.733 [2.713, 2.784] | T | — | 16701 / 471616 / 100.0% est. | passed / 6.55e-06 |
| 65536 | 512 | NATURAL_BOO | HEM k=32/t256/p64/rowseg | 2.719 [2.682, 2.769] | T | — | 16804 / 475250 / 100.0% est. | passed / 7.14e-06 |
| 262144 | 128 | A | block 8×8/min12/sm2048; k=0 | 2.236 [2.210, 2.300] | T | 1.000000 / 292612 | — | passed / 4.99e-06 |
| 262144 | 128 | B | block 8×8/min2/sm2048; k=0 | 3.326 [3.302, 3.448] | T | 1.000000 / 553908 | — | passed / 3.83e-06 |
| 262144 | 128 | C | HEM k=32/t256/p64/rowseg | 3.103 [3.058, 3.174] | T | — | 67961 / 1916328 / 100.0% est. | passed / 6.63e-06 |
| 262144 | 128 | D | block 8×8/min12/sm2048; k=32 | 3.105 [3.080, 3.182] | T | 0.873301 / 329328 | — | passed / 5.16e-06 |
| 262144 | 128 | Dprime | block 8×8/min12/sm2048; k=64 | 2.957 [2.927, 3.036] | T | 0.906343 / 303644 | — | passed / 5.16e-06 |
| 262144 | 128 | E | HEM k=32/t256/p64/rowseg | 3.021 [2.994, 3.092] | T | — | 67961 / 1916328 / 100.0% est. | passed / 6.63e-06 |
| 262144 | 128 | F_PBR | block 4×4/min8/sm2048; k=0 | 5.905 [5.898, 5.907] | T | 0.000000 / 0 | — | passed / 6.66e-06 |
| 262144 | 128 | F_BOO | HEM k=32/t256/p64/rowseg | 2.986 [2.951, 3.061] | T | — | 67879 / 1916215 / 100.0% est. | passed / 7.17e-06 |
| 262144 | 128 | NATURAL_BOO | HEM k=32/t256/p64/rowseg | 2.856 [2.818, 2.946] | T | — | 67922 / 1916265 / 100.0% est. | passed / 6.9e-06 |
| 262144 | 512 | A | block 8×8/min8/sm2048; k=0 | 8.909 [8.861, 9.597] | T | 1.000000 / 292612 | — | passed / 5.03e-06 |
| 262144 | 512 | B | block 8×8/min12/sm2048; k=0 | 13.593 [13.556, 14.649] | T | 1.000000 / 553908 | — | passed / 4.8e-06 |
| 262144 | 512 | C | HEM k=32/t256/p64/rowseg | 10.870 [10.857, 11.548] | T | — | 67961 / 1916328 / 100.0% est. | passed / 8.46e-06 |
| 262144 | 512 | D | block 8×8/min12/sm2048; k=32 | 12.453 [12.350, 13.119] | T | 0.873301 / 329328 | — | passed / 5.45e-06 |
| 262144 | 512 | Dprime | block 8×8/min12/sm2048; k=32 | 11.675 [11.623, 12.302] | T | 0.898340 / 308616 | — | passed / 5.45e-06 |
| 262144 | 512 | E | HEM k=32/t256/p64/rowseg | 11.247 [11.076, 11.871] | T | — | 67961 / 1916328 / 100.0% est. | passed / 8.46e-06 |
| 262144 | 512 | F_PBR | block 4×4/min8/sm2048; k=0 | 23.906 [23.903, 23.911] | T | 0.000000 / 0 | — | passed / 7.9e-06 |
| 262144 | 512 | F_BOO | HEM k=32/t256/p64/rowseg | 10.696 [10.691, 11.191] | T | — | 67879 / 1916215 / 100.0% est. | passed / 7.3e-06 |
| 262144 | 512 | NATURAL_BOO | HEM k=32/t256/p64/rowseg | 10.715 [10.646, 11.455] | T | — | 67922 / 1916265 / 100.0% est. | passed / 7.66e-06 |

### Preprocessing and one-time dense permutations

Entries are median [min,max] over the three jobs. Clustering, order construction, PAPᵀ and format conversion are separately measured CPU wall times. PAPᵀ includes CSR canonicalization; order construction is a separate column. Construction is done once per N/configuration and reused across F; repeated costs in the table are not independent measurements. PAPᵀ here is the additional HEM reorder from B to D/D′/E; construction of the random test input matrices is outside these preprocessing timers. These costs exclude the symmetry check included in E1’s sparse-reorder timer. Cluster time includes recording hierarchy labels. C/D/D′/E share the same cluster computation at a given k: do not add that cost repeatedly.

X and Y columns time **one allocating GPU index_select per sample**, with staged indices, using the median of 20 such calls in each job. They are not the total for 20 operations. External columns map original↔arm coordinates; HEM-only columns map B↔D/D′/E, without charging the node shuffle. Y uses the inverse index. Zero means no permutation needed; it is not a measured zero-time copy. Index construction/upload and H2D format staging are excluded from those dense-operation times; format staging is recorded separately in CSV.

| N | F | arm | cluster ms | order ms | PAPᵀ ms | convert ms | X external ms | Y external ms | X HEM-only ms | Y HEM-only ms |
|---:|---:|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 65536 | 128 | A | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 116.700 [91.996, 229.081] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 128 | B | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 141.024 [111.503, 253.036] | 0.107 [0.096, 0.109] | 0.106 [0.095, 0.110] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 128 | C | 3139.317 [2285.190, 4261.969] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 81.217 [65.276, 105.489] | 0.107 [0.096, 0.109] | 0.106 [0.095, 0.110] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 128 | D | 3139.317 [2285.190, 4261.969] | 2.810 [2.080, 3.825] | 75.165 [64.794, 82.678] | 220.853 [120.891, 269.000] | 0.102 [0.095, 0.104] | 0.102 [0.095, 0.104] | 0.103 [0.095, 0.106] | 0.102 [0.095, 0.106] |
| 65536 | 128 | Dprime | 3139.317 [2285.190, 4261.969] | 7.684 [5.691, 10.535] | 89.445 [67.377, 116.754] | 143.781 [115.626, 260.830] | 0.102 [0.095, 0.105] | 0.102 [0.095, 0.105] | 0.103 [0.095, 0.104] | 0.102 [0.095, 0.105] |
| 65536 | 128 | E | 3139.317 [2285.190, 4261.969] | 2.810 [2.080, 3.825] | 75.165 [64.794, 82.678] | 79.343 [72.254, 107.760] | 0.102 [0.095, 0.104] | 0.102 [0.095, 0.104] | 0.103 [0.095, 0.106] | 0.102 [0.095, 0.106] |
| 65536 | 128 | F_PBR | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 635.631 [442.126, 895.246] | 0.106 [0.095, 0.109] | 0.105 [0.095, 0.109] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 128 | F_BOO | 4511.630 [3365.673, 5760.556] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 105.247 [76.302, 120.953] | 0.106 [0.095, 0.109] | 0.105 [0.095, 0.109] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 128 | NATURAL_BOO | 2683.415 [1922.503, 3486.782] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 75.526 [61.269, 93.434] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 512 | A | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 116.486 [90.377, 227.164] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 512 | B | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 140.915 [111.631, 258.895] | 0.342 [0.334, 0.345] | 0.342 [0.335, 0.346] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 512 | C | 3139.317 [2285.190, 4261.969] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 81.217 [65.276, 105.489] | 0.342 [0.334, 0.345] | 0.342 [0.335, 0.346] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 512 | D | 3139.317 [2285.190, 4261.969] | 2.810 [2.080, 3.825] | 75.165 [64.794, 82.678] | 220.853 [120.891, 269.000] | 0.342 [0.335, 0.343] | 0.342 [0.335, 0.344] | 0.342 [0.334, 0.345] | 0.342 [0.335, 0.344] |
| 65536 | 512 | Dprime | 3197.102 [2310.991, 4278.105] | 9.411 [6.764, 12.051] | 90.269 [66.901, 117.475] | 143.380 [142.727, 261.454] | 0.342 [0.335, 0.345] | 0.342 [0.335, 0.345] | 0.343 [0.334, 0.343] | 0.343 [0.334, 0.344] |
| 65536 | 512 | E | 3139.317 [2285.190, 4261.969] | 2.810 [2.080, 3.825] | 75.165 [64.794, 82.678] | 79.343 [72.254, 107.760] | 0.342 [0.335, 0.343] | 0.342 [0.335, 0.344] | 0.342 [0.334, 0.345] | 0.342 [0.335, 0.344] |
| 65536 | 512 | F_PBR | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 635.631 [442.126, 895.246] | 0.343 [0.335, 0.345] | 0.343 [0.335, 0.346] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 512 | F_BOO | 4511.630 [3365.673, 5760.556] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 105.247 [76.302, 120.953] | 0.343 [0.335, 0.345] | 0.343 [0.335, 0.346] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 65536 | 512 | NATURAL_BOO | 2683.415 [1922.503, 3486.782] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 75.526 [61.269, 93.434] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 128 | A | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 463.296 [304.967, 692.929] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 128 | B | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 559.521 [349.626, 814.169] | 0.348 [0.340, 0.351] | 0.348 [0.340, 0.351] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 128 | C | 14755.561 [10711.817, 20570.142] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 362.416 [286.698, 482.941] | 0.348 [0.340, 0.351] | 0.348 [0.340, 0.351] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 128 | D | 14755.561 [10711.817, 20570.142] | 9.501 [7.496, 13.513] | 258.964 [211.253, 375.879] | 687.922 [412.379, 895.974] | 0.345 [0.336, 0.346] | 0.344 [0.336, 0.346] | 0.345 [0.337, 0.349] | 0.344 [0.336, 0.349] |
| 262144 | 128 | Dprime | 14879.708 [10856.118, 20626.483] | 39.179 [30.304, 44.050] | 284.761 [241.046, 384.858] | 642.872 [417.833, 843.199] | 0.343 [0.336, 0.345] | 0.343 [0.335, 0.345] | 0.344 [0.337, 0.346] | 0.344 [0.336, 0.346] |
| 262144 | 128 | E | 14755.561 [10711.817, 20570.142] | 9.501 [7.496, 13.513] | 258.964 [211.253, 375.879] | 331.537 [289.849, 427.155] | 0.345 [0.336, 0.346] | 0.344 [0.336, 0.346] | 0.345 [0.337, 0.349] | 0.344 [0.336, 0.349] |
| 262144 | 128 | F_PBR | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2676.012 [1544.841, 3489.638] | 0.345 [0.338, 0.349] | 0.346 [0.338, 0.350] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 128 | F_BOO | 21351.663 [16812.934, 29818.431] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 483.967 [366.851, 612.006] | 0.345 [0.338, 0.349] | 0.346 [0.338, 0.350] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 128 | NATURAL_BOO | 12404.001 [8900.708, 16919.565] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 326.813 [260.603, 418.919] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 512 | A | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 459.174 [304.388, 684.288] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 512 | B | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 557.485 [368.436, 822.815] | 1.301 [1.295, 1.304] | 1.302 [1.295, 1.303] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 512 | C | 14755.561 [10711.817, 20570.142] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 362.416 [286.698, 482.941] | 1.301 [1.295, 1.304] | 1.302 [1.295, 1.303] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 512 | D | 14755.561 [10711.817, 20570.142] | 9.501 [7.496, 13.513] | 258.964 [211.253, 375.879] | 687.922 [412.379, 895.974] | 1.300 [1.295, 1.304] | 1.302 [1.295, 1.305] | 1.298 [1.294, 1.303] | 1.301 [1.295, 1.305] |
| 262144 | 512 | Dprime | 14755.561 [10711.817, 20570.142] | 33.720 [26.298, 37.043] | 285.690 [243.082, 389.433] | 656.841 [392.333, 871.985] | 1.299 [1.295, 1.304] | 1.300 [1.295, 1.304] | 1.298 [1.293, 1.303] | 1.300 [1.295, 1.303] |
| 262144 | 512 | E | 14755.561 [10711.817, 20570.142] | 9.501 [7.496, 13.513] | 258.964 [211.253, 375.879] | 331.537 [289.849, 427.155] | 1.300 [1.295, 1.304] | 1.302 [1.295, 1.305] | 1.298 [1.294, 1.303] | 1.301 [1.295, 1.305] |
| 262144 | 512 | F_PBR | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2676.012 [1544.841, 3489.638] | 1.304 [1.299, 1.306] | 1.304 [1.298, 1.306] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 512 | F_BOO | 21351.663 [16812.934, 29818.431] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 483.967 [366.851, 612.006] | 1.304 [1.299, 1.306] | 1.304 [1.298, 1.306] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 262144 | 512 | NATURAL_BOO | 12404.001 [8900.708, 16919.565] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 326.813 [260.603, 418.919] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Group-size distributions

Partitions and group sizes match across the three jobs. D/D′/E use C’s groups. Each cell is min / median / max.

| N | input | k=8 | k=16 | k=32 | k=64 |
|---:|:---|:---|:---|:---|:---|
| 65536 | C | 2 / 8 / 8 | 4 / 16 / 16 | 4 / 28 / 32 | 12 / 56 / 64 |
| 65536 | F_BOO | 3 / 8 / 8 | 4 / 16 / 16 | 4 / 28 / 32 | 4 / 56 / 64 |
| 65536 | NATURAL_BOO | 2 / 8 / 8 | 4 / 16 / 16 | 4 / 28 / 32 | 8 / 56 / 64 |
| 262144 | C | 3 / 8 / 8 | 4 / 16 / 16 | 4 / 28 / 32 | 4 / 56 / 64 |
| 262144 | F_BOO | 3 / 8 / 8 | 3 / 16 / 16 | 4 / 28 / 32 | 4 / 56 / 64 |
| 262144 | NATURAL_BOO | 2 / 8 / 8 | 4 / 16 / 16 | 4 / 28 / 32 | 4 / 56 / 64 |

### Hierarchy alignment control at k=32

Counts from job 1 (the saved partitions/orders match across jobs). Use matching level 1 for pairs, level 2 for quads, and level 3 for octets. A subgroup is contiguous if its vertices occupy one interval; it is aligned if that interval also starts at a multiple of its size. Later levels can contain additional pairs/quads/octets; this table reports these specified early levels only.

| N | order | subgroup size | level | groups | contiguous | aligned |
|---:|:---|---:|---:|---:|---:|---:|
| 65536 | stable | 2 | 1 | 32633 | 16304 | 8821 |
| 65536 | stable | 4 | 2 | 16253 | 16249 | 6517 |
| 65536 | stable | 8 | 3 | 7450 | 2070 | 422 |
| 65536 | hierarchy | 2 | 1 | 32633 | 32633 | 19837 |
| 65536 | hierarchy | 4 | 2 | 16253 | 16253 | 6513 |
| 65536 | hierarchy | 8 | 3 | 7450 | 7450 | 1540 |
| 262144 | stable | 2 | 1 | 130830 | 65529 | 34329 |
| 262144 | stable | 4 | 2 | 65303 | 65296 | 17134 |
| 262144 | stable | 8 | 3 | 30057 | 8465 | 1095 |
| 262144 | hierarchy | 2 | 1 | 130830 | 130830 | 74813 |
| 262144 | hierarchy | 4 | 2 | 65303 | 65303 | 17103 |
| 262144 | hierarchy | 8 | 3 | 30057 | 30057 | 3920 |

### Paired ratios across jobs

Each ratio is computed **within each job**, using the selected fixed configurations above; report median [min,max] across the three ratios. B/A and D/A > 1 indicate a penalty versus natural PBR. C/D < 1 means C has the lower measured time. These remain ratios of tuned arms, potentially with different k.

| N | F | B/A | D/A | C/D |
|---:|---:|:---|:---|:---|
| 65536 | 128 | 1.469 [1.466, 1.473] | 1.352 [1.351, 1.353] | 1.004 [0.998, 1.014] |
| 65536 | 512 | 1.528 [1.524, 1.530] | 1.354 [1.350, 1.356] | 0.919 [0.919, 0.921] |
| 262144 | 128 | 1.495 [1.487, 1.499] | 1.388 [1.383, 1.394] | 0.997 [0.993, 1.000] |
| 262144 | 512 | 1.526 [1.526, 1.530] | 1.386 [1.367, 1.405] | 0.879 [0.873, 0.880] |

### Matched controls: k=32, PBR 8×8/min_nnz=2

These comparisons hold targets and PBR settings fixed rather than comparing separately tuned winners. Values are medians [min,max] of paired per-job ratios. D/D′ > 1 favors hierarchy ordering; C/E > 1 favors contiguous BOO coordinates; F_PBR/B > 1 is the additional DOF-shuffle penalty for PBR; F_BOO/C compares BOO/HEM under the two permutations (reclustering in F).

| N | F | D/D′ | C/E | F_PBR/B | F_BOO/C | natural BOO/A |
|---:|---:|:---|:---|:---|:---|:---|
| 65536 | 128 | 1.076 [1.076, 1.078] | 0.995 [0.988, 0.996] | 1.719 [1.699, 1.754] | 0.987 [0.984, 0.988] | 1.289 [1.286, 1.298] |
| 65536 | 512 | 1.090 [1.088, 1.091] | 1.009 [1.007, 1.009] | 1.885 [1.824, 1.892] | 0.978 [0.976, 0.980] | 1.205 [1.205, 1.216] |
| 262144 | 128 | 1.089 [1.088, 1.093] | 1.026 [1.022, 1.027] | 1.935 [1.865, 1.948] | 0.964 [0.962, 0.965] | 1.266 [1.263, 1.271] |
| 262144 | 512 | 1.078 [1.076, 1.080] | 0.973 [0.967, 0.980] | 1.903 [1.796, 1.914] | 0.984 [0.969, 0.985] | 1.200 [1.196, 1.200] |

### Plain reading

Across the three jobs, node-random PBR takes 1.47–1.53× the time of natural PBR; group-label-reordered PBR still takes 1.35–1.39×. Those are the clear ordering penalties in the selected-arm comparison. The reordered winners select min_nnz=12 and send roughly 10–13% of nonzeros to CSR, so the tuned gain is not a pure ordering effect.

C/D ranges from 0.879 to 1.004 across workload medians. C and D are effectively tied at F=128; C has the lower measured time at F=512, but their separation stays below 1.3×. This does not establish a large BOO-versus-reordered-PBR advantage.

In matched controls, hierarchy ordering gives D/D′ ratios of about 1.08–1.09 across the three jobs. This is a modest repeated difference, not a ≥1.3× effect. It makes matching subtrees contiguous but does not solve tile alignment: at N=262,144/k=32, all 30,057 level-3 octets become contiguous, but only 3,920 start at an 8-aligned offset. C/E stays within roughly 3% of unity and changes direction; contiguous gather addresses do not show a consistent material gain here.

The DOF-level shuffle is a substantially stronger stress case: tuned F_PBR/F_BOO job-ratio medians are 1.63–2.24×. However, every selected F_PBR configuration has **compressed_fraction=0 and zero PBR blocks**: it is the PBR wrapper running entirely through CSR remainder. This is evidence of fixed-tile structure being lost, not a clean PBR-kernel-versus-BOO-kernel measurement. BOO/HEM remains close to its node-shuffle timings. Natural FEM + BOO is also included; its matched-control time is about 1.20–1.29× A, so no ≥1.3× distinction is claimed for that control.

The GPU validation suite passed **755 tests, with 12 skips**. All **1,248** full-run configuration/job outputs passed the restored-coordinate reference check; **24,960** raw matmul samples are retained. Saved hierarchy/permutation arrays match byte-for-byte at the array level across jobs. The report also checks source/extension hashes and per-configuration matrix/input hashes. [Run metadata, logs, job scripts, hierarchy arrays and alignment diagnostics](../../tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928) retain the supporting evidence.

### Data and reproduction

- [All per-job configurations](../../tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928/derived/REORDER_VS_BOO_REPLICATES.csv). Raw matmul and dense-permutation samples are in the [root sample archive](../../RESULTS_ARCHIVE.md).
- [Selected configurations and three job times](../../tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928/derived/REORDER_VS_BOO_BEST.csv), [paired ratios](../../tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928/derived/REORDER_VS_BOO_RATIOS.csv), and [independently selected per-job winners](../../tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928/derived/REORDER_VS_BOO_JOB_WINNERS.csv).
- Run each replicate in a separate Slurm allocation: `python -m benchmark.bench_reorder_vs_boo --replicate 1 --out results/job1` (then replicate 2 and 3 in new jobs).
- Regenerate this report: `python tests/torch_fast_spdmm/benchmark/report_reorder_vs_boo.py tests/torch_fast_spdmm/benchmark/results/reorder_vs_boo_20260928 --report experiments/2026-09-28_reorder_vs_boo/REORDER_VS_BOO.md`.

This experiment changes ordering and format selection while using existing CUDA kernels. It does not provide a fully independent representation-versus-kernel decomposition. No sub-1.3× difference from E1’s single session is treated as established. Cross-job ranges are descriptive, not confidence intervals.
