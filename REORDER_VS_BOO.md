# E1: reordering versus BOO

## Provenance and protocol

Historical E1: Slurm job `21719128`, GPU node `cs-3090-06` (verified with `sacct`), NVIDIA RTX 3090, float32 values, int32 sparse indices, seed 0. N = 16,384 / 65,536 / 262,144; F = 128 / 512. Source base commit: `4492e8c4a5893a776884ca97f0d08a8d45210233`, plus the then-uncommitted benchmark and tests. Measured source SHA-256: `13a7fae70a1c0b1f744e08ef38164c17ac071ffd70457911da0328ecff877d3a`; extension SHA-256: `c10ff765360004030f1f00a2bcaff954a2c4b4b396aebfbb920c4efcfec756da`. Historical metadata retained `unknown` Git fields because the uploaded snapshot had no `.git`; the base commit alone does not identify that benchmark snapshot. This branch commits the harness and evidence.

Raw per-configuration numbers: [REORDER_VS_BOO_E1.csv](REORDER_VS_BOO_E1.csv). All 7,200 per-call samples: [REORDER_VS_BOO_E1_SAMPLES.csv](REORDER_VS_BOO_E1_SAMPLES.csv). Original metadata, saved permutations and logs: [run directory](tests/torch_fast_spdmm/benchmark/results/hem_pbr_20260928).

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

Arm A = natural FEM + PBR; B = node-permuted FEM + PBR; C = node-permuted FEM + BOO/HEM; D = HEM group-label-reordered B + PBR. In the config column `b/min` means square PBR tile size/min_nnz; k denotes HEM target. All other exact settings are fixed above. `—` is inapplicable; `NR` means not recorded in E1, not zero. BOO x_loads and occupancy will be reconstructed separately from saved labels; dense permutation timings require new measurements and cannot be recovered from old timings.

| N | F | arm | config | time ms | timing | PBR fraction / blocks | BOO blocks / x_loads / occupancy | reference check / max abs error |
|---:|---:|:---|:---|---:|:---|:---|:---|:---|
| 16384 | 128 | A | k=0, b/min=8/12 | 0.160256 | T | 1.000000 / 17860 | — | passed / 4.05e-06 |
| 16384 | 128 | B | k=0, b/min=8/2 | 0.226304 | T | 1.000000 / 33954 | — | passed / 2.3e-06 |
| 16384 | 128 | C | k=32, HEM, 256 threads, p=64 | 0.229824 | T | — | 4117 / NR / NR | passed / 4.93e-06 |
| 16384 | 128 | D | k=32, b/min=8/12 | 0.208880 | T | 0.888923 / 20338 | — | passed / 3.73e-06 |
| 16384 | 512 | A | k=0, b/min=8/2 | 0.578560 | T | 1.000000 / 17860 | — | passed / 4.49e-06 |
| 16384 | 512 | B | k=0, b/min=8/12 | 0.878560 | T | 1.000000 / 33954 | — | passed / 4.91e-06 |
| 16384 | 512 | C | k=32, HEM, 256 threads, p=64 | 0.756736 | T | — | 4117 / NR / NR | passed / 4.93e-06 |
| 16384 | 512 | D | k=32, b/min=8/12 | 0.771072 | T | 0.888923 / 20338 | — | passed / 4.58e-06 |
| 65536 | 128 | A | k=0, b/min=8/2 | 0.579584 | T | 1.000000 / 72580 | — | passed / 3.77e-06 |
| 65536 | 128 | B | k=0, b/min=8/12 | 0.852352 | T | 1.000000 / 137602 | — | passed / 2.91e-06 |
| 65536 | 128 | C | k=32, HEM, 256 threads, p=64 | 0.787456 | T | — | 16767 / NR / NR | passed / 5.8e-06 |
| 65536 | 128 | D | k=16, b/min=8/12 | 0.782336 | T | 0.887448 / 85056 | — | passed / 4.51e-06 |
| 65536 | 512 | A | k=0, b/min=8/8 | 2.268160 | T | 1.000000 / 72580 | — | passed / 5.71e-06 |
| 65536 | 512 | B | k=0, b/min=8/12 | 3.459584 | T | 1.000000 / 137602 | — | passed / 3.87e-06 |
| 65536 | 512 | C | k=32, HEM, 256 threads, p=64 | 2.795520 | T | — | 16767 / NR / NR | passed / 7.77e-06 |
| 65536 | 512 | D | k=32, b/min=8/12 | 3.104192 | T | 0.901605 / 84324 | — | passed / 5.21e-06 |
| 262144 | 128 | A | k=0, b/min=8/8 | 2.281472 | T | 1.000000 / 292612 | — | passed / 4.32e-06 |
| 262144 | 128 | B | k=0, b/min=8/12 | 3.368448 | T | 1.000000 / 553908 | — | passed / 3.38e-06 |
| 262144 | 128 | C | k=32, HEM, 256 threads, p=64 | 3.038720 | T | — | 67961 / NR / NR | passed / 6.41e-06 |
| 262144 | 128 | D | k=32, b/min=8/12 | 3.160576 | T | 0.873301 / 329328 | — | passed / 5.16e-06 |
| 262144 | 512 | A | k=0, b/min=8/2 | 9.078736 | T | 1.000000 / 292612 | — | passed / 4.66e-06 |
| 262144 | 512 | B | k=0, b/min=8/2 | 13.815808 | T | 1.000000 / 553908 | — | passed / 5.15e-06 |
| 262144 | 512 | C | k=32, HEM, 256 threads, p=64 | 10.928592 | T | — | 67961 / NR / NR | passed / 8.46e-06 |
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

This one session shows a ≥1.3× ordering penalty relative to A for most B/D cells. C and D are within 1.3× in these selected results; **E1 does not establish a performance difference between them**. No difference below roughly 1.3× is claimed from this single session. Representation and kernel effects remain coupled in C versus D. Expanded replicated arms follow below when measured.

## Expanded experiment (pending)

Three separate Slurm jobs will run N=65,536 and 262,144, F=128 and 512, with the same timing protocol, seed and config sweep. Added arms: D′ (within-group HEM hierarchy ordering), E (contiguous-group BOO with exactly C’s blocks), F-PBR and F-BOO (DOF-level permutation), and natural FEM + BOO/HEM. We will report medians and min/max ranges across job medians, paired per-job ratios, structural counters and one-time dense permutation costs.
