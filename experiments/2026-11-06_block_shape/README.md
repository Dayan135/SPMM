# Block shape: growing k without losing CTAs per SM

Status: gate, occupancy, shape screen and P16 add-on complete (2026-11-07). Confirmation frozen 2026-11-08,
run 2026-11-09; results below.

## Question

Can BOO use larger blocks without losing resident CTAs per SM, and does that make SpMM faster?

Rowseg's arena costs 256 B per column at P64 and 6 B per nonzero, but only 8 B per row. Growing
k_c fills shared memory quickly. Step A also stages the whole column group, so square k leaves
staged X nearly flat. Growing only k_r should merge blocks that share a column group, which cuts
staged X with little arena growth.

Model counts on fem2d_permuted, HEM, N=65536, P64, relative to 32x32. These are counts, not times:

| Shape | Arena | Threads/SM | X staged | X read | Y commits |
|---|---|---|---|---|---|
| 32x32 | 12.6 KB | 100% | 1.00 | 0.46 | 1.00 |
| 64x64 | 26.8 KB | 100% | 0.98 | 0.36 | 0.79 |
| 128x128 | 55.8 KB | 33% | 0.95 | 0.29 | 0.64 |
| 64x32 | 14.2 KB | 100% | 0.72 | 0.36 | 1.00 |
| 128x32 | 15.9 KB | 100% | 0.53 | 0.29 | 1.00 |
| 256x32 | 17.3 KB | 100% | 0.40 | 0.25 | 1.00 |
| 128x16 | 8.7 KB | 100% | 0.42 | 0.29 | 1.33 |

The September grid timed 64x16 only, among the row-heavy shapes: 1.03-1.06x best-vs-best on
fem2d_permuted and 0.93-0.94x on diag_b0.3. 64x32, 128x32 and 256x32 have never been timed.

## Block size against P

The hypothesis is that lowering P while raising k keeps CTAs/SM high, because the arena's X region is
`4 * P * k_c` bytes. The first check uses the 2026-10-14 screen, which timed square k at P8-P128 with
FPT 1 and 2 on one allocation. Speedup is against the best k32/P64 arm for the same input and F,
shown as F128 / F512. Threads/SM is the best available over T.

| Input | 64/P32 | 128/P16 | 128/P64 | Threads/SM, 128/P16 vs 128/P64 |
|---|---|---|---|---|
| fem2d | 0.99 / 0.90 | 0.89 / 0.72 | 0.53 / 0.52 | 1536 vs 512 |
| fem2d_permuted | 0.99 / 0.89 | 0.90 / 0.66 | 0.54 / 0.53 | 1536 vs 512 |
| pwtk | 0.89 / 0.88 | 0.77 / 0.71 | 0.60 / 0.59 | 1024 vs 512 |
| webbase | 0.76 / 0.76 | 0.55 / 0.45 | 0.29 / 0.30 | 1536 vs 512 |
| roadnet | 1.18 / 1.08 | 1.04 / 0.82 | 0.94 / 0.91 | 1536 vs 1024 |
| diag_dd0.05 | 1.11 / 1.04 | 1.05 / 0.72 | 1.08 / 1.06 | 1536 vs 1024 |
| diag_b1.0 | 0.96 / 1.01 | 0.60 / 0.58 | 0.87 / 0.90 | 512 vs 512 |
| diag_skewed | 0.98 / 1.01 | 0.66 / 0.65 | 0.83 / 0.84 | 512 vs 512 |

Lower P restores occupancy except where nonzeros fill the arena, and it recovers much of k128's loss
on the clustered and irregular inputs, but restored occupancy alone rarely beats k32/P64. Rectangular
shapes and FPT4 are untested with P16, so P16 is added as its own stage, reported as a k by P matrix
with the screen's P32-P128 results.

## Stages

Each stage runs only if the previous one justifies it.

| Stage | Job | Output |
|---|---|---|
| 1. Gate (CPU) | `shape_job.sh gate`, array 0-25 | Real BOO structure for 14 shapes on 26 inputs |
| 2. Stage 0 (GPU) | `shape_job.sh stage0` | Rectangular and padded correctness, three sanitizers |
| 3. Sanity (GPU) | `shape_job.sh sanity` | Two inputs, F128: arms run and pass the launch audit |
| 4. Occupancy (GPU) | `shape_job.sh occupancy`, array 0-23 | Time vs CTAs/SM for identical work |
| 5. Shape screen (GPU) | `shape_job.sh screen`, array 0-15 | Time vs shape, two allocations |
| 6. P16 add-on (GPU) | separate manifest, screen manifest unchanged | The same shapes at P16, FPT{1,2,4}, F{128,512}, for the k by P matrix |
| 7. Confirmation (GPU) | manifest written from the screen and add-on | Finalists on three GPUs, with F256 and 1M scale held out, against direct cuSPARSE |
| 8. Active-column lists | only if Step A still dominates | Separate kernel change, not started |

## Fixed before results

**Gate pass.** Relative to the same input and method at 32x32: X staged <= 0.80, Y commits <= 1.10,
100% threads/SM reachable at P64, remainder fraction rise <= 0.005. A pass selects a shape for
timing and says nothing about time.

**Screen objects.** The frozen labeling per input from the 2026-10-14 screen. 32x32, 64x64, 128x128,
64x32 and 128x32 run on every input, so a gate failure can be tested against time. Other shapes with
k_r > k_c run where they pass the gate.

**Screen grid.** P{32,64,128} x T{128,256,512} x FPT{1,2,4}, F{128,512}, 5 warmups, 20 repeats per
allocation. Complete-call timing with `launch_mode='auto'` and an independent shuffle per repeat.

**A shape is a finalist on an input when** its best arm beats the best 32x32 arm by more than 5% in
both allocations, at the same F. The 5% bound comes from the 0.96-1.035 spread of earlier
identical-work controls. Tuning is per (input, F), so best-vs-best is the comparison.

**Occupancy answer.** Padding keeps work and bank mapping fixed, so the ratio of `_ctaN` to `_pad0`
is the cost of occupancy alone. If the slowdown at 33% threads/SM is within the control spread,
occupancy is not the limit on k, and the k128 penalty has another cause.

**Claims.** A speedup claim needs confirmation on three independent GPUs. It is ranked by matmul time
and reported against the strongest direct cuSPARSE arm in the same cell, with preprocessing separate.

## Changed after the first results (2026-11-07)

These rules were added after seeing first-pass data, so they are not part of the pre-registration
above. Results are reported both ways where it matters.

- **Control band, post hoc.** A (allocation, cell) is dropped when any `_dup` or `_pad0` control falls
  outside [0.95, 1.05] of its reference arm. This was added after a noisy webbase occupancy cell
  (alloc2, F128, GPU 991c4f33) appeared. `shape_report.py` lists every dropped cell.
- **Launch audit with Nsight Systems.** The Kineto check inside `bench_shape` captured 0-1 CUDA
  events on long F512 cells, so those arms could not be verified that way. `shape_job.sh audit` reruns
  every F512 cell once under `nsys` (`shape_launch_audit.py`) and checks the launched kernel sequence
  against the expected specializations. The report accepts an arm verified by either method. Kernel
  selection depends on the manifest and device type, not timing, so one audit covers all allocations.
- **Independent GPUs.** Screen allocations 1 and 2 shared one GPU (fe230431, cs-pheno-12) on 5 of 8
  inputs, and occupancy allocations 1 and 2 shared b0aca411 (cs-pheno-11) on pwtk. The `kp` stage
  excludes cs-pheno-12, and occupancy allocation 4 reruns pwtk excluding cs-pheno-11. The finalist
  rule now also requires >= 2 distinct GPU UUIDs, and screen and `kp` allocations are pooled for P >= 32.

## Confirmation, frozen 2026-11-08 before any confirmation timing

Manifest: `shape_run/manifests/confirm.json`, from `shape_confirm_manifest.py`. The selection table is
`analysis/confirm_freeze/selection.csv`. The same selection without the control-band filter is
identical (`selection_unfiltered.csv`).

- **Selection, per (input, F in {128, 512}), from screen and kp data only.** Per allocation, speedup =
  best 32x32 / arm. Score = minimum over allocations, with arms on fewer than two GPUs excluded.
  The winner is the top-scoring non-32x32 arm; the comparator is the top-scoring 32x32 arm.
- **Inputs.** fem2d_permuted, roadnet, fem2d, diag_dd0.05 (sparse noisy diagonal), diag_b1.0
  (dense diagonal), pwtk (borderline: no shape clears 1.05 in every allocation), and webbase
  (negative control). fem2d F128 is also borderline (1.047).
- **Held out.** F256 times the union of the F128 and F512 selections; the primary claim is the
  F512-selected pair. 1M is generated inputs only, each F selection at its own F. pwtk and roadNet
  keep their native size.
- **Baselines, in the same interleaved cell as the BOO arms they are compared with.** Torch CSR and
  four direct algorithms (DEFAULT, CSR_ALG1-3) in original order. For inputs whose labeling is not
  identity (fem2d, fem2d_permuted, roadnet), the same four algorithms on A reordered by the
  winner's labels and by the 32x32 labels: resident X' and Y' (`steady_api`), and per-call X
  gather and Y restore (`gather_api`). Each baseline family is a separate sub-cell that re-times
  the BOO winner and comparator as its anchor, which keeps 1M F512 within memory.
- **Allocations.** Three, on disjoint node sets, so every input runs on three distinct GPUs.
- **Claims.** Minimum over allocations: above 1.05 is a win, 0.95-1.05 is parity, below 0.95 is a
  loss. Shape claim: winner against comparator. cuSPARSE claim: winner against the fastest direct
  algorithm, reported separately for original order, reordered steady state and reordered gather.
  Every table is reported with and without the control-band filter.

### Confirmation run notes (2026-11-09)

- **ise-3090-04 excluded.** It has an Ivy Bridge CPU (Xeon E5-2620 v2, no AVX2), so torch runs at CPU
  capability `DEFAULT` and `torch.randn` gives a different X. The input hash check refused every task
  there (allocation 2 tasks 3-10 and the first audit). No timing ran. Those tasks were rerun on
  other ise nodes. The failed outputs are kept in `shape_run/failed/confirm_alloc2_ise-3090-04`.
- **ise-3090-01 excluded after the fact.** Allocation 2's roadnet task (task 1) ran there. At F256 and
  F512, unrelated kernels (BOO, torch CSR, cuSPARSE CSR_ALG1) each failed the float32 bound with
  exactly 64 entries. The same references pass on the other two allocations, which points to that
  GPU. Task 1 was rerun on another ise node, and the original is kept in `failed/confirm_alloc2_ise-3090-01`.
  Tasks 0 and 2 from ise-3090-01 passed every check and are kept.
- Both exclusions follow correctness failures, not timing. The ise-3090-01 exclusion was decided after
  a partial report of allocations 1 and 3 had been seen. The three allocations remain on disjoint
  node groups.

### Confirmation results (2026-11-09)

All cells have 3 allocations on 3 distinct GPUs. Values are speedup of the frozen BOO winner, minimum
over allocations, for F128 / F256 (F512-selected pair) / F512 at 262,144 rows, then 1M rows at
F128 / F512. Source: `analysis/confirm/report.txt`, `confirm_all.csv`, `confirm_filtered.csv`.

The bins are thresholds on that minimum: > 1.05 win, 0.95-1.05 parity, < 0.95 loss. A win is robust
across GPUs. "Parity" means the minimum fell in the band and does not establish equivalence. A
minimum below 0.95 does not by itself show a consistent loss. Verdicts use the unrounded minimum
(changed 2026-11-10; no verdict moved). The control-band filter changes no verdict. It drops one
group, fem2d F512 allocation 3, which leaves that cell with two allocations (marked `*` in
`confirm_filtered.csv`).

**Shape claim: winner against tuned 32x32**

| Input | Winner (F128; F512) | 262K rows: F128 / F256 / F512 | 1M rows: F128 / F512 |
|---|---|---|---|
| fem2d_permuted | 256x32; 256x64 | 1.22 / 1.26 / 1.28 | 1.21 / 1.27 |
| roadnet | 128x64 | 1.17 / 1.19 / 1.18 | - |
| fem2d | 128x32; 256x64 | 1.04 / 1.13 / 1.10 | 1.08 / 1.16 |
| diag_dd0.05 | 64x64; 128x64 | 1.08 / 1.09 / 1.09 | 1.09 / 1.09 |
| diag_b1.0 | 64x32 | 1.07 / 1.06 / 1.03 | 1.08 / 1.05 |
| pwtk | 128x32; 64x32 | 1.04 / 1.09 / 1.03 | - |
| webbase (negative control) | 64x16 | 1.00 / 1.01 / 1.05 | - |

The shape claim is confirmed on five of the six finalists, including held-out F256 and 1M. fem2d F128
(borderline at selection) and diag_b1.0 F512 are parity, and pwtk is parity except at held-out F256.
Webbase stays at parity, except for a marginal 1.054 at F512. The F128-selected pairs at F256 agree:
1.06-1.32 win everywhere except pwtk (1.04) and webbase (0.98).

**cuSPARSE claim: winner against the fastest direct algorithm in the same cell**

| Input | Original order | Reordered, resident X', Y' | Reordered, gather/restore per call |
|---|---|---|---|
| fem2d_permuted | win 1.19 / 1.29 / 1.99; 1M 1.30 / 2.02 | parity: 0.91 (loss) / 1.00 / 1.00; 1M 0.97 / 1.07 | win 1.30-1.46 |
| fem2d | loss 0.85-0.93; 1M F512 0.98 | loss 0.86-0.94; 1M F512 1.02 | win 1.27-1.42 |
| roadnet | loss 0.68 / 0.86 / n/a | loss 0.66 / 0.85 / n/a | win 1.41 / 1.61 / n/a |
| diag_dd0.05 | loss to parity 0.89-0.98; 1M F512 1.03 | identity labels | identity labels |
| diag_b1.0 | win 2.4-2.9 | identity labels | identity labels |
| pwtk | win 1.07-1.23 | identity labels | identity labels |
| webbase | loss 0.51-0.53 | identity labels | identity labels |

In the measured cells, with the algorithms that fit in memory, BOO's worst case against cuSPARSE on
the same reordering with resident X' and Y' falls in or below the parity band (0.86-1.07). On
fem2d_permuted, most of BOO's lead over original-order cuSPARSE goes away once cuSPARSE gets the same
ordering, so the ordering explains most of it. In every measured cell where cuSPARSE must gather X
and restore Y per call, BOO wins (1.27-1.61). At 1M F512 the resident comparison is DEFAULT-only,
and some 1M F128 anchors are missing (see caveats), so those cells are weaker evidence.

Caveats:
- Memory headroom excluded some cuSPARSE algorithms (282 arms). For roadnet F256 that includes
  CSR_ALG3, which is the fastest algorithm at F128. Roadnet F512 has no original-order or reordered
  cuSPARSE baseline. Most reordered algorithms at 1M F512 were also excluded, leaving DEFAULT.
- Eight BOO anchors in the 1M F128 `perm32x32` sub-cells (allocations 1 and 2) were not
  launch-verified. Kineto captured 0 events, and the Nsight audit only covered F >= 256. Those
  allocations use the other reordering's sub-cell for `perm_steady`.

### Matched-shape profiles (2026-11-10)

`shape_job.sh profile` captured ncu `--set full` of one call for 17 arms (`analysis/confirm/profile_confirm.txt`)
on ise-pheno-07. Each arm passed the float64 check before capture. Outputs are in
`analysis/confirm/profile/` (`report.txt`, `profile_summary.csv`, `profile_pairs.csv`). ncu locks
clocks, so profiled durations are diagnostic only. Predictions were frozen first, in
`analysis/confirm/profile_predictions.md`.

Winner / 32x32, dominant kernel:

| Pair | Time | Global ld sectors | Shared st | Shared ld | L2 reads | DRAM reads | RED sectors | Warp inst | Achieved occ |
|---|---|---|---|---|---|---|---|---|---|
| fem2d_permuted F512 | 0.80 | 0.52 | 0.54 | 1.04 | 0.52 | 0.62 | 0.78 | 0.93 | 1.00 |
| roadnet F128 | 0.80 | 0.61 | 0.61 | 0.93 | 0.61 | 1.01 | 0.85 | 0.80 | 1.00 |
| fem2d F512 | 0.85 | 0.52 | 0.53 | 1.04 | 0.52 | 0.78 | 0.78 | 0.93 | 1.00 |
| diag_b1.0 F128 (matched P/T/FPT) | 0.92 | 0.63 | 0.67 | 1.00 | 0.64 | 1.00 | 1.00 | 0.88 | 0.75 |
| diag_dd0.05 F512 (matched P/T/FPT) | 0.86 | 0.50 | 0.52 | 0.94 | 0.50 | 1.00 | 0.60 | 0.72 | 0.83 |

- **Staging hypothesis supported.** X-load sectors fall as predicted (0.52 vs 0.51 predicted on
  fem2d_permuted, 0.61 vs 0.60 on roadnet). Shared loads stay flat, and occupancy never rises.
- **diag_b1.0 is the cleanest case.** Atomics are unchanged and achieved occupancy drops 25%, yet the
  kernel is 8% faster with 0.63x X loads. Elsewhere, fewer atomics (0.60-0.85) may also contribute.
- **Shared memory per SM, from ncu's allocated size.** It rises to 96% on 256x64 (32.1 KB x 3 CTAs)
  and to 99% on diag_dd0.05 128x64. On diag_dd0.05 that costs a CTA (6 -> 5, achieved occupancy
  98 -> 82%), and the winner still gains.

Against cuSPARSE, dominant kernels:

| Pair | Warp inst | DRAM read + write | L2 reads | Issue active |
|---|---|---|---|---|
| fem2d_permuted F512: BOO winner vs reordered ALG2 | 1733M vs 1145M | 3.9 vs 3.4 GB | 2.1 vs 13.6 GB | 60% vs 35% |
| roadnet F128: BOO winner vs ALG3 | 1555M vs 767M | 3.2 vs 3.2 GB | 2.9 vs 3.6 GB | 62% vs 36% |
| fem2d F512: BOO winner vs ALG2 | 1733M vs 1158M | 3.7 vs 1.9 GB | 2.1 vs 12.6 GB | 60% vs 39% |

BOO now moves about as much DRAM as cuSPARSE on fem2d_permuted and roadnet, but executes 1.5-2x the
instructions. cuSPARSE reads X straight from L2 at 30-75% occupancy. The remaining gap to cuSPARSE
looks like instruction overhead more than memory traffic. This is inferred from counters, with no
source-level breakdown yet.

## Known risks

- Row and column labels come from independent clusterings, so a larger row group may not share
  column groups. If the gate shows this, the next variant is column-group-aware row clustering.
- Dense and banded inputs saturate: the X saving stops near k_r=64 on diag_b1.0.
- Longer CTAs may offset the X saving through load imbalance or a longer Step B.
- Padding changes `max_nnz_blk`, which only sets layout and arena size. It is a multiple of 32 so
  that X and later regions keep their offsets modulo 32 words. Tests and sanitizers check this path.

## Code

- `scripts/research_panel/shape_gate.py`, `shape_gate_report.py`: CPU gate and pass rule
- `scripts/research_panel/shape_manifest.py`: frozen manifests from the gate
- `scripts/research_panel/shape_job.sh`: Slurm runner, snapshot at `/home/dayanb/research/block-shape-20261106`
- `scripts/research_panel/shape_launch_audit.py`: Nsight Systems launch audit per F512 cell
- `scripts/research_panel/shape_report.py`: occupancy table, screen finalists, k by P matrix
- `tests/torch_fast_spdmm/benchmark/bench_shape.py`: GPU driver
- `tests/torch_fast_spdmm/test_boo_shape.py`: correctness tests

Inputs, X and float64 references are the hash-checked artifacts of the 2026-10-14 nnzflat run, and
the extension binary is the one validated there (sha256 checked at job start).
