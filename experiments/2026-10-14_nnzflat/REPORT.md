# BOO nnzflat FPT1/2 — results report (2026-10-14)

Plan: [README.md](README.md). Analysis tables and figures: [analysis/](analysis/).
Every number below is a complete-call median (CUDA events around one
`boo_matmul(launch_mode='auto')`, including Y allocation and zeroing, main kernel,
CSR remainder and join). Ratios are paired per-repeat medians within one cell
(same allocation, A, X and BOO object, independently shuffled repeat order).
Unless stated otherwise, a ratio is `t(other) / t(subject)`, so values above 1 mean
the subject is faster.

## Bottom line

**nnzflat does not become competitive.** No P, T, FPT or k setting makes it
faster than either the independently tuned row-segmented kernel or direct
cuSPARSE on any of the eight inputs, the held-out F256 widths, or the N=1,048,576
scale checks. Across three independent RTX3090 allocations, the best screened
nnzflat candidate is slower than the screen-tuned rowseg in 38/38 confirmed
comparisons (0.30–0.90×). It is also slower than the best direct cuSPARSE
algorithm in 37/37 (0.34–0.90×). Every one of these is a >5% regression with a
consistent direction in all three allocations. **FPT2 is a confirmed regression
for nnzflat**, not an improvement. No default should change. The nnzflat FPT
template is kept as an experimental, explicitly selected path
(`row_seg=False, features_per_thread=2, nnzflat_impl='template'`).

The closest cases are the sparse noisy diagonal (nnzflat 0.82–0.90× of tuned
rowseg) and webbase (0.84–0.88×). The worst are the dense diagonal (0.30–0.33×)
and pwtk/FEM (0.40–0.49×).

## Validity checks (all passed before interpretation)

| Check | Result |
|---|---|
| New nnzflat FPT tests (`test_boo_nnzflat_fpt.py`) | 293 passed |
| BOO suite (`test_boo`, `test_boo_edges`, `test_boo_fpt`) | 585 passed, 13 skipped (pre-existing RCM / 2-GPU skips) |
| PBR/CSR/optimizer regressions | 647 passed, 24 skipped |
| compute-sanitizer memcheck / racecheck / synccheck, boo_sanitizer subset (edges + rowseg FPT + nnzflat FPT) | 348 cases each, 0 errors / 0 hazards / 0 errors |
| CUDA-required gate with GPUs hidden | Failed as designed |
| SASS: template FPT1 vs original nnzflat | Opcode-identical in 30/30 T×P×dtype specializations |
| SASS: FPT2 | `LDS.64` float2 shared X load present; two `ATOMS.CAST.SPIN` CAS loops per item (shared float `atomicAdd` compiles to a CAS spin loop on sm_86, one per feature) |
| Launch attributes, all 75 float32 nnzflat/rowseg specializations | 0 bytes of local memory (no spills); registers and occupancy in [analysis/stage0/launch_attributes.csv](analysis/stage0/launch_attributes.csv) |
| Nsight Systems kernel-name audit | Stage 0 75/75; Stage 2 all 16 cells (2,872 feasible arms); Stage 3 all 34 frozen cells (387 arms, replayed once in a separate GPU job): every timed arm launched exactly its expected `<T, P_LOG2, FPT>` specialization |
| Numerical validation of every timed arm vs CPU float64 reference of the original matrix | All passed (row-wise float32 bound); 0 failed arms in any stage |
| Identical-work controls (reference vs template FPT1; duplicate arm) | Screen: 0.976–1.031; confirmation: 0.946–1.035 (258 control pairs, median 1.000) |

The Kineto (torch.profiler) launch check returned "no capture" for about half of
the Stage 2 arms and some Stage 3 arms in long processes. Those arms were
re-verified independently with the Nsight Systems audit above; none mismatched.

## Coverage

| Stage | Requested nnzflat configurations | Measured | Unsupported | Alias | Failed / missing |
|---|---:|---:|---:|---:|---:|
| 1 controlled panel | 480 | 456 | 24 | 0 | 0 |
| 2 application screen | 1,440 | 1,320 | 120 | 0 | 0 |

Every unsupported cell is an arena that exceeds the device's opt-in shared-memory
limit (101,376 B) at k128. That is P128 for all inputs, plus P64 for the two
dense-diagonal inputs. None was resized, re-labeled or rerouted; see
`analysis/coverage_stage{1,2}.csv`. No identity labeling produced identical labels
across k, so there are no aliases. Baselines and controls are outside these
counts: matched rowseg arms (Stage 1 480 measured; Stage 2 1,416 measured, 24
arena-infeasible) and identical-work controls (Stage 1 48; Stage 2 136 measured,
8 infeasible).

Confirmation: 34 frozen cells × 3 allocations (387 BOO arms per allocation plus
four direct cuSPARSE algorithms and torch CSR in every cell). In roadNet F512, the
original allocations excluded all four direct algorithms through the
memory-headroom check. That cell was re-measured in fresh processes on three new
allocations (labeled `_supp`; arms unchanged). Three algorithms then fit;
CSR_ALG3 remains a recorded resource exclusion. ALG3 was the fastest direct
algorithm at roadNet F128/F256, but its F512 performance is unmeasured. The
F512 comparison therefore uses the strongest measured algorithm; including
ALG3 could increase direct's advantage, but that is not established here.

## Hypotheses

### H1 — more CUDA threads reduce serial work per thread

**Partly supported, but it does not rescue nnzflat.** On the application screen
(Stage 2, 440 paired comparisons each), T256 beats T128 with geometric mean
1.48× (faster in 86%). T512 beats T256 with geometric mean 1.22× (faster in 63%,
slower in 28%). The T512 gain grows with P: P64 1.35× (faster in 70%), P128 1.56× (faster in all),
while at P8/P16 it is mixed (geomean 1.04/1.11, about half slower). The
controlled panel shows the same pattern with smaller gains (T256/T512 geomean
1.07×; P8 0.92×, P128 1.31×). At small P, fewer CTAs fit and per-thread work is
already short, so extra threads mostly add idle lanes and barrier cost. The
screened best nnzflat used T256 on 11 and T512 on 5 of 16 input/F cells. The
confirmed best-alternative comparisons (which include P64/T512 against
P32/T256) are unresolved, i.e. inside the control spread, in 22 of 41 cases.

### H2 — small P improves work distribution and reduces tile memory

**Rejected for P8; P32 is the sweet spot.** P8 is slower than P64 in 69% of
Stage 2 pairs (geomean `t(P64)/t(P8)` = 0.76) and 98% of Stage 1 pairs (0.58).
P16 is roughly neutral (Stage 2 1.08), and P32 is the best fixed choice (1.20×
over P64). P128 loses (0.70). Profiling shows why P8 loses even with 100%
theoretical occupancy (vs 33% at P64). The number of logical shared atomics is
identical, but shared-atomic wavefronts rise 4.3× (Stage 1, 52.7M vs 12.3M),
shared-load bank conflicts rise about 85×, and short-scoreboard stalls rise
2.8×. pwtk at k32 shows the same pattern (P8 3.4× the shared-atomic wavefronts of
P32; 50.3 vs 23.4 ms under ncu). With only 8 features per tile, a warp's 32 lanes
cover 4 nonzeros, often in the same row, so the CAS loops collide. P8 also needs
64 tile passes with three barriers each at F512.

### H3 — large BOO blocks supply enough work and reuse

**Rejected.** The best nnzflat time worsens with k on every dense-block input:
k128 is 1.4–2.5× slower than k32 (pwtk F512: 18.1 / 19.2 / 28.5 ms at
k32/64/128). Only the very sparse inputs prefer k64 (roadNet, diag_dd0.05), and
even there k128 regresses. The arena is sized from the global maxima, so at k128
pwtk runs at 1 CTA/SM (ncu achieved occupancy 16.7% vs 99.5% at k32; nsys kernel
median 54.7 ms vs 19.6 ms). P64/P128 become infeasible at k128 on dense blocks. The same
direction holds on the controlled synthetic panel, where only the grouping
changes: balanced F512 best nnzflat 0.65/0.60/0.73/0.94 ms at k16/32/64/128.
Changing k changes the representation and the kernel together. Block counts,
group sizes, arena and occupancy per k are in `analysis/stage*_best_by_k.csv`
and annotated in the [k curves](analysis/figures/k_curves_s2_F512.png).

### H4 — FPT2 saves sparse index/value loads and addressing

**Rejected: FPT2 is a confirmed nnzflat regression.** Paired FPT2-vs-FPT1 at
identical P/T/k: Stage 2 geomean `t(FPT1)/t(FPT2)` = 0.88; FPT2 is >5% faster in
1 of 660 pairs and >5% slower in 65%. The loss is largest at small P (P8 0.80,
P16 0.84, P64 0.95). Stage 1 agrees (0.88; P8 0.81). In confirmation, FPT1 beats
its FPT2 counterpart in 25 of 38 comparisons consistently across all three
allocations (1.03–1.09×); the remaining 13 are within the control spread. The
mechanism is visible in the counters. FPT2 halves the work items and does read X
as `LDS.64`, but each item still issues two scalar shared atomics, and each is a
CAS spin loop. Shared-atomic wavefronts rise 62–71% (Stage 1 P8: 85.6M vs 52.7M;
pwtk P32: 980M vs 574M), and shared-load bank conflicts rise about 4× on
pwtk/webbase. The index/value reuse is real but small compared with the atomic
cost. A scalar-load FPT2 ablation was not needed: FPT2 did not win, so there is
no gain to attribute.

### H5 — nnzflat benefits from uneven row lengths

**Not supported as a practical effect.** On the controlled pair (identical
dimensions, nnz, base-block nnz and values; only local row lengths differ), skew
slows both kernels. nnzflat's skew penalty `t(skewed)/t(balanced)` is geomean
1.11 (max 1.70); rowseg's is 1.19 (max 2.05). nnzflat is therefore less sensitive
to skew in aggregate (median per-configuration ratio of penalties 1.01, geomean
1.08), but the advantage is far smaller than rowseg's lead. Same-setting rowseg is
2.0× faster on skewed (geomean 0.494) and 2.2× faster on balanced (0.456). nnzflat
pays for skew too: skewed rows concentrate shared atomics on fewer smem_Y rows
(+29% shared-atomic wavefronts at P8). On the applications, the skew stress case
diag_skewed is one of nnzflat's worse inputs (0.42–0.45× of tuned rowseg).

## Same-setting and tuned comparisons

- **Same P/T/FPT/k (kernel effect):** rowseg is faster than nnzflat in 97.7% of
  1,320 Stage 2 pairs (geomean `t(rowseg)/t(nnzflat)` = 0.43). nnzflat wins only
  7 untuned small-P cells on the sparse noisy diagonal, by at most 1.17×.
- **Independently tuned:** the best rowseg beats the best nnzflat on all 16
  screened input/F cells and in every confirmed cell.
- **Global output atomics:** nnzflat also issues more global RED requests than
  rowseg. Its flush writes every row-group slot, while rowseg skips zero
  accumulators: 27.0M vs 16.3M L1 global RED requests on pwtk k32 F512, and
  62.4M vs 35.2M on webbase.

Confirmation medians (three allocations; roadNet F512 direct from the
supplementary allocations):

| Input | F | best nnzflat ms | best rowseg ms | best direct ms | nnzflat / rowseg speed | rowseg / direct speed |
|---|---:|---:|---:|---:|---:|---:|
| pwtk | 512 | 18.69 | 7.43 | 7.80 | 0.40 | 1.05 |
| webbase | 512 | 19.24 | 16.36 | 8.19 | 0.85 | 0.50 |
| roadNet | 512 | 36.16 | 26.68 | 22.01 | 0.74 | 0.83 |
| fem2d | 512 | 17.09 | 8.00 | 5.94 | 0.47 | 0.74 |
| fem2d_permuted | 512 | 17.73 | 8.47 | 12.50 | 0.48 | 1.48 |
| diag_b1.0 | 512 | 22.72 | 6.95 | 13.00 | 0.31 | 1.87 |
| diag_dd0.05 | 512 | 3.25 | 2.65 | 2.57 | 0.82 | 0.97 |
| diag_skewed | 512 | 26.96 | 11.34 | 12.07 | 0.42 | 1.07 |
| fem2d (N=1M) | 512 | 73.38 | 33.34 | 25.46 | 0.45 | 0.76 |
| diag_b1.0 (N=1M) | 512 | 94.19 | 28.28 | 53.78 | 0.30 | 1.90 |

All 34 cells (F128/F256/F512, N=1M) are in `analysis/stage3_best_median.csv` and
the [comparison figure](analysis/figures/stage3_nnzflat_rowseg_direct.png).
Rowseg's standing against direct cuSPARSE matches the earlier tile confirmation
(wins on pwtk, permuted FEM and dense diagonal; loses on natural FEM, webbase
and roadNet); nnzflat does not change it.

## Setup cost and break-even

nnzflat and rowseg use the same BOO object, so choosing nnzflat adds no setup
cost. It also brings no runtime benefit, so there is no break-even point against
rowseg or against direct cuSPARSE in any measured cell. For completeness,
`analysis/setup_costs.csv` lists clustering times from the saved October 13
labels (0.3 s for pwtk identity up to 74–99 s for N=1M HEM FEM) and BOO build
times (0.05–18 s, single-threaded on the preparation node).

## Classification of effects

- **Confirmed (three allocations, consistent direction, outside the control
  spread):** nnzflat slower than tuned rowseg and than best direct (every input,
  F and scale); nnzflat slower than rowseg at its own setting; FPT2 slower than
  FPT1 for nnzflat (25/38); the selected T512-over-T256 gain at small P (both
  arms far from competitive).
- **Screening only (one allocation):** threads/P/k trends across the full grid,
  P32 as the best fixed P, k32 (dense) / k64 (sparse) as the best k, and
  nnzflat's lower aggregate skew sensitivity.
- **Unresolved:** best candidate vs best alternative with a different P or k
  (22 of 41 inside the control spread); FPT2 vs FPT1 in 13 of 38 confirmation
  comparisons.

## Scope notes and deviations

- **Concurrency:** the Eran confirmation job was still running at the start, so
  only one of these GPU jobs ran until it finished; afterwards two. GPU jobs
  requested one RTX3090, 48 GB and no CPU count. Slurm's DefCpuPerGPU=6 and
  MaxMemPerCPU=8 GB mean 48 GB keeps the default 6 CPUs, whereas the earlier
  56 GB request had been raised to 8 CPUs. `OMP_NUM_THREADS=1` throughout.
- **BOO reconstruction on GPU nodes:** BOO has no deserialization path, so GPU
  jobs rebuilt each object from the saved labels (0.05–18 s single-threaded,
  diag_skewed N=1M the largest). Each rebuild had to reproduce the CPU-job BOO
  digest. Inputs, labels, structure statistics and float64 references were
  produced only by CPU jobs and SHA-256-checked by the GPU jobs.
- **Kineto launch check:** often empty in long processes (above). The added
  Nsight Systems audit (`scripts/research_panel/nnzflat_launch_audit.py`)
  verified every Stage 2 and Stage 3 arm.
- **roadNet F512 direct baselines:** re-measured on supplementary allocations;
  CSR_ALG3 remains excluded (above).
- **Stage 1 synthetic design:** N16384, 16×16 base blocks in a ±2 block band, 64
  nonzeros per base block. Balanced: 4 per local row. Skewed: local rows
  16,16,16,2,2,2,1×10, with heavy rows fixed per base block-row. Contiguous
  labels coarsened to k16–k128. Both patterns have 327,296 nnz, identical block
  counts per k, no remainder, and identical value streams.
- **Not done:** an additional labeling method or column-major nnzflat follow-up.
  The plan made these conditional on a winning labeling worth revisiting, and
  there is none.
