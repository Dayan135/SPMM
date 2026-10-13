# BOO row-segmented kernel: profiling, P × FPT, slab width and confirmation — 2026-10-12

The checkout retains the report tables and figures. Per-call samples and raw
profiler captures are in the separate [result archive](../../RESULTS_ARCHIVE.md).

Executes [the profiling continuation task](../../scripts/research_panel/PROFILING_CONTINUATION_TASK.md)
for the plan in [NEXT_EXPERIMENTS.md](../../NEXT_EXPERIMENTS.md). Workload: one fixed sparsity
pattern reused across many SpMM calls, batch 1, float32, threads 256, row-major rowseg, auto
launch, remainder threshold 4, seed 0, saved full-sweep labels (never clustered during GPU timing).
Unprofiled interleaved complete-call timings decide performance. Profiler durations are used
only for diagnosis.

Status: **complete.** Confirmed on three independent RTX 3090 allocations (§7).
Results, raw samples, profiler exports and reproduction:
[`tests/torch_fast_spdmm/benchmark/results/tiles_20261012/`](../../tests/torch_fast_spdmm/benchmark/results/tiles_20261012/README.md).

## 1. Summary

- **FPT1 is limited by the shared-memory pipe; FPT2/4 remove that limit.** At F512 with P64,
  FPT1 runs L1/TEX at 96–99.7% of peak with 30–55% MIO-throttle stalls. FPT2/FPT4 compile the X
  reads to `LDS.64`/`LDS.128`. They cut executed shared loads 1.9×/3.1× on pwtk (655M → 347M → 210M).
- **FPT4 moves the bottleneck to global atomics.** The number of RED requests is unchanged, but the per-thread
  `atomicAdd(Y[f0+j])` loop is lane-strided. **L2 sectors per RED request are 4 / 8 / ~16 for
  FPT1/2/4** (cuSPARSE: 4). This is why FPT4 regresses on atomic-heavy inputs.
- **Confirmed (3 allocations, randomized order), F512 complete call:**

  | Case | Best confirmed variant | vs FPT2/P64/s512 anchor | vs strongest same-cell cuSPARSE |
  |---|---|---|---|
  | pwtk (identity k32) | P64/FPT4 | 1.09–1.14× | 1.16–1.20× |
  | pwtk (HEM k32) | P64/FPT4 | 1.16× | 1.16–1.21× |
  | dense diagonal 262k / 1M | P128/FPT4 | 1.38–1.44× / 1.44–1.48× | 2.65–2.67× / 2.74–2.81× |
  | permuted FEM 262k / 1M | P64/FPT4 | 1.06–1.07× / 1.07–1.09× | 1.54–1.55× / 1.56–1.58× |
  | natural FEM 262k / 1M | P64/FPT4 | 1.10–1.13× / 1.12× | **0.80–0.82× / 0.84–0.87×** (cuSPARSE wins) |
  | webbase | P64/FPT2 (anchor) | FPT4 0.86× (regression confirmed) | **0.50×** |
  | roadNet (METIS k64) | P64/FPT4 ≈ anchor | 1.01–1.02× | **0.67×** (CSR_ALG3) |
  | diag_dd0.05 (held out) | P64/FPT2 (anchor) | FPT4 0.92× (regression) | **0.95–0.97×** (CSR_ALG3) |

- **FPT4 is not a universal default.** It wins on dense-block inputs (pwtk, dense diagonal,
  FEM) but regresses on webbase (−14%) and on the held-out sparse noisy diagonal (−8%). Both
  regressions match the atomic-sector mechanism. FPT2 remains the conservative choice;
  per-matrix selection is needed.
- **P: keep 64**, except the dense diagonal. There, P128/FPT4 removes P64/FPT4's two-row-warp
  shared-memory bank conflicts (80M → 13M) and is faster in 12/12 paired comparisons (+0.6% to +6.4%,
  ~4% typical). That effect is consistent, but its size overlaps the identical-work control spread (below). P128 costs
  5–41% everywhere else (shared-memory occupancy limit 4 CTAs/SM; roadNet 2).
- **Slab width: keep 512.** Narrower slabs repeat sparse staging per slab: slab128 0.87–0.97×,
  slab256 0.92–1.01× at F512/1024. slab1024 at F1024 is within ±3% of slab512.
- **Timing-order sensitivity (methodological).** The inherited rotation order
  (`bench_spmm.round_orders`) runs each variant after a nearly fixed predecessor. In a direct
  control, identical work then differed by up to 7% depending only on the order seed. All conclusions here use per-repeat
  shuffling. Under shuffling, the identical-work control across 75 confirmation cells has median
  0.999 and range 0.961–1.035, with allocation-dependent sign. This is the measured within-cell spread for this protocol, not a universal
  error bound. Rankings from rotation-ordered runs that differ by less than this, including
  the full FPT sweep's small effects, should be treated as provisional.
- **Recommended next experiment:** coalesce the FPT>1 output atomics by redistributing outputs
  across lanes at identical P/FPT/slab ([§9](#9-recommended-next-experiment)).

## 2. What changed (code)

Experiment parameterization only; no default changed; nothing committed.

| File | Change |
|---|---|
| `torch_fast_spdmm/kernels/boo_kernel.cu` | `FEAT_BLOCK` template parameter on `boo_spmm_rowseg_kernel` (default `BOO_FEAT_BLOCK`=512); process-wide `set/get_boo_feat_block` (128/256/512/1024); non-512 widths instantiated for float32/threads256 only, other combinations rejected (never substituted); `static_assert(FEAT_BLOCK % P == 0)`; kernel selection factored into one function shared by the launcher and the new `boo_rowseg_launch_info` (effective P, arena, grid.z, `cudaFuncAttributes`, occupancy API) |
| `torch_fast_spdmm/kernels/boo_kernels.hpp` | declarations for the above |
| `torch_fast_spdmm/bindings/bindings_boo.cu` | binds `set/get_boo_feat_block`, `boo_arena_bytes_rowseg_cuda`, `boo_rowseg_launch_info` |

**Defaults preserved.** All 90 slab512 specializations present in both binaries
(float/double × threads × P × FPT, plus the diagnostic raw-store kernel) have **byte-identical normalized SASS** to
the validated full-sweep binary (`sass/default_vs_validated.csv`). The pre-existing uncommitted
FPT edits differed from the validated build by whitespace only and are preserved.

Measured build: extension SHA-256 `b980dbc28216ed2169adee4cbe0db83b6735e78489cab76db21a6ea6c61c229c`
(validated full-sweep binary: `bc688ff8…a607c`). Source hashes: `build_hashes.json`. The local
kernel/bindings are byte-identical to the measured snapshot.

New drivers (the real-agent sources are not edited):
- `tests/torch_fast_spdmm/benchmark/`: `tiles_common.py`, `bench_tiles.py`, `profile_tiles.py`, `test_tiles.py`.
- `scripts/boo_tile_experiments/`: `job.sh`, `profile_cases.sh`, `analyze_profiles.py`, `audit_launches.py`,
  `sass_summary.py`, `report_tiles.py`, `finalist_resources.py`, `confirm_report.py`, `plot_confirm.py`.

## 3. Correctness and launch verification

- **GPU suite** (job 22711787): 992 passed, 13 skipped. It covers test_tiles, test_boo_fpt, test_boo,
  test_boo_edges, test_boo_plan, test_cusparse_direct and test_real_matrices. test_tiles +
  test_boo_fpt alone: 417 passed, 0 skipped.
- **compute-sanitizer** (22712100) on the `boo_sanitizer` subset of test_tiles (partial slabs, P
  reduction, unaligned `smem_X` offsets, >48 KiB arena; 66 tests per tool): memcheck 0 errors,
  racecheck 0 hazards, synccheck 0 errors.
- **Float64 reference with per-row bound, no violations:**
  - F130: all 7 cases × 12 P/FPT (84 variants).
  - F1000/F1025: slab 128/256/512/1024 × P32/64/128 × FPT1/2/4 on pwtk, dense diagonal, FEM and
    permuted FEM (288 variants, grid.z up to 9).
  - Every timed variant in every run.
- **Launch identity:** the nsys audit (22712709) placed an NVTX range around every validation call
  for all 48 P×FPT×slab specializations on the six tuning cases at F128/F512, both scopes.
  **1,140 BOO calls, 0 mismatches:** each launched exactly the requested `rowseg<float, 256, P, false, FPT, slab>`.
  torch.profiler name checks pass whenever Kineto captures events. Long processes intermittently
  capture zero CUDA events; those are recorded as `UNVERIFIED`, never as a pass. Profiled
  variants' ncu/nsys kernel names match their requests. `DIAG_RAW_STORE` is false everywhere.

## 4. Profiling diagnosis (F512, P64/slab512 unless stated)

**Passes and collection notes.**
- Nsight Systems, the `optimizer.performance.METRICS` stall set, occupancy (LaunchStats/Occupancy +
  `OCCUPANCY_METRICS`), memory (SpeedOfLight/MemoryWorkloadAnalysis/SchedulerStats), instruction
  mix and an atomic-sector pass were each separate invocations.
- `MemoryWorkloadAnalysis_Tables` printed nothing without `--print-details`, so bank conflicts come from
  `l1tex__data_bank_conflicts_*`.
- `sm__pipe_lsu_cycles_active…pct_of_peak_sustained_active` and `smsp__barrier_count.sum` returned
  `n/a` on ncu 2025.2.1 and are recorded as unavailable.
- Registers are 39–40/thread with **no local-memory spills** (0 local ld/st instructions, 0 local bytes) in every BOO variant.

| Case | Limiting cost (evidence) | Controlled timing comparison | Uncertainty | Next action |
|---|---|---|---|---|
| pwtk | FPT1: shared pipe (L1 96%, MIO 30%, 655M LDS). FPT4: global side — long-scoreboard 29%, L2 throughput 73%, RED 4 → 15.4 sectors/request, barrier stalls 18% | FPT1/2/4 (confirmed): 11.9 / 7.7 / 6.9 ms; P128 or P32 at FPT4: no gain | Share of FPT4's remaining time from strided atomics vs. X staging not separated | Coalesced atomics |
| webbase | Atomic/L2 latency on tiny blocks (median 22 nnz/block, 67% of staged X columns unused): long-scoreboard 45%. FPT4: RED sectors 141M → 564M, L2 bytes 17 → 26 GB. Y memset 2.3 ms (14%); CSR remainder 1.1 ms, mostly serialized after the main kernel | FPT2 → FPT4: 16.5 → 19.2 ms (0.86×, confirmed) | Remainder serialization vs. GPU saturation not separated | Coalesced atomics; then memset/remainder |
| roadNet | Global-memory latency: long-scoreboard 38–49%, DRAM 57–63%, 69% unused staged X, median 13 nnz/block. Memset 4.5 ms (16%); remainder 1.35 ms serialized. BOO main kernel alone 20.5 ms vs cuSPARSE ALG3's whole call 18.0 ms (DRAM 79%) | FPT1/2/4: 28.8 / 27.3 / 26.9 ms; P32 ≈ P64; P128 0.60× | Arena caps P64 at 5 CTAs/SM (83%), but P32 (6 CTAs) is not faster, so occupancy is not limiting | Not a tile-tuning case; keep CSR |
| dense diagonal | FPT1: shared pipe saturated (L1 99.7%, MIO 55%, 839M LDS). P64/FPT4: 80M shared bank conflicts (16% of load wavefronts) from 2-row warps | P64 → P128 at FPT4: conflicts 80M → 13M; +0.6% to +6.4% in 12/12 paired confirmations, while theoretical occupancy drops 100% → 67% | Gain size overlaps the control spread | Coalesced atomics (RED 16 sectors/request) |
| natural FEM | BOO main 6.7 ms + 0.6 ms memset vs cuSPARSE ALG2 6.1 ms total. ALG2 exploits native locality (L2-bound 102%, DRAM 1.39 GB); BOO reads 2.5× more DRAM (3.55 GB) with 54% unused staged X | FPT4 1.10–1.13× over the anchor, still 0.80–0.82× of the strongest direct (1M: 0.84–0.87×) | DRAM excess split between staging and atomic write-back not measured | Setup-time selection should keep CSR |
| permuted FEM | cuSPARSE is DRAM-bound (94%, 9.4 GB) from permuted X; BOO's clustering restores locality (4.7 GB) | BOO 1.54–1.58× over strongest direct | — | Coalesced atomics |

**Timeline (nsys, one call).** A Y memset precedes every BOO call: 0.50 ms on pwtk, 0.60 ms on the
262k generated inputs, 2.3 ms on webbase, 4.5 ms on roadNet. cuSPARSE ALG2/ALG3 have no separate
memset; DEFAULT runs a 0.5 ms β-scaling kernel. Host time before the first GPU work is ~0.3 ms for
BOO (22 runtime calls with a remainder, 3 without) vs ~0.15 ms for direct. Overlapping kernel
durations are never summed as call latency; span and union are reported (`profile_f512/analysis/nsys_calls.csv`).

**SASS (static).** FPT2/FPT4 replace 35 scalar `LDS` with 35 `LDS.64`/`LDS.128` and double/quadruple the
FFMA count per unrolled loop. Per-thread `RED` count scales with FPT (1/2/4). Instruction counts
are identical across slab widths (`sass/rowseg_summary.csv`). This is evidence that the FPT gain
comes from vector shared loads plus reuse of each sparse value/index load. A separate scalar-vs-vector
ablation was not needed.

## 5. P × FPT screen (slab512, threads256, F128/512)

Randomized order (job 22713030). Speedup heatmaps annotate arena, CTAs/SM and registers:
[full call](plots/screen_heatmap_full_call_22713030.png),
[C++ scope](plots/screen_heatmap_cpp_22713030.png).
These are single-allocation complete-call medians (ms). Rows not repeated in §7 (the F128 cells, P16/P32) are
**provisional**; differences below the §7 control spread are not rankings.

| Case | F | FPT1/P64 | FPT2/P64 (anchor) | FPT4/P64 | best P/FPT | best vs anchor | best vs strongest direct |
|---|---:|---:|---:|---:|---|---:|---:|
| pwtk | 128 | 2.94 | 2.04 | 1.91 | P64/FPT4 | 1.070 | 1.05× (CSR_ALG2) |
| pwtk | 512 | 11.76 | 7.47 | 6.82 | P64/FPT4 | 1.095 | 1.19× (CSR_ALG2) |
| webbase | 128 | 4.37 | 4.17 | 4.89 | P64/FPT2 | 1.000 | 0.51× (CSR_ALG3) |
| webbase | 512 | 16.97 | 16.44 | 19.18 | P64/FPT2 | 1.000 | 0.50× (CSR_ALG3) |
| roadNet | 128 | 7.64 | 7.26 | 7.07 | P32/FPT4 | 1.052 | 0.65× (CSR_ALG3) |
| roadNet | 512 | 29.18 | 27.60 | 27.33 | P64/FPT4 | 1.010 | see §7 |
| dense diagonal | 128 | 3.18 | 1.81 | 1.34 | P128/FPT4 | 1.402 | 2.51× (DEFAULT) |
| dense diagonal | 512 | 12.58 | 6.84 | 4.96 | P128/FPT4 | 1.409 | 2.71× (CSR_ALG2) |
| natural FEM | 128 | 2.86 | 2.11 | 1.91 | P64/FPT4 | 1.104 | 0.77× (CSR_ALG2) |
| natural FEM | 512 | 11.26 | 8.05 | 7.26 | P64/FPT4 | 1.109 | 0.83× (CSR_ALG2) |
| permuted FEM | 128 | 3.09 | 2.36 | 2.19 | P64/FPT4 | 1.074 | 1.00× (CSR_ALG3) |
| permuted FEM | 512 | 11.46 | 8.42 | 8.00 | P64/FPT4 | 1.053 | 1.56× (CSR_ALG2) |

P16 is slower than P64 at the same FPT in every case (0.54–0.93× of the anchor); P128 is the slowest P on roadNet and webbase. Requested P equals
effective P for F ≥ 128. No infeasible arena at slab512. The earlier rotation-ordered screen (job
22711918) is kept for provenance only.

## 6. Slab width and F1024

Randomized order (job 22713031): P32/P64/P128 × FPT4 and P64/FPT2, widths 128/256/512/1024,
F128/512/1024, on pwtk (identity and HEM), dense diagonal and natural/permuted FEM.
[Plot](plots/slab_full_call.png).
At fixed P the arena is unchanged; only grid.z (8/4/2/1 at F1024) and per-slab restaging of
sparse values/indices/ids change.

- slab128: 0.87–0.97× of slab512 at F512/1024. slab256: 0.92–1.01×. Losses are largest at
  P128, which has the fewest tiles per slab to amortize restaging.
- slab1024 vs slab512 at F1024: 0.97–1.015×, no consistent gain. slab512 stays the default.
- F128 identical-work control (every width runs one slab): within ±2%.
- The rotation-ordered stage A (22712093) suggested up-to-7% slab1024 gains. These did not
  reproduce under shuffling and are attributed to order sensitivity.
- roadNet/webbase were not added: the profiles show atomics/latency, not parallelism, limiting
  them, and F1024 outputs would need 8 GB per Y buffer on roadNet.

## 7. Independent confirmation

**Setup.**
- Three allocations, each on its own RTX 3090 node: jobs 22713805 (cs-3090-05), 22713806
  (cs-3090-04, part of its run alongside the audit job on another GPU of that node) and 22713807 (cs-3090-03).
- Per-repeat shuffled order, distinct order seeds; 5 warmups + 20 repeats per variant; raw samples saved.
- In every cell: finalists P64/FPT4/s512 and P128/FPT4/s512, the anchors P64/FPT2/s512 and P64/FPT1/s512, and
  the identical-work control P64/FPT4/s1024 at F512.
- **All four cuSPARSE algorithms were measured in every allocation and cell** (`direct_coverage.csv`).
  roadNet and the 1M ALG3 runs used separate processes that repeat all BOO variants. Each ratio is paired within one process; the strongest
  direct is the algorithm giving the variant its lowest paired speedup.

[Figure](plots/confirm_vs_direct_f512.png);
per-allocation rows in `confirm_per_allocation.csv`, C++-scope rows alongside (same conclusions).

| Case | F | Variant | median ms (alloc 1/2/3) | vs anchor | vs strongest direct | strongest direct |
|---|---:|---|---|---|---|---|
| pwtk | 512 | P64/FPT2 (anchor) | 7.77 / 7.65 / 7.69 | 1 | 1.01–1.09 | CSR_ALG2/DEFAULT |
| pwtk | 512 | **P64/FPT4** | 6.79 / 6.93 / 7.07 | 1.09–1.14 | **1.16–1.20** | CSR_ALG2/DEFAULT |
| pwtk | 1024 | **P64/FPT4** | 14.19 / 14.40 / 14.21 | 1.09–1.12 | **1.20–1.22** | DEFAULT |
| pwtk HEM | 512 | **P64/FPT4** | 6.83 / 6.76 / 6.91 | 1.16 | **1.16–1.21** | CSR_ALG2/DEFAULT |
| dense diagonal | 512 | P64/FPT4 | 5.01 / 5.07 / 5.17 | 1.36–1.42 | 2.56–2.63 | CSR_ALG2/DEFAULT |
| dense diagonal | 512 | **P128/FPT4** | 4.95 / 5.00 / 4.95 | 1.38–1.44 | **2.65–2.67** | CSR_ALG2/DEFAULT |
| dense diagonal | 1024 | **P128/FPT4** | 9.73 / 9.90 / 9.86 | 1.45–1.51 | **2.76–2.85** | CSR_ALG2/DEFAULT |
| dense diagonal 1M | 512 | **P128/FPT4** | 20.17 / 20.39 / 19.57 | 1.44–1.48 | **2.74–2.81** | CSR_ALG2/DEFAULT |
| permuted FEM | 512 | **P64/FPT4** | 8.07 / 8.11 / 8.11 | 1.06–1.07 | **1.54–1.55** | CSR_ALG2/DEFAULT |
| permuted FEM | 1024 | **P64/FPT4** | 16.22 / 16.34 / 16.29 | 1.07–1.08 | **1.51–1.52** | CSR_ALG2/DEFAULT |
| permuted FEM 1M | 512 | **P64/FPT4** | 32.99 / 32.67 / 32.84 | 1.07–1.09 | **1.56–1.58** | CSR_ALG2/DEFAULT |
| natural FEM | 512 | P64/FPT4 | 7.19 / 7.33 / 7.47 | 1.10–1.13 | 0.80–0.82 | CSR_ALG2/DEFAULT |
| natural FEM 1M | 512 | P64/FPT4 | 30.59 / 30.27 / 30.48 | 1.12 | 0.84–0.87 | CSR_ALG2/DEFAULT |
| webbase | 512 | P64/FPT2 (anchor) | 16.47 / 16.56 / 16.47 | 1 | 0.50 | CSR_ALG3 |
| webbase | 512 | P64/FPT4 | 19.18 / 19.36 / 19.22 | **0.86** | 0.43 | CSR_ALG3 |
| roadNet | 512 | P64/FPT4 | 26.86 / 26.98 / 26.93 | 1.01–1.02 | 0.67 | CSR_ALG3 |
| diag_dd0.05 (held out) | 512 | P64/FPT2 (anchor) | 2.64 / 2.70 / 2.65 | 1 | 0.95–0.97 | CSR_ALG3 |
| diag_dd0.05 (held out) | 512 | P64/FPT4 | 2.87 / 2.92 / 2.89 | **0.92** | 0.88–0.89 | CSR_ALG3 |

**Tuning vs. held-out.**
- *Tuning cases* (pwtk identity, webbase, roadNet, the 262k generated inputs): the confirmation repeats them on new allocations.
- *Not tuned on:* pwtk HEM and the 1M inputs, which reuse saved labels. They confirm the pattern: pwtk HEM is now pwtk's best,
  matching the full-sweep winner's labeling.
- *Truly held out:* diag_dd0.05. The anchor is best there, FPT4 regresses, and BOO still loses to CSR_ALG3. So the gains
  do not transfer to sparse noisy diagonals.

**Identical-work control** (P64/FPT4 slab512 vs slab1024 at F512; 75 cells, both scopes):
median 0.999, range 0.961–1.035, 16/75 beyond ±2%. The sign clusters by allocation (allocs 1–2
mostly <1, alloc 3 >1). Differences of a few percent within one cell are therefore not
resolvable with this protocol. The confirmed gains (≥6% vs anchor, reproduced in all three
allocations with the same sign) exceed this spread. The P128-vs-P64 diag difference does not
clearly exceed it.

## 8. Finalist resources

[`finalist_resources.csv`](../../tests/torch_fast_spdmm/benchmark/results/tiles_20261012/report_shuffle/finalist_resources.csv)
(from the launcher's own report; ncu values where profiled).
- Limits use sm_86 rules: 65536 registers with 256-register warp granularity, 100 KiB shared
  carveout plus 1 KiB per-CTA reservation, 48 warps, 16 CTAs.
- The computed minimum matches `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on all 36 rows.
- Theoretical occupancy is the resource limit; achieved occupancy is the ncu-measured average.

| Variant | Dynamic smem/CTA: pwtk / diag / FEM / roadNet | Regs/thread | Spills | Theoretical CTAs/SM (binding limit) | Theoretical occ. | Achieved occ. (ncu, F512) |
|---|---|---:|---|---|---:|---:|
| P64/FPT2 (anchor) | 13552 / 14740 / 12628 / 18552 B | 40 | none | 6 (registers + warps; diag also shared); roadNet 5 (shared) | 100% (roadNet 83.3%) | 98.8–99.6% (roadNet 83.1%) |
| P64/FPT4 | same | 39 | none | same | 100% (roadNet 83.3%) | 98.8–99.6% (roadNet 83.1%) |
| P128/FPT4 | 21744 / 22932 / 20820 / 34936 B | 39 | none | 4 (shared memory); roadNet 2 | 66.7% (roadNet 33.3%) | 66.1% (diag) |

High achieved occupancy does not mean efficient execution. The P64 variants sit near 99%
while issuing only 0.44–0.68 warps per scheduler per cycle.

## 9. Recommended next experiment

**Coalesced FPT>1 output atomics at identical P/FPT/slab.** Today each FPT>1 thread issues FPT
scalar atomics to adjacent addresses, so each warp-level RED instruction is lane-strided
(8 or ~16 sectors/request vs 4 for FPT1 and cuSPARSE). Before the atomics, redistribute the
accumulators across the FPT lanes of a group with an FPT×FPT `__shfl_sync` transpose. Each RED
instruction then covers 32 contiguous floats. Arithmetic, X staging and the vector shared loads
are unchanged.

**Design.**
- Arms: identical P/FPT/slab with and without redistribution, at P64/FPT2, P64/FPT4 and P128/FPT4, slab512, F512 and F1024.
- Cases: pwtk, webbase (the regression), dense diagonal, permuted FEM, diag_dd0.05.

**Report.**
- RED requests and `lts__t_sectors_op_red` (target: 4 sectors/request), and total L2/DRAM bytes.
- The added SHFL instruction count and the MIO/short-scoreboard stall change it causes.
- Shuffled-order complete-call and C++ timings with all four direct algorithms per cell, on three allocations, with the identical-work control.

**Success criterion.** FPT4 no longer regresses on webbase/diag_dd0.05 and gains beyond the
control spread on pwtk/FEM.

**Deferred without current evidence.**
- Threads sweep: no register or launch limit binds.
- Scalar-vs-vector ablation: SASS plus LDS counts already attribute the FPT gain.
- Active-column compaction: roadNet is staging-heavy, but no tile-level lever helped. Revisit only if
  staged-X traffic dominates after the atomic fix.
- batch>1.
- Memset elimination and remainder overlap are the next fixed costs on webbase/roadNet.

## 10. Setup cost and break-even (F512 unless noted, medians of the 3 allocations)

Setup cost = labels (clustering; identity labels still cost the feasible-first grouping pass) +
BOO build + GPU staging, from the full-sweep records. Each is a single measurement on its own node.
cuSPARSE plan creation (2–28 ms) and variant selection (timing ~5 candidates a few times,
≲0.5 s) are small next to these.
Break-even calls = setup / (strongest-direct ms − BOO ms).

| Case | Variant | BOO ms | strongest direct ms | saving/call | setup (s) | break-even calls | if labels reused |
|---|---|---:|---:|---:|---:|---:|---:|
| pwtk (identity) | P64/FPT4 | 6.93 | 8.25 | 1.31 ms | 0.98 | ~750 | ~420 |
| pwtk (identity), F1024 | P64/FPT4 | 14.21 | 17.34 | 3.13 ms | 0.98 | ~310 | ~180 |
| pwtk (HEM) | P64/FPT4 | 6.83 | 8.21 | 1.38 ms | 25.9 | ~18,800 | ~390 |
| dense diagonal 262k | P128/FPT4 | 4.95 | 13.23 | 8.28 ms | 1.26 | ~150 | ~100 |
| dense diagonal 1M | P128/FPT4 | 20.17 | 55.79 | 35.6 ms | 4.00 | ~110 | ~60 |
| permuted FEM 262k | P64/FPT4 | 8.11 | 12.49 | 4.38 ms | 16.1 | ~3,700 | ~80 |
| permuted FEM 1M | P64/FPT4 | 32.84 | 51.54 | 18.7 ms | 101.3 | ~5,400 | ~130 |
| natural FEM, webbase, roadNet, diag_dd0.05 | best BOO | — | — | negative | — | never | never |

Tuning FPT2 → FPT4 itself needs no new setup (same BOO object). Choosing it per matrix costs a few
timed calls. pwtk identity vs HEM: HEM is ~1.4% faster per call (within the control spread) but its 25 s clustering needs
~18,000 extra calls to pay back, so identity labels are the practical choice for pwtk.

## 11. Handoff

**Edited (uncommitted, not pushed):** `torch_fast_spdmm/kernels/boo_kernel.cu`,
`torch_fast_spdmm/kernels/boo_kernels.hpp`, `torch_fast_spdmm/bindings/bindings_boo.cu`
(on top of the pre-existing uncommitted FPT edits, which are preserved).

**New:**
- `tests/torch_fast_spdmm/benchmark/{tiles_common,bench_tiles,profile_tiles,test_tiles}.py`
- `scripts/boo_tile_experiments/*`
- `TILE_EXPERIMENTS.md`
- `tests/torch_fast_spdmm/benchmark/results/tiles_20261012/`

**Remote root:** `/home/dayanb/research/boo-tiles-20261012` (isolated snapshot, in-place build,
PYTHONPATH only; the shared installed extension was not touched).

**Jobs:**

| Job(s) | Role |
|---|---|
| 22711735 | Build |
| 22711787 | Test suite |
| 22711817 | SASS dump and default-vs-validated comparison |
| 22711822 | Smoke run |
| 22711843 | Structure (CPU) |
| 22711909 | Profiler smoke |
| 22711841 | Screen (failed on a Kineto false negative, fixed) |
| 22711918 | Rotation-ordered screen |
| 22712009 | F130 correctness |
| 22712092, 22712180 | roadNet baseline fills |
| 22712093 | Rotation-ordered slab stage A |
| 22712094 | F1000/F1025 correctness |
| 22712096–8, 22712699–700 | Profiles |
| 22712100 | Sanitizer |
| 22712897 | Order control (22712703 was cancelled) |
| 22712709 | nsys launch audit |
| 22713030 | Randomized P×FPT screen |
| 22713031 | Randomized slab screen |
| 22713805–7 | Three-allocation confirmation |

**Completed:** every item of the task (§§3–8, 10).

**Not done:**
- **Default changes:** none were made, deliberately. The evidence supports per-matrix selection
  between CSR, BOO FPT2 and BOO FPT4/P128-diag rather than a new global default.
- **Not part of this experiment:** batch>1, thread sweeps, compaction.
- **Known methodology issue for other harnesses:** `bench_spmm.round_orders` (used by `bench_real` and the full sweep)
  should be replaced by per-repeat shuffling before small effects from those harnesses are relied on.

**One evidence-supported next experiment:** coalesced FPT>1 output atomics (§9).
