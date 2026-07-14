# Last Experiments

## 1. Occupancy-guided `target_groups` (2026-06-24, commit `a1dbf5b`)

### Motivation

`_build_group_row_starts` computed `target_groups = sm_count × 8`, where 8 came from
`_BLOCKS_PER_SM_EST=4 × _PIPELINE_DEPTH=2` — a hardcoded guess. The real blocks-per-SM for
the rowgroup kernel on an RTX 3090 is 11-12 (varies by `block_rows` and `smem_size`), so
the GPU was being under-filled by ~30-50%.

### Implementation

- `pbr_kernel.cu`: added `get_pbr_rowgroup_blocks_per_sm(block_rows, smem_size)` using
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
  - **Critical**: pass `dynamicSMemSize=0`. The kernel uses only static `__shared__` arrays.
    Passing the array byte size as `dynamicSMemSize` double-counts shared memory and the
    occupancy API returns bps=5 instead of 11. The driver reads static smem from the binary.
- `bindings_kernels.cu`: exposed as `m.def("pbr_rowgroup_blocks_per_sm", ...)`.
- `__init__.py`: removed `_BLOCKS_PER_SM_EST` and `_PIPELINE_DEPTH` constants;
  `_build_group_row_starts` now calls `core.pbr_rowgroup_blocks_per_sm(block_rows, smem_size)`
  at PBRMatrix `.to("cuda")` time.

### Results on RTX 3090

bps: 8 (estimated) → 11-12 (real, varies by config).

Profile comparison (commit `a1dbf5b` vs prior):

| case | waves/SM before | waves/SM after | time before | time after |
|---|---|---|---|---|
| dense2 262144 f512 rowgroup | 0.73 | 0.99 | ~1642 µs | ~1497 µs (~9% faster) |
| scat8x1 262144 f512 rowgroup | 2.78 | 3.92 | ~548 µs | ~548 µs (unchanged) |

Scat8x1 is unchanged because per-block work density (~17 PBR blocks/group) is the bottleneck,
not grid size.

### Benchmark sweep

3×1216 tests (sizes 16384 / 65536 / 262144), W&B runs `a1dbf5b_16384`, `a1dbf5b_65536`,
`a1dbf5b_262144`. Plots: `plots/occupancy/`.

Key findings from `rowgroup_vs_block` heatmap at 262144:

| matrix | f64 | f128 | f256 | f512 |
|---|---|---|---|---|
| scattered_dense2 | 1.24× | 1.29× | 1.41× | **1.83×** |
| scattered_8x1    | 0.94× | 0.99× | 0.99× | 1.00× |
| scattered_2x4    | **0.88×** | 0.95× | 0.98× | 0.98× |
| diag_dd0.001     | 1.00× | 1.00× | 1.00× | 1.00× |

- `scattered_dense2` at high features: biggest winner — atomicAdd contention + good group fill.
- `scattered_2x4` at f64: slight regression (0.88×) — occupancy fix over-partitions when
  features are narrow (feat_tiles=1) and blocks-per-group drops to ~12 (was 17).
- `diag` matrices: flat 1.0× — see Experiment 2 below.

---

## 2. Diagonal matrix profiling (2026-06-24, commit `04858b7`)

### Question

Why does rowgroup show exactly 1.0× vs block on all diagonal matrices in the benchmark?

### Setup

- `profile_pbr.py` extended: `_MATRIX_PROFILES` refactored to `(kind, cfg)` tuples;
  added `diag_dd0.001/002/003` entries using `generate_diag_blocks_plus_noise`.
- New ncu metrics: `smsp__warp_issue_stall_mio_throttle_per_warp_active.pct` and
  `smsp__warp_issue_stall_long_scoreboard_per_warp_active.pct` (atomicAdd throttle and
  memory-latency stalls).
- Job: `jobs/profile_diag/profile_diag_20260624.sh`, output in `ncu_out/profile_diag/`
  (remote only). Profiled `diag_dd0.001` at 262144, block=8, min_nnz=2, f64/f256/f512,
  both kernels.

### Structural finding

```
num_blocks        = 59-70          # out of 32,768 possible diagonal 8×8 blocks
blocks_per_stripe = min=1 max=1    # every row-strip has exactly ONE PBR block
remainder_nnz     ≈ 70,700         # out of total_nnz ≈ 70,800
compressed_fraction ≈ 0.17%        # PBR captures < 0.2% of the work
```

**Root cause**: with `diag_density=0.001` the 8×8 diagonal blocks average ~0.064 nnz each —
nearly all have <2 nnz and fall to CSR remainder. The off-diagonal noise
(`off_diag_density=1e-6 × 262144²`) contributes ~68,700 nnz, all to CSR. The matrix
labelled "diag" is 99.8% off-diagonal uniform noise at this density / size combination.

### Kernel timings

| kernel | PBR portion | CSR remainder | zeros init |
|---|---|---|---|
| block f512 | 5.6 µs | 548 µs | 604 µs |
| rowgroup f512 | 11.9 µs | 547 µs | 606 µs |
| block f64 | 4.5 µs | 86 µs | 77 µs |
| rowgroup f64 | 10.5 µs | 86 µs | 77 µs |

PBR is 1-5% of total. The 6 µs block-vs-rowgroup delta is invisible on a 600 µs wall → 1.0× in benchmark. ✓

Rowgroup is 2× slower than block for the PBR portion because with ~1 PBR block per group the rowgroup
overhead (binary search, smem_Y init, `__syncthreads`) is not amortized.

### Implication

`diag_dd0.001` at size=262144 with `min_nnz_per_block=2` does not exercise PBR at all.
To study PBR on diagonal sparsity, use higher `diag_density`, smaller `system_size`,
or lower `min_nnz_per_block` (to 1).

### Note on stall metrics

`smsp__warp_issue_stall_mio_throttle` and `smsp__warp_issue_stall_long_scoreboard` returned
`n/a` for all kernels on the RTX 3090 with `--profile-from-start off`. These counters require
a dedicated ncu replay pass (e.g., `ncu --set full`).

---

## 3. Tile ("block of blocks") 2D kernel (2026-07-08, branch `feat/pbr-tileblock`)

### Motivation

`rowgroup` (formerly `blockgroup`) reuses only `smem_Y` across a row-strip; `colgroup` reuses only
`smem_X` across a column band. Head-to-head ncu profiling (Experiment note in
`ncu_out/blockgroup_colgroup/`, June 2026) showed `rowgroup` winning on all 3 tested cases — the
hypothesized colgroup win (X-load amortization at low features) never materialized. Neither is a clear
general win because each reuses only *one* smem axis. The `tile` kernel generalizes both: one CUDA block
owns a `tile_rows × tile_cols` group of PBR blocks, holds **all** its row-slots and col-slots resident
in smem simultaneously, and flushes `smem_Y` via atomicAdd **once per tile** (2 `__syncthreads` total)
rather than once per strip (rowgroup) or once per block (colgroup).

### Design

- **Rename** `blockgroup` → `rowgroup` throughout (commit `e6310a5`, mechanical, verified
  behavior-preserving: 624/624 tests including 96 `TestPBRRowgroupKernel` + 96 `TestPBRColgroupKernel`).
  Frees "block" for the new 2D kernel; `rowgroup`/`colgroup` now a symmetric pair, both kept for
  head-to-head benchmarking.
- **Pluggable grouping, fixed kernel contract.** How PBR blocks map to tiles is decided host-side and
  decoupled from the kernel via a metadata contract (`tile_offsets` CSR + per-slot `tile_row/col_origins`
  with `-1` sentinels for unused slots + per-block `tile_row/col_local` slot lookups). The kernel never
  searches and a tile's rows/cols can be **any** set of bands, contiguous or not. Strategies live in
  `_TILE_STRATEGIES` (`__init__.py`); adding one is a single function + registry entry, no kernel change.
  - `'grid'` — fixed `(row_band // tile_rows, col_band // tile_cols)` window partition. Cheap (one
    `np.lexsort`, fully vectorized). Only catches reuse already aligned to fixed band windows.
  - `'greedy'` — seed-and-grow density-maximizing clustering. Catches non-grid-aligned reuse `'grid'`
    structurally misses: e.g. populated row-bands `{0, 2}` land in one 2×2 tile (the motivating toy
    example `[[1,0,1],[0,0,0],[1,0,1]]`), where a fixed window puts each corner in its own tile.
    `O(num_blocks × capacity)`, plain-Python sets; port to C++ later only if it becomes a bottleneck.
- **Grid size is data-determined** — one CUDA block per non-empty tile, no `group_size`/occupancy tuning
  at call time (unlike rowgroup/colgroup). `get_pbr_tile_blocks_per_sm` (with the required
  `dynamicSMemSize=0`, per Experiment 1) is still exposed for diagnostics and a future count-balanced
  group-of-tiles v2.
- **Smem:** `p = SMEM_SIZE / (tile_rows·BLOCK_ROWS + tile_cols·BLOCK_COLS)`; `smem_X[tile_cols·BLOCK_COLS·p]`
  loaded once, `smem_Y[tile_rows·BLOCK_ROWS·p]` zeroed once and flushed once (per-row-slot guarded by the
  `-1` origin sentinel to skip atomicAdd-of-0 on unused slots). v1: 8×8 PBR blocks × 9 tile shapes
  (`{1,2,4}²` plus `(1,1)`) × 3 smem tiers = 27 template instantiations × 4 dtype combos.

### Correctness

`grid`/`greedy` builders validated locally (500+ randomized contract checks: every block placed exactly
once, tile capacity respected, origins match local-slot lookups; `'greedy'` collapses the toy corner
example to a single 2×2 tile, `'grid'` to four singletons). GPU correctness: `TestPBRTileKernel` sweeps
all 9 shapes × {grid, greedy} against the `block` kernel as ground truth, plus sparse-fill,
non-block-aligned-size, and scattered-corners edge cases.

**GPU build + test results:** all 2360 tests pass on RTX 3090 (1728 `TestPBRTileKernel` cases across all
9 shapes × {grid, greedy} × dtypes/sizes/min_nnz, plus sparse-fill / non-aligned-size / scattered-corners
edge cases). Correctness is not in question — the kernel is right, just slow (see below).

### Benchmark results (commit `69041a9`, 5 sizes × 10 matrices × 6 features, RTX 3090)

**`tile` loses to `block` everywhere, and gets *worse* with size.** Fastest-kernel win counts across
~300 (size, matrix, feature) cells: `block` wins 46-60/60 at every size; `tile` wins **1** cell total
(at 65536). Geomean `tile_vs_block` ratio (>1 = tile faster) degrades monotonically: 0.90 (1024) → 0.86
(4096) → 0.80 (16384) → 0.79 (65536) → **0.77 (262144)**. `tile` is also the worst of the three fusion
kernels at 262144 (rowgroup 0.87, colgroup 0.87, tile 0.77 vs block).

**Worst case is the one that should be best.** `diag_b1.0` (fully dense 64-wide diagonal band — every
2×2/4×4 tile the `'grid'` strategy forms is 100% populated, no scattering to fight) shows `tile` at
**2.4× slower** than `block` at 262144 (block 4.0ms vs tile-best 9.5ms; rowgroup/colgroup similarly bad,
9.7/9.3ms). The best `tile` *shape* on this matrix is almost always `1×1` — bigger tiles (more nominal
reuse) make it *worse*. One bright spot: PBR+`block` beats cuSPARSE at scale (1.03× at 16384 → 1.27× at
262144), so the format itself remains a real win; it's specifically the fusion kernels that underperform.

`grid` vs `greedy` grouping is indistinguishable here (0.99-1.00× everywhere) — expected, since this
matrix set (FEM dropped per experiment scope, see commit `69041a9`) has no permutation-hidden structure
for `greedy` to recover that `grid` can't already see.

Full data/config: `plots/tileblock_partial/` (1024/4096 only — see truncation note below), raw CSV at
commit time in W&B project `private-dayan/torch-fast-spdmm`, runs `69041a9_<size>_<matrix>`.

Three infrastructure bugs surfaced and fixed along the way (all in git history on `feat/pbr-tileblock`):
scattered-matrix "density" was a fraction of the whole block grid → O(N²) nnz, OOM at 262144 (rescaled
to scale as `1024/N`, kept O(N)); one job per size timed out at 24h with the full config matrix (split to
one job per (size, matrix)); `wandb.Table` silently truncates at 10k rows, which had dropped FEM and the
densest scattered matrices from the very runs reported as "completed" (fixed by the same per-(size,matrix)
split — each run's table is now ~1.4k rows).

### Root-cause profiling: why does a *perfectly-tiled* dense matrix still lose? (2026-07-13, ncu)

Profiled `block` vs `tile` (1×1/2×2/4×4, `'grid'`) on `diag_b1.0` @ 262144, f64 and f512
(`jobs/profile_tileblock_diag/profile_20260712.sh`, output in `ncu_out/tileblock_diag/`). Two compounding
causes, neither of which is about tile *selection* — `diag_b1.0`'s tiles are already 100% dense:

**1. `tile` has no vectorized (`float4`) inner loop — this alone dominates even when grid geometry
matches.** At f64, `block` and `tile_1×1`/`tile_2×2` all launch the identical grid (`grid.z=1`, same
block count — `p` covers all 64 features in one pass for every variant tested). Yet `tile` is still
**2.8-3.3× slower** at *identical* grid geometry: `block` 1.84ms vs `tile_1×1` 6.03ms, `tile_2×2` 5.14ms.
`block`'s `pbr_spmm_vec_kernel` dispatches a `float4`/`double2`-vectorized path whenever
`features % L == 0` (`pbr_kernel.cu:162-315`); the `tile` kernel added in this branch is plain scalar.
That gap alone accounts for most of the loss, independent of tile shape.

**2. Bigger tiles compound it via more redundant passes.** At f512, `p = SMEM_SIZE/((tile_rows+tile_cols)·
BLOCK_ROWS)` shrinks as tiles grow (128→64→32 for 1×1→2×2→4×4 @ SMEM=2048), forcing more `feat_tile`
passes: `grid.z` = 1 (block) vs 4/8/16 (tile 1×1/2×2/4×4). Each extra pass re-reads the same small
per-tile metadata (`tile_offsets`, `tile_row/col_origins`, `tile_row/col_local`) from global memory, so
`dram__bytes_read.sum` climbs *monotonically with tile size* — 1.15GB (block) → 1.37 → 1.65 → **2.22GB**
(tile 1×1→2×2→4×4) — the opposite of the X-reuse the design intended. This is exactly why bigger tiles
measured worse in the benchmark, and why `1×1` usually won among tile shapes there.

**Implication:** the negative benchmark result is not a tile-*selection* problem (the user's proposed
biclique-cover / co-clustering refinement to `'greedy'`, see conversation 2026-07-12, would not move
this number — `'grid'` already produces ideal tiles on this matrix). It's a kernel-*implementation* gap:
no vectorized load/store path, plus a smem-split formula that trades away occupancy for reuse in a way
that isn't paying for itself yet. Fixing this (vectorize the inner loop; reconsider whether metadata
should be loaded once per tile-invocation-across-feat-tiles rather than once per (tile, feat_tile) pair)
is the prerequisite before any further investment in tile-selection algorithms.

---

## 4. nnz-balanced grouping (2026-07-13, branch `feat/pbr-nnzgroup`)

### Motivation

`WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md` (H3/H4) found `block` launches ~5× more CUDA blocks than
cuSPARSE's own nnz-balanced partition on `diag_b0.3`, but a diagnostic count-balanced `chunk` kernel
(isolating CTA count as the only variable) showed reducing CTA count alone doesn't recover cuSPARSE's
speed — it gets worse past a small chunk size, with DRAM traffic exploding 7.4× for identical total
work. The natural next hypothesis: cuSPARSE's real advantage is that its partition is nnz-*balanced*,
not just count-balanced — does switching the grouping *criterion* (not just reducing count) close the
gap?

### Implementation

Unified `rowgroup_kernel`/`colgroup_kernel`/`chunk_kernel` (`pbr_kernel.cu`) behind a shared
`group_offsets` CSR-style boundary array, replacing the scalar `group_size`/`chunk_size` argument.
`chunk_kernel` additionally dropped its compile-time `#pragma unroll`'d `CHUNK` template param for a
plain runtime loop (nnz-balanced groups have variable block count per CTA). New host-side
`_build_group_offsets(offsets_np, target_groups, grouping)`: `'count'` reproduces the original
`group_idx*group_size` arithmetic bit-for-bit; `'nnz'` builds boundaries via `np.searchsorted` over
the cumulative-nnz prefix sum that `block_offsets`/`col_offsets` already are — no new C++ needed.
Caught one real bug along the way: `_reorder_pbr_arrays`' offsets output was missing the trailing
total-nnz sentinel (`num_blocks` entries instead of the documented `num_blocks+1`), silently breaking
`col_offsets_np`-based nnz math specifically (rowgroup/chunk, which use the untouched row-major
`block_offsets_np`, were unaffected) — fixed and reverified with 200 randomized cases plus a full
remote regression run (3704 tests passing, including every `grouping='count'` case as a
behavior-preservation guard).

### Results: falsified twice, then a real mechanism found

**Ablation 1 (`diag_b0.3`, low per-block nnz variance, CV=0.19):** `grouping='nnz'` produced nearly
identical grid sizes/timings to `'count'` — expected, since there's little imbalance to correct.

**Ablation 2 (`diag_skewed`, new profile — dense diagonal band + off-diagonal noise, genuine bimodal
per-block nnz, CV=2.08):** same null result. `chunk64`: grid=4110 for both `count` and `nnz`, times
22.4773ms vs 22.4812ms. Directly measured the partition quality to rule out "the algorithm isn't
achieving balance": `'nnz'` grouping cut per-group nnz CV from 0.279 (count) to **0.043** (nnz) — a
6.5× tighter balance — with **zero runtime effect**. This proves nnz-imbalance was never the
bottleneck; a smarter block-reordering scheme (clustering nnz-similar blocks before partitioning)
would target the same already-solved metric, so that direction was abandoned before being built.

**Follow-up profiling (the actual mechanism):** added L1TEX-level global-load-request counters
(isolates *requests issued* from *requests reaching DRAM*) and local-memory counters, `block` vs
`chunk` at `chunk_size∈{2,4,8,16,32,64}`, `diag_b0.3`@262144/f512:

| variant | time (ms) | DRAM read (MB) | L1TEX global-load requests (MB) | L2 hit rate | waves/SM |
|---|---|---|---|---|---|
| `block` | 13.01 | 1099.7 | 4826.3 | 94.59% | 290.6 |
| `chunk64` | 16.74 | 8142.2 | 4827.1 | 59.98% | 4.5 |

Local-memory (register spill) traffic = 0 throughout, ruling that out. L1TEX request volume is
completely flat (~4.83 GB) across every variant — the actual logical work never changes. **L2 hit
rate collapses monotonically, 94.6% → 60.0%**, and DRAM traffic tracks it almost exactly. Mechanism:
`block`'s 262,144 tiny CTAs run with massive concurrency (290 waves), so many spatially-nearby
diagonal-band blocks are in flight at once and L2 serves most of that traffic. Grouping blocks into
fewer, fatter CTAs — by count or by nnz, doesn't matter — means each surviving CTA marches
sequentially through a wider, spatially-dispersed swath of the matrix, with far fewer CTAs in flight
to give L2 concurrent spatial overlap to exploit. Neither CTA-count nor nnz-balance targets spatial
locality, so neither could have fixed this.

### Implication

`feat/pbr-nnzgroup` is shelved — no production kernel change recommended. Any future PBR-block
grouping scheme needs to preserve *spatial* locality among the blocks one CTA processes sequentially
(not just balance work count or nnz), or it pays this same L2-locality tax. `rowgroup`/`colgroup`'s
row-strip/col-band locality is already partially protective (unlike `chunk`'s pure array-order
grouping); this may partly explain why they run closer to `block` than `chunk` does at large group
sizes. Full writeup: `WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md` H5.

---

## 5. Instruction-mix profiling: the per-block decode overhead (2026-07-13)

### Motivation

Experiment 4 (H5) explained why *grouping-based* fixes can't close the gap to cuSPARSE (L2-locality
loss), but not why `block` itself — no grouping at all — trails cuSPARSE in the first place. The
open clue: `sm__throughput` at N=262144 was 67.3%/30.3% (f64/f512) for `block` vs 95.4%/58.4% for
cuSPARSE's `csrmm_alg2` — cuSPARSE extracts far more useful work per occupied cycle.

### Hypothesis

`block`'s bitmask-decode logic (`pbr_spmm_zero_idle_kernel`/`pbr_spmm_vec_kernel`, `pbr_kernel.cu`) —
unpacking the 64-bit occupancy code into `packed_active_rows`/`packed_row_offsets`/
`packed_cols_physical`/`packed_cols_logical` via two 8-iteration unrolled loops plus `__popcll`/
`__clz` — is not gated by thread index. Every one of the 128 threads in a CTA redundantly repeats the
same ~16-24 integer/bit-scan ops to decode the *same* code, instead of one thread decoding once and
broadcasting.

### Experiment

Instruction-mix ncu profiling (`sm__inst_executed_pipe_{fma,alu}.sum`, thread-level
`sm__sass_thread_inst_executed_op_{integer,fp32}_pred_on.sum`) on `block` vs cuSPARSE, identical
`diag_b0.3` matrix, N=262144, f64 and f512.

### Results

| | features | pipe ALU:FMA | thread int:fp32 | SM throughput |
|---|---|---|---|---|
| `block` | 64 | 0.99 | **12.80** | 67.36% |
| cuSPARSE | 64 | 1.30 | **2.78** | 95.31% |
| `block` | 512 | 0.74 | **8.19** | 30.25% |
| cuSPARSE | 512 | 1.30 | **2.78** | 58.31% |

`block`'s ALU pipe executes almost as many instructions as its FMA pipe (should be far below 1 for a
matmul kernel). At the thread level, `block` executes **8–13× more integer instructions per
floating-point instruction** than cuSPARSE, whose ratio (2.78) doesn't move with feature count at
all — exactly what a well-amortized design should look like. `block`'s ratio improves somewhat at
f512 (more FMA work amortized against the same fixed per-launch decode cost), but is still ~2.9×
worse than cuSPARSE even there. This is real, wasted instruction volume — but see the follow-up
below: it does not, by itself, explain the `sm__throughput` gap.

### Follow-up: stall-reason verification (2026-07-14) — corrects the original "Implication"

The write-up above originally claimed the instruction-mix ratio "directly explains" the
`sm__throughput` gap. That claim was never checked against actual warp stall reasons — it inferred
causation from instruction *counts* alone. Prompted to verify it, ran `ncu` with
`smsp__warp_issue_stalled_*.pct` metrics (note the `-ed` suffix — `smsp__warp_issue_stall_*` without
it doesn't exist and is likely why earlier attempts in this project logged these as "n/a") plus
`sm__pipe_{alu,fma}_cycles_active.avg.pct_of_peak_sustained_elapsed`, `block` vs cuSPARSE,
`diag_b0.3`@262144/f512:

| stall reason | `block` | cuSPARSE |
|---|---|---|
| `mio_throttle` (shared-mem/atomic pipe) | **21.29%** | 0.21% |
| `barrier` (`__syncthreads()`) | **20.10%** | 0% |
| `long_scoreboard` (DRAM latency) | 17.55% | **78.39%** |
| `short_scoreboard` (shared-mem/fast dependency) | 12.82% | 2.83% |
| `wait` (fixed-latency execution dependency) | 4.52% | 8.33% |
| `math_pipe_throttle` (ALU/FMA pipe) | **0.83%** | 0.32% |
| ALU pipe utilization | 17.23% | 20.79% |
| warps active (occupancy) | 91.00% | 76.04% |

`math_pipe_throttle` — the metric that directly measures "stalled waiting on the ALU pipe" — is
0.83% for `block`, with the ALU pipe itself only 17.23% utilized. **The redundant decode is real
wasted work, but it is not what warps are stalling on**; it executes in otherwise-idle cycles. The
stall-verified dominant causes are instead `mio_throttle` (atomicAdd/shared-mem pipe contention from
`block`'s per-(row,feature) Y flush) and `barrier` (`__syncthreads()` waits in the load→compute→flush
structure) — both near-zero for cuSPARSE, which is instead healthily `long_scoreboard`-bound (78.4%).

### Implication (corrected)

The redundant per-thread decode (every thread re-deriving `n_r`, `n_c`, `packed_active_rows`,
`packed_row_offsets`, `packed_cols_physical`, `packed_cols_logical`, `p`/`vp_log2` identically across
a CTA) is still a genuine compute-once/broadcast antipattern worth fixing — it wastes issue slots and
register pressure. But per the stall-reason data above, it is **not** the lever that closes the
`sm__throughput` gap to cuSPARSE. The concrete, stall-verified next lever is reducing `atomicAdd`
contention (`mio_throttle`) and `__syncthreads()` count (`barrier`) in `block`'s Y-flush structure.
Unlike H3–H5 (launch-configuration/grouping questions that turned out not to be the primary lever),
this is still a genuine kernel-body fix — just targeting a different mechanism than originally
claimed. Full writeup: `WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md` H6.

**Further follow-up (`STALL_REASONS.md`, 2026-07-14):** reran the same stall-reason profiling across
`diag_b0.1`, `scattered_1x8`, `scattered_8x1`, `scattered_2x2` (plus the original `diag_b0.3`) at
N∈{1024, 262144}, f∈{128, 512}. The `mio_throttle`+`barrier` mechanism reproduces on `diag_b0.1` but
is matrix-shape-dependent, not universal: it collapses to ~0% on `scattered_1x8` (single populated
row per PBR block — `block` is `long_scoreboard`+`barrier`-dominated there instead), and on
`scattered_8x1` cuSPARSE itself shows the same `mio_throttle` spike (~21.5%) that `block` does — so
for that matrix shape the effect isn't even specific to `block`'s kernel design. Full data and tables:
`STALL_REASONS.md`.

---

## Open questions / next steps

1. **Auto kernel selection** — pick block vs rowgroup based on expected compute per group
   (`num_pbr_blocks × features / target_groups > threshold`). Python-only change.
   Would recover the scattered_2x4/f64 regression and remove the manual `kernel=` arg.

2. **Meaningful diagonal profiling** — re-profile with `diag_density ≥ 0.1` or
   `min_nnz_per_block=1` to get ≥50% compressed_fraction and actually exercise PBR on
   banded sparsity.

3. **AtomicAdd stall measurement** — use `ncu --set full` (or `--replay-mode kernel`) to
   collect warp stall reason counters. The `mio_throttle` metric is the direct signal for
   atomicAdd backpressure in the block kernel.
