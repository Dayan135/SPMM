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

**GPU build + test results: _pending remote run_** (RTX 3090, `feat/pbr-tileblock`).

### Benchmark plan

Sweep `tile` (9 shapes × {grid, greedy}) against `block`/`rowgroup`/`colgroup` on the existing
scattered/diag families plus two new FEM families: `fem2d` (grid-aligned reuse — `'grid'` should already
do well) and `fem2d_permuted` (mesh nodes relabeled to break index-space locality while preserving reuse
structure — the case that isolates what `'greedy'` recovers that `'grid'` cannot). Results: _pending._

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
