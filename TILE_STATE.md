# Tile ("block of blocks") kernel — state summary

**Status: shelved (2026-07-13), branch `feat/pbr-tileblock`. Negative result — do not resume
without first addressing the root cause below.**

## The idea

`rowgroup` reuses `smem_Y` across a row-strip (1×N); `colgroup` reuses `smem_X` across a column
band (N×1). Neither reuses both axes at once, and profiling had shown no consistent winner between
them. The generalization: a genuine 2D tile — one CUDA block owns a `tile_rows × tile_cols` group
of PBR blocks, caches **both** `smem_X` and `smem_Y` for the whole tile, and flushes once per tile
instead of once per strip/block.

A real design question came up along the way: how do you decide *which* PBR blocks share a tile?
A naive fixed grid partition misses reuse that isn't grid-aligned (e.g. populated blocks at
row-bands `{0,2}` × col-bands `{0,2}` in an otherwise-empty grid — a fixed window never puts row 0
and row 2 in the same tile). This was solved with a pluggable-strategy architecture: the kernel
consumes a fixed metadata contract (`tile_offsets` + per-slot origins + per-block local-slot
lookups), and how tiles are formed is a swappable host-side strategy — `'grid'` (cheap fixed
window) or `'greedy'` (density-maximizing seed-and-grow clustering, catches the non-aligned case).
Both are implemented and correct.

## What's built (all on `feat/pbr-tileblock`, correct, none of it reverted)

- `pbr_spmm_tile_kernel` (`torch_fast_spdmm/kernels/pbr_kernel.cu`) — 9 shapes (`{1,2,4}²` +
  `(1,1)`) × 3 smem tiers, 8×8 PBR blocks only.
- Pluggable grouping: `_TILE_STRATEGIES = {'grid': ..., 'greedy': ...}` in
  `torch_fast_spdmm/torch_fast_spdmm/__init__.py`, cached per `(tile_rows, tile_cols, strategy)`.
- Full bindings, occupancy query (`get_pbr_tile_blocks_per_sm`, correctly using
  `dynamicSMemSize=0`).
- `blockgroup` renamed to `rowgroup` (kept, alongside `colgroup`, as the two 1D-reuse baselines
  tile was meant to generalize).
- Correctness: **all 2360 tests pass** on RTX 3090 — 1728 `TestPBRTileKernel` cases (9 shapes ×
  {grid, greedy} × dtypes/sizes/min_nnz) plus sparse-fill / non-aligned-size / scattered-corners
  edge cases. The kernel is correct; it's just slow.

## Benchmark result: negative

Full sweep, 5 sizes × 10 matrices × 6 features (commit `69041a9`, see `LAST_EXPERIMENTS.md`
Experiment 3 for the complete numbers):

- `tile` wins **1 cell out of ~300**. `block` wins nearly everywhere.
- geomean `tile_vs_block` (>1 = tile faster) **degrades with size**: 0.90 (1024) → 0.77 (262144).
- Worst on `diag_b1.0` — a fully dense diagonal band where every tile the `'grid'` strategy forms
  is 100% populated, no scattering to fight. `tile` is **2.4× slower** than `block` there at
  262144. Best tile *shape* is almost always `1×1` — bigger tiles (more nominal reuse) measure
  *worse*.
- `grid` vs `greedy` grouping is indistinguishable in this matrix set (FEM matrices, the one case
  that would separate them, were dropped from the final sweep per experiment scope) — moot anyway
  since tile loses to block regardless of grouping quality.

## Root cause (ncu profiling, `diag_b1.0`, `jobs/profile_tileblock_diag/`)

Not a tile-selection problem — `diag_b1.0`'s tiles are already ideal. Two compounding
implementation gaps:

1. **No vectorized (`float4`) inner loop.** `block`'s `pbr_spmm_vec_kernel` dispatches a
   vectorized path whenever `features % 4 == 0`; the tile kernel here is plain scalar. At f64,
   `block` and `tile_1×1`/`tile_2×2` launch the *identical grid* (same block count, one pass) —
   tile is still 2.8–3.3× slower on identical geometry. This is most of the gap, independent of
   tile shape.
2. **Smem-split formula shrinks `p` as tiles grow**, forcing more `feat_tile` passes (`grid.z`: 1
   for block vs 4/8/16 for tile 1×1/2×2/4×4 at f512). Each extra pass re-reads the same small
   per-tile metadata from global memory, so DRAM reads climb *monotonically with tile size*
   (1.15→2.22 GB) — the opposite of the reuse the design intended. Explains both why bigger tiles
   measured worse and why `1×1` usually won among tile shapes.

## If this is revisited

Fix the implementation gap before touching tile-selection algorithms again:
1. Vectorize the tile kernel's load/accumulate/flush loop (`float4`/`double2`, mirroring
   `pbr_spmm_vec_kernel`) — the single biggest lever, per the f64 same-grid comparison above.
2. Reconsider whether per-tile metadata (`tile_offsets`, `tile_row/col_origins`,
   `tile_row/col_local`) needs to be re-read once per `(tile, feat_tile)` pair, or can be loaded
   once and reused across a tile's feat_tile passes.
3. Only after (1)+(2) close most of the gap does it make sense to reopen the tile-*selection*
   question (the biclique-cover / co-clustering discussion from 2026-07-12) — a better selector
   cannot fix a kernel that's already slower on perfectly-formed tiles.

`rowgroup`/`colgroup`/`block` are unaffected and remain the working baselines. Nothing here blocks
other work on the branch or elsewhere in the repo.
