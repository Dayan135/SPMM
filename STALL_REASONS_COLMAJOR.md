# Col-major PBR block ordering: L2-locality mechanism confirmed, sign flips with matrix shape

Branch: `feat/pbr-colmajor-locality-profile` (based on `feat/pbr-zeroidle-microbench`, which added
`col_major` to `pbr_batched_matmul_cuda`/`pbr_matmul` — see commits `ca59466`/`8e63dd9`).

## Background

`col_major=True` makes `block`/`vec` read the PBR block arrays already pre-sorted by `(col_origin,
row_origin)` (built for the `colgroup` kernel) instead of the natural `(row_origin, col_origin)` order.
**No kernel code changes either way** — only which CTA (one per PBR block) gets which `blockIdx`, i.e.
only the *launch order* changes.

The wall-clock sweep (`ca59466`/`8e63dd9`, 720 points, `block=2/4/8`, `smem=512/1024/2048`, 5 matrix
profiles × 2 densities × 6 feature counts, N=262144) was run but never analyzed at the mechanism level
— only raw W&B tables existed. Pulling and aggregating that data (median Δtime per matrix profile):

| matrix profile | median Δtime | best (min) | worst (max) |
|---|---|---|---|
| `diag_b1.0` | **−4.7%** | −28.4% | +1.9% |
| `diag_b0.3` | −1.2% | −7.1% | +2.8% |
| `scattered_1x8_d2pct` | −3.6% | −36.6% | +1.1% |
| `scattered_1x8_d0.5pct` | −2.8% | −8.7% | +0.1% |
| `scattered_2x4_d0.5pct` | −4.2% | −12.7% | −0.6% |
| `scattered_2x4_d2pct` | −1.1% | −5.9% | +2.3% |
| `scattered_2x2_d0.5pct` | +1.7% | −5.9% | +22.2% |
| `scattered_2x2_d2pct` | −0.5% | −6.2% | +21.8% |
| `scattered_8x1_d0.5pct` | −0.8% | −9.9% | +12.1% |
| `scattered_8x1_d2pct` | **+0.1%** | −5.2% | **+54.5%** |

Mostly a wash-to-small-win, **except `scattered_8x1_d2pct` regresses up to +54.5%** in specific
configs, all `block=8x8` — the opposite shape from `scattered_1x8` (its biggest winner, −36.6%). This
experiment explains that split mechanistically.

## Method

`profile_pbr.py` didn't expose `col_major` or let a specific `_dXpct` sweep density be reproduced —
added `--col-major` and `--density-pct` (commit `15de944`). Then ran the standard 15-metric
stall-reason list *and* an H5-style L2/DRAM-locality metric list (`dram__bytes_read/write.sum`,
`l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`, `l1tex__t_bytes_pipe_lsu_mem_global_op_*`,
`launch__waves_per_multiprocessor`) as **separate** `ncu` invocations per point — mixing extra counters
into the stall-reason list previously corrupted results by changing `ncu`'s replay-pass structure (see
`STALL_REASONS.md`), so the two lists are never combined in one run (`jobs/profile_colmajor/
profile_20260720.sh`, commit `2f0c2c9`).

Three representative N=262144 points, row-major vs `--col-major`, picked from the sweep above:

| tag | matrix | features | block | min_nnz | smem | from sweep |
|---|---|---|---|---|---|---|
| `diag_b1.0_f128` | `diag_b1.0` | 128 | 2 | 2 | 2048 | best win, −28.4% |
| `scattered_1x8_d2pct_f512` | `scattered_1x8` (d2pct) | 512 | 8 | 2 | 2048 | best scattered win, −36.6% |
| `scattered_8x1_d2pct_f256` | `scattered_8x1` (d2pct) | 256 | 8 | 2 | 512 | worst regression, +54.5% |

Single-shot `ncu`-profiled runs (not repeated trials, unlike the wall-clock sweep) — cross-checked
against the sweep's own numbers below to confirm they reproduce the same phenomenon, not profiling
noise.

## Results

| point | time (row→col) | Δtime | L2 hit rate (row→col) | DRAM read (row→col) | DRAM write (row→col) |
|---|---|---|---|---|---|
| `diag_b1.0_f128` | 22.99→16.41 ms | **−28.6%** | 98.04%→98.06% (flat) | 420.0→419.7 MB (flat) | 134.8→134.5 MB (flat) |
| `scattered_1x8_d2pct_f512` | 1.923→1.002 ms | **−47.9%** | 27.4%→**67.4%** (+146%) | 1512→**670** MB (**−56%**) | 148.7→171.5 MB (+15%) |
| `scattered_8x1_d2pct_f256` | 1.989→2.803 ms | **+40.9%** | 88.2%→**73.6%** (−16.5pp) | 337→**751** MB (**+123%**) | 246.7→679.5 MB (**+176%**) |

(`lts__t_sector_hit_rate.pct` = L2; `grid_size`/`block_size`/`waves_per_multiprocessor` were
byte-for-byte identical row-major vs col-major at every point, confirming this is purely a reordering,
not a work-partitioning change.)

## Mechanism: confirms H5's L2-locality finding, sign depends on which axis is "spread"

**`scattered_1x8` (win) and `scattered_8x1` (regression) are mirror images, and the L2 hit-rate/DRAM
numbers explain both, directly extending H5** (`block`'s advantage over grouped kernels comes from L2
serving spatially-overlapping *concurrent* CTAs, `WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md`):

- `scattered_1x8` = 1 row-band spanning the full N rows × 8 column bands. In natural row-major order
  (sorted primarily by `row_origin`), consecutively-launched CTAs can land in *any* of the 8 column
  bands — poor X-read locality for whichever CTAs the GPU happens to run concurrently. Sorting by
  `(col_origin, row_origin)` instead groups all blocks touching the same narrow column slice together
  in launch order, so concurrently-scheduled CTAs now share almost the same `X` columns — L2 hit rate
  more than doubles (27%→67%) and DRAM read traffic is cut in half.
- `scattered_8x1` = 8 row bands × 1 column band spanning the full N columns — the transposed shape.
  Row-major order *already* clusters concurrent CTAs by row (good `Y`/`atomicAdd`-destination locality,
  since L2 also serves the atomic read-modify-write traffic to `Y`, not just `X` reads). Sorting by
  `(col_origin, row_origin)` instead interleaves all 8 distant row bands for CTAs that now share only a
  column position — concurrently-active CTAs scatter their `Y` writes across 8 far-apart regions of a
  262144-row output. Aggregate L2 hit rate drops 88%→74%, and **both** DRAM read and write traffic
  roughly double — consistent with `lts__t_sector_hit_rate.pct` being a single aggregate counter over
  all L2 traffic (X reads and Y atomic reads+writes share the same cache), so degrading one traffic
  class's locality evicts lines the other class needed too.

This is a clean, falsifiable, and now-confirmed mechanism: **col-major sort trades X-read locality for
Y-write(atomic) locality between concurrently-scheduled CTAs; whichever axis (rows or columns) is the
"spread out" one in a given matrix's block distribution determines which locality class col-major
improves, and whichever axis was already narrow (well-clustered in natural row-major order) is the one
it degrades.** `scattered_2x2`/`scattered_2x4` (moderate spread on both axes) sit in between and show
the sweep's noisiest, most mixed results — consistent with neither axis dominating.

**`diag_b1.0`'s −28.6% win does NOT fit this mechanism** — L2 hit rate (98.04%→98.06%) and DRAM
read/write are both flat within noise. The diagonal band is dense and narrow on both axes at once
(`row_origin ≈ col_origin` for nearly every block), so there's little spatial-locality headroom for
either ordering to win or lose. The real driver there is a large `mio_throttle_pct` drop (3.45%→1.56%,
−54.8%) with roughly flat memory traffic — a shared-memory-pipe-pressure or launch-order/scheduling
effect, not a cache-locality one. **This part of the result is real (cross-validated against the sweep
median, −4.7%, and the sweep's own best single point, −28.4%, essentially exact) but its mechanism is
unresolved** — flagged as an open question rather than force-fit into the L2 story.

## Verdict

**Not a uniform win — matrix-shape-dependent, and now for a well-understood reason (for the scattered
cases).** Applying `col_major=True` unconditionally would help matrices with column-spread block
distributions and actively regress matrices with row-spread ones by as much as +54%. A shape-adaptive
choice (e.g. pick row-major vs col-major per matrix based on comparing row-origin vs col-origin spread
of the PBR block set — cheap to compute at `csr_to_pbr` time) is a plausible follow-up, but is a
decision-policy change, not a kernel edit, and is a distinct next hypothesis from this one. The
`diag_b1.0` mio_throttle-driven win remains unexplained and would need a targeted follow-up (e.g.
per-SM CTA assignment / launch-order instrumentation) if pursued further.
