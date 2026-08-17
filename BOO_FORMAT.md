# BOO — Blocked COO

A clustered, variable-size, bitmap-free sparse-matrix format for SpMM, and the CUDA
kernel that consumes it.

**Status:** implemented and measured. First results in §12. Claims that are still
predictions rather than measurements are marked as such — there is a specific reason for
that discipline, recorded in §8.

---

## 1. Why PBR caps out

PBR stores each block's occupancy as a 64-bit bitmap: bit `r * block_cols + c` is set iff
`(r, c)` is a nonzero. That is a `std::bitset<64>` (`pbr_matrix/pbr_matrix.hpp:13`), so
`block_rows * block_cols <= 64` — an 8×8 ceiling.

The ceiling is not one constant. Four independent caps have to be lifted together:

| cap | location | limit |
|---|---|---|
| `BITS_PER_CODE 64` — the bitmap itself | `pbr_matrix/pbr_matrix.hpp:13` | 64 positions per block |
| `packed_active_rows`, 3 bits per row index | `kernels/pbr_kernel.cu:56` | row index ≤ 7 |
| `packed_row_offsets`, 7 bits per row's data offset | `kernels/pbr_kernel.cu:57` | ≤ 127 nnz per block; uses 56 of 64 bits at 8×8, no headroom |
| `packed_cols_physical` / `packed_cols_logical`, 3 bits per column | `kernels/pbr_kernel.cu:75-76` | column index ≤ 7 |

On top of that, `nnzflat` rebuilds a per-block COO table into **thread-private arrays**
(`nz_row[64]`, `nz_ptr[64]`, `nz_pc[64]`, `kernels/pbr_kernel.cu:721-743`) — ~288 bytes per
thread at 8×8, recomputed identically in every thread of every CTA. That already spilled to
local memory once, measured at up to 1.15B spill instructions per launch. At 16×16 it would
be ~1 KB per thread and is simply not viable.

So a bigger block is not a parameter change. It needs a different encoding.

## 2. The change, in one line

PBR blocks already are `rows(stripe_i) × cols(stripe_j)` intersections of a fixed, aligned
8×8 grid.

**BOO keeps exactly that structure, and replaces the fixed grid with a clustered,
variable-size one — and the bitmap with 1-D `uint16` COO indexing.**

Same shape. Better grid lines. No size cap.

```
PBR                                   BOO
+----+----+----+----+                 +--------+--+------------+
|    |    |    |    |                 |        |  |            |
+----+----+----+----+                 |        +--+------------+
|    |    |    |    |                 +--------+  |            |
+----+----+----+----+                 |        |  |            |
|    |    |    |    |                 |        +--+------------+
+----+----+----+----+                 +--------+--+            |
fixed 8x8, aligned                    clustered, variable
```

## 3. Why block size is the lever

With `n_rg` row groups and `n_cg` column groups:

```
total X loads  = Σ_blocks n_c(b)   ≈  n_rg × (columns touched per row group)
total Y writes = Σ_blocks n_r(b)   ≈  n_cg × rows
```

- **Bigger row groups → fewer row groups → fewer X reads.**
- **Bigger column groups → fewer column groups → fewer Y writes.**

The two dimensions buy different things, which is why they are tuned independently.

The Y-write side is where the measured money is. `STALL_REASONS_NOATOMIC_ABLATION.md`
replaced Step C's global `atomicAdd` with a plain store — numerically wrong, diagnostic
only — and measured **−40% to −66% wall clock**, `mio_throttle` collapsing 17.8 → 0.46, and
SM throughput roughly doubling. BOO cannot delete those atomics (see §6), but larger column
groups cut their *count* proportionally.

The X side compounds. A block's arithmetic intensity is `nnz / n_c` MACs per X scalar
loaded: ~2.5 for an 8×8 PBR block holding 20 nonzeros, ~9.4 for a clustered 32×32 block
holding 300.

## 4. The format

Row groups and column groups are each stored **once**. A block is a pair of references plus
its nonzeros.

```
rows, cols, nb, n_rg, n_cg, total_nnz
max_nr, max_nc, max_nnz_blk            // global maxima; the launcher sizes the arena from these

row_grp_ptr[n_rg+1] -> row_ids[]       int32, GLOBAL row indices, ascending within a group
col_grp_ptr[n_cg+1] -> col_ids[]       int32, GLOBAL col indices, ascending within a group

blk_row_grp[nb]      int32             which row group
blk_col_grp[nb]      int32             which col group
blk_nz_ptr [nb+1]   -> nz_idx[]        uint16
                       nz_val[]        scalar_t
blk_shift  [nb]      uint8             = log2(pad2(|col group of this block|))

rem_indptr[rows+1], rem_col_ind[], rem_vals[]     // may be empty; see §7
```

### 4.1 Worked example

```
        c0  c1  c2  c3  c4  c5          RG0 = {0, 2, 4}    CG0 = {0, 3}
   r0 [  a   .   .   b   .   . ]        RG1 = {1, 3, 5}    CG1 = {1, 2, 4, 5}
   r1 [  .   e   .   .   .   . ]
   r2 [  c   .   .   .   .   . ]                      CG0             CG1
   r3 [  .   .   .   .   f   . ]        RG0    blk0: a, b, c    blk1: d
   r4 [  .   .   .   .   .   d ]        RG1    blk2: g          blk3: e, f
   r5 [  .   .   .   g   .   . ]
```

| field | dtype | meaning | value |
|---|---|---|---|
| `rows`, `cols` | scalar | matrix dimensions | `6`, `6` |
| `total_nnz` | scalar | nonzeros in `A` | `7` |
| `n_rg`, `n_cg` | scalar | row / column groups from clustering | `2`, `2` |
| `nb` | scalar | blocks = non-empty intersections; this is `gridDim.x` | `4` |
| `row_grp_ptr[n_rg+1]` | int32 | offsets into `row_ids`; group *g* owns `row_ids[ptr[g]:ptr[g+1]]` | `[0,3,6]` |
| `row_ids[rows]` | int32 | global row indices, group by group — a permutation of `0..rows-1` | `[0,2,4, 1,3,5]` |
| `col_grp_ptr[n_cg+1]` | int32 | same, columns | `[0,2,6]` |
| `col_ids[cols]` | int32 | global column indices, group by group | `[0,3, 1,2,4,5]` |
| `blk_row_grp[nb]` | int32 | which row group this block sits in | `[0,0,1,1]` |
| `blk_col_grp[nb]` | int32 | which column group | `[0,1,0,1]` |
| `blk_nz_ptr[nb+1]` | int32 | offsets into `nz_idx` / `nz_val` | `[0,3,4,5,7]` |
| `nz_idx[]` | **uint16** | packed local coordinate of each nonzero | `[0,1,2, 11, 5, 0,6]` |
| `nz_val[]` | scalar_t | values, same order | `[a,b,c, d, g, e,f]` |
| `blk_shift[nb]` | uint8 | low bits of `nz_idx` holding the column part | `[1,2,1,2]` |
| `max_nr`, `max_nc`, `max_nnz_blk` | scalar | **largest** `n_r` / `n_c` / `n_nz` over all blocks | `3`, `4`, `3` |
| `rem_*` | int32 / scalar_t | CSR remainder; empty unless the small-block policy fills it | empty |

*(`nz_idx` / `nz_val` shown in the default row-major order — see §4.3.)*

### 4.2 The `nz_idx` encoding

`local_row` and `local_col` are positions **within the block's own row and column group
lists**, not global indices. So `nz_idx` is a coordinate inside the small `n_r × n_c`
rectangle.

Take `d` at global `(row 4, col 5)`, in block 1 = (RG0, CG1):

```
block 1 rows = row_ids[0:3] = [0, 2, 4]      row 4 is at position 2  ->  local_row = 2
block 1 cols = col_ids[2:6] = [1, 2, 4, 5]   col 5 is at position 3  ->  local_col = 3
|CG1| = 4 -> pad2(4) = 4 -> 2 bits           ->  blk_shift[1] = 2

nz_idx = (local_row << blk_shift) | local_col = (2 << 2) | 3 = 11
```

The kernel reverses it in two instructions:

```cuda
const uint32_t r = idx >> shift;
const uint32_t c = idx & ((1u << shift) - 1u);
```

No `__popcll`, no bitmap scan, no select-the-kth-set-bit search. Global indices are needed
only at the boundaries — `sh_row_ids[r]` when storing to `Y`, `sh_col_ids[c]` when loading
`X`. The inner multiply runs entirely in local coordinates against shared memory.

`blk_shift` is **per block** because column groups differ in size. That is what "block size
is flexible, not predefined" means concretely.

**The uint16 constraint never binds.** `max_nr · pad2(max_nc) ≤ 65536` permits 256×256, but
shared memory (§9) caps `n_r + n_c` well before that. The encoding is free; the builder
asserts the bound anyway.

### 4.3 Nonzero ordering — a parameter, not an assumption

`nz_order ∈ {'rowmajor', 'colmajor'}`, **default `rowmajor`**.

Row-major is the order nonzeros already arrive in from CSR, so the builder emits it with no
re-sort, and under row-major `nz_idx` is monotonically increasing within a block — a free
extra invariant to assert.

The existing `nnzflat` kernel uses column-major. Its own header comment on
`feat/pbr-nnzflat-rowmajor-benchmark` records why that is not evidence:

> This gives up column-major's smem_X read-locality property … that property was
> **inherited from `colgroup`, a since-removed kernel variant that never itself beat
> `block`, so it was never independently validated as a win for `nnzflat` specifically.**
> Net effect … **to be measured empirically, not assumed.**

There is local ledger data (`optimizer_ledger_nnzflat_rowmajor_{baseline,candidate}.sqlite3`,
`main` vs `27c8759`, size 262144 / features 512 / block 8 / min_nnz 2):

| matrix | column-major | row-major | |
|---|---|---|---|
| `diag_b1.0` | 57,840,352 | 73,769,088 | col-major 27% faster |
| `diag_b0.3` | 19,003,136 | 25,662,112 | col-major 35% faster |
| `scattered_1x8` | 861,376 | 1,051,456 | col-major 22% faster |
| `pattern_row_skewed` @smem2048 | 1,997,664 | 1,779,936 | row-major 11% faster |
| `pattern_row_skewed` @smem512 | 2,374,144 | 2,014,240 | row-major 15% faster |

But that branch bundles **three** changes — the ordering flip, a cheaper index derivation,
and Step C row-compaction — so it does not isolate ordering.

In BOO, two of those three confounds do not exist: there is no index derivation at all
(§4.2), and row-compaction is free from the format (§4.4). Ordering becomes a pure host-side
sort key, zero kernel cost to flip. So it is a flag, measured directly (§8), not inherited.

### 4.4 Why explicit `row_ids` / `col_ids` instead of block origins

Clustering never has to physically permute `A`, `X`, or `Y`. `Y` stays in natural row order,
so tests compare directly against `torch.sparse.mm` and there is no gather/scatter per
matmul. Index overhead is `4·(rows + cols)` bytes **total**, not per block.

It also delivers `SHOW_TO_ERAN.md` §4's Step-C row compaction for free: a row group only
lists rows the clustering assigned to it, so Step C needs no `if (row_mask == 0) continue`
branch. The 72.3% `branch_resolving` stall measured on `scattered_1x8` is gone by
construction, not by a fix.

### 4.5 Mapping from PBR

| PBR | BOO | why |
|---|---|---|
| `block_coords[2*nb]` = `(row_origin, col_origin)` on a fixed aligned grid | `blk_row_grp` + `blk_col_grp` + the group tables | rows/cols in a group are neither contiguous nor aligned, so a corner coordinate cannot describe them |
| `block_codes[nb]` = uint64 bitmap | `nz_idx[]`, one uint16 per nonzero | a bitmap costs `n_r·n_c` bits and caps at 64; explicit indices cost 16 bits per *nonzero* and cap at 65536 positions |
| `block_data[]`, position implied by bitmap scan order | `nz_val[]`, paired 1:1 with `nz_idx` | position is explicit → no scan, no popcount, no select-kth-set-bit |
| `block_offsets[nb+1]` | `blk_nz_ptr[nb+1]` | same thing, renamed |
| `block_rows` / `block_cols`, two scalars for the whole matrix | `row_grp_ptr` / `col_grp_ptr` differences, per block | the point: block dimensions become per-block data |
| `remainder_indptr/col_ind/vals` | same, kept | the small-block policy is open (§7) |

## 5. Clustering

```python
cluster_axes(A, method, target_nr, target_nc, max_nr, max_nc, max_block_nnz, seed)
    -> (row_labels[rows], col_labels[cols])
```

The clustering target **is** the expected block size. 8 reproduces today's PBR blocking; the
point of the new format is that larger values are now expressible.

Graclus and METIS both output a **hard partition** — one label per vertex — so each global
row belongs to exactly one row group. Blocks are the non-empty `(row group × col group)`
intersections.

- **Symmetric / near-symmetric `A`** (most of the suite: `fem2d`, `laplacian_2d`, `diag_*`):
  one clustering of the vertices used for **both** axes. Diagonal blocks come out dense,
  off-diagonal blocks are the inter-cluster edges. This is the structurally right model when
  `A` is a graph.
- **Rectangular or non-symmetric `A`**: rows from `A·Aᵀ`, columns from `Aᵀ·A`. Same code,
  transposed input.

Detected automatically; overridable.

### 5.1 The similarity graph

```python
A_pat = A.copy(); A_pat.data[:] = 1
G = A_pat @ A_pat.T          # G[i,j] = |col(i) ∩ col(j)|
```

`A·Aᵀ` is symmetric and square for **any** `A`, including rectangular. Do *not* symmetrize
`A` itself — `A + Aᵀ` is the mesh-graph model, a different object, and wrong for rectangular
`A`.

This weight is not a proxy. Merging rows `i` and `j` costs
`deg(i) + deg(j) − |col(i) ∩ col(j)|` columns instead of `deg(i) + deg(j)`, so the saving
*is* the edge weight, and heavy-edge matching on `G` is literally greedy on the true
objective.

Two guards are mandatory, or `diag_b1.0` at N=262144 will OOM (`A·Aᵀ` costs `Σ_j d_j²`):

- **hub-column drop** — exclude columns of degree > `tau` (default 1024) from graph
  construction only. They connect everything and carry no clustering signal.
- **top-k neighbour cap** — build `G` in ~4096-row chunks, keep the `k` (default 32)
  heaviest neighbours per row via `np.argpartition`, deterministic tie-break by smaller
  index. Memory becomes `O(k · rows)`.

### 5.2 Methods

| `method` | implementation | role |
|---|---|---|
| `graclus` | `torch_cluster.graclus_cluster(row, col, weight)`, iterated to the target group size | primary |
| `metis` | `torch.ops.torch_sparse.partition(rowptr, col, weight, nparts)`, `nparts = ceil(n/target)`, vertex weight = row nnz | balance guarantee Graclus lacks |
| `hem` | pure-numpy multilevel heavy-edge matching (§5.3) | dependency-free fallback |
| `rcm` | `scipy.sparse.csgraph.reverse_cuthill_mckee(A + A.T)`, square only, then tile | cheap baseline |
| `degree` | sort by nnz, then tile | controls for balance alone |
| `identity` | `arange(n) // target` | **today's fixed tiling — the honest "no clustering" control** |
| `random` | seeded permutation, then tile | pessimistic control; if the offline metric does not degrade here, the metric is broken |

`torch_cluster` and `torch_sparse` are optional imports. `torch_sparse`'s METIS path also
requires the wheel to be built `WITH_METIS`, which pip wheels often are not. If either
import fails, `graclus` / `metis` degrade to `hem` with a warning.

### 5.3 The `hem` fallback

`L = log2(target)` levels. Fully vectorized, no RNG, deterministic.

1. **Heaviest neighbour** — `np.lexsort((coo.col, -coo.data, coo.row))`, take the first
   entry per row. `lexsort` is stable and the `col`-ascending key breaks weight ties by
   smallest index, so the result is reproducible.
2. **Mutual-max matching** — keep pairs where `best[best[i]] == i` and `i < j`. A maximal
   matching with weight ≥ ½ of optimal. This is the choice that makes the whole thing
   vectorizable in numpy, and it removes all randomness.
3. **Cap enforcement, before contraction** — reject a pair if `size[i]+size[j] > max_nr` or
   `nnz[i]+nnz[j] > max_block_nnz`. Rejected pairs stay unmatched this level.
4. **Residual rounds** — re-run 1–3 twice on the unmatched subgraph. One round matches
   ~50–60%; three reach ~85–90%.
5. **Contract**, then **recompute weights exactly** from the binarized supernode supports:
   ```python
   A_l = binarize(P @ A_prev);  G = A_l @ A_l.T
   ```
   rather than the usual `P·G·Pᵀ` shortcut, which **over-counts** — shared-column counts
   combine by union, not by sum. Preprocessing is not time-bounded and there are only ~5
   levels. `P·G·Pᵀ` stays behind a flag; the exact-vs-approx delta is a cheap ablation.

Complexity: `O(k · rows)` memory, `O(L · k·rows · log(k·rows))` time.

### 5.4 What the graph model does not capture

The exact objective for SpMM blocking is the **hypergraph** connectivity−1 cut — columns as
nets, rows as vertices — i.e. the PaToH / hMETIS model, not METIS's graph model. The graph
model is an approximation of it.

This matters when reading a plateau: a graph-model result flattening out is not evidence
that the *idea* flattened out.

## 6. The kernel

```cuda
template <typename scalar_t, int THREADS, int P_LOG2>
__global__ void __launch_bounds__(THREADS, 1536 / THREADS)
boo_spmm_nnzflat_kernel(...);     // extern __shared__ arena
```

`grid = (nb, batch, ceil(features / FEAT_BLOCK))` — one CUDA block per BOO block.

Arena layout, sized by the launcher from the format's global maxima:

```
[ sh_nz_val : max_nnz_blk scalars ][ sh_nz_idx : max_nnz_blk uint16 ]
[ sh_col_ids : max_nc int32 ][ sh_row_ids : max_nr int32 ]
[ smem_X : max_nc * p ][ smem_Y : max_nr * p ]
```

Load / work / flush:

```
1. LOAD  (hoisted above the feature-tile loop, once per CTA)
        stage nz_val, nz_idx, col_ids, row_ids into shared

2. for base_f in steps of p:
     A. zero smem_Y
        smem_X[c*P + f] = X[sh_col_ids[c]*features + base_f + f]
        __syncthreads()

     B. one thread per (nonzero, feature):
          r = idx >> shift;  c = idx & mask
          atomicAdd(&smem_Y[r*P + f], sh_nz_val[k] * smem_X[c*P + f])
        __syncthreads()

     C. atomicAdd(&Y[sh_row_ids[r]*features + base_f + f], smem_Y[r*P + f])
```

### 6.1 Why the hoist is load-bearing

Today `block_data` is `__ldg`-ed once per (nonzero, feature) **on every feature-tile pass**,
so A-traffic is `4·nnz·(FEAT_BLOCK/p)` — shrinking `p` multiplied it. That is the single
reason a small `p` was previously unaffordable.

Hoisting makes A-traffic `6·nnz`, **independent of `p`**. That is what unlocks §9's choice
of `p = 8`. It also deletes `nnzflat`'s per-thread COO rebuild entirely, so register
pressure *falls* relative to PBR despite fatter blocks.

### 6.2 Why Step C is atomic

A row group appears in one block per column group it intersects, so a row's result is split
across its whole block-row and each block contributes a partial sum. Plain stores would race.

Making Step C a plain store requires a row group to own *all* of its nonzeros in one CUDA
block — a real option, recorded in §11.

### 6.3 The `current_p` bug class, designed out

All four PBR kernels decode with stride `p` (`>> p_log2`, `& p_mask`) but index shared memory
with stride `current_p = min(p, feat_end - base_f)`. Those agree only when `current_p == p`;
otherwise every decode is wrong. Documented-not-fixed at `kernels/pbr_kernel.cu:82-84`
("correct when features is a power-of-2 multiple of p, which holds for all benchmarked
feature counts").

BOO never lets the stride differ from `p`. Shared memory is always indexed with the template
constant `P`; only the *global* accesses are guarded:

```cuda
smem_X[c*P + f] = (base_f + f < features) ? X[...] : scalar_t(0);
if (base_f + f < features) atomicAdd(&Y[...], smem_Y[r*P + f]);
```

There is no `current_p` anywhere. Regression-tested at `features ∈ {100, 200, 520}`.

## 7. The small-block policy — deliberately open

`small_block_policy ∈ {'keep', 'merge', 'remainder'}`.

On a 2-D clustered grid, many non-empty intersections hold only one or two nonzeros. Each
still pays a full `smem_X` load and two barriers for almost no arithmetic. This is exactly
the long tail PBR's `min_nnz` / CSR-remainder rule exists to absorb.

| policy | behaviour | trade |
|---|---|---|
| `keep` | every non-empty intersection becomes a block | `compressed_fraction ≡ 1.0`; makes `SHOW_TO_ERAN.md` §2's CSR-fallback artifact structurally impossible |
| `merge` | a row group's sparse column groups are fused into one wider block | bigger `smem_X`, fewer blocks, fewer Y writes |
| `remainder` | below-threshold blocks spill to CSR, run concurrently on a second stream | exactly PBR's behaviour today |

The format carries the remainder arrays unconditionally (empty under `keep`), and the
bindings keep PBR's two-stream fork/join, so all three are reachable without a schema
change. The choice is made from the measured nnz-per-block histogram, not in advance.

## 8. Bank conflicts — prediction, pending measurement

Shared memory has 32 banks of 4-byte words; `bank = word_address mod 32`.

**Steps A and C are conflict-free at any `p`, unconditionally.** The shared address *is* the
flat work index: with `r = work >> P_LOG2` and `f = work & P_MASK`, `r*P + f == work`. A warp
covers 32 consecutive work indices → 32 consecutive words → 32 distinct banks.

**Step B depends on `nz_order`.** At `p = 8` a warp covers 4 consecutive nonzeros × 8
features (§9.1).

| | `smem_X` read | `smem_Y` atomic |
|---|---|---|
| **column-major** | the 4 nonzeros usually share a column → all four lane-groups read the *same* 8 words → hardware **broadcast**, free | the 4 have distinct, typically consecutive rows `r..r+3` → `r*8+f` collides only if `r ≡ r' (mod 4)`, impossible for 4 consecutive → **conflict-free**, 4 disjoint ranges, atomics in parallel |
| **row-major** | the 4 usually share a row → 4 different columns at `c*8+f` → collide whenever `c ≡ c' (mod 4)` → 2- to 4-way conflict | all 4 share a row → all target the *same* 8 `smem_Y` words → 4-way serialized atomics |

So the model predicts column-major wins at `p = 8`, and wins *more* at `p = 8` than at
`p = 32` (where a warp sits inside a single nonzero and ordering barely matters).

**This is a derivation, not a measurement, and the default is `rowmajor` anyway.** Two
reasons to distrust the prediction until it is measured:

- The row-major `smem_Y` serialization is *within one warp across 8 words*. Shared-atomic
  conflict resolution is cheap; whether it costs measurable time at this scale is unknown.
- Clustered blocks are far denser than PBR's 8×8, so a warp's 4 nonzeros are more likely to
  share a row *or* a column than in the §4.3 data. Both orderings' collision structure
  changes with block density, and the PBR-era numbers may simply not transfer.

Verify with `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` on both settings.
Three outcomes, all actionable: the split reproduces → `nz_order` becomes a per-matrix build
choice; column-major wins everywhere → flip the default (one line); no measurable difference
→ the property was never real and the question closes for good.

**Do not pad the shared-memory pitch.** Padding to `P+1` breaks the `r*P + f == work`
identity that makes Steps A and C conflict-free, and at `p = 8` it would also break the
column-major broadcast. It is a pessimization here, not the free win it looks like.

If column-major does win, it also retires most of `SHOW_TO_ERAN.md` §6's "23 atomics vs 8"
concern — the surplus was never the raw count, it was the collision structure.

## 9. Sizing

### 9.1 Why `p = 8`

`p` is the number of contiguous features held in shared memory per pass, computed at runtime
as the largest power of two that fits after the data regions are reserved, capped at the
requested value (default 8) and at `features`.

The only structural reason to want a large `p` is contiguous global access. 8 fp32 = 32 B =
exactly one sector, which already gives 100% byte efficiency. And a small `p` keeps the
kernel useful when `features` is small.

The cost is real but narrow. In Step A each thread handles one (column, feature) pair —
`c = i >> 3`, `f = i & 7` — so a 32-lane warp covers `32 / 8 = 4` columns, at unrelated
global addresses since `col_ids[c]` are arbitrary:

```
p = 32:  warp reads 1 column × 32 floats = 128 B contiguous
         -> 4 sectors, merged into 1 wavefront
p =  8:  warp reads 4 columns ×  8 floats = 4 × 32 B, non-contiguous
         -> 4 sectors, 4 separate wavefronts
```

Same bytes, **same sector count**, 4× the *wavefronts*. The cost sits in L1/LSU request
throughput, not DRAM traffic. `p` remains a sweep axis (`{8, 16, 32}`).

### 9.2 Occupancy model

Calibrated against a measured number: `SHOW_TO_ERAN.md` §3 reports 11 blocks/SM, limiter =
shared memory, at `smem_size = 2048` (8192 B static).

```
CTAs/SM = min( floor(102400 / (B + 1024)),   # 100 KB smem/SM, 1 KB driver reservation
               floor(1536 / T),              # 48 warps/SM on cc 8.6
               16,                           # hw block limit on cc 8.6
               floor(65536 / (T * R)) )      # registers
```

`floor(102400 / (8192 + 1024)) = 11`. Exact match.

**Corollary: `__launch_bounds__(128, 16)` in the current PBR kernels is arithmetically
impossible on SM86** — 16 × 128 = 2048 threads, but cc 8.6 caps at 1536. All it does is tell
ptxas to chase 65536/(16·128) = 32 registers/thread, a target it misses anyway (measured
36–40). BOO uses `__launch_bounds__(T, 1536/T)`, giving an achievable 42-register target.

### 9.3 Reachable configurations

At `p = 8`, fp32: `B = 36·(n_r + n_c) + 6·nnz_cap`. **The nonzero data, not `smem_X`/
`smem_Y`, dominates** — a direct consequence of the small `p`, and it makes `max_nnz_blk`
the binding cap.

| `n_r × n_c` | nnz cap | arena `B` | best `T` | CTAs/SM | occupancy |
|---|---|---|---|---|---|
| 16×16 | 64 | 1,536 B | 128 | 12 | 100% |
| 32×32 | 256 | 3,840 B | 128 | 12 | 100% |
| 64×64 | 512 | 7,680 B | 128 | 11 | 92% |
| 64×64 | 1024 | 10,752 B | 256 | 6 | 100% |
| 128×128 | 2048 | 21,504 B | 512 | 3 | 100% |
| 128×128 | 4096 | 33,792 B | 512 | 2 | 67% |

fp64 costs `68·(n_r + n_c) + 10·nnz_cap`: 32×32 / 256 nnz = 6,912 B at 100% on `T = 128`;
64×64 / 1024 nnz = 18,944 B, needing `T = 512` for 100%.

`cudaFuncAttributeMaxDynamicSharedMemorySize` is not required below 48 KB/CTA but is set
guarded anyway — without it the failure mode is a swallowed `cudaErrorInvalidValue` at
launch.

### 9.4 Per-CTA dynamic shared memory does not exist

CUDA has no mechanism for a per-CTA shared-memory size. `extern __shared__` takes its size
from the launch config (`kernel<<<grid, block, bytes, stream>>>`), and the hardware gives
every CTA in the grid that same allocation. The only lever is how many launches you issue:

| approach | granularity | cost |
|---|---|---|
| one launch, global max | every CTA pays for the biggest block | simplest; a real tax if max ≫ mean |
| size-class bucketing | blocks bucketed into 2–4 classes, one launch each | k launches on k streams; a block pays only its bucket's ceiling |
| template ladder on size (PBR's `TOTAL_SHARED ∈ {512,1024,2048}`) | same as bucketing, static smem | combinatorial blowup once `THREADS` and `P_LOG2` are also templated; hard 48 KB ceiling |
| build-time splitting | oversized blocks split until they fit a cap | free, and needed anyway for the occupancy caps |

**Chosen:** template only on `scalar_t × THREADS × P_LOG2` — closed sets, 2 × 3 × 3 = 18
instantiations — and pass the arena size as the runtime launch parameter. The arena size
need not be compile-time; all pointer arithmetic into it is runtime regardless, and only
`P_LOG2` benefits from constant-folding (shift/mask, loop unrolling). Caps are enforced by
splitting oversized blocks at build time. Bucketing is added only if the measured histogram
shows `max nnz ≫ mean nnz`.

### 9.5 Sweep axes

`target_nr`, `target_nc` (independently, `{8,16,32,64,128}`), `threads` `{128,256,512}`,
`p` `{8,16,32}`, `cluster` method, `small_block_policy`, `nz_order`, and the L1/shared
carveout (`cudaFuncAttributePreferredSharedMemoryCarveout` — the 100 KB shared carveout
leaves only 28 KB of L1 per SM, and `X` is re-read across CTAs; at `p = 8` the arena is
small enough that the 64 KB-L1 carveout is affordable).

## 10. How this gets validated

Ordered so that a failure localizes.

1. **PBR parity, offline.** `target_nr = target_nc = 8`, `cluster='identity'`,
   `small_block_policy='remainder'` with PBR's `min_nnz` must reproduce PBR's blocking
   *exactly* — same blocks, same nonzero partition, different encoding. The cleanest possible
   anchor: it isolates the encoding change from everything else.
2. **Clustering does something, offline.** `x_loads` and `y_writes` should rank
   `random > identity > degree > rcm > graclus/metis`. If they do not, no CUDA work is
   justified yet.
3. **Round-trip.** `boo_to_csr(csr_to_boo(A)) == A` exactly, across every method, both
   shapes, both scalar types, all three policies.
4. **Format invariants.** `max_nr·pad2(max_nc) ≤ 65536`; every `nz_idx` decodes in range;
   nonzeros strictly increasing in whichever key `nz_order` selects; `row_ids` / `col_ids`
   are exact partitions; `Σ block nnz + remainder nnz == A.nnz`.
5. **A numpy reference matmul** straight from the BOO arrays, so a disagreement localizes to
   format-vs-kernel.
6. **Determinism.** Build twice, assert byte-identical arrays.
7. **GPU correctness** against `pbr_batched_matmul_cuda`, including the `row_skewed` preset,
   a highly-imbalanced-cluster case, and `features ∈ {100, 200, 520}`.
8. **Targeted `ncu` cells** (§10.1), then the full W&B sweep.

### 10.1 The headline experiment

`fem2d_permuted` vs `fem2d`. Both already exist in `_MATRIX_PROFILES`, and
`generate_permuted_fem_2d` was written specifically to destroy grid-aligned locality — the
exact matrix a clustered grid should rescue and a fixed grid cannot. `fem2d` is a built-in
negative control: its natural order is already good, so its clustering delta should be small.

Cheapest falsifiable claim in the project. Run it first.

**Reporting rules, non-negotiable:** always print `compressed_fraction` (state explicitly
that it is 1.0 by construction under `keep`, rather than letting it look suspicious), and
always print `ctas_per_sm` / `occupancy_pct` next to `gpu_time_ns` so an occupancy tax is
visible immediately rather than inferred three experiments later.

`SHOW_TO_ERAN.md` §4 found unchanged code swinging up to 29% between sweep runs from shared-
cluster variance, and §7 found a 41% swing at one cell. Sweeps are for breadth; the
single-config `ncu` cells are the load-bearing evidence, and sweep-level claims under ~30%
are noise unless they reproduce.

## 11. Known risks and open questions

- **The nnz cap binds at `p = 8`**, and CUDA sizes the arena from the largest block for every
  CTA (§9.4). Handled by build-time splitting; escalates to bucketing if the spread is wide.
- **The long tail of near-empty blocks** (§7) — open, decided from measurement.
- **`nz_order` default is `rowmajor`, against the bank model's prediction** (§8) — open,
  decided from measurement.
- **Coarsening tax.** Fewer, fatter CUDA blocks is what killed the earlier
  `rowgroup`/`colgroup`/`tile` variants. What is different here: those kept 128 threads/CTA,
  where the budget is 7.5 KB and the kernel already exceeded it — the `threads` lever was
  never used, and it is a 4× step, not a linear one. Register pressure also falls, since the
  bitmap scan, the packed 3-bit registers, and the per-thread COO arrays all disappear.
- **`torch_cluster` / `torch_sparse` availability** — `hem` covers the gap.
- **The graph model is an approximation of the hypergraph objective** (§5.4).

### Follow-ons, in priority order

1. **Row-exclusive blocks and the plain store.** Folding a row group's whole block-row into
   one CUDA block, accumulating across column groups into a single `smem_Y`, makes each `Y`
   element written exactly once — Step C becomes a plain store, realizing
   `STALL_REASONS_NOATOMIC_ABLATION.md`'s −40..66% as a **correct** result rather than a
   diagnostic ceiling. A schema extension (block → column-group list), not a rewrite. Trades
   grid width for atomic elimination.
2. **`cp.async` / `__pipeline_memcpy_async`** double-buffering the Step A load. Ampere-native,
   removes the register round-trip, and attacks the barrier count a small `p` costs
   (`features/p` passes).
3. **Warp-per-row Step B** with register accumulation and a single plain shared store.
   `SHOW_TO_ERAN.md`'s bottom line names warp-level pre-reduction as the required structural
   change, and BOO makes it a small delta. This is the one variant where smem pitch padding
   would need re-examining.
4. **Size-class bucketing**, if the histogram warrants it.

---

## 12. First measured results

All at **N=65536, features=512, fp32, RTX 3090**, CUDA-event wall time, mean of 20 runs
after 5 warmups. `compressed_fraction = 1.0` everywhere (the `keep` policy, so this is
structural, not a coincidence) — no CSR-fallback confound is possible in any of these cells.

### 12a. The format, at fixed work distribution

BOO uses `nnzflat`'s one-thread-per-(nonzero, feature) work distribution. So the honest
format comparison is against PBR's **`nnzflat`** kernel — comparing against PBR's `block`
(`zero_idle`) varies both the format *and* the work distribution at once.

| matrix | PBR `nnzflat` | **BOO** | BOO speedup |
|---|---|---|---|
| `fem2d_permuted` | 8.24 ms | **4.72 ms** | **1.74×** |
| `fem2d` | 8.50 ms | **4.64 ms** | **1.83×** |

**The format change is worth 1.7–1.8× at fixed work distribution.** That is the bitmap
going away (no scan, no popcount, no select-kth-set-bit, no per-thread COO rebuild), the
block-data hoist, free row compaction, and larger clustered blocks, combined.

### 12b. Clustering, on destroyed locality

`fem2d_permuted` is `fem2d` with mesh nodes randomly relabeled — built specifically to
destroy grid-aligned locality. BOO at `threads=256, p=32`:

| cluster | target | blocks | `x_loads` | occupancy | time |
|---|---|---|---|---|---|
| `identity` | 16 | 133,150 | 2,130,400 | 100% | 10.51 ms |
| `identity` | 32 | 129,624 | 4,147,968 | 100% | 16.20 ms |
| `identity` | 64 | 122,700 | 7,852,800 | 83% | 28.89 ms |
| `hem` | 16 | 35,255 | 495,358 | 100% | 4.98 ms |
| **`hem`** | **32** | **18,485** | **479,712** | **100%** | **4.67 ms** |
| `hem` | 64 | 10,022 | 476,245 | 50% | 6.43 ms |

**3.5× from clustering alone** (16.20 → 4.67 at target=32), tracking an 8.6× `x_loads`
reduction. And it lands on top of the *unpermuted* matrix's own number (4.60 ms,
`identity` on `fem2d`) — **the clustering fully recovers the locality the permutation
destroyed.** That was the cheapest falsifiable claim in the design (§10.1); it holds.

The negative control holds too: on `fem2d`, whose natural order is already good,
`hem` (4.68 ms) vs `identity` (4.60 ms) is a wash, exactly as predicted.

`x_loads` predicts wall time well within BOO, so the offline metric is validated as a
screening tool — clustering methods can be ranked without GPU time.

### 12c. `p = 8` was wrong

The design point was `p = 8`, on the argument that 8 fp32 = 32 B = one sector is all
coalescing needs (§9.1). Measured on `fem2d_permuted` with `hem`, target=32:

| `p` | threads=128 | threads=256 | threads=512 |
|---|---|---|---|
| 8 | 9.91 ms | 10.31 ms | 12.32 ms |
| 16 | 6.42 ms | 6.61 ms | 7.81 ms |
| **32** | 5.53 ms | **4.69 ms** | 5.44 ms |
| 64 | 7.94 ms | 5.06 ms | — |
| 128 | 11.39 ms | 7.95 ms | — |

**`p = 32` is a clear interior optimum, and `p = 8` is 2.1× off it.** Below 32 the barrier
count (`features / p` passes, each with three `__syncthreads()`) and the 4× wavefront
count dominate; above 32 the arena grows and occupancy collapses (at `p=128`,
`threads=128`: 16.7%). The §9.1 reasoning was right that 32 B is enough for *sector*
efficiency and wrong that sector efficiency was the binding constraint.

`threads` behaves as designed — a budget lever, not a direct win: it matters precisely
because it is what lets `p = 32` run at 100% occupancy (58% at `threads=128`).

**Defaults updated: `threads=256, p=32`.**

### 12d. What is left

| matrix | PBR `block` | BOO best |
|---|---|---|
| `fem2d_permuted` | 3.63 ms | 4.72 ms (1.30× slower) |
| `fem2d` | 2.37 ms | 4.64 ms (1.96× slower) |

BOO does not beat PBR's `block` kernel. But `block` is `zero_idle`, which accumulates a
whole row in a *register* and does one plain shared store, where BOO (like `nnzflat`)
issues a shared `atomicAdd` per (nonzero, feature). `SHOW_TO_ERAN.md` already measured
that work-distribution difference at 1.1–2.2× in `zero_idle`'s favour on PBR, and the
residual gap here is the same size — i.e. **the remaining deficit is the work
distribution, not the format**, and §11 follow-on 3 (warp-per-row register accumulation
before the shared store) is the specific fix. BOO makes it a small delta rather than a
rewrite.

Not yet measured: the `nz_order` isolate (§4.3, §8), the small-block policy decision (§7),
fp64, and any matrix other than the `fem2d` pair.

---

## 13. Optimizing Step B — the row-segmented kernel

§12's numbers are the **shared-atomic** kernel. Profiling it (`ncu`, the
`optimizer.performance.METRICS` list, `fem2d` N=65536 f512) showed the problem was not
where §3 assumed:

| metric | BOO (atomic) | PBR `block` | PBR `nnzflat` |
|---|---|---|---|
| gpu_time_ns | 5,488,032 | 2,647,008 | 8,690,816 |
| `sm_throughput_pct` | **92.08** | 30.24 | 68.63 |
| `alu_pipe_pct` | **40.40** | 16.96 | 21.57 |
| `long_scoreboard_pct` | 10.82 | 19.75 | **61.26** |
| `mio_throttle_pct` | 5.30 | **23.54** | 1.12 |
| **`short_scoreboard_pct`** | **29.28** | 13.69 | 6.51 |
| `barrier_pct` | 11.32 | 12.85 | 5.66 |

BOO was **not memory-bound** — `long_scoreboard` 10.8% and `mio_throttle` 5.3%, both far
below PBR. It ran at 92% SM throughput with 40% ALU pipe: issuing an enormous number of
instructions, top stall `short_scoreboard` (29.3%) = warps waiting on shared-memory
dependencies. That is the per-(nonzero, feature) `atomicAdd` into `smem_Y`.

**The fix.** `nz_order = rowmajor` sorts nonzeros by `local_row`, so a per-block row-start
table can be recovered by binary search (branch-free, no atomics, and rows the column
group does not touch fall out for free). Each thread then owns one *(row, feature)* item
and accumulates that row in a **register**, committing once. This removes every shared
atomic, `smem_Y` entirely and its zeroing pass, and one barrier per feature tile. It is
`zero_idle`'s structure on BOO's format — and `zero_idle` is the PBR kernel that wins.

`boo_spmm_nnzflat_kernel` is kept and is still used for `colmajor`, so `nz_order` remains
a real A/B axis.

### 13a. Isolated effect

Same format, same `rowmajor` ordering, same `p=32` — only Step B differs:

| matrix (N=65536) | shared atomic | register accumulate | |
|---|---|---|---|
| `fem2d` | 4.64 ms | **2.94 ms** | **1.58×** |
| `fem2d_permuted` | 4.72 ms | **3.03 ms** | **1.56×** |

Dropping `smem_Y` also shrank the arena enough that **`p = 64` now fits at 100%
occupancy** (12,612 B) where `p = 32` used to be the ceiling. New optimum:
**`threads=256, p=64`**. `p=128` still loses (20,804 B → 67% occupancy).

### 13b. Final standing vs PBR

Best BOO config (`hem`, target 32, `threads=256`, `p=64`) against PBR `block`:

| matrix | N | BOO | PBR `block` | |
|---|---|---|---|---|
| `fem2d_permuted` | 65,536 | **2.92 ms** | 3.66 ms | **BOO 1.25× faster** |
| `fem2d_permuted` | 262,144 | **11.46 ms** | 14.35 ms | **BOO 1.25× faster** |
| `fem2d` | 65,536 | 2.86 ms | **2.38 ms** | BOO 1.20× slower |
| `fem2d` | 262,144 | 10.99 ms | **9.46 ms** | BOO 1.16× slower |

**BOO beats PBR on the permuted matrix at both sizes, by the same 1.25×, and loses on the
naturally-ordered one by ~1.2×.** That split is the mechanism working as designed rather
than a surprise: BOO's advantage is recovering locality, and `fem2d` in its natural order
has none to recover — it is close to BOO's worst case, `fem2d_permuted` closer to its
best. The advantage is stable across a 4× size increase.

Still unmeasured: the `nz_order` isolate with bank-conflict counters, the small-block
policy, fp64, and matrices beyond the `fem2d` pair. Single runs; `SHOW_TO_ERAN.md` §4
recorded up to 29% run-to-run variance on this cluster, so the 1.25× and 1.56× effects
are clear of it but finer distinctions are not.

---

## 14. Full 4-way benchmark

**BOO vs PBR `block` (zero_idle) vs PBR `nnzflat_shared` (v6) vs PBR `nnzflat_onthefly` (v5).**

14 matrix profiles × {16384, 65536, 262144} × {128, 512} features, fp32 — 84 cells, 82
clean. Raw data: `tests/torch_fast_spdmm/benchmark/results/bench_4way_{hem,identity}.csv`.

### 14a. Method

`nnzflat_shared` and `nnzflat_onthefly` rewrite the *same* kernel function in place on
their respective branches, so the repo's existing approach is one build and one sweep per
branch. That is the wrong instrument here: `SHOW_TO_ERAN.md` §4/§7 measured `zero_idle`'s
own **unchanged** code swinging **29–41%** between sweep runs on this cluster — larger than
most of the effects being compared.

So both variants were resurrected verbatim into **one build** (`bench/4way`) as
`pbr_spmm_nnzflat_{shared,onthefly}_kernel`, and all four kernels are timed **back to back
in one process against the same matrix object**, with each cell the median of 3 interleaved
rounds. A node transient then perturbs every kernel in a cell equally. Measured
within-cell spread is **≤ 11%**, which is the noise floor for any claim below. All four
kernels were verified against dense before timing.

### 14b. BOO is only comparable if its config is chosen too

Every published comparison in this repo gives PBR a swept config (best-of-grid over
block size / `min_nnz` / smem tier). Pinning BOO to one clustering method is not the same
courtesy, and it matters enormously:

| | vs `block` | vs `nnzflat_shared` | vs `nnzflat_onthefly` |
|---|---|---|---|
| BOO, `hem` always | 0.828 | 0.949 | 1.421 |
| **BOO, best of {`hem`, `identity`}** | **1.026** | **1.175** | **1.760** |

Geomean over 82 clean cells, >1 = BOO faster. The clustering choice is worth **~1.9–2.0×
in either direction** — `hem` wins 57/82 cells, `identity` 25/82, and picking wrong costs
about as much as everything else in this document combined. `diag_b0.3` alone swings from
**0.40 → 2.08** on that choice.

**This is not an oracle.** `x_loads` (offline, no GPU) picks the faster clustering in
**75/82 cells (91%)**, and when it is wrong the penalty is negligible — geomean **1.0025×**,
worst cell 1.073×. So the best-of-two column is achievable in practice by running
`boo_info` on both labelings and taking the lower `x_loads`.

### 14c. Results

Geomean speedup of BOO (best-of-two) over each PBR kernel, clean cells only:

| by size | n | vs `block` | vs `nnzflat_shared` | vs `nnzflat_onthefly` |
|---|---|---|---|---|
| 16,384 | 28 | 0.974 | 1.126 | 1.646 |
| 65,536 | 28 | 1.013 | 1.173 | 1.803 |
| 262,144 | 26 | **1.098** | 1.233 | 1.842 |
| **overall** | **82** | **1.026** | **1.175** | **1.760** |

BOO is the fastest of the four in **28/82** clean cells.

| by matrix | vs `block` | | by matrix | vs `block` |
|---|---|---|---|---|
| `diag_b0.3` | **2.077** | | `pattern_corner_cluster` | 0.962 |
| `diag_b0.1` | **2.020** | | `diag_b1.0` | 0.935 |
| `fem2d_permuted` | **1.169** | | `scattered_2x4` | 0.893 |
| `scattered_8x1` | **1.099** | | `scattered_2x2` | 0.886 |
| `pattern_checkerboard` | 1.020 | | `scattered_1x8` | 0.906 |
| `pattern_row_skewed` | 1.019 | | `fem2d` | 0.791 |
| | | | `pattern_diag_stripe` | 0.781 |
| | | | `diag_skewed` | 0.503 † |

† `diag_skewed` is the one profile where PBR's `compressed_fraction` is 0.92, so those
cells are partly CSR-vs-CSR rather than a kernel comparison (`SHOW_TO_ERAN.md` §2's trap).
Excluded from nothing above, but do not read it as a clean loss.

### 14d. What this says

- **BOO edges `block` overall (1.026×) and the advantage grows with size** (0.974 → 1.013 →
  1.098). At the largest size it is a ~10% win. That is the opposite of PBR's own scaling
  story, where the block kernel loses ground as N grows.
- **BOO beats both `nnzflat` variants** — 1.18× vs the better one (`shared`), 1.76× vs
  `onthefly`. Since BOO shares `nnzflat`'s heritage, this is the cleanest statement that
  the *format* is doing real work.
- **`nnzflat_shared` (v6) is much better than `nnzflat_onthefly` (v5)** — the shared
  position table is worth ~1.5× — and v6 is close to `block` on many profiles, which is a
  more favourable reading of that line of work than `SHOW_TO_ERAN.md` §7 reached.
- **The remaining losses cluster on `scattered_*` and `pattern_diag_stripe`** — profiles
  whose nonzeros are structurally scattered, so there is little locality for clustering to
  find and BOO pays the format's per-block overhead without the reward. `fem2d` (0.791)
  is the same story: already well-ordered, nothing to recover.
- **The single largest lever is the clustering choice, not the kernel.** Getting it right
  is worth ~2× and costs nothing at runtime.
