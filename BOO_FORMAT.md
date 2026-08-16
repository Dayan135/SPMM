# BOO — Blocked COO

A clustered, variable-size, bitmap-free sparse-matrix format for SpMM, and the CUDA
kernel that consumes it.

**Status:** design. Nothing in this document has been measured yet. Claims that are
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
