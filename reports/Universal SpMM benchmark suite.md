**Universal SpMM benchmark suite**

Use **11 primary matrices: eight real SuiteSparse workloads and three deterministic controls**. This suite spans web and social graphs, roads, finite elements, molecular structure, circuit simulation, rectangular optimization, and a synthetic block-pruned neural layer. It deliberately combines favorable block structure with cases where grouping and padding should struggle. Treat it as a compact coverage suite, not a claim that eleven matrices represent every application. Compare all formats using the same canonical matrix and dense right-hand side, and report preprocessing separately from repeated SpMM execution.

**1. Dataset roster**

Density below is **100 × numerical NNZ / (rows × columns)**, expressed as a percentage. Real-matrix statistics are published SuiteSparse values for the full matrix, including symmetric expansion and excluding explicit zeros; verify them against the actual canonical artifacts during ingestion. Generated counts follow exactly from the constructions below. Global density is distinct from occupancy within stored blocks.

| Matrix / collection identifier | Domain | Dimensions, rows × columns | Numerical NNZ | Density (%) |
|---|---|---:|---:|---:|
| [SNAP/web-Google](https://sparse.tamu.edu/SNAP/web-Google) | Directed web hyperlinks | 916,428 × 916,428 | 5,105,039 | 0.000607858 |
| [SNAP/soc-Pokec](https://sparse.tamu.edu/SNAP/soc-Pokec) | Directed social network | 1,632,803 × 1,632,803 | 30,622,564 | 0.001148614 |
| [SNAP/roadNet-CA](https://sparse.tamu.edu/SNAP/roadNet-CA) | Road network | 1,971,281 × 1,971,281 | 5,533,214 | 0.000142390 |
| [Williams/cant](https://sparse.tamu.edu/Williams/cant) | Finite-element cantilever | 62,451 × 62,451 | 4,007,383 | 0.102750054 |
| [Williams/pdb1HYS](https://sparse.tamu.edu/Williams/pdb1HYS) | Protein structure | 36,417 × 36,417 | 4,344,765 | 0.327610610 |
| [Sandia/ASIC_680k](https://sparse.tamu.edu/Sandia/ASIC_680k) | Circuit simulation | 682,862 × 682,862 | 2,638,997 | 0.000565943 |
| [SNAP/amazon0312](https://sparse.tamu.edu/SNAP/amazon0312) | Product co-purchase network | 400,727 × 400,727 | 3,200,440 | 0.001993024 |
| [Mittelmann/rail4284](https://sparse.tamu.edu/Mittelmann/rail4284) | Railway set-cover / linear programming | 4,284 × 1,096,894 | 11,284,032 | 0.240132082 |
| `uniform16` — generated | Uniform-column, fixed-degree control | 262,144 × 262,144 | 4,194,304 | 0.006103516 |
| `blockdiag32` — generated | Independent dense subdomains | 262,144 × 262,144 | 8,388,608 | 0.012207031 |
| `pruned16x16` — generated | Block-pruned linear-layer proxy | 4,096 × 16,384 | 4,194,304 | 6.25 |

Three source distinctions matter. **ASIC_680k contains 1,232,776 explicit zeros**, giving 3,871,773 stored pattern entries but only 2,638,997 numerical nonzeros; remove the zeros for the primary track and retain both counts in metadata. Preserve SuiteSparse dimensions for web-Google and roadNet-CA even though their embedded SNAP descriptions quote smaller active-node counts. Pin **Mittelmann/rail4284** specifically: a historical Williams variant omits columns and is a different workload. Amazon is a product graph, not a rectangular user–item ratings matrix. These distinctions follow the linked collection records.

**2. Structural categories and hardware rationale**

The following mechanisms are hypotheses for profiling, not measured outcomes. Web/social graph labels do not establish a mathematical power-law fit; measure degree distributions before claiming one. Likewise, cant and pdb1HYS probe local dense structure but should not be labeled exactly block-diagonal. The generated control supplies that exact property.

| Structural category | Members | What this tests in PBR, BOO and BCOO |
|---|---|---|
| Hub-heavy / candidate scale-free | web-Google, soc-Pokec | Skewed row work stresses lane utilization and scheduling. Dispersed columns challenge input-feature cache reuse. Splitting output rows across groups exposes partial-sum and atomic costs where the kernel uses them. Compare row-length tails and groups touching each output row. |
| Community structure | amazon0312 | Tests whether grouping discovers useful input reuse and compact blocks. Compare native ordering with a separately labeled reordered track; measure clustering cost, block occupancy and remaining scattered entries. |
| Ultra-sparse, short-row | roadNet-CA | At about 2.8 NNZ/row, metadata, barriers and launches have little arithmetic to amortize them. Tests PBR/BOO small-block policies and CSR remainder coverage; dense-tile BCOO risks severe padding overhead. |
| Local dense structure and exact block diagonal | cant, pdb1HYS, blockdiag32 | Tests reuse of staged X across rows and the benefit of compact block encodings. Larger groups can save loads while raising shared-memory/register demand and reducing resident work. blockdiag32 gives a favorable reference for dense-tile BCOO. |
| Irregular circuit structure | ASIC_680k | Tests grouping robustness, small-block filtering and heterogeneous work sizes. Measure row and group-size distributions before attributing behavior to extreme rows. Explicit-zero handling also exercises dataset normalization. |
| Strongly rectangular, long-row | rail4284 | About 2,634 NNZ/row and a much wider input dimension stress reductions, X footprint and column grouping. Multiple blocks contributing to the small output-row set can expose output contention; square-matrix assumptions become visible. |
| Uniform scattered columns, balanced rows | uniform16 | Removes row-degree imbalance while retaining poor spatial grouping. Isolates gather reuse, metadata and block padding costs; reveals how much BOO/PBR work is delegated to the remainder. |
| Relatively dense, block-pruned | pruned16x16 | Regular long rows and fully occupied 16×16 tiles stress feature tiling and shared-memory capacity. Tests dense-tile BCOO efficiency and output accumulation; include dense GEMM as an additional baseline. |

**Random sparse indices do not automatically imply uncoalesced warp accesses.** When adjacent lanes process adjacent features, accesses to `X[j, f]` can coalesce even when `j` is random. The larger effect can be reduced reuse between rows or warps. Similarly, degree skew can produce inactive lanes or unequal CTA runtimes without branch divergence. Profile these mechanisms separately using transaction efficiency, L2 hit rate, active-lane behavior, shared-memory usage, occupancy and output-update traffic. NVIDIA documents the distinction between coalescing and redundant access, and the resource tradeoffs affecting occupancy. [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/).

For every matrix, record row-NNZ median/p95/p99/max and coefficient of variation, empty rows/columns, column-degree skew, useful block occupancy, allocated bytes per useful NNZ, remainder fraction, and groups contributing to each output row. These measurements explain results more reliably than global density alone.

**3. Exact definitions of the generated controls**

For `uniform16`, initialize NumPy `Generator(PCG64(20261117))`. Visit rows in ascending order; sample 16 distinct column IDs uniformly without replacement from `[0, 262144)`, sort them, and assign FP32 value `1.0`. Self-loops are allowed; do not enforce symmetry. This is a uniformly sampled column pattern with fixed row degree, **not** an Erdős–Rényi Bernoulli graph. Its exact NNZ is `262144 × 16`.

For `blockdiag32`, create 8,192 fully occupied 32×32 blocks on the diagonal: `A[r,c] = 1.0` exactly when `r // 32 == c // 32`. Include diagonal entries and omit everything else. Its exact NNZ is `8192 × 32 × 32`. This supplies ideal block fill and disjoint subdomains without relying on a real matrix's ordering.

For `pruned16x16`, use a 256×1,024 grid of 16×16 blocks. Initialize a fresh `Generator(PCG64(20261118))`; visit block rows in ascending order, sample 64 distinct block columns uniformly without replacement, sort them, and fill each selected tile with FP32 `1.0`. The scalar matrix is 4,096×16,384 with `256 × 64 × 16 × 16` nonzeros and exactly 1,024 NNZ per row. This is a synthetic block-pruning proxy, not a trained model or Ampere 2:4 sparsity workload; it does not establish Tensor Core applicability.

Pin the NumPy version and hash the generated CSR arrays because a seed alone does not freeze high-level sampling implementations. Optional permutation or signed-value checks must carry separate case names and hashes; do not count them as additional primary matrices. Use a seeded, nonconstant dense X shared across formats.

**4. File format and loading contract**

Use **SuiteSparse Matrix Market archives → canonical CPU SciPy CSR → format adapters**. Download the Matrix Market archive linked on each roster page, retain the archive, and select its primary matrix member. Matrix Market supports coordinate storage, one-based indices and symmetry declarations. `scipy.io.mmread` handles these conventions, expands symmetric storage and supplies unit values for pattern input; do not add the transpose again. [Matrix Market specification](https://math.nist.gov/MatrixMarket/formats.html), [SciPy reader](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html), [reader implementation](https://github.com/scipy/scipy/blob/main/scipy/io/_fast_matrix_market/__init__.py).

Normalize once: retain published shape and original ordering, sum duplicate coordinates in float64, remove explicit zeros, sort columns, reject complex/nonfinite values, and cast to FP32. Remove and report any additional zeros introduced by the cast. Preserve directed edges and existing diagonals; do not silently symmetrize, transpose, normalize, add loops or remove isolated vertices. Define graph multiplication as `Y[i,:] = Σj A[i,j] X[j,:]`; an aggregation convention requiring `A.T` becomes a separately named case. Canonical CSR has sorted columns and no duplicates, while raw `.nnz` can count stored zeros. [SciPy CSR documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).

Cache a CSR `.npz` and JSON manifest outside the timed path. Record group/name, resolved URL, archive SHA256, member path, original header, published and canonical shape/NNZ, zero/duplicate policy, value/index dtypes, package versions, generator specification, canonical-array hash and any permutations. Fail ingestion on unexplained count discrepancies. PyG is unnecessary for this mixed numerical/graph roster; a graph loader is acceptable only if it reproduces this exact contract.

The current repository accepts `scipy.sparse.csr_matrix` specifically; force that class rather than relying on `csr_array`. The following is an **unexecuted integration sketch**, using the existing public adapters:

```python
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch_fast_spdmm as fast

raw = sio.mmread(matrix_path)
if np.iscomplexobj(raw.data):
    raise ValueError("Real-valued suite required")
a = sp.csr_matrix(raw, dtype=np.float64)
a.sum_duplicates()
a.eliminate_zeros()
a.sort_indices()
if not np.isfinite(a.data).all():
    raise ValueError("Nonfinite matrix values")
a = a.astype(np.float32)
a.eliminate_zeros()  # record any NNZ change caused by casting
if max(*a.shape, a.nnz) >= 2**31:
    raise ValueError("Use a separately declared int64 track")
a.indices = np.ascontiguousarray(a.indices, dtype=np.int32)
a.indptr = np.ascontiguousarray(a.indptr, dtype=np.int32)

pbr = fast.csr_to_pbr(a, block_rows=8, block_cols=8,
                      min_nnz_per_block=4)
boo = fast.csr_to_boo(a, target_nr=32, target_nc=32, seed=0)
csr_gpu = torch.sparse_csr_tensor(
    torch.from_numpy(a.indptr), torch.from_numpy(a.indices),
    torch.from_numpy(a.data), size=a.shape, device="cuda")
```

These PBR/BOO settings are starting points, not recommended universal optima. Here PBR refers to the repository's **Packed Block Representation**, matching the implementation being evaluated despite the request's “Pattern-Based” expansion. The repository's BOO stores sparse entries within clustered groups; it is distinct from a dense-block format. **No separate BCOO public adapter was found.** For this proposal, define BCOO as fixed `(block_rows, block_cols)` coordinates with dense value tiles, including padding. Document tail handling and preserve logical shape; report useful NNZ divided by allocated tile slots. If BCOO instead means sparse coordinates within each block, revise this contract before comparing results. Do not silently substitute PyTorch BSR or JAX's differently named BCOO. [Repository BOO design](../docs/reference/BOO_FORMAT.md), [PBR API](../torch_fast_spdmm/torch_fast_spdmm/__init__.py), [BOO API](../torch_fast_spdmm/torch_fast_spdmm/boo.py).

**5. RTX 3090 evaluation protocol**

Use FP32 with contiguous row-major `X[n,F]`, int32 indices where valid, and **F = 32, 64, 128, 256, 512**. Add 33, 65 and 129 as separate feature-tail probes; keep FP64 results separate. Compare direct cuSPARSE CSR, the naive CSR baseline and each custom format. Warm up, interleave candidate order, synchronize correctly across all participating streams, and report medians plus dispersion. CUDA execution is asynchronous, so host dispatch time alone is not kernel latency. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).

Report preprocessing, kernel timing and complete-call timing separately. The complete call must include required output initialization, CSR remainder and stream joins. Also report `Tprep/K + Tcall` for reuse count K. Native-order results form the primary track; reordered results must include corresponding X/Y permutations and their cost. Tune on a declared subset and freeze the chosen heuristic for held-out matrices; distinguish that result from an oracle selecting each case's best block shape.

The RTX 3090 has 24 GB of memory. One FP32 X/Y pair requires `4F(n+m)` bytes before sparse formats, workspaces or validation buffers; roadNet-CA at F=512 alone requires about **7.52 GiB**. Run matrices and candidate formats sequentially, budget peak memory explicitly, and never densify the large sparse inputs. [NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/).

Prepared 17 November 2026. This is a dataset and protocol design: published statistics were researched, but archives were not downloaded, generated controls were not materialized, and no benchmark results are claimed. All extension compilation, correctness tests and performance runs must execute through SSH on the remote GPU node under the repository policy.
