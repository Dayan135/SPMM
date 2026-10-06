# METIS and Graclus through the 16-bit BOO limit

The BOO builder accepts `cluster='metis'` and `cluster='graclus'`. Both use the
same binary sparsity-pattern similarity graph as HEM: shared-column counts,
hub degree 1024, top 32 neighbors before symmetrization. Symmetric patterns with
equal targets share row/column labels; other inputs cluster each axis separately.

```python
boo = csr_to_boo(A, cluster='metis', target_nr=256, target_nc=256, seed=0)
boo = csr_to_boo(A, cluster='graclus', target_nr=256, target_nc=256, seed=0)
```

## Backends and installation

- **METIS:** PyMETIS >=2025.2.2, weighted k-way partitioning with
  `ceil(n / target)` partitions and unit vertex weights (row-count balance).
  The shared cap pass splits any oversize partition and enforces the optional
  nonzero cap. This can increase partition count and reduce partition quality;
  target sizes are upper bounds in the BOO builder, not guaranteed exact sizes.
- **Graclus:** `torch_cluster.graclus_cluster` on CPU, iterated on unions of
  column supports. Infeasible edges are removed before each matching call, so
  merges respect size and optional nonzero caps. Seeded CPU visitation is
  isolated from the caller's Torch RNG. This is the PyG greedy matching routine,
  not the full original Graclus normalized-cut partitioner with refinement.
- Requests for unavailable backends raise an error. They never substitute HEM.
- HEM remains the default; its existing implementation is unchanged.

Install on the remote GPU environment. PyMETIS is available through the `metis`
extra (`pip install -e 'torch_fast_spdmm[metis]' --no-build-isolation`) or
`pip install 'pymetis>=2025.2.2'`. Install a `torch-cluster` wheel matching the
installed PyTorch/CUDA versions from the official PyG wheel index; the `graclus`
extra declares the dependency but does not select a compatible binary wheel.

Backend references: [PyMETIS API](https://documen.tician.de/pymetis/functionality.html),
[torch-cluster installation and Graclus](https://github.com/rusty1s/pytorch_cluster).

## Encoding and GPU feasibility

Packed coordinates use `index = (row << ceil(log2(ncols))) | col`. The largest
coordinate must be <=65535. Thus **256x256 is valid**, including coordinate
(255,255). Powers-of-two rectangles such as 128x512 also fit, but this experiment
uses square groups. For non-power-of-two widths, account for the padded column
stride rather than checking `nrows*ncols` alone.

Encoding capacity is independent of shared-memory capacity. For float32 and the
row-segmented kernel, the launch requests approximately

```
6 * max_block_nnz + 4*(max_nr + max_nc) + 4*(max_nr + 1)
  + 4*p*max_nc + 16 bytes
```

A dense 256x256 block needs 393216 bytes for sparse values and indices alone.
It cannot launch on the current RTX 3090 kernel. Sparse 256x256 groups can fit.
The sweep selects the largest fitting feature tile from 64,32,16,8 (also bounded
by feature count). If P=8 still fails the device limit, it records the candidate
as unsupported, without changing its groups. The selected tile is reported;
comparisons therefore measure clustering plus this fixed launch-selection rule.

## Reproduction

Run all compilation and tests on a GPU node, per CLAUDE.md. After building:

```bash
pytest tests/torch_fast_spdmm/test_clustering_backends.py \
  tests/torch_fast_spdmm/test_boo_edges.py \
  tests/torch_fast_spdmm/test_boo.py --boo-require-cuda -q

cd tests/torch_fast_spdmm
python -m benchmark.bench_clustering \
  --methods identity hem metis graclus --groups 8 16 32 64 128 256 \
  --sizes 16384 65536 --features 128 512 \
  --profiles fem2d fem2d_permuted diag_b0.3 diag_b1.0 \
  --out results/clustering
```

The output directory must be new. The benchmark saves backend versions, source
and extension hashes, matrix/input hashes, labels, preprocessing/staging times,
actual group sizes, X loads, Y writes, remainder fraction, shared-memory use,
per-call timing samples and correctness results. Timings interleave all supported
configurations in one process against identical A/X, with a PyTorch CSR baseline.
Clustering and construction are outside the timed matmul calls. No W&B run is
created. These are single-session measurements, not replicated performance claims.

## Results

Initial run: Slurm job `22326694`, RTX 3090 on `cs-3090-05`, N=16384,
F=128/512, seed 0, PyMETIS 2025.2.2, torch-cluster 1.6.3, Torch 2.10.0+cu128.
The GPU suite passed 508 tests with 13 skips. The subsequent smoke and full
benchmark runs completed. The full run has 196 validated configuration/feature
rows and four unsupported rows (identity/METIS at target 256 on `diag_b1.0`,
for both feature counts). The unsupported cases exceed shared memory even at P=8.

At N=16384, best measured full-call times at F=512 (milliseconds; group target
in parentheses) are:

| Matrix | HEM | METIS | Graclus |
|---|---:|---:|---:|
| fem2d | 0.703 (16) | 0.761 (32) | 0.709 (16) |
| fem2d_permuted | 0.730 (32) | 0.751 (32) | 0.732 (32) |
| diag_b0.3 | 0.390 (32) | 0.389 (32) | 0.416 (32) |
| diag_b1.0 | 0.878 (32) | 0.825 (32) | 0.825 (32) |

These are post-hoc minima from a single sweep, not held-out or replicated wins.
At target 256, FEM groups actually reach 256 with both new backends, and the
GPU multiplication passes. Their feature tile shrinks to P=32. On the diagonal
profiles, HEM and Graclus stop at 64-row connected components even when asked for
128/256; METIS may combine disconnected components to fill the requested parts.
Those larger targets therefore do not imply identical actual group sizes.

Artifacts: [raw results and saved labels](tests/torch_fast_spdmm/benchmark/results/clustering_20261006).
The uploaded source is retained as `measured_source_initial.tar.gz`; its source
fingerprint matches the initial metadata (`f42072bf…`). The base local commit is
`c2ccde58c8d6849ebdc2b86956032516538f86fa`, plus this implementation. The remote
snapshot has no Git checkout; metadata records the actual source/extension hashes.

Final-source GPU validation in job `22331033` passed **509 tests, with 13 skips**.
This includes all 24 new backend/launch-selection cases; neither backend was
skipped. The final source archive is `measured_source_final.tar.gz` (fingerprint
`30b74612…`); the source change after the initial run only updates tests.

The N=65536 sweep completed in job `22331033` on `dt-3090-01`, with the same
backend versions and protocol. Its source fingerprint matches the final archive.
It also has 196 validated rows and four unsupported rows, with the same
identity/METIS target-256 memory exclusions on `diag_b1.0`. Both main sweeps
together retain **7840 individual matmul samples**; the smoke run adds 240.

Best measured F=512 times at N=65536 (milliseconds; requested group target in
parentheses):

| Matrix | HEM | METIS | Graclus |
|---|---:|---:|---:|
| fem2d | 2.658 (32) | 2.763 (32) | 2.596 (32) |
| fem2d_permuted | 2.874 (32) | 2.964 (32) | 2.825 (16) |
| diag_b0.3 | 1.245 (256*) | 1.244 (64) | 1.248 (256*) |
| diag_b1.0 | 3.171 (256*) | 3.031 (32) | 3.048 (32) |

\* Actual groups have 64 rows. Targets 64/128/256 produce the same labels in
these cases; small timing differences between them are repeated measurements
of the same grouping, not evidence favoring a target of 256.

For a fixed target of 32, CPU clustering times at N=65536 (seconds):

| Matrix | HEM | METIS | Graclus |
|---|---:|---:|---:|
| fem2d | 3.96 | 3.75 | 2.34 |
| fem2d_permuted | 4.66 | 4.26 | 3.08 |
| diag_b0.3 | 25.97 | 6.60 | 4.18 |
| diag_b1.0 | 4.31 | 2.98 | 2.14 |

These preprocessing numbers are single CPU wall-clock measurements. Matmul time
is the primary metric here (preprocessing is amortized over repeated
multiplications), and on that metric Graclus is generally close to HEM; neither
new backend establishes a universal runtime win. Graclus did reduce preprocessing
time on all four tested profiles, which is a secondary benefit only.

Actual 256-row FEM groups pass validation, but are slower: on permuted FEM,
N=65536/F=512, METIS takes 7.116 ms at target 256 versus 2.964 ms at target 32;
Graclus takes 7.612 ms versus 2.866 ms at target 32. The 256 target uses P=32,
while the 32 target uses P=64. This is the measured combined effect of larger
groups and the launch-memory constraint. The 16-bit coordinate ceiling is usable,
but effective group sizes around 16–64 were preferable on these matrices.

Detailed outputs: [N=16384 summary](tests/torch_fast_spdmm/benchmark/results/clustering_20261006/results/full/summary.csv),
[N=65536 summary](tests/torch_fast_spdmm/benchmark/results/clustering_20261006/results/large/summary.csv),
[final GPU validation log](tests/torch_fast_spdmm/benchmark/results/clustering_20261006/job-22331033.log).

## Plots

Generated locally from the two main sweeps (no GPU needed):

```bash
cd tests/torch_fast_spdmm
conda run -n fem python -m benchmark.plot_clustering   # -> plots/clustering_20261006/
```

| Figure | Question |
|---|---|
| `A1_time_vs_target_{16k,64k}.png` | Matmul time vs target group size, per profile × F (p25–p75 band, cuSPARSE dashed, × = unsupported) |
| `A2_best_speedup.png` | Best-target speedup over cuSPARSE per backend |
| `A3_relative_heatmap.png` | Every (backend, target) relative to the best config per (profile, F, N) |
| `B4_group_sizes_*.png` | Requested target vs actual mean/max row and column group size |
| `B5_structure_vs_time_*.png` | Remainder fraction, max nnz/block, #blocks, X-row loads/N, smem arena vs time |
| `B6_tile_p_*.png` | Feature tile P selected by the shared-memory fit |
| `C7_scaling.png` | Best speedup at N=16k vs 65k |
| `C8_distributions_*.png` | Per-sample timings of the top-3 configs + cuSPARSE |
| `D9_breakeven.png` | Matmuls needed to amortize preprocessing vs cuSPARSE (secondary) |
| `best_per_backend.{csv,md}` | Table view of the best config per backend |

Observations from the plots (matmul time; same single-session caveat as above):

- **Backend choice barely matters once rows are clustered.** At the best target, HEM, METIS
  and Graclus are within 10% of each other on every profile, with METIS usually the slowest
  on FEM (5–9%). `identity` is equally good except on `fem2d_permuted`, where it is
  1.7–2.2× slower. Clustering pays off
  only when the input ordering is scrambled.
- **Target ≈32 is the robust choice.** In A3 it is best or within 1.10× of the best in
  nearly every column for every backend.
- **BOO beats cuSPARSE only in a few cells:** `diag_b1.0` at N=65536 (1.03–1.09×)
  and `fem2d_permuted` F=512 at N=65536 (HEM 1.01×, Graclus 1.03×). Everywhere else
  it is slower, down to about 0.55× on `fem2d`. The gap to cuSPARSE shrinks from 16k to 65k (C7).
- **The large-group slowdown is not only the tile shrink.** On FEM, time already roughly
  doubles at target 128 while P is still 64 (B6). P drops to 32 only at 256.
- In B5, X-row loads/N is the clearest predictor of the worst (identity) configs; the other
  structural metrics separate configs weakly when all profiles are pooled.
