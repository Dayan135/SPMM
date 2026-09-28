# HEM-reordered FEM with the existing PBR implementation

The experiment is complete. **Turning the HEM groups into a permutation gives PBR a modest improvement, but does not recover original FEM performance or explain away BOO's advantage on the largest case.** The BOO kernel choice also has a substantial, independently measured effect.

Results were collected on an RTX 3090 in one isolated build/process, Slurm job `21719128`, on 2026-09-28. The full run has 360 validated configuration summaries and 7,200 timing samples: N=16,384 / 65,536 / 262,144, feature widths 128 / 512, seed 0, five warmups and 20 interleaved timing rounds. These repetitions measure timing variability for one input seed and one GPU job.

## Requested comparison

For the largest workload, **N=262,144, features=512**, median complete-matmul CUDA-event times were:

| Case | Fixed parameters | Best measured configuration |
|:---|---:|---:|
| Original `fem2d` + PBR | 9.079 ms | 9.079 ms |
| Randomly permuted `fem2d` + PBR | 13.816 ms | 13.816 ms |
| Randomly permuted `fem2d` + BOO/HEM | 10.929 ms | 10.929 ms |
| **HEM-reordered `fem2d_permuted` + PBR** | **13.717 ms** | **12.473 ms** |

Fixed parameters mean PBR 8×8, `min_nnz=2`, `smem_size=2048`, existing `block` kernel; HEM target/cap 32; BOO rowseg, 256 threads, p=64, legacy launch, remainder threshold 4. Neither the PBR implementation nor any CUDA kernel was changed.

The tuning sweep gives every PBR order the same eight `(block size, min_nnz)` choices: `(2,2)`, `(2,3)`, `(4,2)`, `(4,4)`, `(4,8)`, `(8,2)`, `(8,8)`, `(8,12)`. Both BOO and reordered PBR sweep HEM targets 8/16/32/64. The largest-case reordered winner is HEM k=16, PBR 8×8/min_nnz=12; BOO's winner is k=32. These are post-hoc minima, not held-out tuning results.

Across all six workloads:

- Fixed-parameter random-PBR / HEM→PBR speedup is **0.98–1.07×**: little benefit, including one small regression.
- Best-measured random-PBR / HEM→PBR speedup is **1.07–1.14×**. Original-order PBR remains fastest in every workload.
- At N=262,144/f=512, BOO/HEM is **1.26× faster than random PBR**, **1.26× faster than fixed HEM→PBR**, and **1.14× faster than tuned HEM→PBR**.
- Tuned HEM→PBR sends **9.8–14.1% of nonzeros to CSR remainder**. Thus its tuned gain includes a change in the division of work between PBR and CSR. The fixed-parameter table avoids threshold tuning.

The [full comparison](tests/torch_fast_spdmm/benchmark/results/hem_pbr_20260928/results/full/comparison.md) contains every workload's headline timings, structural statistics and extra controls. The [summary CSV](tests/torch_fast_spdmm/benchmark/results/hem_pbr_20260928/results/full/summary.csv) retains all configurations and IQRs; [samples.csv](tests/torch_fast_spdmm/benchmark/results/hem_pbr_20260928/results/full/samples.csv) retains all raw timings.

## How the permutation was constructed

Start with symmetric original FEM `A0`. The existing `fem2d_permuted` profile shuffles **whole four-DOF mesh nodes**, preserving the DOFs within each node. Using its permutation `q`, construct `A = A0[q][:, q]`; the harness checks equality against the existing profile generator.

Compute HEM once on `A`, using symmetric row/column labels, equal targets/caps and `feasible_first=True`. Give those exact labels to BOO. For PBR, use:

```python
p = np.argsort(hem_labels, kind='stable')  # new vertex index -> old vertex index
A_prime = A[p][:, p].tocsr()              # P A P.T, with P[i, p[i]] = 1
pbr = csr_to_pbr(A_prime, ...)
```

This concatenates groups in increasing label order and retains increasing vertex IDs within each group, matching BOO's group order. There is no padding, second clustering pass, or reconstruction of the mesh order.

The dense input follows the same coordinate changes: `X = X0[q]`, `X_prime = X[p]`. After multiplying, `Y0[q[p]] = Y_prime` restores the original order. Every configuration is checked this way against `A0.astype(float64) @ X0.astype(float64)`, at rtol=atol=1e-4. Tests also cover a non-self-inverse permutation, exact BOO group-order agreement, symmetry and inverse sparse reconstruction.

## Clustering, representation and kernel

**Ordering benefit with PBR held fixed.** Random PBR versus HEM→PBR isolates the performance effect of this particular HEM-derived ordering. It does not establish that HEM has reconstructed mesh locality. HEM is a partitioning method; its label order and the order inside each group do not optimize a fixed PBR grid.

**Representation and block partition.** At N=262,144 and HEM k=32, original PBR has 292,612 blocks; random PBR has 553,908; HEM→PBR still has 537,058, whereas BOO directly represents the HEM partition with 67,961 blocks. Fixed HEM→PBR retains 99.93% of nonzeros in PBR, and BOO retains 100%. Thus the small fixed-ordering gain is not predominantly CSR fallback. The variable HEM group boundaries do not generally align with PBR's fixed 8×8 tiles; the large remaining block-count difference is consistent with that mismatch. Block counts alone do not quantify runtime or prove a pure representation speedup.

**Kernel effect on identical representation.** The harness additionally runs BOO's existing shared-atomic and rowseg kernels on the **same BOO object**. At k=32, rowseg is **1.55–1.74× faster** across the six workloads. At the largest case the timings are approximately 18.97 ms shared-atomic versus 10.93 ms rowseg. This is a direct kernel control without changing clustering or stored data.

**Coordinate-locality control.** The same HEM groups are also represented as BOO after relabeling to the reordered coordinates, using `labels[p]` rather than reclustering. At k=32, random-BOO / reordered-BOO speedup is approximately **0.98–1.00×**. Simply making the vertex coordinates contiguous does not materially accelerate this BOO configuration in this run.

The requested four cases separate HEM ordering from the combined BOO execution stack. The added same-format kernel control isolates one kernel effect. **They do not yield a complete additive decomposition into clustering, representation and kernel costs:** PBR and BOO still have different block partitions, metadata and execution mechanics.

## Preprocessing and scope

Timing covers a complete matmul with GPU-resident inputs, including output initialization and auxiliary-stream joins. It excludes HEM, sparse permutation, format construction, staging, and dense input/output permutations. It is appropriate when working repeatedly in reordered coordinates; it is not the total cost of a one-off call with external inputs/outputs in original coordinates.

HEM takes roughly **0.59 s / 2.3 s / 10.3–10.6 s** for the three sizes. Sparse reordering takes approximately **14–15 ms / 58–67 ms / 263–271 ms**, including the symmetry check and CSR canonicalization. BOO and HEM→PBR share the same one-time HEM computation; it must not be counted twice. Format construction and staging are recorded separately per configuration. Dense permutation costs were not benchmarked.

## Reproduce and validate

Build on a GPU node using the repository's `CLAUDE.md` instructions, then from `tests/torch_fast_spdmm`:

```bash
python -m benchmark.bench_hem_pbr --out results/hem-pbr
python -m benchmark.report_hem_pbr results/hem-pbr
```

`--sizes`, `--features`, `--seeds`, `--hem-k`, `--pbr-configs`, `--warmup` and `--repeats` control the sweep. The output directory must be new. The experiment does not upload to W&B.

GPU validation: **748 passed, 12 skipped**, covering the new permutation tests, matrix generators, BOO and PBR tests. All 360 full-run configuration outputs and all smoke-run outputs passed the independent numerical check. Logs and the exact job script are in [the run directory](tests/torch_fast_spdmm/benchmark/results/hem_pbr_20260928).

The isolated source snapshot came from commit `4492e8c4a5893a776884ca97f0d08a8d45210233` plus the new benchmark and test files. Raw metadata includes measured source/extension hashes. Because the snapshot had no `.git` directory and its provenance sidecar was written after execution, raw `git_commit`/`git_status` say `unknown`; the sidecar records the origin. Its inherited `baseline` description mentions cuSPARSE, but **no cuSPARSE case was run**; the benchmark source now overrides that descriptive field. Raw measurement files are retained unchanged.
