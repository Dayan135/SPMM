# PBR / BOO / cuSPARSE CSR comparison

For the controlled symmetric FEM experiment that applies the same HEM grouping
to BOO and to a permutation before running existing PBR, see
[HEM_PBR_EXPERIMENT.md](../2026-09-28_reorder_vs_boo/HEM_PBR_EXPERIMENT.md). It includes the four requested
ordering/format cases, same-format BOO kernel controls, timings and raw results.

Run `benchmark.bench_spmm` from `tests/torch_fast_spdmm` after building the current
source on the remote GPU node (see `CLAUDE.md`). This harness is separate from
the historical four-way PBR kernel experiment.

```bash
python -m benchmark.bench_spmm --out results/spmm-current
```

Defaults:

- PBR `block` kernel: `(block size, min_nnz)` = `(2,2)`, `(2,3)`, `(4,2)`,
  `(4,4)`, `(4,8)`, `(8,2)`, `(8,8)`, `(8,12)`; `smem_size=2048`.
- BOO HEM: `k=8,16,32,64`, meaning `target_nr=target_nc=k`, with matching caps.
  Row-major/row-segmented kernel, 256 threads, P=64, remainder policy, threshold 4.
- CSR baseline: CUDA `torch.sparse.mm`, using float32 values and int32 indices.
  This measures the PyTorch cuSPARSE path, not a manually tuned direct cuSPARSE API.
- The existing 14 matrix profiles, requested sizes 16,384 / 65,536 / 262,144,
  and feature widths 128 / 512. Actual matrix dimensions are recorded separately.
- Five warmup calls per configuration, then 20 individually measured calls per
  configuration in interleaved rounds. A seeded initial order rotates each round.
  No outliers are discarded; the headline time is the median in milliseconds.

Every configuration in a workload uses exactly the same seeded original matrix
and dense input. The FEM/permuted-FEM pair uses the same seed. Every output is
checked against SciPy multiplication of the original matrix in float64 before
warmup, with rtol/atol 1e-4. Failures abort the run and retain completed cells.

Timing uses CUDA events around a complete matmul call with resident inputs.
It includes output allocation/initialization effects, dispatch gaps and auxiliary
stream joins. It excludes format construction, host-to-device staging,
correctness checks and logging. This is neither isolated kernel duration nor
synchronized CPU request latency. Conversion and staging wall times are recorded
separately in milliseconds. Repeated input reuse represents steady-state use.

`summary.csv` contains every configuration, median/mean/std/min/max/quartiles,
preprocessing/staging times, correctness error, format population, and speedup
`median CSR time / median configuration time`. `samples.csv` contains all 20 raw
times with round and execution position. `metadata.json` records arguments,
source and extension hashes, original Git state, package paths, GPU and library
versions, W&B URL and completion status. Matrices and dense inputs have hashes.
Completed workload cells are flushed locally before W&B logging.

W&B uses project `torch-fast-spdmm` by default, the authenticated default entity,
per-workload tables, paginated aggregate tables, and artifacts containing the
complete CSV/metadata files. `--entity`, `--project`, and `--name` can override
the destination. `--wandb-mode offline` retains an uploadable local W&B run;
the default is online. Logging occurs outside timing.

Use a new output directory for each run; existing results are never overwritten.
Sweep options include:

```bash
python -m benchmark.bench_spmm \
  --pbr-configs 2:2 4:4 8:8 --pbr-smem 512 1024 2048 \
  --boo-clusters hem identity --boo-k 8 16 32 64 \
  --features 16 32 64 128 256 512 --out results/spmm-expanded
```

All configurations remain visible; no workloads are excluded based on compressed
fraction. Choosing the fastest configuration afterward is tuned performance and
must be labelled accordingly. Twenty repeats characterize timing variability for
one seeded input, not variability across different inputs or GPU jobs.

Generate a comparison table and optional plot locally (no CUDA needed):

```bash
python tests/torch_fast_spdmm/benchmark/report_spmm.py \
  tests/torch_fast_spdmm/benchmark/results/spmm_20260923/full/summary.csv --plot
```

This produces `comparison.md`, `best_measured.csv`, and `comparison.png` beside
the input. PBR/BOO selections are explicitly labelled best measured configurations;
the original summary still contains the full grid. The plot uses milliseconds
on a logarithmic axis and shows interquartile ranges.

## Recorded run

The 2026-09-23 run completed all 84 workloads with 1,092 configuration summaries
and 21,840 raw timing samples. The GPU validation suite passed 1,152 tests with
13 expected skips. Results and interpretation are in
[`results/spmm_20260923/full/ASSESSMENT.md`](../../tests/torch_fast_spdmm/benchmark/results/spmm_20260923/full/ASSESSMENT.md)
and [W&B](https://wandb.ai/private-dayan/torch-fast-spdmm/runs/4nk9coe9).
