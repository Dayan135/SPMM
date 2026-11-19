# `bench.1`: benchmark output reference

`run_benchmarks(kernel_name, matrix_datasets)` returns this object. It is the only benchmark data the optimization
agent sees each iteration, so it is kept small. Everything else goes into the raw CSV named in `raw`. The JSON Schema is
[`schemas/bench.v1.json`](schemas/bench.v1.json), and a full example is
[`schemas/examples/bench.example.json`](schemas/examples/bench.example.json).

## How a benchmark run is measured

To read the fields, you need to know how a run works:

1. **Two builds: candidate and baseline.**
   - The candidate is the commit under test (`cand`).
   - The baseline is the last commit that passed every check (`base`, the `opt/verified` git tag by default).
   - Each build is a separately compiled `.so` in its own build directory.
2. **One Slurm job, one GPU.** Both builds are timed in the same job on the same node. So `x_base` never compares two
   different GPUs.
3. **Rounds.** Python can't load two versions of the same extension module in one process, so each build runs in its
   own Python process.
   - The job runs `env.rounds` rounds (default 3).
   - Each round runs the baseline process and the candidate process in a fresh random order. Neither build always
     goes first (warmer GPU, different clocks).
4. **Inside each process, every case is timed** (case = matrix × feature count F). The timed arms are:
   - the kernel in `cpp_only` scope;
   - a duplicate of the same `cpp_only` arm (`_dup`);
   - the kernel in `full_call` scope;
   - cuSPARSE (`DirectSpMM`, `cpp_only`).

   The arm order is shuffled on every repeat (`bench_nnzflat.measure`). There are `env.warmup` untimed calls, then
   `env.reps` timed calls per round.
5. **Correctness is checked before any timing.** Every arm's output is compared with a float64 reference using a
   per-row error bound. A failing arm is never timed.

**Two timing scopes** (the project learned this the hard way, see `CUSPARSE_DIRECT_SYMMETRIC.md`):

| Scope | What is inside the timed interval | Used for |
|---|---|---|
| `cpp_only` | CUDA events placed in C++: zeroing Y, the format kernel(s), the CSR-remainder kernel, the stream join. No Python, no pybind11, no argument checks. | `k_ms`, `base_ms`, `cusp_ms`, and every speedup |
| `full_call` | CUDA events placed in Python around one complete public API call (`pbr_matmul` / `boo_matmul`) | `call_ms` only (wrapper overhead) |

Speedups always compare like with like: `cpp_only` against `cpp_only`.

## Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `v` | `"bench.1"` | Schema name and version. If the layout ever changes, this becomes `bench.2`, so a reader can't confuse old and new results. |
| `ok` | bool | `true` only if no case failed correctness (no `F` verdict). `opt/verified` may advance only when this is `true` **and** the pytest suite passed. |
| `fmt` | `"pbr"` \| `"boo"` | Sparse format under test. BCOO would be added later as a new value. |
| `kernel` | string | Kernel variant inside the format, e.g. `block`, `nnzflat` (PBR) or `rowseg`, `nnzflat` (BOO). It is the part after `:` in the `kernel_name` argument (`"boo:rowseg"`). |
| `cand` | short sha | Candidate commit. |
| `base` | short sha | Baseline commit. |
| `env.node` | string | Cluster node that ran the job. Check it against the excluded-node list when results look odd. |
| `env.job` | int | Slurm job id, for finding the `.out`/`.err` logs. |
| `env.dtype` | `"fp32"` | Value precision. FP64 would be a separate run, never mixed in. |
| `env.reps` / `env.warmup` / `env.rounds` | int | Timed calls per round, untimed warm-up calls, number of base/candidate process rounds. |
| `noise` | fraction | This job's timing noise floor (see below). For example, `0.031` means differences under about 3.1% are indistinguishable from noise **in this job**. |
| `sum` | object | Aggregates; see the next table. |
| `prep_ms` | {matrix: ms} | Format-construction time for the candidate (`csr_to_pbr` / `csr_to_boo`, host side, plus upload). It is reported once per matrix because it does not depend on F. It is never included in `k_ms`. For BOO it excludes HEM clustering: `agent_bench` caches the labels per (matrix, params, `clustering.py` hash) under `suite/boo_labels/`. An attempt that edits clustering misses the cache. |
| `cols` | fixed list | Column names for every row in `rows`, in order. They are listed once so each row doesn't repeat them, which saves context. |
| `rows` | list of lists | One row per measured case; see "Per-case columns". |
| `skipped` | list of `[mat, F, reason]` | Cases not measured: `unsupported: rectangular`, `skip:oom-budget` (predicted memory > 24 GB budget), `unsupported in base` (the baseline lacks an API the candidate uses), and so on. Skips are **not** failures. |
| `raw` | path | Path of the full per-sample CSV on the local machine (synced back from the cluster). It holds every sample, arm, position and round, for anything the summary leaves out. |

### `noise`: how it is computed
For each case, the tool takes two quantities:
- **Same-process noise:** |median(`_dup`) / median(kernel) − 1|. Identical work, timed twice.
- **Cross-process drift:** |cuSPARSE median in the base process / cuSPARSE median in the cand process − 1|. cuSPARSE
  runs identical code in both processes, so any difference is drift between the processes.

`noise` is the **95th percentile, over all cases, of the larger of the two.** Past runs on this cluster put
identical-work spread at about 0.96–1.035, which is why the verdict threshold never drops below 4%.

## `sum`: aggregates

`sum.tune` and `sum.held` have the same shape. **tune** covers the matrices the loop is allowed to optimize
against (cant, web-Google, roadNet-CA, uniform16, blockdiag32). **held** covers the held-out matrices, which only
decide promotion (soc-Pokec, pdb1HYS, ASIC_680k, amazon0312, rail4284, pruned16x16). A change that improves `tune` but
not `held` is overfitting.

| Field | Meaning |
|---|---|
| `n` | Number of measured cases in the subset (skips and failures excluded). |
| `geo_x_base` | Geometric mean of `x_base` over those cases. **>1 means the candidate is faster than the baseline overall.** The geometric mean is used because speedups are ratios: a 2× win and a 2× loss cancel to 1. |
| `geo_x_cusp` | Geometric mean of `x_cusp`. **>1 means faster than cuSPARSE.** This is the project's real target. |
| `W` / `L` / `T` | Counts of win, loss and tie verdicts. |
| `sum.bw_pct_max` | Highest `gbps` over all cases, as % of the RTX 3090's 936 GB/s nominal peak. It feeds the CHANGELOG's "Peak Memory Bandwidth". |
| `sum.fail` | Number of `F` cases. Any value above 0 makes `ok` false. |
| `sum.skip` | `len(skipped)`. |

## Per-case columns (`cols` / `rows`)

| # | Column | Unit | Meaning and how to read it |
|---|---|---|---|
| 0 | `mat` | | Matrix id from the suite manifest (`cant`, `web-Google`, `uniform16`, …). |
| 1 | `F` | | Number of dense feature columns in X and Y (32, 64, 128, 256, 512; plus 33, 65, 129 when tail probes are on). |
| 2 | `k_ms` | ms | **Candidate median `cpp_only` time.** This is the number that matters. It pools all rounds. |
| 3 | `k_iqr` | % | (p75 − p25) / median of the candidate's `cpp_only` samples. A large value (> 5%) means unstable timing for this case, so trust its verdict less. |
| 4 | `call_ms` | ms | Candidate median `full_call` time. `call_ms − k_ms` ≈ Python/pybind/dispatch overhead. A kernel change normally doesn't move it, so if it does, look at the Python wrapper. |
| 5 | `base_ms` | ms | Baseline median `cpp_only` time, same job, same GPU. |
| 6 | `cusp_ms` | ms | cuSPARSE `DirectSpMM` median `cpp_only` time on the same matrix and X (native order). |
| 7 | `x_base` | ratio | `base_ms / k_ms`. **>1 = candidate faster than baseline.** 1.07 means 7% faster. |
| 8 | `x_cusp` | ratio | `cusp_ms / k_ms`. **>1 = candidate faster than cuSPARSE.** |
| 9 | `gbps` | GB/s | Effective bandwidth = minimum bytes the operation must move ÷ `k_ms`. Minimum bytes = `4·(2·nnz + m + 1)` (CSR values, column indices and row pointers) `+ 4·F·(x_rows + m)` (each X row that is actually referenced read once, Y written once). This is a lower bound on real traffic, so `gbps` near 936 means the kernel is close to the memory roofline. A low value on a large case means time is going to something other than compulsory traffic: re-reads, atomics, sync, or latency. |
| 10 | `mem_mib` | MiB | Peak GPU memory allocated during the case (`torch.cuda.max_memory_allocated` after a reset): the format, X, Y and workspaces. Watch it on large inputs against the 24 GB card. |
| 11 | `v` | | Verdict: see below. |

### Verdict `v`
With `thr = max(noise, 0.04)`:

| `v` | Rule | Meaning |
|---|---|---|
| `W` | `x_base > 1 + thr` | Real win over the baseline. |
| `L` | `x_base < 1 / (1 + thr)` | Real regression. |
| `T` | otherwise | Within noise: no evidence either way. |
| `F` | per-row bound violated | Wrong results. `x_base` and the rest are still filled from the validation call where possible, but the case is excluded from all aggregates and `ok` becomes `false`. |

**Correctness bound.** Every output element y_ij is compared with a float64 reference r_ij. It fails when
|y_ij − r_ij| > tol_i · (|A||X|)_ij, where tol_i = (2(k_i + 1) + 2) · u32, k_i is row i's NNZ, and u32 is float32 unit
roundoff. NaN always fails. This is the bound already used in `bench_real.Reference`. It scales with each row's length
and magnitude, so a long row isn't flagged for ordinary float32 rounding.

## Reading a result: quick guide
- **Promotion candidate:** `ok`, `tune.geo_x_base > 1.04`, `held.geo_x_base ≥ 1.0`, and no new `L` in `held`.
- **Win only on a few matrices:** look at which structural category (from `suite_manifest.json`) the `W` rows share.
  The result may belong behind a per-matrix heuristic rather than in the default.
- **`x_base` up but `x_cusp` still < 1:** progress, but cuSPARSE remains ahead. Report both.
- **High `k_iqr` or `noise` > 0.05:** rerun before drawing conclusions. The node may be shared or throttling.
