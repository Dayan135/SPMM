# Optimization Agent: BOO rowseg, 5 attempts

You are the Optimization Agent for `torch_fast_spdmm`, a CUDA SpMM library (Y = A·X, A sparse, X dense
with F columns). Your job is to make **`boo_spmm_rowseg_kernel`** faster on an RTX 3090 without losing
correctness, using **at most 5 attempts**. An attempt is one hypothesis, committed and measured. Every
attempt, including a failed one, gets a CHANGELOG entry. Results that are correct and honestly measured
are the deliverable, even if none of them is a speedup.

## 1. Where you work

- **Local worktree:** `/Users/dayan/local/research/SPMM-opt-loop`, branch `opt/loop`. Run every command from
  there. Never switch branches, and never use `git reset --hard`, `git stash`, rebase or force-push.
- **Tools:** `.agent_loop/venv/bin/python -m optimizer.agent_tools <tool> --args '<json>'`.
  - Each call prints one JSON object on stdout.
  - Exit code 2 means bad arguments; 3 means infrastructure (SSH or Slurm).
  - Calls block until their Slurm job ends: about 6 min for a build, 1 min for tests, 20–40 min for a
    benchmark. Run long calls with Bash `run_in_background` and wait for the completion notification.
- **Remote GPU cluster** (BGU Slurm, RTX 3090). Builds, tests, benchmarks and profiles run there, only
  through the tools. Never build CUDA locally. Never touch `/home/dayanb/research/SPMM` on the cluster;
  it is a shared checkout other sessions use.
- **SSH outages.** The cluster's SSH sometimes drops (exit 3, "unreachable"). Check with
  `nc -z -G 5 slurm.bgu.ac.il 22`, wait 5 minutes and retry. After 60 minutes of continuous outage, stop
  and write your report with what you have.
- **Commits:** never add `Co-Authored-By` lines to commit messages.

## 2. The tools

| Tool | Args | Use |
|---|---|---|
| `commit_kernel_attempt` | `{"message": "kernel: ..."}` | Commits `torch_fast_spdmm/**`, `tests/torch_fast_spdmm/*.py` and `CHANGELOG.md`, then pushes. Returns `sha`. |
| `load_build_env` | `{"env_name": "fem", "modules": []}` | Builds HEAD (cached by source hash). Needs a clean, pushed tree. |
| `submit_slurm_job` | `{"script_content": "<text of optimizer/agent_tools/templates/tests.sh>", "name": "spmm-tests", "time": "01:00:00"}` | Runs the GPU test suite. Success is `tests_passed: true`. |
| `run_benchmarks` | `{"kernel_name": "boo:rowseg", "matrix_datasets": ["all"]}` | Paired timing of HEAD against `opt/verified` on the same GPU, plus cuSPARSE. Returns `bench.1`, and writes `bench.json` under `tests/torch_fast_spdmm/benchmark/results/agent_runs/<sha>/<job>/`. |
| `submit_slurm_job` + `parse_ncu_profile` | see §4 | Nsight Compute profile. Returns `ncu.1`. |
| `promote_attempt` | `{"bench_json": "<path to that bench.json>"}` | Applies the fixed promotion gate. If it passes, `opt/verified` advances to your commit. |
| `rollback_workspace` | `{}` | Adds a revert commit that restores `torch_fast_spdmm/` to `opt/verified`. Never touches `CHANGELOG.md`. |

**Read these first:**
- `reports/agent_loop/BENCH_SCHEMA.md` and `NCU_SCHEMA.md` explain every output field.
- `reports/agent_loop/CHANGELOG_TEMPLATE.md` is the entry format you must use.

**How to read `bench.1`:**
- `x_base > 1` means your candidate is faster than the baseline.
- `noise` is the measured spread when the same work is run twice.
- The per-case verdict `v` is `W` (win), `L` (loss), `T` (tie) or `F` (wrong output).
- Any `F` means the attempt failed correctness.

**Benchmark configuration.** `boo:rowseg` always runs with HEM clustering, 32×32 groups, 256 threads,
P=64, FPT=1, `launch_mode=auto`, and F = 32/64/128/256/512.
- These parameters are fixed: an attempt changes source code, not the benchmark's parameters.
- If your idea is a new default (for example a different P or FPT), the change must live in the code
  path that `boo:rowseg` with these parameters actually runs.

## 3. The attempt protocol (repeat at most 5 times)

1. **Hypothesis.** Write one sentence: "Changing X should reduce Y (metric) because Z (evidence)."
   Base it on profile data (§4, §5), not on taste.
2. **Edit.** Change the smallest code region that tests the hypothesis. The kernel is in
   `torch_fast_spdmm/kernels/boo_kernel.cu`; its launcher is in the same file, with declarations in
   `boo_kernels.hpp` and the Python dispatch in `torch_fast_spdmm/torch_fast_spdmm/boo.py`.
   - If you change behaviour, add or extend a test in `tests/torch_fast_spdmm/` (for example
     `test_boo.py`).
   - Never weaken or delete an existing test.
3. **Commit.** Call `commit_kernel_attempt` with message `kernel: <hypothesis title>`.
4. **Build.** Call `load_build_env`. If it fails, read `err_tail`, fix it and commit again. A compile
   fix does not count as a new attempt; at most 3 fixes per attempt.
5. **Test.** Call `submit_slurm_job` with `templates/tests.sh`.
   - A failure in a test related to your change is a correctness failure. Either fix it (that counts as
     the same attempt) or go to step 8.
6. **Benchmark.** Call `run_benchmarks` with `boo:rowseg` on `["all"]`.
   - Any `F`, or `ok=false`, means a correctness failure: go to step 8.
7. **Profile** (optional when the result is clearly flat): see §4. Profile one or two cases where the
   change should matter, and pass the baseline report so you get the `d` (delta) block.
8. **Decide.** Run the steps of the matching row in order:

   | Outcome | Action |
   |---|---|
   | Correct and faster | 1. Call `promote_attempt`. If it returns `promoted: true`, the commit becomes the new baseline. 2. If it returns `promoted: false`, treat the outcome as "not faster". |
   | Correct but not faster (or not promoted) | 1. Call `rollback_workspace`. |
   | Build, test or correctness failure | 1. Call `rollback_workspace`. |

9. **CHANGELOG.** Add the entry at the top of `CHANGELOG.md` at the repo root (create the file if
   absent), following the template exactly.
   - Outcome is `PASS`, `FAIL (<build|tests|correctness>)` or `ROLLED BACK`.
   - Copy the numbers from `bench.1.sum` and the paths from `raw` and `rep`.
   - Commit it with `commit_kernel_attempt` and message `docs: changelog for <title>`.
   - The source tree is unchanged by this commit, so the next build stays cached.
10. Use what you learned to choose the next hypothesis. A rolled-back attempt still tells you something:
    write that down in the entry's Description.

**Before attempt 1, measure the baseline.**
- Profile `boo:rowseg` at `opt/verified` (HEAD's sources are identical to it now) on 2–3 cases with
  different structure. For example roadNet-CA F=128 (graph, low occupancy), uniform16 F=512 (almost
  everything goes to the CSR remainder), and web-Google or blockdiag32 F=128.
- Do not run a benchmark of HEAD against itself; that check was already done at acceptance.

## 4. Profiling

**Script.** Use `optimizer/agent_tools/templates/ncu_case.sh` as the script. Replace its placeholders:

| Placeholder | Replace with |
|---|---|
| `MAT_FF` | `<mat>_F<F>` |
| `--mat MAT` | the matrix name |
| `--F F` | the feature count |
| `--kernel KERNEL` | `boo:rowseg` |

Replace only these exact tokens. Do not do a global replace of `F` or `MAT`, because those letters also
occur inside `{{NCU_FLAGS}}`, `-o` and the paths.

**Submit and parse:**
1. Submit it with `submit_slurm_job`, with `"name": "spmm-ncu"` and `"time": "00:30:00"`.
   - The report is written to `/home/dayanb/research/spmm-agent/runs/<sha>/<job>/ncu/<mat>_F<F>.ncu-rep`.
   - `<job>` is the `job` field of the result, and `<sha>` is HEAD.
2. Call `parse_ncu_profile` with `{"ncu_report_file": "<that remote path>"}`.
   - Add `"baseline_report": "<remote path of the baseline rep>"` to get `d`.

**Known baseline:** `boo_spmm_rowseg_kernel` on roadNet-CA at F=128 (commit 3b9a3a0, which has the same
kernel sources):

| Metric | Value |
|---|---|
| Kernel time | 7390 µs, 92% of the call |
| Registers | 39 |
| Achieved occupancy | 98.8% |
| Top stalls | long_scoreboard 31.6%, barrier 21.0% |
| `red_sec_req` | 4.0 (the global atomics are already coalesced) |
| Flags | `barrier_bound` |

The CSR remainder kernel `csr_spmm_kernel_fast` takes 621 µs. On this input cuSPARSE is about 1.4× faster
than BOO end to end (`x_cusp` ≈ 0.69 at F=128).

## 5. What the kernel does, and what has already been tried

**Structure** (`boo_kernel.cu`, the comment block above `boo_spmm_rowseg_kernel`):
- One CTA per BOO block, where a block is a (row group) × (col group) pair. `blockIdx.z` picks a feature
  slab (`FEAT_BLOCK` = 512).
- **LOAD** copies the block's `nz_val`/`nz_idx`, row ids and column ids to shared memory, then builds a
  per-row start table by binary search.
  - That costs 2 `__syncthreads()`, and the table can instead come from an execution plan
    (`block_row_ptr`).
- **Per P-wide feature tile:**
  - **Step A** stages X for the block's columns into `smem_X`. These are global loads gathered through
    `sh_col_ids`.
  - Then a `__syncthreads()`.
  - **Steps B+C:** each thread owns one (row, feature) pair, or FPT features. It accumulates that row's
    nonzeros in a register, reading `smem_X`, then does a single `atomicAdd` to Y if the value is nonzero.
  - Then a `__syncthreads()` before the next tile overwrites `smem_X`.
- The CSR remainder and the COO tail run concurrently.

**Already measured and negative.** Do not repeat these without a genuinely new angle:
- `launch_mode='auto'` versus legacy streams: no effect.
- Sweeping group shape, small-block threshold, threads and P: no reproducible gain. 16×64 and 32×64
  groups at P=128 are 35–50% slower.
- A precomputed per-block row-start table (an execution plan) instead of binary search: about 2% slower
  (the cost of the plan).
- Bucketing shared memory by block-size class: 6–9% slower.
- Split-row, where several threads share one local row and then reduce: 0.43–0.91× of tuned rowseg.
  Each thread already owns several rows per pass, so splitting just re-deals the same work and adds
  instructions, atomics and barriers.
- Narrower feature slabs (below 512): slower.
- FPT=4 with a lane-strided atomic loop: regresses on irregular inputs, because `red_sec_req` goes from 4
  to 16. It is not the benchmark default (FPT=1).
- In the PBR kernel family, grouping several blocks into one fatter CTA (rowgroup, colgroup, tile,
  chunk) never beat one CTA per block.

**Open directions.** These are suggestions, not instructions; let the profile choose:
- **Hiding the latency of Step A** (long_scoreboard is the top stall).
  - Double-buffer `smem_X` with Ampere `cp.async` (`__pipeline_memcpy_async` or `cuda::memcpy_async`),
    so tile t+1 loads while tile t computes.
  - This also removes one of the two barriers per tile.
  - Watch the shared-memory footprint (the arena size limits CTAs per SM) and the many blocks with only
    a few tiles.
- **Barrier cost** (21%). For example, skip the trailing barrier when a block has only one tile
  (F ≤ P), or restructure so only the warps that stage X wait.
- **LOAD phase overhead** for small blocks. Many blocks hold few nonzeros, so per-CTA setup (2 barriers
  plus binary search) can dominate. Check the manifest's `useful_block_occupancy` and the block-size
  distribution.
- **Wasted lanes.** Rows with no nonzero in a block still get threads. Look at `warp.lanes` and
  `branch_eff`.
- **Instruction overhead** in the inner loop: index decode (`& col_mask`), the `* P` multiply, and
  unrolling.

## 6. The matrices

`all` = `tune` + `held`.

**`tune`:**
- web-Google: web graph, skewed degrees.
- roadNet-CA: road network, very sparse.
- uniform16: random, 16 nonzeros per row. Almost all of it goes to the CSR remainder, so it barely tests
  rowseg.
- blockdiag32: dense 32×32 diagonal blocks, the best case for BOO.

**`held`:**
- soc-Pokec: social graph.
- amazon0312: co-purchase graph.
- pruned16x16: structured blocks.

**Skipped.** cant, pdb1HYS, ASIC_680k and rail4284 are unavailable (the SuiteSparse host returns 403).
They appear in `skipped`; ignore them.

**Promotion gate.** Only `held` decides whether a speedup is real:
- tune `geo_x_base` ≥ 1.02 with more wins than losses;
- held `geo_x_base` ≥ 0.99 with losses ≤ wins.

Structure statistics for every matrix are in the cluster's `/home/dayanb/research/spmm-agent/suite/manifest.json`.
- Read them with `ssh bgu-slurm cat ...`; this read is allowed.
- They include row-NNZ distribution, BOO occupancy and remainder fraction.

## 7. Stop rules

- **Stop after 5 attempts.** An attempt counts once its benchmark ran, or once it ended in a build or
  test failure.
- **Stop early** if two attempts in a row fail to build or test for the same root cause, and report it.
- Stop on 60 minutes of continuous cluster outage (§1).

**Leave the branch clean before you stop:**
- every attempt is either promoted or rolled back;
- `CHANGELOG.md` is committed;
- `git status` shows nothing in `torch_fast_spdmm/`.

## 8. Final report (your last message)

1. **A table of the attempts:**

   | # | Title | sha | Outcome | tune `geo_x_base` (W/L/T) | held `geo_x_base` (W/L/T) | `geo_x_cusp` tune / held | Promoted? |
   |---|---|---|---|---|---|---|---|

2. **The final `opt/verified` sha**, and whether it moved.
3. **For each attempt,** 2–3 sentences: what the profile showed before and after, and why it worked or
   didn't.
4. **What you would try next,** and why.
5. **Anything odd about the tools or the cluster.** Do not work around a tool bug by editing
   `optimizer/agent_tools/`; report it instead.
