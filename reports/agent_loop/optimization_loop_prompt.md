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
  - Exit code 2 means bad arguments; 3 means infrastructure (SSH or Slurm); 4 means the tool's side
    effect (commit, push, promotion, job) already happened but its audit record could not be written.
    Never retry an exit-4 call.
    - With a recovery record (`audit_finalization: "pending_recovery"`, `recovery_record` set): continue;
      the next tool call reconciles the record.
    - Without one (`audit_finalization: "manual_recovery_required"`, `recovery_record: null`): do not
      continue the campaign. Stop and report to the user that the side effect succeeded, with the full
      returned result.
  - Calls block until their Slurm job ends: about 6 min for a build, 1 min for tests, 20–40 min for a
    benchmark. Run long calls with Bash `run_in_background` and wait for the completion notification.
- **Remote GPU cluster** (BGU Slurm, RTX 3090). Builds, tests, benchmarks and profiles run there, only
  through the tools. Never build CUDA locally. Never touch `/home/dayanb/research/SPMM` on the cluster;
  it is a shared checkout other sessions use.
- **SSH outages.** The cluster's SSH sometimes drops (exit 3, "unreachable"). Check with
  `nc -z -G 5 slurm.bgu.ac.il 22`, wait 5 minutes and retry. Notify the user (to turn on VPN). After 60 minutes of continuous outage, stop
  and write your report with what you have.
- **Commits:** never add `Co-Authored-By` lines to commit messages.

## 2. The tools

| Tool | Args | Use |
|---|---|---|
| `begin_attempt` | `{"title": "...", "hypothesis": "..."}` | Starts a tracked attempt in `.agent_loop/attempts.sqlite3`, verifies a clean starting point matching `opt/verified`, and enforces one active attempt at a time and the 5-attempt cap. Required before `commit_kernel_attempt`, `rollback_workspace`, `run_benchmarks`, or `promote_attempt`. |
| `commit_kernel_attempt` | `{"message": "kernel: ..."}` | Commits `torch_fast_spdmm/**`, top-level `tests/torch_fast_spdmm/test_*.py` (excluding `test_utils.py`), and `CHANGELOG.md`, then pushes. Returns `sha`. |
| `load_build_env` | `{"env_name": "fem", "modules": []}` | Builds HEAD (cached by source hash). Needs a clean, pushed tree. |
| `submit_slurm_job` | `{"script_content": "<text of optimizer/agent_tools/templates/tests.sh>", "name": "spmm-tests", "time": "01:00:00"}` | Runs the GPU test suite. Success is `tests_passed: true`. |
| `run_benchmarks` | `{"kernel_name": "boo:rowseg", "matrix_datasets": ["tune"]}` (or `["all"]`, see §3 step 6) | Paired timing of HEAD against `opt/verified` on the same GPU, plus cuSPARSE. Returns `bench.1`, and writes `bench.json` under `tests/torch_fast_spdmm/benchmark/results/agent_runs/<sha>/<job>/`. |
| `setup_controls` | `{"stage": "all"}` (or `"status"`) | Prepares or checks the isolated `suite/controls/` namespace (`fem2d`, `fem2d_permuted`, `diag_b1.0`). Separate from `setup_suite`. |
| `synthetic_controls` | `{}` | Optional, non-gating paired diagnostic benchmark (`fem2d`, `fem2d_permuted`, `diag_b1.0` at N=262144, F=128/512) using the same paired driver. Writes `synthetic_controls.json` (`gating: false`, `promotion_eligible: false`). |
| `submit_slurm_job` + `parse_ncu_profile` | see §4 | Nsight Compute profile. Returns `ncu.1`. |
| `promote_attempt` | `{"bench_json": "<path to the bench.json of an [\"all\"] run>"}` | Applies the fixed promotion gate (§6). If it passes, `opt/verified` advances to your commit. The file must be exactly as `run_benchmarks` wrote it. |
| `rollback_workspace` | `{}` | Adds a revert commit that restores `torch_fast_spdmm/**` and top-level `test_*.py` to `opt/verified`. Never touches `CHANGELOG.md` or the benchmark harness. |
| `finish_attempt` | `{"outcome": "...", "decision_reason": "...", "promoted": false, "rolled_back": true, "changelog_sha": "<sha>", ...}` | Validates consistency with `opt/verified`, workspace state, `CHANGELOG.md`, and benchmark artifacts, then closes the active attempt in `.agent_loop/attempts.sqlite3`. |
| `close_run` | `{"status": "completed", "reason": "..."}` | Closes the active campaign as `completed` or `aborted` (recording `ended_at` and preventing further attempts on that run). Call before `export_attempt_log`. |
| `get_loop_status` | `{}` | Reports the active/latest run and attempt, counted attempts remaining (campaign-wide: the 5-attempt budget is shared by every run in `attempts.sqlite3`), recent attempts, and any unfinished or interrupted tool calls. |
| `clean_attempt_workspace` | `{}` | Recovery with **no** active attempt: discards uncommitted edits to `torch_fast_spdmm/**` and top-level `test_*.py` only. Never commits or pushes; refuses committed drift (stop and report). |
| `export_attempt_log` | `{}` | Exports deterministic JSON and Markdown summaries (including the §8 final-report table and closed run status/end time) to `reports/agent_loop/runs/<run-id>/`. |

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

1. **Hypothesis & `begin_attempt`.** Write one sentence: "Changing X should reduce Y (metric) because Z (evidence)."
   Base it on profile data (§4, §5), not on taste. Call `begin_attempt` with `{"title": "<short title>", "hypothesis": "<that sentence>"}` before editing code.
   - Only one attempt may be active at a time. Compile fixes stay inside the same active attempt.
   - Check `get_loop_status` when resuming after an interruption to inspect remaining attempts and any interrupted tool calls.
2. **Edit.** Change the smallest code region that tests the hypothesis. The kernel is in
   `torch_fast_spdmm/kernels/boo_kernel.cu`; its launcher is in the same file, with declarations in
   `boo_kernels.hpp` and the Python dispatch in `torch_fast_spdmm/torch_fast_spdmm/boo.py`.
   - If you change behaviour, add or extend a test in `tests/torch_fast_spdmm/` (for example
     `test_boo.py`).
   - Never weaken or delete an existing test. `promote_attempt` refuses a candidate that deletes
     lines from an existing `tests/torch_fast_spdmm/test_*.py`.
   - Do not edit the measurement harness: `optimizer/`, `tests/torch_fast_spdmm/benchmark/`,
     `conftest.py`, `test_utils.py`. A candidate that changes any of them cannot be promoted.
3. **Commit.** Call `commit_kernel_attempt` with message `kernel: <hypothesis title>`.
4. **Build.** Call `load_build_env`. If it fails, read `err_tail`, fix it and commit again. A compile
   fix does not count as a new attempt; at most 3 fixes per attempt.
5. **Test.** Call `submit_slurm_job` with `templates/tests.sh`.
   - A failure in a test related to your change is a correctness failure. Either fix it (that counts as
     the same attempt) or go to step 8.
6. **Benchmark, in two stages.**
   - **6a.** Call `run_benchmarks` with `boo:rowseg` on `["tune"]`. Any `F`, or `ok=false` (including
     `"failure": "candidate_runtime"`, a crash of the candidate process), means a correctness failure: go
     to step 8. Do not re-run it.
   - **6b.** Only if tune `geo_x_base` ≥ 1.02 with W > L, call it again on `["all"]` (default features,
     reps and rounds). That run is the one `promote_attempt` judges.
   - Use `held` numbers only for the promotion decision, never to choose or tune a hypothesis;
     otherwise `held` stops being held out.
   - If a tool call fails with exit 3 mid-job, call it again with the same arguments: it reattaches to
     the running or finished Slurm job instead of submitting a duplicate. If that job genuinely reached a
     terminal failure (`FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, etc.), reattaching reports the failure and
     clears its pending record rather than silently resubmitting it.
7. **Profile / synthetic controls** (optional when the result is clearly flat): see §4. Profile one or two cases where the
   change should matter, and pass the baseline report so you get the `d` (delta) block.
   - **`synthetic_controls` (auxiliary, non-gating):** normally invoked only after the tune gate passes,
     and only when its result can distinguish mechanisms relevant to the hypothesis (not required for every
     attempt). It may run earlier only if the pre-registered hypothesis names a specific diagnostic
     question that requires it.
   - Builder and Auditor may inspect `synthetic_controls.json` because the controls (`fem2d`,
     `fem2d_permuted`, `diag_b1.0` at N=262144, F=128/512) are not held-out.
   - Controls results never enter `tune`/`held`/`all` aggregates, never influence `promote_attempt`, and
     never compensate for real-matrix regressions. A controls failure is not a promotion failure unless an
     independent promotion check fails (though a candidate correctness failure or crash on controls is
     recorded as a candidate correctness failure and not retried as infrastructure).
   - `setup_controls` (`suite/controls/`) and `setup_suite` (`suite/`) remain completely separate.
8. **Decide.** Run the steps of the matching row in order:

   | Outcome | Action |
   |---|---|
   | Correct and faster (passed 6b) | 1. Call `promote_attempt` with the 6b `bench.json`. If it returns `promoted: true`, the commit becomes the new baseline. 2. If it returns `promoted: false`, treat the outcome as "not faster". |
   | Correct but not faster (or not promoted) | 1. Call `rollback_workspace`. |
   | Build, test or correctness failure | 1. Call `rollback_workspace`. |

9. **CHANGELOG.** Add the entry at the top of `CHANGELOG.md` at the repo root (create the file if
   absent), following the template exactly.
   - Outcome is `PASS`, `FAIL (<build|tests|correctness>)` or `ROLLED BACK`.
   - Copy the numbers from `bench.1.sum` and the paths from `raw` and `rep`.
   - Commit it with `commit_kernel_attempt` and message `docs: changelog for <title>`.
   - The source tree is unchanged by this commit, so the next build stays cached.
10. **Close attempt (`finish_attempt`).** Only after step 8 (promotion or rollback) and step 9 (committed
    `CHANGELOG.md`), call `finish_attempt` with the matching `outcome`, `decision_reason`, `promoted` /
    `rolled_back`, `changelog_sha` (the CHANGELOG commit made by `commit_kernel_attempt` in this
    attempt), and any `tune_bench_json` / `bench_json` / `profile_paths`.
    - `.agent_loop/attempts.sqlite3` is the machine-readable source of truth and `CHANGELOG.md` is the
      human-readable record; they must agree. `finish_attempt` refuses inconsistent completion data.
11. Use what you learned to choose the next hypothesis. A rolled-back attempt still tells you something:
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
- **Barrier cost** (21%). The trailing `__syncthreads()` at the end of each feature tile in
  `boo_spmm_rowseg_kernel` only protects `smem_X` from the next tile's Step A. After the *final* tile
  nothing overwrites `smem_X`, so that barrier looks removable: guard it to non-final tiles. That halves
  the per-tile barriers for F ≤ P without changing arithmetic. Check nothing after the loop reads
  shared memory first. A larger version: restructure so only the warps that stage X wait.
- **LOAD phase overhead** for small blocks. Many blocks hold few nonzeros, so per-CTA setup (2 barriers
  plus binary search) can dominate. Check the manifest's `useful_block_occupancy`. The manifest has no
  per-block size histogram; if you need one, compute it from `csr_to_boo` output inside a profile job.
- **Wasted lanes.** Rows with no nonzero in a block still get threads. Look at `warp.lanes` and
  `branch_eff`. Permanent per-block active-row compaction (done once at format build time) removes that
  work. It differs from the failed row-start execution plan, which only replaced the binary search.
  roadNet-CA and amazon0312 have very low useful occupancy, so they are where it should show.
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
They appear in `skipped`; ignore them. They are the only approved skips (`state.approved_skips`).

**Promotion gate** (enforced by `promote_attempt`; all must hold):
- tune `geo_x_base` ≥ 1.02 with more wins than losses;
- held `geo_x_base` ≥ 0.99 with losses ≤ wins, so a tune win cannot hide a held regression;
- the run covers every tune and held matrix at F = 32/64/128/256/512 with reps ≥ 30 and rounds ≥ 3,
  apart from the approved skips;
- tests passed for the candidate, the baseline is the current `opt/verified`, the candidate is HEAD;
- the harness is unchanged and no existing test lost lines.

Structure statistics for every matrix are in the cluster's `/home/dayanb/research/spmm-agent/suite/manifest.json`.
- Read them with `ssh bgu-slurm cat ...`; this read is allowed.
- They include row-NNZ statistics, BOO useful occupancy, remainder fraction and groups touching each row.
  They do not include per-block size distributions.

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

Before writing your final message:
1. Call `close_run` with `{"status": "completed", "reason": "<brief completion summary>"}` (or `"aborted"` if
   stopping due to an unrecovered outage/early-abort).
2. Call `export_attempt_log` to write the stable JSON and Markdown summaries (including the closed status and
   end time) under `reports/agent_loop/runs/<run-id>/`, and verify that your report table agrees with it.

1. **A table of the attempts:**

   | # | Title | sha | Outcome | tune `geo_x_base` (W/L/T) | held `geo_x_base` (W/L/T) | `geo_x_cusp` tune / held | Promoted? |
   |---|---|---|---|---|---|---|---|

2. **The final `opt/verified` sha**, and whether it moved.
3. **For each attempt,** 2–3 sentences: what the profile showed before and after, and why it worked or
   didn't.
4. **What you would try next,** and why.
5. **Anything odd about the tools or the cluster.** Do not work around a tool bug by editing
   `optimizer/agent_tools/`; report it instead.
