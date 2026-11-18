# Tool-Builder prompt: SpMM agent-loop tools

> **You are the Tool-Builder Agent.** You are a DevOps/MLOps engineer. You implement six command-line tools that a
> separate *Optimization Agent* will use to edit, build, test, benchmark and profile custom CUDA SpMM kernels on a
> Slurm GPU cluster. You do **not** optimize kernels yourself and you do **not** change any `.cu`/`.cpp` file, except the narrow timing-binding exception in §8. Read
> this whole document before writing code. Everything you need is here or in the files it names.

---

## 0. Goal and definition of done

Build `optimizer/agent_tools/`, a Python package invoked as

```bash
python -m optimizer.agent_tools <tool> --args '<json object>'
```

It exposes exactly six tools: `load_build_env`, `submit_slurm_job`, `commit_kernel_attempt`, `rollback_workspace`,
`run_benchmarks` and `parse_ncu_profile`. There is also one setup-only command, `setup_suite` (§5), which the loop does
not use.

Every invocation prints **exactly one JSON object on stdout** and nothing else. Logs, progress and ssh noise go to
stderr. Exit codes:
- `0` means the tool ran and its JSON describes the outcome, including a failed build or failed tests.
- `2` means bad arguments.
- `3` means an infrastructure error (ssh unreachable after retries, Slurm rejected the job). Codes 2 and 3 still print
  `{"tool": "...", "error": "...", "detail": "..."}`.

`run_benchmarks` must print an object that validates against `bench.v1.json`, and `parse_ncu_profile` one that
validates against `ncu.v1.json` (Appendix A). Validate before printing. A schema violation is a bug in your code,
never something to pass along.

You are done when every acceptance check in §9 passes, and you have reported back as described in §11.

---

## 1. Project and cluster facts (non-negotiable)

**Repository.** The local repo is `/Users/dayan/local/research/SPMM-boo` (a git worktree; macOS, no GPU). Read its
`CLAUDE.md` first. The package is `torch_fast_spdmm`: custom CUDA SpMM for two sparse formats, **PBR** (Packed Block
Representation) and **BOO** (clustered blocks). There is also a naive CSR kernel and a direct cuSPARSE wrapper.

**Nothing CUDA runs locally.** Every build, test, benchmark and profile runs on the BGU Slurm cluster:

| Item | Value |
|---|---|
| SSH alias | `bgu-slurm` (env override `OPTIMIZER_REMOTE_HOST`). It can intermittently time out depending on the user's VPN. Retry **ssh-level** failures (exit 255) 3× with backoff, then exit 3. Never retry a job that ran and failed. |
| Shared remote clone | `/home/dayanb/research/SPMM` (env `OPTIMIZER_REMOTE_DIR`). **Other sessions use it. Never `checkout`, `reset` or `pull` in it.** Only `git fetch` there, then extract snapshots with `git archive`. |
| Your remote root | `/home/dayanb/research/spmm-agent/` (create it). |
| Slurm | `--partition=rtx3090 --account=erant --qos=erant --gres=gpu:rtx_3090:1`. Header template: `SBATCH_HEADER` in `optimizer/remote.py`. |
| Excluded nodes | `--exclude=ise-3090-01,ise-3090-04` on **every** GPU job, read from state (`exclude_nodes`). `ise-3090-04` (Ivy Bridge, no AVX2) produced a different `torch.randn` X for the same seed. `ise-3090-01` returned wrong outputs (suspected bad GPU memory). |
| Conda env | `fem`. It contains Python, PyTorch, **nvcc 12.9** and Nsight Compute. **There is no `/usr/local/cuda` and no CUDA `module`.** |
| Env preamble | `BUILD_ENV_PREAMBLE` in `optimizer/remote.py`, plus `export TORCH_CUDA_ARCH_LIST=8.6 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1`. `CUDA_HOME="$CONDA_PREFIX"` must be set **after** `conda activate fem`. `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:..."` is needed at **import time too**; without it you get `CXXABI_1.3.15 not found`. |
| ncu binary | `$CONDA_PREFIX/nsight-compute-2025.2.1/ncu` |
| Matrix cache | `SPMM_MATRIX_CACHE=/home/dayanb/research/real-matrices-20261008/matrix_cache` |
| GPU | RTX 3090 (sm_86, 82 SMs, 24 GB, 936 GB/s nominal) |
| Package shadowing | The repo root contains a non-package directory `torch_fast_spdmm/` that shadows the installed package. **Never run Python with cwd = a repo root.** Existing benchmarks run as `python -m benchmark.<x>` from `tests/torch_fast_spdmm/`. |

**Proven reference job.** Read `experiments/2026-11-11_phase1/phase1_job.sh` end to end. It builds several snapshots in
one job, switches between them with a `use()` function that sets `PYTHONPATH`, checks `.so` sha256 hashes, runs pytest,
times the snapshots in rotating order, and profiles with `ncu --set full` then `--import --page raw --csv`. Your tools
generalize that script; copy its conventions.

---

## 2. Existing code to reuse (do not rebuild these)

| Need | Reuse | Notes |
|---|---|---|
| ssh, scp, rsync, sbatch, waiting for jobs | `optimizer/remote.py`: `ssh`, `push_file`, `pull_file`, `pull_dir`, `render_and_submit`, `job_is_finished`, `wait_for_job`, `read_remote_text`, `SBATCH_HEADER`, `BUILD_ENV_PREAMBLE` | **Do not use `sync_and_build`.** It checks out the shared clone. `render_and_submit` takes header fields; add `--exclude`. Use `sacct -j <id> -n -P -o State,ExitCode,NodeList,Elapsed` after the job leaves `squeue`. |
| Timing loop | `tests/torch_fast_spdmm/benchmark/bench_nnzflat.py`: `Arm`, `measure()`, `event_timer`, `GpuReference` | `measure` validates each arm, warms up, shuffles arm order on every repeat, and returns median/p25/p75/MAD. Use it unchanged. |
| C++-scope timers | `core.PBRTimedSpMM`, `core.BOOTimedSpMM`, `core.DirectSpMM` (bindings in `torch_fast_spdmm/bindings/*_timed.cu`, `bindings_cusparse.cu`) | `timed_run()` brackets Y memset, kernels, remainder and join in C++. See how `bench_real.py` / `bench_tiles.py` construct them, and `bench_nnzflat.baseline_arms` for `DirectSpMM`. |
| Correctness bound | `bench_real.Reference` (CPU float64, chunked over features) and `bench_nnzflat.GpuReference.check` | Bound: \|y−r\| ≤ (2(k_i+1)+2)·u32·(\|A\|\|X\|)_i per row; NaN fails. |
| Real-matrix download and canonicalization | `tests/torch_fast_spdmm/benchmark/real_matrices.py`: `MatrixSpec`, `download` (login node, network), `prepare` (CPU node), `matrix_hash`, `load`, `digest_arrays`, cache layout `<group>/<name>/{archive, .mtx, source.json, canonical_<keep\|drop>.npz}` | It already holds roadNet-CA. Add the 7 new SuiteSparse matrices as `MatrixSpec`s. |
| NCU raw-page parsing | `scripts/research_panel/nnzflat_profile_report.py` (`COUNTERS`, `STALLS`, `STALL_FMT`, `rows_of`, `num`) and `shape_profile_report.py` (`kernels(raw, label)`) | Raw CSV = header row + units row + one row per kernel launch. `num()` handles mixed unit prefixes. |
| Single-case profiling driver | `scripts/research_panel/shape_profile.py` | Pattern for `cudaProfilerStart/Stop` around the profiled call when running `ncu --profile-from-start off`. |
| Public API | `torch_fast_spdmm/torch_fast_spdmm/__init__.py` (`csr_to_pbr`, `pbr_matmul(..., kernel=block\|chunkcols\|nnzflat\|nnzflat_shared\|nnzflat_onthefly)`) and `boo.py` (`csr_to_boo(mat, cluster='hem', target_nr, target_nc, ...)`, `boo_matmul(boo, x, threads, p, csr_mode, row_seg, launch_mode, features_per_thread, nnzflat_impl, ...)`) | Read the signatures yourself. Do not trust this summary over the code. |
| Commit/rollback pattern | `.claude/skills/kernel-quick-edit/SKILL.md` | Isolated branch, commit, push, compare. |

Known bug, not yours to fix, but don't call it: `optimizer/orchestrator.py:58` passes `kernels=` to `enumerate_grid`,
which doesn't accept it.

---

## 3. Layout you create

```
optimizer/agent_tools/
  __main__.py          # argparse: <tool> --args JSON; dispatch; JSON-only stdout; exit codes
  state.py             # .agent_loop/state.json read/write (file lock)
  slurm.py             # submit_slurm_job + script rendering (+ dry_run)
  build.py             # load_build_env + build cache
  vcs.py               # commit_kernel_attempt, rollback_workspace, verified pointer
  bench.py             # run_benchmarks: job render, rsync back, aggregate -> bench.1
  ncu.py               # parse_ncu_profile -> ncu.1 (+ NCU_FLAGS constant)
  suite.py             # setup_suite (one-time ingestion orchestration)
  schemas/bench.v1.json, schemas/ncu.v1.json   # byte-identical copies of reports/agent_loop/schemas/
  templates/build.sh, tests.sh, bench.sh, ncu_case.sh
tests/torch_fast_spdmm/benchmark/
  universal_suite.py   # roster, generators, normalization, manifest + structural stats, X generation, reference cache
  agent_bench.py       # cluster-side: one process = one build; runs cases via bench_nnzflat.measure; writes samples CSV
  agent_profile.py     # cluster-side: one case, warm-up, cudaProfilerStart/Stop around ONE call, writes <stem>.case.json
tests/agent_tools/     # offline unit tests (no GPU, no ssh)
.agent_loop/           # local state; add to .gitignore
```

**Remote layout**, under `/home/dayanb/research/spmm-agent/`:

```
builds/<src_hash>/torch_fast_spdmm/      # git archive of the torch_fast_spdmm/ subtree at some sha, built in place
builds/<src_hash>/BUILD_OK.json          # {sha, src_hash, so_sha256, nvcc, torch, build_s}
src/<sha>/                               # git archive of <sha> EXCLUDING torch_fast_spdmm/ and benchmark results
runs/<sha>/<job_id>/                     # job outputs (samples CSV, ncu reps, logs)
suite/manifest.json, suite/generated/*.npz, suite/refs/<mat>_F<F>.{ref,absref,tol}.npy + .json
```

**Snapshot isolation rule.** A process that tests build `B` with driver sources from candidate sha `S` runs:

```bash
cd src/S/tests/torch_fast_spdmm
PYTHONPATH="builds/B/torch_fast_spdmm:src/S/tests/torch_fast_spdmm:src/S"
```

- The **compiled extension and the Python package** come from the build. **Drivers and tests always come from the
  candidate sha**, so the baseline build can be older than `agent_bench.py`.
- Because `src/S` excludes `torch_fast_spdmm/`, shadowing is impossible.
- Every process must log `torch_fast_spdmm.__file__` and `torch_fast_spdmm_core.__file__` and **assert both are
  under `builds/B`**. Record them in the samples CSV.
- Generate a `use <build_dir> <src_dir>` shell function into the preamble that does exactly this.

**`src_hash`** = first 12 hex chars of `git rev-parse <sha>:torch_fast_spdmm`, the git tree id of the whole package
subtree (kernels, bindings, pbr_matrix, python package, setup.py). Commits that only touch tests, docs or `CHANGELOG.md`
reuse the build.

**Local state**, `.agent_loop/state.json`:

```json
{"remote_host": "bgu-slurm", "remote_repo": "/home/dayanb/research/SPMM",
 "agent_root": "/home/dayanb/research/spmm-agent", "env_name": "fem", "modules": [],
 "preamble": "<rendered text>", "exclude_nodes": ["ise-3090-01", "ise-3090-04"],
 "loop_branch": "opt/loop-YYYYMMDD", "verified": "<sha>",
 "builds": {"<src_hash>": {"sha": "...", "so_sha256": "..."}},
 "tests_passed": {"<sha>": true}}
```

---

## 4. Tool contracts

Arguments are a JSON object. Names in **bold** are the required signature; everything else is optional with the
default shown.

### 4.1 `load_build_env` — **`env_name: str`, `modules: list[str]`**

Purpose: (a) make the env usable by every later job, and (b) make sure the **current HEAD** has a compiled build.

1. Run `ssh bgu-slurm 'bash -lc "conda activate <env_name> && which nvcc && nvcc --version | tail -1 && python -c \"import torch;print(torch.__version__)\""'`.
   Fail if nvcc or torch is missing.
2. For each module, check that `module load <m>` succeeds. **A named module that doesn't exist is a hard error.** The
   default `[]` is correct for this cluster. Do not invent `cuda/12.x`.
3. Render the preamble (env lines, the `use` function, the module loads) and store it in state. It is a script
   fragment that every later job starts with; activation cannot persist across tool calls, because each call is a
   fresh session.
4. Build HEAD:
   - **Refuse** if HEAD isn't pushed to `origin`, or if `torch_fast_spdmm/` has uncommitted changes. Return
     `ok:false` with the reason.
   - Compute `src_hash`. If `builds/<src_hash>/BUILD_OK.json` exists remotely, return it with `cached:true`.
   - Otherwise run `git -C <remote_repo> fetch origin`, then `git archive <sha> torch_fast_spdmm | tar -x -C builds/<src_hash>`.
   - Submit `templates/build.sh` on a GPU node: `cd builds/<src_hash>/torch_fast_spdmm && python setup.py build_ext --inplace`, then sha256 the `.so`, import-smoke it under `use`, and write `BUILD_OK.json`.
   - The build has a 45-min limit. On failure, delete the half-built dir so the next call retries.
5. Also make sure `src/<sha>` exists (via `git archive` with `:(exclude)torch_fast_spdmm` and
   `:(exclude)tests/torch_fast_spdmm/benchmark/results`). Verify that the exclude pathspec works with the cluster's
   git; fall back to extract-then-delete if it doesn't.

Returns:
```json
{"ok": true, "cached": false, "sha": "a1b2c3d", "src_hash": "9f3c0e1a2b4d", "build_dir": ".../builds/9f3c0e1a2b4d",
 "so_sha256": "...", "nvcc": "12.9.86", "torch": "2.x", "build_s": 212.4, "warnings": 3, "err_tail": null}
```
- `warnings` is the count of nvcc `warning` lines.
- `err_tail` is the last 40 lines of the build log on failure, else null.

Also export `ensure_build(sha) -> dict` for `run_benchmarks` to reuse. It must work for **any** pushed sha (the
baseline), not just HEAD.

### 4.2 `submit_slurm_job` — **`script_content: str`**

Optional arguments: `time="02:00:00"`, `mem="64G"`, `cpus=8`, `name="spmm-agent"`, `gpu=true`, `dry_run=false`.

1. If the script has no `#SBATCH --partition`, prepend `SBATCH_HEADER`, filled in from the optional arguments, plus
   `--exclude=<state.exclude_nodes>`. If the script has its own header, **still enforce** the exclude list and
   `rtx3090` when `gpu=true`. If `gpu=false`, there is no `--gres` and no partition override; the CPU partition comes
   from what the existing `real_matrices prepare` jobs use.
2. Insert the preamble right after the header, unless the script contains `# AGENT_NO_PREAMBLE`.
3. Substitute these placeholders. **Error if any `{{...}}` remains** after substitution.

   | Placeholder | Value |
   |---|---|
   | `{{BUILD_DIR}}` | Build of current HEAD |
   | `{{BASE_BUILD_DIR}}` | Build of `state.verified` |
   | `{{SRC_DIR}}` | `src/<HEAD sha>` |
   | `{{RUN_DIR}}` | `runs/<HEAD sha>/$SLURM_JOB_ID`; create it at job start |
   | `{{AGENT_ROOT}}` | Remote agent root |
   | `{{NCU}}` | ncu binary path |
   | `{{NCU_FLAGS}}` | the standard flags, §4.6 |
   | `{{SHA}}` | HEAD sha |
4. With `dry_run`, return the rendered script in `{"rendered": "..."}` and don't submit.
5. Otherwise submit, wait (poll `squeue` every 20 s; no time limit beyond the job's own), then query `sacct`.
6. If stdout contains a line `AGENT_TESTS_PASSED <sha>` and the job state is `COMPLETED` with exit 0, set
   `state.tests_passed[sha]=true`. `templates/tests.sh` prints that line only when pytest exits 0.

Returns:
```json
{"job": 23141, "state": "COMPLETED", "exit": 0, "node": "ise-3090-07", "elapsed_s": 341,
 "out_tail": "...last 60 lines...", "err_tail": "...last 60 lines...",
 "out_path": "/home/.../23141.out", "err_path": "/home/.../23141.err", "tests_passed": true}
```
`tests_passed` is present only if the marker logic applies.

**`templates/tests.sh`** (the Optimization Agent will pass its contents to this tool):
- `use {{BUILD_DIR}} {{SRC_DIR}}`.
- Run `python -m pytest test_pbr.py test_csr.py test_boo.py test_boo_edges.py test_boo_shape.py test_kcp_shape.py
  test_boo_fpt.py test_boo_nnzflat_fpt.py test_cusparse_direct.py --boo-require-cuda -q -p no:cacheprovider`
  from `tests/torch_fast_spdmm`. Check these file names exist at implementation time; drop missing ones with a
  comment.
- Then echo the marker on success.

### 4.3 `commit_kernel_attempt` — **`message: str`**

- Must run on the loop branch `state.loop_branch`, and refuses on `main`. The loop branch must start from a commit
  that **already contains `optimizer/agent_tools/` and the three benchmark drivers**, because benchmark drivers always
  come from the candidate sha. `state.verified` starts at that commit. If no loop branch exists yet, return an error
  telling the user to create one from the tools commit; never pick a start point yourself.
- Stage only:
  - `torch_fast_spdmm/**`, excluding build artifacts that are already gitignored;
  - `tests/torch_fast_spdmm/*.py` and `tests/torch_fast_spdmm/benchmark/*.py`;
  - `CHANGELOG.md`.
- Report every other dirty path in `ignored` and don't stage it.
- Commit with `message` exactly as given. Reject an empty message. **Never add `Co-Authored-By` or any other
  trailer.**
- `git push origin <loop_branch>` (never `--force`).
- Return `{"sha": "a1b2c3d", "branch": "opt/loop-20261118", "files": [...], "pushed": true, "ignored": [...]}`, or
  `{"error": "nothing to commit"}`.

### 4.4 `rollback_workspace` — no arguments

Brings `torch_fast_spdmm/` (and the agent's test files) back to the last verified state **without rewriting
history**:

1. Discard uncommitted changes: `git restore --staged --worktree -- torch_fast_spdmm tests/torch_fast_spdmm`, and
   `git clean -fd -- torch_fast_spdmm tests/torch_fast_spdmm/*.py`. Do not use `-x`; ignored files stay.
2. If `git diff --quiet <verified> HEAD -- torch_fast_spdmm tests/torch_fast_spdmm` reports differences:
   - `git checkout <verified> -- torch_fast_spdmm tests/torch_fast_spdmm`, excluding `benchmark/results`;
   - commit `revert: restore torch_fast_spdmm to verified <v> after failed <head>`;
   - push.
3. **Never** run `git reset --hard`, never force-push, never touch `CHANGELOG.md` or any path outside those two.

Return `{"reverted_from": "<head>", "restored_to": "<verified>", "revert_sha": "<sha>|null", "discarded": [...]}`.

### 4.5 `run_benchmarks` — **`kernel_name: str`, `matrix_datasets: list[str]`**

Optional arguments:

| Argument | Default |
|---|---|
| `features` | `[32, 64, 128, 256, 512]` (`[128]` when `matrix_datasets == ["smoke"]`) |
| `tail` | `false` (adds 33, 65, 129) |
| `baseline_ref` | `state.verified` |
| `reps` | `30` |
| `warmup` | `5` |
| `rounds` | `3` |
| `promote` | `false` |
| `time` | auto: about 25 min per matrix × F-count, minimum 1 h |

**`kernel_name`** is `"<fmt>:<variant>[?k=v&...]"`.
- `fmt` is `pbr` or `boo`.
- PBR variant = the `pbr_matmul` `kernel=` value. Default params: `block_rows=8, block_cols=8, min_nnz_per_block=4`.
- BOO variant is one of:
  - `rowseg` (`row_seg=True`);
  - `nnzflat` (with `impl`);
  - `legacy`.

  Defaults: `cluster='hem', target_nr=32, target_nc=32, threads=256, p=64, features_per_thread=1, launch_mode='auto'`.
- Query params override the defaults, e.g. `boo:rowseg?p=64&fpt=4&kr=128&kc=32`.
- Unknown params are an argument error (exit 2).
- Echo the full variant string in `bench.1.kernel`.

**`matrix_datasets`** holds matrix ids and/or subset names, expanded and de-duplicated:

| Subset | Members |
|---|---|
| `tune` | cant, web-Google, roadNet-CA, uniform16, blockdiag32 |
| `held` | soc-Pokec, pdb1HYS, ASIC_680k, amazon0312, rail4284, pruned16x16 |
| `smoke` | cant, roadNet-CA, uniform16 |
| `all` | tune ∪ held |

**Procedure**

1. `ensure_build(HEAD)` and `ensure_build(baseline_ref)`. Make sure `src/<HEAD>` exists, and that the suite manifest
   and the references for every needed (matrix, F) exist. If references are missing, build them on a CPU node first,
   or fail with a clear message pointing to `setup_suite`.
2. Render `templates/bench.sh` into one GPU job. For round `r` in `1..rounds`, shuffle `[base, cand]` with
   `random.Random(seed*1000003 + r)`. For each build, run
   `use <build> {{SRC_DIR}}; python -m benchmark.agent_bench --role <base|cand> --round r --kernel <kernel_name> --cases <json> --reps --warmup --out {{RUN_DIR}}`.
3. **`agent_bench.py`**, for each case:
   - Load the canonical CSR and the seeded X (§5).
   - Construct the format and time the construction (`prep_ms`).
   - Build these arms, then call `bench_nnzflat.measure(torch, arms, warmup, reps, order_seed)`:

     | Arm | Scope |
     |---|---|
     | `kernel` | `cpp_only` (via `PBRTimedSpMM`/`BOOTimedSpMM`) |
     | `kernel_dup` | same as `kernel`, a duplicate for noise |
     | `kernel_api` | `full_call` (event-timed `pbr_matmul`/`boo_matmul`) |
     | `cusparse` | `cpp_only` (`DirectSpMM`, default algorithm) |
   - Record `torch.cuda.max_memory_allocated()` (after `reset_peak_memory_stats()`) per case.
   - Write every sample and summary row to `samples.csv` / `summary.csv`. Include role, round, node, module file
     paths and `x_sha256`.
   - Before building anything, predict the case's GPU bytes: X, Y, format ≈ 3× CSR bytes, cuSPARSE workspace and a
     64 MiB margin. If that exceeds 22 GiB, emit `skip:oom-budget`.
   - A format constructor or launcher that raises an explicit "unsupported" error → `unsupported: <msg>`.
   - If the baseline build lacks an API the candidate uses → `unsupported in base`.
   - **Wrong numbers are never a skip.** They are correctness failures.
4. Wait, then rsync `RUN_DIR` back to the local `tests/torch_fast_spdmm/benchmark/results/agent_runs/<cand>/<job>/`.
   Aggregate **locally**:

   | Output field | Rule |
   |---|---|
   | `k_ms`, `call_ms`, `base_ms`, `cusp_ms` | Medians over **all rounds' samples** of the cand `kernel`, cand `kernel_api`, base `kernel`, and cand `cusparse` arms |
   | `k_iqr` | Candidate (p75−p25)/median × 100 |
   | `x_base` | `base_ms/k_ms` |
   | `x_cusp` | `cusp_ms/k_ms` |
   | `gbps` | `(4·(2·nnz + m + 1) + 4·F·(x_rows_referenced + m)) / (k_ms·1e6)`; `x_rows_referenced` = nonempty columns of A, from the manifest |
   | `mem_mib` | Candidate peak |
   | `noise` | 95th percentile over cases of max(\|median(kernel_dup)/median(kernel) − 1\| for cand, \|cusp_base/cusp_cand − 1\|) |
   | `v` | `thr = max(noise, 0.04)`; `F` if any correctness failure for the cand `kernel`/`kernel_api` arms, else `W` if `x_base > 1+thr`, `L` if `x_base < 1/(1+thr)`, else `T` |
   | `sum.*` | Per §2a of `BENCH_SCHEMA.md`. Geometric means use only `W`/`L`/`T` rows. `bw_pct_max = max(gbps)/936·100` |
   | `ok` | `fail == 0` |
   | `raw` | Repo-relative path of the local `samples.csv` |

5. If `promote` and `ok` and `state.tests_passed[cand]`: set `state.verified = cand` and move the **local** tag
   `opt/verified`. It is not pushed; remote jobs always use explicit shas.
6. Validate against `bench.v1.json`, then print. Also write the JSON next to the samples, as `bench.json`.

### 4.6 `parse_ncu_profile` — **`ncu_report_file: str`**

Optional arguments: `baseline_report=null`.

- **Inputs.** A local or remote (`bgu-slurm:/abs/path` or a path under the agent root) `.ncu-rep` **or** an already
  exported `_raw.csv`.
- **Export.** For a `.ncu-rep`, run `<ncu> --import X.ncu-rep --csv --page raw > X_raw.csv` remotely on the login node
  (ncu import needs no GPU), then pull the CSV.
- **Case metadata.** Read it from the sidecar `X.case.json` written by `agent_profile.py`: `mat`, `F`, `fmt`,
  `kernel`, `commit`.
- **Collection flags.** Define `NCU_FLAGS` exactly as:
  ```
  --set full --metrics smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum --profile-from-start off --target-processes all --import-source no -k regex:pbr_spmm|boo_spmm|csr|cusparse -f
  ```
  `--set full` alone lacks the fadd/fmul counters.
- **`templates/ncu_case.sh`**:
  ```
  use {{BUILD_DIR}} {{SRC_DIR}}; mkdir -p {{RUN_DIR}}/ncu
  {{NCU}} {{NCU_FLAGS}} -o {{RUN_DIR}}/ncu/<mat>_F<F> python -m benchmark.agent_profile --kernel <kernel_name> --mat <mat> --F <F> --out {{RUN_DIR}}/ncu/<mat>_F<F>
  ```
  The agent replaces the `<...>` values.
- **Mapping.** Use exactly the mapping in Appendix B. Every metric name there was checked against a real raw page from
  this cluster (`experiments/2026-11-06_block_shape/analysis/raw_gpu/profile_confirm/fp512_win_raw.csv`), except the
  three FLOP counters added by `--metrics`.
- **Kernels.** Group raw rows by kernel name with template arguments stripped. Sum durations; time-weight percentages
  and ratios. Sort by `us`. Kernels with share ≥ 0.15 get the full entry, the others the minor entry.
- **Missing counters.** A counter absent from the CSV → field `null` and its name in `missing`. Never substitute a
  different counter, except the documented `roof.src = "ffma_only"` fallback.
- **`d`.** With `baseline_report`, parse it the same way, match the main kernel by name, and emit dotted-path deltas
  (`us` relative, the rest absolute) only where |Δ| > 5% of the baseline value.
- **`flags`.** Apply the rules in Appendix B to the kernel with the largest `share`.
- Validate against `ncu.v1.json`, then print. Also write `X.ncu.json` next to the report.

---

## 5. Universal dataset suite (`universal_suite.py` + `setup_suite`)

Normative source: `reports/Universal SpMM benchmark suite.md` (read §1, §3, §4, §5). The essentials:

| id | SuiteSparse | shape | numerical NNZ |
|---|---|---|---|
| web-Google | SNAP/web-Google | 916,428² | 5,105,039 |
| soc-Pokec | SNAP/soc-Pokec | 1,632,803² | 30,622,564 |
| roadNet-CA | SNAP/roadNet-CA | 1,971,281² | 5,533,214 |
| cant | Williams/cant | 62,451² | 4,007,383 |
| pdb1HYS | Williams/pdb1HYS | 36,417² | 4,344,765 |
| ASIC_680k | Sandia/ASIC_680k | 682,862² | 2,638,997 (+1,232,776 explicit zeros, removed) |
| amazon0312 | SNAP/amazon0312 | 400,727² | 3,200,440 |
| rail4284 | **Mittelmann**/rail4284 (not the Williams variant) | 4,284 × 1,096,894 | 11,284,032 |
| uniform16 | generated | 262,144² | 4,194,304 |
| blockdiag32 | generated | 262,144² | 8,388,608 |
| pruned16x16 | generated | 4,096 × 16,384 | 4,194,304 |

**Normalization.** Use `scipy.io.mmread`, then `sp.csr_matrix(..., dtype=float64)`.
1. `sum_duplicates`, `eliminate_zeros`, `sort_indices`.
2. Reject complex or non-finite values.
3. Cast to float32, then `eliminate_zeros` again, and **record** the count removed.
4. int32 indices.
5. Keep the published shape and order. **Never symmetrize, transpose, add loops or drop isolated vertices.**

**Fail ingestion if the canonical NNZ ≠ the table's value** (after the documented zero removal). Use
`real_matrices.MatrixSpec`, `download` (login node) and `prepare` (CPU node). Check `real_matrices.prepare`'s
`explicit_zeros` option against this contract (the `drop` policy).

**Generators** (exact, from report §3; pin and record the numpy version; hash the CSR arrays):
- `uniform16`: `Generator(PCG64(20261117))`. Rows in ascending order; 16 distinct columns sampled uniformly without
  replacement from [0, 262144), sorted, value 1.0.
- `blockdiag32`: `A[r,c] = 1.0` iff `r//32 == c//32`.
- `pruned16x16`: a 256 × 1024 grid of 16×16 tiles. Fresh `Generator(PCG64(20261118))`. For each block row in order,
  sample 64 distinct block columns without replacement, sort them, and fill each tile with 1.0.

**Dense X.** Must be seeded, non-constant, shared across formats, and **identical on every node**. Generate it with
numpy `Generator(PCG64(seed)).standard_normal((n_cols, F), dtype=np.float32)`, not `torch.randn`, which differed on a
non-AVX2 node. Record `x_sha256` per (matrix, F, seed).

**Manifest.** Write `suite/manifest.json`, one entry per matrix, with:
- provenance: URL, archive sha256, member path, original header, published and canonical shape/NNZ, zeros removed,
  dtypes, package versions, CSR hash;
- `nonempty_cols`, used for `gbps`;
- structural stats: row-NNZ median/p95/p99/max/CV, empty rows/cols, column-degree max/CV;
- for the **default** PBR (8×8, min 4) and BOO (HEM 32×32) parameters: useful block occupancy, allocated bytes per
  useful NNZ, remainder fraction, and the mean/max number of groups touching an output row.

**References.** For each (matrix, F) in the suite, compute `bench_real.Reference` once on a CPU node, chunked, with
`--mem` large enough. soc-Pokec at F=512 needs about 7 GB for `ref` + `absref`. Store it under `suite/refs/` with a
sha256 json, in the same style as `GpuReference`'s stem files, and load it with `GpuReference`.

**`setup_suite`** (setup only, not a loop tool):
- args `{"stage": "download"|"prepare"|"refs"|"all"}`;
- runs download on the login node via ssh, and prepare/refs as CPU sbatch jobs;
- returns `{"matrices": {...id: {"nnz":..., "ok": true}}, "manifest": "<remote path>"}`;
- is idempotent.

Memory: 24 GB card. roadNet-CA at F=512 needs about 7.5 GiB for X+Y alone. Run cases sequentially, never densify a
large sparse matrix, and use `torch.cuda.empty_cache()` between cases.

---

## 6. Loop order the tools must support

The Optimization Agent will run each iteration in this order. Make sure every hand-off works, and that each tool's
error message says what to do next.

1. Edit kernel source.
2. `commit_kernel_attempt`.
3. `load_build_env`. A build failure leads to `rollback_workspace`.
4. `submit_slurm_job(tests.sh)`. Failing tests lead to `rollback_workspace`.
5. `run_benchmarks(..., ["tune"])`, then `["held"]` (or `["all"]`) for promotion candidates.
6. `submit_slurm_job(ncu_case.sh)` for 2–4 cases.
7. `parse_ncu_profile(rep, baseline_report)`.
8. The agent writes the `CHANGELOG.md` entry (`reports/agent_loop/CHANGELOG_TEMPLATE.md`) and commits it with the next
   `commit_kernel_attempt`.

---

## 7. Implementation rules

- **Python 3.10+**, standard library plus `jsonschema` locally. Add it to the project's dev dependencies, or vendor a
  minimal validator if adding the dependency is not possible; say which you chose. The cluster side uses only what is
  already in `fem`.
- **Make every remote step idempotent and restartable.** Use marker files (`BUILD_OK.json`, `summary.csv` present, and
  so on) as `phase1_job.sh` does.
- **Keep every tool's stdout under about 4 KB**, except `run_benchmarks` with `all` (about 2k tokens) and
  `submit_slurm_job` tails. Full data always goes to files.
- **Don't import torch locally.** The local side only does ssh, git, file parsing and JSON.
- **Concurrency.** The state file uses a lock. Two `run_benchmarks` calls for different candidates must not collide,
  because run dirs are keyed by sha and job id.
- **Every number in a JSON output is rounded**: times to 3 significant decimals, percentages to 1 decimal, ratios to 3
  decimals.

## 8. Do-not list

- No `git reset --hard`, no `git push --force`, no commits on `main`, no edits in the shared remote clone besides `git fetch`.
- No `.cu`/`.cpp`/`.hpp` edits, **with one exception**: additive changes to the timing bindings `torch_fast_spdmm/bindings/bindings_pbr_timed.cu` and `bindings_boo_timed.cu`, so that `cpp_only` can time every variant `run_benchmarks` accepts. That means a PBR kernel selector (`block`, `chunkcols`, `nnzflat*`), and BOO `launch_mode`/execution-plan support mirroring `boo_matmul`. Conditions:
  - **Don't touch** kernel files (`kernels/*.cu`) or the non-timed bindings, except to expose an existing dispatch function the timer needs to call.
  - **New constructor arguments get defaults** that reproduce today's behavior exactly (`block`, legacy fork/join), so `bench_real`/`bench_tiles` results stay reproducible.
  - **The bracketed interval is unchanged in meaning:** Y zeroing, the kernel(s) actually launched, and the join when there is one.
  - **Add GPU tests:** for every supported variant, `timed.run()` matches `pbr_matmul(kernel=...)` / `boo_matmul(launch_mode=...)`, and the Kineto launch audit (`nc.verify_call`) sees the requested kernel names.
  - **Commit them separately** with scope `bindings`.
  - **Fallback:** any combination still not representable is reported as `unsupported: no cpp_only timer`, never timed through a different path.
- No changes to existing benchmark semantics: wrap `bench_nnzflat.measure`, don't fork it.
- No local CUDA builds or GPU runs. No `torch.randn` for benchmark X.
- No densifying large sparse inputs. No running Python with cwd at any repo root.
- No "estimated" values for missing counters, no silent skips of failing cases, no cross-node speedup ratios.
- No `Co-Authored-By` or other trailers in commits.

## 9. Acceptance checks (all must pass, run in this order)

1. **Offline unit tests** (`pytest tests/agent_tools -q`, locally, no ssh):
   - `parse_ncu_profile` on `experiments/2026-11-06_block_shape/analysis/raw_gpu/profile_confirm/fp512_win_raw.csv`
     (plus a synthetic `case.json`) yields a valid `ncu.1` with `roof.src == "ffma_only"`. `missing` contains exactly the three raw FLOP counters (`smsp__sass_thread_inst_executed_op_{ffma,fadd,fmul}_pred_on.sum`, none of which this fixture has; the derived FFMA fallback is what makes `ffma_only` possible), and `ridge_ai` ≈ 32 (2·10496·1.394e9 / (96·9.492e9)).
   - A second, multi-kernel raw CSV from `experiments/` parses with correct `share` ordering.
   - The bench aggregator on a synthetic `samples.csv` yields hand-computed `x_base`, `noise`, verdicts and geometric
     means, and a `F` case gives `ok=false`.
   - `submit_slurm_job` `dry_run` contains `--exclude=ise-3090-01,ise-3090-04`, the preamble, and no unresolved
     `{{`.
   - `commit_kernel_attempt` and `rollback_workspace` work in a throwaway local git repo (a temp dir with a bare
     "origin"). Rollback creates a revert commit, leaves `CHANGELOG.md` untouched, and never moves HEAD backwards.
   - Both `reports/agent_loop/schemas/examples/*.json` validate.
2. `setup_suite all` succeeds. All 11 NNZ counts match §5, and the manifest has every field.
3. `load_build_env {"env_name": "fem", "modules": []}` builds the current HEAD. Running it a second time returns
   `cached: true` in seconds.
4. `submit_slurm_job` with `templates/tests.sh` → `COMPLETED`, exit 0, `tests_passed: true`.
5. **A/A smoke:** `run_benchmarks {"kernel_name": "boo:rowseg", "matrix_datasets": ["smoke"], "baseline_ref": "<HEAD>"}`
   gives valid `bench.1`, `ok: true`, every `v == "T"`, and `noise ≤ 0.04`. Repeat with `pbr:block`.
6. Profile `cant` at F=128 with `templates/ncu_case.sh` via `submit_slurm_job`. `parse_ncu_profile` on the resulting
   rep gives valid `ncu.1` with `missing == []` and `roof.src == "full"`.
7. `run_benchmarks` on `["all"]` with `pbr:block` completes. Rows that cannot run appear in `skipped` with a reason,
   and none are silently absent.

## 10. Documents you must keep in sync

- `reports/agent_loop/BENCH_SCHEMA.md` and `NCU_SCHEMA.md` are the human-readable spec for every field. If you find a
  definition that is impossible to implement as written, **stop and report it.** Do not change the schema
  unilaterally.
- Add a short "Agent loop tools" section to `CLAUDE.md`'s Key files table that lists `optimizer/agent_tools/` and the
  three new benchmark drivers.

## 11. What to report back when done

- Files created or changed.
- Acceptance results 1–7, with job ids and pasted tool outputs (trim `rows` to 5).
- Any deviation from this spec and why.
- Measured wall time of a `tune` run and an `all` run.
- Anything that surprised you about the cluster.

Commit your work on a branch `tools/agent-loop` in a separate worktree (not the loop branch), with commit scopes
`build`/`tests`/`docs` per `CLAUDE.md`. Do not merge it.

---

## Appendix B: NCU field mapping and flag rules (normative)

Every name was checked against a real `--page raw` export from this cluster, except the three `--metrics` FLOP
counters. Durations come in mixed units, so use `num()` from `nnzflat_profile_report.py`.

| `ncu.1` field | Raw counter(s) / formula |
|---|---|
| `name` | `Kernel Name`, with template arguments and parameter list stripped |
| `us` | `gpu__time_duration.sum` → µs |
| `launch.grid` / `block` / `regs` | `launch__grid_size` / `launch__block_size` / `launch__registers_per_thread` |
| `launch.smem_kb` | (`launch__shared_mem_per_block_static` + `launch__shared_mem_per_block_dynamic`) / 1024 |
| `sol.sm` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| `sol.mem` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| `sol.dram` | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| `sol.l2` | `lts__throughput.avg.pct_of_peak_sustained_elapsed` |
| `sol.l1` | `l1tex__throughput.avg.pct_of_peak_sustained_active` |
| FLOPs | `2·smsp__sass_thread_inst_executed_op_ffma_pred_on.sum + smsp__sass_thread_inst_executed_op_fadd_pred_on.sum + smsp__sass_thread_inst_executed_op_fmul_pred_on.sum` (`src="full"`). Fallback: `derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2` (`src="ffma_only"`). |
| DRAM bytes | `dram__bytes_read.sum + dram__bytes_write.sum` |
| `roof.ai` | FLOPs / DRAM bytes |
| `roof.gflops` | FLOPs / duration / 1e9 |
| `roof.ridge_ai` | (2 · `sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained` · `sm__cycles_elapsed.avg.per_second`) / (`dram__bytes.sum.peak_sustained` · `dram__cycles_elapsed.avg.per_second`) |
| `roof.bound` | `memory` if ai < ridge_ai else `compute` |
| `occ.ach` | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| `occ.theo` | `sm__maximum_warps_per_active_cycle_pct` |
| `occ.limiter` | argmin over `launch__occupancy_limit_{registers,shared_mem,warps,blocks}` → `registers`/`smem`/`warps`/`blocks` (tie → first in that order) |
| `occ.waves` | `launch__waves_per_multiprocessor` |
| `warp.lanes` | `smsp__thread_inst_executed_per_inst_executed.ratio` |
| `warp.branch_eff` | `smsp__sass_average_branch_targets_threads_uniform.pct` |
| `warp.div_br` | `smsp__sass_branch_targets_threads_divergent.sum` |
| `warp.ipc` | `sm__inst_executed.avg.per_cycle_active` |
| `stall` | For every reason in `STALLS`, value = `smsp__average_warps_issue_stalled_<r>_per_issue_active.ratio`. Normalize to % of the sum over all reasons, sort descending, keep the top 5 (top 3 for minor kernels). |
| `mem.l1_hit` / `l2_hit` | `l1tex__t_sector_hit_rate.pct` / `lts__t_sector_hit_rate.pct` |
| `mem.dram_gb` | DRAM bytes / 1e9 |
| `mem.ld_sec_req` | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` |
| `mem.smem_conf_ld` | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum / smsp__sass_inst_executed_op_shared_ld.sum` |
| `mem.smem_conf_st` | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum / smsp__sass_inst_executed_op_shared_st.sum` |
| `mem.red_sec_req` | `l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum / l1tex__t_requests_pipe_lsu_mem_global_op_red.sum` |
| `mem.atom_req` | `l1tex__t_requests_pipe_lsu_mem_global_op_red.sum + l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum` |
| `mem.smem_inst` | `smsp__sass_inst_executed_op_shared_ld.sum + smsp__sass_inst_executed_op_shared_st.sum` |

A ratio with a zero denominator is `null`, and `missing` is not affected. For example, a kernel with no atomics gets
`red_sec_req: null`.

**Flag rules**, applied to the kernel with the largest `share`. Skip a rule if any input is null.

| Flag | Rule |
|---|---|
| `smem_pipe_pressure` | stall share of `mio_throttle` > 25 |
| `barrier_bound` | stall share of `barrier` > 20 |
| `occ_limiter:<limiter>` | `occ.ach < 0.7·occ.theo` or `occ.theo < 50` |
| `lane_underuse` | `warp.lanes < 24` |
| `divergence` | `warp.branch_eff < 90` |
| `uncoalesced_ld` | `mem.ld_sec_req > 5` |
| `uncoalesced_red` | `mem.red_sec_req > 4` |
| `memory_bound` | `sol.dram > 70` and `sol.sm < 40` |
| `latency_bound` | max(`sol.*`) < 40 |

**`d` key set:**
- `us`, as a relative change (cand/base − 1);
- `sol.*`, `occ.ach`, `occ.theo`, `warp.lanes`, `warp.branch_eff`, `mem.*` and `stall.<reason>` (from the full
  normalized stall vector, not only the top 5), as absolute changes.

Include a key only when |cand − base| > 0.05·|base| (for `us`: |rel| > 0.05).

---

## Appendix A: approved JSON Schemas and examples (normative, copy byte-identically)

The human-readable field definitions are in `reports/agent_loop/BENCH_SCHEMA.md` and `reports/agent_loop/NCU_SCHEMA.md`.
Read both: they define every formula used above.

### A.1 `bench.v1.json`
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "spmm-agent-loop/bench.v1.json",
  "title": "bench.1 - compact SpMM benchmark result",
  "description": "Output of run_benchmarks. Field meanings: reports/agent_loop/BENCH_SCHEMA.md.",
  "type": "object",
  "additionalProperties": false,
  "required": ["v", "ok", "fmt", "kernel", "cand", "base", "env", "noise", "sum", "prep_ms", "cols", "rows", "skipped", "raw"],
  "$defs": {
    "sha": {"type": "string", "pattern": "^[0-9a-f]{7,40}$"},
    "subset": {
      "type": "object",
      "additionalProperties": false,
      "required": ["n", "geo_x_base", "geo_x_cusp", "W", "L", "T"],
      "properties": {
        "n": {"type": "integer", "minimum": 0},
        "geo_x_base": {"type": ["number", "null"], "exclusiveMinimum": 0},
        "geo_x_cusp": {"type": ["number", "null"], "exclusiveMinimum": 0},
        "W": {"type": "integer", "minimum": 0},
        "L": {"type": "integer", "minimum": 0},
        "T": {"type": "integer", "minimum": 0}
      }
    },
    "pos": {"type": "number", "exclusiveMinimum": 0},
    "posOrNull": {"type": ["number", "null"], "exclusiveMinimum": 0}
  },
  "properties": {
    "v": {"const": "bench.1"},
    "ok": {"type": "boolean"},
    "fmt": {"enum": ["pbr", "boo"]},
    "kernel": {"type": "string", "minLength": 1},
    "cand": {"$ref": "#/$defs/sha"},
    "base": {"$ref": "#/$defs/sha"},
    "env": {
      "type": "object",
      "additionalProperties": false,
      "required": ["node", "job", "dtype", "reps", "warmup", "rounds"],
      "properties": {
        "node": {"type": "string"},
        "job": {"type": "integer"},
        "dtype": {"const": "fp32"},
        "reps": {"type": "integer", "minimum": 1},
        "warmup": {"type": "integer", "minimum": 0},
        "rounds": {"type": "integer", "minimum": 1}
      }
    },
    "noise": {"type": "number", "minimum": 0},
    "sum": {
      "type": "object",
      "additionalProperties": false,
      "required": ["tune", "held", "bw_pct_max", "fail", "skip"],
      "properties": {
        "tune": {"$ref": "#/$defs/subset"},
        "held": {"$ref": "#/$defs/subset"},
        "bw_pct_max": {"type": ["number", "null"], "minimum": 0},
        "fail": {"type": "integer", "minimum": 0},
        "skip": {"type": "integer", "minimum": 0}
      }
    },
    "prep_ms": {"type": "object", "additionalProperties": {"$ref": "#/$defs/pos"}},
    "cols": {
      "const": ["mat", "F", "k_ms", "k_iqr", "call_ms", "base_ms", "cusp_ms", "x_base", "x_cusp", "gbps", "mem_mib", "v"]
    },
    "rows": {
      "type": "array",
      "items": {
        "type": "array",
        "prefixItems": [
          {"type": "string"},
          {"type": "integer", "minimum": 1},
          {"$ref": "#/$defs/posOrNull"},
          {"type": ["number", "null"], "minimum": 0},
          {"$ref": "#/$defs/posOrNull"},
          {"$ref": "#/$defs/posOrNull"},
          {"$ref": "#/$defs/posOrNull"},
          {"$ref": "#/$defs/posOrNull"},
          {"$ref": "#/$defs/posOrNull"},
          {"type": ["number", "null"], "minimum": 0},
          {"type": ["number", "null"], "minimum": 0},
          {"enum": ["W", "L", "T", "F"]}
        ],
        "items": false,
        "minItems": 12
      }
    },
    "skipped": {
      "type": "array",
      "items": {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}, {"type": "string"}],
        "items": false,
        "minItems": 3
      }
    },
    "raw": {"type": "string"}
  }
}
```

### A.2 Example `bench.1` object
```json
{
  "v": "bench.1", "ok": true,
  "fmt": "boo", "kernel": "rowseg", "cand": "a1b2c3d", "base": "e4f5a6b",
  "env": {"node": "ise-3090-07", "job": 23141, "dtype": "fp32", "reps": 30, "warmup": 5, "rounds": 3},
  "noise": 0.031,
  "sum": {
    "tune": {"n": 25, "geo_x_base": 1.062, "geo_x_cusp": 0.941, "W": 9, "L": 1, "T": 15},
    "held": {"n": 29, "geo_x_base": 1.018, "geo_x_cusp": 0.902, "W": 4, "L": 2, "T": 23},
    "bw_pct_max": 71.4, "fail": 0, "skip": 1
  },
  "prep_ms": {"cant": 182.0, "web-Google": 941.3},
  "cols": ["mat","F","k_ms","k_iqr","call_ms","base_ms","cusp_ms","x_base","x_cusp","gbps","mem_mib","v"],
  "rows": [
    ["cant",128,0.412,1.8,0.455,0.441,0.398,1.070,0.966,512.3,1210,"W"],
    ["web-Google",512,6.210,2.4,6.390,6.180,5.020,0.995,0.808,388.1,4410,"T"]
  ],
  "skipped": [["rail4284",512,"unsupported: rectangular"]],
  "raw": "tests/torch_fast_spdmm/benchmark/results/agent_runs/a1b2c3d/23141/samples.csv"
}
```

### A.3 `ncu.v1.json`
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "spmm-agent-loop/ncu.v1.json",
  "title": "ncu.1 - compact Nsight Compute bottleneck summary",
  "description": "Output of parse_ncu_profile. Field meanings: reports/agent_loop/NCU_SCHEMA.md.",
  "type": "object",
  "additionalProperties": false,
  "required": ["v", "rep", "case", "k", "d", "flags", "missing"],
  "$defs": {
    "pct": {"type": ["number", "null"], "minimum": 0},
    "num": {"type": ["number", "null"]},
    "stallList": {
      "type": "array",
      "maxItems": 5,
      "items": {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "number", "minimum": 0, "maximum": 100}],
        "items": false,
        "minItems": 2
      }
    },
    "sol": {
      "type": "object",
      "additionalProperties": false,
      "required": ["sm", "mem", "dram", "l2", "l1"],
      "properties": {
        "sm": {"$ref": "#/$defs/pct"}, "mem": {"$ref": "#/$defs/pct"}, "dram": {"$ref": "#/$defs/pct"},
        "l2": {"$ref": "#/$defs/pct"}, "l1": {"$ref": "#/$defs/pct"}
      }
    },
    "kernelFull": {
      "type": "object",
      "additionalProperties": false,
      "required": ["name", "us", "share", "launch", "sol", "roof", "occ", "warp", "stall", "mem"],
      "properties": {
        "name": {"type": "string"},
        "us": {"type": "number", "minimum": 0},
        "share": {"type": "number", "minimum": 0, "maximum": 1},
        "launch": {
          "type": "object", "additionalProperties": false,
          "required": ["grid", "block", "regs", "smem_kb"],
          "properties": {
            "grid": {"type": ["integer", "null"]}, "block": {"type": ["integer", "null"]},
            "regs": {"type": ["integer", "null"]}, "smem_kb": {"$ref": "#/$defs/num"}
          }
        },
        "sol": {"$ref": "#/$defs/sol"},
        "roof": {
          "type": "object", "additionalProperties": false,
          "required": ["ai", "gflops", "ridge_ai", "bound", "src"],
          "properties": {
            "ai": {"$ref": "#/$defs/num"}, "gflops": {"$ref": "#/$defs/num"},
            "ridge_ai": {"$ref": "#/$defs/num"},
            "bound": {"enum": ["memory", "compute", null]},
            "src": {"enum": ["full", "ffma_only", null]}
          }
        },
        "occ": {
          "type": "object", "additionalProperties": false,
          "required": ["ach", "theo", "limiter", "waves"],
          "properties": {
            "ach": {"$ref": "#/$defs/pct"}, "theo": {"$ref": "#/$defs/pct"},
            "limiter": {"enum": ["registers", "smem", "warps", "blocks", null]},
            "waves": {"$ref": "#/$defs/num"}
          }
        },
        "warp": {
          "type": "object", "additionalProperties": false,
          "required": ["lanes", "branch_eff", "div_br", "ipc"],
          "properties": {
            "lanes": {"type": ["number", "null"], "minimum": 0, "maximum": 32},
            "branch_eff": {"$ref": "#/$defs/pct"},
            "div_br": {"$ref": "#/$defs/num"},
            "ipc": {"$ref": "#/$defs/num"}
          }
        },
        "stall": {"$ref": "#/$defs/stallList"},
        "mem": {
          "type": "object", "additionalProperties": false,
          "required": ["l1_hit", "l2_hit", "dram_gb", "ld_sec_req", "smem_conf_ld", "smem_conf_st", "red_sec_req", "atom_req", "smem_inst"],
          "properties": {
            "l1_hit": {"$ref": "#/$defs/pct"}, "l2_hit": {"$ref": "#/$defs/pct"},
            "dram_gb": {"$ref": "#/$defs/num"}, "ld_sec_req": {"$ref": "#/$defs/num"},
            "smem_conf_ld": {"$ref": "#/$defs/num"}, "smem_conf_st": {"$ref": "#/$defs/num"},
            "red_sec_req": {"$ref": "#/$defs/num"}, "atom_req": {"$ref": "#/$defs/num"},
            "smem_inst": {"$ref": "#/$defs/num"}
          }
        }
      }
    },
    "kernelMinor": {
      "type": "object",
      "additionalProperties": false,
      "required": ["name", "us", "share", "sol", "stall"],
      "properties": {
        "name": {"type": "string"},
        "us": {"type": "number", "minimum": 0},
        "share": {"type": "number", "minimum": 0, "maximum": 1},
        "sol": {"$ref": "#/$defs/sol"},
        "stall": {"type": "array", "maxItems": 3, "items": {"$ref": "#/$defs/stallList/items"}}
      }
    }
  },
  "properties": {
    "v": {"const": "ncu.1"},
    "rep": {"type": "string"},
    "case": {
      "type": "object",
      "additionalProperties": false,
      "required": ["mat", "F", "fmt", "kernel", "commit"],
      "properties": {
        "mat": {"type": "string"},
        "F": {"type": "integer", "minimum": 1},
        "fmt": {"enum": ["pbr", "boo"]},
        "kernel": {"type": "string"},
        "commit": {"type": "string", "pattern": "^[0-9a-f]{7,40}$"}
      }
    },
    "k": {
      "type": "array",
      "minItems": 1,
      "items": {"oneOf": [{"$ref": "#/$defs/kernelFull"}, {"$ref": "#/$defs/kernelMinor"}]}
    },
    "d": {"type": "object", "additionalProperties": {"type": "number"}},
    "flags": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string",
        "pattern": "^(smem_pipe_pressure|barrier_bound|occ_limiter:(registers|smem|warps|blocks)|lane_underuse|divergence|uncoalesced_ld|uncoalesced_red|memory_bound|latency_bound)$"
      }
    },
    "missing": {"type": "array", "items": {"type": "string"}}
  }
}
```

### A.4 Example `ncu.1` object
```json
{
  "v": "ncu.1", "rep": "tests/torch_fast_spdmm/benchmark/results/agent_runs/a1b2c3d/23150/ncu/cant_F128.ncu-rep",
  "case": {"mat": "cant", "F": 128, "fmt": "boo", "kernel": "rowseg", "commit": "a1b2c3d"},
  "k": [
    {
      "name": "boo_spmm_rowseg_kernel", "us": 412.0, "share": 0.91,
      "launch": {"grid": 7812, "block": 256, "regs": 40, "smem_kb": 34.5},
      "sol":  {"sm": 42.1, "mem": 78.3, "dram": 61.0, "l2": 55.4, "l1": 70.2},
      "roof": {"ai": 0.31, "gflops": 210.0, "ridge_ai": 32.1, "bound": "memory", "src": "full"},
      "occ":  {"ach": 48.2, "theo": 66.7, "limiter": "smem", "waves": 3.1},
      "warp": {"lanes": 27.9, "branch_eff": 96.1, "div_br": 1234, "ipc": 1.10},
      "stall": [["mio_throttle",31.2],["barrier",18.0],["long_scoreboard",14.5],["lg_throttle",6.1],["short_scoreboard",4.0]],
      "mem":  {"l1_hit": 41.0, "l2_hit": 72.5, "dram_gb": 0.21, "ld_sec_req": 4.1,
               "smem_conf_ld": 0.02, "smem_conf_st": 0.0, "red_sec_req": 8.0, "atom_req": 1.2e6, "smem_inst": 3.4e6}
    },
    {
      "name": "spdmm_csr_naive_kernel", "us": 40.1, "share": 0.09,
      "sol": {"sm": 12.0, "mem": 35.2, "dram": 30.1, "l2": 22.0, "l1": 28.4},
      "stall": [["long_scoreboard",55.0],["lg_throttle",12.0],["wait",8.0]]
    }
  ],
  "d": {"us": -0.071, "stall.mio_throttle": -12.4, "occ.ach": 10.1},
  "flags": ["smem_pipe_pressure", "occ_limiter:smem", "uncoalesced_red"],
  "missing": []
}
```
