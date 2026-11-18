# CHANGELOG entry template (agent optimization loop)

Every structural change gets one entry in `CHANGELOG.md` at the repository root, newest first. Failed and
rolled-back attempts are logged too: a negative result is a result. `rollback_workspace` never modifies
`CHANGELOG.md`.

## Format

```markdown
### <short title of the hypothesis>
- **Timestamp:** 2026-11-18T14:05:00+02:00
- **Commit:** a1b2c3d  (baseline e4f5a6b)
- **Change Type:** Kernel Logic | Memory Structure | Build Flag
- **Description:** <what was changed, in which kernel/file, and the hypothesis it tests: "expected X to drop because Y">
- **Outcome Summary:** <PASS|FAIL|ROLLED BACK> · tune <geo_x_base>× (W/L/T) · held <geo_x_base>× (W/L/T) · vs cuSPARSE tune <geo_x_cusp>× held <geo_x_cusp>× · peak BW <bw_pct_max>% of 936 GB/s · noise <noise>
- **Reference:** <bench.1 `raw` path> ; <ncu.1 `rep` path(s)> ; Slurm job <env.job>
```

## Where each field comes from

| Field | Source |
|---|---|
| Timestamp | Wall-clock time when the entry is written, ISO 8601 with offset. |
| Commit | `commit_kernel_attempt` → `sha`. Baseline = `bench.1.base`. |
| Change Type | `Kernel Logic` = control flow / work mapping. `Memory Structure` = data layout, shared-memory staging, format arrays. `Build Flag` = nvcc / setup.py options. |
| Outcome PASS/FAIL | `PASS` = pytest suite passed **and** `bench.1.ok`. `FAIL` = build, test or correctness failure (state which). `ROLLED BACK` = correct but not kept (no gain on `held`, or a regression). |
| Speedups | `bench.1.sum.tune.geo_x_base`, `sum.held.geo_x_base`, `geo_x_cusp`, and the W/L/T counts. |
| Peak Memory Bandwidth | `bench.1.sum.bw_pct_max`. |
| Reference | `bench.1.raw`, `ncu.1.rep`, Slurm job id (`bench.1.env.job` or `submit_slurm_job` → `job`). |

## Example

```markdown
### Coalesced FPT>1 atomics in boo_spmm_rowseg (shfl transpose before RED)
- **Timestamp:** 2026-11-18T14:05:00+02:00
- **Commit:** a1b2c3d  (baseline e4f5a6b)
- **Change Type:** Kernel Logic
- **Description:** Step C of `boo_spmm_rowseg_kernel` (boo_kernel.cu) now transposes FPT=4 partial sums with `__shfl_sync` so each warp's atomicAdd hits 32 consecutive floats. Hypothesis: `red_sec_req` 16→4 and lower L2 atomic load on hub-heavy inputs (web-Google, soc-Pokec).
- **Outcome Summary:** PASS · tune 1.051× (6/0/19) · held 1.012× (3/1/25) · vs cuSPARSE tune 0.962× held 0.910× · peak BW 68.2% of 936 GB/s · noise 0.028
- **Reference:** tests/torch_fast_spdmm/benchmark/results/agent_runs/a1b2c3d/23141/samples.csv ; tests/torch_fast_spdmm/benchmark/results/agent_runs/a1b2c3d/23150/ncu/web-Google_F512.ncu-rep ; Slurm job 23141
```
