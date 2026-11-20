### Final-tile trailing barrier elimination and LOAD-phase overlap with binary search
- **Timestamp:** 2026-11-20T16:33:25+02:00
- **Commit:** 209772f  (baseline 78f8a18)
- **Change Type:** Kernel Logic
- **Description:** Reordered `LOAD` in `boo_spmm_rowseg_kernel` (`boo_kernel.cu`) so `sh_nz_idx` loads before Barrier 1 while `sh_nz_val`, `sh_col_ids`, and `sh_row_ids` load concurrently with `sh_row_start` binary search before Barrier 2, and guarded the trailing `__syncthreads()` with `base_f + P < feat_end`. Hypothesis: expected `barrier` stalls (19.6–21.0%) to drop at zero smem/register cost. `load_build_env` failed with `ok: false`, `warnings: 0`, `err_tail: ""` (Slurm job 24345273) — same root cause as Attempt 2, triggering the §7 early stop rule.
- **Outcome Summary:** FAIL (build) · tune n/a (0/0/0) · held n/a (0/0/0) · vs cuSPARSE tune n/a held n/a · peak BW n/a · noise n/a
- **Reference:** Slurm job 24345273

### P/2-strided 2-feature register accumulation with coalesced atomics and final-tile barrier elimination
- **Timestamp:** 2026-11-20T16:29:15+02:00
- **Commit:** 67f5692  (baseline 78f8a18)
- **Change Type:** Kernel Logic
- **Description:** Modified `FPT == 1` path of `boo_spmm_rowseg_kernel` (`boo_kernel.cu`) to accumulate two `P/2`-strided features (`f0` and `f0 + P/2`) per thread and guard the trailing `__syncthreads()` with `base_f + P < feat_end`. Hypothesis: expected `smem_inst` and `mio_throttle` (50.1% on `blockdiag32_F128`) and `barrier` stalls to drop with zero arena increase. `load_build_env` failed 3 times (`ok: false`, `warnings: 0`, `err_tail: ""`, Slurm jobs 24345076, 24345144, 24345243).
- **Outcome Summary:** FAIL (build) · tune n/a (0/0/0) · held n/a (0/0/0) · vs cuSPARSE tune n/a held n/a · peak BW n/a · noise n/a
- **Reference:** Slurm jobs 24345076, 24345144, 24345243

### Double-buffered smem_X with Ampere cp.async and final-tile barrier elimination
- **Timestamp:** 2026-11-20T16:19:00+02:00
- **Commit:** 8e43f3e  (baseline 78f8a18)
- **Change Type:** Kernel Logic
- **Description:** Double-buffered `smem_X` in `boo_spmm_rowseg_kernel` (`boo_kernel.cu`, `boo.py`) using Ampere `cp.async` (gated to <= 32 KB arena) and eliminated the final-tile trailing barrier. Hypothesis: expected `long_scoreboard` (31.6–32.2%) and `barrier` (21.0%) stalls to drop by overlapping tile t+1 X loads with tile t compute. Tune gate not met (0.992x, 1W/2L/17T); `web-Google` F=64/128 regressed (0.908x/0.949x) and `blockdiag32` F=128 was 0.963x, consistent with 2x `smem_X` reducing resident CTAs/SM.
- **Outcome Summary:** ROLLED BACK · tune 0.992× (1/2/17) · held n/a (0/0/0) · vs cuSPARSE tune 0.820× held n/a · peak BW 31.0% of 936 GB/s · noise 0.005
- **Reference:** tests/torch_fast_spdmm/benchmark/results/agent_runs/8e43f3e96b44d2e4df17c017e6cead9abf19337e/24338529/samples.csv ; Slurm job 24338529
