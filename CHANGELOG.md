### Double-buffered smem_X with Ampere cp.async and final-tile barrier elimination
- **Timestamp:** 2026-11-20T16:19:00+02:00
- **Commit:** 8e43f3e  (baseline 78f8a18)
- **Change Type:** Kernel Logic
- **Description:** Double-buffered `smem_X` in `boo_spmm_rowseg_kernel` (`boo_kernel.cu`, `boo.py`) using Ampere `cp.async` (gated to <= 32 KB arena) and eliminated the final-tile trailing barrier. Hypothesis: expected `long_scoreboard` (31.6–32.2%) and `barrier` (21.0%) stalls to drop by overlapping tile t+1 X loads with tile t compute. Tune gate not met (0.992x, 1W/2L/17T); `web-Google` F=64/128 regressed (0.908x/0.949x) and `blockdiag32` F=128 was 0.963x, consistent with 2x `smem_X` reducing resident CTAs/SM.
- **Outcome Summary:** ROLLED BACK · tune 0.992× (1/2/17) · held n/a (0/0/0) · vs cuSPARSE tune 0.820× held n/a · peak BW 31.0% of 936 GB/s · noise 0.005
- **Reference:** tests/torch_fast_spdmm/benchmark/results/agent_runs/8e43f3e96b44d2e4df17c017e6cead9abf19337e/24338529/samples.csv ; Slurm job 24338529
