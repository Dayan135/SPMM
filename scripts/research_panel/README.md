# Synthetic profiling panel

The other research track owns `benchmark/real_matrices.py` and `benchmark/bench_real.py`.
This track owns `benchmark/research_panel.py`, `benchmark/panel_structure.py`,
their focused tests, and these job/report scripts.

Run all builds and tests on the remote GPU node. Use a new isolated snapshot
with `ROOT` set to its absolute path. `job.sh build` uses `build_ext --inplace`;
it does not reinstall or modify the shared environment's package.

1. Submit `job.sh build` on RTX 3090 (8 CPUs, 32 GB, 30 minutes).
2. After successful build/tests, submit `job.sh smoke` on RTX 3090.
3. Submit `job.sh cluster` on a CPU node (8 CPUs, 64 GB, normal QoS). It prepares
   the six synthetic profiles at N=262144, identity/HEM, k=32/64. It requires
   the built extension to import the package, but sees no GPU.
4. After smoke and clustering succeed, run `job.sh benchmark` on RTX 3090
   (8 CPUs, 64 GB). It uses saved labels and F=128/512, with full-call and
   preallocated C++ measurements. It excludes incorrect raw-store diagnostics.
5. Run `job.sh profile` on RTX 3090. The initial profiler subset is natural FEM,
   dense diagonal, and row-skewed matrices at F=512/k=32, comparing BOO/torch/direct.
   Nsight Systems, the established stall-counter set, and memory/occupancy
   counters run separately. Direct profiling selects the fastest validated
   full-call algorithm from the completed panel, matching exact A/X hashes.

`PANEL_SIZE` overrides N for cluster/benchmark/profile; it must match the labels.
Smoke always uses N=1024 with separate labels. Each GPU output directory must be
new. On retry, preserve previous artifacts and choose another directory/snapshot.

The benchmark records matrix, input, label, source and extension hashes. CSV
ratios match timing scopes. API BOO uses `auto`, while the existing preallocated
C++ harness uses a legacy fork/join; this distinction is retained explicitly.
Preprocessing, format build, staging and detailed support distributions are
outside matmul timing. Three allocation replicates and held-out matrices remain
necessary before promoting an optimization.

After collecting results:

```bash
python scripts/research_panel/report.py <results-directory>
```

The report keeps unprofiled timing comparisons separate from profiler counters.
`profile_kernels.csv` contains per-kernel trace durations, which can overlap;
do not sum them and call the result complete-call elapsed time.

## Paired FPT controls

`research_panel --phase benchmark --boo-fpt 1 2 4` interleaves FPT variants
using the same BOO object and labels. Include FPT1 as the paired reference.
The preallocated C++ wrapper sets the process-wide selector before every call.
For a single profiled variant, use `--phase profile --boo-fpt 2` (or 1/4).

`fpt_control.sh` runs the focused harness and FPT correctness suites, then four
synthetic controls at HEM k32, N262144, F128/512. Its binary hash intentionally
pins the validated experimental build; use `ROOT` for an isolated copy and
`LABELS` for existing synthetic labels. Request the GPU model explicitly:
`--partition=rtx3090 --account=erant --qos=erant --gres=gpu:rtx_3090:1`.
Do not treat a generic `--gres=gpu:1` allocation as a hardware match.
