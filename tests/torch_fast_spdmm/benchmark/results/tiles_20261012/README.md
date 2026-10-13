# BOO tile experiments — results (2026-10-12)

Report: [TILE_EXPERIMENTS.md](../../../../../experiments/2026-10-12_tile_optimization/TILE_EXPERIMENTS.md). Remote root:
`/home/dayanb/research/boo-tiles-20261012` (isolated snapshot; `.nsys-rep` files and the
large audit `trace.sqlite` remain there only). Extension SHA-256 `b980dbc2…c229c`
(`build_hashes.json`).

This checkout keeps the reports, selected figures, summaries, and metadata.
Raw samples, profiling captures, logs, and failed-run diagnostics are in the
separate [result archive](../../../../../RESULTS_ARCHIVE.md). The layout below
describes the complete archived run; restore it before regenerating reports.

**Timing protocol.** Only unprofiled timings decide performance. Directories marked *rotation*
used the inherited rotation order and are kept for provenance; their small differences are not
used. Everything else used per-repeat shuffled order (`order_mode` column).

## Layout

| Path | Contents | Job(s) |
|---|---|---|
| `build_hashes.json`, `logs/` | Build, source hashes, GPU records, Slurm/test/sanitizer logs | 22711735, 22711787, 22712100 |
| `sass/` | Static SASS opcode counts per rowseg specialization; default-vs-validated SASS identity; binary hashes | 22711817 |
| `structure/<case>/structure.json` | CPU block statistics, warp row-span efficiency, staging models | 22711843 |
| `profile_smoke/`, `profile_f512/` | ncu CSVs (stalls/occ/mem/instmix/atomics), nsys SQLite, per-run `*_meta/profile.json`; `analysis/` tidy tables | 22711909, 22712096–8, 22712699–700 |
| `screen_pxfpt/` | P×FPT screen (*rotation*) | 22711918 |
| `screen_pxfpt_failed_22711841/` | Partial run that aborted on a Kineto false negative (pwtk only) | 22711841 |
| `screen_pxfpt_shuffle/` | P×FPT screen, randomized | 22713030 |
| `slab_stageA/` | Slab stage A (*rotation*) | 22712093 |
| `slab_shuffle/` | Slab × P/FPT screen, randomized | 22713031 |
| `order_control/{rotation,shuffle,rotation_seed7}/` | Order-sensitivity control on one allocation | 22712897 |
| `correctness_f130/`, `correctness_slab_f1000_f1025/` | Validation-focused runs (3 repeats) | 22712009, 22712094 |
| `roadnet_f512_direct_fill/`, `roadnet_f512_alg3_fill/` | Missing roadNet F512 baselines with repeated BOO anchors | 22712092, 22712180 |
| `audit_screen/` | nsys launch audit: `audit.csv` (1,140 BOO ranges, 0 mismatches) | 22712709 |
| `confirm/alloc{1,2,3}/<process>/` | Confirmation: `main_f512`, `main_f1024`, `scale_f512`, `scale_f512_alg3`, `roadnet_f512`, `roadnet_f512_alg3` | 22713805–7 |
| `report*/` | Derived tables and figures (below) | local |
| `plans/` | Exact profiling plans and confirmation argument files | — |
| `debug/` | Throwaway diagnostics (Kineto capture test) | — |

Every bench directory contains:
- `summary.csv`: medians, IQR, correctness, launch report and verification status.
- `samples.csv`: every timed sample, with predecessor and order mode.
- `launch.csv`: the launcher's own report for each requested variant, including infeasible ones.
- `metadata.json`: source/extension hashes, node, job, arguments and per-case input hashes.

## Derived outputs

- `report_shuffle/`:
  - `screen_cells.csv`
  - `screen_heatmap_{full_call,cpp}_22713030.png|pdf`
  - `finalist_resources.csv`
- `report_slab_shuffle/`: `slab_cells.csv`, `slab_full_call.png|pdf`.
- `report_confirm/`:
  - `confirm_summary.csv` and `confirm_per_allocation.csv`
  - `identical_work_control.csv` and `direct_coverage.csv`
  - `allocations.csv`
  - `confirm_vs_direct_f512.png|pdf`
- `report/`, `report_slabA/`: the same tables for the rotation-ordered runs (provenance only).

## Reproduce

Run on the cluster, from the snapshot root:

```bash
R=/home/dayanb/research/boo-tiles-20261012
S="--partition=rtx3090 --account=erant --qos=erant --gres=gpu:rtx_3090:1 --cpus-per-task=8 --mem=120G --export=ALL,ROOT=$R"

# Build in place + tests + sanitizer + SASS
sbatch $S scripts/boo_tile_experiments/job.sh build
sbatch $S scripts/boo_tile_experiments/job.sh tests
sbatch $S scripts/boo_tile_experiments/job.sh sanitizer
sbatch $S scripts/boo_tile_experiments/job.sh sass

# Randomized P x FPT screen; slab screen
sbatch $S scripts/boo_tile_experiments/job.sh bench --features 128 512 --p 16 32 64 128 --fpt 1 2 4 \
    --slab 512 --cpp --order shuffle --order-seed 11 --out $R/results/screen_pxfpt_shuffle
sbatch $S scripts/boo_tile_experiments/job.sh bench --cases pwtk pwtk_hem diag_b1.0 fem2d fem2d_permuted \
    --features 128 512 1024 --configs <P:FPT:slab ...> --cpp --order shuffle --order-seed 12 --out $R/results/slab_shuffle

# Profiles (plan line: case F method P FPT slab direct_alg); add PASSES=atomics for the sector pass
sbatch $S scripts/boo_tile_experiments/job.sh profile $R/results/plans/prof_a.txt $R/results/profile_f512

# Launch audit; confirmation (one args file per allocation)
sbatch $S scripts/boo_tile_experiments/job.sh audit $R/results/audit_screen --features 128 512 \
    --p 16 32 64 128 --fpt 1 2 4 --slab 128 256 512 1024 --cpp --direct-algorithms
sbatch $S scripts/boo_tile_experiments/job.sh benchseq $R/results/plans/confirm_alloc1.txt   # also 2, 3
```

Derived outputs are produced locally, from the repository root (fem env for plots):

```bash
D=tests/torch_fast_spdmm/benchmark/results/tiles_20261012
python3 scripts/boo_tile_experiments/analyze_profiles.py $D/profile_f512
conda run -n fem python scripts/boo_tile_experiments/report_tiles.py screen $D/report_shuffle $D/screen_pxfpt_shuffle
conda run -n fem python scripts/boo_tile_experiments/report_tiles.py slab $D/report_slab_shuffle $D/slab_shuffle
python3 scripts/boo_tile_experiments/finalist_resources.py $D/report_shuffle/finalist_resources.csv \
    64:4:512 128:4:512 64:2:512 -- $D/screen_pxfpt_shuffle/launch.csv -- $D/profile_f512/analysis/ncu_wide.csv
python3 scripts/boo_tile_experiments/confirm_report.py $D/confirm $D/report_confirm
conda run -n fem python scripts/boo_tile_experiments/plot_confirm.py \
    $D/report_confirm/confirm_per_allocation.csv $D/report_confirm/confirm_vs_direct_f512.png
```
