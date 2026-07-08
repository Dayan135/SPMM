#!/bin/bash
#SBATCH --job-name=bench_tileblock
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=jobs/logs/%A_%a.out
#SBATCH --error=jobs/logs/%A_%a.err

# Full block-of-blocks benchmark sweep: block / rowgroup / colgroup / tile
# (9 shapes x {grid,greedy}) over 8 PBR configs x 3 smem tiers, 13 matrices,
# 6 feature counts. One SLURM array task per system size.
#
# Each task sets SPDMM_ONLY_SIZE so pytest collects (and W&B names the run for)
# just that size -- ~18.4k benchmarks/task. .so assumed already built by
# jobs/build_spdmm_20260623.sh on this commit.

SYS_SIZES=(1024 4096 16384 65536 262144)
SYS_SIZE=${SYS_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  task=$SLURM_ARRAY_TASK_ID  size=$SYS_SIZE  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export SPDMM_COMMIT=$(git rev-parse --short HEAD)
export SPDMM_ONLY_SIZE=$SYS_SIZE
echo "[job] commit=$SPDMM_COMMIT  size=$SYS_SIZE"

# --benchmark-max-time caps per-benchmark timed duration; min-rounds keeps stats
# stable. Warm-up (CUDA launch + tile-meta build) is done explicitly in the test.
pytest tests/torch_fast_spdmm/test_benchmark.py -q --benchmark-only \
    --benchmark-min-rounds=5 --benchmark-max-time=0.3 --benchmark-warmup=off

echo "[job] done $(date)"
