#!/bin/bash
#SBATCH --job-name=bench_tb_mat
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-49
#SBATCH --output=jobs/logs/%A_%a.out
#SBATCH --error=jobs/logs/%A_%a.err

# Full block-of-blocks sweep, one SLURM array task per (size, matrix).
#   task_id = size_idx * 10 + matrix_idx   (5 sizes x 10 matrices = 50 tasks)
# Each task: 234 kernel/config combos x 6 features (~1.4k benchmarks) -> never
# times out and logs a small W&B table (avoids the 10k-row session-table cap).
# Rerun of the per-size sweep after: density rescaled O(N) (was O(N^2), OOM at
# 262144), and per-(size,matrix) split (per-size tasks timed out at 16384+).
# .so assumed already built by jobs/build_spdmm_20260623.sh on this commit.

SYS_SIZES=(1024 4096 16384 65536 262144)
MATRICES=(diag_b0.3 diag_b1.0 \
          scattered_1x8_d0.5pct scattered_1x8_d2pct \
          scattered_8x1_d0.5pct scattered_8x1_d2pct \
          scattered_2x2_d0.5pct scattered_2x2_d2pct \
          scattered_2x4_d0.5pct scattered_2x4_d2pct)
NM=${#MATRICES[@]}

SIDX=$(( SLURM_ARRAY_TASK_ID / NM ))
MIDX=$(( SLURM_ARRAY_TASK_ID % NM ))
SYS_SIZE=${SYS_SIZES[$SIDX]}
MATRIX=${MATRICES[$MIDX]}

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  task=$SLURM_ARRAY_TASK_ID  size=$SYS_SIZE  matrix=$MATRIX  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export SPDMM_COMMIT=$(git rev-parse --short HEAD)
export SPDMM_ONLY_SIZE=$SYS_SIZE
export SPDMM_ONLY_MATRIX=$MATRIX
echo "[job] commit=$SPDMM_COMMIT  size=$SYS_SIZE  matrix=$MATRIX"

pytest tests/torch_fast_spdmm/test_benchmark.py -q --benchmark-only \
    --benchmark-min-rounds=5 --benchmark-max-time=0.3 --benchmark-warmup=off

echo "[job] done $(date)"
