#!/bin/bash
#SBATCH --job-name=bench_scalar_diag_colmajor
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=jobs/logs/%A_%a.out
#SBATCH --error=jobs/logs/%A_%a.err

# Full benchmark sweep: cusparse + csr_naive baselines vs the zero_idle-kernel
# ('scalar_diag') baseline/mergeBC variants, col-sorted (col_major=True), at
# the user-requested (block_size, min_nnz) configs:
#   2x2: min_nnz 2, 3
#   4x4: min_nnz 4, 8
#   8x8: min_nnz 2, 12, 20
# Across all 10 regular matrix families and the full existing SIZES x FEATURES
# grid (test_benchmark.py defaults). One SLURM array task per size (existing
# per-size splitting convention -- see bench_blockgroup_20260629.sh). -k
# restricts to just the 2 baselines + scalar_diag PBR runs, not the full
# existing block/rowgroup/colgroup/tile suite.

SYS_SIZES=(1024 4096 16384 65536 262144)
SYS_SIZE=${SYS_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  task=$SLURM_ARRAY_TASK_ID  size=$SYS_SIZE  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export SPDMM_COMMIT=$(git rev-parse --short HEAD)
echo "[job] commit=$SPDMM_COMMIT"

echo "[job] benchmarking size=$SYS_SIZE (cusparse, csr_naive, scalar_diag only)"
pytest tests/torch_fast_spdmm/test_benchmark.py -v --benchmark-only \
    -k "_${SYS_SIZE} and (cusparse or csr_naive or scalar_diag)"

echo "[job] done $(date)"
