#!/bin/bash
#SBATCH --job-name=bench_diag
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=jobs/logs/%A_%a.out
#SBATCH --error=jobs/logs/%A_%a.err

# Benchmark the three reworked diagonal-block matrix profiles (64×64 blocks):
#   diag_dd0.017  diag_dd0.033  diag_dd0.05
# Diagonal density is now dominant (≥95% of nnz), so this sweep actually
# exercises PBR on banded sparsity — unlike the previous dd0.001/002/003 runs
# where >97% of nnz fell to the CSR remainder.
#
# No CUDA changes since a1dbf5b; .so is assumed already built.

SYS_SIZES=(16384 65536 262144)
SYS_SIZE=${SYS_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  task=$SLURM_ARRAY_TASK_ID  size=$SYS_SIZE  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export WANDB_MODE=offline
export SPDMM_COMMIT=$(git rev-parse --short HEAD)
echo "[job] commit=$SPDMM_COMMIT"

echo "[job] benchmarking size=$SYS_SIZE  (diag families only)"
pytest tests/torch_fast_spdmm/test_benchmark.py -v --benchmark-only \
    -k "diag_dd and _${SYS_SIZE}"

echo "[job] done $(date)"
