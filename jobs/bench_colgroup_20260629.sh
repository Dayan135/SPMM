#!/bin/bash
#SBATCH --job-name=bench_colgroup
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

# Full benchmark sweep for colgroup vs block vs blockgroup kernels.
# Runs all 3 sizes as a SLURM array (one task per size).
# .so assumed already built by build_test_colgroup_20260629.sh.

SYS_SIZES=(16384 65536 262144)
SYS_SIZE=${SYS_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  task=$SLURM_ARRAY_TASK_ID  size=$SYS_SIZE  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export SPDMM_COMMIT=$(git rev-parse --short HEAD)
echo "[job] commit=$SPDMM_COMMIT"

echo "[job] benchmarking size=$SYS_SIZE"
pytest tests/torch_fast_spdmm/test_benchmark.py -v --benchmark-only \
    -k "_${SYS_SIZE}"

echo "[job] done $(date)"
