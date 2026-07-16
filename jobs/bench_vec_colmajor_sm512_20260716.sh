#!/bin/bash
#SBATCH --job-name=bench_vec_colmajor_sm512
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

# Same sweep as bench_vec_colmajor_20260716.sh (vectorized 'block' kernel,
# col-sorted, baseline only, same 7 (block_size, min_nnz) configs), but at
# smem_size=512 instead of 2048. -k filters to just the new sm512 configs
# (plus baselines) so the already-benchmarked sm2048 configs aren't rerun.

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

echo "[job] benchmarking size=$SYS_SIZE (cusparse, csr_naive, block_colmajor sm512 only)"
pytest tests/torch_fast_spdmm/test_benchmark.py -v --benchmark-only \
    -k "_${SYS_SIZE} and (cusparse or csr_naive or (block_colmajor and sm512))"

echo "[job] done $(date)"
