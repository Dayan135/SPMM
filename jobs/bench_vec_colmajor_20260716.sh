#!/bin/bash
#SBATCH --job-name=bench_vec_colmajor
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

# Same sweep as bench_scalar_diag_colmajor_20260716.sh, but over the
# vectorized 'block' kernel (pbr_spmm_vec_kernel -- auto-selected since every
# features value here is %4==0) instead of the forced-scalar zero_idle kernel,
# and WITHOUT mergeBC this round (baseline vec kernel only). Same
# user-requested (block_size, min_nnz) configs, col-sorted (col_major=True),
# fixed smem_size=2048. One SLURM array task per size (existing convention).

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

echo "[job] benchmarking size=$SYS_SIZE (cusparse, csr_naive, block_colmajor only)"
pytest tests/torch_fast_spdmm/test_benchmark.py -v --benchmark-only \
    -k "_${SYS_SIZE} and (cusparse or csr_naive or block_colmajor)"

echo "[job] done $(date)"
