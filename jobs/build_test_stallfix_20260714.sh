#!/bin/bash
#SBATCH --job-name=build_test_stallfix
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

# Build + full correctness test for feat/pbr-block-stall-fix (Step B/C merge,
# smem_Y removal in pbr_spmm_vec_kernel / pbr_spmm_zero_idle_kernel). Pure
# refactor -- must reproduce identical results, no behavior change expected.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

pip install -e torch_fast_spdmm --no-build-isolation

pytest tests/torch_fast_spdmm/ -v

echo "[job] done $(date)"
