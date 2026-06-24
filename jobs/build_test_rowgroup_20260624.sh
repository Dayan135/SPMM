#!/bin/bash
#SBATCH --job-name=build_rowgroup
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

# Build + correctness tests for the row-group fusion kernel.
# Runs the full PBR and CSR test suites plus the new TestPBRRowgroupKernel class.

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

echo "[job] building editable install"
pip install -e torch_fast_spdmm --no-build-isolation

echo "[job] running correctness tests"
pytest tests/torch_fast_spdmm/test_pbr.py \
       tests/torch_fast_spdmm/test_csr.py -v

echo "[job] done $(date)"
