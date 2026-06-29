#!/bin/bash
#SBATCH --job-name=build_blockgroup
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

# Build + correctness tests + benchmark for the blockgroup kernel.
# Pulls feat/pbr-blockgroup, builds the .so, runs test_pbr (blockgroup
# correctness) and test_benchmark (block vs blockgroup sweep).

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

echo "[job] pulling feat/pbr-blockgroup"
git fetch origin
git checkout feat/pbr-blockgroup
git pull

export SPDMM_COMMIT=$(git rev-parse --short HEAD)
echo "[job] commit=$SPDMM_COMMIT"

echo "[job] building editable install"
pip install -e torch_fast_spdmm --no-build-isolation

echo "[job] === correctness tests ==="
pytest tests/torch_fast_spdmm/test_pbr.py \
       tests/torch_fast_spdmm/test_csr.py -v

echo "[job] === benchmark (size=16384) ==="
export WANDB_MODE=offline
pytest tests/torch_fast_spdmm/test_benchmark.py -v --benchmark-only \
    -k "_16384"

echo "[job] done $(date)"
