#!/bin/bash
#SBATCH --job-name=cusparse-direct
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dayanb/research/cusparse-direct-20260929/job-%j.log
#SBATCH --error=/home/dayanb/research/cusparse-direct-20260929/job-%j.err
source ~/.bashrc
conda activate fem
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX="$CXX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=4 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export NSYS="$CONDA_PREFIX/nsight-compute-2025.2.1/host/target-linux-x64/nsys"
cd /home/dayanb/research/cusparse-direct-20260929
export PYTHONPATH="$PWD/torch_fast_spdmm:$PWD/tests/torch_fast_spdmm:${PYTHONPATH:-}"
if [ "$1" = validate ]; then
    cd torch_fast_spdmm
    python setup.py build_ext --inplace > ../build.log 2>&1
    cd ..
    python -m pytest tests/torch_fast_spdmm/test_cusparse_direct.py tests/torch_fast_spdmm/test_boo.py tests/torch_fast_spdmm/test_pbr.py --boo-require-cuda -q > correctness.log 2>&1
    cd tests/torch_fast_spdmm
    python -m benchmark.bench_spmm --direct-cusparse --sizes 1024 --profiles fem2d_permuted diag_skewed --features 32 128 --pbr-configs 4:4 8:2 --boo-k 8 32 --repeats 3 --warmup 2 --wandb-mode disabled --out ../../results/smoke
    python -m benchmark.profile_torch_grid --grid --sizes 1024 --profiles fem2d_permuted --features 128 --out ../../results/profile_smoke
else
    cd tests/torch_fast_spdmm
    python -m benchmark.bench_spmm --direct-cusparse --wandb-mode disabled --out "../../results/job$1"
    if [ "$1" = 1 ]; then
        "$NSYS" --version > ../../results/nsys_version.txt
        python -m benchmark.profile_torch_grid --grid --out ../../results/profiles
    fi
fi
