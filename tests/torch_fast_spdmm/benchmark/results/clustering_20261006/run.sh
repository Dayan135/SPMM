#!/bin/bash
#SBATCH --job-name=spmm-clustering
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dayanb/research/clustering-20261006/job-%j.log
source ~/.bashrc
conda activate fem
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=8
cd /home/dayanb/research/clustering-20261006
python -m pip install --target deps --no-deps 'pymetis>=2025.2.2'
export PYTHONPATH="$PWD/deps:$PWD/torch_fast_spdmm:$PWD/tests/torch_fast_spdmm"
cd torch_fast_spdmm
python setup.py build_ext --inplace
cd ..
pytest tests/torch_fast_spdmm/test_clustering_backends.py tests/torch_fast_spdmm/test_boo_edges.py tests/torch_fast_spdmm/test_boo.py --boo-require-cuda -q
cd tests/torch_fast_spdmm
python -m benchmark.bench_clustering --sizes 1024 --profiles fem2d_permuted diag_b1.0 --features 128 --warmup 2 --repeats 5 --out ../../results/smoke
python -m benchmark.bench_clustering --sizes 16384 --profiles fem2d fem2d_permuted diag_b0.3 diag_b1.0 --out ../../results/full
