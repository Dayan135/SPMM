#!/bin/bash
#SBATCH --job-name=hem-pbr
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dayanb/research/hem-pbr-20260928/job.log
#SBATCH --error=/home/dayanb/research/hem-pbr-20260928/job.err
source ~/.bashrc
conda activate fem
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX="$CXX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=4
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/dayanb/research/hem-pbr-20260928/torch_fast_spdmm:/home/dayanb/research/hem-pbr-20260928/tests/torch_fast_spdmm:${PYTHONPATH:-}"
cd /home/dayanb/research/hem-pbr-20260928/torch_fast_spdmm
python setup.py build_ext --inplace > ../build.log 2>&1
cd ..
python - <<'PY'
import pathlib, torch_fast_spdmm, torch_fast_spdmm_core
for module in (torch_fast_spdmm, torch_fast_spdmm_core):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(pathlib.Path.cwd())
    print(module.__file__)
PY
python -m pytest tests/torch_fast_spdmm/benchmark/test_hem_pbr.py tests/torch_fast_spdmm/benchmark/test_matrices.py tests/torch_fast_spdmm/test_boo.py tests/torch_fast_spdmm/test_pbr.py --boo-require-cuda -q > correctness.log 2>&1
cd tests/torch_fast_spdmm
python -m benchmark.bench_hem_pbr --sizes 1024 --features 32 --hem-k 8 32 --pbr-configs 4:4 8:2 --repeats 3 --warmup 2 --out ../../results/smoke
python -m benchmark.bench_hem_pbr --out ../../results/full
