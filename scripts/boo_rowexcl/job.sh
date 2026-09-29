#!/bin/bash
#SBATCH --job-name=boo-rowexcl
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dayanb/research/boo-rowexcl-20260929/job-%j.log
#SBATCH --error=/home/dayanb/research/boo-rowexcl-20260929/job-%j.err
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
cd /home/dayanb/research/boo-rowexcl-20260929
export PYTHONPATH="$PWD/torch_fast_spdmm:$PWD/tests/torch_fast_spdmm:${PYTHONPATH:-}"
if [ "$1" = validate ]; then
    # First build precedes imports; nothing is installed into the shared env.
    (cd torch_fast_spdmm && python setup.py build_ext --inplace) > build.log 2>&1
fi
python - <<'CHECK'
import pathlib, torch_fast_spdmm as m, torch_fast_spdmm_core as core
root = '/home/dayanb/research/boo-rowexcl-20260929/'
for module in (m, core):
    p = str(pathlib.Path(module.__file__).resolve())
    print(p)
    assert p.startswith(root), p
CHECK
if [ "$1" = validate ]; then
    python -m pytest tests/torch_fast_spdmm/test_boo_rowexcl.py tests/torch_fast_spdmm/test_boo.py tests/torch_fast_spdmm/test_cusparse_direct.py --boo-require-cuda -q > correctness.log 2>&1
elif [ "$1" = smoke ]; then
    cd tests/torch_fast_spdmm
    python -m benchmark.bench_boo_rowexcl --out ../../results/smoke --replicate 1 --sizes 1024 --profiles fem2d_permuted diag_skewed pattern_row_skewed --features 128 512 --boo-k 8 32 --warmup 2 --repeats 3
else
    cd tests/torch_fast_spdmm
    python -m benchmark.bench_boo_rowexcl --out "../../results/phase_a_job$1" --replicate "$1"
fi
