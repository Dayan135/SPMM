#!/bin/bash
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
: "${ROOT:?Set isolated sweep root}"
export CUDA_HOME="$CONDA_PREFIX"
export CC="$(which x86_64-conda-linux-gnu-cc)"
export CXX="$(which x86_64-conda-linux-gnu-c++)"
export CUDAHOSTCXX="$CXX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
export PYTHONUNBUFFERED=1
export SPMM_MATRIX_CACHE=/home/dayanb/research/real-matrices-20261008/matrix_cache
cd "$ROOT"
RUN_ROOT="$ROOT/eran_run"
SMOKE_ROOT="$RUN_ROOT/${SMOKE_ROOT_SUFFIX:-smoke}"
MODE=${1:?Expected preflight, cluster or matmul}
if [ "$MODE" = cluster ]; then
    export OMP_NUM_THREADS=8
else
    # GPU jobs use Slurm's default CPU allocation.
    export OMP_NUM_THREADS=1
fi
if [ "$MODE" = preflight ]; then
    (cd "$ROOT/torch_fast_spdmm" && python setup.py build_ext --inplace)
    python -c 'import torch, torch_fast_spdmm_core as c; print("torch", torch.__version__, "extension", c.__file__)'
    python -m pytest tests/torch_fast_spdmm/benchmark/test_research_panel.py tests/torch_fast_spdmm/benchmark/test_real_matrices.py tests/torch_fast_spdmm/test_boo_fpt.py --boo-require-cuda -q
    CUDA_VISIBLE_DEVICES= SWEEP_SMOKE_PHASE=cluster python scripts/research_panel/full_sweep.py --phase smoke --root "$SMOKE_ROOT" --cache "$SPMM_MATRIX_CACHE"
    SWEEP_SMOKE_PHASE=matmul python scripts/research_panel/full_sweep.py --phase smoke --root "$SMOKE_ROOT" --cache "$SPMM_MATRIX_CACHE"
else
    if [ "$MODE" = cluster ]; then export CUDA_VISIBLE_DEVICES=; fi
    python scripts/research_panel/full_sweep.py --phase "$MODE" --root "$RUN_ROOT" --cache "$SPMM_MATRIX_CACHE"
fi
