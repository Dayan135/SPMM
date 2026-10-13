#!/bin/bash
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
: "${ROOT:?Set isolated sweep root}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export SPMM_MATRIX_CACHE=/home/dayanb/research/real-matrices-20261008/matrix_cache
cd "$ROOT"
MODE=${1:?Expected preflight, cluster or matmul}
if [ "$MODE" = preflight ]; then
    python -c 'import hashlib, torch; import torch_fast_spdmm_core as c; from pathlib import Path; assert hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest() == "bc688ff876317ddee01a888d5de6d9ac08b7ea87a331090e03021daaa0ff607c"'
    python -m pytest tests/torch_fast_spdmm/benchmark/test_research_panel.py tests/torch_fast_spdmm/benchmark/test_real_matrices.py tests/torch_fast_spdmm/test_boo_fpt.py --boo-require-cuda -q
    CUDA_VISIBLE_DEVICES= SWEEP_SMOKE_PHASE=cluster python scripts/research_panel/full_sweep.py --phase smoke --root "$ROOT/smoke" --cache "$SPMM_MATRIX_CACHE"
    SWEEP_SMOKE_PHASE=matmul python scripts/research_panel/full_sweep.py --phase smoke --root "$ROOT/smoke" --cache "$SPMM_MATRIX_CACHE"
else
    if [ "$MODE" = cluster ]; then export CUDA_VISIBLE_DEVICES=; fi
    python scripts/research_panel/full_sweep.py --phase "$MODE" --root "$ROOT" --cache "$SPMM_MATRIX_CACHE"
fi
