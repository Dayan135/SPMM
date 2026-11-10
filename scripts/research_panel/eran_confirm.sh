#!/bin/bash
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
: "${ROOT:?Set isolated sweep root}"
: "${SLURM_ARRAY_TASK_ID:?Expected allocation index 0, 1 or 2}"
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$ROOT"
python scripts/research_panel/eran_confirm.py \
    --root "$ROOT/eran_run" \
    --allocation-index "$((SLURM_ARRAY_TASK_ID + 1))" \
    --cache /home/dayanb/research/real-matrices-20261008/matrix_cache
