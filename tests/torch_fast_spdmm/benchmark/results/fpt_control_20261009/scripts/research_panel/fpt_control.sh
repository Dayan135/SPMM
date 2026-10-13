#!/bin/bash
# An isolated copy of the already validated FPT extension; no shared installation changes.
source ~/.bashrc
conda activate fem
set -euo pipefail
: "${ROOT:?Set ROOT to the isolated FPT control snapshot}"
: "${LABELS:?Set LABELS to the completed synthetic panel labels (read-only)}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/results"
cd "$ROOT"
python -c 'import hashlib, torch; import torch_fast_spdmm_core as c; from pathlib import Path; actual=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(); assert actual == "bc688ff876317ddee01a888d5de6d9ac08b7ea87a331090e03021daaa0ff607c", actual; print("Validated FPT binary:", actual)'
python -m pytest tests/torch_fast_spdmm/benchmark/test_research_panel.py \
    tests/torch_fast_spdmm/test_boo_fpt.py --boo-require-cuda -q > results/control-tests.log 2>&1
cd tests/torch_fast_spdmm
python -m benchmark.research_panel --phase benchmark --labels "$LABELS" \
    --out "$ROOT/results/panel" --sizes 262144 \
    --profiles fem2d fem2d_permuted diag_b1.0 pattern_row_skewed \
    --features 128 512 --methods hem --groups 32 --boo-fpt 1 2 4
cd "$ROOT"
python scripts/research_panel/report.py results
