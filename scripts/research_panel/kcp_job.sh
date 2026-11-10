#!/bin/bash
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
: "${ROOT:?isolated snapshot}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export SPMM_MATRIX_CACHE=/home/dayanb/research/real-matrices-20261008/matrix_cache
RUN="$ROOT/shape_run"
LABELS=/home/dayanb/research/eran-fpt-20261013/labels
ART=/home/dayanb/research/nnzflat-20261014/nnzflat_run
cd "$ROOT"
printf '%s  %s\n' 9d4318d5149ce732a5dc38d8c3e0bd9a507491561069b956563cb0d21214f1cd "$ROOT/torch_fast_spdmm/torch_fast_spdmm_core.cpython-312-x86_64-linux-gnu.so" | sha256sum -c -
MODE=${1:?mode}
case "$MODE" in
gate)
 CUDA_VISIBLE_DEVICES= python scripts/research_panel/kcp_gate.py --run "$RUN" --source-labels "$LABELS" --old-labels /home/dayanb/research/block-shape-20261106/shape_run/labels --cache "$SPMM_MATRIX_CACHE" --task "$SLURM_ARRAY_TASK_ID"
 ;;
screen)
 : "${ALLOC:?allocation}"
 python -m benchmark.bench_shape --artifact-root "$ART" --run "$RUN" --source-labels "$LABELS" --manifest "$RUN/manifests/kcp.json" --task "$SLURM_ARRAY_TASK_ID" --out "$RUN/results/kcp/alloc$ALLOC/task_$(printf %03d "$SLURM_ARRAY_TASK_ID")" --cache "$SPMM_MATRIX_CACHE" --allocation "kcp$ALLOC-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID" --order-seed-offset "$ALLOC"
 ;;
stage0)
 python -m pytest tests/torch_fast_spdmm/test_kcp_shape.py --boo-require-cuda -q
 for TOOL in memcheck racecheck synccheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool "$TOOL" --error-exitcode 1 python -m pytest tests/torch_fast_spdmm/test_kcp_shape.py --boo-require-cuda -m boo_sanitizer -q
 done
 ;;
audit)
 STAGES=kcp AUDIT_MIN_F=128 bash scripts/research_panel/shape_job.sh audit
 ;;
profile)
 NCU=${NCU:-$CONDA_PREFIX/nsight-compute-2025.2.1/ncu}
 OUT="$RUN/profile/kcp"; mkdir -p "$OUT"
 while read -r LABEL CELL ARM; do
  [[ -z "$LABEL" || "$LABEL" == \#* ]] && continue
  STEM="$OUT/$LABEL"
  [ -f "${STEM}_raw.csv" ] && continue
  "$NCU" --profile-from-start off --target-processes all --set full --import-source no -o "$STEM" -f python scripts/research_panel/shape_profile.py --artifact-root "$ART" --run "$RUN" --source-labels "$LABELS" --manifest "$RUN/manifests/kcp.json" --cell "$CELL" --arm "$ARM" --cache "$SPMM_MATRIX_CACHE" --out "${STEM}_meta" > "$STEM.log" 2>&1
  "$NCU" --import "$STEM.ncu-rep" --csv --page raw > "${STEM}_raw.csv"
 done < "$RUN/manifests/profile_kcp.txt"
 ;;
*) exit 2;;
esac
