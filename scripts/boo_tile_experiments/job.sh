#!/bin/bash
# BOO P x FPT x slab experiments (PROFILING_CONTINUATION_TASK.md). Submit with an
# explicit RTX 3090:
#   sbatch --partition=rtx3090 --account=erant --qos=erant --gres=gpu:rtx_3090:1 \
#          --cpus-per-task=8 --mem=96G --time=... \
#          --export=ALL,ROOT=<snapshot> scripts/boo_tile_experiments/job.sh <mode> [args]
# ROOT is an isolated snapshot built in place; the shared installed extension
# is never touched (PYTHONPATH points at the snapshot).
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
: "${ROOT:?Set ROOT to the isolated source snapshot}"
export CUDA_HOME="$CONDA_PREFIX"
export CC="$(command -v x86_64-conda-linux-gnu-cc)"
export CXX="$(command -v x86_64-conda-linux-gnu-c++)"
export CUDAHOSTCXX="$CXX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
export OMP_NUM_THREADS=8 MAX_JOBS=8 PYTHONUNBUFFERED=1
export SPMM_MATRIX_CACHE=/home/dayanb/research/real-matrices-20261008/matrix_cache
export TILE_LABELS=${TILE_LABELS:-/home/dayanb/research/boo-full-fpt-20261009/labels}
MODE=${1:?Expected build, tests, sanitizer, bench, benchseq, audit, profile, structure or sass}
shift
RESULTS="$ROOT/results"
mkdir -p "$RESULTS" "$RESULTS/logs"
cd "$ROOT"

gpu_record() {
    {
        echo "job=${SLURM_JOB_ID:-none} node=$(hostname) date=$(date -Is)"
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm --format=csv
    } > "$1"
}

case "$MODE" in
build)
    gpu_record "$RESULTS/logs/build-gpu-${SLURM_JOB_ID:-local}.txt"
    (cd torch_fast_spdmm && python setup.py build_ext --inplace) \
        > "$RESULTS/logs/build-${SLURM_JOB_ID:-local}.log" 2>&1
    python - <<'EOF' > "$RESULTS/build_hashes.json"
import hashlib, json, torch
from pathlib import Path
import torch_fast_spdmm_core as c
root = Path('.')
src = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(
    list(root.glob('torch_fast_spdmm/kernels/*')) + list(root.glob('torch_fast_spdmm/bindings/*'))
    + list(root.glob('torch_fast_spdmm/torch_fast_spdmm/*.py')) + [root / 'torch_fast_spdmm/setup.py'])
    if p.is_file()}
print(json.dumps(dict(extension=c.__file__, extension_sha256=hashlib.sha256(
    Path(c.__file__).read_bytes()).hexdigest(), source_sha256=src,
    torch=torch.__version__, cuda=torch.version.cuda), indent=2))
EOF
    cat "$RESULTS/build_hashes.json" | head -5
    ;;
tests)
    gpu_record "$RESULTS/logs/tests-gpu-${SLURM_JOB_ID:-local}.txt"
    python -m pytest tests/torch_fast_spdmm/benchmark/test_tiles.py \
        tests/torch_fast_spdmm/test_boo_fpt.py tests/torch_fast_spdmm/test_boo.py \
        tests/torch_fast_spdmm/test_boo_edges.py tests/torch_fast_spdmm/test_boo_plan.py \
        tests/torch_fast_spdmm/test_cusparse_direct.py \
        tests/torch_fast_spdmm/benchmark/test_real_matrices.py \
        --boo-require-cuda -q -p no:cacheprovider "$@" \
        > "$RESULTS/logs/tests-${SLURM_JOB_ID:-local}.log" 2>&1
    tail -3 "$RESULTS/logs/tests-${SLURM_JOB_ID:-local}.log"
    ;;
sanitizer)
    # Representative new boundary/alignment cases under memcheck/racecheck/synccheck.
    SAN=$(command -v compute-sanitizer || echo "$CONDA_PREFIX/bin/compute-sanitizer")
    for TOOL in memcheck racecheck synccheck; do
        "$SAN" --tool "$TOOL" --error-exitcode 9 --target-processes all \
            python -m pytest tests/torch_fast_spdmm/benchmark/test_tiles.py -m boo_sanitizer \
            --boo-require-cuda -q -p no:cacheprovider -x \
            > "$RESULTS/logs/sanitizer-$TOOL-${SLURM_JOB_ID:-local}.log" 2>&1
        echo "$TOOL: $(tail -1 "$RESULTS/logs/sanitizer-$TOOL-${SLURM_JOB_ID:-local}.log")"
    done
    ;;
bench)
    gpu_record "$RESULTS/logs/bench-gpu-${SLURM_JOB_ID:-local}.txt"
    cd tests/torch_fast_spdmm
    python -m benchmark.bench_tiles --labels "$TILE_LABELS" --cache "$SPMM_MATRIX_CACHE" "$@"
    ;;
benchseq)
    # Each non-comment line of the args file is one bench_tiles invocation, run in
    # order in THIS allocation (same GPU), e.g. anchors repeated around variants.
    gpu_record "$RESULTS/logs/benchseq-gpu-${SLURM_JOB_ID:-local}.txt"
    ARGS_FILE=${1:?args file}
    cd tests/torch_fast_spdmm
    while read -r -u 3 LINE; do
        [[ -z "$LINE" || "$LINE" == \#* ]] && continue
        eval "set -- $LINE"
        python -m benchmark.bench_tiles --labels "$TILE_LABELS" --cache "$SPMM_MATRIX_CACHE" "$@"
    done 3< "$ARGS_FILE"
    ;;
audit)
    # Launch audit: the given bench_tiles arguments under nsys with 1 repeat; every
    # validation call's NVTX range must contain exactly the requested specialization.
    NSYS=${NSYS:-$CONDA_PREFIX/nsight-compute-2025.2.1/host/target-linux-x64/nsys}
    OUTDIR=${1:?audit output dir}
    shift
    mkdir -p "$OUTDIR"
    cd tests/torch_fast_spdmm
    "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --export=sqlite \
        --force-overwrite=true -o "$OUTDIR/trace" python -m benchmark.bench_tiles \
        --labels "$TILE_LABELS" --cache "$SPMM_MATRIX_CACHE" --warmup 0 --repeats 1 \
        --out "$OUTDIR/bench" "$@" > "$OUTDIR/nsys.log" 2>&1
    python "$ROOT/scripts/boo_tile_experiments/audit_launches.py" "$OUTDIR/trace.sqlite" \
        "$OUTDIR/bench/launch.csv" > "$OUTDIR/audit.csv" 2> "$OUTDIR/audit_summary.txt" || true
    cat "$OUTDIR/audit_summary.txt"
    rm -f "$OUTDIR/trace.nsys-rep"
    ;;
profile)
    gpu_record "$RESULTS/logs/profile-gpu-${SLURM_JOB_ID:-local}.txt"
    cd tests/torch_fast_spdmm
    bash "$ROOT/scripts/boo_tile_experiments/profile_cases.sh" "$@"
    ;;
structure)
    # CPU only: block support, warp row-span efficiency and staging models per case.
    export CUDA_VISIBLE_DEVICES=
    cd tests/torch_fast_spdmm
    for CASE in "$@"; do
        python -m benchmark.profile_tiles --structure --case "$CASE" --labels "$TILE_LABELS" \
            --cache "$SPMM_MATRIX_CACHE" --out "$RESULTS/structure/$CASE"
    done
    ;;
sass)
    SO=$(python -c 'import torch, torch_fast_spdmm_core as c; print(c.__file__)')
    OUT="$RESULTS/sass"
    mkdir -p "$OUT"
    cuobjdump -sass "$SO" > "$OUT/all.sass"
    python "$ROOT/scripts/boo_tile_experiments/sass_summary.py" "$OUT/all.sass" > "$OUT/rowseg_summary.csv"
    # Default (slab512) specializations vs the validated full-sweep binary.
    VALIDATED=/home/dayanb/research/boo-full-fpt-20261009/torch_fast_spdmm/torch_fast_spdmm_core.cpython-312-x86_64-linux-gnu.so
    sha256sum "$VALIDATED" "$SO" > "$OUT/binaries.sha256"
    cuobjdump -sass "$VALIDATED" > "$OUT/validated.sass"
    python "$ROOT/scripts/boo_tile_experiments/sass_summary.py" --compare "$OUT/validated.sass" \
        "$OUT/all.sass" > "$OUT/default_vs_validated.csv"
    python -m pytest tests/torch_fast_spdmm/benchmark/test_tiles.py tests/torch_fast_spdmm/test_boo_fpt.py \
        --boo-require-cuda -q -rs -p no:cacheprovider > "$OUT/skip_reasons.log" 2>&1 || true
    ;;
*)
    printf 'Unknown mode: %s\n' "$MODE" >&2
    exit 2
    ;;
esac
