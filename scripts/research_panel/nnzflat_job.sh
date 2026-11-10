#!/bin/bash
# nnzflat FPT1/2 experiment runner (experiments/2026-10-14_nnzflat). ROOT is an
# isolated source snapshot; RUN=$ROOT/nnzflat_run holds artifacts and results.
#
# GPU modes (one RTX3090, <= 48 GB, no explicit CPU request, OMP_NUM_THREADS=1):
#   stage0 | screen1 | screen2 (array over inputs) | confirm (array over allocations) | profile
# CPU modes (cpu partition): prep (array over inputs) | manifest | finalize | report
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
: "${ROOT:?Set ROOT to the isolated nnzflat snapshot}"
export CUDA_HOME="$CONDA_PREFIX"
export CC="$(command -v x86_64-conda-linux-gnu-cc)"
export CXX="$(command -v x86_64-conda-linux-gnu-c++)"
export CUDAHOSTCXX="$CXX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
export PYTHONUNBUFFERED=1
export SPMM_MATRIX_CACHE=/home/dayanb/research/real-matrices-20261008/matrix_cache
SOURCE_LABELS=/home/dayanb/research/eran-fpt-20261013/labels
RUN="$ROOT/nnzflat_run"
NSYS="$CONDA_PREFIX/nsight-compute-2025.2.1/host/target-linux-x64/nsys"
MODE=${1:?mode}
mkdir -p "$RUN/logs" "$ROOT/results"
cd "$ROOT"
echo "mode=$MODE job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_TASK_ID:-} node=$(hostname) $(date -Is)"

gpu_env() {
    export OMP_NUM_THREADS=1
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
    echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-} SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
}

case "$MODE" in
stage0)
    gpu_env
    S0="$RUN/stage0"
    mkdir -p "$S0"
    (cd torch_fast_spdmm && MAX_JOBS=${SLURM_CPUS_ON_NODE:-4} python setup.py build_ext --inplace) > "$S0/build.log" 2>&1
    python - <<'EOF' > "$ROOT/results/build_hashes.json"
import hashlib, json, pathlib, torch, torch_fast_spdmm_core as c
p = pathlib.Path(c.__file__).resolve()
print(json.dumps({'extension_path': str(p), 'extension_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}))
EOF
    cat "$ROOT/results/build_hashes.json"
    T=tests/torch_fast_spdmm
    python -m pytest $T/test_boo_nnzflat_fpt.py --boo-require-cuda -ra -q \
        --junitxml "$S0/nnzflat-fpt-results.xml" > "$S0/nnzflat-fpt.log" 2>&1 && echo "nnzflat tests PASS" || { tail -40 "$S0/nnzflat-fpt.log"; exit 1; }
    python -m pytest $T/test_boo.py $T/test_boo_edges.py $T/test_boo_fpt.py --boo-require-cuda -ra -q \
        --junitxml "$S0/boo-results.xml" > "$S0/boo-suite.log" 2>&1 && echo "BOO suite PASS" || { tail -40 "$S0/boo-suite.log"; exit 1; }
    python -m pytest $T/test_pbr.py $T/test_csr.py $T/test_sparse_block_mat.py tests/optimizer -q \
        --junitxml "$S0/regression-results.xml" > "$S0/regression.log" 2>&1 && echo "regressions PASS" || { tail -40 "$S0/regression.log"; exit 1; }
    for TOOL in memcheck racecheck synccheck; do
        PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool $TOOL --error-exitcode 1 \
            python -m pytest $T/test_boo_edges.py $T/test_boo_fpt.py $T/test_boo_nnzflat_fpt.py \
            --boo-require-cuda -m boo_sanitizer -q > "$S0/$TOOL.log" 2>&1 \
            && echo "$TOOL PASS: $(tail -1 "$S0/$TOOL.log")" || { tail -40 "$S0/$TOOL.log"; exit 1; }
    done
    CUDA_VISIBLE_DEVICES= python -m pytest $T/test_boo_nnzflat_fpt.py --boo-require-cuda -q \
        > "$S0/cuda-required.log" 2>&1 && { echo "no-GPU gate unexpectedly passed"; exit 1; } || echo "no-GPU gate failed as expected"
    python scripts/research_panel/nnzflat_stage0.py launch --out "$S0" | tee "$S0/launch.log"
    "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
        --capture-range-end=stop --export=sqlite -o "$S0/sequence_nsys" --force-overwrite true \
        python scripts/research_panel/nnzflat_stage0.py sequence --out "$S0" > "$S0/nsys.log" 2>&1
    python scripts/research_panel/nnzflat_stage0.py audit --out "$S0" --sqlite "$S0/sequence_nsys.sqlite"
    SO=$(python -c 'import torch, torch_fast_spdmm_core as c; print(c.__file__)')
    cuobjdump -sass "$SO" > "$S0/all.sass"
    python scripts/research_panel/nnzflat_stage0.py sass --out "$S0" --sass "$S0/all.sass" | tee "$S0/sass.log"
    gzip -f "$S0/all.sass"
    echo "NNZFLAT_STAGE0_DONE"
    ;;
prep)
    export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
    INPUTS=(syn_balanced syn_skewed pwtk webbase roadnet fem2d fem2d_permuted diag_b1.0 diag_dd0.05
            diag_skewed fem2d_1m fem2d_permuted_1m diag_b1.0_1m diag_dd0.05_1m diag_skewed_1m)
    NAME=${INPUTS[$SLURM_ARRAY_TASK_ID]}
    python scripts/research_panel/nnzflat_prep.py --root "$RUN" --cache "$SPMM_MATRIX_CACHE" \
        --source-labels "$SOURCE_LABELS" --input "$NAME" --features 128 256 512
    ;;
manifest)
    export CUDA_VISIBLE_DEVICES=
    python scripts/research_panel/nnzflat_manifest.py --root "$RUN" --stage 1
    python scripts/research_panel/nnzflat_manifest.py --root "$RUN" --stage 2
    ;;
screen1)
    gpu_env
    python -m benchmark.bench_nnzflat --root "$RUN" --manifest "$RUN/manifests/stage1.json" --task 0 \
        --out "$RUN/results/stage1/task_000" --cache "$SPMM_MATRIX_CACHE" --allocation "s1-${SLURM_JOB_ID}"
    ;;
screen2)
    gpu_env
    I=${SLURM_ARRAY_TASK_ID:?}
    python -m benchmark.bench_nnzflat --root "$RUN" --manifest "$RUN/manifests/stage2.json" --task "$I" \
        --out "$RUN/results/stage2/task_$(printf %03d "$I")" --cache "$SPMM_MATRIX_CACHE" \
        --allocation "s2-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$I"
    ;;
finalize)
    export CUDA_VISIBLE_DEVICES=
    python scripts/research_panel/nnzflat_analyze.py --root "$RUN" audit
    python scripts/research_panel/nnzflat_analyze.py --root "$RUN" freeze
    ;;
confirm)
    gpu_env
    A=$((SLURM_ARRAY_TASK_ID + 1))
    M="$RUN/manifests/stage3.json"
    N=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['tasks']))" "$M")
    for ((t = 0; t < N; t++)); do
        python -m benchmark.bench_nnzflat --root "$RUN" --manifest "$M" --task "$t" \
            --out "$RUN/results/stage3/alloc$A/task_$(printf %03d "$t")" --cache "$SPMM_MATRIX_CACHE" \
            --allocation "alloc$A-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
            --order-seed-offset "$A"
    done
    ;;
confirm_supp)
    # Supplementary confirmation cell(s), fresh process per allocation (MANIFEST_NAME, OUT_NAME).
    gpu_env
    A=$((SLURM_ARRAY_TASK_ID + 1))
    M="$RUN/manifests/${MANIFEST_NAME:?}.json"
    python -m benchmark.bench_nnzflat --root "$RUN" --manifest "$M" --task 0 \
        --out "$RUN/results/${OUT_NAME:?}/alloc$A/task_000" --cache "$SPMM_MATRIX_CACHE" \
        --allocation "supp$A-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
        --order-seed-offset "$((A + 10))"
    ;;
launchaudit)
    gpu_env
    STAGE=${2:?stage}
    M="$RUN/manifests/stage$STAGE.json"
    python - "$M" <<'EOF2' > "$RUN/logs/launchaudit_cells_s$STAGE.txt"
import json, sys
m = json.load(open(sys.argv[1]))
for t in m['tasks']:
    for c in range(len(t['cells'])):
        print(t['task'], c)
EOF2
    while read -r T C; do
        D="$RUN/launch_audit/stage$STAGE/task_$(printf %03d "$T")_cell$C"
        mkdir -p "$D"
        "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
            --capture-range-end=stop --export=sqlite -o "$D/trace" --force-overwrite true \
            python scripts/research_panel/nnzflat_launch_audit.py run --root "$RUN" --manifest "$M" \
            --task "$T" --cell "$C" --cache "$SPMM_MATRIX_CACHE" --out "$D" > "$D/run.log" 2>&1
        python scripts/research_panel/nnzflat_launch_audit.py check --out "$D" --sqlite "$D/trace.sqlite"
    done < "$RUN/logs/launchaudit_cells_s$STAGE.txt"
    ;;
profile)
    gpu_env
    RUN="$RUN" NSYS="$NSYS" bash scripts/research_panel/nnzflat_profile.sh
    ;;
report)
    export CUDA_VISIBLE_DEVICES=
    python scripts/research_panel/nnzflat_analyze.py --root "$RUN" report
    ;;
*)
    echo "unknown mode $MODE" >&2
    exit 2
    ;;
esac
echo "done mode=$MODE $(date -Is)"
