#!/bin/bash
# Block-shape experiment runner (experiments/2026-11-06_block_shape). ROOT is an isolated
# source snapshot holding the validated 2026-10-14 extension; RUN=$ROOT/shape_run holds results.
#
# CPU modes (cpu partition): gate (array over inputs)
# GPU modes (one RTX3090, <= 48 GB, no explicit CPU request, OMP_NUM_THREADS=1):
#   stage0 | sanity | screen (array 0-15) | occupancy (array 0-23) | kp (array 0-15) | audit
#   confirm_sanity | confirm (array 0-10, ALLOC=1|2|3; submit each allocation on a disjoint node set) | profile
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
: "${ROOT:?Set ROOT to the isolated block-shape snapshot}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm:$ROOT"
export PYTHONUNBUFFERED=1
export SPMM_MATRIX_CACHE=/home/dayanb/research/real-matrices-20261008/matrix_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
SOURCE_LABELS=/home/dayanb/research/eran-fpt-20261013/labels
NNZFLAT_RUN=/home/dayanb/research/nnzflat-20261014/nnzflat_run
NNZFLAT_BOO="$NNZFLAT_RUN/artifacts/boo"
EXTENSION_SHA256=9d4318d5149ce732a5dc38d8c3e0bd9a507491561069b956563cb0d21214f1cd
RUN="$ROOT/shape_run"
MODE=${1:?mode}
mkdir -p "$RUN/logs"
cd "$ROOT"
echo "mode=$MODE job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_TASK_ID:-} node=$(hostname) $(date -Is)"

SO="$ROOT/torch_fast_spdmm/torch_fast_spdmm_core.cpython-312-x86_64-linux-gnu.so"
echo "$EXTENSION_SHA256  $SO" | sha256sum -c -

gpu_env() {
    export OMP_NUM_THREADS=1
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
}

# bench <manifest> <task> <out> <allocation> <order-seed-offset>
bench() {
    python -m benchmark.bench_shape --artifact-root "$NNZFLAT_RUN" --run "$RUN" \
        --source-labels "$SOURCE_LABELS" --manifest "$RUN/manifests/$1.json" --task "$2" --out "$3" \
        --cache "$SPMM_MATRIX_CACHE" --allocation "$4" --order-seed-offset "$5"
}

case "$MODE" in
gate)
    CUDA_VISIBLE_DEVICES= python scripts/research_panel/shape_gate.py \
        --run "$RUN" --source-labels "$SOURCE_LABELS" --nnzflat-artifacts "$NNZFLAT_BOO" \
        --cache "$SPMM_MATRIX_CACHE"
    ;;
stage0)
    gpu_env
    S0="$RUN/stage0"
    mkdir -p "$S0"
    T=tests/torch_fast_spdmm
    python -m pytest $T/test_boo_shape.py --boo-require-cuda -ra -q \
        --junitxml "$S0/shape-results.xml" > "$S0/shape.log" 2>&1 \
        && echo "shape tests PASS: $(tail -1 "$S0/shape.log")" || { tail -40 "$S0/shape.log"; exit 1; }
    for TOOL in memcheck racecheck synccheck; do
        PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool $TOOL --error-exitcode 1 \
            python -m pytest $T/test_boo_shape.py --boo-require-cuda -m boo_sanitizer -q > "$S0/$TOOL.log" 2>&1 \
            && echo "$TOOL PASS: $(tail -1 "$S0/$TOOL.log")" || { tail -40 "$S0/$TOOL.log"; exit 1; }
    done
    echo "SHAPE_STAGE0_DONE"
    ;;
sanity)
    gpu_env
    bench sanity 0 "$RUN/results/sanity/task_000" "sanity-${SLURM_JOB_ID}" 0
    ;;
screen)
    # Array 0-15: eight inputs on two allocations, with independent shuffles.
    gpu_env
    I=${SLURM_ARRAY_TASK_ID:?}
    A=$((I / 8 + 1))
    TASK=$((I % 8))
    bench screen "$TASK" "$RUN/results/screen/alloc$A/task_$(printf %03d "$TASK")" \
        "screen$A-${SLURM_ARRAY_JOB_ID}_$I" "$A"
    ;;
occupancy)
    # Array 0-23: eight inputs on three allocations, with independent shuffles.
    gpu_env
    I=${SLURM_ARRAY_TASK_ID:?}
    A=$((I / 8 + 1))
    TASK=$((I % 8))
    bench occupancy "$TASK" "$RUN/results/occupancy/alloc$A/task_$(printf %03d "$TASK")" \
        "occ$A-${SLURM_ARRAY_JOB_ID}_$I" "$A"
    ;;
kp)
    # Array 0-15: eight inputs on two allocations; P16-P128 in the same cells.
    gpu_env
    I=${SLURM_ARRAY_TASK_ID:?}
    A=$((I / 8 + 1))
    TASK=$((I % 8))
    bench kp "$TASK" "$RUN/results/kp/alloc$A/task_$(printf %03d "$TASK")" \
        "kp$A-${SLURM_ARRAY_JOB_ID}_$I" "$((A + 20))"
    ;;
confirm_sanity)
    # Reduced copy of confirm task 0's F128 cells (orig + reordered), 1 warmup, 3 repeats.
    gpu_env
    python - "$RUN/manifests/confirm.json" "$RUN/manifests/confirm_sanity.json" <<'EOF2'
import json, sys
m = json.load(open(sys.argv[1]))
t = m['tasks'][0]
m.update(stage='confirm_sanity', warmup=1, repeats=3,
         tasks=[dict(t, task=0, cells=[c for c in t['cells'] if c['features'] == 128])])
json.dump(m, open(sys.argv[2], 'w'), indent=1)
EOF2
    bench confirm_sanity 0 "$RUN/results/confirm_sanity/task_000_${SLURM_JOB_ID}" "csan-${SLURM_JOB_ID}" 0
    ;;
confirm)
    # Array 0-10 over inputs; ALLOC=1|2|3 is one independent allocation (own shuffle seeds).
    gpu_env
    : "${ALLOC:?Set ALLOC=1|2|3}"
    I=${SLURM_ARRAY_TASK_ID:?}
    bench confirm "$I" "$RUN/results/confirm/alloc$ALLOC/task_$(printf %03d "$I")" \
        "confirm$ALLOC-${SLURM_ARRAY_JOB_ID}_$I" "$((ALLOC + 30))"
    ;;
audit)
    # Nsight Systems launch audit of every cell with F >= AUDIT_MIN_F (default 512) in STAGES
    # (default: occupancy screen kp).
    gpu_env
    NSYS="$CONDA_PREFIX/nsight-compute-2025.2.1/host/target-linux-x64/nsys"
    for STAGE in ${STAGES:-occupancy screen kp}; do
        M="$RUN/manifests/$STAGE.json"
        python - "$M" "${AUDIT_MIN_F:-512}" <<'EOF2' > "$RUN/logs/audit_cells_$STAGE.txt"
import json, sys
m = json.load(open(sys.argv[1]))
for t in m['tasks']:
    for c, cell in enumerate(t['cells']):
        if cell['features'] >= int(sys.argv[2]):
            print(t['task'], c)
EOF2
        while read -r T C; do
            D="$RUN/launch_audit/$STAGE/task_$(printf %03d "$T")_cell$C"
            mkdir -p "$D"
            "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
                --capture-range-end=stop --export=sqlite -o "$D/trace" --force-overwrite true \
                python scripts/research_panel/shape_launch_audit.py run --artifact-root "$NNZFLAT_RUN" \
                --run "$RUN" --source-labels "$SOURCE_LABELS" --manifest "$M" --task "$T" --cell "$C" \
                --cache "$SPMM_MATRIX_CACHE" --out "$D" > "$D/run.log" 2>&1 || { tail -20 "$D/run.log"; exit 1; }
            python scripts/research_panel/shape_launch_audit.py check --out "$D" --sqlite "$D/trace.sqlite"
            rm -f "$D/trace.nsys-rep"
        done < "$RUN/logs/audit_cells_$STAGE.txt"
    done
    echo "SHAPE_AUDIT_DONE"
    ;;
profile)
    # Matched-shape Nsight Compute profiles. Reads $RUN/manifests/profile_confirm.txt: one
    # "<label> <cell> <arm>" per line (# comments). One process and one full ncu capture per arm,
    # all kernels in the capture range (BOO main, CSR remainder, cuSPARSE internals).
    gpu_env
    NCU=${NCU:-$(command -v ncu || echo "$CONDA_PREFIX/nsight-compute-2025.2.1/ncu")}
    OUT="$RUN/profile/confirm"
    mkdir -p "$OUT"
    "$NCU" --version > "$OUT/ncu-version.txt"
    grep -v '^#' "$RUN/manifests/profile_confirm.txt" | while read -r LABEL CELL ARM; do
        [ -z "$LABEL" ] && continue
        STEM="$OUT/$LABEL"
        if [ -f "$STEM.ncu-rep" ]; then echo "keep $LABEL"; continue; fi
        echo "profile $LABEL $CELL $ARM $(date -Is)"
        "$NCU" --profile-from-start off --target-processes all --set full --import-source no \
            -o "$STEM" -f python scripts/research_panel/shape_profile.py --artifact-root "$NNZFLAT_RUN" \
            --run "$RUN" --source-labels "$SOURCE_LABELS" --manifest "$RUN/manifests/confirm.json" \
            --cell "$CELL" --arm "$ARM" --cache "$SPMM_MATRIX_CACHE" --out "${STEM}_meta" \
            > "$STEM.log" 2>&1 || { tail -20 "$STEM.log"; continue; }
        "$NCU" --import "$STEM.ncu-rep" --csv --page raw > "${STEM}_raw.csv"
    done
    echo "SHAPE_PROFILE_DONE"
    ;;
*)
    echo "unknown mode $MODE" >&2
    exit 2
    ;;
esac
echo "done $(date -Is)"
