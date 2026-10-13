#!/bin/bash
# Submit with explicit CPU/GPU Slurm resources. ROOT is an isolated snapshot.
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
export OMP_NUM_THREADS=8 MAX_JOBS=4 PYTHONUNBUFFERED=1
MODE=${1:?Expected build, cluster, smoke, benchmark or profile}
mkdir -p "$ROOT/results"
cd "$ROOT"
if [ "$MODE" = build ]; then
    cd torch_fast_spdmm
    python setup.py build_ext --inplace > "$ROOT/results/build.log" 2>&1
    cd "$ROOT"
    python -m pytest tests/torch_fast_spdmm/benchmark/test_research_panel.py \
        tests/torch_fast_spdmm/test_boo.py tests/torch_fast_spdmm/test_cusparse_direct.py \
        --boo-require-cuda -q > "$ROOT/results/correctness.log" 2>&1
    exit
fi
cd "$ROOT/tests/torch_fast_spdmm"
if [ "$MODE" = cluster ]; then
    export CUDA_VISIBLE_DEVICES=
    python -m benchmark.research_panel --phase cluster --labels "$ROOT/labels" \
        --sizes "${PANEL_SIZE:-262144}" --methods identity hem --groups 32 64
elif [ "$MODE" = smoke ]; then
    CUDA_VISIBLE_DEVICES= python -m benchmark.research_panel --phase cluster \
        --labels "$ROOT/smoke_labels" --sizes 1024 --profiles fem2d diag_skewed \
        --methods identity hem --groups 32
    python -m benchmark.research_panel --phase benchmark --labels "$ROOT/smoke_labels" \
        --out "$ROOT/results/smoke" --sizes 1024 --profiles fem2d diag_skewed \
        --methods identity hem --groups 32 --features 32 128 --warmup 2 --repeats 3
elif [ "$MODE" = benchmark ]; then
    python -m pytest "$ROOT/tests/torch_fast_spdmm/benchmark/test_research_panel.py" \
        --boo-require-cuda -q > "$ROOT/results/panel-tests.log" 2>&1
    python -m benchmark.research_panel --phase benchmark --labels "$ROOT/labels" \
        --out "$ROOT/results/panel" --sizes "${PANEL_SIZE:-262144}"
elif [ "$MODE" = profile ]; then
    NCU=${NCU:-$(command -v ncu)}
    NSYS=${NSYS:-$CONDA_PREFIX/nsight-compute-2025.2.1/host/target-linux-x64/nsys}
    "$NCU" --version > "$ROOT/results/ncu-version.txt"
    "$NSYS" --version > "$ROOT/results/nsys-version.txt"
    METRICS=$(python -c 'from optimizer.performance import METRICS; print(",".join(METRICS))')
    # Initial diagnostic panel: contrasting regimes, one fixed group size/F.
    # Keep stall counters separate from memory/launch counters (see optimizer/performance.py).
    for PROFILE in fem2d diag_b1.0 pattern_row_skewed; do
        for METHOD in boo torch direct; do
            STEM="$ROOT/results/${PROFILE}_${METHOD}"
            ARGS=(--phase profile --labels "$ROOT/labels" --sizes "${PANEL_SIZE:-262144}"
                  --profiles "$PROFILE" --features 512 --methods hem --groups 32
                  --profile-method "$METHOD" --baseline-summary "$ROOT/results/panel/summary.csv")
            "$NSYS" profile --trace=cuda,nvtx --cuda-event-trace=false --sample=none --cpuctxsw=none \
                --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite \
                -o "${STEM}_nsys" python -m benchmark.research_panel "${ARGS[@]}" \
                --out "${STEM}_nsys_meta" > "${STEM}_nsys.log" 2>&1
            "$NCU" --profile-from-start off --target-processes all --metrics "$METRICS" --csv \
                --log-file "${STEM}_stalls.csv" python -m benchmark.research_panel "${ARGS[@]}" \
                --out "${STEM}_stalls_meta" > "${STEM}_stalls.log" 2>&1
            "$NCU" --profile-from-start off --target-processes all \
                --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis --csv \
                --log-file "${STEM}_memory.csv" python -m benchmark.research_panel "${ARGS[@]}" \
                --out "${STEM}_memory_meta" > "${STEM}_memory.log" 2>&1
        done
    done
else
    printf 'Unknown mode: %s\n' "$MODE" >&2
    exit 2
fi
