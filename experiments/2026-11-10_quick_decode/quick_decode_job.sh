#!/bin/bash
# Quick A/B for "decode the smem_X column offset once per CTA" (BOO rowseg). One GPU job:
# A = validated source, B = A + the edit; both rebuilt here with one toolchain.
set -eo pipefail
source ~/.bashrc
conda activate fem
set -u
Q=/home/dayanb/research/quick-decode-20261110
SRC=/home/dayanb/research/kc-p-20261110
RUN=$SRC/shape_run                      # labels only (read)
LABELS=/home/dayanb/research/eran-fpt-20261013/labels
ART=/home/dayanb/research/nnzflat-20261014/nnzflat_run
ORIG_SHA=38f37f959f3010df7296246a5b05b81e1e74898e44f22d3b506e0fc63dc43226
EDIT_SHA=33c39b926091c775cda9272413ebe9c8bf639bbc51145a446be7a1a262a9dc21
export CUDA_HOME="$CONDA_PREFIX" CC=$(which x86_64-conda-linux-gnu-cc) CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX TORCH_CUDA_ARCH_LIST=8.6
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export SPMM_MATRIX_CACHE=/home/dayanb/research/real-matrices-20261008/matrix_cache
mkdir -p "$Q/logs"
echo "node=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,uuid --format=csv,noheader

use() { export PYTHONPATH="$Q/$1/deps:$Q/$1/torch_fast_spdmm:$Q/$1/tests/torch_fast_spdmm:$Q/$1"; cd "$Q/$1"; }

# 1. snapshots
for V in A B; do
    [ -d "$Q/$V" ] || rsync -a --exclude shape_run --exclude build "$SRC/" "$Q/$V/"
done
cp "$Q/boo_kernel.edited.cu" "$Q/B/torch_fast_spdmm/kernels/boo_kernel.cu"
echo "$ORIG_SHA  $Q/A/torch_fast_spdmm/kernels/boo_kernel.cu" | sha256sum -c -
echo "$EDIT_SHA  $Q/B/torch_fast_spdmm/kernels/boo_kernel.cu" | sha256sum -c -

# 2. build both with the same command
for V in A B; do
    (cd "$Q/$V/torch_fast_spdmm" && rm -rf build torch_fast_spdmm_core*.so \
        && python setup.py build_ext --inplace > "$Q/logs/build_$V.log" 2>&1) || { tail -30 "$Q/logs/build_$V.log"; exit 1; }
    sha256sum "$Q/$V/torch_fast_spdmm/"torch_fast_spdmm_core*.so | tee "$Q/logs/so_$V.sha256"
done

# 3. correctness: full BOO suite on both; sanitizers on B
TESTS="tests/torch_fast_spdmm/test_boo.py tests/torch_fast_spdmm/test_boo_fpt.py tests/torch_fast_spdmm/test_boo_edges.py tests/torch_fast_spdmm/test_boo_shape.py tests/torch_fast_spdmm/test_kcp_shape.py"
for V in A B; do
    use $V
    python -m pytest $TESTS --boo-require-cuda -q -p no:cacheprovider 2>&1 | tail -3 | tee "$Q/logs/tests_$V.txt"
done
use B
for TOOL in memcheck racecheck synccheck; do
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool "$TOOL" --error-exitcode 1 \
        python -m pytest tests/torch_fast_spdmm/test_boo_shape.py tests/torch_fast_spdmm/test_kcp_shape.py \
        --boo-require-cuda -m boo_sanitizer -q -p no:cacheprovider 2>&1 | tail -2 | tee "$Q/logs/sanitizer_B_$TOOL.txt"
done

# 4. timing, alternating A/B on this GPU (round r: AB if r odd, BA if even)
for R in 1 2 3; do
    ORDER="A B"; [ $((R % 2)) -eq 0 ] && ORDER="B A"
    for V in $ORDER; do
        use $V
        for T in 0 1; do
            OUT="$Q/results/quick_decode/alloc${V}r$R/task_$(printf %03d $T)"
            [ -f "$OUT/summary.csv" ] && continue
            rm -rf "$OUT"
            python -m benchmark.bench_shape --artifact-root "$ART" --run "$RUN" --source-labels "$LABELS" \
                --manifest "$Q/quick.json" --task $T --out "$OUT" --cache "$SPMM_MATRIX_CACHE" \
                --allocation "quick-$V-r$R" --order-seed-offset $R > "$Q/logs/bench_${V}_r${R}_t$T.log" 2>&1 \
                || { tail -20 "$Q/logs/bench_${V}_r${R}_t$T.log"; exit 1; }
        done
    done
done

# 5. ncu profiles of both
NCU=$CONDA_PREFIX/nsight-compute-2025.2.1/ncu
for V in A B; do
    use $V
    mkdir -p "$Q/profile/$V"
    while read -r LABEL CELL ARM; do
        [ -z "$LABEL" ] && continue
        STEM="$Q/profile/$V/$LABEL"
        [ -f "${STEM}_raw.csv" ] && continue
        "$NCU" --profile-from-start off --target-processes all --set full --import-source no -o "$STEM" -f \
            python scripts/research_panel/shape_profile.py --artifact-root "$ART" --run "$RUN" \
            --source-labels "$LABELS" --manifest "$Q/quick.json" --cell "$CELL" --arm "$ARM" \
            --cache "$SPMM_MATRIX_CACHE" --out "${STEM}_meta" > "$STEM.log" 2>&1
        "$NCU" --import "$STEM.ncu-rep" --csv --page raw > "${STEM}_raw.csv"
    done < "$Q/profile_quick.txt"
done
echo "QUICK_DECODE_DONE $(date -Is)"
