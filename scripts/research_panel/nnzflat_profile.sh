#!/bin/bash
# Profiling subset for the nnzflat experiment (run via nnzflat_job.sh profile, GPU node).
# Reads $RUN/manifests/profile.txt: one "input k features arm" per line (# comments).
# Per configuration: nsys timeline of 20 complete calls (per-phase kernel times),
# one full Nsight Compute capture of 1 call (all sections, main + remainder kernels),
# and its raw-page CSV export. Unprofiled bench_nnzflat timings decide performance.
set -eo pipefail
: "${RUN:?}" "${NSYS:?}"
LIST="$RUN/manifests/profile.txt"
OUT="$RUN/profile"
mkdir -p "$OUT"
ncu --version > "$OUT/ncu-version.txt"
"$NSYS" --version > "$OUT/nsys-version.txt"
grep -v '^#' "$LIST" | while read -r INPUT K F ARM; do
    [ -z "$INPUT" ] && continue
    STEM="$OUT/${INPUT}_k${K}_F${F}_${ARM}"
    if [ -f "$STEM.ncu-rep" ]; then echo "keep $STEM"; continue; fi
    echo "profile $INPUT k$K F$F $ARM $(date -Is)"
    ARGS=(--root "$RUN" --input "$INPUT" --k "$K" --features "$F" --arm "$ARM" --cache "$SPMM_MATRIX_CACHE")
    "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
        --capture-range-end=stop --export=sqlite -o "${STEM}_nsys" --force-overwrite true \
        python -m benchmark.profile_nnzflat "${ARGS[@]}" --calls 20 --out "${STEM}_nsys_meta" \
        > "${STEM}_nsys.log" 2>&1
    "$NSYS" stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --format csv \
        -o "${STEM}_nsys" "${STEM}_nsys.sqlite" > /dev/null 2>&1 || true
    ncu --profile-from-start off --target-processes all --set full --import-source no \
        --kernel-name regex:'boo_spmm|csr' -o "$STEM" -f \
        python -m benchmark.profile_nnzflat "${ARGS[@]}" --calls 1 --out "${STEM}_ncu_meta" \
        > "${STEM}_ncu.log" 2>&1
    ncu --import "$STEM.ncu-rep" --csv --page raw > "${STEM}_raw.csv"
    ncu --import "$STEM.ncu-rep" --csv --page details > "${STEM}_details.csv"
done
echo "NNZFLAT_PROFILE_DONE"
