#!/bin/bash
# Profile each line of a plan file: "case features method p fpt slab direct_alg".
# method is boo | direct | torch (p/fpt/slab ignored for baselines; use '-').
# Five SEPARATE invocations per variant (never merged, see optimizer/performance.py):
#   nsys     timeline (kernels, memset, streams, overlap, host gaps)
#   stalls   exactly optimizer.performance.METRICS
#   occ      LaunchStats + Occupancy sections + OCCUPANCY_METRICS (regs/smem/limits/spills)
#   mem      SpeedOfLight + MemoryWorkloadAnalysis(+_Tables) + SchedulerStats sections
#   instmix  executed instruction mix + shared wavefronts/bank conflicts
#   atomics  (opt-in via PASSES) global RED/LD requests vs sectors, L2 read/write sectors
# Called from job.sh profile (cwd tests/torch_fast_spdmm, env already set).
set -uo pipefail  # no -e: one failing pass is recorded, the plan continues
PLAN=${1:?plan file}
OUT=${2:?output directory}
PASSES=${PASSES:-"nsys stalls occ mem instmix"}
mkdir -p "$OUT"
NCU=${NCU:-$(command -v ncu)}
NSYS=${NSYS:-$CONDA_PREFIX/nsight-compute-2025.2.1/host/target-linux-x64/nsys}
"$NCU" --version > "$OUT/ncu-version.txt"
"$NSYS" --version > "$OUT/nsys-version.txt"
"$NCU" --list-sections > "$OUT/ncu-sections.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,clocks.max.memory --format=csv > "$OUT/gpu.txt"
cp "$PLAN" "$OUT/plan.txt"
STALLS=$(python -c 'from optimizer.performance import METRICS; print(",".join(METRICS))')
OCC=$(python -c 'from optimizer.performance import OCCUPANCY_METRICS; print(",".join(OCCUPANCY_METRICS))')
INSTMIX="gpu__time_duration.sum,smsp__inst_executed.sum,smsp__thread_inst_executed.sum,\
smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,\
smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_red.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
ATOMICS="gpu__time_duration.sum,lts__t_sectors_op_red.sum,lts__t_requests_op_red.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum"
have_section() { grep -qw "$1" "$OUT/ncu-sections.txt"; }
MEM_SECTIONS=()
for S in SpeedOfLight MemoryWorkloadAnalysis MemoryWorkloadAnalysis_Tables SchedulerStats; do
    if have_section "$S"; then MEM_SECTIONS+=(--section "$S"); else echo "missing section $S" >> "$OUT/unavailable.txt"; fi
done
OCC_SECTIONS=()
for S in LaunchStats Occupancy; do
    if have_section "$S"; then OCC_SECTIONS+=(--section "$S"); else echo "missing section $S" >> "$OUT/unavailable.txt"; fi
done

while read -r -u 3 CASE F METHOD P FPT SLAB DALG; do
    [[ -z "$CASE" || "$CASE" == \#* ]] && continue
    if [ "$METHOD" = boo ]; then
        STEM="$OUT/${CASE}_F${F}_boo_p${P}_fpt${FPT}_s${SLAB}"
        ARGS=(--case "$CASE" --features "$F" --method boo --p "$P" --fpt "$FPT" --slab "$SLAB")
    elif [ "$METHOD" = direct ]; then
        STEM="$OUT/${CASE}_F${F}_direct_${DALG}"
        ARGS=(--case "$CASE" --features "$F" --method direct --direct-algorithm "$DALG")
    else
        STEM="$OUT/${CASE}_F${F}_torch"
        ARGS=(--case "$CASE" --features "$F" --method torch)
    fi
    ARGS+=(--labels "$TILE_LABELS" --cache "$SPMM_MATRIX_CACHE")
    for PASS in $PASSES; do
        if [ -e "${STEM}_${PASS}.done" ]; then continue; fi
        case "$PASS" in
        nsys)
            "$NSYS" profile --trace=cuda,nvtx --cuda-event-trace=false --sample=none --cpuctxsw=none \
                --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite --force-overwrite=true \
                -o "${STEM}_nsys" python -m benchmark.profile_tiles "${ARGS[@]}" --out "${STEM}_nsys_meta" \
                > "${STEM}_nsys.log" 2>&1 ;;
        stalls)
            "$NCU" --profile-from-start off --target-processes all --metrics "$STALLS" --csv \
                --log-file "${STEM}_stalls.csv" python -m benchmark.profile_tiles "${ARGS[@]}" \
                --out "${STEM}_stalls_meta" > "${STEM}_stalls.log" 2>&1 ;;
        occ)
            "$NCU" --profile-from-start off --target-processes all "${OCC_SECTIONS[@]}" --metrics "$OCC" \
                --csv --log-file "${STEM}_occ.csv" python -m benchmark.profile_tiles "${ARGS[@]}" \
                --out "${STEM}_occ_meta" > "${STEM}_occ.log" 2>&1 ;;
        mem)
            "$NCU" --profile-from-start off --target-processes all "${MEM_SECTIONS[@]}" \
                --csv --log-file "${STEM}_mem.csv" python -m benchmark.profile_tiles "${ARGS[@]}" \
                --out "${STEM}_mem_meta" > "${STEM}_mem.log" 2>&1 ;;
        instmix)
            "$NCU" --profile-from-start off --target-processes all --metrics "$INSTMIX" --csv \
                --log-file "${STEM}_instmix.csv" python -m benchmark.profile_tiles "${ARGS[@]}" \
                --out "${STEM}_instmix_meta" > "${STEM}_instmix.log" 2>&1 ;;
        atomics)
            "$NCU" --profile-from-start off --target-processes all --metrics "$ATOMICS" --csv \
                --log-file "${STEM}_atomics.csv" python -m benchmark.profile_tiles "${ARGS[@]}" \
                --out "${STEM}_atomics_meta" > "${STEM}_atomics.log" 2>&1 ;;
        esac
        STATUS=$?
        grep -q PROFILE_TILES_DONE "${STEM}_${PASS}.log" && touch "${STEM}_${PASS}.done" \
            || echo "FAILED $PASS ${STEM##*/} (status $STATUS)" | tee -a "$OUT/failures.txt"
    done
    echo "profiled ${STEM##*/}"
done 3< "$PLAN"
echo PROFILE_CASES_DONE
