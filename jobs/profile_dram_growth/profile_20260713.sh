#!/bin/bash
#SBATCH --job-name=ncu_dram_growth
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/profile_dram_growth/logs/%j.out
#SBATCH --error=jobs/profile_dram_growth/logs/%j.err

# WHY does DRAM traffic grow with chunk/group size for IDENTICAL total work?
# Across every ablation so far (H4's count-balanced sweep, and this branch's
# count-vs-nnz rerun on both diag_b0.3 and diag_skewed), gpu__time_duration
# stays roughly flat/slightly worse and dram__bytes_read/write climb sharply
# as chunk_size grows (e.g. diag_b0.3 f512/262144: 1099MB @ chunk=1 (block) ->
# 8143MB @ chunk=64, 7.4x) -- identical regardless of grouping criterion
# (count vs nnz) or whether the loop is #pragma-unrolled or a plain runtime
# loop. Something about coarser per-CTA grouping itself is the cause, not
# grouping quality.
#
# Adds two new counter families to isolate WHERE the extra DRAM traffic goes:
#   l1tex__t_bytes_pipe_lsu_mem_local_op_{ld,st}.sum -- LOCAL memory (register
#     spill) traffic specifically. If chunk_size correlates with rising local
#     memory bytes, that's spilling, not global-memory access pattern decay.
#   l1tex__t_sector_hit_rate.pct / lts__t_sector_hit_rate.pct (already have
#     these) -- falling L1/L2 hit rate as chunk grows would point to cache
#     locality degradation instead (wider per-CTA working set, less reuse).
#   launch__registers_per_thread / launch__local_mem_per_thread_static --
#     direct register/local-mem allocation size per thread.
#
# All runs: diag_b0.3, size=262144 (where the effect is largest), f512 only
# (largest gap), chunk_size sweep 1(=block)/2/4/8/16/32/64, grouping='count'
# only (grouping doesn't matter per the prior ablation).

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/dram_growth"
mkdir -p "$OUT"

METRICS="\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,\
launch__registers_per_thread,launch__shared_mem_per_block_static,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct"

profile_pbr() {
    local tag=$1 kernel=$2 chunk=${3:-8}
    local extra=()
    [ "$kernel" = "chunk" ] && extra=(--chunk-size "$chunk" --grouping count)
    echo "[job] profiling $tag (kernel=$kernel chunk=$chunk)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_pbr \
            --size 262144 --features 512 \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile diag_b0.3 --kernel "$kernel" "${extra[@]}" \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

profile_pbr diag_b0.3_262144_f512_block block
for chunk in 2 4 8 16 32 64; do
    profile_pbr diag_b0.3_262144_f512_chunk${chunk} chunk "$chunk"
done

echo "[job] all profiles done $(date)"
