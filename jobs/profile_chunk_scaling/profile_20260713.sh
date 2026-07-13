#!/bin/bash
#SBATCH --job-name=ncu_chunk_verify
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/profile_chunk_scaling/logs/%j.out
#SBATCH --error=jobs/profile_chunk_scaling/logs/%j.err

# Direct verification of WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md's H3: does
# reducing ONLY the CTA count (chunk kernel: identical per-block body to
# 'block', just chunk_size PBR blocks per CTA via a #pragma-unrolled loop,
# no smem reuse) close the gap to cusparse at scale, without any of the
# extra reuse/sync machinery rowgroup/colgroup/tile also add?
#
# Sweeps chunk_size in {2,4,8,16,32,64} vs block vs cusparse on the same
# diag_b0.3 matrix at N=1024 (GPU has slack, block wins at f64) and
# N=262144 (GPU saturated, block loses at both f64/f512 per H3).
#
# Prediction if H3 is right: grid.x / waves-per-SM should drop roughly
# proportionally to chunk_size, and gpu__time_duration should improve
# monotonically toward cusparse's csrmm_alg2 numbers as chunk_size grows,
# with NO change in DRAM traffic pattern (same per-block work, just fewer
# launches) -- confirming the effect is occupancy/launch-geometry, not
# something else.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/chunk_scaling"
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
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct"

# profile_pbr TAG KERNEL SIZE FEATURES [CHUNK_SIZE]
profile_pbr() {
    local tag=$1 kernel=$2 size=$3 features=$4 chunk=${5:-8}
    local extra=()
    [ "$kernel" = "chunk" ] && extra=(--chunk-size "$chunk")
    echo "[job] profiling $tag (kernel=$kernel size=$size features=$features chunk=$chunk)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_pbr \
            --size "$size" --features "$features" \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile diag_b0.3 --kernel "$kernel" "${extra[@]}" \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

# profile_cusparse TAG SIZE FEATURES
profile_cusparse() {
    local tag=$1 size=$2 features=$3
    echo "[job] profiling $tag (cusparse size=$size features=$features)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_cusparse \
            --size "$size" --features "$features" --matrix-profile diag_b0.3 \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

for size in 1024 262144; do
    for feat in 64 512; do
        profile_pbr diag_b0.3_${size}_f${feat}_block block "$size" "$feat"
        for chunk in 2 4 8 16 32 64; do
            profile_pbr diag_b0.3_${size}_f${feat}_chunk${chunk} chunk "$size" "$feat" "$chunk"
        done
        profile_cusparse diag_b0.3_${size}_f${feat}_cusparse "$size" "$feat"
    done
done

echo "[job] all profiles done $(date)"
