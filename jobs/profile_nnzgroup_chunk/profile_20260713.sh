#!/bin/bash
#SBATCH --job-name=ncu_nnz_chunk
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/profile_nnzgroup_chunk/logs/%j.out
#SBATCH --error=jobs/profile_nnzgroup_chunk/logs/%j.err

# Step 1 of the nnz-balanced grouping experiment: does switching chunk's
# grouping criterion from 'count' (H4's original sweep, jobs/profile_chunk_
# scaling/) to 'nnz' -- everything else identical, same zero-reuse kernel
# body -- close any of the gap to cuSPARSE that count-balancing alone could
# not? Mirrors jobs/profile_chunk_scaling/profile_20260713.sh exactly, adding
# a grouping=nnz variant at each chunk_size for direct comparison.
#
# All runs: diag_b0.3, size in {1024, 262144}, features in {64, 512},
# block=8, min_nnz=2, smem=2048.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/nnzgroup_chunk"
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

# profile_pbr TAG KERNEL SIZE FEATURES [CHUNK_SIZE GROUPING]
profile_pbr() {
    local tag=$1 kernel=$2 size=$3 features=$4 chunk=${5:-8} grouping=${6:-count}
    local extra=()
    [ "$kernel" = "chunk" ] && extra=(--chunk-size "$chunk" --grouping "$grouping")
    echo "[job] profiling $tag (kernel=$kernel size=$size features=$features chunk=$chunk grouping=$grouping)"
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
            profile_pbr diag_b0.3_${size}_f${feat}_chunk${chunk}_count chunk "$size" "$feat" "$chunk" count
            profile_pbr diag_b0.3_${size}_f${feat}_chunk${chunk}_nnz   chunk "$size" "$feat" "$chunk" nnz
        done
        profile_cusparse diag_b0.3_${size}_f${feat}_cusparse "$size" "$feat"
    done
done

echo "[job] all profiles done $(date)"
