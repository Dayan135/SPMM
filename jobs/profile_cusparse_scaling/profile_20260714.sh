#!/bin/bash
#SBATCH --job-name=ncu_cusp_scale
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=jobs/profile_cusparse_scaling/logs/%j.out
#SBATCH --error=jobs/profile_cusparse_scaling/logs/%j.err

# Why does PBR's advantage over cuSPARSE erode with N for a FIXED PBR config?
#
# test_benchmark.py's full sweep showed: with a fixed 8x8/min_nnz=2/smem=2048
# 'block' config (not per-size-retuned), cusparse/block advantage DEGRADES on
# several matrices, e.g. diag_b0.3: 0.71 (1024) -> 0.48 (262144). A structural
# check (pure-Python block-density scan, no GPU needed) ruled out "more work
# falling to the CSR remainder": compressed_fraction = 1.0 at EVERY size for
# diag_b0.3 -- essentially all nnz goes through the compressed PBR path
# regardless of N, so this isn't a format-coverage problem. It has to be the
# kernel's own scaling behaviour relative to cusparse's.
#
# Profiles 'block' vs cusparse on the IDENTICAL diag_b0.3 matrix (shared
# build_matrix() in profile_pbr.py / profile_cusparse.py) at the best-relative
# size (1024) and the worst (262144), at f64 and f512, to see where the extra
# relative cost shows up: zero-init overhead vs real kernel, occupancy/wave
# count, DRAM traffic, cache hit rates.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/cusparse_scaling"
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

# profile_block TAG SIZE FEATURES
profile_block() {
    local tag=$1 size=$2 features=$3
    echo "[job] profiling $tag (block  size=$size features=$features)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_pbr \
            --size "$size" --features "$features" \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile diag_b0.3 --kernel block \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

# profile_cusparse TAG SIZE FEATURES
profile_cusparse() {
    local tag=$1 size=$2 features=$3
    echo "[job] profiling $tag (cusparse  size=$size features=$features)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_cusparse \
            --size "$size" --features "$features" \
            --matrix-profile diag_b0.3 \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

for size in 1024 262144; do
    for feat in 64 512; do
        profile_block    diag_b0.3_${size}_f${feat}_block    "$size" "$feat"
        profile_cusparse diag_b0.3_${size}_f${feat}_cusparse "$size" "$feat"
    done
done

echo "[job] all profiles done $(date)"
