#!/bin/bash
#SBATCH --job-name=ncu_smem_block_data
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=jobs/profile_smem_block_data/logs/%j.out
#SBATCH --error=jobs/profile_smem_block_data/logs/%j.err

# Compares static shared-memory footprint / occupancy between the baseline
# `block` kernel (pbr_spmm_zero_idle_kernel, no block_data cache) on
# research/pbr-cusparse-scalability-gap and the smem_block_data variant on
# feat/pbr-cache-block-data (adds a static __shared__ s_block_data[BLOCK_ROWS
# *BLOCK_COLS] buffer -- see STALL_REASONS_CACHE_BLOCK_DATA.md). Question:
# does the extra static smem actually reduce blocks-per-SM (the mechanism
# proposed to explain that experiment's wall-clock regressions), and if so,
# is shared memory actually the binding occupancy limiter or something else?
#
# features are chosen NOT divisible by 4 (129, 517) so use_vec=false and the
# kernel dispatches to pbr_spmm_zero_idle_kernel specifically (the scalar
# kernel named in this request), not pbr_spmm_vec_kernel.
#
# Metrics are a SEPARATE ncu invocation from the project's standard
# stall-reason METRICS list (optimizer/performance.py) -- mixing extra
# counters into that list previously corrupted its numbers by changing ncu's
# replay-pass structure (see STALL_REASONS.md methodology note). This job
# only collects launch-config / occupancy metrics, plus wall time for
# reference.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

OUT="$SLURM_SUBMIT_DIR/ncu_out/smem_block_data"
mkdir -p "$OUT"

METRICS="\
gpu__time_duration.sum,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic,\
launch__shared_mem_config_size,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_warps,\
launch__occupancy_limit_blocks,\
launch__waves_per_multiprocessor,\
launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active"

MATRICES="diag_b0.3 scattered_1x8 scattered_8x1"
BLOCKS="2 4 8"
SMEMS="512 2048"
FEATURES="129 517"
SIZE=262144
MIN_NNZ=2

profile_ref () {
    local tag="$1" ref="$2"
    echo "[job] === checking out $ref (tag=$tag) ==="
    git fetch origin
    git checkout "$ref"
    git pull origin "$ref" || true
    local commit
    commit=$(git rev-parse HEAD)
    echo "[job] building $ref @ $commit"
    pip install -e torch_fast_spdmm --no-build-isolation

    cd tests/torch_fast_spdmm
    for matrix in $MATRICES; do
        for block in $BLOCKS; do
            for smem in $SMEMS; do
                for feat in $FEATURES; do
                    cfg="${tag}_${matrix}_b${block}_s${smem}_f${feat}"
                    echo "[job] profiling $cfg"
                    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
                        python -m benchmark.profile_pbr --size $SIZE --features $feat \
                            --block $block --min-nnz $MIN_NNZ --smem-size $smem \
                            --matrix-profile "$matrix" --kernel block \
                        > "$OUT/${cfg}.csv" 2> "$OUT/${cfg}.log" \
                        || { echo "  FAILED $cfg"; tail -5 "$OUT/${cfg}.log"; }
                done
            done
        done
    done
    cd "$SLURM_SUBMIT_DIR"
}

profile_ref baseline research/pbr-cusparse-scalability-gap
profile_ref smem_block_data feat/pbr-cache-block-data

echo "[job] batch complete $(date)"
