#!/bin/bash
#SBATCH --job-name=racecheck_nosyncAll_fullgrid
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=jobs/racecheck_nosync_fullgrid/logs/%j_nosyncAll.out
#SBATCH --error=jobs/racecheck_nosync_fullgrid/logs/%j_nosyncAll.err

# Full-grid compute-sanitizer racecheck sweep for scalar_diag variant
# nosyncAll, across all 10 matrices x 7 (block_size,min_nnz) configs
# x 6 features = 420 configs for this variant (1680 total across the 4
# nosync* variants, split one sbatch job per variant). Each of the 10
# matrices is ONE compute-sanitizer process (racecheck_probe_batch.py loops
# the 42 (block,min_nnz,features) sub-configs internally per process) -- 10
# processes per job, 40 total across all 4 jobs, instead of 1680 fresh
# processes.
#
# See STALL_REASONS_NOSYNC_RACECHECK_FULLGRID.md for the writeup and
# racecheck_probe_batch.py's docstring for the batching rationale.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME  variant=nosyncAll"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "[job] building"
pip install -e torch_fast_spdmm --no-build-isolation

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/racecheck_nosync_fullgrid"
mkdir -p "$OUT"

VARIANT="nosyncAll"
MATRICES="diag_b0.3 diag_b1.0 scattered_1x8_d0.5pct scattered_1x8_d2pct scattered_8x1_d0.5pct scattered_8x1_d2pct scattered_2x2_d0.5pct scattered_2x2_d2pct scattered_2x4_d0.5pct scattered_2x4_d2pct"

for matrix in $MATRICES; do
    tag="${VARIANT}_${matrix}"
    echo "[job] racecheck $tag (42 sub-configs)"
    compute-sanitizer --tool racecheck python -m benchmark.racecheck_probe_batch \
        --variant "$VARIANT" --matrix "$matrix" \
        > "$OUT/${tag}.log" 2>&1 \
        || { echo "  FAILED $tag"; tail -20 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
done

echo "[job] === SUMMARY ==="
for f in "$OUT"/${VARIANT}_*.log; do
    tag=$(basename "$f" .log)
    line=$(grep "RACECHECK SUMMARY" "$f" || echo "NO SUMMARY LINE")
    echo "$tag : $line"
done

echo "[job] batch complete $(date)"
