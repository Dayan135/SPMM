#!/bin/bash
#SBATCH --job-name=density-bands
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
source ~/.bashrc
conda activate fem
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX="$CXX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=4
# DENSITY_DIR: an isolated git clone of feat/density-bands, already built
# in-place (setup.py build_ext --inplace, not pip install -e), set by the
# caller so this script doesn't need hand-editing per run.
: "${DENSITY_DIR:?Set DENSITY_DIR to the isolated, already-built density-bands clone}"
cd "$DENSITY_DIR"
export PYTHONPATH="$PWD/torch_fast_spdmm:$PWD/tests/torch_fast_spdmm:${PYTHONPATH:-}"
python -c "import torch_fast_spdmm as m; p=m.__file__; print(p); assert p.startswith('$DENSITY_DIR/'), p"
cd tests/torch_fast_spdmm
python -m benchmark.bench_density_bands --wandb-mode disabled --out "../../results/job$1"
