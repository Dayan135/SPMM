# SPMM — torch_fast_spdmm

Custom CUDA SpMM kernels for sparse-dense matrix multiplication using PBR (Packed Block Representation). A naive CSR SpMM kernel is also kept as a benchmark baseline.

## Remote machine requirement

**All compilation and testing must run on the remote GPU machine via SSH.**
Do not attempt to build CUDA extensions or run tests locally — they will fail without a GPU.

## Build

Full copy-paste recipe (BGU Slurm). Build **on a GPU node** — clone lives at
`/home/dayanb/research/SPMM`:

```bash
# 1. Get onto an rtx3090 GPU node (build needs nvcc from the env)
srun --partition=rtx3090 --account=erant --qos=erant \
     --gres=gpu:rtx_3090:1 --time=00:30:00 --pty bash -l

# 2. Activate env and set toolchain (CUDA_HOME MUST be set after activate)
conda activate fem
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
# Runtime/import: .so links the conda libstdc++ (newer than system /lib64);
# without this, `import torch_fast_spdmm` fails with `CXXABI_1.3.15 not found`.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 3. Build the editable install
cd /home/dayanb/research/SPMM
pip install -e torch_fast_spdmm --no-build-isolation
```

After any change to `.cu` or `.cpp` files, reinstall before running tests.
`LD_LIBRARY_PATH` is needed at **import/run time** too, not just for the build.

> **CUDA_HOME on the BGU Slurm cluster (`slurm.bgu.ac.il`):** the GPU nodes have
> **no `/usr/local/cuda`** — `nvcc` (12.9) ships *inside* the `fem` conda env.
> So `CUDA_HOME` must point at `$CONDA_PREFIX` (set it *after* `conda activate fem`).
> Hardcoding a path like `/usr/local/cuda-12.5` makes the editable build fail with
> `nvcc not found at '/usr/local/cuda-12.5/bin/nvcc'`.

## Running tests

```bash
pytest tests/torch_fast_spdmm/ -v
```

## Plotting (local)

Benchmark plots are generated locally from W&B using the `fem` conda env,
which has `wandb`, `pandas`, and `matplotlib` installed.

```bash
conda run -n fem python -m benchmark.plot_benchmarks \
    --project private-dayan/torch-fast-spdmm \
    --outdir plots/<experiment> \
    --runs <commit>_16384 <commit>_65536 <commit>_262144
```

Run from `tests/torch_fast_spdmm/`. If `wandb`, `pandas`, or `matplotlib`
are missing in the local `fem` env: `conda run -n fem pip install wandb pandas matplotlib`.

## Key files

| Path | Purpose |
|------|---------|
| `torch_fast_spdmm/kernels/pbr_kernel.cu` | PBR SpMM CUDA kernel |
| `torch_fast_spdmm/kernels/kernels.hpp` | CUDA error-check utilities |
| `torch_fast_spdmm/pbr_matrix/pbr_matrix.cpp` | PBR format CPU implementations |
| `torch_fast_spdmm/pbr_matrix/pbr_matrix.hpp` | PBR format data structures |
| `torch_fast_spdmm/bindings/bindings_kernels.cu` | pybind11 CUDA kernel wrappers |
| `torch_fast_spdmm/bindings/bindings.cpp` | pybind11 module entry point |
| `torch_fast_spdmm/torch_fast_spdmm/__init__.py` | Python API |
| `tests/torch_fast_spdmm/test_pbr.py` | PBR correctness tests |
| `tests/torch_fast_spdmm/test_csr.py` | CSR-naive correctness tests |
| `tests/torch_fast_spdmm/conftest.py` | Self-contained pytest fixtures |
| `tests/torch_fast_spdmm/test_benchmark.py` | GPU benchmark suite (W&B) |
| `tests/torch_fast_spdmm/benchmark/profile_pbr.py` | ncu profiling driver (scattered + diag) |
| `LAST_EXPERIMENTS.md` | Detailed log of recent experiments and findings |

## Current kernel state (as of 2026-06-29, branch `feat/pbr-blockgroup`)

Two PBR kernels exist. Choose via `kernel=` arg in `pbr_matmul`:

- **`block`** — one CUDA block per PBR block; `atomicAdd` to Y; concurrent with CSR on a separate stream.
- **`blockgroup`** — one CUDA block per block-count-balanced group; `atomicAdd` to Y; concurrent with CSR.
  `group_size = ceil(num_pbr_blocks / target_groups)` — computed at call time (no CPU scan or GPU array).
  `target_groups = sm_count × get_pbr_blockgroup_blocks_per_sm(block_rows, smem_size)` — occupancy query
  uses `dynamicSMemSize=0` (static `__shared__` only). smem_Y accumulates an entire strip before
  atomicAdd, reducing atomic pressure vs the block kernel.

`blockgroup` solves the wide-row scalability problem: it always launches `target_groups` CUDA blocks
regardless of strip count, unlike the old `rowgroup` which was capped at `num_strips`.

See `LAST_EXPERIMENTS.md` for rowgroup/block profiling data that motivated this design.

## Commit scopes

- `kernel` — CUDA kernels (`pbr_kernel.cu`, `csr_kernels.cu`)
- `bindings` — pybind11 wrappers
- `pbr` — PBR format construction (`pbr_matrix/`)
- `tests` — test suite
- `build` — `setup.py`, `pyproject.toml`, `CMakeLists.txt`
- `docs` — README, notes
