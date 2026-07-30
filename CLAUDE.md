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
| `torch_fast_spdmm/kernels/pbr_kernel.cu` | PBR SpMM CUDA kernel (`block`: scalar `zero_idle` + vectorized variants) |
| `torch_fast_spdmm/kernels/pbr_kernels.hpp` | Launch-function + occupancy-query declarations for the PBR kernels |
| `torch_fast_spdmm/kernels/kernels.hpp` | CUDA error-check utilities |
| `torch_fast_spdmm/kernels/csr_kernels.cu` | Naive CSR SpMM kernel (benchmark baseline / PBR remainder) |
| `torch_fast_spdmm/kernels/csr_kernels.hpp` | Declares `spdmm_csr_naive` |
| `torch_fast_spdmm/pbr_matrix/pbr_matrix.cpp` | PBR format CPU implementations |
| `torch_fast_spdmm/pbr_matrix/pbr_matrix.hpp` | PBR format data structures |
| `torch_fast_spdmm/bindings/bindings_kernels.cu` | pybind11 CUDA kernel wrappers |
| `torch_fast_spdmm/bindings/bindings.cpp` | pybind11 module entry point |
| `torch_fast_spdmm/torch_fast_spdmm/__init__.py` | Python API |
| `torch_fast_spdmm/torch_fast_spdmm/py.typed` | PEP 561 marker |
| `torch_fast_spdmm/setup.py` | Editable-install build (the live build path — see Build above) |
| `torch_fast_spdmm/pyproject.toml` | Package metadata / deps |
| `torch_fast_spdmm/CMakeLists.txt` | Alternate CMake build path |
| `tests/torch_fast_spdmm/test_pbr.py` | PBR correctness tests |
| `tests/torch_fast_spdmm/test_csr.py` | CSR-naive correctness tests |
| `tests/torch_fast_spdmm/test_sparse_block_mat.py` | Tests using the `torch.sparse_csr` fixture |
| `tests/torch_fast_spdmm/conftest.py` | Self-contained pytest fixtures |
| `tests/torch_fast_spdmm/test_utils.py` | dtype/format conversion helpers (numpy/torch/scipy) |
| `tests/torch_fast_spdmm/cuda_benchmark.py` | CUDA-aware subclass of pytest-benchmark's fixture |
| `tests/torch_fast_spdmm/test_benchmark.py` | GPU benchmark suite (W&B) |
| `tests/torch_fast_spdmm/benchmark/matrices.py` | Synthetic sparse matrix generators |
| `tests/torch_fast_spdmm/benchmark/plot_benchmarks.py` | Pulls W&B tables, renders comparison plots/heatmaps |
| `tests/torch_fast_spdmm/benchmark/profile_pbr.py` | ncu profiling driver (scattered + diag) |
| `tests/torch_fast_spdmm/benchmark/profile_cusparse.py` | ncu profiling driver for the cuSPARSE baseline |
| `tests/torch_fast_spdmm/benchmark/csr_kernel_sweep.py` | CSR-kernel-fast vs fallback head-to-head experiment |

## Current kernel state

One PBR kernel family (`block`), dispatched from `pbr_matmul`/`pbr_batched_matmul_cuda`:

- **`pbr_spmm_zero_idle_kernel`** (scalar) and **`pbr_spmm_vec_kernel`** (vectorized, `float4`/`double2`
  lanes) — one CUDA block per PBR block, `atomicAdd` to Y, concurrent with the CSR remainder kernel on a
  separate stream. Both are structured as three steps per feature tile: Step A loads `X` into `smem_X`,
  Step B computes into `smem_Y` (one thread per (row, feature) work-item), Step C flushes `smem_Y` to
  global `Y` via `atomicAdd`.

Earlier `rowgroup`/`colgroup`/`tile`/`chunk` kernel variants (grouping PBR blocks into fewer, "fatter"
CUDA blocks to amortize `smem_X`/`smem_Y` reuse or atomic frequency) were investigated and removed —
none beat `block` in the end; see git history prior to this cleanup for the investigation record.

## Commit scopes

- `kernel` — CUDA kernels (`pbr_kernel.cu`, `csr_kernels.cu`)
- `bindings` — pybind11 wrappers
- `pbr` — PBR format construction (`pbr_matrix/`)
- `tests` — test suite
- `build` — `setup.py`, `pyproject.toml`, `CMakeLists.txt`
- `docs` — README, notes
