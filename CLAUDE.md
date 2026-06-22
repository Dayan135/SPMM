# SPMM — torch_fast_spdmm

Custom CUDA SpMM kernels for sparse-dense matrix multiplication using PBR (Packed Block Representation), CSR, and CSC formats.

## Remote machine requirement

**All compilation and testing must run on the remote GPU machine via SSH.**
Do not attempt to build CUDA extensions or run tests locally — they will fail without a GPU.

## Build

Full copy-paste recipe (BGU Slurm). Build **on a GPU node** — clone lives at
`/home/dayanb/research1/SPMM`:

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
cd /home/dayanb/research1/SPMM
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
| `tests/torch_fast_spdmm/test_csr.py` | CSR correctness tests |
| `tests/torch_fast_spdmm/test_csc.py` | CSC correctness tests |
| `tests/torch_fast_spdmm/test_benchmark.py` | GPU benchmark suite (W&B) |

## Commit scopes

- `kernel` — CUDA kernels (`pbr_kernel.cu`, `csr_kernels.cu`, `csc_kernels.cu`)
- `bindings` — pybind11 wrappers
- `pbr` — PBR format construction (`pbr_matrix/`)
- `tests` — test suite
- `build` — `setup.py`, `pyproject.toml`, `CMakeLists.txt`
- `docs` — README, notes
