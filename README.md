# torch_fast_spdmm

CUDA library for sparse-dense matrix multiplication (SpMM). Implements three storage formats — PBR, CSR, CSC — exposed as a PyTorch extension via pybind11.

The main focus is **PBR (Packed Block Representation)**: a block-compressed sparse format that exploits repeated sparsity patterns across rows to enable coalesced memory access and vectorized loads on GPU. Benchmark results on graph matrices (e.g. enwiki-2018 PPR) show ~1.75× throughput vs. cuSPARSE at typical block configurations.

## Requirements

- Linux + CUDA-capable GPU
- CUDA 12.x (nvcc inside conda env, not system-wide — see build notes)
- Python ≥ 3.12
- PyTorch with CUDA support
- pybind11, numpy, scipy

## Install

```bash
conda activate fem   # or your env with PyTorch+CUDA

export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX

pip install -e torch_fast_spdmm --no-build-isolation
```

After any change to `.cu` or `.cpp` sources, re-run the install command.

## Usage

```python
import scipy.sparse as sp
from torch_fast_spdmm import csr_to_pbr, pbr_matmul
import torch

# Convert a scipy CSR matrix to PBR
A_pbr = csr_to_pbr(A_csr, block_rows=8, block_cols=8, min_nnz=4)
A_pbr = A_pbr.to("cuda")

# Dense feature matrix
X = torch.randn(A_csr.shape[1], 128, device="cuda")

# SpMM
out = pbr_matmul(A_pbr, X)
```

## Formats

| Format | Description |
|--------|-------------|
| **PBR** | Block-compressed: encodes sparsity patterns as 64-bit bitsets; CSR fallback for low-density blocks |
| **CSR** | Standard compressed sparse row |
| **CSC** | Compressed sparse column |

## Tests

```bash
pytest tests/torch_fast_spdmm/ -v
```

Benchmarks log to Weights & Biases project `torch-fast-spdmm`:

```bash
pytest tests/torch_fast_spdmm/test_benchmark.py -v
```

## Repository layout

```
torch_fast_spdmm/          # C++/CUDA extension + Python package
  bindings/                # pybind11 entry points
  kernels/                 # CUDA kernels (PBR, CSR, CSC)
  pbr_matrix/              # PBR format data structures and CPU ops
  torch_fast_spdmm/        # Python API (__init__.py)
  setup.py / pyproject.toml
tests/
  torch_fast_spdmm/        # pytest suite
    benchmark/             # matrix generators and kernel sweeps
```
