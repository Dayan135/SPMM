# torch_fast_spdmm

CUDA library for sparse-dense matrix multiplication (SpMM), exposed as a PyTorch extension via pybind11. It includes BOO (Blocked COO), PBR (Packed Block Representation), and a naive CSR benchmark baseline.

Current research focuses on BOO grouping, row-segmented CUDA kernels, and comparisons with direct cuSPARSE. BOO performance depends strongly on matrix structure and preprocessing cost; see [the confirmed tile experiment](experiments/2026-10-12_tile_optimization/TILE_REVIEW.md) and [the full FPT sweep](experiments/2026-10-09_full_fpt/FULL_FPT_BENCHMARK.md). PBR remains implemented and tested as a separate format.

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
from torch_fast_spdmm import csr_to_boo, boo_matmul
import torch

# A_csr is a float32 scipy CSR matrix. Identity grouping avoids an offline
# clustering step; HEM, Graclus, and METIS are also available.
A_boo = csr_to_boo(A_csr, cluster="identity", target_nr=32, target_nc=32)
A_boo = A_boo.to("cuda")

# Dense feature matrix
X = torch.randn(A_csr.shape[1], 128, device="cuda", dtype=torch.float32)

# SpMM
out = boo_matmul(A_boo, X)
```

## Formats

| Format | Description |
|--------|-------------|
| **BOO** | Groups rows and columns, stores nonzeros by block, and routes small blocks to a CSR remainder by default |
| **PBR** | Block-compressed: encodes sparsity patterns as 64-bit bitsets; CSR remainder for low-density blocks |
| **CSR** | Standard compressed sparse row — naive kernel kept as a benchmark baseline |

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
  kernels/                 # BOO, PBR, and naive CSR CUDA kernels
  boo_matrix/              # BOO format data structures and CPU ops
  pbr_matrix/              # PBR format data structures and CPU ops
  torch_fast_spdmm/        # Python API, clustering, and execution plans
  setup.py / pyproject.toml
tests/
  torch_fast_spdmm/        # pytest suite
    benchmark/             # matrix generators, sweeps, and selected results
scripts/                   # experiment drivers and report generators
experiments/               # dated reports, explanations, and selected figures
docs/reference/            # shared format and testing notes
```

## Research record and result archive

- [Research index](RESEARCH_INDEX.md)
- [Dated experiments and figures](experiments/README.md)
- [BOO format and kernel design](docs/reference/BOO_FORMAT.md)
- [PBR / BOO / CSR benchmark](experiments/2026-09-23_spmm_baseline/SPMM_BENCHMARK.md)
- [Full FPT sweep](experiments/2026-10-09_full_fpt/FULL_FPT_BENCHMARK.md) and [plots](experiments/2026-10-09_full_fpt/plots/README.md)
- [Confirmed tile experiment](experiments/2026-10-12_tile_optimization/TILE_REVIEW.md) and [detailed report](experiments/2026-10-12_tile_optimization/TILE_EXPERIMENTS.md)
- [Result archive policy and restore instructions](RESULTS_ARCHIVE.md)

The repository keeps reports, selected figures, summary tables, and metadata.
Per-call samples, saved matrix/label arrays, and profiler captures are in the
separate result archive described in `RESULTS_ARCHIVE.md`.
