# BOO correctness and coverage

Run compilation and tests on a remote GPU node using the environment described
in `CLAUDE.md`. A passing numerical suite is evidence for the cases exercised,
not a proof of correctness for every possible input.

## Input contract

- Build from a valid 2D SciPy CSR matrix or CPU PyTorch CSR tensor with float32
  or float64 values. Explicit zero entries, duplicates, and unsorted entries are
  supported; multiplication follows their additive interpretation.
- Explicit labels are supplied for both axes together. Each array is 1D, integer,
  matches its axis length, and has labels in `[0, axis_length)`. Group members may
  be noncontiguous and group sizes may differ. Local coordinates must fit uint16.
- Dense X is a CPU/CUDA tensor shaped `[cols, features]` or
  `[batch, cols, features]`, with the same scalar dtype as the matrix. Strided
  views are accepted and made contiguous for CUDA. Empty matrices, batches, and
  feature dimensions are supported.
- CUDA metadata and X must reside on the same GPU; execution need not start with
  that GPU selected as the current device. GPU dimensions and metadata use int32.
- CUDA threads: 128, 256, or 512. Requested feature tile `p`: 8, 16, 32, 64, or
  128. Narrow features use a minimum P=8 allocation with padding. `csr_mode` is
  0, 1, or 2. Row-segmented execution requires row-major entries; nnzflat accepts
  either order. Oversized shared-memory requests raise CUDA errors.

## Coverage map

`tests/torch_fast_spdmm/test_boo.py` retains format round trips, invariants,
complete-array determinism, the NumPy decoder, dense references, clustering,
imbalanced matrices, and policy integration. Dense inputs have repeatable seeds.

`tests/torch_fast_spdmm/test_boo_edges.py` adds:

| Area | Cases |
|---|---|
| Kernel paths | Explicit rowseg, explicit row-major nnzflat, automatic column-major nnzflat |
| Feature boundaries | 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1025; FP32/FP64 |
| Launch specializations | All 3 supported thread counts crossed with all 5 requested tile sizes |
| Format structure | Explicit irregular groups, empty local/global rows, rectangular batches, singleton axes |
| Sparse partition | All BOO, mixed BOO+CSR/COO, and entirely CSR/COO; asserts actual path populations |
| Indices | int32/int64 host matrices and CUDA staging, uint16 high bit and 65535 endpoint, overflow rejection |
| Work loops | Multiple strides per thread and >48 KiB dynamic shared-memory opt-in |
| Empty inputs | Empty matrices including zero axes, zero features, zero batch |
| Tensor views | Transpose, offset slice, broadcast/zero stride, batched transpose; input preservation |
| Integration | Nondefault-stream producers/consumers, temporary contiguous copies, device staging round trip, exact accumulation from 257 blocks into the same rows; multi-GPU test when available |
| Numerical inputs | Signed values, explicit zeros, duplicate cancellation, unsorted CSR, twelve reproducible randomized problems |
| Invalid inputs | Shape, dtype, layout, label bounds, malformed CSR, unsupported parameters/order, excessive shared memory |

New product tests compare against CPU float64 multiplication of the original
matrix. They do not derive the expected result from BOO or another custom GPU
kernel. Each randomized pytest ID identifies the seed. Thread/tile combinations
and difficult boundary families are separated to keep the suite bounded rather
than multiplying every parameter into one enormous Cartesian product.

## Commands after rebuilding

From the repository root on the GPU node:

```bash
python -m pytest tests/torch_fast_spdmm/test_boo.py \
    tests/torch_fast_spdmm/test_boo_edges.py --boo-require-cuda -ra
```

`--boo-require-cuda` rejects a run without a visible GPU instead of silently
accepting skipped CUDA tests. The multi-GPU case still skips on a one-GPU node.
The optimizer correctness gate includes both BOO files and this flag.

Run the small device-safety subset under each sanitizer, separately from timing:

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck --error-exitcode 1 \
    python -m pytest tests/torch_fast_spdmm/test_boo_edges.py \
    --boo-require-cuda -m boo_sanitizer -q
PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool racecheck --error-exitcode 1 \
    python -m pytest tests/torch_fast_spdmm/test_boo_edges.py \
    --boo-require-cuda -m boo_sanitizer -q
PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool synccheck --error-exitcode 1 \
    python -m pytest tests/torch_fast_spdmm/test_boo_edges.py \
    --boo-require-cuda -m boo_sanitizer -q
```

Follow with the PBR/CSR correctness files and `tests/optimizer` for regressions.
Benchmark timing belongs in a separate run without sanitizers.

## Limits

The suite does not claim exhaustive data/branch coverage. Multi-GPU behavior needs
two visible GPUs; allocations near int32 limits are not attempted. Numerical
checks target finite values, not a defined NaN/Inf contract or autograd behavior.
Results on one GPU architecture do not establish portability to every GPU.

## Validation record

Validated on 2026-09-16 on an RTX 3090 in an isolated source/build directory:
`/home/dayanb/research/boo-coverage.hoHA6Y`. The existing remote checkout and
editable installation were not changed. Slurm jobs: `21391587` (build, initial
suite, regressions, sanitizers) and `21391804` (final suite and no-GPU gate).

| Check | Result |
|---|---|
| Final BOO correctness suite | 485 passed, 13 skipped, about 11 seconds |
| PBR/CSR/optimizer regressions | 647 passed |
| Compute Sanitizer memcheck | 123 selected cases passed, 0 errors |
| Compute Sanitizer racecheck | 123 selected cases passed, 0 hazards/errors/warnings |
| Compute Sanitizer synccheck | 123 selected cases passed, 0 errors |
| CUDA-required gate with GPUs hidden | Failed as expected with an explicit CUDA-unavailable error |

The skips are 12 unsupported rectangular-RCM combinations from the original
suite and one test requiring two visible GPUs. The suites emitted PyTorch's
standard sparse-CSR beta warning. Sanitizers ran on the same functional kernel
implementation; the final five added tests concern int64 staging and cross-block
accumulation and do not alter that sanitizer subset.

Remote logs: `correctness-final.log`, `regression.log`, `memcheck.log`,
`racecheck.log`, `synccheck.log`, and `cuda-required.log`. JUnit reports:
`boo-final-results.xml` and `regression-results.xml`.
