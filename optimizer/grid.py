"""Enumerates the PBR kernel parameter grid as flat CandidateConfig points.

Implementation note (deviation from the original plan): the plan called for
importing these constants live from ``tests/torch_fast_spdmm/test_benchmark.py``
and ``.../benchmark/profile_pbr.py``. Both of those modules transitively
``import torch_fast_spdmm_core`` (the compiled CUDA extension) at module load
time, which only exists on a built GPU node -- so a live import would make
this module (and anything that tests it) unusable in a plain local/CI
environment with no GPU. Instead, the enumeration constants below are kept as
explicit, self-contained values that must be **manually kept in sync** with:
  - ``tests/torch_fast_spdmm/test_benchmark.py``: _ALL_SIZES, _PBR_BLOCK_CONFIGS,
    _SMEM_TIERS
  - ``tests/torch_fast_spdmm/benchmark/profile_pbr.py``: _MATRIX_PROFILES (keys only)
This keeps optimizer/grid.py importable and unit-testable with plain Python,
no torch/CUDA/GPU required.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# -- Matrix profile names -- keep in sync with profile_pbr.py's _MATRIX_PROFILES keys.
MATRIX_PROFILES = [
    "scattered_8x1",
    "scattered_1x8",
    "scattered_2x4",
    "scattered_2x2",
    "scattered_dense2",
    "diag_dd0.017",
    "diag_dd0.033",
    "diag_dd0.05",
    "diag_b0.1",
    "diag_b0.3",
    "diag_b1.0",
    "diag_skewed",
    "fem2d",
    "fem2d_permuted",
]

# -- Sizes -- keep in sync with test_benchmark.py's _ALL_SIZES.
ALL_SIZES = [1024, 4096, 16384, 65536, 262144]

# -- PBR block-size / min-nnz variants -- keep in sync with _PBR_BLOCK_CONFIGS.
PBR_BLOCK_CONFIGS = [
    (2, 2), (2, 3),
    (4, 2), (4, 4), (4, 8),
    (8, 2), (8, 8), (8, 12),
]  # (block_size, min_nnz)

# -- Shared-memory tiers -- keep in sync with _SMEM_TIERS.
SMEM_TIERS = [512, 1024, 2048]

FEATURES_DEFAULT = [128, 512]


@dataclass(frozen=True)
class CandidateConfig:
    """One point in the PBR parameter grid -- maps 1:1 onto profile_pbr.py's CLI args."""

    matrix_profile: str
    size: int
    features: int
    block: int
    min_nnz: int
    smem_size: int

    def config_key(self) -> str:
        """Stable string key identifying this exact config point, for ledger dedup."""
        return "|".join([
            self.matrix_profile, str(self.size), str(self.features),
            str(self.block), str(self.min_nnz), str(self.smem_size),
        ])

    def cli_args(self) -> list[str]:
        """The exact argv this config maps to for profile_pbr.py."""
        return [
            "--size", str(self.size),
            "--features", str(self.features),
            "--block", str(self.block),
            "--min-nnz", str(self.min_nnz),
            "--smem-size", str(self.smem_size),
            "--matrix-profile", self.matrix_profile,
        ]

    def as_dict(self) -> dict:
        return asdict(self)


def enumerate_pbr_kernel_configs(block_configs=None, smem_tiers=None):
    """Yield pbr_cfg_kwargs dicts -- one per (block_size, min_nnz, smem) point."""
    block_configs = block_configs if block_configs is not None else PBR_BLOCK_CONFIGS
    smem_tiers = smem_tiers if smem_tiers is not None else SMEM_TIERS

    for block_size, min_nnz in block_configs:
        for smem in smem_tiers:
            yield {"block": block_size, "min_nnz": min_nnz, "smem_size": smem}


def enumerate_grid(matrices=None, sizes=None, features=None,
                    block_configs=None, smem_tiers=None) -> list[CandidateConfig]:
    """Flatten (matrix x size x features x pbr-config) into CandidateConfig points."""
    matrices = matrices if matrices is not None else MATRIX_PROFILES
    sizes = sizes if sizes is not None else ALL_SIZES
    features = features if features is not None else FEATURES_DEFAULT

    configs: list[CandidateConfig] = []
    for matrix in matrices:
        for size in sizes:
            for feat in features:
                for pbr_cfg in enumerate_pbr_kernel_configs(block_configs, smem_tiers):
                    configs.append(CandidateConfig(
                        matrix_profile=matrix,
                        size=size,
                        features=feat,
                        block=pbr_cfg["block"],
                        min_nnz=pbr_cfg["min_nnz"],
                        smem_size=pbr_cfg["smem_size"],
                    ))
    return configs
