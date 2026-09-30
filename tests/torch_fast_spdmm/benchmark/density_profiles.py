"""37 new matrix configurations spanning three nnz/row density bands:
(a) ~1 (accept 0.8-1.2), (b) ~7 (accept 6-8), (c) ~48 (accept 32-64) -- plus
FEM's own natural densities (~9/36/72). Self-contained: does not modify or
reuse profile_pbr.py's existing _MATRIX_PROFILES, so old results stay
reproducible. Reuses matrices.py's existing generators unmodified.

For the two tile-based families (scattered, pattern), each active tile
contributes a fixed number of nonzeros (`nnz_per_tile`); a tile lands in any
given tile-row with probability ~ 1/grid_dim, and grid_dim = N/8. Solving
E[nnz/row] = target for the generator's `density` parameter (see
generate_scattered_block_matrix/generate_patterned_block_matrix in
matrices.py):

    density = 64 * target_nnz_per_row / (nnz_per_tile * N)

(equivalently: tiles_per_tile_row = 8 * target / nnz_per_tile, matching the
requested derivation directly.) Verified empirically against the actual
generated matrix's nnz/row after construction -- see verify_bands() below.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_matrices_spec = importlib.util.spec_from_file_location(
    "matrices", str(Path(__file__).parent / "matrices.py"))
matrices = importlib.util.module_from_spec(_matrices_spec)
_matrices_spec.loader.exec_module(matrices)

BANDS = {"a": (0.8, 1.2), "b": (6, 8), "c": (32, 64)}
BAND_TARGET = {"a": 1, "b": 7, "c": 48}

_TILE_BLOCK = 8
_SCATTERED_SHAPES = {  # name -> (n_r, n_c), nnz_per_tile = n_r*n_c
    "scattered_1x8": (1, 8), "scattered_8x1": (8, 1),
    "scattered_2x4": (2, 4), "scattered_2x2": (2, 2),
}
_PATTERN_MASKS = {  # name -> mask; nnz_per_tile = mask.sum()
    "pattern_diag_stripe": matrices.PBR_PATTERN_PRESETS["diag_stripe"],
    "pattern_corner_cluster": matrices.PBR_PATTERN_PRESETS["corner_cluster"],
    "pattern_checkerboard": matrices.PBR_PATTERN_PRESETS["checkerboard"],
    "pattern_row_skewed": matrices.PBR_PATTERN_PRESETS["row_skewed"],
}


def _tile_density(system_size: int, target_nnz_per_row: float, nnz_per_tile: int) -> float:
    return 64.0 * target_nnz_per_row / (nnz_per_tile * system_size)


def _build_scattered(system_size, name, band, seed):
    n_r, n_c = _SCATTERED_SHAPES[name]
    target = BAND_TARGET[band]
    density = _tile_density(system_size, target, n_r * n_c)
    return matrices.generate_scattered_block_matrix(
        system_size, block_size=_TILE_BLOCK, profiles=[(n_r, n_c, density)], seed=seed)


def _build_pattern(system_size, name, band, seed):
    mask = _PATTERN_MASKS[name]
    target = BAND_TARGET[band]
    density = _tile_density(system_size, target, int(mask.sum()))
    return matrices.generate_patterned_block_matrix(
        system_size, block_size=_TILE_BLOCK, patterns=[(mask, density)], seed=seed)


def _build_diag_fill(system_size, band, seed):
    target = BAND_TARGET[band]
    diag_density = min(target / 64.0, 1.0)
    return matrices.generate_diag_blocks_plus_noise(
        system_size, block_size=64, diag_density=diag_density, off_diag_density=0.0, seed=seed)


def _build_diag_dense(system_size, band, seed):
    block_size = {"b": 8, "c": 64}[band]
    return matrices.generate_diag_blocks_plus_noise(
        system_size, block_size=block_size, diag_density=1.0, off_diag_density=0.0, seed=seed)


def _build_diag_skewed(system_size, band, seed):
    # Same off_diag_density as the existing (pre-band) diag_skewed profile in
    # profile_pbr.py -- only block_size varies here to hit band b vs c.
    block_size = {"b": 8, "c": 64}[band]
    return matrices.generate_diag_blocks_plus_noise(
        system_size, block_size=block_size, diag_density=1.0, off_diag_density=0.00002, seed=seed)


def _build_fem(system_size, permuted, dofs_per_node, seed):
    gen = matrices.generate_permuted_fem_2d if permuted else matrices.generate_fem_2d
    return gen(system_size=system_size, dofs_per_node=dofs_per_node, seed=seed)


# name -> (builder, band_label, flagged_note)
DENSITY_PROFILES: dict = {}
for _name in _SCATTERED_SHAPES:
    for _band in ("a", "b", "c"):
        DENSITY_PROFILES[f"{_name}@{_band}"] = (
            lambda N, s, n=_name, b=_band: _build_scattered(N, n, b, s), _band, None)
for _name in _PATTERN_MASKS:
    for _band in ("a", "b", "c"):
        DENSITY_PROFILES[f"{_name}@{_band}"] = (
            lambda N, s, n=_name, b=_band: _build_pattern(N, n, b, s), _band, None)
for _band in ("a", "b", "c"):
    DENSITY_PROFILES[f"diag_fill@{_band}"] = (
        lambda N, s, b=_band: _build_diag_fill(N, b, s), _band, None)
for _band in ("b", "c"):
    DENSITY_PROFILES[f"diag_dense@{_band}"] = (
        lambda N, s, b=_band: _build_diag_dense(N, b, s), _band, None)
    DENSITY_PROFILES[f"diag_skewed@{_band}"] = (
        lambda N, s, b=_band: _build_diag_skewed(N, b, s), _band, None)
for _profile, _permuted in (("fem2d", False), ("fem2d_permuted", True)):
    DENSITY_PROFILES[f"{_profile}@dofs1"] = (
        lambda N, s, p=_permuted: _build_fem(N, p, 1, s), "b", "labeled ~9, closest to band b")
    DENSITY_PROFILES[f"{_profile}@dofs4"] = (
        lambda N, s, p=_permuted: _build_fem(N, p, 4, s), "c", None)
    DENSITY_PROFILES[f"{_profile}@dofs8"] = (
        lambda N, s, p=_permuted: _build_fem(N, p, 8, s), "c",
        "~72 nnz/row, just above band c -- included because it aligns with PBR's 8x8 tiles")

assert len(DENSITY_PROFILES) == 37, len(DENSITY_PROFILES)


def build_density_matrix(system_size: int, name: str, seed: int = 0):
    builder, band, note = DENSITY_PROFILES[name]
    mat = builder(system_size, seed).astype(np.float32).tocsr()
    mat.sum_duplicates()
    mat.sort_indices()
    return mat, band, note


def verify_bands(system_size: int, seed: int = 0):
    """Build every profile once and check actual nnz/row lands in its band
    (except @dofsN, which have no target -- report only). Returns a list of
    dicts, one per profile."""
    rows = []
    for name in DENSITY_PROFILES:
        mat, band, note = build_density_matrix(system_size, name, seed)
        actual = mat.nnz / system_size
        target = BAND_TARGET.get(band)
        lo, hi = BANDS[band]
        in_band = lo <= actual <= hi if "dofs" not in name else None
        rows.append(dict(profile=name, band=band, target_nnz_per_row=target,
                         actual_nnz_per_row=actual, nnz=mat.nnz, in_band=in_band, note=note))
    return rows


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    rows = verify_bands(N)
    bad = [r for r in rows if r["in_band"] is False]
    for r in rows:
        flag = "" if r["in_band"] in (True, None) else "  <-- OUT OF BAND"
        print(f"{r['profile']:28s} band={r['band']} target={r['target_nnz_per_row']} "
              f"actual={r['actual_nnz_per_row']:.2f}{flag}")
    if bad:
        print(f"\n{len(bad)} profiles out of band at N={N}")
        sys.exit(1)
    print(f"\nAll bands verified at N={N}")
