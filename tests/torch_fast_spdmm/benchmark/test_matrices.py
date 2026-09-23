import numpy as np
from pytest import mark, raises

from benchmark.matrices import (generate_patterned_block_matrix, PBR_PATTERN_PRESETS,
                                generate_scattered_block_matrix, generate_diag_blocks_plus_noise)


def test_presets_are_valid_8x8_masks():
    for name, mask in PBR_PATTERN_PRESETS.items():
        assert mask.shape == (8, 8), name
        assert mask.dtype == bool, name
        assert mask.sum() > 0, name


def test_row_skewed_preset_is_maximally_imbalanced():
    mask = PBR_PATTERN_PRESETS["row_skewed"]
    row_counts = mask.sum(axis=1)
    assert row_counts[0] == 8
    assert (row_counts[1:] == 1).all()


@mark.parametrize("pattern_name", list(PBR_PATTERN_PRESETS))
def test_generated_matrix_matches_preset_shape_and_dtype(pattern_name):
    mat = generate_patterned_block_matrix(
        1024, 8, [(PBR_PATTERN_PRESETS[pattern_name], 0.02)], seed=0,
    )
    assert mat.shape == (1024, 1024)
    assert mat.dtype == np.float32
    assert mat.nnz > 0


def test_reproducible_with_same_seed():
    a = generate_patterned_block_matrix(1024, 8, [(PBR_PATTERN_PRESETS["diag_stripe"], 0.02)], seed=42)
    b = generate_patterned_block_matrix(1024, 8, [(PBR_PATTERN_PRESETS["diag_stripe"], 0.02)], seed=42)
    assert (a != b).nnz == 0


@mark.parametrize('generator,kwargs', [
    (generate_scattered_block_matrix, dict(block_size=8, profiles=[(2, 4, 0.02)])),
    (generate_diag_blocks_plus_noise, dict(diag_density=0.3)),
])
def test_explicit_seed_is_reproducible_and_preserves_global_rng(generator, kwargs):
    before = np.random.get_state()
    a = generator(1024, seed=42, **kwargs)
    b = generator(1024, seed=42, **kwargs)
    c = generator(1024, seed=43, **kwargs)
    after = np.random.get_state()
    assert (a != b).nnz == 0
    assert (a != c).nnz > 0
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@mark.parametrize('profile', ['scattered_2x4', 'diag_b0.3', 'pattern_checkerboard',
                                   'fem2d', 'fem2d_permuted'])
def test_benchmark_profile_seed_reproduces_exact_csr(profile):
    from benchmark.profile_pbr import build_matrix
    a = build_matrix(1024, profile, seed=42)
    b = build_matrix(1024, profile, seed=42)
    for field in ('indptr', 'indices', 'data'):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_fem_pair_uses_same_values_before_permutation():
    from benchmark.profile_pbr import build_matrix
    a = build_matrix(1024, 'fem2d', seed=42)
    b = build_matrix(1024, 'fem2d_permuted', seed=42)
    np.testing.assert_array_equal(np.sort(a.data), np.sort(b.data))


def test_low_density_nnz_matches_pattern_popcount_times_block_count():
    # Density low enough that with-replacement block-position collisions
    # (same sampling scheme as generate_scattered_block_matrix) are rare.
    system_size, block_size, density = 4096, 8, 0.001
    mask = PBR_PATTERN_PRESETS["row_skewed"]
    mat = generate_patterned_block_matrix(system_size, block_size, [(mask, density)], seed=0)
    expected_blocks = int((system_size // block_size) ** 2 * density)
    expected_nnz = expected_blocks * int(mask.sum())
    assert abs(mat.nnz - expected_nnz) / expected_nnz < 0.05


def test_mixed_patterns_combine():
    mat = generate_patterned_block_matrix(1024, 8, [
        (PBR_PATTERN_PRESETS["diag_stripe"], 0.01),
        (PBR_PATTERN_PRESETS["checkerboard"], 0.01),
    ], seed=1)
    assert mat.nnz > 0


def test_rejects_wrong_mask_shape():
    with raises(ValueError, match="does not match"):
        generate_patterned_block_matrix(1024, 8, [(np.ones((4, 4), dtype=bool), 0.1)])


def test_rejects_empty_mask():
    with raises(ValueError, match="no True cells"):
        generate_patterned_block_matrix(1024, 8, [(np.zeros((8, 8), dtype=bool), 0.1)])


def test_rejects_density_producing_zero_blocks():
    with raises(ValueError, match="produced 0 blocks"):
        generate_patterned_block_matrix(8, 8, [(PBR_PATTERN_PRESETS["diag_stripe"], 0.0001)])
