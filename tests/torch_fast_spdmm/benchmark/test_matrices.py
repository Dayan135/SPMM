import numpy as np
from pytest import mark, raises

from benchmark.matrices import generate_patterned_block_matrix, PBR_PATTERN_PRESETS


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
