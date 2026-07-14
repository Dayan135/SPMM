from optimizer.grid import (
    CandidateConfig, enumerate_grid, enumerate_pbr_kernel_configs,
    MATRIX_PROFILES, ALL_SIZES, PBR_BLOCK_CONFIGS, SMEM_TIERS,
    TILE_SHAPES, TILE_STRATEGIES,
)


def test_config_key_is_stable_and_unique():
    a = CandidateConfig("diag_b0.3", 262144, 512, "block", 8, 2, 2048)
    b = CandidateConfig("diag_b0.3", 262144, 512, "block", 8, 2, 2048)
    c = CandidateConfig("diag_b0.3", 262144, 128, "block", 8, 2, 2048)
    assert a.config_key() == b.config_key()
    assert a.config_key() != c.config_key()


def test_config_key_distinguishes_tile_shape():
    a = CandidateConfig("diag_b0.3", 262144, 512, "tile", 8, 2, 2048, tile_rows=2, tile_cols=2, tile_strategy="grid")
    b = CandidateConfig("diag_b0.3", 262144, 512, "tile", 8, 2, 2048, tile_rows=4, tile_cols=4, tile_strategy="grid")
    assert a.config_key() != b.config_key()


def test_cli_args_shape_for_block_kernel():
    cfg = CandidateConfig("diag_b0.3", 262144, 512, "block", 8, 2, 2048)
    args = cfg.cli_args()
    assert "--matrix-profile" in args and "diag_b0.3" in args
    assert "--kernel" in args and "block" in args
    assert "--tile-rows" not in args


def test_cli_args_shape_for_tile_kernel():
    cfg = CandidateConfig("diag_b0.3", 262144, 512, "tile", 8, 2, 2048,
                           tile_rows=2, tile_cols=2, tile_strategy="grid")
    args = cfg.cli_args()
    assert "--tile-rows" in args and "2" in args
    assert "--tile-strategy" in args and "grid" in args


def test_enumerate_pbr_kernel_configs_tile_only_for_8x8():
    pairs = list(enumerate_pbr_kernel_configs(
        block_configs=[(4, 4), (8, 2)], smem_tiers=[2048],
        tile_shapes=TILE_SHAPES, tile_strategies=TILE_STRATEGIES,
        kernels=("block",),
    ))
    tile_pairs = [p for p in pairs if p[0]["kernel"] == "tile"]
    non_tile_pairs = [p for p in pairs if p[0]["kernel"] != "tile"]
    # tile only ever paired with block_size==8
    assert all(pbr_cfg["block"] == 8 for _, pbr_cfg in tile_pairs)
    # block_size==4 never gets a tile pairing
    assert not any(pbr_cfg["block"] == 4 for kspec, pbr_cfg in pairs if kspec["kernel"] == "tile")
    assert len(non_tile_pairs) == 2  # one "block" kernel per block_config


def test_enumerate_grid_produces_expected_count_for_small_slice():
    configs = enumerate_grid(
        matrices=["diag_b0.3"], sizes=[262144], features=[512],
        block_configs=[(8, 2)], smem_tiers=[2048],
        kernels=("block", "rowgroup"),
    )
    # 1 matrix x 1 size x 1 feature x (2 kernels, no tile since kernels excludes tile) = 2
    assert len(configs) == 2
    assert {c.kernel for c in configs} == {"block", "rowgroup"}


def test_enumerate_grid_default_covers_all_matrix_profiles():
    configs = enumerate_grid(sizes=[1024], features=[128],
                              block_configs=[(8, 2)], smem_tiers=[2048],
                              kernels=("block",))
    assert {c.matrix_profile for c in configs} == set(MATRIX_PROFILES)


def test_grid_constants_are_nonempty():
    assert len(MATRIX_PROFILES) > 0
    assert len(ALL_SIZES) > 0
    assert len(PBR_BLOCK_CONFIGS) > 0
    assert len(SMEM_TIERS) > 0
    assert len(TILE_SHAPES) > 0
    assert len(TILE_STRATEGIES) > 0
