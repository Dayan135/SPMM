from optimizer.grid import (
    CandidateConfig, enumerate_grid, enumerate_pbr_kernel_configs,
    MATRIX_PROFILES, ALL_SIZES, PBR_BLOCK_CONFIGS, SMEM_TIERS,
)


def test_config_key_is_stable_and_unique():
    a = CandidateConfig("diag_b0.3", 262144, 512, 8, 2, 2048)
    b = CandidateConfig("diag_b0.3", 262144, 512, 8, 2, 2048)
    c = CandidateConfig("diag_b0.3", 262144, 128, 8, 2, 2048)
    assert a.config_key() == b.config_key()
    assert a.config_key() != c.config_key()


def test_cli_args_shape_for_block_kernel():
    cfg = CandidateConfig("diag_b0.3", 262144, 512, 8, 2, 2048)
    args = cfg.cli_args()
    assert "--matrix-profile" in args and "diag_b0.3" in args
    assert "--block" in args and "8" in args


def test_enumerate_pbr_kernel_configs_covers_every_block_config():
    cfgs = list(enumerate_pbr_kernel_configs(
        block_configs=[(4, 4), (8, 2)], smem_tiers=[2048],
    ))
    assert len(cfgs) == 2
    assert {c["block"] for c in cfgs} == {4, 8}


def test_enumerate_grid_produces_expected_count_for_small_slice():
    configs = enumerate_grid(
        matrices=["diag_b0.3"], sizes=[262144], features=[512],
        block_configs=[(8, 2), (4, 4)], smem_tiers=[2048],
    )
    # 1 matrix x 1 size x 1 feature x 2 block_configs = 2
    assert len(configs) == 2
    assert {c.block for c in configs} == {8, 4}


def test_enumerate_grid_default_covers_all_matrix_profiles():
    configs = enumerate_grid(sizes=[1024], features=[128],
                              block_configs=[(8, 2)], smem_tiers=[2048])
    assert {c.matrix_profile for c in configs} == set(MATRIX_PROFILES)


def test_grid_constants_are_nonempty():
    assert len(MATRIX_PROFILES) > 0
    assert len(ALL_SIZES) > 0
    assert len(PBR_BLOCK_CONFIGS) > 0
    assert len(SMEM_TIERS) > 0
