from optimizer.performance import parse_ncu_csv

# A trimmed synthetic ncu --csv capture matching the real format this project's
# jobs actually produce (see ncu_out/stall_reasons/*.csv from the stall-reason
# investigation) -- includes an unrelated FillFunctor kernel row that must be
# filtered out, and a pbr_spmm_vec_kernel row that must be picked up.
_SAMPLE_CSV = '''==PROF== Connected to process 12345
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","12345","python3.12","127.0.0.1","void at::vectorized_elementwise_kernel<4, at::FillFunctor<float>, std::array<char *, 1>>(int, T2, T3)","1","7","(128, 1, 1)","(131072, 1, 1)","0","8.6","Command line profiler metrics","gpu__time_duration.sum","ns","500"
"1","12345","python3.12","127.0.0.1","void pbr_spmm_vec_kernel<int, float, 8, 8, 2048, 128, 512>(unsigned int)","1","93","(128, 1, 1)","(262144, 1, 1)","0","8.6","Command line profiler metrics","gpu__time_duration.sum","ns","12,986,144"
"1","12345","python3.12","127.0.0.1","void pbr_spmm_vec_kernel<int, float, 8, 8, 2048, 128, 512>(unsigned int)","1","93","(128, 1, 1)","(262144, 1, 1)","0","8.6","Command line profiler metrics","smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct","%","21.32"
"1","12345","python3.12","127.0.0.1","void pbr_spmm_vec_kernel<int, float, 8, 8, 2048, 128, 512>(unsigned int)","1","93","(128, 1, 1)","(262144, 1, 1)","0","8.6","Command line profiler metrics","smsp__warp_issue_stalled_barrier_per_warp_active.pct","%","20.11"
'''


def test_parse_ncu_csv_picks_up_target_kernel_only():
    metrics = parse_ncu_csv(_SAMPLE_CSV, kernel_name_prefix="pbr_spmm")
    assert metrics is not None
    assert metrics["gpu_time_ns"] == 12986144.0
    assert metrics["mio_throttle_pct"] == 21.32
    assert metrics["barrier_pct"] == 20.11


def test_parse_ncu_csv_returns_none_when_no_matching_kernel():
    metrics = parse_ncu_csv(_SAMPLE_CSV, kernel_name_prefix="csrmm_alg2")
    assert metrics is None


def test_parse_ncu_csv_handles_empty_input():
    assert parse_ncu_csv("") is None
