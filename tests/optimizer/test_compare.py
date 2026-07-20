from optimizer.compare import compare_ledgers, ComparisonRow
from optimizer.grid import CandidateConfig
from optimizer.ledger import Ledger


def _cfg(**overrides):
    defaults = dict(matrix_profile="diag_b0.3", size=262144, features=512,
                     kernel="block", block=8, min_nnz=2, smem_size=2048)
    defaults.update(overrides)
    return CandidateConfig(**defaults)


def test_compare_matches_identical_config_across_ledgers(tmp_path):
    baseline = Ledger(tmp_path / "baseline.sqlite3")
    candidate = Ledger(tmp_path / "candidate.sqlite3")

    cfg = _cfg()
    baseline.record(cfg, git_commit="base123", correctness_pass=True,
                     metrics={"gpu_time_ns": 12986144.0, "mio_throttle_pct": 21.32, "barrier_pct": 20.11},
                     verdict="measured")
    candidate.record(cfg, git_commit="cand456", correctness_pass=True,
                      metrics={"gpu_time_ns": 10000000.0, "mio_throttle_pct": 10.0, "barrier_pct": 5.0},
                      verdict="measured")

    rows = compare_ledgers(str(tmp_path / "baseline.sqlite3"), str(tmp_path / "candidate.sqlite3"))
    assert len(rows) == 1
    row = rows[0]
    assert row.matrix_profile == "diag_b0.3"
    assert row.verdict() == "WIN"  # ~23% faster


def test_compare_no_matching_points_returns_empty(tmp_path):
    baseline = Ledger(tmp_path / "baseline.sqlite3")
    candidate = Ledger(tmp_path / "candidate.sqlite3")

    baseline.record(_cfg(matrix_profile="diag_b0.1"), git_commit="c1", correctness_pass=True,
                     metrics={"gpu_time_ns": 1.0}, verdict="measured")
    candidate.record(_cfg(matrix_profile="diag_b0.3"), git_commit="c2", correctness_pass=True,
                      metrics={"gpu_time_ns": 1.0}, verdict="measured")

    rows = compare_ledgers(str(tmp_path / "baseline.sqlite3"), str(tmp_path / "candidate.sqlite3"))
    assert rows == []


def test_compare_ignores_failed_records(tmp_path):
    baseline = Ledger(tmp_path / "baseline.sqlite3")
    candidate = Ledger(tmp_path / "candidate.sqlite3")

    cfg = _cfg()
    baseline.record(cfg, git_commit="c1", correctness_pass=True, metrics={"gpu_time_ns": 1.0}, verdict="measured")
    candidate.record(cfg, git_commit="c2", correctness_pass=True, metrics=None, verdict="failed_no_data")

    rows = compare_ledgers(str(tmp_path / "baseline.sqlite3"), str(tmp_path / "candidate.sqlite3"))
    assert rows == []


def test_verdict_classification_thresholds():
    row_win = ComparisonRow("k", "m", 1, 1, "block", 8, 2048,
                             baseline_metrics={"gpu_time_ns": 100.0},
                             candidate_metrics={"gpu_time_ns": 90.0})  # -10%
    row_regression = ComparisonRow("k", "m", 1, 1, "block", 8, 2048,
                                    baseline_metrics={"gpu_time_ns": 100.0},
                                    candidate_metrics={"gpu_time_ns": 110.0})  # +10%
    row_wash = ComparisonRow("k", "m", 1, 1, "block", 8, 2048,
                              baseline_metrics={"gpu_time_ns": 100.0},
                              candidate_metrics={"gpu_time_ns": 101.0})  # +1%
    assert row_win.verdict() == "WIN"
    assert row_regression.verdict() == "REGRESSION"
    assert row_wash.verdict() == "WASH"


def test_whack_a_mole_visible_in_metric_deltas():
    """A wash on time despite the target metric improving -- the whack-a-mole
    pattern found in every prior kernel-edit attempt -- must still be visible
    per-metric, not hidden by the single time-based verdict."""
    row = ComparisonRow(
        "k", "diag_b0.3", 262144, 512, "block", 8, 2048,
        baseline_metrics={"gpu_time_ns": 100.0, "barrier_pct": 20.0, "mio_throttle_pct": 5.0},
        candidate_metrics={"gpu_time_ns": 100.5, "barrier_pct": 0.0, "mio_throttle_pct": 25.0},
    )
    assert row.verdict() == "WASH"
    assert row.delta("barrier_pct") == -20.0  # barrier improved...
    assert row.delta("mio_throttle_pct") == 20.0  # ...but mio_throttle absorbed it
