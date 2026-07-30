from optimizer.grid import CandidateConfig
from optimizer.ledger import Ledger


def _cfg(**overrides):
    defaults = dict(matrix_profile="diag_b0.3", size=262144, features=512,
                     block=8, min_nnz=2, smem_size=2048)
    defaults.update(overrides)
    return CandidateConfig(**defaults)


def test_has_measured_false_before_record(tmp_path):
    ledger = Ledger(tmp_path / "test.sqlite3")
    cfg = _cfg()
    assert ledger.has_measured(cfg) is False


def test_has_measured_true_after_record(tmp_path):
    ledger = Ledger(tmp_path / "test.sqlite3")
    cfg = _cfg()
    ledger.record(cfg, git_commit="abc123", correctness_pass=True,
                   metrics={"gpu_time_ns": 12986144.0, "mio_throttle_pct": 21.32},
                   verdict="measured")
    assert ledger.has_measured(cfg) is True


def test_distinct_configs_are_distinct_in_ledger(tmp_path):
    ledger = Ledger(tmp_path / "test.sqlite3")
    a = _cfg(features=128)
    b = _cfg(features=512)
    ledger.record(a, git_commit="abc123", correctness_pass=True, metrics={"gpu_time_ns": 1.0}, verdict="measured")
    assert ledger.has_measured(a) is True
    assert ledger.has_measured(b) is False


def test_record_persists_across_reopen(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    ledger1 = Ledger(db_path)
    ledger1.record(_cfg(), git_commit="abc123", correctness_pass=True,
                    metrics={"gpu_time_ns": 1.0}, verdict="measured")

    ledger2 = Ledger(db_path)
    assert ledger2.has_measured(_cfg()) is True
    assert len(ledger2.all_records()) == 1


def test_negative_results_are_recorded_too(tmp_path):
    ledger = Ledger(tmp_path / "test.sqlite3")
    cfg = _cfg()
    ledger.record(cfg, git_commit="abc123", correctness_pass=True, metrics=None, verdict="failed_no_data")
    records = ledger.all_records()
    assert len(records) == 1
    assert records[0]["verdict"] == "failed_no_data"
    # dedup still applies -- a failed/no-data attempt still counts as "measured"
    # so the sweep doesn't retry it forever.
    assert ledger.has_measured(cfg) is True


def test_best_per_matrix_picks_lowest_time(tmp_path):
    ledger = Ledger(tmp_path / "test.sqlite3")
    fast = _cfg(smem_size=2048)
    slow = _cfg(smem_size=1024)
    ledger.record(fast, git_commit="c1", correctness_pass=True,
                   metrics={"gpu_time_ns": 100.0}, verdict="measured")
    ledger.record(slow, git_commit="c1", correctness_pass=True,
                   metrics={"gpu_time_ns": 200.0}, verdict="measured")

    best = ledger.best_per_matrix(metric="gpu_time_ns")
    key = "diag_b0.3/262144/512"
    assert key in best
    assert best[key]["smem_size"] == 2048


def test_best_per_matrix_ignores_failed_records(tmp_path):
    ledger = Ledger(tmp_path / "test.sqlite3")
    ledger.record(_cfg(), git_commit="c1", correctness_pass=True,
                   metrics=None, verdict="failed_no_data")
    best = ledger.best_per_matrix()
    assert best == {}
