"""SQLite-backed parameter database (the "ExperimentRecord" ledger).

Durable, queryable substrate for every phase of MULTIAGENT_PBR_OPTIMIZATION.md:
Phase 0 uses it purely for dedup (skip already-measured config points) and
reporting; later phases (Bayesian optimization, Knowledge Curator retrieval)
read from the same table without any schema change needed.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from optimizer.grid import CandidateConfig

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "optimizer_ledger.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key        TEXT NOT NULL,
    matrix_profile    TEXT NOT NULL,
    size              INTEGER NOT NULL,
    features          INTEGER NOT NULL,
    block             INTEGER NOT NULL,
    min_nnz           INTEGER NOT NULL,
    smem_size         INTEGER NOT NULL,
    git_commit        TEXT,
    correctness_pass  INTEGER,
    metrics_json      TEXT,
    verdict           TEXT,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_config_key ON experiments(config_key);
CREATE INDEX IF NOT EXISTS idx_matrix_size_feat ON experiments(matrix_profile, size, features);
"""


class Ledger:
    """Thin wrapper around a single SQLite file. Not process-safe for concurrent
    writers beyond SQLite's own file-locking -- fine for Phase 0's single
    orchestrator process; revisit if Phase 4's parallel-agent scale-out needs
    concurrent writes from multiple processes."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def has_measured(self, config: CandidateConfig) -> bool:
        """The Static Verifier's cheap, zero-GPU-cost dedup check."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM experiments WHERE config_key = ? LIMIT 1",
                (config.config_key(),),
            ).fetchone()
        return row is not None

    def record(self, config: CandidateConfig, *, git_commit: Optional[str],
               correctness_pass: Optional[bool], metrics: Optional[dict],
               verdict: str) -> int:
        """Write one ExperimentRecord. Called whether the outcome is a pass,
        fail, or inconclusive -- negative results are recorded exactly like
        positive ones (see MULTIAGENT_PBR_OPTIMIZATION.md §0c)."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO experiments
                   (config_key, matrix_profile, size, features, block, min_nnz,
                    smem_size, git_commit, correctness_pass, metrics_json, verdict, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    config.config_key(), config.matrix_profile, config.size, config.features,
                    config.block, config.min_nnz, config.smem_size,
                    git_commit, None if correctness_pass is None else int(correctness_pass),
                    None if metrics is None else json.dumps(metrics),
                    verdict, time.time(),
                ),
            )
            return cur.lastrowid

    def all_records(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM experiments ORDER BY created_at").fetchall()
        out = []
        for row in rows:
            d = dict(row)
            if d.get("metrics_json"):
                d["metrics"] = json.loads(d["metrics_json"])
            out.append(d)
        return out

    def best_per_matrix(self, metric: str = "gpu_time_ns") -> dict[str, dict]:
        """Report helper: cheapest-`metric` record per (matrix_profile, size, features)."""
        best: dict[tuple, dict] = {}
        for rec in self.all_records():
            if rec.get("verdict") != "measured" or not rec.get("metrics"):
                continue
            value = rec["metrics"].get(metric)
            if value is None:
                continue
            key = (rec["matrix_profile"], rec["size"], rec["features"])
            if key not in best or value < best[key]["metrics"][metric]:
                best[key] = rec
        return {f"{m}/{s}/{f}": rec for (m, s, f), rec in best.items()}
