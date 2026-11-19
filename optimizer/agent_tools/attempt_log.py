"""Durable SQLite experiment and tool-call audit log for the optimization loop."""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
import uuid

from . import state, vcs
from .schema import validate
from .util import ArgumentError, InfrastructureError

MAX_ATTEMPTS_DEFAULT = 5
BUSY_TIMEOUT_MS = 30_000
MAX_SUMMARY_BYTES = 4096
MAX_ERROR_CHARS = 1024
MAX_ARG_STR_CHARS = 4096
PROMPT_REL_PATH = 'reports/agent_loop/optimization_loop_prompt.md'

SAFE_CONFIG_KEYS = (
    'loop_branch', 'approved_skips', 'exclude_nodes',
    'env_name', 'modules', 'agent_root', 'remote_host', 'remote_repo',
)

OUTCOME_ALIASES = {
    'pass': 'PASS',
    'rolled back': 'ROLLED BACK',
    'rolled_back': 'ROLLED BACK',
    'fail (build)': 'FAIL (build)',
    'fail_build': 'FAIL (build)',
    'fail:build': 'FAIL (build)',
    'fail (tests)': 'FAIL (tests)',
    'fail_tests': 'FAIL (tests)',
    'fail:tests': 'FAIL (tests)',
    'fail (correctness)': 'FAIL (correctness)',
    'fail_correctness': 'FAIL (correctness)',
    'fail:correctness': 'FAIL (correctness)',
    'aborted': 'ABORTED',
}
COUNTED_OUTCOMES = frozenset({
    'PASS', 'ROLLED BACK', 'FAIL (build)', 'FAIL (tests)', 'FAIL (correctness)',
})
SHA_RE = re.compile(r'^[0-9a-f]{7,40}$')

# Dictionary keys whose values are always secret. Matched on whole snake_case words of the
# normalized key (camelCase and '-' are folded to '_'), so 'max_tokens' or 'environment_name'
# are NOT redacted, while 'api_token', 'client_secret' or 'GITHUB_TOKEN' are.
SECRET_KEY_SUFFIXES = (
    'token', 'secret', 'password', 'passwd', 'pwd', 'passphrase', 'credential', 'credentials',
    'authorization', 'cookie', 'cookies', 'private_key', 'api_key', 'apikey', 'access_key',
    'secret_key', 'ssh_key', 'auth_header', 'bearer', 'session_key',
)
SECRET_KEY_EXACT = frozenset({
    'auth', 'env', 'env_vars', 'envvars', 'environ', 'environment', 'ssh_config',
})
# Kept for backward compatibility of imports; key matching uses _is_secret_key().
SECRET_KEY_RE = re.compile(r'(?!)')
INLINE_PRIVATE_KEY_RE = re.compile(
    r'-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|$)'
)
INLINE_BEARER_RE = re.compile(r'(?i)\b(Bearer|Basic|Token)(\s+)[A-Za-z0-9._~\-+/=]{8,}')
# Value-level secret patterns, applied to every stored string (arguments, result summaries,
# error text, free-text attempt fields). Each entry is (pattern, replacement); replacements keep
# the surrounding non-secret text (header names, flag names, variable names, URL host).
INLINE_SECRET_SUBS = (
    (INLINE_PRIVATE_KEY_RE, '[REDACTED_PRIVATE_KEY]'),
    (re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}'), '[REDACTED_GITHUB_TOKEN]'),
    (re.compile(r'\bgithub_pat_[A-Za-z0-9_]{20,}'), '[REDACTED_GITHUB_TOKEN]'),
    (re.compile(r'\bglpat-[A-Za-z0-9_\-]{20,}'), '[REDACTED_GITLAB_TOKEN]'),
    (re.compile(r'\bsk-[A-Za-z0-9_\-]{16,}'), '[REDACTED_API_KEY]'),
    (re.compile(r'\bhf_[A-Za-z0-9]{30,}'), '[REDACTED_HF_TOKEN]'),
    (re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b'), '[REDACTED_AWS_KEY_ID]'),
    (re.compile(r'\bAIza[0-9A-Za-z_\-]{35}'), '[REDACTED_GOOGLE_KEY]'),
    (re.compile(r'\bya29\.[0-9A-Za-z_\-]{20,}'), '[REDACTED_GOOGLE_TOKEN]'),
    (re.compile(r'\bxox[abprs]-[A-Za-z0-9\-]{10,}'), '[REDACTED_SLACK_TOKEN]'),
    (re.compile(r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}'), '[REDACTED_JWT]'),
    # scheme://user:password@host -> scheme://[REDACTED]@host
    (re.compile(r'([a-zA-Z][a-zA-Z0-9+.\-]*://)[^\s:@/]+:[^\s@/]+@'), r'\1[REDACTED]@'),
    # Authorization/Cookie headers: keep the header name.
    (re.compile(r'(?i)\b((?:Proxy-)?Authorization|Cookie|Set-Cookie|X-Api-Key|X-Auth-Token)(\s*[:=]\s*)'
                r'(?!\[REDACTED)[^\r\n"\']+'), r'\1\2[REDACTED]'),
    (INLINE_BEARER_RE, r'\1\2[REDACTED]'),
    # --password=x / --token x: keep the flag.
    (re.compile(r'(?i)(--(?:password|passwd|token|secret|api-key|apikey|access-key)(?:=|\s+))'
                r'(?!\[REDACTED)[^\s"\']+'), r'\1[REDACTED]'),
    # NAME=value / NAME: value where NAME names a secret (env vars, config lines, JSON-ish text).
    (re.compile(r'(?i)\b([A-Za-z0-9_]*?(?:TOKEN|SECRET|PASSW(?:OR)?D|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|'
                r'SECRET_?KEY|CREDENTIALS?|PASSPHRASE))(["\']?\s*[:=]\s*["\']?)'
                r'(?!\[REDACTED|\$\{?[A-Za-z_]|["\'\s,;}]|$)[^\s"\',;}]+'), r'\1\2[REDACTED]'),
)
SCRIPT_SECRET_PATTERNS = tuple(pat for pat, _ in INLINE_SECRET_SUBS)


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        branch TEXT NOT NULL,
        initial_verified_sha TEXT NOT NULL,
        prompt_path TEXT NOT NULL,
        prompt_sha256 TEXT NOT NULL,
        suite_manifest_sha256 TEXT,
        suite_identity TEXT,
        config_json TEXT NOT NULL,
        max_attempts INTEGER NOT NULL DEFAULT 5 CHECK (max_attempts >= 1),
        started_at TEXT NOT NULL,
        ended_at TEXT,
        status TEXT NOT NULL CHECK (status IN ('active', 'completed', 'aborted')),
        close_reason TEXT
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_single_active
        ON runs(status) WHERE status = 'active'
    """,
    """
    CREATE TABLE IF NOT EXISTS attempts (
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
        title TEXT NOT NULL,
        hypothesis TEXT NOT NULL,
        baseline_sha TEXT NOT NULL,
        start_head_sha TEXT,
        start_reflog_len INTEGER,
        candidate_shas_json TEXT NOT NULL DEFAULT '[]',
        final_candidate_sha TEXT,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        status TEXT NOT NULL CHECK (status IN ('active', 'closed')),
        outcome TEXT CHECK (outcome IS NULL OR outcome IN (
            'PASS', 'ROLLED BACK', 'FAIL (build)', 'FAIL (tests)',
            'FAIL (correctness)', 'ABORTED'
        )),
        promoted INTEGER NOT NULL DEFAULT 0 CHECK (promoted IN (0, 1)),
        rolled_back INTEGER NOT NULL DEFAULT 0 CHECK (rolled_back IN (0, 1)),
        decision_reason TEXT,
        changelog_sha TEXT,
        counts_toward_limit INTEGER NOT NULL DEFAULT 1 CHECK (counts_toward_limit IN (0, 1)),
        tune_bench_path TEXT,
        all_bench_path TEXT,
        metrics_summary_json TEXT,
        profile_paths_json TEXT NOT NULL DEFAULT '[]',
        UNIQUE (run_id, ordinal),
        UNIQUE (attempt_id, run_id),
        CHECK (NOT (promoted = 1 AND rolled_back = 1)),
        CHECK ((status = 'active' AND ended_at IS NULL AND outcome IS NULL)
               OR (status = 'closed' AND ended_at IS NOT NULL AND outcome IS NOT NULL))
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_attempts_single_active
        ON attempts(run_id) WHERE status = 'active'
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_calls (
        call_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
        attempt_id INTEGER REFERENCES attempts(attempt_id) ON DELETE RESTRICT,
        tool_name TEXT NOT NULL,
        args_json TEXT NOT NULL,
        args_sha256 TEXT NOT NULL,
        head_sha TEXT,
        verified_sha TEXT,
        pid INTEGER NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
        status TEXT NOT NULL CHECK (status IN ('running', 'success', 'failure', 'interrupted')),
        exit_code INTEGER,
        result_summary_json TEXT,
        error_class TEXT,
        error_detail TEXT,
        slurm_job_id INTEGER,
        slurm_resume_key TEXT,
        slurm_reattached INTEGER CHECK (slurm_reattached IS NULL OR slurm_reattached IN (0, 1)),
        artifact_paths_json TEXT NOT NULL DEFAULT '[]',
        FOREIGN KEY (attempt_id, run_id) REFERENCES attempts(attempt_id, run_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_calls_run_attempt
        ON tool_calls(run_id, attempt_id, call_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_calls_status
        ON tool_calls(status)
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
        attempt_id INTEGER REFERENCES attempts(attempt_id) ON DELETE RESTRICT,
        call_id INTEGER REFERENCES tool_calls(call_id) ON DELETE RESTRICT,
        kind TEXT NOT NULL,
        local_path TEXT,
        remote_path TEXT,
        sha256 TEXT,
        size_bytes INTEGER CHECK (size_bytes IS NULL OR size_bytes >= 0),
        created_at TEXT NOT NULL,
        CHECK (local_path IS NOT NULL OR remote_path IS NOT NULL),
        CHECK (attempt_id IS NOT NULL OR call_id IS NOT NULL),
        FOREIGN KEY (attempt_id, run_id) REFERENCES attempts(attempt_id, run_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_artifacts_attempt
        ON artifacts(attempt_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_artifacts_call
        ON artifacts(call_id)
    """,
) + (
    # Campaign-wide attempt budget: one row per ledger file. The five-attempt budget is shared by
    # every run in this attempts.sqlite3; only a user-authorized fresh ledger resets it.
    """
    CREATE TABLE IF NOT EXISTS campaign (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        budget INTEGER NOT NULL CHECK (budget >= 1),
        created_at TEXT NOT NULL
    )
    """,
    # Commits made through commit_kernel_attempt, bound to the attempt that made them.
    """
    CREATE TABLE IF NOT EXISTS attempt_commits (
        sha TEXT PRIMARY KEY,
        attempt_id INTEGER NOT NULL REFERENCES attempts(attempt_id) ON DELETE RESTRICT,
        call_id INTEGER REFERENCES tool_calls(call_id) ON DELETE RESTRICT,
        files_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_attempt_commits_attempt
        ON attempt_commits(attempt_id)
    """,
)
_MIGRATION_TABLES = ('campaign', 'attempt_commits')


@dataclass
class _CallContext:
    call_id: int
    run_id: str
    attempt_id: int | None
    db_path: Path
    root: Path
    slurm_job_id: int | None = None
    slurm_resume_key: str | None = None
    slurm_reattached: bool | None = None
    artifacts: list[tuple[str, str | None, str | None]] = field(default_factory=list)
    suite_manifest_sha256: str | None = None


@dataclass
class CallHandle:
    ctx: _CallContext
    token: contextvars.Token
    started_mono: float
    lock_file: object | None
    lock_path: Path | None


_ACTIVE_CALL: contextvars.ContextVar[_CallContext | None] = contextvars.ContextVar(
    'agent_tools_active_call', default=None
)


def default_db_path() -> Path:
    return state.STATE.parent / 'attempts.sqlite3'


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> tuple[str | None, int | None]:
    try:
        if path.is_file():
            data = path.read_bytes()
            return _sha256_bytes(data), len(data)
    except OSError:
        return None, None
    return None, None


def _rel_path(path: Path | str, root: Path) -> str:
    p = Path(path)
    try:
        resolved_root = root.resolve()
        resolved = p.resolve() if p.is_absolute() else (resolved_root / p).resolve()
        return str(resolved.relative_to(resolved_root))
    except ValueError:
        return str(p)


def _resolve_local(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else root / p


def _safe_git(repo: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        res = subprocess.run(
            ['git', *args], cwd=repo, text=True, capture_output=True
        )
        return res
    except OSError:
        return None


def _is_git_repo(repo: Path) -> bool:
    res = _safe_git(repo, 'rev-parse', '--is-inside-work-tree')
    return res is not None and res.returncode == 0 and res.stdout.strip() == 'true'


def _git_head(repo: Path) -> str | None:
    res = _safe_git(repo, 'rev-parse', 'HEAD')
    if res is not None and res.returncode == 0:
        return res.stdout.strip()
    return None


def _git_branch(repo: Path, cfg: dict) -> str:
    res = _safe_git(repo, 'branch', '--show-current')
    if res is not None and res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()
    return str(cfg.get('loop_branch') or 'unknown')


def _resolve_verified_sha(repo: Path, cfg: dict) -> str | None:
    verified = cfg.get('verified')
    if not verified:
        return None
    if _is_git_repo(repo):
        res = _safe_git(repo, 'rev-parse', f'{verified}^{{commit}}')
        if res is not None and res.returncode == 0:
            return res.stdout.strip()
    return str(verified)


def _chmod_owner_only(path: Path, mode: int) -> None:
    try:
        if path.exists():
            os.chmod(path, mode)
    except OSError:
        pass


def _is_secret_key(key: str) -> bool:
    norm = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', key)
    norm = re.sub(r'[^A-Za-z0-9]+', '_', norm).strip('_').lower()
    if norm in SECRET_KEY_EXACT:
        return True
    return any(norm == suf or norm.endswith('_' + suf) for suf in SECRET_KEY_SUFFIXES)


def _contains_embedded_secret(text: str) -> bool:
    return _redact_inline_secrets(text) != text


def _redact_inline_secrets(text: str) -> str:
    """Replace secret values anywhere in a string, keeping the surrounding text."""
    for pattern, replacement in INLINE_SECRET_SUBS:
        text = pattern.sub(replacement, text)
    return text


def redact_text(value):
    """Public helper: redact secret values in free text (None passes through)."""
    return None if value is None else _redact_inline_secrets(str(value))


def _save_script_artifact(script_content: str, db_path: Path, root: Path) -> tuple[str | None, str, int]:
    raw = script_content.encode('utf-8')
    digest = _sha256_bytes(raw)
    if _contains_embedded_secret(script_content):
        return None, digest, len(raw)
    scripts_dir = db_path.parent / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    _chmod_owner_only(db_path.parent, 0o700)
    _chmod_owner_only(scripts_dir, 0o700)
    target = scripts_dir / f'{digest[:16]}.sh'
    if not target.exists():
        tmp = scripts_dir / f'{digest[:16]}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp'
        tmp.write_bytes(raw)
        _chmod_owner_only(tmp, 0o600)
        os.replace(tmp, target)
    _chmod_owner_only(target, 0o600)
    return _rel_path(target, root), digest, len(raw)


def redact_and_sanitize(
    value,
    *,
    db_path: Path | None = None,
    root: Path | None = None,
    script_artifacts: list[tuple[str, str | None, str | None]] | None = None,
    max_str_chars: int = MAX_ARG_STR_CHARS,
):
    """Recursively redact secrets and offload/bound large script or string values."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key_str = str(k)
            if _is_secret_key(key_str):
                out[key_str] = '[REDACTED]'
            elif key_str == 'script_content' and isinstance(v, str) and db_path is not None and root is not None:
                rel, digest, nbytes = _save_script_artifact(v, db_path, root)
                if rel is not None:
                    if script_artifacts is not None:
                        script_artifacts.append(('slurm_script', rel, None))
                    out[key_str] = {
                        'redacted_type': 'script_artifact',
                        'artifact_path': rel,
                        'sha256': digest,
                        'bytes': nbytes,
                    }
                else:
                    out[key_str] = {
                        'redacted_type': 'script_omitted_sensitive',
                        'artifact_path': None,
                        'sha256': digest,
                        'bytes': nbytes,
                    }
            else:
                out[key_str] = redact_and_sanitize(
                    v,
                    db_path=db_path,
                    root=root,
                    script_artifacts=script_artifacts,
                    max_str_chars=max_str_chars,
                )
        return out
    if isinstance(value, list):
        return [
            redact_and_sanitize(
                item,
                db_path=db_path,
                root=root,
                script_artifacts=script_artifacts,
                max_str_chars=max_str_chars,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            redact_and_sanitize(
                item,
                db_path=db_path,
                root=root,
                script_artifacts=script_artifacts,
                max_str_chars=max_str_chars,
            )
            for item in value
        ]
    if isinstance(value, str):
        cleaned = _redact_inline_secrets(value)
        if len(cleaned) > max_str_chars:
            raw = cleaned.encode('utf-8')
            return {
                'truncated': True,
                'sha256': _sha256_bytes(raw),
                'bytes': len(raw),
                'preview': cleaned[:256],
            }
        return cleaned
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _redact_inline_secrets(str(value))[:max_str_chars]


def _cap_json(obj, max_bytes: int = MAX_SUMMARY_BYTES) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)
    raw = encoded.encode('utf-8')
    if len(raw) <= max_bytes:
        return encoded
    fallback = {
        'truncated': True,
        'original_bytes': len(raw),
        'sha256': _sha256_bytes(raw),
        'preview': encoded[: max(256, max_bytes // 4)],
    }
    return json.dumps(fallback, sort_keys=True, separators=(',', ':'))


def summarize_result(tool_name: str, result) -> str:
    """Produce a compact, bounded, secret-free JSON summary of a tool return value."""
    if not isinstance(result, dict):
        return _cap_json(redact_and_sanitize(result, max_str_chars=512))
    if 'error' in result and len(result) <= 4:
        summary = redact_and_sanitize(result, max_str_chars=512)
    elif tool_name == 'run_benchmarks' and result.get('v') == 'bench.1':
        summary = {
            'v': result.get('v'),
            'ok': result.get('ok'),
            'fmt': result.get('fmt'),
            'kernel': result.get('kernel'),
            'cand': result.get('cand'),
            'base': result.get('base'),
            'env': result.get('env'),
            'noise': result.get('noise'),
            'sum': result.get('sum'),
            'row_count': len(result.get('rows', [])),
            'skip_count': len(result.get('skipped', [])),
            'raw': result.get('raw'),
        }
    elif tool_name == 'parse_ncu_profile' and result.get('v') == 'ncu.1':
        kernels = result.get('k', [])
        top = kernels[0] if kernels else None
        summary = {
            'v': result.get('v'),
            'rep': result.get('rep'),
            'case': result.get('case'),
            'kernel_count': len(kernels),
            'top_kernel': {'name': top.get('name'), 'us': top.get('us'), 'share': top.get('share')}
            if isinstance(top, dict) else None,
            'flags': result.get('flags'),
            'missing_count': len(result.get('missing', [])),
            'd_keys': sorted(result.get('d', {}).keys()),
        }
    elif tool_name == 'submit_slurm_job':
        if 'rendered' in result:
            raw = str(result['rendered']).encode('utf-8')
            summary = {'dry_run': True, 'rendered_sha256': _sha256_bytes(raw), 'rendered_bytes': len(raw)}
        else:
            summary = {
                k: result.get(k)
                for k in ('job', 'state', 'exit', 'node', 'elapsed_s', 'tests_passed',
                          'resume_key', 'reattached', 'out_path', 'err_path')
                if k in result
            }
    elif tool_name == 'load_build_env':
        summary = {
            k: result.get(k)
            for k in ('ok', 'cached', 'sha', 'src_hash', 'so_sha256', 'build_s',
                      'warnings', 'job', 'reason')
            if k in result
        }
    elif tool_name == 'commit_kernel_attempt':
        summary = {
            k: result.get(k)
            for k in ('sha', 'branch', 'files', 'pushed', 'ignored', 'error')
            if k in result
        }
    elif tool_name == 'rollback_workspace':
        summary = {
            k: result.get(k)
            for k in ('reverted_from', 'restored_to', 'revert_sha', 'discarded')
            if k in result
        }
    elif tool_name == 'promote_attempt':
        summary = {
            k: result.get(k)
            for k in ('promoted', 'cand', 'verified', 'previous_verified', 'reasons')
            if k in result
        }
    elif tool_name in ('begin_attempt', 'finish_attempt', 'export_attempt_log'):
        summary = redact_and_sanitize(result, max_str_chars=512)
    elif tool_name == 'get_loop_status':
        summary = {
            'run_id': (result.get('run') or {}).get('run_id'),
            'active_attempt_ordinal': (result.get('active_attempt') or {}).get('ordinal'),
            'counted_attempts_used': result.get('counted_attempts_used'),
            'counted_attempts_remaining': result.get('counted_attempts_remaining'),
            'unfinished_tool_call_count': len(result.get('unfinished_tool_calls', [])),
        }
    else:
        summary = redact_and_sanitize(result, max_str_chars=512)
    return _cap_json(redact_and_sanitize(summary, max_str_chars=512), MAX_SUMMARY_BYTES)


def _connect(db_path: Path) -> sqlite3.Connection:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _chmod_owner_only(db_path.parent, 0o700)
        conn = sqlite3.connect(
            str(db_path),
            timeout=BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA synchronous = FULL')
        conn.execute(f'PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}')
        _ensure_schema(conn)
        _chmod_owner_only(db_path, 0o600)
        _chmod_owner_only(Path(f'{db_path}-wal'), 0o600)
        _chmod_owner_only(Path(f'{db_path}-shm'), 0o600)
        return conn
    except (sqlite3.Error, OSError) as exc:
        raise InfrastructureError(f'attempt database unavailable at {db_path}: {exc}') from exc


def _schema_current(conn: sqlite3.Connection) -> bool:
    tables = {r['name'] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {'artifacts', *_MIGRATION_TABLES} <= tables:
        return False
    run_cols = {r['name'] for r in conn.execute('PRAGMA table_info(runs)').fetchall()}
    att_cols = {r['name'] for r in conn.execute('PRAGMA table_info(attempts)').fetchall()}
    if 'close_reason' not in run_cols or not {'start_head_sha', 'start_reflog_len'} <= att_cols:
        return False
    return conn.execute('SELECT 1 FROM campaign WHERE id = 1').fetchone() is not None


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create a fresh schema or migrate an older ledger in place (idempotent)."""
    if _schema_current(conn):
        return
    with _write_tx(conn):
        tables = {r['name'] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        fresh = 'artifacts' not in tables
        for stmt in _SCHEMA_STATEMENTS:
            conn.execute(stmt)
        if not fresh:
            run_cols = {r['name'] for r in conn.execute('PRAGMA table_info(runs)').fetchall()}
            if 'close_reason' not in run_cols:
                conn.execute('ALTER TABLE runs ADD COLUMN close_reason TEXT')
            att_cols = {r['name'] for r in conn.execute('PRAGMA table_info(attempts)').fetchall()}
            if 'start_head_sha' not in att_cols:
                conn.execute('ALTER TABLE attempts ADD COLUMN start_head_sha TEXT')
            if 'start_reflog_len' not in att_cols:
                conn.execute('ALTER TABLE attempts ADD COLUMN start_reflog_len INTEGER')
        conn.execute(
            'INSERT OR IGNORE INTO campaign (id, budget, created_at) VALUES (1, ?, ?)',
            (MAX_ATTEMPTS_DEFAULT, _now_utc()),
        )


def _campaign_counts(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """(budget, counted_used, remaining) aggregated across ALL runs in this ledger."""
    row = conn.execute('SELECT budget FROM campaign WHERE id = 1').fetchone()
    budget = int(row['budget']) if row is not None else MAX_ATTEMPTS_DEFAULT
    used = int(conn.execute(
        "SELECT COUNT(*) FROM attempts WHERE status = 'closed' AND counts_toward_limit = 1"
    ).fetchone()[0])
    return budget, used, max(0, budget - used)


@contextlib.contextmanager
def _write_tx(conn: sqlite3.Connection):
    conn.execute('BEGIN IMMEDIATE')
    try:
        yield
        conn.execute('COMMIT')
    except BaseException:
        if conn.in_transaction:
            conn.execute('ROLLBACK')
        raise


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _call_lock_path(db_path: Path, call_id: int) -> Path:
    return db_path.parent / 'call_locks' / f'{call_id}.lock'


def _acquire_call_lock(db_path: Path, call_id: int) -> tuple[object, Path]:
    lock_path = _call_lock_path(db_path, call_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_owner_only(lock_path.parent, 0o700)
    fh = lock_path.open('a+')
    _chmod_owner_only(lock_path, 0o600)
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    fh.seek(0)
    fh.truncate()
    fh.write(f'{os.getpid()}\n')
    fh.flush()
    return fh, lock_path


def _release_call_lock(lock_file, lock_path: Path | None) -> None:
    if lock_file is not None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            lock_file.close()
        except OSError:
            pass
    if lock_path is not None:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _is_call_dead(db_path: Path, call_id: int, pid: int) -> bool:
    lock_path = _call_lock_path(db_path, call_id)
    if lock_path.exists():
        try:
            with lock_path.open('a+') as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            return False
        except OSError:
            return not _pid_alive(pid)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        return True
    if pid == os.getpid():
        return True
    return not _pid_alive(pid)


def recover_interrupted_calls(
    conn: sqlite3.Connection, db_path: Path, *, exclude_call_id: int | None = None
) -> list[int]:
    """Replay pending audit-recovery records, then mark dead 'running' calls 'interrupted'."""
    reconcile_audit_recovery(conn, db_path)
    rows = conn.execute(
        "SELECT call_id, pid FROM tool_calls WHERE status = 'running'"
    ).fetchall()
    dead_ids = [
        int(row['call_id'])
        for row in rows
        if int(row['call_id']) != exclude_call_id
        and _is_call_dead(db_path, int(row['call_id']), int(row['pid']))
    ]
    if not dead_ids:
        return []
    now = _now_utc()
    with _write_tx(conn):
        for cid in dead_ids:
            conn.execute(
                """
                UPDATE tool_calls
                   SET status = 'interrupted',
                       ended_at = ?,
                       error_class = 'ProcessInterrupted',
                       error_detail = 'process exited before finalizing tool call'
                 WHERE call_id = ? AND status = 'running'
                """,
                (now, cid),
            )
    return dead_ids


def _safe_config_snapshot(cfg: dict) -> dict:
    snap = {k: cfg[k] for k in SAFE_CONFIG_KEYS if k in cfg}
    return redact_and_sanitize(snap)


def _prompt_digest(root: Path) -> str:
    p = root / PROMPT_REL_PATH
    digest, _ = _file_digest(p)
    return digest or ''


def _create_run_in_tx(
    conn: sqlite3.Connection, *, root: Path, cfg: dict, max_attempts: int = MAX_ATTEMPTS_DEFAULT
) -> sqlite3.Row:
    now = _now_utc()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'{stamp}-{uuid.uuid4().hex[:8]}'
    branch = _git_branch(root, cfg)
    verified = _resolve_verified_sha(root, cfg) or _git_head(root) or 'unknown'
    suite_sha = cfg.get('suite_manifest_sha256')
    suite_identity = json.dumps(
        {
            'manifest_remote': f"{str(cfg.get('agent_root', '')).rstrip('/')}/suite/manifest.json",
            'approved_skips': list(cfg.get('approved_skips', [])),
        },
        sort_keys=True,
        separators=(',', ':'),
    )
    conn.execute(
        """
        INSERT INTO runs (
            run_id, branch, initial_verified_sha, prompt_path, prompt_sha256,
            suite_manifest_sha256, suite_identity, config_json, max_attempts,
            started_at, status, close_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL)
        """,
        (
            run_id,
            branch,
            verified,
            PROMPT_REL_PATH,
            _prompt_digest(root),
            suite_sha,
            suite_identity,
            json.dumps(_safe_config_snapshot(cfg), sort_keys=True, separators=(',', ':')),
            max_attempts,
            now,
        ),
    )
    return conn.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()


def _select_or_init_run_in_tx(
    conn: sqlite3.Connection, *, root: Path, cfg: dict, max_attempts: int = MAX_ATTEMPTS_DEFAULT
) -> sqlite3.Row:
    """Return active run if present; else latest closed run; create initial run only when DB has no runs."""
    row = _select_run(conn, None)
    if row is not None:
        return row
    return _create_run_in_tx(conn, root=root, cfg=cfg, max_attempts=max_attempts)


def _select_run(conn: sqlite3.Connection, run_id: str | None) -> sqlite3.Row | None:
    if run_id is not None:
        return conn.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
    row = conn.execute(
        "SELECT * FROM runs WHERE status = 'active' ORDER BY started_at DESC, rowid DESC LIMIT 1"
    ).fetchone()
    if row is not None:
        return row
    return conn.execute('SELECT * FROM runs ORDER BY started_at DESC, rowid DESC LIMIT 1').fetchone()


def _insert_artifact_in_tx(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    attempt_id: int | None,
    call_id: int | None,
    kind: str,
    local_path: str | None,
    remote_path: str | None,
    root: Path,
) -> None:
    if local_path is None and remote_path is None:
        return
    if attempt_id is None and call_id is None:
        return
    norm_local = None
    digest = None
    size_bytes = None
    if local_path is not None:
        resolved = _resolve_local(local_path, root)
        norm_local = _rel_path(resolved, root)
        digest, size_bytes = _file_digest(resolved)
    conn.execute(
        """
        INSERT INTO artifacts (
            run_id, attempt_id, call_id, kind, local_path, remote_path,
            sha256, size_bytes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, attempt_id, call_id, kind, norm_local, remote_path, digest, size_bytes, _now_utc()),
    )


def attach_slurm_info(
    *,
    job_id: int | None = None,
    resume_key: str | None = None,
    reattached: bool | None = None,
    out_path: str | None = None,
    err_path: str | None = None,
) -> None:
    """Attach Slurm execution metadata to the active top-level CLI tool call."""
    ctx = _ACTIVE_CALL.get()
    if ctx is None:
        return
    if job_id is not None:
        ctx.slurm_job_id = int(job_id)
    if resume_key is not None:
        ctx.slurm_resume_key = str(resume_key)
    if reattached is not None:
        ctx.slurm_reattached = bool(reattached)
    if out_path:
        ctx.artifacts.append(('slurm_out', None, str(out_path)))
    if err_path:
        ctx.artifacts.append(('slurm_err', None, str(err_path)))


def attach_artifact(
    *,
    kind: str,
    local_path: str | Path | None = None,
    remote_path: str | None = None,
) -> None:
    """Attach an artifact reference to the active top-level CLI tool call."""
    ctx = _ACTIVE_CALL.get()
    if ctx is None:
        return
    ctx.artifacts.append((
        kind,
        str(local_path) if local_path is not None else None,
        str(remote_path) if remote_path is not None else None,
    ))


def record_suite_manifest_sha256(
    digest: str, *, db_path: Path | None = None, run_id: str | None = None
) -> None:
    """Record the suite manifest SHA-256 on the run; raise InfrastructureError on mid-run drift."""
    if not isinstance(digest, str) or not digest.strip():
        raise ArgumentError('suite_manifest_sha256 must be a nonempty string')
    ctx = _ACTIVE_CALL.get()
    target = db_path or (ctx.db_path if ctx is not None else default_db_path())
    target_run_id = run_id or (ctx.run_id if ctx is not None else None)
    if not target.exists():
        if ctx is not None:
            ctx.suite_manifest_sha256 = digest
        return
    conn = _connect(target)
    try:
        with _write_tx(conn):
            row = (
                conn.execute(
                    'SELECT run_id, suite_manifest_sha256 FROM runs WHERE run_id = ?',
                    (target_run_id,),
                ).fetchone()
                if target_run_id
                else conn.execute(
                    "SELECT run_id, suite_manifest_sha256 FROM runs WHERE status = 'active' ORDER BY started_at DESC, rowid DESC LIMIT 1"
                ).fetchone()
            )
            if row is None:
                return
            existing = row['suite_manifest_sha256']
            if existing is not None and existing != digest:
                raise InfrastructureError(
                    f"suite manifest drift detected for run {row['run_id']}: "
                    f'expected sha256 {existing[:16]}, got {digest[:16]}'
                )
            if existing is None:
                conn.execute(
                    'UPDATE runs SET suite_manifest_sha256 = ? WHERE run_id = ?',
                    (digest, row['run_id']),
                )
    finally:
        conn.close()
    if ctx is not None:
        ctx.suite_manifest_sha256 = digest


def start_tool_call(
    tool_name: str,
    arguments: dict,
    *,
    db_path: Path | None = None,
    state_path: Path | None = None,
    repo: Path | None = None,
) -> CallHandle:
    """Insert a 'running' audit record before dispatching a top-level CLI tool."""
    root = repo or state.ROOT
    target_db = db_path or default_db_path()
    cfg = state.read(state_path) if state_path is not None else state.read()
    script_artifacts: list[tuple[str, str | None, str | None]] = []
    sanitized_args = redact_and_sanitize(
        arguments,
        db_path=target_db,
        root=root,
        script_artifacts=script_artifacts,
    )
    args_json = json.dumps(sanitized_args, sort_keys=True, separators=(',', ':'))
    args_sha = _sha256_bytes(args_json.encode('utf-8'))
    head_sha = _git_head(root)
    verified_sha = _resolve_verified_sha(root, cfg)
    now = _now_utc()
    pid = os.getpid()

    conn = _connect(target_db)
    try:
        recover_interrupted_calls(conn, target_db)
        with _write_tx(conn):
            run = _select_or_init_run_in_tx(conn, root=root, cfg=cfg)
            run_id = str(run['run_id'])
            active = conn.execute(
                "SELECT attempt_id FROM attempts WHERE run_id = ? AND status = 'active'",
                (run_id,),
            ).fetchone()
            attempt_id = int(active['attempt_id']) if active is not None else None
            cur = conn.execute(
                """
                INSERT INTO tool_calls (
                    run_id, attempt_id, tool_name, args_json, args_sha256,
                    head_sha, verified_sha, pid, started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (run_id, attempt_id, tool_name, args_json, args_sha,
                 head_sha, verified_sha, pid, now),
            )
            call_id = int(cur.lastrowid)
            for kind, local_path, remote_path in script_artifacts:
                _insert_artifact_in_tx(
                    conn,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    call_id=call_id,
                    kind=kind,
                    local_path=local_path,
                    remote_path=remote_path,
                    root=root,
                )
        lock_file, lock_path = _acquire_call_lock(target_db, call_id)
    except InfrastructureError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise InfrastructureError(f'failed to write tool_call start record: {exc}') from exc
    finally:
        conn.close()

    ctx = _CallContext(
        call_id=call_id,
        run_id=run_id,
        attempt_id=attempt_id,
        db_path=target_db,
        root=root,
        artifacts=list(script_artifacts),
    )
    token = _ACTIVE_CALL.set(ctx)
    return CallHandle(
        ctx=ctx,
        token=token,
        started_mono=time.monotonic(),
        lock_file=lock_file,
        lock_path=lock_path,
    )


def finish_tool_call(
    handle: CallHandle,
    *,
    tool_name: str,
    exit_code: int,
    result: dict | None = None,
    error: BaseException | None = None,
) -> None:
    """Finalize the 'running' tool_calls row, attaching artifacts and attempt updates."""
    ctx = handle.ctx
    duration_ms = max(0, int((time.monotonic() - handle.started_mono) * 1000))
    ended_at = _now_utc()
    status = 'success' if exit_code == 0 and error is None else 'failure'
    summary_json = summarize_result(tool_name, result) if result is not None else None
    err_class = type(error).__name__ if error is not None else None
    err_detail = None
    if error is not None:
        err_detail = _redact_inline_secrets(str(error))[:MAX_ERROR_CHARS]
    elif isinstance(result, dict) and 'error' in result and exit_code != 0:
        err_class = str(result.get('error'))
        err_detail = _redact_inline_secrets(str(result.get('detail', '')))[:MAX_ERROR_CHARS]

    # Pull Slurm fields from result if not already attached via context.
    if isinstance(result, dict):
        if ctx.slurm_job_id is None:
            if tool_name == 'submit_slurm_job' and isinstance(result.get('job'), int):
                ctx.slurm_job_id = result['job']
            elif tool_name == 'run_benchmarks' and isinstance(result.get('env'), dict):
                job = result['env'].get('job')
                if isinstance(job, int):
                    ctx.slurm_job_id = job
            elif tool_name == 'load_build_env' and isinstance(result.get('job'), int):
                ctx.slurm_job_id = result['job']
        if ctx.slurm_resume_key is None and isinstance(result.get('resume_key'), str):
            ctx.slurm_resume_key = result['resume_key']
        if ctx.slurm_reattached is None and isinstance(result.get('reattached'), bool):
            ctx.slurm_reattached = result['reattached']

    # Associate begin_attempt's own call (and in-memory context) with the run and attempt it created.
    orig_run_id = ctx.run_id
    orig_attempt_id = ctx.attempt_id
    run_id = orig_run_id
    attempt_id = orig_attempt_id
    if tool_name == 'begin_attempt' and status == 'success' and isinstance(result, dict):
        if isinstance(result.get('run_id'), str):
            run_id = result['run_id']
        if isinstance(result.get('attempt_id'), int):
            attempt_id = result['attempt_id']
        ctx.run_id = run_id
        ctx.attempt_id = attempt_id

    # Collect artifact paths list (deduplicated, order-preserving).
    artifact_paths: list[str] = []
    seen_art: set[tuple[str, str | None, str | None]] = set()
    new_artifacts: list[tuple[str, str | None, str | None]] = []
    for kind, local_path, remote_path in ctx.artifacts:
        norm_local = _rel_path(local_path, ctx.root) if local_path else None
        key = (kind, norm_local, remote_path)
        if key in seen_art:
            continue
        seen_art.add(key)
        # Skip slurm_script because it was already inserted in start_tool_call.
        if kind != 'slurm_script':
            new_artifacts.append((kind, norm_local, remote_path))
        display = norm_local or remote_path
        if display and display not in artifact_paths:
            artifact_paths.append(display)

    commit_info = None
    if (
        tool_name == 'commit_kernel_attempt'
        and status == 'success'
        and isinstance(result, dict)
        and isinstance(result.get('sha'), str)
    ):
        commit_info = {'sha': result['sha'], 'files': [str(f) for f in (result.get('files') or [])]}
    payload = {
        'v': 1,
        'tool_name': tool_name,
        'call_id': ctx.call_id,
        'run_id': run_id,
        'orig_run_id': orig_run_id,
        'attempt_id': attempt_id,
        'orig_attempt_id': orig_attempt_id,
        'ended_at': ended_at,
        'duration_ms': duration_ms,
        'status': status,
        'exit_code': exit_code,
        'summary_json': summary_json,
        'err_class': err_class,
        'err_detail': err_detail,
        'slurm_job_id': ctx.slurm_job_id,
        'slurm_resume_key': ctx.slurm_resume_key,
        'slurm_reattached': None if ctx.slurm_reattached is None else int(ctx.slurm_reattached),
        'artifact_paths': artifact_paths,
        'new_artifacts': [list(a) for a in new_artifacts],
        'suite_manifest_sha256': ctx.suite_manifest_sha256,
        'commit_info': commit_info,
        'root': str(ctx.root),
    }

    storage_exc: BaseException | None = None
    try:
        try:
            conn = _connect(ctx.db_path)
        except InfrastructureError as exc:
            storage_exc = exc
        else:
            try:
                _finalize_tx(conn, payload)
            except (sqlite3.Error, OSError) as exc:
                storage_exc = exc
            finally:
                conn.close()
        if storage_exc is not None:
            record = _write_audit_recovery(ctx.db_path, payload) if exit_code == 0 else None
            raise AuditFinalizationError(
                f'failed to finalize tool_call {ctx.call_id}: {storage_exc}', record
            ) from storage_exc
    finally:
        _ACTIVE_CALL.reset(handle.token)
        _release_call_lock(handle.lock_file, handle.lock_path)


class AuditFinalizationError(InfrastructureError):
    """The tool's own work finished, but its audit row could not be written.

    `recovery_record` is the path of the emergency record that the next agent_tools invocation
    replays into the ledger (None if even that could not be written, or the tool had failed).
    """

    def __init__(self, message: str, recovery_record: str | None = None):
        super().__init__(message)
        self.recovery_record = recovery_record


def _recovery_dir(db_path: Path) -> Path:
    return db_path.parent / 'audit_recovery'


def _write_audit_recovery(db_path: Path, payload: dict) -> str | None:
    """Persist a finalization payload atomically; return its path or None on failure."""
    try:
        target_dir = _recovery_dir(db_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        _chmod_owner_only(target_dir, 0o700)
        target = target_dir / f"{int(payload['call_id'])}.json"
        tmp = target_dir / f"{int(payload['call_id'])}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(payload, sort_keys=True, separators=(',', ':')), encoding='utf-8')
        _chmod_owner_only(tmp, 0o600)
        os.replace(tmp, target)
        return str(target)
    except OSError:
        return None


def reconcile_audit_recovery(conn: sqlite3.Connection, db_path: Path) -> list[int]:
    """Replay emergency finalization records into the ledger; delete each once applied."""
    directory = _recovery_dir(db_path)
    if not directory.is_dir():
        return []
    applied = []
    for record in sorted(directory.glob('*.json')):
        try:
            payload = json.loads(record.read_text(encoding='utf-8'))
            _finalize_tx(conn, payload, replay=True)
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError, InfrastructureError):
            continue
        try:
            record.unlink()
        except FileNotFoundError:
            pass
        applied.append(int(payload['call_id']))
    return applied


def _finalize_tx(conn: sqlite3.Connection, payload: dict, *, replay: bool = False) -> None:
    """Write one tool call's final state (idempotent, so a recovery replay is safe)."""
    run_id, attempt_id = payload['run_id'], payload['attempt_id']
    root = Path(payload['root'])
    with _write_tx(conn):
        if attempt_id is not None:
            att_owner = conn.execute(
                'SELECT run_id FROM attempts WHERE attempt_id = ?',
                (attempt_id,),
            ).fetchone()
            if att_owner is None or str(att_owner['run_id']) != run_id:
                raise InfrastructureError(
                    f"attempt/run mismatch for call {payload['call_id']}: "
                    f'attempt {attempt_id} does not belong to run {run_id}'
                )
        conn.execute(
            """
            UPDATE tool_calls
               SET run_id = ?,
                   attempt_id = ?,
                   ended_at = ?,
                   duration_ms = ?,
                   status = ?,
                   exit_code = ?,
                   result_summary_json = ?,
                   error_class = ?,
                   error_detail = ?,
                   slurm_job_id = ?,
                   slurm_resume_key = ?,
                   slurm_reattached = ?,
                   artifact_paths_json = ?
             WHERE call_id = ?
            """,
            (
                run_id,
                attempt_id,
                payload['ended_at'],
                payload['duration_ms'],
                payload['status'],
                payload['exit_code'],
                payload['summary_json'],
                payload['err_class'],
                payload['err_detail'],
                payload['slurm_job_id'],
                payload['slurm_resume_key'],
                payload['slurm_reattached'],
                json.dumps(payload['artifact_paths'], separators=(',', ':')),
                payload['call_id'],
            ),
        )
        if run_id != payload['orig_run_id'] or attempt_id != payload['orig_attempt_id']:
            conn.execute(
                'UPDATE artifacts SET run_id = ?, attempt_id = ? WHERE call_id = ?',
                (run_id, attempt_id, payload['call_id']),
            )
        existing_arts = set()
        if replay:
            existing_arts = {
                (r['kind'], r['local_path'], r['remote_path'])
                for r in conn.execute(
                    'SELECT kind, local_path, remote_path FROM artifacts WHERE call_id = ?',
                    (payload['call_id'],),
                )
            }
        for kind, local_path, remote_path in payload['new_artifacts']:
            if (kind, local_path, remote_path) in existing_arts:
                continue
            _insert_artifact_in_tx(
                conn,
                run_id=run_id,
                attempt_id=attempt_id,
                call_id=payload['call_id'],
                kind=kind,
                local_path=local_path,
                remote_path=remote_path,
                root=root,
            )
        suite_sha = payload.get('suite_manifest_sha256')
        if suite_sha:
            existing_row = conn.execute(
                'SELECT suite_manifest_sha256 FROM runs WHERE run_id = ?',
                (run_id,),
            ).fetchone()
            if existing_row is not None:
                existing = existing_row['suite_manifest_sha256']
                if existing is not None and existing != suite_sha and not replay:
                    raise InfrastructureError(
                        f'suite manifest drift detected for run {run_id}: '
                        f'expected sha256 {existing[:16]}, got {suite_sha[:16]}'
                    )
                if existing is None:
                    conn.execute(
                        'UPDATE runs SET suite_manifest_sha256 = ? WHERE run_id = ?',
                        (suite_sha, run_id),
                    )
        commit_info = payload.get('commit_info')
        if commit_info and attempt_id is not None:
            sha, files = commit_info['sha'], list(commit_info['files'])
            # Every governed commit (kernel or CHANGELOG) is bound to its attempt; this is the only
            # source finish_attempt accepts for changelog_sha.
            conn.execute(
                """
                INSERT OR IGNORE INTO attempt_commits (sha, attempt_id, call_id, files_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sha, attempt_id, payload['call_id'], json.dumps(files, separators=(',', ':')),
                 payload['ended_at']),
            )
            # Kernel/test candidate commits (including compile fixes) also go into candidate_shas;
            # pure CHANGELOG commits do not.
            if [f for f in files if f != 'CHANGELOG.md']:
                row = conn.execute(
                    'SELECT candidate_shas_json FROM attempts WHERE attempt_id = ?',
                    (attempt_id,),
                ).fetchone()
                if row is not None:
                    shas = json.loads(row['candidate_shas_json'] or '[]')
                    if sha not in shas:
                        shas.append(sha)
                    conn.execute(
                        'UPDATE attempts SET candidate_shas_json = ?, final_candidate_sha = ? WHERE attempt_id = ?',
                        (json.dumps(shas, separators=(',', ':')), sha, attempt_id),
                    )


def _normalize_outcome(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArgumentError('outcome must be a nonempty string')
    key = re.sub(r'\s+', ' ', value.strip()).lower()
    if key not in OUTCOME_ALIASES:
        allowed = sorted(set(OUTCOME_ALIASES.values()))
        raise ArgumentError(f'unknown outcome {value!r}; expected one of {allowed}')
    return OUTCOME_ALIASES[key]


def _validate_bench_file(path_str: str, root: Path) -> tuple[str, dict]:
    p = _resolve_local(path_str, root)
    try:
        data = json.loads(p.read_text())
        schema = json.loads((Path(__file__).parent / 'schemas/bench.v1.json').read_text())
        validate(data, schema)
    except (OSError, ValueError, TypeError) as exc:
        raise ArgumentError(f'invalid bench.1 file {path_str!r}: {exc}') from exc
    return _rel_path(p, root), data


def _extract_metrics_summary(data: dict) -> dict:
    return {
        'fmt': data.get('fmt'),
        'kernel': data.get('kernel'),
        'cand': data.get('cand'),
        'base': data.get('base'),
        'ok': data.get('ok'),
        'noise': data.get('noise'),
        'job': (data.get('env') or {}).get('job'),
        'node': (data.get('env') or {}).get('node'),
        'raw': data.get('raw'),
        'sum': data.get('sum'),
    }


def _sha_matches(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def _dirty_attempt_code_paths(repo: Path) -> list[str]:
    """Return uncommitted (staged, unstaged, or untracked) attempt source/test paths."""
    res = _safe_git(repo, 'status', '--porcelain', '-z', '-uall')
    if res is None or res.returncode != 0:
        raise ArgumentError('could not inspect git working tree status')
    raw = res.stdout.split('\0')
    dirty = []
    i = 0
    while i < len(raw) and raw[i]:
        record = raw[i]
        kind, path = record[:2], record[3:]
        if 'R' in kind or 'C' in kind:
            if vcs.is_attempt_code_path(path):
                dirty.append(path)
            i += 1
            if i < len(raw) and raw[i] and vcs.is_attempt_code_path(raw[i]):
                dirty.append(raw[i])
        elif vcs.is_attempt_code_path(path):
            dirty.append(path)
        i += 1
    return sorted(set(dirty))


def _drifted_attempt_code_paths(repo: Path, baseline_sha: str) -> list[str]:
    """Return committed attempt source/test paths differing between baseline_sha and HEAD."""
    rev = _safe_git(repo, 'rev-parse', f'{baseline_sha}^{{commit}}')
    if rev is None or rev.returncode != 0:
        raise ArgumentError(f'baseline {baseline_sha!r} does not resolve to a commit')
    diff = _safe_git(
        repo, 'diff', '--name-only', baseline_sha, 'HEAD', '--',
        'torch_fast_spdmm', 'tests/torch_fast_spdmm',
    )
    if diff is None or diff.returncode != 0:
        raise ArgumentError(f'could not diff HEAD against baseline {baseline_sha[:7]}')
    return sorted(p for p in diff.stdout.splitlines() if vcs.is_attempt_code_path(p))


def _is_ancestor(repo: Path, older: str, newer: str) -> bool:
    res = _safe_git(repo, 'merge-base', '--is-ancestor', older, newer)
    return res is not None and res.returncode == 0


def _touches_attempt_paths(repo: Path, sha: str) -> bool:
    res = _safe_git(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', '-m', '--root', sha)
    if res is None or res.returncode != 0:
        return True  # cannot prove otherwise: fail closed
    return any(vcs.is_attempt_code_path(p) for p in res.stdout.splitlines())


def _head_reflog(repo: Path) -> list[str] | None:
    """HEAD reflog commit ids, newest first (None if unavailable)."""
    res = _safe_git(repo, 'reflog', 'show', '--format=%H', 'HEAD')
    if res is None or res.returncode != 0:
        return None
    return res.stdout.split()


def _attempt_path_commits_since(repo: Path, start_head: str, start_reflog_len: int | None) -> list[str]:
    """Commits touching attempt source/test paths made since the attempt began.

    Covers commits made outside commit_kernel_attempt (plain git), including ones later reverted
    (still in history since start_head) or reset away (still in the HEAD reflog entries added
    after begin_attempt recorded the reflog length). Returns sentinel entries, so the caller fails
    closed, when history no longer descends from the start or the reflog was truncated.
    """
    found: list[str] = []
    head = _git_head(repo)
    if head is None:
        return ['<no HEAD>']
    if not _is_ancestor(repo, start_head, head):
        found.append(f'<history rewritten: HEAD {head[:7]} does not descend from start {start_head[:7]}>')
    log = _safe_git(repo, 'rev-list', f'{start_head}..HEAD')
    for sha in (log.stdout.split() if log is not None and log.returncode == 0 else []):
        if _touches_attempt_paths(repo, sha):
            found.append(sha)
    if start_reflog_len is not None:
        reflog = _head_reflog(repo)
        if reflog is None or len(reflog) < start_reflog_len:
            found.append('<HEAD reflog unavailable or truncated since the attempt began>')
        else:
            for sha in reflog[:len(reflog) - start_reflog_len]:
                if sha in found or _is_ancestor(repo, sha, start_head):
                    continue
                if _touches_attempt_paths(repo, sha):
                    found.append(sha)
    return found


def _verify_clean_starting_point(repo: Path, cfg: dict, baseline_sha: str) -> None:
    """Enforce branch == state.loop_branch != main, clean attempt paths, HEAD == verified."""
    if not _is_git_repo(repo):
        return
    branch_res = _safe_git(repo, 'branch', '--show-current')
    branch = branch_res.stdout.strip() if branch_res and branch_res.returncode == 0 else ''
    expected = cfg.get('loop_branch')
    if branch == 'main' or not expected or branch != expected:
        raise ArgumentError(
            f'current branch {branch!r} must equal state.loop_branch {expected!r} and must not be main'
        )
    dirty = _dirty_attempt_code_paths(repo)
    if dirty:
        raise ArgumentError(
            f'uncommitted changes exist in attempt source/test paths: {dirty}; '
            'call clean_attempt_workspace (allowed only with no active attempt) to discard them, '
            'then retry begin_attempt'
        )
    drifted = _drifted_attempt_code_paths(repo, baseline_sha)
    if drifted:
        raise ArgumentError(
            f'HEAD differs from state.verified ({baseline_sha[:7]}) on attempt source/test paths: '
            f'{drifted}; committed drift needs user intervention (no tool reverts committed '
            'attempt-path changes outside an active attempt)'
        )


def begin_attempt(
    title: str,
    hypothesis: str,
    *,
    run_id: str | None = None,
    new_run: bool = False,
    db_path: Path | None = None,
    state_path: Path | None = None,
    repo: Path | None = None,
) -> dict:
    """Start a new optimization attempt after validating clean baseline, run status, and 5-attempt cap."""
    if not isinstance(title, str) or not title.strip():
        raise ArgumentError('title must be a nonempty string')
    title = ' '.join(title.strip().split())
    if len(title) > 200:
        raise ArgumentError('title must be at most 200 characters')
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ArgumentError('hypothesis must be a nonempty one-sentence string')
    if '\n\n' in hypothesis.strip():
        raise ArgumentError('hypothesis must be a single sentence, not multiple paragraphs')
    hypothesis = ' '.join(hypothesis.strip().split())
    title = _redact_inline_secrets(title)
    hypothesis = _redact_inline_secrets(hypothesis)
    if len(hypothesis) > 1000:
        raise ArgumentError('hypothesis must be at most 1000 characters')
    if not isinstance(new_run, bool):
        raise ArgumentError('new_run must be a boolean')
    if run_id is not None and new_run:
        raise ArgumentError('cannot specify both run_id and new_run=true')

    root = repo or state.ROOT
    target_db = db_path or default_db_path()
    cfg = state.read(state_path) if state_path is not None else state.read()
    baseline_sha = _resolve_verified_sha(root, cfg)
    if not baseline_sha:
        raise ArgumentError('state.verified is unset; cannot establish attempt baseline')
    _verify_clean_starting_point(root, cfg, baseline_sha)
    start_head = _git_head(root)
    start_reflog = _head_reflog(root)
    start_reflog_len = len(start_reflog) if start_reflog is not None else None

    conn = _connect(target_db)
    try:
        recover_interrupted_calls(conn, target_db)
        with _write_tx(conn):
            budget, counted, remaining = _campaign_counts(conn)
            if remaining <= 0:
                raise ArgumentError(
                    f'attempt limit reached ({counted}/{budget} counted attempts across the campaign '
                    'ledger); cannot begin another attempt. A new campaign needs a fresh, '
                    'user-authorized ledger (attempts.sqlite3); no tool argument resets the budget.'
                )
            if run_id is not None:
                run = _select_run(conn, run_id)
                if run is None:
                    raise ArgumentError(f'unknown run_id {run_id!r}')
                if run['status'] != 'active':
                    raise ArgumentError(
                        f"run {run['run_id']} is closed ({run['status']}); "
                        'cannot begin new attempts on a closed run'
                    )
            else:
                active_run = conn.execute(
                    "SELECT * FROM runs WHERE status = 'active' ORDER BY started_at DESC, rowid DESC LIMIT 1"
                ).fetchone()
                if active_run is not None:
                    if new_run:
                        raise ArgumentError(
                            f"run {active_run['run_id']} is still active; call close_run before starting a new run"
                        )
                    run = active_run
                else:
                    latest = conn.execute(
                        'SELECT * FROM runs ORDER BY started_at DESC, rowid DESC LIMIT 1'
                    ).fetchone()
                    if latest is not None and not new_run:
                        raise ArgumentError(
                            f"latest run {latest['run_id']} is closed ({latest['status']}); "
                            'cannot begin new attempts on a closed run (new_run=true opens another run '
                            'that shares the remaining campaign budget)'
                        )
                    run = _create_run_in_tx(conn, root=root, cfg=cfg, max_attempts=remaining)
            rid = str(run['run_id'])
            active = conn.execute(
                "SELECT ordinal, title FROM attempts WHERE run_id = ? AND status = 'active'",
                (rid,),
            ).fetchone()
            if active is not None:
                raise ArgumentError(
                    f"attempt {active['ordinal']} ({active['title']!r}) is already active; "
                    'call finish_attempt before starting another'
                )
            max_ord = conn.execute(
                'SELECT COALESCE(MAX(ordinal), 0) FROM attempts WHERE run_id = ?',
                (rid,),
            ).fetchone()[0]
            ordinal = int(max_ord) + 1
            now = _now_utc()
            cur = conn.execute(
                """
                INSERT INTO attempts (
                    run_id, ordinal, title, hypothesis, baseline_sha, start_head_sha,
                    start_reflog_len, started_at, status, counts_toward_limit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', 1)
                """,
                (rid, ordinal, title, hypothesis, baseline_sha, start_head, start_reflog_len, now),
            )
            attempt_id = int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ArgumentError(f'cannot begin attempt (concurrent conflict): {exc}') from exc
    finally:
        conn.close()

    return {
        'run_id': rid,
        'attempt_id': attempt_id,
        'ordinal': ordinal,
        'title': title,
        'hypothesis': hypothesis,
        'baseline_sha': baseline_sha,
        'start_head_sha': start_head,
        'campaign_budget': budget,
        'counted_attempts_used': counted,
        'counted_attempts_remaining': remaining,
        'started_at': now,
    }


def finish_attempt(
    outcome: str,
    decision_reason: str,
    *,
    counts_toward_limit: bool | None = None,
    promoted: bool = False,
    rolled_back: bool = False,
    changelog_sha: str | None = None,
    bench_json: str | None = None,
    tune_bench_json: str | None = None,
    profile_paths: list[str] | None = None,
    candidate_shas: list[str] | None = None,
    db_path: Path | None = None,
    state_path: Path | None = None,
    repo: Path | None = None,
) -> dict:
    """Validate completion consistency and transactionally close the active attempt."""
    canonical_outcome = _normalize_outcome(outcome)
    if not isinstance(decision_reason, str) or not decision_reason.strip():
        raise ArgumentError('decision_reason must be a nonempty string')
    decision_reason = _redact_inline_secrets(decision_reason.strip())
    if not isinstance(promoted, bool) or not isinstance(rolled_back, bool):
        raise ArgumentError('promoted and rolled_back must be booleans')
    if promoted and rolled_back:
        raise ArgumentError('an attempt cannot be both promoted and rolled_back')
    if canonical_outcome == 'PASS' and (not promoted or rolled_back):
        raise ArgumentError('outcome PASS requires promoted=true and rolled_back=false')
    if canonical_outcome == 'ROLLED BACK' and (promoted or not rolled_back):
        raise ArgumentError('outcome ROLLED BACK requires promoted=false and rolled_back=true')
    if canonical_outcome.startswith('FAIL (') and (promoted or not rolled_back):
        raise ArgumentError(f'outcome {canonical_outcome} requires promoted=false and rolled_back=true')
    if canonical_outcome == 'ABORTED' and promoted:
        raise ArgumentError('outcome ABORTED cannot have promoted=true')

    if counts_toward_limit is not None and not isinstance(counts_toward_limit, bool):
        raise ArgumentError('counts_toward_limit must be a boolean when provided')
    if counts_toward_limit is None:
        counts = canonical_outcome in COUNTED_OUTCOMES
    else:
        counts = counts_toward_limit
        if canonical_outcome in COUNTED_OUTCOMES and not counts:
            raise ArgumentError(
                f'outcome {canonical_outcome} must count toward the five-attempt limit'
            )

    if profile_paths is not None:
        if not isinstance(profile_paths, list) or any(
            not isinstance(p, str) or not p.strip() for p in profile_paths
        ):
            raise ArgumentError('profile_paths must be a list of nonempty path strings')
    if candidate_shas is not None:
        if not isinstance(candidate_shas, list) or any(
            not isinstance(s, str) or not SHA_RE.fullmatch(s) for s in candidate_shas
        ):
            raise ArgumentError('candidate_shas must be a list of 7-40 char hex SHAs')

    root = repo or state.ROOT
    target_db = db_path or default_db_path()
    cfg = state.read(state_path) if state_path is not None else state.read()
    current_verified = _resolve_verified_sha(root, cfg)
    is_git = _is_git_repo(root)

    if canonical_outcome != 'ABORTED':
        if not isinstance(changelog_sha, str) or not SHA_RE.fullmatch(changelog_sha):
            raise ArgumentError('changelog_sha (7-40 hex SHA) is required for completed/failed attempts')
    elif changelog_sha is not None and (not isinstance(changelog_sha, str) or not SHA_RE.fullmatch(changelog_sha)):
        raise ArgumentError('changelog_sha must be a 7-40 hex SHA')

    if is_git and changelog_sha is not None:
        rev = _safe_git(root, 'rev-parse', f'{changelog_sha}^{{commit}}')
        if rev is None or rev.returncode != 0:
            raise ArgumentError(f'changelog_sha {changelog_sha!r} does not resolve to a commit')
        changelog_sha = rev.stdout.strip()
        touched = _safe_git(root, 'show', '--name-only', '--format=', changelog_sha)
        files_in_commit = set((touched.stdout if touched else '').splitlines())
        if 'CHANGELOG.md' not in files_in_commit:
            raise ArgumentError(f'changelog_sha {changelog_sha[:7]} did not modify CHANGELOG.md')
        dirty_log = _safe_git(root, 'status', '--porcelain', '--', 'CHANGELOG.md')
        if dirty_log is not None and dirty_log.stdout.strip():
            raise ArgumentError('CHANGELOG.md has uncommitted modifications; commit it before finish_attempt')

    all_bench_rel, all_bench_data = (None, None)
    if bench_json is not None:
        all_bench_rel, all_bench_data = _validate_bench_file(bench_json, root)

    tune_bench_rel, tune_bench_data = (None, None)
    if tune_bench_json is not None:
        tune_bench_rel, tune_bench_data = _validate_bench_file(tune_bench_json, root)

    if promoted and all_bench_data is None:
        raise ArgumentError('promoted=true requires bench_json from the full ["all"] benchmark run')
    if promoted and all_bench_data is not None:
        if not all_bench_data['ok'] or all_bench_data['sum']['fail']:
            raise ArgumentError('cannot finish promoted attempt with a failing bench_json')
    if canonical_outcome == 'FAIL (correctness)':
        checked_benches = [b for b in (all_bench_data, tune_bench_data) if b is not None]
        if checked_benches and all(b['ok'] and not b['sum']['fail'] for b in checked_benches):
            raise ArgumentError('outcome FAIL (correctness) contradicts bench.json where ok=true and fail=0')

    conn = _connect(target_db)
    try:
        recover_interrupted_calls(conn, target_db)
        with _write_tx(conn):
            run = _select_run(conn, None)
            if run is None:
                raise ArgumentError('no optimization run found; call begin_attempt first')
            run_id = str(run['run_id'])
            active = conn.execute(
                "SELECT * FROM attempts WHERE run_id = ? AND status = 'active'",
                (run_id,),
            ).fetchone()
            if active is None:
                raise ArgumentError('no active attempt to finish')
            attempt_id = int(active['attempt_id'])
            ordinal = int(active['ordinal'])
            baseline_sha = str(active['baseline_sha'])
            start_head = active['start_head_sha'] or baseline_sha

            # changelog_sha must be a governed commit_kernel_attempt commit of THIS attempt,
            # touching CHANGELOG.md, made after the attempt started, reachable from HEAD, and
            # not already claimed by another attempt.
            if changelog_sha is not None:
                bound = conn.execute(
                    'SELECT attempt_id, files_json FROM attempt_commits WHERE sha = ?',
                    (changelog_sha,),
                ).fetchone()
                if bound is None:
                    raise ArgumentError(
                        f'changelog_sha {changelog_sha[:7]} was not made by commit_kernel_attempt during '
                        'this attempt (plain git commits and older commits are not accepted); commit the '
                        'CHANGELOG entry with commit_kernel_attempt'
                    )
                if int(bound['attempt_id']) != attempt_id:
                    raise ArgumentError(
                        f"changelog_sha {changelog_sha[:7]} belongs to attempt id {bound['attempt_id']}, "
                        f'not the active attempt {attempt_id}'
                    )
                if 'CHANGELOG.md' not in json.loads(bound['files_json'] or '[]'):
                    raise ArgumentError(
                        f'changelog_sha {changelog_sha[:7]} did not modify CHANGELOG.md (recorded files)'
                    )
                reused = conn.execute(
                    'SELECT ordinal FROM attempts WHERE changelog_sha = ? AND attempt_id != ?',
                    (changelog_sha, attempt_id),
                ).fetchone()
                if reused is not None:
                    raise ArgumentError(
                        f"changelog_sha {changelog_sha[:7]} is already used by attempt {reused['ordinal']}"
                    )
                if is_git:
                    if changelog_sha == start_head or not _is_ancestor(root, start_head, changelog_sha):
                        raise ArgumentError(
                            f'changelog_sha {changelog_sha[:7]} is not after the attempt start {start_head[:7]}'
                        )
                    if not _is_ancestor(root, changelog_sha, 'HEAD'):
                        raise ArgumentError(
                            f'changelog_sha {changelog_sha[:7]} is not reachable from HEAD'
                        )

            merged_shas: list[str] = list(json.loads(active['candidate_shas_json'] or '[]'))
            for s in candidate_shas or []:
                if s not in merged_shas:
                    merged_shas.append(s)
            for bdata in (all_bench_data, tune_bench_data):
                if bdata is not None and bdata['cand'] not in merged_shas:
                    merged_shas.append(bdata['cand'])
            final_cand = merged_shas[-1] if merged_shas else active['final_candidate_sha']

            # If a benchmark already ran in this attempt, it counts even if later aborted.
            bench_ran = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM tool_calls
                     WHERE attempt_id = ? AND tool_name = 'run_benchmarks' AND status = 'success'
                    """,
                    (attempt_id,),
                ).fetchone()[0]
            ) > 0
            if (bench_ran or all_bench_data is not None or tune_bench_data is not None) and not counts:
                raise ArgumentError(
                    'a benchmark ran for this attempt, so counts_toward_limit must be true'
                )

            # Baseline / state.verified / worktree cleanliness consistency checks.
            if promoted:
                if current_verified and _sha_matches(current_verified, baseline_sha):
                    raise ArgumentError(
                        'promoted=true but state.verified is still equal to attempt baseline_sha; '
                        'call promote_attempt first'
                    )
                if all_bench_data is not None:
                    if not _sha_matches(all_bench_data['base'], baseline_sha):
                        raise ArgumentError(
                            f"bench_json base {all_bench_data['base'][:7]} does not match "
                            f'attempt baseline {baseline_sha[:7]}'
                        )
                    if current_verified and not _sha_matches(all_bench_data['cand'], current_verified):
                        raise ArgumentError(
                            f"bench_json cand {all_bench_data['cand'][:7]} does not match "
                            f'state.verified {current_verified[:7]}'
                        )
            elif rolled_back or canonical_outcome == 'ABORTED':
                if current_verified and not _sha_matches(current_verified, baseline_sha):
                    raise ArgumentError(
                        f'{canonical_outcome} requires state.verified ({current_verified[:7]}) '
                        f'to match attempt baseline ({baseline_sha[:7]})'
                    )
                if is_git:
                    dirty_paths = _dirty_attempt_code_paths(root)
                    if dirty_paths:
                        raise ArgumentError(
                            f'attempt source/test paths have uncommitted changes ({dirty_paths}); '
                            'call rollback_workspace before finish_attempt'
                        )
                    drifted_paths = _drifted_attempt_code_paths(root, baseline_sha)
                    if drifted_paths:
                        raise ArgumentError(
                            f'attempt source/test paths at HEAD differ from baseline ({drifted_paths}); '
                            'call rollback_workspace before finish_attempt'
                        )
                if canonical_outcome == 'ABORTED' and not counts:
                    if merged_shas:
                        raise ArgumentError(
                            'a candidate commit was made during this attempt; non-counting ABORTED is forbidden '
                            '(rollback_workspace and close with an appropriate counted outcome)'
                        )
                    if is_git:
                        ungoverned = _attempt_path_commits_since(
                            root, start_head, active['start_reflog_len']
                        )
                        if ungoverned:
                            raise ArgumentError(
                                'a candidate commit was made during this attempt (commits touching attempt '
                                f'source/test paths since its start: {[c[:12] for c in ungoverned][:5]}); '
                                'non-counting ABORTED is forbidden; close with a counted outcome'
                            )

            for label, bdata in (('bench_json', all_bench_data), ('tune_bench_json', tune_bench_data)):
                if bdata is not None and not _sha_matches(bdata['base'], baseline_sha):
                    raise ArgumentError(
                        f"{label} base {bdata['base'][:7]} does not match attempt baseline {baseline_sha[:7]}"
                    )

            metrics_source = all_bench_data or tune_bench_data
            metrics_summary = _extract_metrics_summary(metrics_source) if metrics_source else None
            # Merge tune summary if we have both tune and all bench files.
            if all_bench_data is not None and tune_bench_data is not None:
                metrics_summary['tune_stage_sum'] = tune_bench_data.get('sum', {}).get('tune')

            profiles = list(profile_paths or [])
            ended_at = _now_utc()
            conn.execute(
                """
                UPDATE attempts
                   SET candidate_shas_json = ?,
                       final_candidate_sha = ?,
                       ended_at = ?,
                       status = 'closed',
                       outcome = ?,
                       promoted = ?,
                       rolled_back = ?,
                       decision_reason = ?,
                       changelog_sha = ?,
                       counts_toward_limit = ?,
                       tune_bench_path = ?,
                       all_bench_path = ?,
                       metrics_summary_json = ?,
                       profile_paths_json = ?
                 WHERE attempt_id = ?
                """,
                (
                    json.dumps(merged_shas, separators=(',', ':')),
                    final_cand,
                    ended_at,
                    canonical_outcome,
                    int(promoted),
                    int(rolled_back),
                    decision_reason,
                    changelog_sha,
                    int(counts),
                    tune_bench_rel,
                    all_bench_rel,
                    json.dumps(metrics_summary, sort_keys=True, separators=(',', ':'))
                    if metrics_summary is not None else None,
                    json.dumps(profiles, separators=(',', ':')),
                    attempt_id,
                ),
            )
            if all_bench_rel:
                _insert_artifact_in_tx(
                    conn, run_id=run_id, attempt_id=attempt_id, call_id=None,
                    kind='bench_all_json', local_path=all_bench_rel, remote_path=None, root=root,
                )
                if all_bench_data and all_bench_data.get('raw'):
                    _insert_artifact_in_tx(
                        conn, run_id=run_id, attempt_id=attempt_id, call_id=None,
                        kind='bench_raw_csv', local_path=all_bench_data['raw'], remote_path=None, root=root,
                    )
            if tune_bench_rel:
                _insert_artifact_in_tx(
                    conn, run_id=run_id, attempt_id=attempt_id, call_id=None,
                    kind='bench_tune_json', local_path=tune_bench_rel, remote_path=None, root=root,
                )
            for prof in profiles:
                local_candidate = _resolve_local(prof, root)
                if local_candidate.exists():
                    _insert_artifact_in_tx(
                        conn, run_id=run_id, attempt_id=attempt_id, call_id=None,
                        kind='ncu_profile', local_path=prof, remote_path=None, root=root,
                    )
                else:
                    _insert_artifact_in_tx(
                        conn, run_id=run_id, attempt_id=attempt_id, call_id=None,
                        kind='ncu_profile', local_path=None, remote_path=prof, root=root,
                    )

            budget, counted_used, remaining = _campaign_counts(conn)
    finally:
        conn.close()

    return {
        'run_id': run_id,
        'attempt_id': attempt_id,
        'ordinal': ordinal,
        'outcome': canonical_outcome,
        'promoted': promoted,
        'rolled_back': rolled_back,
        'counts_toward_limit': counts,
        'candidate_shas': merged_shas,
        'changelog_sha': changelog_sha,
        'campaign_budget': budget,
        'counted_attempts_used': counted_used,
        'counted_attempts_remaining': remaining,
        'ended_at': ended_at,
    }


def close_run(
    status: str,
    reason: str,
    *,
    run_id: str | None = None,
    db_path: Path | None = None,
    state_path: Path | None = None,
    repo: Path | None = None,
) -> dict:
    """Close the active optimization run as 'completed' or 'aborted'."""
    if status not in ('completed', 'aborted'):
        raise ArgumentError("status must be 'completed' or 'aborted'")
    if not isinstance(reason, str) or not reason.strip():
        raise ArgumentError('reason must be a nonempty string')
    reason = _redact_inline_secrets(reason.strip())

    target_db = db_path or default_db_path()
    current_ctx = _ACTIVE_CALL.get()
    current_call_id = current_ctx.call_id if current_ctx is not None else None

    conn = _connect(target_db)
    try:
        recover_interrupted_calls(conn, target_db, exclude_call_id=current_call_id)
        with _write_tx(conn):
            run = _select_run(conn, run_id)
            if run is None:
                raise ArgumentError('no optimization run found to close')
            rid = str(run['run_id'])
            if run['status'] != 'active':
                raise ArgumentError(f"run {rid} is already closed ({run['status']})")
            active_attempt = conn.execute(
                "SELECT ordinal, title FROM attempts WHERE run_id = ? AND status = 'active'",
                (rid,),
            ).fetchone()
            if active_attempt is not None:
                raise ArgumentError(
                    f"cannot close run {rid} while attempt {active_attempt['ordinal']} "
                    f"({active_attempt['title']!r}) is active; call finish_attempt first"
                )
            if status == 'completed':
                live_running = conn.execute(
                    """
                    SELECT call_id, tool_name, status FROM tool_calls
                     WHERE run_id = ? AND status = 'running'
                       AND (? IS NULL OR call_id != ?)
                     ORDER BY call_id ASC
                    """,
                    (rid, current_call_id, current_call_id),
                ).fetchall()
                if live_running:
                    summary = [(int(r['call_id']), r['tool_name'], r['status']) for r in live_running[:5]]
                    raise ArgumentError(
                        f'cannot mark run {rid} completed while live running tool calls exist: {summary}'
                    )
            ended_at = _now_utc()
            conn.execute(
                "UPDATE runs SET status = ?, ended_at = ?, close_reason = ? WHERE run_id = ? AND status = 'active'",
                (status, ended_at, reason, rid),
            )
            run_counted = int(
                conn.execute(
                    "SELECT COUNT(*) FROM attempts WHERE run_id = ? AND status = 'closed' AND counts_toward_limit = 1",
                    (rid,),
                ).fetchone()[0]
            )
            total_attempts = int(
                conn.execute('SELECT COUNT(*) FROM attempts WHERE run_id = ?', (rid,)).fetchone()[0]
            )
            budget, counted_used, remaining = _campaign_counts(conn)
    finally:
        conn.close()

    return {
        'run_id': rid,
        'status': status,
        'reason': reason,
        'ended_at': ended_at,
        'campaign_budget': budget,
        'counted_attempts_used': counted_used,
        'counted_attempts_remaining': remaining,
        'run_counted_attempts': run_counted,
        'total_attempts': total_attempts,
    }


def _row_to_attempt(row: sqlite3.Row) -> dict:
    return {
        'attempt_id': int(row['attempt_id']),
        'ordinal': int(row['ordinal']),
        'title': row['title'],
        'hypothesis': row['hypothesis'],
        'baseline_sha': row['baseline_sha'],
        'candidate_shas': json.loads(row['candidate_shas_json'] or '[]'),
        'final_candidate_sha': row['final_candidate_sha'],
        'started_at': row['started_at'],
        'ended_at': row['ended_at'],
        'status': row['status'],
        'outcome': row['outcome'],
        'promoted': bool(row['promoted']),
        'rolled_back': bool(row['rolled_back']),
        'decision_reason': row['decision_reason'],
        'changelog_sha': row['changelog_sha'],
        'counts_toward_limit': bool(row['counts_toward_limit']),
        'tune_bench_path': row['tune_bench_path'],
        'all_bench_path': row['all_bench_path'],
        'metrics_summary': json.loads(row['metrics_summary_json'])
        if row['metrics_summary_json'] else None,
        'profile_paths': json.loads(row['profile_paths_json'] or '[]'),
    }


def _row_to_call(row: sqlite3.Row) -> dict:
    return {
        'call_id': int(row['call_id']),
        'attempt_id': int(row['attempt_id']) if row['attempt_id'] is not None else None,
        'tool_name': row['tool_name'],
        'args_sha256': row['args_sha256'],
        'head_sha': row['head_sha'],
        'verified_sha': row['verified_sha'],
        'pid': int(row['pid']),
        'started_at': row['started_at'],
        'ended_at': row['ended_at'],
        'duration_ms': row['duration_ms'],
        'status': row['status'],
        'exit_code': row['exit_code'],
        'error_class': row['error_class'],
        'error_detail': row['error_detail'],
        'slurm_job_id': row['slurm_job_id'],
        'slurm_resume_key': row['slurm_resume_key'],
        'slurm_reattached': None if row['slurm_reattached'] is None else bool(row['slurm_reattached']),
        'artifact_paths': json.loads(row['artifact_paths_json'] or '[]'),
    }


def get_loop_status(
    *,
    run_id: str | None = None,
    recent_limit: int = 10,
    db_path: Path | None = None,
    state_path: Path | None = None,
    repo: Path | None = None,
) -> dict:
    """Return active run/attempt, remaining counted attempts, recent attempts, and unfinished calls."""
    if not isinstance(recent_limit, int) or recent_limit < 1 or recent_limit > 100:
        raise ArgumentError('recent_limit must be an integer between 1 and 100')
    root = repo or state.ROOT
    target_db = db_path or default_db_path()
    cfg = state.read(state_path) if state_path is not None else state.read()
    current_ctx = _ACTIVE_CALL.get()
    current_call_id = current_ctx.call_id if current_ctx is not None else None

    conn = _connect(target_db)
    try:
        recover_interrupted_calls(conn, target_db, exclude_call_id=current_call_id)
        with _write_tx(conn):
            run = _select_run(conn, run_id)
            if run is None and run_id is None:
                run = _select_or_init_run_in_tx(conn, root=root, cfg=cfg)
        if run is None:
            raise ArgumentError(f'unknown run_id {run_id!r}')
        rid = str(run['run_id'])
        max_attempts = int(run['max_attempts'])
        active_row = conn.execute(
            "SELECT * FROM attempts WHERE run_id = ? AND status = 'active'",
            (rid,),
        ).fetchone()
        run_counted = int(
            conn.execute(
                "SELECT COUNT(*) FROM attempts WHERE run_id = ? AND status = 'closed' AND counts_toward_limit = 1",
                (rid,),
            ).fetchone()[0]
        )
        budget, counted_used, remaining = _campaign_counts(conn)
        total_attempts = int(
            conn.execute('SELECT COUNT(*) FROM attempts WHERE run_id = ?', (rid,)).fetchone()[0]
        )
        recent_rows = conn.execute(
            'SELECT * FROM attempts WHERE run_id = ? ORDER BY ordinal DESC LIMIT ?',
            (rid, recent_limit),
        ).fetchall()
        unfinished_rows = conn.execute(
            """
            SELECT * FROM tool_calls
             WHERE run_id = ? AND status IN ('running', 'interrupted')
               AND (? IS NULL OR call_id != ?)
             ORDER BY call_id ASC
            """,
            (rid, current_call_id, current_call_id),
        ).fetchall()
    finally:
        conn.close()

    current_verified = _resolve_verified_sha(root, cfg)
    return {
        'run': {
            'run_id': rid,
            'branch': run['branch'],
            'initial_verified_sha': run['initial_verified_sha'],
            'current_verified_sha': current_verified,
            'prompt_path': run['prompt_path'],
            'prompt_sha256': run['prompt_sha256'],
            'suite_manifest_sha256': run['suite_manifest_sha256'],
            'suite_identity': json.loads(run['suite_identity']) if run['suite_identity'] else None,
            'max_attempts': max_attempts,
            'started_at': run['started_at'],
            'ended_at': run['ended_at'],
            'status': run['status'],
            'close_reason': run['close_reason'],
        },
        'active_attempt': _row_to_attempt(active_row) if active_row is not None else None,
        'campaign_budget': budget,
        'counted_attempts_used': counted_used,
        'counted_attempts_remaining': remaining,
        'run_counted_attempts': run_counted,
        'total_attempts': total_attempts,
        'recent_attempts': [_row_to_attempt(r) for r in recent_rows],
        'unfinished_tool_calls': [_row_to_call(r) for r in unfinished_rows],
    }


def _fmt_geo_wlt(subset: dict | None) -> str:
    if not subset or not subset.get('n'):
        return '—'
    geo = subset.get('geo_x_base')
    geo_s = f'{geo:.3f}×' if isinstance(geo, (int, float)) else '—'
    return f"{geo_s} ({subset.get('W', 0)}/{subset.get('L', 0)}/{subset.get('T', 0)})"


def _fmt_cusp_pair(summary: dict | None) -> str:
    if not summary:
        return '—'
    s = summary.get('sum') or {}
    tune = (s.get('tune') or {}).get('geo_x_cusp')
    held = (s.get('held') or {}).get('geo_x_cusp')
    ts = f'{tune:.3f}×' if isinstance(tune, (int, float)) else '—'
    hs = f'{held:.3f}×' if isinstance(held, (int, float)) else '—'
    return f'{ts} / {hs}'


def _render_markdown(payload: dict) -> str:
    run = payload['run']
    counts = payload['counts']
    status_line = f"- **Started:** `{run['started_at']}`"
    if run.get('ended_at'):
        status_line += f" · **Ended:** `{run['ended_at']}`"
    status_line += f" · **Status:** `{run['status']}`"
    lines = [
        f"# Optimization Loop Attempt Log: `{run['run_id']}`",
        '',
        f"- **Branch:** `{run['branch']}`",
        f"- **Initial `opt/verified`:** `{run['initial_verified_sha']}`",
        f"- **Current/Final `opt/verified`:** `{run['final_verified_sha']}` (moved: {'yes' if run['verified_moved'] else 'no'})",
        f"- **Prompt SHA-256:** `{run['prompt_sha256']}` (`{run['prompt_path']}`)",
        f"- **Suite Manifest SHA-256:** `{run['suite_manifest_sha256'] or 'unrecorded'}`",
        f"- **Counted Attempts (campaign ledger):** {counts['counted_attempts_used']} / {counts['campaign_budget']} (remaining: {counts['counted_attempts_remaining']}; this run: {counts['run_counted_attempts']})",
        f"- **Interrupted Tool Calls:** {counts['interrupted_call_count']}",
        status_line,
    ]
    if run.get('close_reason'):
        lines.append(f"- **Close Reason:** {run['close_reason']}")
    lines.extend([
        '',
        '## Attempts',
        '',
        '| # | Title | sha | Outcome | tune `geo_x_base` (W/L/T) | held `geo_x_base` (W/L/T) | `geo_x_cusp` tune / held | Promoted? |',
        '|---|---|---|---|---|---|---|---|',
    ])
    for att in payload['attempts']:
        sha = (att['final_candidate_sha'] or '')[:7] or '—'
        outcome = att['outcome'] or att['status']
        ms = att.get('metrics_summary') or {}
        sums = ms.get('sum') or {}
        tune_str = _fmt_geo_wlt(sums.get('tune'))
        held_str = _fmt_geo_wlt(sums.get('held'))
        cusp_str = _fmt_cusp_pair(ms)
        prom = 'Yes' if att['promoted'] else 'No'
        title = att['title'].replace('|', '\\|')
        lines.append(
            f"| {att['ordinal']} | {title} | `{sha}` | {outcome} | {tune_str} | {held_str} | {cusp_str} | {prom} |"
        )
    if not payload['attempts']:
        lines.append('| — | *(no attempts recorded)* | — | — | — | — | — | — |')

    lines.extend(['', '## Attempt Details', ''])
    for att in payload['attempts']:
        lines.append(f"### Attempt {att['ordinal']}: {att['title']}")
        lines.append(f"- **Hypothesis:** {att['hypothesis']}")
        lines.append(f"- **Baseline SHA:** `{att['baseline_sha']}`")
        cand_list = ', '.join(f'`{s[:7]}`' for s in att['candidate_shas']) or '*(none)*'
        lines.append(f"- **Candidate SHA(s):** {cand_list}")
        lines.append(
            f"- **Outcome:** `{att['outcome'] or att['status']}` · "
            f"**Promoted:** {att['promoted']} · **Rolled Back:** {att['rolled_back']} · "
            f"**Counts Toward Limit:** {att['counts_toward_limit']}"
        )
        if att['decision_reason']:
            lines.append(f"- **Decision/Reason:** {att['decision_reason']}")
        if att['changelog_sha']:
            lines.append(f"- **CHANGELOG Commit:** `{att['changelog_sha']}`")
        lines.append(f"- **Tool Calls:** {len(att['tool_calls'])}")
        if att['artifacts']:
            lines.append('- **Artifacts:**')
            for art in att['artifacts']:
                loc = art['local_path'] or art['remote_path']
                sha_note = f" (`sha256:{art['sha256'][:12]}`)" if art['sha256'] else ''
                lines.append(f"  - `{art['kind']}`: `{loc}`{sha_note}")
        lines.append('')

    if payload['unfinished_tool_calls']:
        lines.extend(['## Unfinished / Interrupted Tool Calls', ''])
        for call in payload['unfinished_tool_calls']:
            lines.append(
                f"- call `{call['call_id']}` (`{call['tool_name']}`, status=`{call['status']}`, "
                f"started=`{call['started_at']}`, job=`{call['slurm_job_id']}`): "
                f"{call['error_detail'] or 'in progress'}"
            )
        lines.append('')

    return '\n'.join(lines)


def export_attempt_log(
    *,
    run_id: str | None = None,
    output_dir: str | None = None,
    db_path: Path | None = None,
    state_path: Path | None = None,
    repo: Path | None = None,
) -> dict:
    """Export stable JSON and Markdown summaries under reports/agent_loop/runs/<run-id>/."""
    root = repo or state.ROOT
    target_db = db_path or default_db_path()
    cfg = state.read(state_path) if state_path is not None else state.read()
    current_ctx = _ACTIVE_CALL.get()
    current_call_id = current_ctx.call_id if current_ctx is not None else None

    conn = _connect(target_db)
    try:
        recover_interrupted_calls(conn, target_db, exclude_call_id=current_call_id)
        with _write_tx(conn):
            run = _select_run(conn, run_id)
            if run is None and run_id is None:
                run = _select_or_init_run_in_tx(conn, root=root, cfg=cfg)
        if run is None:
            raise ArgumentError(f'unknown run_id {run_id!r}')
        rid = str(run['run_id'])
        max_attempts = int(run['max_attempts'])
        attempt_rows = conn.execute(
            'SELECT * FROM attempts WHERE run_id = ? ORDER BY ordinal ASC',
            (rid,),
        ).fetchall()
        call_rows = conn.execute(
            """
            SELECT * FROM tool_calls
             WHERE run_id = ? AND (? IS NULL OR call_id != ?)
             ORDER BY call_id ASC
            """,
            (rid, current_call_id, current_call_id),
        ).fetchall()
        art_rows = conn.execute(
            'SELECT * FROM artifacts WHERE run_id = ? ORDER BY artifact_id ASC',
            (rid,),
        ).fetchall()
        budget, campaign_used, campaign_remaining = _campaign_counts(conn)
    finally:
        conn.close()

    calls_by_attempt: dict[int | None, list[dict]] = {}
    for cr in call_rows:
        aid = int(cr['attempt_id']) if cr['attempt_id'] is not None else None
        calls_by_attempt.setdefault(aid, []).append(_row_to_call(cr))

    arts_by_attempt: dict[int | None, list[dict]] = {}
    for ar in art_rows:
        aid = int(ar['attempt_id']) if ar['attempt_id'] is not None else None
        arts_by_attempt.setdefault(aid, []).append({
            'artifact_id': int(ar['artifact_id']),
            'call_id': int(ar['call_id']) if ar['call_id'] is not None else None,
            'kind': ar['kind'],
            'local_path': ar['local_path'],
            'remote_path': ar['remote_path'],
            'sha256': ar['sha256'],
            'size_bytes': ar['size_bytes'],
            'created_at': ar['created_at'],
        })

    attempts_payload = []
    for ar in attempt_rows:
        att = _row_to_attempt(ar)
        aid = att['attempt_id']
        att['tool_calls'] = calls_by_attempt.get(aid, [])
        att['artifacts'] = arts_by_attempt.get(aid, [])
        attempts_payload.append(att)

    run_counted = sum(1 for a in attempts_payload if a['status'] == 'closed' and a['counts_toward_limit'])
    promoted_count = sum(1 for a in attempts_payload if a['promoted'])
    unfinished = [
        _row_to_call(cr)
        for cr in call_rows
        if cr['status'] in ('running', 'interrupted')
    ]
    current_verified = _resolve_verified_sha(root, cfg) or run['initial_verified_sha']
    payload = {
        'schema': 'attempt_log.1',
        'run': {
            'run_id': rid,
            'branch': run['branch'],
            'initial_verified_sha': run['initial_verified_sha'],
            'final_verified_sha': current_verified,
            'verified_moved': current_verified != run['initial_verified_sha'],
            'prompt_path': run['prompt_path'],
            'prompt_sha256': run['prompt_sha256'],
            'suite_manifest_sha256': run['suite_manifest_sha256'],
            'suite_identity': json.loads(run['suite_identity']) if run['suite_identity'] else None,
            'config': json.loads(run['config_json']),
            'max_attempts': max_attempts,
            'started_at': run['started_at'],
            'ended_at': run['ended_at'],
            'status': run['status'],
            'close_reason': run['close_reason'],
        },
        'counts': {
            'campaign_budget': budget,
            'counted_attempts_used': campaign_used,
            'counted_attempts_remaining': campaign_remaining,
            'run_counted_attempts': run_counted,
            'total_attempts': len(attempts_payload),
            'promoted_count': promoted_count,
            'tool_call_count': len(call_rows),
            'interrupted_call_count': sum(1 for c in unfinished if c['status'] == 'interrupted'),
        },
        'attempts': attempts_payload,
        'setup_tool_calls': calls_by_attempt.get(None, []),
        'unfinished_tool_calls': unfinished,
    }

    # Exports may only be written under reports/agent_loop/runs/, never into attempt source/test
    # paths or anywhere else in the worktree.
    runs_base = (root / 'reports/agent_loop/runs').resolve()
    out_dir = (_resolve_local(output_dir, root) if output_dir else (runs_base / rid)).resolve()
    if out_dir != runs_base and runs_base not in out_dir.parents:
        raise ArgumentError(
            f'output_dir must be inside reports/agent_loop/runs/ (got {output_dir!r})'
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'summary.json'
    md_path = out_dir / 'summary.md'

    json_text = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    md_text = _render_markdown(payload)

    for path, content in ((json_path, json_text), (md_path, md_text)):
        tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
        tmp.write_text(content, encoding='utf-8')
        os.replace(tmp, path)

    json_rel = _rel_path(json_path, root)
    md_rel = _rel_path(md_path, root)
    attach_artifact(kind='export_json', local_path=json_rel)
    attach_artifact(kind='export_md', local_path=md_rel)

    return {
        'run_id': rid,
        'json_path': json_rel,
        'json_sha256': _sha256_bytes(json_text.encode('utf-8')),
        'markdown_path': md_rel,
        'markdown_sha256': _sha256_bytes(md_text.encode('utf-8')),
        'attempts_exported': len(attempts_payload),
    }
