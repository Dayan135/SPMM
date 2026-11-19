"""Governed, non-gating synthetic_controls diagnostic panel."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import statistics

from . import attempt_log, bench, state
from .build import ensure_build, load_build_env
from .schema import validate
from .slurm import release_job, submit_slurm_job
from .util import ArgumentError, InfrastructureError, checked_ssh, git, head, ssh, transfer

CONTROLS_PANEL = 'synthetic_controls'
CONTROLS_SIZE = 262144
CONTROLS_FEATURES = (128, 512)
CONTROLS_PROFILES = ('fem2d', 'fem2d_permuted', 'diag_b1.0')
CONTROLS_KERNEL = 'boo:rowseg'
EXPECTED_ROLES = ('cand', 'base')
EXPECTED_ARMS = ('kernel', 'kernel_dup', 'kernel_api', 'cusparse')
CONTROLS_KERNEL_CONFIG = {
    'implementation': 'boo:rowseg',
    'threads': 256,
    'p': 64,
    'features_per_thread': 1,
    'launch_mode': 'auto',
    'cluster_k': 32,
}
CONTROLS_PROFILE_META = {
    'fem2d': {
        'diagnostic_role': 'regression_negative_control',
        'label_method': 'hem',
        'k': 32,
        'generator': 'generate_fem_2d',
        'generator_params': {
            'system_size': CONTROLS_SIZE,
            'dofs_per_node': 4,
            'coupling': 'anisotropic',
            'noise_density': 0.0,
            'seed': 0,
        },
        'seed': 0,
    },
    'fem2d_permuted': {
        'diagnostic_role': 'positive_ordering_locality_control',
        'label_method': 'hem',
        'k': 32,
        'generator': 'generate_permuted_fem_2d',
        'generator_params': {
            'system_size': CONTROLS_SIZE,
            'dofs_per_node': 4,
            'coupling': 'anisotropic',
            'noise_density': 0.0,
            'seed': 0,
        },
        'seed': 0,
    },
    'diag_b1.0': {
        'diagnostic_role': 'ideal_dense_block_ceiling_control',
        'label_method': 'identity',
        'k': 32,
        'generator': 'generate_diag_blocks_plus_noise',
        'generator_params': {
            'system_size': CONTROLS_SIZE,
            'block_size': 64,
            'diag_density': 1.0,
            'off_diag_density': 0.0,
            'seed': 0,
        },
        'seed': 0,
    },
}
SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
SHA_RE = re.compile(r'^[0-9a-f]{7,40}$')


def _controls_root(cfg: dict) -> str:
    return f"{cfg['agent_root'].rstrip('/')}/suite/controls"


def setup_controls(stage: str = 'all') -> dict:
    """Prepare or inspect the isolated suite/controls/ namespace on the cluster."""
    if stage not in ('prepare', 'refs', 'labels', 'all', 'status'):
        raise ArgumentError('stage must be prepare, refs, labels, all or status')
    cfg = state.read()
    if not cfg['preamble']:
        built = load_build_env(cfg['env_name'], cfg['modules'])
    else:
        built = ensure_build(head(), config=cfg)
    if not built.get('ok'):
        return {'ok': False, 'reason': built.get('reason', 'build failed')}
    cfg = state.read()
    sha = head()
    controls_root = _controls_root(cfg)
    source = f"{cfg['agent_root'].rstrip('/')}/src/{sha}"
    build = built['build_dir']
    for phase in ('prepare', 'refs', 'labels'):
        if stage not in (phase, 'all'):
            continue
        body = (f'use {shlex.quote(build)} {shlex.quote(source)}\n'
                f'python -m benchmark.controls_suite {phase} --root {shlex.quote(controls_root)}\n')
        job = submit_slurm_job(body, name=f'spmm-controls-{phase}', gpu=False,
                               time='04:00:00', mem='64G', cpus=16, sha=sha, config=cfg)
        if job['state'] != 'COMPLETED' or job['exit']:
            return {'ok': False, 'stage': phase, 'job': job['job'],
                    'reason': job['err_tail'] or job['out_tail']}
    cmd = ('bash -lc ' + shlex.quote(
        cfg['preamble'] + '\n'
        f'use {shlex.quote(build)} {shlex.quote(source)}\n'
        f'python -m benchmark.controls_suite status --root {shlex.quote(controls_root)}'
    ))
    res = checked_ssh(cmd, timeout=300, config=cfg)
    return json.loads(res.stdout)


def _validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict) or manifest.get('panel') != CONTROLS_PANEL:
        raise RuntimeError(f"invalid controls manifest panel: {manifest.get('panel')!r}")
    if manifest.get('gating') is not False or manifest.get('promotion_eligible') is not False:
        raise RuntimeError('controls manifest must have gating=false and promotion_eligible=false')
    if manifest.get('version') != 1:
        raise RuntimeError(f"unexpected controls manifest version: {manifest.get('version')!r}")
    np_ver = manifest.get('numpy_version')
    sp_ver = manifest.get('scipy_version')
    py_ver = manifest.get('python_version')
    if not all(isinstance(v, str) and v for v in (np_ver, sp_ver, py_ver)):
        raise RuntimeError('controls manifest missing numpy_version/scipy_version/python_version')
    matrices = manifest.get('matrices')
    if not isinstance(matrices, dict):
        raise RuntimeError('controls manifest missing matrices map')
    if set(matrices.keys()) != set(CONTROLS_PROFILES):
        raise RuntimeError(
            f'controls manifest profiles mismatch: expected {sorted(CONTROLS_PROFILES)}, '
            f'got {sorted(matrices.keys())}'
        )
    gen_src_hashes = set()
    for profile in CONTROLS_PROFILES:
        rec = matrices[profile]
        exp = CONTROLS_PROFILE_META[profile]
        if rec.get('id') != profile:
            raise RuntimeError(f'{profile}: id mismatch {rec.get("id")!r}')
        if rec.get('diagnostic_role') != exp['diagnostic_role']:
            raise RuntimeError(f'{profile}: diagnostic_role mismatch {rec.get("diagnostic_role")!r}')
        if rec.get('generator') != exp['generator'] or rec.get('seed') != exp['seed']:
            raise RuntimeError(f'{profile}: generator/seed mismatch')
        if rec.get('generator_params') != exp['generator_params']:
            raise RuntimeError(f'{profile}: generator_params mismatch')
        gsrc = str(rec.get('generator_src_sha256', ''))
        if not SHA256_RE.match(gsrc):
            raise RuntimeError(f'{profile}: missing or invalid generator_src_sha256')
        gen_src_hashes.add(gsrc)
        if rec.get('canonical_shape') != [CONTROLS_SIZE, CONTROLS_SIZE]:
            raise RuntimeError(f'{profile}: unexpected shape {rec.get("canonical_shape")}')
        nnz = rec.get('canonical_nnz')
        ncols = rec.get('nonempty_cols')
        if not isinstance(nnz, int) or isinstance(nnz, bool) or nnz <= 0:
            raise RuntimeError(f'{profile}: invalid canonical_nnz {nnz!r}')
        if not isinstance(ncols, int) or isinstance(ncols, bool) or not (0 < ncols <= CONTROLS_SIZE):
            raise RuntimeError(f'{profile}: invalid nonempty_cols {ncols!r}')
        if rec.get('label_method') != exp['label_method'] or rec.get('k') != exp['k']:
            raise RuntimeError(f'{profile}: label_method/k mismatch')
        if rec.get('data_dtype') != 'float32' or rec.get('index_dtype') != 'int32':
            raise RuntimeError(f'{profile}: unexpected dtypes {rec.get("data_dtype")}/{rec.get("index_dtype")}')
        if not SHA256_RE.match(str(rec.get('csr_hash', ''))):
            raise RuntimeError(f'{profile}: invalid csr_hash')
        if rec.get('numpy_version') != np_ver or rec.get('scipy_version') != sp_ver:
            raise RuntimeError(f'{profile}: mixed numpy/scipy generation provenance in controls manifest')
    if len(gen_src_hashes) != 1:
        raise RuntimeError('mixed generator_src_sha256 across controls manifest profiles')


def _arm_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    med = statistics.median(values)
    return {
        'median_ms': bench._sig(med),
        'mean_ms': bench._sig(statistics.mean(values)),
        'min_ms': bench._sig(min(values)),
        'max_ms': bench._sig(max(values)),
        'p25_ms': bench._sig(bench._percentile(values, 25)),
        'p75_ms': bench._sig(bench._percentile(values, 75)),
        'mad_ms': bench._sig(statistics.median(abs(v - med) for v in values)),
    }


def _clean_build_fingerprints(build_fingerprints: dict) -> dict:
    clean = {}
    for role in EXPECTED_ROLES:
        fp = build_fingerprints.get(role)
        if not isinstance(fp, dict):
            raise RuntimeError(f'missing build_fingerprints[{role!r}]')
        src_hash = fp.get('src_hash')
        build_dir = fp.get('build_dir')
        so_sha256 = fp.get('so_sha256')
        if not all(isinstance(v, str) and v.strip() for v in (src_hash, build_dir, so_sha256)):
            raise RuntimeError(f'incomplete build_fingerprints for {role}: {fp!r}')
        clean[role] = {
            'src_hash': src_hash,
            'build_dir': build_dir,
            'so_sha256': so_sha256,
        }
    return clean


def _collect_and_verify_env(
    run_dir: Path,
    summaries: list[dict],
    samples: list[dict],
    *,
    cand: str,
    base: str,
    clean_fp: dict,
    job: int,
    node: str,
    reps: int,
    warmup: int,
    rounds: int,
    compiler: str | None,
) -> dict:
    if not isinstance(node, str) or not node.strip() or node == 'unknown':
        raise RuntimeError(f'invalid job node provenance: {node!r}')
    role_paths: dict[str, tuple[str, str]] = {}
    gpus = set()
    cudas = set()
    torches = set()
    pythons = set()

    for role in EXPECTED_ROLES:
        bdir = PurePosixPath(clean_fp[role]['build_dir'])
        pkgs = set()
        cores = set()
        for r in range(1, rounds + 1):
            env_path = run_dir / f'env_{role}_r{r}.json'
            if not env_path.exists():
                raise RuntimeError(f'missing environment file {env_path.name}')
            e = json.loads(env_path.read_text())
            if e.get('role') != role or int(e.get('round', -1)) != r:
                raise RuntimeError(f'{env_path.name}: role/round mismatch ({e.get("role")}, {e.get("round")})')
            if e.get('node') != node:
                raise RuntimeError(f'{env_path.name}: node mismatch ({e.get("node")!r} != {node!r})')
            gpu = e.get('gpu')
            if not isinstance(gpu, str) or not gpu.strip() or gpu == 'unknown':
                raise RuntimeError(f'{env_path.name}: missing or unknown GPU identity')
            gpus.add(gpu)
            cudas.add(e.get('cuda'))
            torches.add(e.get('torch'))
            pythons.add(e.get('python'))
            pkg = e.get('package_file')
            core = e.get('core_file')
            if not isinstance(pkg, str) or not isinstance(core, str) or not pkg or not core:
                raise RuntimeError(f'{env_path.name}: missing package_file/core_file')
            if not PurePosixPath(pkg).is_relative_to(bdir) or not PurePosixPath(core).is_relative_to(bdir):
                raise RuntimeError(
                    f'{env_path.name}: {role} module path ({pkg}, {core}) outside expected build_dir {bdir}'
                )
            pkgs.add(pkg)
            cores.add(core)
        if len(pkgs) != 1 or len(cores) != 1:
            raise RuntimeError(f'mixed {role} module paths across rounds')
        role_paths[role] = (next(iter(pkgs)), next(iter(cores)))

    if cand != base and clean_fp['cand']['build_dir'] != clean_fp['base']['build_dir']:
        if role_paths['cand'] == role_paths['base']:
            raise RuntimeError('candidate and baseline resolved to identical module paths despite distinct builds')

    if len(gpus) != 1 or len(cudas) != 1 or len(torches) != 1 or len(pythons) != 1:
        raise RuntimeError('mixed GPU/CUDA/PyTorch/Python environment across roles or rounds')

    for label, rows in (('summary.csv', summaries), ('samples.csv', samples)):
        for row in rows:
            role = row.get('role')
            if role not in EXPECTED_ROLES:
                raise RuntimeError(f'unexpected role {role!r} in {label}')
            if row.get('node') != node:
                raise RuntimeError(f'mixed/mismatched node {row.get("node")!r} in {label}')
            exp_pkg, exp_core = role_paths[role]
            if row.get('package_file') != exp_pkg or row.get('core_file') != exp_core:
                raise RuntimeError(
                    f'{label}: {role} module path ({row.get("package_file")}, {row.get("core_file")}) '
                    f'does not match {role} build ({exp_pkg}, {exp_core})'
                )

    return {
        'node': node,
        'job': job,
        'gpu': next(iter(gpus)),
        'cuda': next(iter(cudas)),
        'torch': next(iter(torches)),
        'python': next(iter(pythons)),
        'compiler': compiler,
        'dtype': 'fp32',
        'reps': reps,
        'warmup': warmup,
        'rounds': rounds,
    }


def aggregate_controls(
    run_dir: Path,
    manifest: dict,
    *,
    manifest_sha256: str,
    run_id: str,
    attempt_id: int,
    attempt_ordinal: int,
    cand: str,
    base: str,
    tool_commit: str,
    build_fingerprints: dict,
    job_info: dict,
    reps: int,
    warmup: int,
    rounds: int,
) -> dict:
    if not SHA_RE.match(cand) or not SHA_RE.match(base) or not SHA_RE.match(tool_commit):
        raise RuntimeError('invalid cand, base, or tool_commit SHA')
    _validate_manifest(manifest)
    clean_fp = _clean_build_fingerprints(build_fingerprints)

    samples_path = run_dir / 'samples.csv'
    summary_path = run_dir / 'summary.csv'
    with samples_path.open(newline='') as file:
        samples = list(csv.DictReader(file))
    with summary_path.open(newline='') as file:
        summaries = list(csv.DictReader(file))

    env = _collect_and_verify_env(
        run_dir, summaries, samples,
        cand=cand, base=base, clean_fp=clean_fp,
        job=int(job_info['job']), node=str(job_info['node']),
        reps=reps, warmup=warmup, rounds=rounds,
        compiler=build_fingerprints.get('cand', {}).get('compiler'),
    )

    expected_grid = {(m, f) for m in CONTROLS_PROFILES for f in CONTROLS_FEATURES}
    expected_summary_keys = {
        (m, f, role, r, arm)
        for m, f in expected_grid
        for role in EXPECTED_ROLES
        for r in range(1, rounds + 1)
        for arm in EXPECTED_ARMS
    }

    summary_map: dict[tuple[str, int, str, int, str], dict] = {}
    for row in summaries:
        if row.get('status') == 'unsupported':
            raise RuntimeError('controls panel does not allow unsupported/skipped cases')
        try:
            key = (row['mat'], int(row['F']), row['role'], int(row['round']), row['arm'])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f'malformed summary row: {row!r}') from exc
        if key not in expected_summary_keys:
            raise RuntimeError(f'unexpected summary record {key}')
        if key in summary_map:
            raise RuntimeError(f'duplicate summary record {key}')
        if row.get('status') not in ('measured', 'failed'):
            raise RuntimeError(f'invalid summary status {row.get("status")!r} for {key}')
        if row['status'] == 'measured':
            # Measured arms require all finite, nonnegative numeric correctness fields with 0 violations.
            for col in ('max_abs_error', 'max_scaled_error', 'bound_violations'):
                if row.get(col) in (None, ''):
                    raise RuntimeError(f'missing correctness field {col!r} for {key}')
            try:
                mae = float(row['max_abs_error'])
                mse = float(row['max_scaled_error'])
                bv = int(row['bound_violations'])
                if not (math.isfinite(mae) and mae >= 0 and math.isfinite(mse) and mse >= 0 and bv == 0):
                    raise ValueError('measured row requires finite nonnegative errors and bound_violations == 0')
            except ValueError as exc:
                raise RuntimeError(f'invalid correctness values for {key}: {exc}') from exc
        else:
            # Failed arms require a nonempty reason; numeric correctness fields may be absent for exceptions.
            if not str(row.get('reason') or '').strip():
                raise RuntimeError(f'missing failure reason for failed arm {key}')
            for col in ('max_abs_error', 'max_scaled_error'):
                if row.get(col) not in (None, ''):
                    try:
                        val = float(row[col])
                        if not (math.isfinite(val) and val >= 0):
                            raise ValueError(f'{col} must be finite and nonnegative')
                    except ValueError as exc:
                        raise RuntimeError(f'invalid correctness values for {key}: {exc}') from exc
            if row.get('bound_violations') not in (None, ''):
                try:
                    bv = int(row['bound_violations'])
                    if bv < 0:
                        raise ValueError('negative bound_violations')
                except ValueError as exc:
                    raise RuntimeError(f'invalid correctness values for {key}: {exc}') from exc
        summary_map[key] = row

    if set(summary_map.keys()) != expected_summary_keys:
        missing = sorted(expected_summary_keys - set(summary_map.keys()))
        raise RuntimeError(f'incomplete controls protocol; missing {len(missing)} summary records, e.g. {missing[:4]}')

    samples_by_key: dict[tuple[str, int, str, int, str], list[float]] = {}
    repeats_by_key: dict[tuple[str, int, str, int, str], set[int]] = {}
    sample_x_by_case: dict[tuple[str, int], set[str]] = {}
    times: dict[tuple[str, str, int, str], list[float]] = {}
    for row in samples:
        try:
            skey = (row['mat'], int(row['F']), row['role'], int(row['round']), row['arm'])
            rep = int(row['repeat'])
            tval = float(row['time_ms'])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f'malformed sample row: {row!r}') from exc
        if skey not in expected_summary_keys:
            raise RuntimeError(f'unexpected sample record {skey}')
        if not (1 <= rep <= reps):
            raise RuntimeError(f'out-of-range sample repeat={rep} for {skey} (expected 1..{reps})')
        seen_reps = repeats_by_key.setdefault(skey, set())
        if rep in seen_reps:
            raise RuntimeError(f'duplicate sample repeat={rep} for {skey}')
        seen_reps.add(rep)
        sx = row.get('x_sha256')
        if not SHA256_RE.match(str(sx or '')):
            raise RuntimeError(f'missing or invalid sample x_sha256 for {skey} repeat={rep}')
        sample_x_by_case.setdefault((skey[0], skey[1]), set()).add(sx)
        if not math.isfinite(tval) or tval <= 0:
            raise RuntimeError(f'invalid sample time_ms={tval} for {skey}')
        samples_by_key.setdefault(skey, []).append(tval)
        times.setdefault((skey[2], skey[0], skey[1], skey[4]), []).append(tval)

    expected_reps = set(range(1, reps + 1))
    for skey, srow in summary_map.items():
        got_reps = repeats_by_key.get(skey, set())
        got_n = len(samples_by_key.get(skey, []))
        if srow['status'] == 'measured':
            if got_n != reps or got_reps != expected_reps:
                raise RuntimeError(
                    f'sample repeat/count mismatch for {skey}: expected 1..{reps}, '
                    f'got count={got_n}, repeats={sorted(got_reps)[:6]}'
                )
        else:
            if got_n != 0:
                raise RuntimeError(f'failed arm {skey} should not have timing samples (got {got_n})')

    base_failures = [
        (skey, srow)
        for skey, srow in sorted(summary_map.items())
        if skey[2] == 'base' and srow['status'] == 'failed'
    ]
    if base_failures:
        details = '; '.join(
            f'{m}_F{f} round={r} arm={arm}: {attempt_log.redact_text(str(srow.get("reason") or ""))}'
            for (m, f, _role, r, arm), srow in base_failures[:4]
        )
        raise InfrastructureError(
            f'controls integrity failure: verified baseline or shared harness failed ({details})'
        )

    cases_sorted = sorted(expected_grid)
    noise, per_case_noise = bench.compute_noise_95(times, cases_sorted)

    cases_out = []
    for mat, f in cases_sorted:
        meta = manifest['matrices'][mat]
        exp_meta = CONTROLS_PROFILE_META[mat]
        records = [
            summary_map[(mat, f, role, r, arm)]
            for role in EXPECTED_ROLES
            for r in range(1, rounds + 1)
            for arm in EXPECTED_ARMS
        ]
        x_shas = {r.get('x_sha256') for r in records}
        ref_hashes = {r.get('ref_hash') for r in records}
        if len(x_shas) != 1 or not SHA256_RE.match(str(next(iter(x_shas)) or '')):
            raise RuntimeError(f'{mat}_F{f}: missing or mismatched x_sha256 across roles/rounds')
        if len(ref_hashes) != 1 or not (next(iter(ref_hashes)) or '').strip():
            raise RuntimeError(f'{mat}_F{f}: missing or mismatched ref_hash across roles/rounds')
        x_sha = next(iter(x_shas))
        ref_hash = next(iter(ref_hashes))
        case_sample_xs = sample_x_by_case.get((mat, f), set())
        if case_sample_xs and case_sample_xs != {x_sha}:
            raise RuntimeError(f'{mat}_F{f}: sample x_sha256 {sorted(case_sample_xs)} does not match summary {x_sha}')

        label_tuples = {
            (
                r.get('label_method'),
                r.get('label_k'),
                r.get('label_hash'),
                r.get('label_cached'),
                r.get('label_cache_path'),
                r.get('clustering_src_sha256'),
            )
            for r in records
        }
        if len(label_tuples) != 1:
            raise RuntimeError(f'{mat}_F{f}: label provenance mismatch across roles/rounds')
        l_method, l_k_raw, l_hash, l_cached_raw, l_cache_path, l_csrc = next(iter(label_tuples))
        l_k = int(l_k_raw) if l_k_raw not in (None, '') else -1
        if l_method != exp_meta['label_method'] or l_k != exp_meta['k']:
            raise RuntimeError(f'{mat}_F{f}: label method/k mismatch ({l_method}, {l_k})')
        if not isinstance(l_hash, str) or not l_hash.strip():
            raise RuntimeError(f'{mat}_F{f}: missing label_hash')

        cand_records = [r for r in records if r['role'] == 'cand']
        failures = [r for r in cand_records if r['status'] == 'failed']
        passed = (len(cand_records) == rounds * len(EXPECTED_ARMS)
                  and not failures
                  and all(int(r['bound_violations']) == 0 for r in cand_records))
        abs_vals = [float(r['max_abs_error']) for r in cand_records if r.get('max_abs_error') not in (None, '')]
        scaled_vals = [float(r['max_scaled_error']) for r in cand_records if r.get('max_scaled_error') not in (None, '')]
        bv_vals = [int(r['bound_violations']) for r in cand_records if r.get('bound_violations') not in (None, '')]
        max_abs = max(abs_vals) if abs_vals else None
        max_scaled = max(scaled_vals) if scaled_vals else None
        violations = max(bv_vals) if bv_vals else None
        err_detail = '; '.join(r['reason'] for r in failures if r.get('reason')) or None

        def vals(role, arm):
            return times.get((role, mat, f, arm), [])

        k = vals('cand', 'kernel')
        kd = vals('cand', 'kernel_dup')
        ka = vals('cand', 'kernel_api')
        b = vals('base', 'kernel')
        cc = vals('cand', 'cusparse')
        bc = vals('base', 'cusparse')

        km = statistics.median(k) if k else None
        bm = statistics.median(b) if b else None
        cm = statistics.median(cc) if cc else None
        call = statistics.median(ka) if ka else None
        iqr = 100 * (bench._percentile(k, 75) - bench._percentile(k, 25)) / km if k and km else None
        xb = bm / km if bm is not None and km else None
        xc = cm / km if cm is not None and km else None
        m_rows, nnz, ncols = meta['canonical_shape'][0], meta['canonical_nnz'], meta['nonempty_cols']
        gbps = (4 * (2 * nnz + m_rows + 1) + 4 * f * (ncols + m_rows)) / (km * 1e6) if km else None
        peak = max((float(r['mem_mib']) for r in cand_records if r.get('mem_mib')), default=0.0)
        prep_vals = [float(r['prep_ms']) for r in cand_records if r.get('prep_ms')]
        prep_ms = bench._sig(statistics.median(prep_vals)) if prep_vals else None

        verdict = bench.compute_verdict(xb, bool(failures or not passed), noise)
        cases_out.append({
            'mat': mat,
            'diagnostic_role': exp_meta['diagnostic_role'],
            'F': f,
            'shape': meta['canonical_shape'],
            'nnz': int(nnz),
            'generator': {
                'name': meta['generator'],
                'params': meta['generator_params'],
                'seed': int(meta['seed']),
            },
            'matrix_hash': meta['csr_hash'],
            'label_method': l_method,
            'k': l_k,
            'label_provenance': {
                'cached': (l_cached_raw in ('True', 'true', True)
                           if l_cached_raw not in (None, '') else None),
                'cache_path': l_cache_path or None,
                'clustering_src_sha256': l_csrc or None,
            },
            'label_hash': l_hash,
            'ref_hash': ref_hash,
            'x_sha256': x_sha,
            'actual_kernel_config': {
                'implementation': CONTROLS_KERNEL_CONFIG['implementation'],
                'threads': CONTROLS_KERNEL_CONFIG['threads'],
                'p': CONTROLS_KERNEL_CONFIG['p'],
                'features_per_thread': CONTROLS_KERNEL_CONFIG['features_per_thread'],
                'launch_mode': CONTROLS_KERNEL_CONFIG['launch_mode'],
                'cluster': l_method,
                'k': l_k,
            },
            'correctness': {
                'passed': bool(passed),
                'max_abs_error': max_abs,
                'max_scaled_error': max_scaled,
                'bound_violations': violations,
                'error_detail': err_detail,
            },
            'timing': {
                'k_ms': bench._sig(km),
                'k_iqr': bench._round(iqr, 1),
                'call_ms': bench._sig(call),
                'base_ms': bench._sig(bm),
                'cusp_ms': bench._sig(cm),
                'dup_noise': bench._round(per_case_noise.get((mat, f))),
                'prep_ms': prep_ms,
                'gbps': bench._round(gbps, 1),
                'mem_mib': bench._sig(peak),
                'cand_stats': _arm_stats(k),
                'base_stats': _arm_stats(b),
                'cusp_stats': _arm_stats(cc),
            },
            'timing_samples': {
                'cand_kernel_ms': k,
                'cand_kernel_dup_ms': kd,
                'cand_kernel_api_ms': ka,
                'base_kernel_ms': b,
                'cand_cusparse_ms': cc,
                'base_cusparse_ms': bc,
            },
            'x_base': bench._round(xb),
            'x_cusp': bench._round(xc),
            'verdict': verdict,
        })

    passing = [c for c in cases_out if c['verdict'] != 'F']
    def _geo(key):
        vals = [c[key] for c in passing if c[key] is not None and c[key] > 0]
        return bench._round(math.exp(statistics.mean(map(math.log, vals)))) if vals else None

    gbps_vals = [c['timing']['gbps'] for c in cases_out if c['timing']['gbps'] is not None]
    result = {
        'v': 'synthetic_controls.1',
        'panel': CONTROLS_PANEL,
        'gating': False,
        'promotion_eligible': False,
        'ok': all(c['verdict'] != 'F' for c in cases_out),
        'fmt': 'boo',
        'kernel': 'rowseg',
        'kernel_config': dict(CONTROLS_KERNEL_CONFIG),
        'run_id': run_id,
        'attempt_id': attempt_id,
        'attempt_ordinal': attempt_ordinal,
        'cand': cand,
        'base': base,
        'tool_commit': tool_commit,
        'build_fingerprints': clean_fp,
        'env': env,
        'controls_manifest_sha256': manifest_sha256,
        'noise': bench._round(noise),
        'synthetic_control_summary': {
            'n': len(passing),
            'control_geomean_x_base': _geo('x_base'),
            'control_geomean_x_cusp': _geo('x_cusp'),
            'W': sum(c['verdict'] == 'W' for c in passing),
            'L': sum(c['verdict'] == 'L' for c in passing),
            'T': sum(c['verdict'] == 'T' for c in passing),
            'fail': sum(c['verdict'] == 'F' for c in cases_out),
            'bw_pct_max': bench._round(max(gbps_vals, default=0.0) / 936.0 * 100.0, 1) if gbps_vals else None,
        },
        'cases': cases_out,
        'job_provenance': {
            'job': int(job_info['job']),
            'node': str(job_info['node']),
            'state': str(job_info['state']),
            'exit': int(job_info['exit']),
            'reattached': bool(job_info.get('reattached', False)),
            'resume_key': str(job_info['resume_key']),
        },
        'raw': str(samples_path.relative_to(state.ROOT)),
    }
    schema = json.loads((Path(__file__).parent / 'schemas/synthetic_controls.v1.json').read_text())
    try:
        validate(result, schema)
    except ValueError as exc:
        raise RuntimeError(f'internal synthetic_controls.1 schema violation: {exc}') from exc
    return result


def _resolve_verified_baseline(cfg: dict) -> str:
    if not cfg.get('verified'):
        raise ArgumentError('state.verified is unset; initialize it to the tools baseline commit')
    verified = git('rev-parse', f'{cfg["verified"]}^{{commit}}').stdout.strip()
    tag_res = git('rev-parse', f'{bench.TAG}^{{commit}}', check=False)
    if tag_res.returncode != 0:
        raise ArgumentError(f'required tag {bench.TAG} is missing or does not resolve to a commit')
    tag_sha = tag_res.stdout.strip()
    if tag_sha != verified:
        raise ArgumentError(
            f'state.verified ({verified[:7]}) does not match {bench.TAG} ({tag_sha[:7]})'
        )
    return verified


def synthetic_controls(
    *,
    reps: int = 30,
    warmup: int = 5,
    rounds: int = 3,
    time: str | None = None,
) -> dict:
    """Run the governed, non-gating synthetic_controls paired benchmark for the active attempt."""
    active = attempt_log.active_attempt_context()
    if active is None:
        raise ArgumentError('synthetic_controls requires an active attempt; call begin_attempt first')
    if any(not isinstance(x, int) or x < minimum for x, minimum in ((reps, 1), (warmup, 0), (rounds, 1))):
        raise ArgumentError('reps/rounds must be positive and warmup nonnegative')
    cfg = state.read()
    verified = _resolve_verified_baseline(cfg)
    cand, base, built_cand, built_base = bench.resolve_cand_base_builds(cfg)
    if base != verified:
        raise RuntimeError(f'resolved base {base[:7]} does not match verified {verified[:7]}')
    tool_commit = git(
        'log', '-1', '--format=%H', '--', 'optimizer', 'tests/torch_fast_spdmm/benchmark',
        check=False,
    ).stdout.strip() or head()
    controls_root = _controls_root(cfg)
    if ssh(f'test -f {shlex.quote(controls_root)}/manifest.json', config=cfg).returncode:
        raise ArgumentError('controls manifest missing; run setup_controls with stage=all')
    manifest_text = checked_ssh(f'cat {shlex.quote(controls_root)}/manifest.json', config=cfg).stdout
    manifest_sha256 = hashlib.sha256(manifest_text.encode('utf-8')).hexdigest()
    manifest = json.loads(manifest_text)
    _validate_manifest(manifest)

    present_refs = set(checked_ssh(f'ls {shlex.quote(controls_root)}/refs/ 2>/dev/null || true', config=cfg).stdout.split())
    cases = []
    missing_refs = []
    for m in CONTROLS_PROFILES:
        for f in CONTROLS_FEATURES:
            if f'{m}_F{f}.json' in present_refs:
                cases.append({'mat': m, 'F': f})
            else:
                missing_refs.append(f'{m}_F{f}')
    if missing_refs:
        raise ArgumentError(f'incomplete controls references {missing_refs}; run setup_controls with stage=all')

    build_fingerprints = {
        'cand': {
            'src_hash': built_cand.get('src_hash', ''),
            'build_dir': built_cand['build_dir'],
            'so_sha256': built_cand.get('so_sha256'),
            'compiler': built_cand.get('compiler'),
        },
        'base': {
            'src_hash': built_base.get('src_hash', ''),
            'build_dir': built_base['build_dir'],
            'so_sha256': built_base.get('so_sha256'),
            'compiler': built_base.get('compiler'),
        },
    }
    clean_fp = _clean_build_fingerprints(build_fingerprints)

    order = bench.paired_order(rounds, seed=20261117)
    script = (Path(__file__).parent / 'templates/bench.sh').read_text() + bench.bench_runner(
        order, base_dir=built_base['build_dir'], cand_dir=built_cand['build_dir'],
        kernel_name=CONTROLS_KERNEL, cases_json=json.dumps(cases, separators=(',', ':')),
        reps=reps, warmup=warmup, suite_mode='controls',
    )
    minutes = max(60, 25 * len(cases))
    limit = time or f'{minutes // 60:02d}:{minutes % 60:02d}:00'
    job = submit_slurm_job(script, name='spmm-controls', time=limit, mem='64G',
                           sha=cand, config=cfg, hold=True)
    kind, markers = bench.classify_bench_job(job)
    if kind == 'candidate_crash':
        release_job(job['resume_key'])
        run_local = state.ROOT / 'tests/torch_fast_spdmm/benchmark/results/agent_runs' / cand / str(job['job'])
        run_local.mkdir(parents=True, exist_ok=True)
        raw_tail = '\n'.join(str(job.get('err_tail') or '').splitlines()[-20:])
        failure = {
            'v': 'controls_failure.1',
            'panel': CONTROLS_PANEL,
            'gating': False,
            'promotion_eligible': False,
            'ok': False,
            'failure': 'candidate_runtime',
            'verdict': 'F',
            'fmt': 'boo',
            'kernel': 'rowseg',
            'kernel_config': dict(CONTROLS_KERNEL_CONFIG),
            'run_id': active['run_id'],
            'attempt_id': active['attempt_id'],
            'attempt_ordinal': active['ordinal'],
            'cand': cand,
            'base': base,
            'tool_commit': tool_commit,
            'build_fingerprints': clean_fp,
            'controls_manifest_sha256': manifest_sha256,
            'job_provenance': {
                'job': int(job['job']),
                'node': str(job['node']),
                'state': str(job['state']),
                'exit': int(job['exit']),
                'reattached': bool(job.get('reattached', False)),
                'resume_key': str(job['resume_key']),
            },
            'failures': [m for m in markers if m['role'] == 'cand'],
            'err_tail': str(attempt_log.redact_text(raw_tail)),
            'action': 'candidate runtime crash on synthetic_controls (non-gating); do not re-run',
        }
        failure_schema = json.loads((Path(__file__).parent / 'schemas/controls_failure.v1.json').read_text())
        try:
            validate(failure, failure_schema)
        except ValueError as exc:
            raise RuntimeError(f'internal controls_failure.1 schema violation: {exc}') from exc
        failure_path = run_local / 'controls_failure.json'
        failure_path.write_text(json.dumps(failure, indent=2) + '\n')
        attempt_log.attach_artifact(kind='synthetic_controls_failure_json', local_path=failure_path)
        return failure
    if kind != 'ok':
        release_job(job['resume_key'])
        label = ('controls integrity failure (baseline or harness failed; not attributable to the '
                 'candidate - report it)' if kind == 'integrity' else 'controls infrastructure failure')
        raise InfrastructureError(
            f'{label}: job {job["job"]} ({job["state"]}, exit={job["exit"]}) '
            f'{job["err_tail"] or job["out_tail"]}'
        )

    run_remote = f"{cfg['agent_root'].rstrip('/')}/runs/{cand}/{job['job']}"
    run_local = state.ROOT / 'tests/torch_fast_spdmm/benchmark/results/agent_runs' / cand / str(job['job'])
    transfer('pull_dir', run_remote, run_local, config=cfg)
    try:
        result = aggregate_controls(
            run_local, manifest,
            manifest_sha256=manifest_sha256,
            run_id=active['run_id'],
            attempt_id=active['attempt_id'],
            attempt_ordinal=active['ordinal'],
            cand=cand,
            base=base,
            tool_commit=tool_commit,
            build_fingerprints=build_fingerprints,
            job_info=job,
            reps=reps,
            warmup=warmup,
            rounds=rounds,
        )
        controls_path = run_local / 'synthetic_controls.json'
        controls_path.write_text(json.dumps(result, indent=2) + '\n')
        attempt_log.attach_artifact(kind='synthetic_controls_json', local_path=controls_path)
        attempt_log.attach_artifact(kind='synthetic_controls_raw_csv', local_path=run_local / 'samples.csv')
    finally:
        release_job(job['resume_key'])
    return result
