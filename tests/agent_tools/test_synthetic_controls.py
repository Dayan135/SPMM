"""Offline tests for the governed, non-gating synthetic_controls diagnostic panel."""
from __future__ import annotations

import ast
import copy
import csv
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys

import pytest

from optimizer.agent_tools import __main__ as cli
from optimizer.agent_tools import attempt_log, bench, controls, state
from optimizer.agent_tools.schema import validate
from optimizer.agent_tools.util import ArgumentError, InfrastructureError

from test_attempt_log import _git, _init_loop_repo, _run_cli
from test_campaign_safety import _fail_finalize_once

KERNEL = 'torch_fast_spdmm/kernels/boo_kernel.cu'


def _realistic_manifest() -> dict:
    manifest = {
        'panel': 'synthetic_controls',
        'gating': False,
        'promotion_eligible': False,
        'version': 1,
        'numpy_version': '2.0.0',
        'scipy_version': '1.14.0',
        'python_version': '3.12.0',
        'matrices': {},
    }
    for prof, meta in controls.CONTROLS_PROFILE_META.items():
        manifest['matrices'][prof] = {
            'id': prof,
            'diagnostic_role': meta['diagnostic_role'],
            'generator': meta['generator'],
            'generator_params': copy.deepcopy(meta['generator_params']),
            'generator_src_sha256': '9' * 64,
            'seed': meta['seed'],
            'label_method': meta['label_method'],
            'k': meta['k'],
            'canonical_shape': [262144, 262144],
            'canonical_nnz': 4000000,
            'nonempty_cols': 262144,
            'data_dtype': 'float32',
            'index_dtype': 'int32',
            'csr_hash': ('a' if prof == 'fem2d' else 'b' if prof == 'fem2d_permuted' else 'c') * 64,
            'numpy_version': '2.0.0',
            'scipy_version': '1.14.0',
        }
    return manifest


def _default_fp(cand_dir: str = '/b/cand', base_dir: str = '/b/base') -> dict:
    return {
        'cand': {'src_hash': '111', 'build_dir': cand_dir, 'so_sha256': '1' * 64, 'compiler': 'nvcc'},
        'base': {'src_hash': '222', 'build_dir': base_dir, 'so_sha256': '2' * 64, 'compiler': 'nvcc'},
    }


def _default_job(node: str = 'gpu-01') -> dict:
    return {'job': 503, 'node': node, 'state': 'COMPLETED', 'exit': 0, 'reattached': False, 'resume_key': 'rk3'}


def _write_fake_controls_run(
    run_dir: Path,
    *,
    cand_dir: str = '/b/cand',
    base_dir: str = '/b/base',
    node: str = 'gpu-01',
    gpu: str = 'NVIDIA GeForce RTX 3090',
    rounds: int = 3,
    reps: int = 3,
    fail_case: tuple[str, int] | None = None,
    mix_x_sha: bool = False,
    bad_label_method: bool = False,
    skip_case: bool = False,
    drop_round: int | None = None,
    drop_role: str | None = None,
    drop_arm: tuple[str, str] | None = None,
    sample_delta: int = 0,
    duplicate_summary: bool = False,
    extra_round: bool = False,
    extra_arm: bool = False,
    missing_correctness: bool = False,
    omit_env: tuple[str, int] | None = None,
    mix_round2_cand_dir: str | None = None,
    swap_pkg_paths: bool = False,
    exception_fail_case: tuple[str, int] | None = None,
    dup_repeat_replace: bool = False,
    repeat_zero: bool = False,
    repeat_over: bool = False,
    mismatch_sample_x: bool = False,
    base_exception_fail: tuple[str, int, str] | None = None,
    base_num_fail: tuple[str, int, str] | None = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _realistic_manifest()
    cand_pkg = f'{cand_dir}/torch_fast_spdmm/__init__.py'
    base_pkg = f'{base_dir}/torch_fast_spdmm/__init__.py'
    if swap_pkg_paths:
        cand_pkg, base_pkg = base_pkg, cand_pkg

    for r in range(1, rounds + 1):
        for role, pkg in (('cand', cand_pkg), ('base', base_pkg)):
            if omit_env == (role, r):
                continue
            r_pkg = pkg
            if role == 'cand' and r == 2 and mix_round2_cand_dir:
                r_pkg = f'{mix_round2_cand_dir}/torch_fast_spdmm/__init__.py'
            (run_dir / f'env_{role}_r{r}.json').write_text(json.dumps({
                'role': role, 'round': r, 'node': node, 'gpu': gpu,
                'cuda': '12.4', 'torch': '2.5.0', 'python': '3.12.0',
                'package_file': r_pkg, 'core_file': r_pkg.replace('__init__.py', 'core.so'),
            }))

    cases = [(m, f) for m in controls.CONTROLS_PROFILES for f in controls.CONTROLS_FEATURES]
    sample_fields = ['role', 'round', 'mat', 'F', 'arm', 'repeat', 'position', 'previous',
                     'time_ms', 'node', 'package_file', 'core_file', 'x_sha256']
    summary_fields = ['role', 'round', 'mat', 'F', 'arm', 'status', 'reason', 'median_ms',
                      'mean_ms', 'min_ms', 'max_ms', 'p25_ms', 'p75_ms', 'mad_ms',
                      'prep_ms', 'mem_mib', 'max_abs_error', 'max_scaled_error',
                      'bound_violations', 'label_method', 'label_k', 'label_hash',
                      'label_cached', 'label_cache_path', 'clustering_src_sha256',
                      'ref_hash', 'node', 'package_file', 'core_file', 'x_sha256']

    with (run_dir / 'samples.csv').open('w', newline='') as sf, \
         (run_dir / 'summary.csv').open('w', newline='') as mf:
        sw = csv.DictWriter(sf, fieldnames=sample_fields)
        mw = csv.DictWriter(mf, fieldnames=summary_fields)
        sw.writeheader()
        mw.writeheader()
        dup_written = False
        for mat, f in cases:
            lmethod = controls.CONTROLS_PROFILE_META[mat]['label_method']
            if bad_label_method and mat == 'diag_b1.0':
                lmethod = 'hem'
            for role, pkg in (('cand', cand_pkg), ('base', base_pkg)):
                if drop_role == role:
                    continue
                core_f = pkg.replace('__init__.py', 'core.so')
                xsha = ('1' if (mix_x_sha and role == 'base') else '0') * 64
                round_list = list(range(1, rounds + 1)) + ([rounds + 1] if (extra_round and role == 'cand' and mat == 'fem2d' and f == 128) else [])
                for r in round_list:
                    if drop_round == r:
                        continue
                    arms = list(controls.EXPECTED_ARMS) + (['extra_arm'] if (extra_arm and role == 'cand' and r == 1 and mat == 'fem2d' and f == 128) else [])
                    for arm in arms:
                        if drop_arm == (role, arm):
                            continue
                        is_num_fail = ((fail_case == (mat, f) and role == 'cand' and arm == 'kernel')
                                       or (base_num_fail == (mat, f, arm) and role == 'base' and r == 1))
                        is_exc_fail = ((exception_fail_case == (mat, f) and role == 'cand' and arm == 'kernel')
                                       or (base_exception_fail == (mat, f, arm) and role == 'base' and r == 1))
                        is_fail = is_num_fail or is_exc_fail
                        t_val = 1.0 if role == 'cand' else (1.2 if mat != 'fem2d' else 0.8)
                        if arm == 'cusparse':
                            t_val = 1.5
                        elif arm == 'kernel_dup':
                            t_val = t_val * 1.01
                        elif arm == 'kernel_api':
                            t_val = t_val * 1.05
                        if not is_fail:
                            n_samples = max(0, reps + (sample_delta if (mat == 'fem2d' and f == 128 and role == 'cand' and r == 1 and arm == 'kernel') else 0))
                            for rep in range(1, n_samples + 1):
                                rep_id = rep
                                sx = xsha
                                if mat == 'fem2d' and f == 128 and role == 'cand' and r == 1 and arm == 'kernel':
                                    if dup_repeat_replace and rep == n_samples:
                                        rep_id = 1
                                    elif repeat_zero and rep == 1:
                                        rep_id = 0
                                    elif repeat_over and rep == n_samples:
                                        rep_id = reps + 1
                                    if mismatch_sample_x and rep == 1:
                                        sx = '9' * 64
                                sw.writerow({
                                    'role': role, 'round': r, 'mat': mat, 'F': f, 'arm': arm,
                                    'repeat': rep_id, 'position': 0, 'previous': '',
                                    'time_ms': t_val, 'node': node,
                                    'package_file': pkg, 'core_file': core_f, 'x_sha256': sx,
                                })
                        srow = {
                            'role': role, 'round': r, 'mat': mat, 'F': f, 'arm': arm,
                            'status': 'unsupported' if skip_case else ('failed' if is_fail else 'measured'),
                            'reason': ('RuntimeError: CUDA illegal memory access' if is_exc_fail
                                       else 'numerical: bound violation' if is_num_fail else ''),
                            'median_ms': '' if is_fail else t_val, 'mean_ms': '' if is_fail else t_val,
                            'min_ms': '' if is_fail else t_val, 'max_ms': '' if is_fail else t_val,
                            'p25_ms': '' if is_fail else t_val, 'p75_ms': '' if is_fail else t_val,
                            'mad_ms': '' if is_fail else 0.0,
                            'prep_ms': 12.0, 'mem_mib': 256.0,
                            'max_abs_error': ('' if (is_exc_fail or (missing_correctness and role == 'cand' and r == 1 and arm == 'kernel' and mat == 'fem2d' and f == 128))
                                              else (0.5 if is_num_fail else 1e-6)),
                            'max_scaled_error': '' if is_exc_fail else (0.1 if is_num_fail else 1e-7),
                            'bound_violations': '' if is_exc_fail else (42 if is_num_fail else 0),
                            'label_method': lmethod, 'label_k': 32,
                            'label_hash': 'd' * 64, 'label_cached': True,
                            'label_cache_path': f'/suite/controls/labels/{mat}_{lmethod}_k32.npz',
                            'clustering_src_sha256': 'e' * 64, 'ref_hash': 'f' * 64,
                            'node': node, 'package_file': pkg, 'core_file': core_f, 'x_sha256': xsha,
                        }
                        mw.writerow(srow)
                        if duplicate_summary and not dup_written:
                            mw.writerow(srow)
                            dup_written = True
    return manifest


def test_fixed_control_panel_and_generators():
    bench_root = str(state.ROOT / 'tests/torch_fast_spdmm')
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)
    from benchmark import controls_suite as cs

    assert controls.CONTROLS_PANEL == 'synthetic_controls'
    assert controls.CONTROLS_PROFILES == ('fem2d', 'fem2d_permuted', 'diag_b1.0')
    assert controls.CONTROLS_SIZE == 262144
    assert controls.CONTROLS_FEATURES == (128, 512)
    assert controls.CONTROLS_KERNEL == 'boo:rowseg'
    assert controls.CONTROLS_KERNEL_CONFIG == {
        'implementation': 'boo:rowseg',
        'threads': 256,
        'p': 64,
        'features_per_thread': 1,
        'launch_mode': 'auto',
        'cluster_k': 32,
    }
    assert cs.CONTROL_SPECS['fem2d']['label_method'] == 'hem'
    assert cs.CONTROL_SPECS['fem2d']['k'] == 32
    assert cs.CONTROL_SPECS['fem2d_permuted']['label_method'] == 'hem'
    assert cs.CONTROL_SPECS['fem2d_permuted']['k'] == 32
    assert cs.CONTROL_SPECS['diag_b1.0']['label_method'] == 'identity'
    assert cs.CONTROL_SPECS['diag_b1.0']['k'] == 32

    pbr_text = (state.ROOT / 'tests/torch_fast_spdmm/benchmark/profile_pbr.py').read_text()
    m = re.search(r'"diag_b1\.0":\s*(\([^)]+\))', pbr_text)
    assert m is not None
    assert ast.literal_eval(m.group(1)) == ('diag', {
        'block_size': 64, 'diag_density': 1.0, 'off_diag_density': 0.0,
    })


def test_deterministic_golden_hashes_and_cache_provenance(tmp_path):
    bench_root = str(state.ROOT / 'tests/torch_fast_spdmm')
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)
    from benchmark import controls_suite as cs
    from benchmark import real_matrices as rm
    from benchmark.matrices import generate_diag_blocks_plus_noise, generate_fem_2d, generate_permuted_fem_2d

    golden = {
        'fem2d': ((1024, 1024), 33856, '2924e7ca4fc9e70f6a989ef73a59a715f8ec1429be896f7742f3441793506450'),
        'fem2d_permuted': ((1024, 1024), 33856, '8a81b9cb2b38fb5288f1cf4a71f5a39b4d06ec69f4ce6f9b4beb953a5116b1a8'),
        'diag_b1.0': ((1024, 1024), 65536, 'ff11d8ef092bf38bd34b33c4f033feef869de7604dbcf62ff83c98db3c352ba7'),
    }
    for name, fn, kw in (
        ('fem2d', generate_fem_2d, {'system_size': 1024, 'seed': 0}),
        ('fem2d_permuted', generate_permuted_fem_2d, {'system_size': 1024, 'seed': 0}),
        ('diag_b1.0', generate_diag_blocks_plus_noise, {'system_size': 1024, 'block_size': 64, 'diag_density': 1.0, 'off_diag_density': 0.0, 'seed': 0}),
    ):
        m = fn(**kw).tocsr()
        m.sum_duplicates(); m.sort_indices()
        exp_shape, exp_nnz, exp_hash = golden[name]
        assert (m.shape, m.nnz, rm.matrix_hash(m)) == (exp_shape, exp_nnz, exp_hash)

    # Parameter mutation changes hash
    mutated = generate_diag_blocks_plus_noise(system_size=1024, block_size=64, diag_density=0.5, off_diag_density=0.0, seed=0).tocsr()
    mutated.sum_duplicates(); mutated.sort_indices()
    assert rm.matrix_hash(mutated) != golden['diag_b1.0'][2]

    # Cache provenance invalidation in controls_suite.prepare when numpy_version differs
    import numpy as np
    import scipy.sparse as sp
    root = tmp_path / 'controls_cache'
    (root / 'generated').mkdir(parents=True)
    # Create a tiny cached file with stale numpy_version; prepare must regenerate rather than lie
    stale_npz = root / 'generated' / 'diag_b1.0.npz'
    sp.save_npz(stale_npz, mutated)
    stale_npz.with_suffix('.json').write_text(json.dumps({
        'generator': 'generate_diag_blocks_plus_noise',
        'generator_params': cs.CONTROL_SPECS['diag_b1.0']['generator_params'],
        'generator_src_sha256': cs._generator_src_sha256(),
        'numpy_version': '0.0.1-stale',
        'scipy_version': '0.0.1-stale',
        'csr_hash': rm.matrix_hash(mutated),
    }))
    # Monkeypatch CONTROL_SIZE/CONTROL_PROFILES for fast offline cache-invalidation check
    orig_profiles = cs.CONTROL_PROFILES
    orig_specs = copy.deepcopy(cs.CONTROL_SPECS)
    try:
        cs.CONTROL_PROFILES = ('diag_b1.0',)
        cs.CONTROL_SPECS['diag_b1.0']['size'] = 1024
        cs.CONTROL_SPECS['diag_b1.0']['generator_params']['system_size'] = 1024
        man = cs.prepare(root)
        assert man['matrices']['diag_b1.0']['numpy_version'] == np.__version__
        assert man['matrices']['diag_b1.0']['csr_hash'] == golden['diag_b1.0'][2]
    finally:
        cs.CONTROL_PROFILES = orig_profiles
        cs.CONTROL_SPECS.clear()
        cs.CONTROL_SPECS.update(orig_specs)


def test_paired_protocol_completeness_negative_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(state, 'ROOT', tmp_path)
    fp = _default_fp()
    job = _default_job()

    # Valid complete 3-round, 3-rep run passes
    ok_dir = tmp_path / 'ok'
    man = _write_fake_controls_run(ok_dir, rounds=3, reps=3)
    res = controls.aggregate_controls(
        ok_dir, man, manifest_sha256='0' * 64, run_id='r1', attempt_id=1, attempt_ordinal=1,
        cand='a' * 40, base='b' * 40, tool_commit='c' * 40, build_fingerprints=fp,
        job_info=job, reps=3, warmup=5, rounds=3,
    )
    validate(res, json.loads((Path(controls.__file__).parent / 'schemas/synthetic_controls.v1.json').read_text()))

    negative_cases = [
        ('missing_round', dict(drop_round=2), 'incomplete controls protocol'),
        ('missing_role', dict(drop_role='base'), 'incomplete controls protocol'),
        ('missing_cand_arm', dict(drop_arm=('cand', 'kernel_dup')), 'incomplete controls protocol'),
        ('missing_base_arm', dict(drop_arm=('base', 'cusparse')), 'incomplete controls protocol'),
        ('too_few_samples', dict(sample_delta=-1), 'sample repeat/count mismatch'),
        ('too_many_samples', dict(sample_delta=1), 'out-of-range sample repeat|sample repeat/count mismatch'),
        ('duplicate_summary', dict(duplicate_summary=True), 'duplicate summary record'),
        ('unexpected_round', dict(extra_round=True), 'unexpected summary record'),
        ('unexpected_arm', dict(extra_arm=True), 'unexpected summary record'),
        ('missing_correctness', dict(missing_correctness=True), 'missing correctness field'),
        ('unsupported_case', dict(skip_case=True), 'unsupported/skipped'),
        ('dup_repeat_replace', dict(dup_repeat_replace=True), 'duplicate sample repeat'),
        ('repeat_zero', dict(repeat_zero=True), 'out-of-range sample repeat'),
        ('repeat_over', dict(repeat_over=True), 'out-of-range sample repeat'),
        ('mismatch_sample_x', dict(mismatch_sample_x=True), 'does not match summary'),
    ]
    for name, kw, pattern in negative_cases:
        d = tmp_path / name
        m = _write_fake_controls_run(d, rounds=3, reps=3, **kw)
        with pytest.raises(RuntimeError, match=pattern):
            controls.aggregate_controls(
                d, m, manifest_sha256='0' * 64, run_id='r1', attempt_id=1, attempt_ordinal=1,
                cand='a' * 40, base='b' * 40, tool_commit='c' * 40, build_fingerprints=fp,
                job_info=job, reps=3, warmup=5, rounds=3,
            )


def test_strict_manifest_contract_rejections():
    base_man = _realistic_manifest()
    controls._validate_manifest(base_man)

    mutations = [
        ('wrong_panel', lambda m: m.update(panel='other'), 'panel'),
        ('gating_true', lambda m: m.update(gating=True), 'gating=false'),
        ('promo_true', lambda m: m.update(promotion_eligible=True), 'promotion_eligible=false'),
        ('wrong_ver', lambda m: m.update(version=2), 'version'),
        ('missing_prof', lambda m: m['matrices'].pop('fem2d'), 'profiles mismatch'),
        ('extra_prof', lambda m: m['matrices'].update(extra={}), 'profiles mismatch'),
        ('bad_diag_bsize', lambda m: m['matrices']['diag_b1.0']['generator_params'].update(block_size=32), 'generator_params'),
        ('bad_diag_dens', lambda m: m['matrices']['diag_b1.0']['generator_params'].update(diag_density=0.5), 'generator_params'),
        ('bad_diag_off', lambda m: m['matrices']['diag_b1.0']['generator_params'].update(off_diag_density=0.01), 'generator_params'),
        ('bad_diag_seed', lambda m: m['matrices']['diag_b1.0'].update(seed=1), 'generator/seed'),
        ('bad_gen_src', lambda m: m['matrices']['fem2d'].update(generator_src_sha256='bad'), 'generator_src_sha256'),
        ('bad_dtype', lambda m: m['matrices']['fem2d'].update(data_dtype='float64'), 'unexpected dtypes'),
        ('mixed_numpy', lambda m: m['matrices']['fem2d'].update(numpy_version='1.26.0'), 'mixed numpy/scipy'),
        ('mixed_gen_src', lambda m: m['matrices']['fem2d'].update(generator_src_sha256='1' * 64), 'mixed generator_src_sha256'),
    ]
    for name, mut, pat in mutations:
        m = copy.deepcopy(base_man)
        mut(m)
        with pytest.raises(RuntimeError, match=pat):
            controls._validate_manifest(m)


def test_exception_failed_arm_without_numeric_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(state, 'ROOT', tmp_path)
    d = tmp_path / 'exc_fail'
    man = _write_fake_controls_run(d, rounds=3, reps=3, exception_fail_case=('fem2d', 128))
    res = controls.aggregate_controls(
        d, man, manifest_sha256='0' * 64, run_id='r1', attempt_id=1, attempt_ordinal=1,
        cand='a' * 40, base='b' * 40, tool_commit='c' * 40, build_fingerprints=_default_fp(),
        job_info=_default_job(), reps=3, warmup=5, rounds=3,
    )
    validate(res, json.loads((Path(controls.__file__).parent / 'schemas/synthetic_controls.v1.json').read_text()))
    assert res['ok'] is False
    bad_case = next(c for c in res['cases'] if c['mat'] == 'fem2d' and c['F'] == 128)
    assert bad_case['verdict'] == 'F'
    assert bad_case['correctness']['passed'] is False
    assert 'CUDA illegal memory access' in (bad_case['correctness']['error_detail'] or '')


def test_controls_status_rejects_stale_or_invalid_manifest(tmp_path):
    bench_root = str(state.ROOT / 'tests/torch_fast_spdmm')
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)
    from benchmark import controls_suite as cs

    root = tmp_path / 'bad_status'
    root.mkdir(parents=True)
    man = _realistic_manifest()
    man['matrices']['fem2d']['generator_src_sha256'] = '0' * 64  # stale
    (root / 'manifest.json').write_text(json.dumps(man))
    st = cs.status(root, check_labels=False)
    assert st['ready'] is False
    assert 'stale' in st['reason'] or 'version' in st['reason']


def test_verified_tag_missing_or_mismatched_fails_closed(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    assert _run_cli(capsys, 'begin_attempt', {'title': 'T1', 'hypothesis': 'H1.'})[0] == 0

    # Mismatched opt/verified tag
    _git(repo, 'commit', '--allow-empty', '-m', 'other')
    other_sha = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'tag', '-f', 'opt/verified', other_sha)
    code, res = _run_cli(capsys, 'synthetic_controls', {})
    assert code == 2 and 'does not match opt/verified' in res['detail']

    # Missing opt/verified tag
    _git(repo, 'tag', '-d', 'opt/verified')
    code, res = _run_cli(capsys, 'synthetic_controls', {})
    assert code == 2 and 'opt/verified is missing' in res['detail']


def test_baseline_arm_failures_are_integrity_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(state, 'ROOT', tmp_path)
    for name, kw, exp_substr in (
        ('base_exc', dict(base_exception_fail=('fem2d', 128, 'kernel')), 'fem2d_F128 round=1 arm=kernel'),
        ('base_num', dict(base_num_fail=('diag_b1.0', 512, 'cusparse')), 'diag_b1.0_F512 round=1 arm=cusparse'),
    ):
        d = tmp_path / name
        man = _write_fake_controls_run(d, rounds=3, reps=3, **kw)
        with pytest.raises(InfrastructureError, match='controls integrity failure: verified baseline or shared harness failed') as exc_info:
            controls.aggregate_controls(
                d, man, manifest_sha256='0' * 64, run_id='r1', attempt_id=1, attempt_ordinal=1,
                cand='a' * 40, base='b' * 40, tool_commit='c' * 40, build_fingerprints=_default_fp(),
                job_info=_default_job(), reps=3, warmup=5, rounds=3,
            )
        assert exp_substr in str(exc_info.value)


def test_bench_runner_run_dir_placeholder_and_slurm_render(tmp_path, monkeypatch):
    from optimizer.agent_tools import slurm

    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    cand_sha = _git(repo, 'rev-parse', 'HEAD')
    cfg = state.read()
    order = [('base', 1), ('cand', 1)]

    promo_body = bench.bench_runner(
        order, base_dir='/b/base', cand_dir='/b/cand',
        kernel_name='boo:rowseg', cases_json='[{"mat":"cant","F":128}]',
        reps=30, warmup=5, suite_mode='promotion',
    )
    assert '--out {{RUN_DIR}} ) || rc=$?' in promo_body
    assert '--out {RUN_DIR}' not in promo_body
    assert '--suite controls' not in promo_body

    ctrl_body = bench.bench_runner(
        order, base_dir='/b/base', cand_dir='/b/cand',
        kernel_name='boo:rowseg', cases_json='[{"mat":"fem2d","F":128}]',
        reps=30, warmup=5, suite_mode='controls',
    )
    assert '--out {{RUN_DIR}} --suite controls ) || rc=$?' in ctrl_body
    assert '--out {RUN_DIR}' not in ctrl_body

    expected_out = f"{cfg['agent_root'].rstrip('/')}/runs/{cand_sha}/$SLURM_JOB_ID"
    expected_src = f"{cfg['agent_root'].rstrip('/')}/src/{cand_sha}"
    for body, exp_suite in ((promo_body, False), (ctrl_body, True)):
        rendered = slurm.render(body, name='spmm-test', config=cfg, sha=cand_sha)
        assert f'--out {expected_out}' + (' --suite controls' if exp_suite else '') + ' ) || rc=$?' in rendered
        assert f'use "$build" {expected_src}' in rendered
        assert f'local done_file="{expected_out}/${{role}}_${{round}}.done"' in rendered
        assert '{{RUN_DIR}}' not in rendered
        assert '{RUN_DIR}' not in rendered
        assert not re.search(r'\{(?:RUN_DIR|SRC_DIR|BUILD_DIR|BASE_BUILD_DIR|AGENT_ROOT|NCU|NCU_FLAGS|SHA)\}', rendered)


def test_dotted_profile_reference_metadata_naming(tmp_path):
    bench_root = str(state.ROOT / 'tests/torch_fast_spdmm')
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)
    from benchmark import controls_suite as cs

    root = tmp_path / 'dotted_refs'
    orig_profiles = cs.CONTROL_PROFILES
    orig_specs = copy.deepcopy(cs.CONTROL_SPECS)
    orig_size = cs.CONTROL_SIZE
    try:
        cs.CONTROL_SIZE = 1024
        cs.CONTROL_PROFILES = ('diag_b1.0',)
        cs.CONTROL_SPECS['diag_b1.0']['size'] = 1024
        cs.CONTROL_SPECS['diag_b1.0']['generator_params']['system_size'] = 1024

        man = cs.prepare(root)
        written = cs.refs(root)
        assert set(written.keys()) == {'diag_b1.0_F128', 'diag_b1.0_F512'}
        assert not (root / 'refs' / 'diag_b1.json').exists()

        for f in (128, 512):
            meta_path = root / 'refs' / f'diag_b1.0_F{f}.json'
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert meta['features'] == f
            assert meta['matrix_sha256'] == man['matrices']['diag_b1.0']['csr_hash']
            for part in ('ref', 'absref', 'tol'):
                part_path = root / 'refs' / f'diag_b1.0_F{f}.{part}.npy'
                assert part_path.exists()
                assert part in meta['sha256']

        st = cs.status(root, check_labels=False)
        assert st['ready'] is True
        assert st['missing_matrices'] == []
        assert st['missing_refs'] == []
    finally:
        cs.CONTROL_SIZE = orig_size
        cs.CONTROL_PROFILES = orig_profiles
        cs.CONTROL_SPECS.clear()
        cs.CONTROL_SPECS.update(orig_specs)

    # Direct regression assertion for every fixed (profile, F) reference metadata path
    for prof in controls.CONTROLS_PROFILES:
        for f in controls.CONTROLS_FEATURES:
            assert Path(f"{root / 'refs' / f'{prof}_F{f}'}.json") == root / 'refs' / f'{prof}_F{f}.json'


def test_runtime_build_binding_and_env_rejections(tmp_path):
    fp = _default_fp()
    job = _default_job()

    # Swapped cand/base paths
    d1 = tmp_path / 'swapped'
    m1 = _write_fake_controls_run(d1, swap_pkg_paths=True)
    with pytest.raises(RuntimeError, match='outside expected build_dir'):
        controls.aggregate_controls(d1, m1, manifest_sha256='0' * 64, run_id='r', attempt_id=1, attempt_ordinal=1,
                                    cand='a' * 40, base='b' * 40, tool_commit='c' * 40, build_fingerprints=fp,
                                    job_info=job, reps=3, warmup=5, rounds=3)

    # Omitted env file
    d2 = tmp_path / 'omit_env'
    m2 = _write_fake_controls_run(d2, omit_env=('base', 2))
    with pytest.raises(RuntimeError, match='missing environment file'):
        controls.aggregate_controls(d2, m2, manifest_sha256='0' * 64, run_id='r', attempt_id=1, attempt_ordinal=1,
                                    cand='a' * 40, base='b' * 40, tool_commit='c' * 40, build_fingerprints=fp,
                                    job_info=job, reps=3, warmup=5, rounds=3)

    # Mixed build between rounds
    d3 = tmp_path / 'mix_round'
    m3 = _write_fake_controls_run(d3, mix_round2_cand_dir='/b/other')
    with pytest.raises(RuntimeError, match='outside expected build_dir'):
        controls.aggregate_controls(d3, m3, manifest_sha256='0' * 64, run_id='r', attempt_id=1, attempt_ordinal=1,
                                    cand='a' * 40, base='b' * 40, tool_commit='c' * 40, build_fingerprints=fp,
                                    job_info=job, reps=3, warmup=5, rounds=3)

    # Unknown GPU rejected
    d4 = tmp_path / 'bad_gpu'
    m4 = _write_fake_controls_run(d4, gpu='unknown')
    with pytest.raises(RuntimeError, match='unknown GPU'):
        controls.aggregate_controls(d4, m4, manifest_sha256='0' * 64, run_id='r', attempt_id=1, attempt_ordinal=1,
                                    cand='a' * 40, base='b' * 40, tool_commit='c' * 40, build_fingerprints=fp,
                                    job_info=job, reps=3, warmup=5, rounds=3)


def test_controls_setup_isolated_from_promotion_suite(tmp_path, monkeypatch):
    _init_loop_repo(tmp_path, monkeypatch)
    submitted_scripts = []

    monkeypatch.setattr(controls, 'load_build_env', lambda env_name, modules: {
        'ok': True, 'build_dir': '/remote/builds/abc', 'src_hash': 'abc123', 'so_sha256': '1' * 64,
    })
    monkeypatch.setattr(controls, 'ensure_build', lambda sha, config=None: {
        'ok': True, 'build_dir': '/remote/builds/abc', 'src_hash': 'abc123', 'so_sha256': '1' * 64,
    })
    monkeypatch.setattr(controls, 'submit_slurm_job', lambda body, **kw: (
        submitted_scripts.append((kw.get('name'), body)) or
        {'job': 101, 'state': 'COMPLETED', 'exit': 0, 'out_tail': '', 'err_tail': ''}
    ))

    class _Res:
        stdout = json.dumps({'panel': 'synthetic_controls', 'ready': True, 'missing_matrices': [], 'missing_refs': [], 'missing_labels': []})
        returncode = 0

    monkeypatch.setattr(controls, 'checked_ssh', lambda cmd, **kw: _Res())
    res = controls.setup_controls('all')
    assert res['ready'] is True
    assert all('/suite/controls' in body for _, body in submitted_scripts)
    assert all('universal_suite' not in body for _, body in submitted_scripts)

    # stage='all' raises InfrastructureError if final status is ready=false
    class _NotReady:
        stdout = json.dumps({
            'panel': 'synthetic_controls', 'ready': False,
            'missing_matrices': [], 'missing_refs': ['diag_b1.0_F128'], 'missing_labels': [],
        })
        returncode = 0

    monkeypatch.setattr(controls, 'checked_ssh', lambda cmd, **kw: _NotReady())
    with pytest.raises(InfrastructureError, match='not ready'):
        controls.setup_controls('all')
    # stage='status' returns structured ready=false without raising
    st = controls.setup_controls('status')
    assert st['ready'] is False and st['missing_refs'] == ['diag_b1.0_F128']

    class _Missing:
        returncode = 1
        stdout = ''
        stderr = ''

    monkeypatch.setattr(bench, 'ensure_build', lambda sha, config=None: {
        'ok': True, 'build_dir': '/remote/builds/abc', 'src_hash': 'abc123',
    })
    monkeypatch.setattr(bench, 'ssh', lambda cmd, **kw: _Missing())
    with pytest.raises(ArgumentError, match='suite manifest missing'):
        bench.run_benchmarks('boo:rowseg', ['tune'])


def test_active_attempt_ledger_ownership_crash_and_recovery(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']

    # Active-attempt requirement
    code, res = _run_cli(capsys, 'synthetic_controls', {})
    assert code == 2 and 'requires an active attempt' in res['detail']

    # Arbitrary baseline_ref is disallowed
    assert _run_cli(capsys, 'begin_attempt', {'title': 'T1', 'hypothesis': 'H1.'})[0] == 0
    code, res = _run_cli(capsys, 'synthetic_controls', {'baseline_ref': 'HEAD~1'})
    assert code == 2 and 'unexpected keyword argument' in res['detail']

    def _mock_cluster(*, crash=False, infra_fail=False, numerical_fail=False):
        manifest = _realistic_manifest()
        cand_sha = _git(repo, 'rev-parse', 'HEAD')
        base_sha = state.read()['verified']
        monkeypatch.setattr(bench, 'resolve_cand_base_builds', lambda cfg, ref=None: (
            cand_sha, base_sha,
            {'build_dir': '/b/cand', 'src_hash': '111', 'so_sha256': '1' * 64, 'compiler': 'nvcc'},
            {'build_dir': '/b/base', 'src_hash': '111', 'so_sha256': '1' * 64, 'compiler': 'nvcc'},
        ))

        class _Ok:
            returncode = 0
            stdout = ''

        class _Cat:
            returncode = 0
            stdout = json.dumps(manifest)

        class _Ls:
            returncode = 0
            stdout = '\n'.join(f'{m}_F{f}.json' for m in controls.CONTROLS_PROFILES for f in controls.CONTROLS_FEATURES)

        monkeypatch.setattr(controls, 'ssh', lambda cmd, **kw: _Ok())
        monkeypatch.setattr(controls, 'checked_ssh', lambda cmd, **kw: _Cat() if 'cat ' in cmd else _Ls())
        monkeypatch.setattr(controls, 'release_job', lambda key: None)

        def fake_submit(script, **kw):
            if infra_fail:
                return {'job': 501, 'node': 'gpu-01', 'state': 'NODE_FAIL', 'exit': 1, 'reattached': False,
                        'resume_key': 'rk1', 'out_tail': '', 'err_tail': 'node died'}
            if crash:
                return {'job': 502, 'node': 'gpu-01', 'state': 'FAILED', 'exit': bench.BENCH_EXIT_CANDIDATE_CRASH,
                        'reattached': False, 'resume_key': 'rk2', 'out_tail': 'AGENT_BENCH_VERDICT candidate_crash',
                        'err_tail': 'AGENT_BENCH_FAILURE role=cand round=1 rc=139 TOKEN=ghp_12345678901234567890123456'}
            return {'job': 503, 'node': 'gpu-01', 'state': 'COMPLETED', 'exit': 0, 'reattached': False,
                    'resume_key': 'rk3', 'out_tail': 'AGENT_BENCH_VERDICT ok', 'err_tail': ''}

        monkeypatch.setattr(controls, 'submit_slurm_job', fake_submit)
        monkeypatch.setattr(controls, 'transfer', lambda kind, src, dst, **kw: _write_fake_controls_run(
            Path(dst), rounds=3, reps=30, fail_case=('fem2d', 128) if numerical_fail else None
        ))

    # Infrastructure failure -> exit 3
    _mock_cluster(infra_fail=True)
    code, res = _run_cli(capsys, 'synthetic_controls', {})
    assert code == 3 and res['error'] == 'infrastructure'

    # Candidate crash -> exit 0, governed controls_failure.1, redacted err_tail
    _mock_cluster(crash=True)
    code, res = _run_cli(capsys, 'synthetic_controls', {})
    assert code == 0 and res['v'] == 'controls_failure.1' and res['ok'] is False
    assert res['failure'] == 'candidate_runtime' and res['verdict'] == 'F'
    assert res['panel'] == 'synthetic_controls' and res['gating'] is False and res['promotion_eligible'] is False
    assert 'ghp_1234567890' not in res['err_tail']
    validate(res, json.loads((Path(controls.__file__).parent / 'schemas/controls_failure.v1.json').read_text()))

    # Correctness failure (numerical) -> exit 0, ok=False, verdict='F'
    _mock_cluster(numerical_fail=True)
    code, res = _run_cli(capsys, 'synthetic_controls', {})
    assert code == 0 and res['ok'] is False
    assert any(c['verdict'] == 'F' and not c['correctness']['passed'] for c in res['cases'])

    # Normal success + ledger tool_calls + artifacts
    _mock_cluster()
    code, res = _run_cli(capsys, 'synthetic_controls', {})
    assert code == 0 and res['ok'] is True
    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    arts = conn.execute("SELECT kind, attempt_id, run_id FROM artifacts WHERE kind LIKE 'synthetic_controls%'").fetchall()
    assert {r['kind'] for r in arts} >= {'synthetic_controls_json', 'synthetic_controls_raw_csv', 'synthetic_controls_failure_json'}
    assert all(r['attempt_id'] == 1 for r in arts)
    conn.close()

    # Secret redaction in arguments
    code, res = _run_cli(capsys, 'synthetic_controls', {'secret_token': 'ghp_12345678901234567890123456'})
    assert code == 2
    conn = sqlite3.connect(env['db_path'])
    row = conn.execute("SELECT args_json, error_detail FROM tool_calls WHERE tool_name = 'synthetic_controls' ORDER BY call_id DESC LIMIT 1").fetchone()
    conn.close()
    assert 'ghp_1234567890' not in row[0]

    # promote_attempt rejects synthetic_controls.json
    cand_sha = _git(repo, 'rev-parse', 'HEAD')
    ctrl_path = repo / f'tests/torch_fast_spdmm/benchmark/results/agent_runs/{cand_sha}/503/synthetic_controls.json'
    assert ctrl_path.exists()
    code, res = _run_cli(capsys, 'promote_attempt', {'bench_json': str(ctrl_path)})
    assert code == 2 and 'synthetic_controls' in res['detail']

    # Cross-attempt ownership rejection
    assert _run_cli(capsys, 'finish_attempt', {'outcome': 'ABORTED', 'decision_reason': 'test', 'counts_toward_limit': False})[0] == 0
    assert _run_cli(capsys, 'begin_attempt', {'title': 'T2', 'hypothesis': 'H2.'})[0] == 0
    code, res = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'test', 'counts_toward_limit': False,
        'tune_bench_json': str(ctrl_path),
    })
    assert code == 2

    # Pending-recovery (exit 4) replay for synthetic_controls
    fired = _fail_finalize_once(monkeypatch, 'synthetic_controls')
    code, res = _run_cli(capsys, 'synthetic_controls', {})
    assert code == 4 and res['audit_finalization'] == 'pending_recovery'
    assert Path(res['recovery_record']).name == f'{fired[0]}.json'
    assert _run_cli(capsys, 'get_loop_status', {})[0] == 0


def test_bench1_aggregation_unchanged_with_controls_present():
    target = state.ROOT / 'tests/torch_fast_spdmm/benchmark/results/agent_runs/offline_isolation_test'
    target.mkdir(parents=True, exist_ok=True)
    try:
        sample = target / 'samples.csv'
        summary = target / 'summary.csv'
        with sample.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['role', 'mat', 'F', 'arm', 'time_ms'])
            w.writeheader()
            for mat, c, b in [('cant', 1.0, 1.1), ('soc-Pokec', 2.0, 1.8)]:
                for role, arm, v in [('cand', 'kernel', c), ('cand', 'kernel_dup', c * 1.01),
                                     ('cand', 'kernel_api', c * 1.1), ('cand', 'cusparse', 3.0),
                                     ('base', 'kernel', b), ('base', 'cusparse', 3.03)]:
                    for _ in range(3):
                        w.writerow(dict(role=role, mat=mat, F=128, arm=arm, time_ms=v))
        with summary.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['role', 'mat', 'F', 'arm', 'status', 'reason', 'mem_mib', 'prep_ms'])
            w.writeheader()
            for mat in ('cant', 'soc-Pokec'):
                for role in ('cand', 'base'):
                    for arm in ('kernel', 'kernel_dup', 'kernel_api', 'cusparse'):
                        w.writerow(dict(role=role, mat=mat, F=128, arm=arm, status='measured',
                                        reason='', mem_mib=100, prep_ms=10))
        manifest = {'matrices': {mat: {'canonical_shape': [100, 100], 'canonical_nnz': 1000, 'nonempty_cols': 100}
                                 for mat in ('cant', 'soc-Pokec')}}
        before = bench.aggregate(sample, summary, manifest, fmt='boo', kernel='rowseg',
                                 cand='a079cba', base='a079cba', job=1, node='test', reps=3, warmup=0, rounds=1)
        (target / 'synthetic_controls.json').write_text(json.dumps({
            'v': 'synthetic_controls.1', 'panel': 'synthetic_controls', 'gating': False, 'promotion_eligible': False,
        }))
        after = bench.aggregate(sample, summary, manifest, fmt='boo', kernel='rowseg',
                                cand='a079cba', base='a079cba', job=1, node='test', reps=3, warmup=0, rounds=1)
        assert before == after
    finally:
        shutil.rmtree(target, ignore_errors=True)
