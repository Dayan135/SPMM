"""Paired same-node benchmark jobs and local bench.1 aggregation."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import random
import shlex
import statistics
from urllib.parse import parse_qsl

from optimizer import remote
from . import state
from .build import ensure_build
from .schema import validate
from .slurm import submit_slurm_job
from .util import ArgumentError, checked_ssh, git, head, ssh, transfer

COLS = ['mat', 'F', 'k_ms', 'k_iqr', 'call_ms', 'base_ms', 'cusp_ms',
        'x_base', 'x_cusp', 'gbps', 'mem_mib', 'v']
PBR_VARIANTS = {'block', 'chunkcols', 'nnzflat', 'nnzflat_shared', 'nnzflat_onthefly'}
BOO_VARIANTS = {'rowseg', 'nnzflat', 'legacy'}
TUNE = {'cant', 'web-Google', 'roadNet-CA', 'uniform16', 'blockdiag32'}
HELD = {'soc-Pokec', 'pdb1HYS', 'ASIC_680k', 'amazon0312', 'rail4284', 'pruned16x16'}
SUBSETS = {'tune': ['cant', 'web-Google', 'roadNet-CA', 'uniform16', 'blockdiag32'],
           'held': ['soc-Pokec', 'pdb1HYS', 'ASIC_680k', 'amazon0312', 'rail4284', 'pruned16x16'],
           'smoke': ['cant', 'roadNet-CA', 'uniform16']}
SUBSETS['all'] = SUBSETS['tune'] + SUBSETS['held']


def parse_kernel(value: str) -> dict:
    if not isinstance(value, str) or ':' not in value:
        raise ArgumentError('kernel_name must be fmt:variant[?k=v]')
    fmt, rest = value.split(':', 1)
    variant, _, query = rest.partition('?')
    if fmt not in ('pbr', 'boo') or variant not in (PBR_VARIANTS if fmt == 'pbr' else BOO_VARIANTS):
        raise ArgumentError(f'unknown kernel_name {value!r}')
    params = dict(block_rows=8, block_cols=8, min_nnz_per_block=4) if fmt == 'pbr' else dict(
        cluster='hem', target_nr=32, target_nc=32, threads=256, p=64,
        features_per_thread=1, launch_mode='auto', impl='reference', csr_mode=0)
    allowed = {'br': 'block_rows', 'bc': 'block_cols', 'min': 'min_nnz_per_block',
               'block_rows': 'block_rows', 'block_cols': 'block_cols',
               'min_nnz_per_block': 'min_nnz_per_block'} if fmt == 'pbr' else {
        'cluster': 'cluster', 'kr': 'target_nr', 'kc': 'target_nc', 'threads': 'threads',
        'p': 'p', 'fpt': 'features_per_thread', 'launch_mode': 'launch_mode',
        'impl': 'impl', 'csr_mode': 'csr_mode', 'target_nr': 'target_nr',
        'target_nc': 'target_nc', 'features_per_thread': 'features_per_thread'}
    for key, item in parse_qsl(query, strict_parsing=True):
        if key not in allowed:
            raise ArgumentError(f'unknown kernel parameter {key}')
        mapped = allowed[key]
        if mapped in ('cluster', 'launch_mode', 'impl'):
            params[mapped] = item
        else:
            try:
                params[mapped] = int(item)
            except ValueError as exc:
                raise ArgumentError(f'{key} must be an integer') from exc
    if fmt == 'boo' and params['launch_mode'] not in ('auto', 'legacy'):
        raise ArgumentError('launch_mode must be auto or legacy')
    if fmt == 'boo' and variant == 'nnzflat' and params['features_per_thread'] == 2 and 'impl=' not in query:
        params['impl'] = 'template'
    for key, item in params.items():
        if isinstance(item, int) and (item < 0 if key == 'csr_mode' else item < 1):
            raise ArgumentError(f'{key} must be positive')
    if fmt == 'boo':
        if params['threads'] not in (128, 256, 512) or params['p'] not in (8,16,32,64,128):
            raise ArgumentError('BOO threads or p unsupported')
        if params['impl'] not in ('reference', 'template') or params['csr_mode'] not in (0,1,2):
            raise ArgumentError('BOO impl or csr_mode unsupported')
        if params['features_per_thread'] not in (1,2,4):
            raise ArgumentError('BOO fpt must be 1, 2 or 4')
        if variant == 'nnzflat' and params['features_per_thread'] == 4:
            raise ArgumentError('BOO nnzflat supports fpt 1 or 2')
    return {'fmt': fmt, 'variant': variant, 'params': params,
            'kernel': rest, 'full': value}


def expand(names: list[str]) -> list[str]:
    if not isinstance(names, list) or not names:
        raise ArgumentError('matrix_datasets must be a nonempty list')
    result = []
    for name in names:
        for item in SUBSETS.get(name, [name]):
            if item not in TUNE | HELD:
                raise ArgumentError(f'unknown matrix {item}')
            if item not in result:
                result.append(item)
    return result


def _percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] * (hi-pos) + ordered[hi] * (pos-lo) if lo != hi else ordered[lo]


def _round(value, decimals=3):
    return None if value is None else round(value, decimals)


def _sig(value):
    return None if value is None else float(f'{value:.3g}')


def aggregate(samples_path: Path, summary_path: Path, manifest: dict, *, fmt: str,
              kernel: str, cand: str, base: str, job: int, node: str,
              reps: int, warmup: int, rounds: int) -> dict:
    with samples_path.open(newline='') as file:
        samples = list(csv.DictReader(file))
    with summary_path.open(newline='') as file:
        summaries = list(csv.DictReader(file))
    times = {}
    for row in samples:
        key = (row['role'], row['mat'], int(row['F']), row['arm'])
        times.setdefault(key, []).append(float(row['time_ms']))
    cases = sorted({(r['mat'], int(r['F'])) for r in summaries})
    per_case_noise = []
    for mat, f in cases:
        def med(role, arm):
            xs = times.get((role, mat, f, arm), [])
            return statistics.median(xs) if xs else None
        a, b, c, d = med('cand', 'kernel'), med('cand', 'kernel_dup'), med('base', 'cusparse'), med('cand', 'cusparse')
        if all(x is not None and x > 0 for x in (a, b, c, d)):
            per_case_noise.append(max(abs(b/a-1), abs(c/d-1)))
    noise = _percentile(per_case_noise, 95) or 0.
    rows, skipped, prep = [], [], {}
    for mat, f in cases:
        records = [r for r in summaries if r['mat'] == mat and int(r['F']) == f]
        failures = [r for r in records if r['role'] == 'cand' and r['arm'] in ('kernel', 'kernel_api')
                    and r['status'] == 'failed']
        skip = next((r['reason'] for r in records if r['status'] == 'unsupported'), None)
        if skip and not failures:
            skipped.append([mat, f, skip])
            continue
        def vals(role, arm):
            return times.get((role, mat, f, arm), [])
        k = vals('cand', 'kernel')
        ka = vals('cand', 'kernel_api')
        b = vals('base', 'kernel')
        cusp = vals('cand', 'cusparse')
        if not failures and not (k and ka and b and cusp):
            skipped.append([mat, f, 'unsupported: missing timed arm'])
            continue
        km = statistics.median(k) if k else None
        bm = statistics.median(b) if b else None
        cm = statistics.median(cusp) if cusp else None
        call = statistics.median(ka) if ka else None
        iqr = 100 * (_percentile(k, 75)-_percentile(k, 25))/km if k else None
        xb = bm/km if bm is not None and km else None
        xc = cm/km if cm is not None and km else None
        meta = manifest['matrices'][mat]
        m, nnz, ncols = meta['canonical_shape'][0], meta['canonical_nnz'], meta['nonempty_cols']
        gbps = (4*(2*nnz+m+1)+4*f*(ncols+m))/(km*1e6) if km else None
        peak = max((float(r['mem_mib']) for r in records if r['role'] == 'cand'
                    and r.get('mem_mib')), default=0.)
        if mat not in prep:
            values = [float(r['prep_ms']) for r in records if r['role'] == 'cand'
                      and r.get('prep_ms')]
            if values:
                prep[mat] = _sig(statistics.median(values))
        threshold = max(noise, .04)
        verdict = ('F' if failures else 'W' if xb is not None and xb > 1+threshold else
                   'L' if xb is not None and xb < 1/(1+threshold) else 'T')
        rows.append([mat, f, _sig(km), _round(iqr, 1), _sig(call),
                     _sig(bm), _sig(cm), _round(xb), _round(xc),
                     _round(gbps, 1), _sig(peak), verdict])
    def subset(members):
        selected = [r for r in rows if r[0] in members and r[11] != 'F']
        def geo(index):
            values = [r[index] for r in selected if r[index] is not None and r[index] > 0]
            return _round(math.exp(statistics.mean(map(math.log, values)))) if values else None
        return {'n': len(selected), 'geo_x_base': geo(7), 'geo_x_cusp': geo(8),
                'W': sum(r[11] == 'W' for r in selected),
                'L': sum(r[11] == 'L' for r in selected),
                'T': sum(r[11] == 'T' for r in selected)}
    result = {
        'v': 'bench.1', 'ok': not any(r[11] == 'F' for r in rows),
        'fmt': fmt, 'kernel': kernel, 'cand': cand, 'base': base,
        'env': {'node': node, 'job': job, 'dtype': 'fp32', 'reps': reps,
                'warmup': warmup, 'rounds': rounds},
        'noise': _round(noise),
        'sum': {'tune': subset(TUNE), 'held': subset(HELD),
                'bw_pct_max': _round(max((r[9] for r in rows if r[9] is not None), default=0)/936*100, 1)
                              if rows else None,
                'fail': sum(r[11] == 'F' for r in rows), 'skip': len(skipped)},
        'prep_ms': prep, 'cols': COLS, 'rows': rows, 'skipped': skipped,
        'raw': str(samples_path.relative_to(state.ROOT)),
    }
    try:
        validate(result, json.loads((Path(__file__).parent/'schemas/bench.v1.json').read_text()))
    except ValueError as exc:
        raise RuntimeError(f'internal bench.1 schema violation: {exc}') from exc
    return result


def run_benchmarks(kernel_name: str, matrix_datasets: list[str], *, features=None,
                   tail=False, baseline_ref=None, reps=30, warmup=5, rounds=3,
                   promote=False, time=None) -> dict:
    if promote:
        raise ArgumentError('promote is retired; call promote_attempt with the bench.json path')
    parsed = parse_kernel(kernel_name)
    matrices = expand(matrix_datasets)
    if features is None:
        features = [128] if matrix_datasets == ['smoke'] else [32,64,128,256,512]
    if tail:
        features = list(dict.fromkeys([*features, 33,65,129]))
    if not isinstance(features, list) or not features or any(not isinstance(f,int) or f < 1 for f in features):
        raise ArgumentError('features must be positive integers')
    if any(not isinstance(x, int) or x < minimum for x, minimum in
           ((reps, 1), (warmup, 0), (rounds, 1))):
        raise ArgumentError('reps/rounds must be positive and warmup nonnegative')
    cfg = state.read()
    cand = head()
    if not baseline_ref and not cfg.get('verified'):
        raise ArgumentError('state.verified is unset; initialize it to the tools baseline commit')
    base = git('rev-parse', f'{baseline_ref or cfg["verified"]}^{{commit}}').stdout.strip()
    built_cand = ensure_build(cand, config=cfg)
    built_base = ensure_build(base, config=cfg)
    if not built_cand.get('ok') or not built_base.get('ok'):
        raise ArgumentError('candidate or baseline build unavailable; run load_build_env first')
    root = cfg['agent_root'].rstrip('/')
    if ssh(f'test -f {root}/suite/manifest.json', config=cfg).returncode:
        raise ArgumentError('suite manifest missing; run setup_suite with stage=all')
    # Cases whose reference was never ingested (e.g. a download that failed upstream) are
    # reported in `skipped` instead of aborting the run; the job runs the rest.
    present = set(checked_ssh(f'ls {root}/suite/refs/ 2>/dev/null || true', config=cfg).stdout.split())
    cases, unavailable = [], []
    for m in matrices:
        for f in features:
            if f'{m}_F{f}.json' in present:
                cases.append({'mat': m, 'F': f})
            else:
                unavailable.append([m, f, 'missing: reference not in suite (run setup_suite)'])
    if not cases:
        raise ArgumentError('no requested case has a reference; run setup_suite with stage=all or refs')
    manifest = json.loads(checked_ssh(f'cat {root}/suite/manifest.json', config=cfg).stdout)
    seed = 20261117
    command = shlex.quote(json.dumps(cases, separators=(',', ':')))
    runner = []
    for r in range(1, rounds+1):
        roles = ['base', 'cand']
        random.Random(seed*1000003+r).shuffle(roles)
        for role in roles:
            directory = built_base['build_dir'] if role == 'base' else built_cand['build_dir']
            runner.append(f'if [ ! -f "{{{{RUN_DIR}}}}/{role}_{r}.done" ]; then')
            runner.append(f'  use {shlex.quote(directory)} {{{{SRC_DIR}}}}')
            runner.append('  python -m benchmark.agent_bench '
                          f'--role {role} --round {r} --kernel {shlex.quote(kernel_name)} '
                          f'--cases {command} --reps {reps} --warmup {warmup} '
                          '--out {{RUN_DIR}}')
            runner.append(f'  touch "{{{{RUN_DIR}}}}/{role}_{r}.done"')
            runner.append('fi')
    script = (Path(__file__).parent/'templates/bench.sh').read_text() + '\n'.join(runner) + '\n'
    minutes = max(60, 25*len(cases))
    limit = time or f'{minutes//60:02d}:{minutes%60:02d}:00'
    job = submit_slurm_job(script, name='spmm-bench', time=limit, mem='64G', sha=cand, config=cfg)
    if job['state'] != 'COMPLETED' or job['exit']:
        raise ArgumentError(f'benchmark job {job["job"]} failed: {job["err_tail"] or job["out_tail"]}')
    run_remote = f'{root}/runs/{cand}/{job["job"]}'
    run_local = state.ROOT / 'tests/torch_fast_spdmm/benchmark/results/agent_runs' / cand / str(job['job'])
    transfer('pull_dir', run_remote, run_local, config=cfg)
    result = aggregate(run_local/'samples.csv', run_local/'summary.csv', manifest,
                       fmt=parsed['fmt'], kernel=parsed['kernel'], cand=cand, base=base,
                       job=job['job'], node=job['node'], reps=reps, warmup=warmup, rounds=rounds)
    result['skipped'].extend(unavailable)
    result['sum']['skip'] = len(result['skipped'])
    expected = {(c['mat'], c['F']) for c in cases} | {(u[0], u[1]) for u in unavailable}
    reported = {(r[0], r[1]) for r in result['rows']} | {(r[0], r[1]) for r in result['skipped']}
    if reported != expected:
        raise RuntimeError(f'benchmark case coverage mismatch: missing={sorted(expected-reported)}, '
                           f'extra={sorted(reported-expected)}')
    (run_local/'bench.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


# Promotion gate: fixed rules, no judgment. A promoted attempt becomes the baseline for
# every later attempt, so a regression on `held` blocks promotion even if `tune` improved.
PROMOTE_GATE = {'tune_geo_min': 1.02, 'held_geo_min': 0.99}


def promote_attempt(bench_json: str) -> dict:
    """Advance state.verified and tag opt/verified to the candidate if it passes the gate.

    Requires: tests_passed for the candidate; the bench.1 result is ok, measured against
    the current verified commit, and covers both `tune` and `held`; tune geo_x_base >=
    1.02 with W > L; held geo_x_base >= 0.99 with L <= W.
    """
    path = Path(bench_json)
    if not path.is_absolute():
        path = state.ROOT / path
    result = json.loads(path.read_text())
    validate('bench', result)
    cfg = state.read()
    cand, base = result['cand'], result['base']
    verified = git('rev-parse', f'{cfg["verified"]}^{{commit}}').stdout.strip()
    tune, held = result['sum']['tune'], result['sum']['held']
    reasons = []
    if not cfg['tests_passed'].get(cand):
        reasons.append('tests have not passed for the candidate')
    if not result['ok']:
        reasons.append('bench.ok is false (a correctness failure)')
    if base != verified:
        reasons.append(f'baseline {base[:7]} is not the current verified {verified[:7]}')
    if cand != head():
        reasons.append(f'candidate {cand[:7]} is not HEAD')
    if not tune['n'] or not held['n']:
        reasons.append('the run must cover both tune and held (use matrix_datasets ["all"])')
    else:
        if tune['geo_x_base'] < PROMOTE_GATE['tune_geo_min'] or tune['W'] <= tune['L']:
            reasons.append(f"tune geo_x_base {tune['geo_x_base']} (W/L {tune['W']}/{tune['L']}) "
                           f"below gate {PROMOTE_GATE['tune_geo_min']} with W > L")
        if held['geo_x_base'] < PROMOTE_GATE['held_geo_min'] or held['L'] > held['W']:
            reasons.append(f"held geo_x_base {held['geo_x_base']} (W/L {held['W']}/{held['L']}) "
                           f"below gate {PROMOTE_GATE['held_geo_min']} or L > W")
    if reasons:
        return {'promoted': False, 'cand': cand, 'verified': verified, 'reasons': reasons}
    with state.locked() as current:
        current['verified'] = cand
    git('tag', '-f', 'opt/verified', cand)
    git('push', '-f', 'origin', 'refs/tags/opt/verified')
    return {'promoted': True, 'cand': cand, 'previous_verified': verified, 'reasons': []}
