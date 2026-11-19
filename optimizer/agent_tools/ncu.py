"""Strict Nsight Compute raw-page reduction to ncu.1."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import shlex

from .schema import validate
from scripts.research_panel.nnzflat_profile_report import STALLS, STALL_FMT, num, rows_of

from . import attempt_log, state
from .util import ArgumentError, checked_ssh, transfer
from optimizer import remote

NCU_FLAGS = ('--set full --metrics '
             'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,'
             'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,'
             'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum '
             '--profile-from-start off --target-processes all --import-source no '
             '-k regex:pbr_spmm|boo_spmm|csr|cusparse -f')

FLOP = tuple(f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum'
             for op in ('ffma', 'fadd', 'fmul'))
DERIVED_FFMA = 'derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2'
SOL = {'sm': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
       'mem': 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
       'dram': 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
       'l2': 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
       'l1': 'l1tex__throughput.avg.pct_of_peak_sustained_active'}
DIRECT = {
    'launch.grid': 'launch__grid_size', 'launch.block': 'launch__block_size',
    'launch.regs': 'launch__registers_per_thread',
    'occ.ach': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'occ.theo': 'sm__maximum_warps_per_active_cycle_pct',
    'occ.waves': 'launch__waves_per_multiprocessor',
    'warp.lanes': 'smsp__thread_inst_executed_per_inst_executed.ratio',
    'warp.branch_eff': 'smsp__sass_average_branch_targets_threads_uniform.pct',
    'warp.div_br': 'smsp__sass_branch_targets_threads_divergent.sum',
    'warp.ipc': 'sm__inst_executed.avg.per_cycle_active',
    'mem.l1_hit': 'l1tex__t_sector_hit_rate.pct',
    'mem.l2_hit': 'lts__t_sector_hit_rate.pct',
}
EXTRA = (
    'gpu__time_duration.sum', 'launch__shared_mem_per_block_static',
    'launch__shared_mem_per_block_dynamic',
    *(f'launch__occupancy_limit_{name}' for name in ('registers', 'shared_mem', 'warps', 'blocks')),
    *FLOP, 'dram__bytes_read.sum', 'dram__bytes_write.sum',
    'sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained',
    'sm__cycles_elapsed.avg.per_second', 'dram__bytes.sum.peak_sustained',
    'dram__cycles_elapsed.avg.per_second',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'smsp__sass_inst_executed_op_shared_ld.sum',
    'smsp__sass_inst_executed_op_shared_st.sum',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum',
    'l1tex__t_requests_pipe_lsu_mem_global_op_red.sum',
    'l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum',
    *(STALL_FMT.format(s) for s in STALLS),
)
EXPECTED = tuple(dict.fromkeys((*SOL.values(), *DIRECT.values(), *EXTRA)))
RAW_FIELDS = EXPECTED + (DERIVED_FFMA,)
COUNTS = {'warp.div_br', 'mem.atom_req', 'mem.smem_inst'}
COLS = ['name', 'us', 'share', 'launch', 'sol', 'roof', 'occ', 'warp', 'stall', 'mem']


def _q(value, digits=3):
    if value is None or not math.isfinite(value):
        return None
    return float(f'{value:.{digits}g}')


def _ratio(a, b):
    return None if a is None or b is None or b == 0 else a / b


def _sum(a, b):
    return None if a is None or b is None else a + b


def _name(value: str) -> str:
    value = value.split('(')[0].strip()
    value = re.sub(r'<.*', '', value)
    return value.split()[-1]


def _value(row, idx, units, key):
    # Launch metrics carry a per-block unit ('Kbyte/block'); scale by the prefix part.
    return num(row[idx[key]], units[idx[key]].removesuffix('/block')) if key in idx else None


def _load(path: Path):
    header, units, rows = rows_of(path)
    idx = {name: i for i, name in enumerate(header)}
    if 'Kernel Name' not in idx:
        raise ArgumentError(f'{path}: no Kernel Name column')
    missing = sorted(name for name in EXPECTED if name not in idx)
    by_name = defaultdict(list)
    for row in rows:
        raw = {name: _value(row, idx, units, name) for name in RAW_FIELDS}
        by_name[_name(row[idx['Kernel Name']])].append(raw)
    if not by_name:
        raise ArgumentError(f'{path}: no kernel rows')
    return by_name, missing


def _aggregate(name, launches):
    durations = [r['gpu__time_duration.sum'] / 1000 for r in launches]
    us = sum(durations)
    total = sum(durations)

    def v(key, *, count=False):
        vals = [row[key] for row in launches]
        if any(x is None for x in vals):
            return None
        return sum(vals) if count else sum(x * w for x, w in zip(vals, durations)) / total

    def div(n, d):
        return _ratio(v(n, count=True), v(d, count=True))

    smem = _sum(v('launch__shared_mem_per_block_static'),
                v('launch__shared_mem_per_block_dynamic'))
    limits = [(kind, v(f'launch__occupancy_limit_{metric}'))
              for kind, metric in [('registers', 'registers'), ('smem', 'shared_mem'),
                                   ('warps', 'warps'), ('blocks', 'blocks')]]
    limiter = min(limits, key=lambda p: p[1])[0] if all(x is not None for _, x in limits) else None
    flop_counts = [v(key, count=True) for key in FLOP]
    if all(x is not None for x in flop_counts):
        flops, src = 2 * flop_counts[0] + flop_counts[1] + flop_counts[2], 'full'
    elif v(DERIVED_FFMA, count=True) is not None:
        flops, src = v(DERIVED_FFMA, count=True), 'ffma_only'
    else:
        flops, src = None, None
    bytes_ = _sum(v('dram__bytes_read.sum', count=True),
                  v('dram__bytes_write.sum', count=True))
    ridge = _ratio(2 * v('sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained') *
                   v('sm__cycles_elapsed.avg.per_second'),
                   v('dram__bytes.sum.peak_sustained') *
                   v('dram__cycles_elapsed.avg.per_second')) if all(
                       v(k) is not None for k in (
                           'sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained',
                           'sm__cycles_elapsed.avg.per_second', 'dram__bytes.sum.peak_sustained',
                           'dram__cycles_elapsed.avg.per_second')) else None
    ai = _ratio(flops, bytes_)
    stall_raw = {reason: v(STALL_FMT.format(reason)) for reason in STALLS}
    stall_sum = sum(x for x in stall_raw.values() if x is not None)
    stalls = {name: 100 * val / stall_sum for name, val in stall_raw.items()
              if val is not None and stall_sum > 0}
    top = sorted(stalls.items(), key=lambda p: -p[1])
    full = {
        'name': name, 'us': _q(us), 'share': 0.,
        'launch': {'grid': round(v(DIRECT['launch.grid'])) if v(DIRECT['launch.grid']) is not None else None,
                   'block': round(v(DIRECT['launch.block'])) if v(DIRECT['launch.block']) is not None else None,
                   'regs': round(v(DIRECT['launch.regs'])) if v(DIRECT['launch.regs']) is not None else None,
                   'smem_kb': _q(smem / 1024) if smem is not None else None},
        'sol': {k: round(v(metric), 1) if v(metric) is not None else None
                for k, metric in SOL.items()},
        'roof': {'ai': _q(ai), 'gflops': _q(flops / (us * 1000)) if flops is not None else None,
                 'ridge_ai': _q(ridge),
                 'bound': ('memory' if ai < ridge else 'compute') if ai is not None and ridge is not None else None,
                 'src': src},
        'occ': {'ach': round(v(DIRECT['occ.ach']), 1) if v(DIRECT['occ.ach']) is not None else None,
                'theo': round(v(DIRECT['occ.theo']), 1) if v(DIRECT['occ.theo']) is not None else None,
                'limiter': limiter, 'waves': _q(v(DIRECT['occ.waves']))},
        'warp': {k: (round(v(DIRECT[f'warp.{k}']), 1) if v(DIRECT[f'warp.{k}']) is not None
                     else None) if k == 'branch_eff' else
                 _q(v(DIRECT[f'warp.{k}'], count=k == 'div_br'))
                 for k in ('lanes', 'branch_eff', 'div_br', 'ipc')},
        'stall': [[reason, round(pct, 1)] for reason, pct in top[:5]],
        'mem': {
            'l1_hit': round(v(DIRECT['mem.l1_hit']), 1) if v(DIRECT['mem.l1_hit']) is not None else None,
            'l2_hit': round(v(DIRECT['mem.l2_hit']), 1) if v(DIRECT['mem.l2_hit']) is not None else None,
            'dram_gb': _q(bytes_ / 1e9) if bytes_ is not None else None,
            'ld_sec_req': _q(div('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
                                 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum')),
            'smem_conf_ld': _q(div('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
                                   'smsp__sass_inst_executed_op_shared_ld.sum')),
            'smem_conf_st': _q(div('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
                                   'smsp__sass_inst_executed_op_shared_st.sum')),
            'red_sec_req': _q(div('l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum',
                                  'l1tex__t_requests_pipe_lsu_mem_global_op_red.sum')),
            'atom_req': _q(_sum(v('l1tex__t_requests_pipe_lsu_mem_global_op_red.sum', count=True),
                                v('l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum', count=True))),
            'smem_inst': _q(_sum(v('smsp__sass_inst_executed_op_shared_ld.sum', count=True),
                                 v('smsp__sass_inst_executed_op_shared_st.sum', count=True))),
        },
    }
    return full, stalls


def _flags(kernel, stalls):
    result = []
    def add(test, flag):
        if test:
            result.append(flag)
    add(stalls.get('mio_throttle', 0) > 25, 'smem_pipe_pressure')
    add(stalls.get('barrier', 0) > 20, 'barrier_bound')
    occ = kernel['occ']
    if occ['limiter'] and occ['ach'] is not None and occ['theo'] is not None:
        add(occ['ach'] < .7 * occ['theo'] or occ['theo'] < 50,
            'occ_limiter:' + occ['limiter'])
    warp, mem, sol = kernel['warp'], kernel['mem'], kernel['sol']
    for value, bound, flag in [(warp['lanes'], 24, 'lane_underuse'),
                               (warp['branch_eff'], 90, 'divergence')]:
        add(value is not None and value < bound, flag)
    add(mem['ld_sec_req'] is not None and mem['ld_sec_req'] > 5, 'uncoalesced_ld')
    add(mem['red_sec_req'] is not None and mem['red_sec_req'] > 4, 'uncoalesced_red')
    add(sol['dram'] is not None and sol['sm'] is not None and
        sol['dram'] > 70 and sol['sm'] < 40, 'memory_bound')
    add(all(x is not None for x in sol.values()) and max(sol.values()) < 40,
        'latency_bound')
    return result


def _flatten(kernel, stalls):
    d = {'us': kernel['us']}
    for section in ('sol', 'occ', 'warp', 'mem'):
        d.update({f'{section}.{k}': v for k, v in kernel[section].items()
                  if isinstance(v, (int, float))})
    d.update({f'stall.{k}': v for k, v in stalls.items()})
    return d


def _local_path(location: str) -> tuple[Path, str]:
    cfg = state.read()
    if ':' in location and location.split(':', 1)[0] == cfg['remote_host']:
        remote_path = location.split(':', 1)[1]
        local = state.ROOT / 'tests/torch_fast_spdmm/benchmark/results/agent_runs' / Path(remote_path).name
        local.parent.mkdir(parents=True, exist_ok=True)
        transfer('pull', remote_path, local, config=cfg)
        stem = remote_path[:-8] if remote_path.endswith('_raw.csv') else str(Path(remote_path).with_suffix(''))
        local_stem = local.name[:-8] if local.name.endswith('_raw.csv') else local.stem
        transfer('pull', stem + '.case.json', local.with_name(local_stem + '.case.json'),
                 config=cfg)
        return local, remote_path
    p = Path(location)
    if p.exists():
        return p, location
    if str(p).startswith(cfg['agent_root']):
        return _local_path(cfg['remote_host'] + ':' + str(p))
    raise ArgumentError(f'profile not found: {location}')


def _export(location: str) -> tuple[Path, str]:
    path, display = _local_path(location)
    if path.suffix == '.csv':
        return path, display
    if path.suffix != '.ncu-rep':
        raise ArgumentError('ncu_report_file must be .ncu-rep or _raw.csv')
    cfg = state.read()
    remote_path = display.split(':', 1)[-1] if ':' in display else None
    if remote_path is None:
        import uuid
        remote_path = f"{cfg['agent_root'].rstrip('/')}/imports/{uuid.uuid4().hex}/{path.name}"
        checked_ssh(f'mkdir -p {shlex.quote(str(Path(remote_path).parent))}', config=cfg)
        transfer('push', path, remote_path, config=cfg)
    raw_remote = str(Path(remote_path).with_suffix('')) + '_raw.csv'
    cmd = (f'"$CONDA_PREFIX/nsight-compute-2025.2.1/ncu" --import '
           f'{shlex.quote(remote_path)} --csv --page raw > {shlex.quote(raw_remote)}')
    checked_ssh('bash -lc ' + shlex.quote('source ~/.bashrc; conda activate ' + cfg['env_name'] + '; ' + cmd),
                timeout=600, config=cfg)
    raw = path.with_name(path.stem + '_raw.csv')
    transfer('pull', raw_remote, raw, config=cfg)
    return raw, display


def _parse(location: str):
    path, display = _export(location)
    sidecar = path.with_name(path.name[:-8] + '.case.json') if path.name.endswith('_raw.csv') else path.with_suffix('.case.json')
    if not sidecar.exists():
        raise ArgumentError(f'case metadata missing: {sidecar}')
    case = json.loads(sidecar.read_text())
    groups, missing = _load(path)
    ordered = sorted(groups.items(), key=lambda item: -sum(
        row['gpu__time_duration.sum'] for row in item[1]))
    entries = [_aggregate(name, launches) for name, launches in ordered]
    total = sum(sum(row['gpu__time_duration.sum'] for row in launches)/1000
                for _, launches in ordered)
    out = []
    for (kernel, stalls), (_, launches) in zip(entries, ordered):
        kernel['share'] = _q(sum(row['gpu__time_duration.sum'] for row in launches)/1000 / total)
        if kernel['share'] < .15:
            kernel = {key: kernel[key] for key in ('name', 'us', 'share', 'sol')}
            kernel['stall'] = [[reason, round(pct, 1)] for reason, pct in
                               sorted(stalls.items(), key=lambda p: -p[1])[:3]]
        out.append(kernel)
    return case, out, entries, missing, display, path


def parse_ncu_profile(ncu_report_file: str, baseline_report: str | None = None) -> dict:
    case, kernels, entries, missing, display, path = _parse(ncu_report_file)
    differences = {}
    main, stalls = entries[0]
    if baseline_report:
        base_case, _, base_entries, _, _, _ = _parse(baseline_report)
        if any(case[k] != base_case[k] for k in ('mat', 'F', 'fmt', 'kernel')):
            raise ArgumentError('baseline profile is for a different case')
        matching = next(((k, s) for k, s in base_entries if k['name'] == main['name']), None)
        if matching:
            candidate_values = _flatten(main, stalls)
            base_values = _flatten(*matching)
            for key, cand in candidate_values.items():
                base = base_values.get(key)
                if base is None or cand is None:
                    continue
                if key == 'us':
                    delta = cand / base - 1 if base else None
                    if delta is not None and abs(delta) > .05:
                        differences[key] = _q(delta)
                elif abs(cand - base) > .05 * abs(base):
                    differences[key] = _q(cand - base)
    result = {'v': 'ncu.1', 'rep': display, 'case': case, 'k': kernels,
              'd': differences, 'flags': _flags(main, stalls), 'missing': missing}
    schema = json.loads((Path(__file__).parent / 'schemas/ncu.v1.json').read_text())
    try:
        validate(result, schema)
    except ValueError as exc:
        raise RuntimeError(f'internal ncu.1 schema violation: {exc}') from exc
    stem = path.name[:-8] if path.name.endswith('_raw.csv') else path.stem
    out = path.with_name(stem + '.ncu.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    attempt_log.attach_artifact(kind='ncu_json', local_path=out)
    attempt_log.attach_artifact(
        kind='ncu_report',
        local_path=path if path.exists() else None,
        remote_path=display if display != str(path) else None,
    )
    return result
