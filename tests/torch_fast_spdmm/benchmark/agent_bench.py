"""One build per process; validated, interleaved C++ and API timing arms."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from benchmark import bench_nnzflat as bn
from benchmark import bench_real as br
from benchmark import real_matrices as rm
from benchmark import universal_suite as suite
from optimizer.agent_tools.bench import parse_kernel

SAMPLE_FIELDS = ('role', 'round', 'mat', 'F', 'arm', 'repeat', 'position', 'previous',
                 'time_ms', 'node', 'package_file', 'core_file', 'x_sha256')
SUMMARY_FIELDS = ('role', 'round', 'mat', 'F', 'arm', 'status', 'reason', 'median_ms',
                  'prep_ms', 'mem_mib', 'node', 'package_file', 'core_file', 'x_sha256')


def _append(path, rows, columns):
    with path.open('a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def _pbr_plan(core, pbr, x, kernel):
    meta = pbr._gpu_meta
    return core.PBRTimedSpMM(
        pbr.cols, pbr.rows, pbr.block_rows, pbr.block_cols,
        meta.codes, meta.coords, meta.offsets, meta.data,
        meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
        x, 0, 2048, False, kernel)


def _boo_plan(core, boo, x, params, variant):
    meta = boo._gpu_meta
    rowseg = variant != 'nnzflat'
    impl = params['impl']
    if variant == 'legacy':
        launch_mode = 'legacy'
    else:
        launch_mode = params['launch_mode']
    if rowseg:
        core.set_boo_rowseg_fpt(params['features_per_thread'])
        n_fpt, reference = 1, True
    else:
        n_fpt, reference = params['features_per_thread'], impl == 'reference'
    return core.BOOTimedSpMM(
        boo.rows, boo.cols, boo.max_nr, boo.max_nc, boo.max_nnz_blk,
        meta.row_grp_ptr, meta.row_ids, meta.col_grp_ptr, meta.col_ids,
        meta.blk_row_grp, meta.blk_col_grp, meta.blk_nz_ptr, meta.blk_shift,
        meta.nz_idx, meta.nz_val, meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
        meta.tail_row, meta.tail_col, meta.tail_val, x, params['threads'], params['p'],
        params['csr_mode'], rowseg, False, n_fpt, reference, launch_mode)


def _case(torch, spmm, core, case, spec, role, round_, out, root, manifest):
    mat, f = case['mat'], int(case['F'])
    base = dict(role=role, round=round_, mat=mat, F=f, node=socket.gethostname(),
                package_file=str(Path(spmm.__file__).resolve()),
                core_file=str(Path(core.__file__).resolve()))
    csr, _ = suite.load_matrix(root, mat, os.environ['SPMM_MATRIX_CACHE'])
    csr_bytes = csr.indptr.nbytes + csr.indices.nbytes + csr.data.nbytes
    x_bytes = csr.shape[1]*f*4
    y_bytes = csr.shape[0]*f*4
    predicted = x_bytes + y_bytes + 3*csr_bytes + (1 << 30) + (64 << 20)
    if predicted > 22 * (1 << 30):
        return [], [dict(base, arm='kernel', status='unsupported', reason='skip:oom-budget')]
    x_cpu = suite.make_x(csr.shape[1], f)
    xsha = suite.x_hash(x_cpu)
    base['x_sha256'] = xsha
    ref = bn.GpuReference(torch, root/'refs'/f'{mat}_F{f}',
                          manifest['matrices'][mat]['csr_hash'], xsha, f)
    torch.cuda.reset_peak_memory_stats()
    x = torch.from_numpy(x_cpu).cuda()
    start = time.perf_counter()
    try:
        if spec['fmt'] == 'pbr':
            params = spec['params']
            sparse = spmm.csr_to_pbr(csr, **params)
        else:
            params = spec['params']
            sparse = spmm.csr_to_boo(csr, cluster=params['cluster'],
                                      target_nr=params['target_nr'], target_nc=params['target_nc'])
        sparse.to('cuda')
        torch.cuda.synchronize()
        prep_ms = 1000*(time.perf_counter()-start)
        if spec['fmt'] == 'pbr':
            try:
                plan = _pbr_plan(core, sparse, x, spec['variant'])
            except TypeError:
                return [], [dict(base, arm='kernel', status='unsupported',
                                 reason='unsupported in base' if role == 'base' else
                                        'unsupported: no cpp_only timer')]
            call = lambda: spmm.pbr_matmul(sparse, x, kernel=spec['variant'])
            prepare = lambda: None
        else:
            try:
                plan = _boo_plan(core, sparse, x, params, spec['variant'])
            except TypeError:
                return [], [dict(base, arm='kernel', status='unsupported',
                                 reason='unsupported in base' if role == 'base' else
                                        'unsupported: no cpp_only timer')]
            rowseg = spec['variant'] != 'nnzflat'
            launch = 'legacy' if spec['variant'] == 'legacy' else params['launch_mode']
            prepare = lambda: core.set_boo_rowseg_fpt(params['features_per_thread']) if rowseg else None
            call = lambda: spmm.boo_matmul(
                sparse, x, threads=params['threads'], p=params['p'], csr_mode=params['csr_mode'],
                row_seg=rowseg, launch_mode=launch,
                features_per_thread=params['features_per_thread'],
                nnzflat_impl=params['impl'] if not rowseg else None)
        a = br._to_torch_csr(torch, csr)
        direct = core.DirectSpMM(a.crow_indices(), a.col_indices(), a.values(), x, 'DEFAULT')
        info = direct.info()
        if not info['supported']:
            return [], [dict(base, arm='cusparse', status='unsupported',
                             reason='unsupported: '+info['unsupported_reason'])]
        shared = dict(base, prep_ms=prep_ms)
        plan_run = lambda: (prepare(), plan.run())[1]
        plan_timed = lambda: (prepare(), plan.timed_run())[1]
        arms = [bn.Arm('kernel', dict(shared, arm='kernel'), prepare, plan_run,
                       plan_timed, ref.check),
                bn.Arm('kernel_dup', dict(shared, arm='kernel_dup'), prepare, plan_run,
                       plan_timed, ref.check),
                bn.Arm('kernel_api', dict(shared, arm='kernel_api'), prepare, call,
                       bn.event_timer(torch, prepare, call), ref.check),
                bn.Arm('cusparse', dict(shared, arm='cusparse'), lambda: None,
                       direct.run, direct.timed_run, ref.check)]
        summary, samples, failed = bn.measure(torch, arms, args.warmup, args.reps,
                                               20261117 + round_*1009 + f)
        for row in failed:
            reason = row.get('reason', '')
            if role == 'base' and reason.startswith('AttributeError:'):
                row.update(status='unsupported', reason='unsupported in base')
            elif 'unsupported' in reason.lower() and not row.get('bound_violations'):
                row.update(status='unsupported', reason='unsupported: '+reason[:200])
        peak = torch.cuda.max_memory_allocated()/(1 << 20)
        for row in summary + failed:
            row.update(base, prep_ms=prep_ms, mem_mib=peak)
        for row in samples:
            row.update(base)
        return samples, summary + failed
    except AttributeError as exc:
        if role == 'base':
            return [], [dict(base, arm='kernel', status='unsupported',
                             reason='unsupported in base')]
        raise
    except (NotImplementedError, RuntimeError) as exc:
        if isinstance(exc, NotImplementedError) or 'unsupported' in str(exc).lower():
            return [], [dict(base, arm='kernel', status='unsupported',
                             reason='unsupported: '+str(exc)[:200])]
        raise
    finally:
        torch.cuda.empty_cache()


def main(argv=None):
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', choices=['base', 'cand'], required=True)
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--kernel', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--reps', type=int, required=True)
    ap.add_argument('--warmup', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    build_dir = Path(os.environ['PYTHONPATH'].split(':')[0]).resolve()
    assert Path(spmm.__file__).resolve().is_relative_to(build_dir)
    assert Path(core.__file__).resolve().is_relative_to(build_dir)
    print('MODULE_PATHS', spmm.__file__, core.__file__, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ['AGENT_ROOT'])/'suite'
    manifest = json.loads((root/'manifest.json').read_text())
    spec = parse_kernel(args.kernel)
    for case in json.loads(args.cases):
        samples, summaries = _case(torch, spmm, core, case, spec, args.role,
                                    args.round, args.out, root, manifest)
        _append(args.out/'samples.csv', samples, SAMPLE_FIELDS)
        _append(args.out/'summary.csv', summaries, SUMMARY_FIELDS)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
