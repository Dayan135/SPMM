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
from benchmark import controls_suite as cs
from benchmark import real_matrices as rm
from benchmark import universal_suite as suite
from optimizer.agent_tools.bench import parse_kernel

SAMPLE_FIELDS = ('role', 'round', 'mat', 'F', 'arm', 'repeat', 'position', 'previous',
                 'time_ms', 'node', 'package_file', 'core_file', 'x_sha256')
SUMMARY_FIELDS = ('role', 'round', 'mat', 'F', 'arm', 'status', 'reason', 'median_ms',
                  'mean_ms', 'min_ms', 'max_ms', 'p25_ms', 'p75_ms', 'mad_ms',
                  'prep_ms', 'mem_mib', 'max_abs_error', 'max_scaled_error',
                  'bound_violations', 'label_method', 'label_k', 'label_hash',
                  'label_cached', 'label_cache_path', 'clustering_src_sha256',
                  'ref_hash', 'node', 'package_file', 'core_file', 'x_sha256')


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


def _boo_labels(spmm, csr, mat, params, root, manifest):
    """HEM clustering labels, cached per (matrix, params, clustering.py source).

    Clustering is pure Python and dominates BOO preparation (about 11 minutes on
    uniform16), yet kernel attempts never change it. The key includes the SHA-256 of
    the clustering.py that this build imports, so an attempt that edits clustering
    misses the cache and re-clusters.
    """
    import torch_fast_spdmm.clustering as clustering
    src = rm.file_sha256(clustering.__file__)[:16]
    key = f"{mat}_{params['cluster']}_{params['target_nr']}x{params['target_nc']}_{src}"
    path = Path(root)/'boo_labels'/f'{key}.npz'
    csr_hash = manifest['matrices'][mat]['csr_hash']
    if path.exists():
        cached = np.load(path)
        if str(cached['csr_hash']) == csr_hash:
            return cached['row_labels'], cached['col_labels'], True
    row, col = clustering.cluster_axes(csr, method=params['cluster'],
                                       target_nr=params['target_nr'], target_nc=params['target_nc'],
                                       max_nr=params['target_nr'], max_nc=params['target_nc'],
                                       max_block_nnz=None, seed=0, symmetric=None,
                                       feasible_first=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez(tmp, row_labels=row, col_labels=col, csr_hash=np.array(csr_hash))
    os.replace(tmp, path)
    return row, col, False


def _case(torch, spmm, core, case, spec, role, round_, out, root, manifest, suite_mode='promotion'):
    mat, f = case['mat'], int(case['F'])
    base = dict(role=role, round=round_, mat=mat, F=f, node=socket.gethostname(),
                package_file=str(Path(spmm.__file__).resolve()),
                core_file=str(Path(core.__file__).resolve()))
    if suite_mode == 'controls':
        csr, _ = cs.load_matrix(root, mat, manifest)
    else:
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
    base['ref_hash'] = ref.sha256
    torch.cuda.reset_peak_memory_stats()
    x = torch.from_numpy(x_cpu).cuda()
    start = time.perf_counter()
    try:
        if spec['fmt'] == 'pbr':
            params = spec['params']
            sparse = spmm.csr_to_pbr(csr, **params)
        else:
            params = spec['params']
            # prep_ms excludes clustering on a cache hit; base and cand share the labels.
            if suite_mode == 'controls':
                row_labels, col_labels, lmeta = cs.load_or_build_labels(csr, mat, root, manifest)
                base.update(
                    label_method=lmeta['label_method'],
                    label_k=lmeta['k'],
                    label_hash=lmeta['label_hash'],
                    label_cached=lmeta['cached'],
                    label_cache_path=lmeta['cache_path'],
                    clustering_src_sha256=lmeta['clustering_src_sha256'],
                )
                cluster_method = lmeta['label_method']
                target_k = lmeta['k']
            else:
                row_labels, col_labels, _ = _boo_labels(spmm, csr, mat, params, root, manifest)
                cluster_method = params['cluster']
                target_k = params['target_nr']
            start = time.perf_counter()
            sparse = spmm.csr_to_boo(csr, cluster=cluster_method,
                                      target_nr=target_k, target_nc=target_k,
                                      row_labels=row_labels, col_labels=col_labels)
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
    ap.add_argument('--suite', choices=['promotion', 'controls'], default='promotion')
    args = ap.parse_args(argv)
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    build_dir = Path(os.environ['PYTHONPATH'].split(':')[0]).resolve()
    assert Path(spmm.__file__).resolve().is_relative_to(build_dir)
    assert Path(core.__file__).resolve().is_relative_to(build_dir)
    print('MODULE_PATHS', spmm.__file__, core.__file__, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ['AGENT_ROOT']) / ('suite/controls' if args.suite == 'controls' else 'suite')
    manifest = json.loads((root/'manifest.json').read_text())
    if args.suite == 'controls':
        if manifest.get('panel') != cs.PANEL_NAME:
            raise ValueError(f'expected {cs.PANEL_NAME} manifest, got {manifest.get("panel")!r}')
        env_info = {
            'role': args.role,
            'round': args.round,
            'node': socket.gethostname(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unknown',
            'cuda': getattr(torch.version, 'cuda', None),
            'torch': getattr(torch, '__version__', None),
            'python': sys.version.split()[0],
            'package_file': str(Path(spmm.__file__).resolve()),
            'core_file': str(Path(core.__file__).resolve()),
        }
        (args.out / f'env_{args.role}_r{args.round}.json').write_text(
            json.dumps(env_info, indent=2, sort_keys=True) + '\n'
        )
    spec = parse_kernel(args.kernel)
    for case in json.loads(args.cases):
        samples, summaries = _case(torch, spmm, core, case, spec, args.role,
                                    args.round, args.out, root, manifest,
                                    suite_mode=args.suite)
        _append(args.out/'samples.csv', samples, SAMPLE_FIELDS)
        _append(args.out/'summary.csv', summaries, SUMMARY_FIELDS)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
