"""Interleaved timing of row-segmented BOO P x FPT x slab variants vs original-order CSR.

One cell = (case, F). Every cell validates and times, in one seeded-rotation
interleaved order: torch CSR, each requested direct cuSPARSE algorithm (C++ and
complete-call scopes, original coordinates), and each requested BOO
(P, FPT, slab) variant at complete-call scope (boo_matmul, auto launch) and
optionally C++ scope (BOOTimedSpMM, legacy fork/join -- identical to auto
whenever a CSR remainder exists). One BOO object per case is reused by all
variants. FPT and slab width are process-wide: every BOO entry applies both
immediately before the call, outside the timed interval for slab (FPT is also
re-applied inside boo_matmul, exactly as in the full sweep).

Before timing, each BOO variant is validated against the float64 reference and
its launched kernel name is checked against the requested specialization
(torch.profiler); infeasible requests are recorded, never substituted.

  python -m benchmark.bench_tiles --labels L --cache C --out OUT \
      --cases pwtk fem2d --features 128 512 --p 16 32 64 128 --fpt 1 2 4 --slab 512 --cpp
  python -m benchmark.bench_tiles ... --configs 64:2:512 32:4:256   # explicit (P:FPT:slab) list
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import random
import socket
import statistics
import time

import numpy as np

from benchmark import bench_real as br
from benchmark import real_matrices as rm
from benchmark import tiles_common as tc

ANCHOR = (64, 2, 512)
FIELDS = (
    'allocation', 'slurm_job_id', 'node', 'case', 'role', 'matrix', 'labeling', 'cluster', 'k',
    'rows', 'cols', 'stored_entries', 'features', 'config', 'family', 'algorithm', 'direct_algorithm',
    'scope', 'coords', 'supported', 'unsupported_reason', 'correctness', 'max_abs_error',
    'max_scaled_error', 'bound_violations', 'median_ms', 'mean_ms', 'min_ms', 'max_ms', 'p25_ms',
    'p75_ms', 'repeats', 'warmup', 'speedup_vs_torch_csr', 'speedup_vs_best_direct_same_scope',
    'threads', 'p', 'effective_p', 'features_per_thread', 'feat_block', 'grid_z', 'arena_bytes',
    'registers_per_thread', 'local_bytes_per_thread', 'max_active_blocks_per_sm',
    'launch_verified', 'launch_mode', 'num_blocks', 'max_nr', 'max_nc', 'max_nnz_blk',
    'compressed_nnz', 'remainder_nnz', 'plan_build_ms', 'workspace_bytes', 'preprocess_status',
    'cusparse_version', 'matrix_sha256', 'label_sha256', 'x_sha256', 'seed', 'order_seed',
    'threshold', 'explicit_zeros', 'order_mode')
SAMPLE_FIELDS = ('allocation', 'case', 'features', 'config', 'repeat', 'position', 'previous', 'time_ms',
                 'order_mode')
LAUNCH_FIELDS = ('case', 'features', 'p', 'features_per_thread_requested', 'feat_block_requested',
                 'supported', 'unsupported_reason', 'requested_p', 'effective_p', 'threads',
                 'features_per_thread', 'feat_block', 'grid_z', 'arena_bytes',
                 'registers_per_thread', 'local_bytes_per_thread', 'static_shared_bytes',
                 'max_threads_per_block', 'max_dynamic_shared_bytes', 'max_active_blocks_per_sm',
                 'shared_limit_bytes', 'num_blocks')


def boo_name(p, fpt, slab):
    return f'boo_p{p}_fpt{fpt}_s{slab}'


def prepared_timer(torch, prepare, fn):
    """CUDA events around fn() only; prepare() (settings) runs before the start event."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()

    def timed():
        prepare()
        start.record()
        y = fn()
        end.record()
        end.synchronize()
        del y
        return start.elapsed_time(end)
    return timed


def orders(count, repeats, seed, mode):
    """'rotation': bench_spmm.round_orders (one cyclic order rotated, so each variant
    always follows the same predecessor). 'shuffle': an independent seeded shuffle per
    repeat, so predecessor effects (clock/power/cache carry-over) become noise
    instead of a fixed per-variant bias."""
    if mode == 'rotation':
        from benchmark.bench_spmm import round_orders
        return round_orders(count, repeats, seed)
    out = []
    for r in range(repeats):
        order = list(range(count))
        random.Random(seed * 1000003 + r).shuffle(order)
        out.append(order)
    return out


def measure(torch, variants, reference, warmup, repeats, seed, order_mode):
    """bench_real.measure with a selectable interleaving order; samples record predecessors."""
    import math
    kept, unsupported = [], []
    for v in variants:
        try:
            # Named range around the validation call (outside timing): an nsys run of the
            # same command audits which kernel each configuration launched.
            torch.cuda.nvtx.range_push(f"validate:{v.config['config']}")
            y = v.run()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.nvtx.range_pop()
            torch.cuda.empty_cache()
            unsupported.append(dict(v.config, supported=False,
                                    unsupported_reason='GPU out of memory at validation call'))
            continue
        result = reference.check(y, v.row_perm)
        del y
        if result['bound_violations']:
            raise AssertionError(f"{v.config['config']}: {result['bound_violations']} entries exceed "
                                 f"the float32 error bound (max scaled {result['max_scaled_error']:.3g})")
        kept.append((v, dict(result, correctness='passed')))
    for v, _ in kept:
        for _ in range(warmup):
            v.run()
    torch.cuda.synchronize()
    samples = []
    previous = ''
    for repeat, order in enumerate(orders(len(kept), repeats, seed, order_mode)):
        for position, index in enumerate(order):
            v = kept[index][0]
            t = v.timed()
            if not math.isfinite(t) or t <= 0:
                raise RuntimeError(f"invalid elapsed time for {v.config['config']}: {t}")
            samples.append(dict(config=v.config['config'], repeat=repeat + 1, position=position,
                                previous=previous, time_ms=t))
            previous = v.config['config']
    summaries = []
    for v, check in kept:
        values = [s['time_ms'] for s in samples if s['config'] == v.config['config']]
        summaries.append(dict(v.config, **check, supported=True, unsupported_reason='',
                              median_ms=statistics.median(values), mean_ms=statistics.mean(values),
                              min_ms=min(values), max_ms=max(values),
                              p25_ms=float(np.percentile(values, 25)),
                              p75_ms=float(np.percentile(values, 75)),
                              repeats=repeats, warmup=warmup, order_mode=order_mode))
    br._speedups(summaries)
    return summaries, samples, unsupported


def write_csv(path, rows, fields):
    exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerows(rows)


def build_variants(torch, core, spmm, csr, x, boo, info, configs, args, limit, launch_rows):
    rows, features = csr.shape[0], x.shape[1]
    out_bytes = rows * features * 4
    transient = 2 * out_bytes + (64 << 20)
    variants, unsupported = [], []

    def skip(config, reason):
        unsupported.append(dict(config, supported=False, unsupported_reason=reason))

    a = br._to_torch_csr(torch, csr)
    if not args.no_torch:
        variants.append(br.Variant(dict(config='torch_csr', family='torch', algorithm='torch_csr',
                                        scope='full_call', coords='original'),
                                   lambda: torch.sparse.mm(a, x),
                                   br._event_timer(torch, lambda: torch.sparse.mm(a, x))))
    boo_common = dict(algorithm='boo', threads=args.threads, launch_mode='auto', coords='original')
    cpp_plans = {}
    boo_variants = []
    for p, fpt, slab in configs:
        launch, reason = tc.launch_info(core, boo, features, args.threads, p, fpt, slab, limit)
        launch_rows.append(dict(launch or {}, case=info['case'], features=features, p=p,
                                features_per_thread_requested=fpt, feat_block_requested=slab,
                                supported=reason is None, unsupported_reason=reason or '',
                                shared_limit_bytes=limit, num_blocks=info['num_blocks']))
        config = dict(boo_common, config=boo_name(p, fpt, slab), family='boo', scope='full_call',
                      p=p, features_per_thread=fpt, feat_block=slab,
                      **({k: launch[k] for k in ('effective_p', 'grid_z', 'arena_bytes',
                                                 'registers_per_thread', 'local_bytes_per_thread',
                                                 'max_active_blocks_per_sm')} if launch else {}))
        if reason:
            skip(config, reason)
            if args.cpp:
                skip(dict(config, config=config['config'] + '_cpp', scope='cpp_only',
                          launch_mode='legacy_forkjoin', family='boo_cpp'), reason)
            continue
        prepare = (lambda fpt=fpt, slab=slab: tc.apply_settings(core, fpt, slab))
        call = (lambda p=p, fpt=fpt: spmm.boo_matmul(boo, x, threads=args.threads, p=p, row_seg=True,
                                                     csr_mode=0, launch_mode='auto',
                                                     features_per_thread=fpt))
        run = (lambda prepare=prepare, call=call: (prepare(), call())[1])
        status = tc.verify_call(torch, run, launch)
        if status != 'verified' and not status.startswith('UNVERIFIED'):
            raise RuntimeError(f"{config['config']}: {status}")
        config['launch_verified'] = status
        boo_variants.append(br.Variant(config, run, prepared_timer(torch, prepare, call)))
        if args.cpp:
            cconfig = dict(config, config=config['config'] + '_cpp', scope='cpp_only',
                           launch_mode='legacy_forkjoin', family='boo_cpp')
            ep = launch['effective_p']
            if ep not in cpp_plans:
                if not br._headroom(torch, transient + out_bytes):
                    cpp_plans[ep] = 'skipped: GPU memory budget (BOOTimedSpMM output)'
                else:
                    meta = boo._gpu_meta
                    cpp_plans[ep] = core.BOOTimedSpMM(
                        boo.rows, boo.cols, boo.max_nr, boo.max_nc, boo.max_nnz_blk,
                        meta.row_grp_ptr, meta.row_ids, meta.col_grp_ptr, meta.col_ids,
                        meta.blk_row_grp, meta.blk_col_grp, meta.blk_nz_ptr, meta.blk_shift,
                        meta.nz_idx, meta.nz_val, meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
                        meta.tail_row, meta.tail_col, meta.tail_val, x, args.threads, ep, 0, True,
                        False)
            plan = cpp_plans[ep]
            if isinstance(plan, str):
                skip(cconfig, plan)
                continue

            def crun(plan=plan, prepare=prepare):
                prepare()
                return plan.run()

            def ctimed(plan=plan, prepare=prepare):
                prepare()  # before every C++ entry; the C++ events bracket only the call
                return plan.timed_run()
            boo_variants.append(br.Variant(cconfig, crun, ctimed))
    variants += boo_variants

    for alg in args.direct_algorithms:
        dconf = dict(family='direct', algorithm='cusparse_direct', direct_algorithm=alg)
        configs_d = [dict(dconf, config=f'direct_{alg}', scope='cpp_only', coords='original'),
                     dict(dconf, config=f'direct_{alg}_api', scope='full_call', coords='original')]
        if not br._headroom(torch, transient + out_bytes):
            for c in configs_d:
                skip(c, 'skipped: GPU memory budget (needs one call of transient headroom)')
            continue
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            plan = core.DirectSpMM(a.crow_indices(), a.col_indices(), a.values(), x, alg)
            torch.cuda.synchronize()
            build_ms = 1000 * (time.perf_counter() - start)
        except torch.cuda.OutOfMemoryError:
            for c in configs_d:
                skip(c, 'GPU out of memory at plan construction')
            continue
        pinfo = plan.info()
        if not pinfo['supported']:
            for c in configs_d:
                skip(dict(c, **pinfo), pinfo['unsupported_reason'])
            continue
        for c in configs_d:
            c.update(plan_build_ms=build_ms, **pinfo)
        variants.append(br.Variant(configs_d[0], plan.run, plan.timed_run))
        variants.append(br.Variant(configs_d[1], plan.run, br._event_timer(torch, plan.run)))
    return variants, unsupported


def parse_configs(args):
    if args.configs:
        out = []
        for item in args.configs:
            p, fpt, slab = (int(v) for v in item.split(':'))
            out.append((p, fpt, slab))
    else:
        out = list(itertools.product(args.p, args.fpt, args.slab))
    if args.anchor and ANCHOR not in out:
        out.append(ANCHOR)
    if len(set(out)) != len(out):
        raise SystemExit('duplicate (P, FPT, slab) configurations')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', nargs='+', choices=sorted(tc.CASES), default=list(tc.TUNING_CASES))
    ap.add_argument('--features', nargs='+', type=int, default=[128, 512])
    ap.add_argument('--p', nargs='+', type=int, choices=[8, 16, 32, 64, 128], default=[16, 32, 64, 128])
    ap.add_argument('--fpt', nargs='+', type=int, choices=[1, 2, 4], default=[1, 2, 4])
    ap.add_argument('--slab', nargs='+', type=int, choices=[128, 256, 512, 1024], default=[512])
    ap.add_argument('--configs', nargs='+', help='explicit P:FPT:slab list (overrides the product)')
    ap.add_argument('--no-anchor', dest='anchor', action='store_false',
                    help='do not add the P64/FPT2/slab512 anchor')
    ap.add_argument('--cpp', action='store_true', help='also time BOOTimedSpMM (C++ scope)')
    ap.add_argument('--direct-algorithms', nargs='*', choices=br.DIRECT_PRIORITY,
                    default=list(br.DIRECT_PRIORITY))
    ap.add_argument('--no-torch', action='store_true')
    ap.add_argument('--threads', type=int, default=256, choices=[128, 256, 512])
    ap.add_argument('--threshold', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0, help='labels/X seed (labels were made with 0)')
    ap.add_argument('--order-seed', type=int, default=0, help='interleaving order seed')
    ap.add_argument('--order', choices=['shuffle', 'rotation'], default='shuffle',
                    help='per-repeat independent shuffle (default) or the inherited rotation')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--allocation', default=os.environ.get('SLURM_JOB_ID', 'local'))
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=rm.default_cache())
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    configs = parse_configs(args)

    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    from benchmark.bench_spmm import source_metadata
    if not torch.cuda.is_available():
        raise SystemExit('GPU required')
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    meta = source_metadata(root)
    build = root / 'results' / 'build_hashes.json'
    meta.update(build_hashes=json.loads(build.read_text()) if build.exists() else None,
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                configs=configs, node=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), status='running', cases={},
                timing='seeded-rotation interleaving of all variants of one (case, F) cell')
    meta_path = args.out / 'metadata.json'
    rm._write_json(meta_path, meta)
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    host = dict(allocation=args.allocation, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                node=socket.gethostname(), order_seed=args.order_seed)
    try:
        with torch.inference_mode():
            for name in args.cases:
                case = tc.CASES[name]
                t0 = time.perf_counter()
                csr, boo, info = tc.load_case(case, args.cache, args.labels, threshold=args.threshold,
                                              seed=args.seed)
                info['load_build_s'] = time.perf_counter() - t0
                meta['cases'][name] = info
                boo.to('cuda')
                for features in args.features:
                    x_cpu = torch.randn(csr.shape[1], features,
                                        generator=torch.Generator().manual_seed(args.seed))
                    reference = br.Reference(csr, x_cpu.numpy())
                    x = x_cpu.cuda()
                    common = dict(info, **host, features=features,
                                  x_sha256=rm.digest_arrays(x_cpu.numpy()))
                    print(f'cell {name} F={features}', flush=True)
                    launch_rows = []
                    variants, unsupported = build_variants(torch, core, spmm, csr, x, boo, info,
                                                           configs, args, limit, launch_rows)
                    summaries, samples, more = measure(torch, variants, reference, args.warmup,
                                                       args.repeats, args.order_seed, args.order)
                    tc.apply_settings(core, 1, tc.DEFAULT_SLAB)
                    rows_out = [dict(common, **s) for s in summaries + unsupported + more]
                    write_csv(args.out / 'summary.csv', rows_out, FIELDS)
                    write_csv(args.out / 'samples.csv', [dict(common, **s, order_mode=args.order)
                                                         for s in samples],
                              SAMPLE_FIELDS)
                    write_csv(args.out / 'launch.csv', launch_rows, LAUNCH_FIELDS)
                    for r in sorted(rows_out, key=lambda r: (r.get('median_ms') is None,
                                                              r.get('median_ms') or 0)):
                        if r.get('median_ms') is None:
                            print(f"  {r['config']:28s} UNSUPPORTED: {r['unsupported_reason']}", flush=True)
                        else:
                            print(f"  {r['config']:28s} {r['scope']:9s} {r['median_ms']:9.4f} ms "
                                  f"[{r['p25_ms']:.4f}, {r['p75_ms']:.4f}]", flush=True)
                    del variants, reference, x
                    torch.cuda.empty_cache()
                boo.to('cpu')
                del csr, boo
                torch.cuda.empty_cache()
                rm._write_json(meta_path, meta)
        meta['status'] = 'complete'
    except BaseException as exc:
        meta.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        tc.apply_settings(core, 1, tc.DEFAULT_SLAB)
        rm._write_json(meta_path, meta)
    print(f'BENCH_TILES_DONE {args.out}', flush=True)


if __name__ == '__main__':
    main()
