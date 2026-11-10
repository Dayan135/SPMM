"""GPU timing driver for the BOO nnzflat FPT1/2 experiment (one manifest task).

  python -m benchmark.bench_nnzflat --root RUN --manifest RUN/manifests/stage2.json \
      --task 3 --out RUN/results/stage2/task_003 --cache C [--allocation alloc1]

A task holds cells. A cell is one feature width F with a list of BOO objects
(input, k, arms) and optional per-input baselines; every arm of a cell is
validated, launch-verified and then timed in one interleaved run with an
independent seeded shuffle per repeat, on the same allocation, A and X. All arms
of one (input, k) share one BOO object; objects at different k are different
representations of the same A. Nothing here generates matrices, clusters, or computes the CPU
reference: inputs, labels, BOO digests and float64 references come from the CPU
preparation job and are hash-checked before use (benchmark.nnzflat_common).

Arm names (benchmark.nnzflat_common.arm_name):
  <input>@k<k>:<arm> where <arm> is
  nf_p{P}_t{T}_f{R}       nnzflat FPT template (R=1 is the identical-work control)
  nfref_p{P}_t{T}_f1      original nnzflat kernel (FPT1 reference)
  rs_p{P}_t{T}_f{R}       row-segmented kernel at the same P/T/FPT
  *_dup                   the same arm timed twice (pure measurement-noise control)
  direct_<ALG>[_api], torch_csr   original-order baselines (confirmation)
Infeasible arms are recorded with the launcher's reason; failures are recorded
and the remaining arms continue. Timing is complete-call (CUDA events around one
boo_matmul with launch_mode='auto': Y allocation + zeroing, main kernel,
remainder and join); settings are applied before the start event.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import time
import traceback

import numpy as np

from benchmark import bench_real as br
from benchmark import nnzflat_common as nc
from benchmark import real_matrices as rm

SUMMARY_FIELDS = (
    'stage', 'allocation', 'slurm_job_id', 'node', 'gpu_name', 'gpu_uuid', 'task', 'cell',
    'input', 'matrix', 'method', 'k', 'labeling', 'features', 'arm', 'family', 'kernel', 'impl',
    'p', 'threads', 'fpt', 'duplicate', 'requested', 'status', 'reason', 'alias_of',
    'scope', 'coords', 'direct_algorithm', 'correctness', 'max_abs_error', 'max_scaled_error',
    'bound_violations', 'median_ms', 'mean_ms', 'min_ms', 'max_ms', 'p25_ms', 'p75_ms', 'mad_ms',
    'repeats', 'warmup', 'launch_verified', 'launched_kernel', 'effective_p', 'arena_bytes',
    'registers_per_thread', 'local_bytes_per_thread', 'static_shared_bytes',
    'max_active_blocks_per_sm', 'grid_x', 'grid_y', 'grid_z', 'shared_limit_bytes',
    'tile_passes', 'stepb_items', 'stepb_max_trips_p50', 'stepb_max_trips_max',
    'active_fraction_mean', 'active_fraction_nnz_weighted', 'barriers_per_cta', 'shared_atomics',
    'padding_items', 'num_blocks', 'max_nr', 'max_nc', 'max_nnz_blk', 'compressed_nnz',
    'remainder_nnz', 'remainder_fraction', 'nnz_per_block_p50', 'nnz_per_block_p95',
    'row_nnz_in_block_p95', 'block_skew_nnz_weighted', 'active_col_fraction',
    'plan_build_ms', 'workspace_bytes', 'preprocess_status', 'cusparse_version',
    'matrix_sha256', 'label_sha256', 'boo_digest', 'x_sha256', 'ref_sha256', 'order_seed')
SAMPLE_FIELDS = ('stage', 'allocation', 'task', 'cell', 'input', 'k', 'features', 'arm', 'repeat',
                 'position', 'previous', 'time_ms', 'order_seed')
STRUCT_KEYS = ('num_blocks', 'max_nr', 'max_nc', 'max_nnz_blk', 'compressed_nnz', 'remainder_nnz',
               'remainder_fraction', 'nnz_per_block_p50', 'nnz_per_block_p95',
               'row_nnz_in_block_p95', 'block_skew_nnz_weighted', 'active_col_fraction')
LAUNCH_KEYS = ('effective_p', 'arena_bytes', 'registers_per_thread', 'local_bytes_per_thread',
               'static_shared_bytes', 'max_active_blocks_per_sm', 'grid_x', 'grid_y', 'grid_z')
WORK_KEYS = ('tile_passes', 'stepb_items', 'stepb_max_trips_p50', 'stepb_max_trips_max',
             'active_fraction_mean', 'active_fraction_nnz_weighted', 'barriers_per_cta',
             'shared_atomics', 'padding_items')


class GpuReference:
    """CPU float64 reference (saved by the CPU job), checked on the GPU in row chunks.

    Same bound as bench_real.Reference: |y - ref| <= tol_i * (|A||X|)_i, NaN fails.
    """

    def __init__(self, torch, stem, matrix_sha256, x_sha256, features):
        meta = json.loads(Path(f'{stem}.json').read_text())
        if (meta['matrix_sha256'], meta['x_sha256'], meta['features']) != (
                matrix_sha256, x_sha256, features):
            raise ValueError(f'{stem}: reference does not match matrix/X/F')
        for part in ('ref', 'absref', 'tol'):
            got = nc.sha256_file(f'{stem}.{part}.npy')
            if got != meta['sha256'][part]:
                raise ValueError(f'{stem}.{part}.npy: sha256 mismatch')
        self.torch = torch
        self.ref = np.load(f'{stem}.ref.npy', mmap_mode='r')
        self.absref = np.load(f'{stem}.absref.npy', mmap_mode='r')
        self.tol = torch.from_numpy(np.load(f'{stem}.tol.npy')).cuda()
        self.sha256 = nc.sha256_text(json.dumps(meta['sha256'], sort_keys=True))

    def check(self, y, chunk=32768):
        torch = self.torch
        y = y.reshape(y.shape[-2], y.shape[-1])
        violations, max_abs, max_scaled = 0, 0.0, 0.0
        for i in range(0, y.shape[0], chunk):
            ref = torch.from_numpy(np.ascontiguousarray(self.ref[i:i + chunk])).cuda().double()
            absref = torch.from_numpy(np.ascontiguousarray(self.absref[i:i + chunk])).cuda()
            err = (y[i:i + chunk].double() - ref).abs()
            bound = self.tol[i:i + chunk, None] * absref.double()
            violations += int((~(err <= bound)).sum())
            if err.numel():
                max_abs = max(max_abs, float(torch.nan_to_num(err, nan=float('inf')).max()))
            nz = absref > 0
            if bool(nz.any()):
                max_scaled = max(max_scaled, float((err[nz] / absref[nz].double()).max()))
        return dict(max_abs_error=max_abs, max_scaled_error=max_scaled, bound_violations=violations)


def event_timer(torch, prepare, fn):
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


def write_csv(path, rows, fields):
    exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerows(rows)


class Arm:
    def __init__(self, name, row, prepare=None, run=None, timed=None, check=None, verify=None):
        self.name, self.row = name, row
        self.prepare, self.run, self.timed, self.check, self.verify = prepare, run, timed, check, verify


def load_input_ctx(torch, root, cache, name, features):
    """Canonical CSR, X and the hash-checked float64 reference for one (input, F)."""
    inp = nc.INPUTS[name]
    csr, record = nc.load_input(root, name, cache)
    x_cpu = nc.make_x(csr.shape[1], features)
    x_sha = rm.digest_arrays(x_cpu.numpy())
    ref = GpuReference(torch, nc.ref_stem(root, name, features), record['matrix_sha256'], x_sha,
                       features)
    return dict(input=name, matrix=inp.key, method=inp.method, csr=csr, record=record,
                features=features, x=x_cpu.cuda(), x_sha256=x_sha, ref=ref, ref_sha256=ref.sha256,
                matrix_sha256=record['matrix_sha256'])


def prepare_object(root, ictx, k, label_hashes):
    """Saved labels -> BOO rebuilt on this node, digest-checked against the CPU artifact."""
    inp = nc.INPUTS[ictx['input']]
    csr, record = ictx['csr'], ictx['record']
    rows, cols, prov = nc.load_input_labels(root, inp, k, csr, record)
    alias_of = None
    for other_k, h in sorted((int(a), b) for a, b in label_hashes.get(inp.name, {}).items()):
        if other_k < k and h == prov['label_sha256']:
            alias_of = nc.labeling_name(inp, other_k)
            break
    labeling = nc.labeling_name(inp, k)
    boo_meta = json.loads(nc.boo_artifact_path(root, inp.name, labeling).read_text())
    if (boo_meta['label_sha256'], boo_meta['matrix_sha256']) != (prov['label_sha256'],
                                                                 record['matrix_sha256']):
        raise ValueError(f'{inp.name} {labeling}: BOO artifact does not match labels/matrix')
    ctx = dict(ictx, k=k, labeling=labeling, label_sha256=prov['label_sha256'], alias_of=alias_of,
               structure=boo_meta['structure'], boo_digest=boo_meta['boo_digest'], boo=None)
    if alias_of:
        return ctx
    t0 = time.perf_counter()
    boo = nc.build_boo(csr, rows, cols)
    if nc.boo_digest(boo) != boo_meta['boo_digest']:
        raise ValueError(f'{inp.name} {labeling}: rebuilt BOO digest differs from the CPU artifact')
    ctx.update(boo=boo.to('cuda'), boo_rebuild_s=time.perf_counter() - t0)
    return ctx


def boo_arms(torch, core, spmm, ctx, arm_names, limit, requested):
    arms, rows = [], []
    boo, x, features = ctx['boo'], ctx['x'], ctx['features']
    for name in arm_names:
        arm = nc.parse_arm(name)
        base = dict(arm=f"{ctx['input']}@k{ctx['k']}:{name}", family=arm['kernel'], kernel=arm['kernel'],
                    impl=arm['impl'], p=arm['p'], threads=arm['threads'], fpt=arm['fpt'],
                    duplicate=arm['duplicate'], requested=name in requested,
                    scope='full_call', coords='original',
                    **{k: ctx['structure'].get(k) for k in STRUCT_KEYS})
        if arm['kernel'] == 'nnzflat':
            base.update(nc.work_model(boo, features, arm['p'], arm['threads'], arm['fpt']))
        info, reason = nc.launch_info(core, boo, features, arm, limit)
        base.update({k: (info or {}).get(k) for k in LAUNCH_KEYS}, shared_limit_bytes=limit)
        if reason:
            rows.append(dict(base, status='unsupported', reason=reason))
            continue
        kwargs = dict(threads=arm['threads'], p=arm['p'], csr_mode=0, launch_mode='auto',
                      features_per_thread=arm['fpt'], row_seg=arm['kernel'] == 'rowseg')
        if arm['kernel'] == 'nnzflat':
            kwargs['nnzflat_impl'] = arm['impl']
        prepare = (lambda arm=arm: nc.apply_arm(core, arm))
        call = (lambda kwargs=kwargs: spmm.boo_matmul(boo, x, **kwargs))
        run = (lambda prepare=prepare, call=call: (prepare(), call())[1])
        arms.append(Arm(base['arm'], base, prepare, run, event_timer(torch, prepare, call),
                        check=ctx['ref'].check, verify=arm))
    return arms, rows


def baseline_arms(torch, core, ctx, cell):
    """Original-order torch CSR and direct cuSPARSE algorithms for one input (confirmation).

    cell: {'torch_csr': bool, 'direct': [algorithm, ...]}. Memory exclusions are recorded.
    """
    arms, rows = [], []
    csr, x = ctx['csr'], ctx['x']
    out_bytes = csr.shape[0] * ctx['features'] * 4
    transient = 2 * out_bytes + (64 << 20)
    a = br._to_torch_csr(torch, csr)
    if cell.get('torch_csr'):
        call = lambda: torch.sparse.mm(a, x)
        arms.append(Arm(f"{ctx['input']}:torch_csr",
                        dict(arm=f"{ctx['input']}:torch_csr", family='torch_csr', scope='full_call',
                             coords='original'),
                        lambda: None, call, event_timer(torch, lambda: None, call), ctx['ref'].check))
    for alg in cell.get('direct', []):
        names = [(f"{ctx['input']}:direct_{alg}", 'cpp_only'), (f"{ctx['input']}:direct_{alg}_api", 'full_call')]
        base = [dict(arm=n, family='direct', direct_algorithm=alg, scope=s, coords='original')
                for n, s in names]
        if not br._headroom(torch, transient + out_bytes):
            rows += [dict(b, status='unsupported', reason='resource exclusion: GPU memory budget '
                          '(needs one call of transient headroom)') for b in base]
            continue
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            plan = core.DirectSpMM(a.crow_indices(), a.col_indices(), a.values(), x, alg)
            torch.cuda.synchronize()
            build_ms = 1000 * (time.perf_counter() - t0)
        except torch.cuda.OutOfMemoryError:
            rows += [dict(b, status='unsupported', reason='resource exclusion: GPU out of memory '
                          'at plan construction') for b in base]
            continue
        info = plan.info()
        if not info['supported']:
            rows += [dict(b, status='unsupported', reason=info['unsupported_reason']) for b in base]
            continue
        extra = dict(plan_build_ms=build_ms, workspace_bytes=info.get('workspace_bytes'),
                     preprocess_status=info.get('preprocess_status'),
                     cusparse_version=info.get('cusparse_version'))
        arms.append(Arm(names[0][0], dict(base[0], **extra), lambda: None, plan.run, plan.timed_run,
                        ctx['ref'].check))
        arms.append(Arm(names[1][0], dict(base[1], **extra), lambda: None, plan.run,
                        event_timer(torch, lambda: None, plan.run), ctx['ref'].check))
    return arms, rows


def measure(torch, arms, warmup, repeats, order_seed):
    kept, failed = [], []
    for arm in arms:
        try:
            torch.cuda.nvtx.range_push(f'validate:{arm.name}')
            y = arm.run()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            check = arm.check(y)
            del y
            if check['bound_violations']:
                failed.append(dict(arm.row, status='failed', correctness='failed', **check,
                                   reason=f"numerical: {check['bound_violations']} entries exceed "
                                          f"the float32 bound (max scaled {check['max_scaled_error']:.3g})"))
                continue
            row = dict(arm.row, correctness='passed', **check)
            if arm.verify is not None:
                status, found = nc.verify_call(torch, arm.run, arm.verify)
                row.update(launch_verified=status, launched_kernel=';'.join(found))
                if status != 'verified' and not status.startswith('UNVERIFIED'):
                    failed.append(dict(row, status='failed', reason=f'launch audit: {status}'))
                    continue
            kept.append((arm, row))
        except Exception as exc:  # recorded, never silently substituted
            torch.cuda.synchronize()
            failed.append(dict(arm.row, status='failed',
                               reason=f'{type(exc).__name__}: {exc}'.replace('\n', ' ')[:500]))
            torch.cuda.empty_cache()
    for arm, _ in kept:
        for _ in range(warmup):
            arm.run()
    torch.cuda.synchronize()
    samples, previous = [], ''
    for r in range(repeats):
        order = list(range(len(kept)))
        random.Random(order_seed * 1000003 + r).shuffle(order)
        for position, i in enumerate(order):
            arm = kept[i][0]
            t = arm.timed()
            if not math.isfinite(t) or t <= 0:
                raise RuntimeError(f'invalid elapsed time for {arm.name}: {t}')
            samples.append(dict(arm=arm.name, repeat=r + 1, position=position, previous=previous,
                                time_ms=t))
            previous = arm.name
    summaries = []
    for arm, row in kept:
        v = [s['time_ms'] for s in samples if s['arm'] == arm.name]
        med = statistics.median(v)
        summaries.append(dict(row, status='measured', median_ms=med, mean_ms=statistics.mean(v),
                              min_ms=min(v), max_ms=max(v), p25_ms=float(np.percentile(v, 25)),
                              p75_ms=float(np.percentile(v, 75)),
                              mad_ms=statistics.median(abs(t - med) for t in v),
                              repeats=repeats, warmup=warmup))
    return summaries, samples, failed


def cell_seed(base, task, index):
    return base * 7919 + task * 101 + index


def run_task(args):
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    from benchmark.bench_spmm import source_metadata
    if not torch.cuda.is_available():
        raise SystemExit('GPU required')
    manifest = json.loads(args.manifest.read_text())
    task = manifest['tasks'][args.task]
    args.out.mkdir(parents=True, exist_ok=False)
    root_src = Path(__file__).resolve().parents[3]
    props = torch.cuda.get_device_properties(0)
    host = dict(stage=manifest['stage'], allocation=args.allocation,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), node=socket.gethostname(),
                gpu_name=props.name, gpu_uuid=str(getattr(props, 'uuid', '')), task=args.task)
    meta = dict(source_metadata(root_src), **host, manifest=str(args.manifest),
                manifest_sha256=nc.sha256_file(args.manifest), task_spec=task, status='running',
                cells_done=0, omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                timing='complete call (auto launch); per-repeat independent seeded shuffle',
                started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    build = root_src / 'results' / 'build_hashes.json'
    if build.exists():
        meta['build_hashes'] = json.loads(build.read_text())
    meta_path = args.out / 'metadata.json'
    rm._write_json(meta_path, meta)
    limit = props.shared_memory_per_block_optin
    requested = set(manifest.get('requested_arm_names', []))
    try:
        with torch.inference_mode():
            for ci, cell in enumerate(task['cells']):
                order_seed = cell_seed(manifest['order_seed'] + 1000 * args.order_seed_offset, args.task, ci)
                features = cell['features']
                common = dict(host, cell=cell['cell'], features=features, order_seed=order_seed)
                print(f"cell {cell['cell']}", flush=True)
                all_arms, rows, objects, inputs = [], [], [], {}
                for obj in cell['objects']:
                    name, k = obj['input'], obj['k']
                    if name not in inputs:
                        inputs[name] = load_input_ctx(torch, args.root, args.cache, name, features)
                    ctx = prepare_object(args.root, inputs[name], k, manifest['label_hashes'])
                    objects.append(ctx)
                    info = {key: ctx[key] for key in ('input', 'matrix', 'method', 'k', 'labeling',
                                                      'matrix_sha256', 'label_sha256', 'boo_digest',
                                                      'x_sha256', 'ref_sha256')}
                    if ctx['alias_of']:
                        for arm_name in obj['arms']:
                            arm = nc.parse_arm(arm_name)
                            rows.append(dict(common, **info, arm=f'{name}@k{k}:{arm_name}',
                                             kernel=arm['kernel'], impl=arm['impl'], p=arm['p'],
                                             threads=arm['threads'], fpt=arm['fpt'],
                                             duplicate=arm['duplicate'],
                                             requested=arm_name in requested, status='alias',
                                             alias_of=ctx['alias_of'],
                                             reason='identical saved labels; not re-measured'))
                        continue
                    arms, unsupported = boo_arms(torch, core, spmm, ctx, obj['arms'], limit, requested)
                    for a in arms:
                        a.row.update(common, **info)
                    all_arms += arms
                    rows += [dict(common, **info, **r) for r in unsupported]
                for name, base in cell.get('baselines', {}).items():
                    ictx = inputs[name]
                    barms, bunsup = baseline_arms(torch, core, ictx, base)
                    info = {key: ictx[key] for key in ('input', 'matrix', 'method', 'matrix_sha256',
                                                       'x_sha256', 'ref_sha256')}
                    for a in barms:
                        a.row.update(common, **info)
                    all_arms += barms
                    rows += [dict(common, **info, **r) for r in bunsup]
                summaries, samples, failed = measure(torch, all_arms, manifest['warmup'],
                                                     manifest['repeats'], order_seed)
                nc.reset_settings(core)
                for smp in samples:
                    who = smp['arm'].split(':')[0]
                    smp.update(stage=manifest['stage'], allocation=args.allocation, task=args.task,
                               cell=cell['cell'], input=who.split('@')[0],
                               k=int(who.split('@k')[1]) if '@k' in who else '',
                               features=features, order_seed=order_seed)
                write_csv(args.out / 'summary.csv', summaries + failed + rows, SUMMARY_FIELDS)
                write_csv(args.out / 'samples.csv', samples, SAMPLE_FIELDS)
                for r in sorted(summaries, key=lambda r: r['median_ms']):
                    print(f"  {r['arm']:44s} {r['median_ms']:9.4f} ms "
                          f"[{r['p25_ms']:.4f}, {r['p75_ms']:.4f}] {r.get('launch_verified', '')}", flush=True)
                for r in failed + rows:
                    print(f"  {r['arm']:44s} {r['status'].upper()}: {r.get('reason', '')}", flush=True)
                meta.setdefault('cells', {})[cell['cell']] = [
                    dict(input=o['input'], k=o['k'], boo_rebuild_s=o.get('boo_rebuild_s'),
                         alias_of=o['alias_of']) for o in objects]
                for o in objects:
                    if o.get('boo') is not None:
                        o['boo'].to('cpu')
                del all_arms, objects, inputs
                torch.cuda.empty_cache()
                meta['cells_done'] = ci + 1
                rm._write_json(meta_path, meta)
        meta['status'] = 'complete'
    except BaseException as exc:
        meta.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc())
        raise
    finally:
        nc.reset_settings(core)
        meta['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        rm._write_json(meta_path, meta)
    print(f'BENCH_NNZFLAT_DONE {args.out}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=rm.default_cache())
    ap.add_argument('--allocation', default=os.environ.get('SLURM_JOB_ID', 'local'))
    ap.add_argument('--order-seed-offset', type=int, default=0,
                    help='added to the manifest order seed (independent shuffles per allocation)')
    run_task(ap.parse_args(argv))


if __name__ == '__main__':
    main()
