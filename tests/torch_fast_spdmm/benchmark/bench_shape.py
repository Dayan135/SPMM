"""GPU timing driver for the block-shape experiment (one manifest task).

  python -m benchmark.bench_shape --artifact-root NNZFLAT_RUN --run RUN \
      --manifest RUN/manifests/screen.json --task 3 --out RUN/results/screen/a1/task_003

See experiments/2026-11-06_block_shape/README.md. A cell is one feature width F; its objects are
(input, method, k_r, k_c) rebuilt from saved labels and checked against the digest the CPU gate
recorded. Row labels come from the k_r labeling and column labels from the k_c labeling. Every arm
of a cell runs in one interleaved run with an independent seeded shuffle per repeat, using the
validation, launch audit and timing from benchmark.bench_nnzflat. Inputs, X and float64 references
are the hash-checked 2026-10-14 artifacts, so results compare directly with that screen.

Arm names, within <input>@k<kr>x<kc>:
  rs_p{P}_t{T}_f{R}         rowseg through boo_matmul
  rs_p{P}_t{T}_f{R}_dup     the same arm timed twice (measurement-noise control)
  rs_p{P}_t{T}_f{R}_pad0    rowseg through the direct binding call, no padding (code-path control)
  rs_p{P}_t{T}_f{R}_cta{N}  the same work with the arena padded down to N resident CTAs/SM
Baselines, within <input>:
  direct_<ALG>[_api], torch_csr               original-order cuSPARSE / torch (cell 'baselines')
  perm<kr>x<kc>_direct_<ALG>[_steady_api|_gather_api]
                                              cuSPARSE on A reordered by an object's labels
                                              (cell 'reordered'; see reordered_arms)

Padding raises only the max_nnz_blk passed to the launcher. The kernel lays out shared memory from
that argument, so X and the id tables move by whole 32-word units and the block's data is untouched.
Same work, same bank mapping, fewer CTAs per SM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import time
import traceback
from types import SimpleNamespace

import numpy as np

from benchmark import bench_nnzflat as bn
from benchmark import bench_real as br
from benchmark import nnzflat_common as nc
from benchmark import real_matrices as rm

SMEM_PER_SM, DRIVER_RESERVED, MARGIN = 102400, 1024, 512
PAD_UNIT = 32
SCALAR_BYTES = 4
PAD_RE = re.compile(r'^(rs_p\d+_t\d+_f\d+)_(pad0|cta(\d+))$')
SUMMARY_FIELDS = bn.SUMMARY_FIELDS + ('kr', 'kc', 'pad_nnz', 'target_cta', 'row_label_sha256',
                                      'col_label_sha256', 'path', 'includes_permutation', 'perm_build_ms',
                                      'role')


def pad_for_cta(arena, target):
    """Whole PAD_UNITs of nonzeros that bring `arena` just inside `target` CTAs/SM, or None."""
    room = SMEM_PER_SM // target - DRIVER_RESERVED - MARGIN - arena
    if room < 0:
        return None
    return room // (PAD_UNIT * (SCALAR_BYTES + 2)) * PAD_UNIT


def matmul_padded(core, boo, x, threads, p, pad):
    """boo_matmul's auto-launch rowseg call with max_nnz_blk raised by `pad`."""
    meta = boo._gpu_meta
    y = core.boo_spmm_cuda_auto_f32(
        boo.rows, boo.cols, int(boo.max_nr), int(boo.max_nc), int(boo.max_nnz_blk) + pad,
        meta.row_grp_ptr, meta.row_ids, meta.col_grp_ptr, meta.col_ids,
        meta.blk_row_grp, meta.blk_col_grp, meta.blk_nz_ptr, meta.blk_shift,
        meta.nz_idx, meta.nz_val,
        meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
        meta.tail_row, meta.tail_col, meta.tail_val,
        x, threads, p, 0, True)
    return y.squeeze(0) if x.dim() == 2 else y


def load_side(ictx, method, k, label_dirs):
    prefix = ictx['matrix'].replace('/', '__', 1)
    name = f'{prefix}__keep__{method}_k{k}.npz'
    path = next((Path(d) / name for d in label_dirs if (Path(d) / name).exists()), None)
    if path is None:
        raise FileNotFoundError(f'{name} not found in {label_dirs}')
    csr, record = ictx['csr'], ictx['record']
    return br.load_labels(path, matrix=ictx['matrix'], matrix_sha256=record['matrix_sha256'],
                          shape=csr.shape, explicit_zeros='keep', method=method, k=k, seed=nc.SEED)


def prepare_object(ictx, obj, label_dirs):
    method, kr, kc = obj['method'], obj['kr'], obj['kc']
    rows, _, prov_r = load_side(ictx, method, kr, label_dirs)
    _, cols, prov_c = load_side(ictx, method, kc, label_dirs)
    if (prov_r['label_sha256'], prov_c['label_sha256']) != (obj['row_label_sha256'], obj['col_label_sha256']):
        raise ValueError(f"{ictx['input']} {method} {kr}x{kc}: labels differ from the gate")
    t0 = time.perf_counter()
    boo = nc.build_boo(ictx['csr'], rows, cols)
    digest = nc.boo_digest(boo)
    if digest != obj['boo_digest']:
        raise ValueError(f"{ictx['input']} {method} {kr}x{kc}: BOO digest differs from the gate")
    structure = nc.boo_structure(boo)
    return dict(ictx, method=method, k=f'{kr}x{kc}', kr=kr, kc=kc, labeling=f'{method}_r{kr}_c{kc}',
                row_label_sha256=prov_r['label_sha256'], col_label_sha256=prov_c['label_sha256'],
                label_sha256=f"{prov_r['label_sha256']}:{prov_c['label_sha256']}", alias_of=None,
                structure=structure, boo_digest=digest, boo=boo.to('cuda'),
                boo_rebuild_s=time.perf_counter() - t0)


def padded_arms(torch, core, ctx, names, limit, requested):
    arms, rows = [], []
    boo, x, features = ctx['boo'], ctx['x'], ctx['features']
    for name in names:
        match = PAD_RE.match(name)
        arm = nc.parse_arm(match.group(1))
        target = int(match.group(3)) if match.group(3) else None
        base = dict(arm=f"{ctx['input']}@k{ctx['k']}:{name}", family='rowseg', kernel='rowseg',
                    impl='', p=arm['p'], threads=arm['threads'], fpt=arm['fpt'], duplicate=False,
                    requested=name in requested, scope='full_call', coords='original',
                    path='direct_binding', target_cta=target,
                    **{k: ctx['structure'].get(k) for k in bn.STRUCT_KEYS})
        arena = nc.arena_rowseg(boo, arm['p'])
        pad = 0 if target is None else pad_for_cta(arena, target)
        if pad is None:
            rows.append(dict(base, status='unsupported',
                             reason=f'arena {arena} B already above the {target} CTA/SM size'))
            continue
        base['pad_nnz'] = pad
        shim = SimpleNamespace(max_nr=boo.max_nr, max_nc=boo.max_nc, max_nnz_blk=int(boo.max_nnz_blk) + pad,
                               num_blocks=boo.num_blocks)
        info, reason = nc.launch_info(core, shim, features, arm, limit)
        base.update({k: (info or {}).get(k) for k in bn.LAUNCH_KEYS}, shared_limit_bytes=limit)
        if not reason and target is not None and info['max_active_blocks_per_sm'] != target:
            reason = f"launcher reports {info['max_active_blocks_per_sm']} CTAs/SM, target {target}"
        if reason:
            rows.append(dict(base, status='unsupported', reason=reason))
            continue
        prepare = (lambda arm=arm: nc.apply_arm(core, arm))
        call = (lambda arm=arm, pad=pad: matmul_padded(core, boo, x, arm['threads'], arm['p'], pad))
        run = (lambda prepare=prepare, call=call: (prepare(), call())[1])
        arms.append(bn.Arm(base['arm'], base, prepare, run, bn.event_timer(torch, prepare, call),
                           check=ctx['ref'].check, verify=arm))
    return arms, rows


def reordered_arms(torch, core, ictx, spec, label_dirs):
    """Direct cuSPARSE on A' = A[pr][:, pc], with pr/pc from the same labels a BOO object uses.

    spec: {'method', 'kr', 'kc', 'row_label_sha256', 'col_label_sha256', 'direct': [alg, ...]}.
    Per algorithm, one plan and three arms (bench_real's definitions):
      perm<kr>x<kc>_direct_<alg>             cusparseSpMM only on resident X', Y'   coords=reordered
      perm<kr>x<kc>_direct_<alg>_steady_api  plan.run() on resident X', Y'          coords=reordered
      perm<kr>x<kc>_direct_<alg>_gather_api  X'<-X[pc], plan.run(), Y<-Y'[pr^-1]    coords=original
    Reordered outputs are validated after mapping back to original rows (outside timing).
    """
    arms, rows = [], []
    obj = dict(spec, input=ictx['input'])
    rlab, _, prov_r = load_side(ictx, spec['method'], spec['kr'], label_dirs)
    _, clab, prov_c = load_side(ictx, spec['method'], spec['kc'], label_dirs)
    if (prov_r['label_sha256'], prov_c['label_sha256']) != (spec['row_label_sha256'], spec['col_label_sha256']):
        raise ValueError(f"{ictx['input']} perm {spec['kr']}x{spec['kc']}: labels differ from the gate")
    pr, pc = br.permutation_from_labels(rlab), br.permutation_from_labels(clab)
    t0 = time.perf_counter()
    a_perm = br._to_torch_csr(torch, br.reorder(ictx['csr'], pr, pc))
    pr_inv = torch.from_numpy(np.argsort(pr)).cuda()
    pc_t = torch.from_numpy(pc).cuda()
    x = ictx['x']
    x_perm = x.index_select(0, pc_t)
    torch.cuda.synchronize()
    reorder_ms = 1000 * (time.perf_counter() - t0)
    tag = f"perm{spec['kr']}x{spec['kc']}"
    out_bytes = ictx['csr'].shape[0] * ictx['features'] * 4
    transient = 2 * out_bytes + (64 << 20)

    def check_reordered(y):
        return ictx['ref'].check(y.reshape(y.shape[-2], y.shape[-1]).index_select(0, pr_inv))

    for alg in spec.get('direct', []):
        names = [(f'{tag}_direct_{alg}', 'cpp_only', 'reordered', False),
                 (f'{tag}_direct_{alg}_steady_api', 'full_call', 'reordered', False),
                 (f'{tag}_direct_{alg}_gather_api', 'full_call', 'original', True)]
        base = [dict(arm=f"{ictx['input']}:{n}", family='direct_perm', direct_algorithm=alg, scope=s,
                     coords=c, includes_permutation=g, labeling=f"{spec['method']}_r{spec['kr']}_c{spec['kc']}",
                     row_label_sha256=spec['row_label_sha256'], col_label_sha256=spec['col_label_sha256'],
                     perm_build_ms=reorder_ms) for n, s, c, g in names]
        if not br._headroom(torch, transient + out_bytes):
            rows += [dict(b, status='unsupported', reason='resource exclusion: GPU memory budget '
                          '(needs one call of transient headroom)') for b in base]
            continue
        try:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            plan = core.DirectSpMM(a_perm.crow_indices(), a_perm.col_indices(), a_perm.values(), x_perm, alg)
            torch.cuda.synchronize()
            build_ms = 1000 * (time.perf_counter() - t1)
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

        def gather(plan=plan):
            torch.index_select(x, 0, pc_t, out=x_perm)
            return plan.run().index_select(0, pr_inv)
        noop = lambda: None
        arms.append(bn.Arm(base[0]['arm'], dict(base[0], **extra), noop, plan.run, plan.timed_run, check_reordered))
        arms.append(bn.Arm(base[1]['arm'], dict(base[1], **extra), noop, plan.run,
                           bn.event_timer(torch, noop, plan.run), check_reordered))
        arms.append(bn.Arm(base[2]['arm'], dict(base[2], **extra), noop, gather,
                           bn.event_timer(torch, noop, gather), ictx['ref'].check))
    return arms, rows


def object_arms(torch, core, spmm, ctx, names, limit, requested):
    plain = [n for n in names if not PAD_RE.match(n)]
    padded = [n for n in names if PAD_RE.match(n)]
    arms, rows = bn.boo_arms(torch, core, spmm, ctx, plain, limit, requested)
    for a in arms:
        a.row.update(path='boo_matmul')
    more, more_rows = padded_arms(torch, core, ctx, padded, limit, requested)
    return arms + more, rows + more_rows


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
    props = torch.cuda.get_device_properties(0)
    host = dict(stage=manifest['stage'], allocation=args.allocation,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), node=socket.gethostname(),
                gpu_name=props.name, gpu_uuid=str(getattr(props, 'uuid', '')), task=args.task)
    meta = dict(source_metadata(Path(__file__).resolve().parents[3]), **host, manifest=str(args.manifest),
                manifest_sha256=nc.sha256_file(args.manifest), task_spec=task, status='running',
                cells_done=0, omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                timing='complete call (auto launch); per-repeat independent seeded shuffle',
                started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    meta_path = args.out / 'metadata.json'
    rm._write_json(meta_path, meta)
    limit = props.shared_memory_per_block_optin
    requested = set(manifest.get('requested_arm_names', []))
    label_dirs = [args.source_labels, args.run / 'labels']
    keys = ('input', 'matrix', 'method', 'k', 'kr', 'kc', 'labeling', 'matrix_sha256', 'label_sha256',
            'row_label_sha256', 'col_label_sha256', 'boo_digest', 'x_sha256', 'ref_sha256')
    try:
        with torch.inference_mode():
            for ci, cell in enumerate(task['cells']):
                seed = bn.cell_seed(manifest['order_seed'] + 1000 * args.order_seed_offset, args.task, ci)
                features = cell['features']
                common = dict(host, cell=cell['cell'], features=features, order_seed=seed, role=cell.get('role', ''))
                print(f"cell {cell['cell']}", flush=True)
                all_arms, rows, objects, inputs = [], [], [], {}
                for obj in cell['objects']:
                    name = obj['input']
                    if name not in inputs:
                        inputs[name] = bn.load_input_ctx(torch, args.artifact_root, args.cache, name, features)
                    ctx = prepare_object(inputs[name], obj, label_dirs)
                    objects.append(ctx)
                    info = {key: ctx[key] for key in keys}
                    arms, unsupported = object_arms(torch, core, spmm, ctx, obj['arms'], limit, requested)
                    for a in arms:
                        a.row.update(common, **info)
                    all_arms += arms
                    rows += [dict(common, **info, **r) for r in unsupported]
                for name, base in cell.get('baselines', {}).items():
                    ictx = inputs[name]
                    barms, bunsup = bn.baseline_arms(torch, core, ictx, base)
                    info = {key: ictx[key] for key in ('input', 'matrix', 'matrix_sha256', 'x_sha256', 'ref_sha256')}
                    for a in barms:
                        a.row.update(common, **info)
                    all_arms += barms
                    rows += [dict(common, **info, **r) for r in bunsup]
                for spec in cell.get('reordered', []):
                    ictx = inputs[spec['input']]
                    rarms, runsup = reordered_arms(torch, core, ictx, spec, label_dirs)
                    info = {key: ictx[key] for key in ('input', 'matrix', 'matrix_sha256', 'x_sha256', 'ref_sha256')}
                    for a in rarms:
                        a.row.update(common, **info)
                    all_arms += rarms
                    rows += [dict(common, **info, **r) for r in runsup]
                summaries, samples, failed = bn.measure(torch, all_arms, manifest['warmup'], manifest['repeats'], seed)
                nc.reset_settings(core)
                for smp in samples:
                    who = smp['arm'].split(':')[0]
                    smp.update(stage=manifest['stage'], allocation=args.allocation, task=args.task,
                               cell=cell['cell'], input=who.split('@')[0],
                               k=who.split('@k')[1] if '@k' in who else '', features=features, order_seed=seed)
                bn.write_csv(args.out / 'summary.csv', summaries + failed + rows, SUMMARY_FIELDS)
                bn.write_csv(args.out / 'samples.csv', samples, bn.SAMPLE_FIELDS)
                for r in sorted(summaries, key=lambda r: r['median_ms']):
                    print(f"  {r['arm']:52s} {r['median_ms']:9.4f} ms [{r['p25_ms']:.4f}, {r['p75_ms']:.4f}] "
                          f"{r.get('launch_verified', '')}", flush=True)
                for r in failed + rows:
                    print(f"  {r['arm']:52s} {r['status'].upper()}: {r.get('reason', '')}", flush=True)
                meta.setdefault('cells', {})[cell['cell']] = [
                    dict(input=o['input'], k=o['k'], method=o['method'], boo_rebuild_s=o['boo_rebuild_s'])
                    for o in objects]
                for o in objects:
                    o['boo'].to('cpu')
                del all_arms, objects, inputs
                torch.cuda.empty_cache()
                meta['cells_done'] = ci + 1
                rm._write_json(meta_path, meta)
        meta['status'] = 'complete'
    except BaseException as exc:
        meta.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        nc.reset_settings(core)
        meta['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        rm._write_json(meta_path, meta)
    print(f'BENCH_SHAPE_DONE {args.out}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifact-root', type=Path, required=True, help='2026-10-14 run: inputs and references')
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--source-labels', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=rm.default_cache())
    ap.add_argument('--allocation', default=os.environ.get('SLURM_JOB_ID', 'local'))
    ap.add_argument('--order-seed-offset', type=int, default=0)
    run_task(ap.parse_args(argv))


if __name__ == '__main__':
    main()
