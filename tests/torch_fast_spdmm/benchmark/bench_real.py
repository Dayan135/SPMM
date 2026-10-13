"""BOO vs CSR baselines on real SuiteSparse matrices (fixed pattern, many SpMM calls).

Phases (run from tests/torch_fast_spdmm; see benchmark.real_matrices for the cache):
  --phase cluster   CPU node, no GPU: cluster every (matrix, method, k); save labels
                    keyed to the canonical matrix hash, with clustering time and host.
  --phase matmul    GPU node: load + validate saved labels (never clusters), build,
                    validate numerically, then time interleaved samples.
  --phase report    Print the matched-scope comparison from an --out directory.

One measurement cell = (matrix, F, labeling). Every cell re-measures the original
baselines, so all speedups are paired within one interleaved cell:

  scope=full_call (Python CUDA events around one complete call)
    torch_csr                 torch.sparse.mm(A, X)                       original coords
    direct_<alg>_api          DirectSpMM.run() on A (persistent Y)         original coords
    boo                       boo_matmul(launch_mode='auto')               original coords
    torch_perm_steady         torch.sparse.mm(A', X')                      reordered coords
    torch_perm_gather         X'<-X[pc] (preallocated), A'X', Y<-Y'[pr^-1] original coords
    direct_perm_<alg>_steady_api / _gather_api   same, with DirectSpMM on A'
  scope=cpp_only (C++ CUDA events, no Python/dispatcher in the interval)
    direct_<alg>, direct_perm_<alg>   cusparseSpMM only (Y, workspace, preprocess reused)
    boo_cpp                    BOOTimedSpMM: Y memset + kernels + join (legacy fork/join;
                               identical to auto launch whenever a CSR remainder exists)
  scope=permutation_only
    perm_gather_x, perm_restore_y   the two dense permutations alone

A' = A[pr][:, pc] with pr/pc the stable argsort of the SAME labels BOO uses (asserted
equal to boo.row_ids/col_ids). BOO needs no external permutation: row_ids/col_ids
apply the equivalent mapping inside the kernel, so BOO has no separate steady state.
Identity labels give identity permutations, so reordered variants are omitted there.

Preprocessing (parse, clustering, BOO build, A' build, staging, cuSPARSE plan
creation incl. bufferSize/preprocess) is recorded but never inside a timed interval.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time
from typing import Callable, Optional

import numpy as np

from benchmark import real_matrices as rm

METHODS = ('identity', 'hem')
DIRECT_PRIORITY = ('DEFAULT', 'CSR_ALG2', 'CSR_ALG1', 'CSR_ALG3')
U32 = 2.0 ** -24


# ---------------------------------------------------------------------------
# labels (CPU)
# ---------------------------------------------------------------------------

def label_path(directory, spec, explicit_zeros, method, k):
    return Path(directory) / f'{spec.group}__{spec.name}__{explicit_zeros}__{method}_k{k}.npz'


def save_labels(path, rows, cols, *, matrix, matrix_sha256, shape, explicit_zeros, method, k,
                seed, feasible_first, cluster_ms, host):
    rows = np.ascontiguousarray(rows, np.int32)
    cols = np.ascontiguousarray(cols, np.int32)
    tmp = Path(path).with_suffix('.tmp.npz')
    np.savez_compressed(tmp, rows=rows, cols=cols, label_sha256=rm.digest_arrays(rows, cols),
                        matrix=matrix, matrix_sha256=matrix_sha256,
                        shape=np.asarray(shape, np.int64), explicit_zeros=explicit_zeros,
                        method=method, k=k, seed=seed, feasible_first=feasible_first,
                        cluster_ms=cluster_ms, host=json.dumps(host))
    tmp.replace(path)


def load_labels(path, *, matrix, matrix_sha256, shape, explicit_zeros, method, k, seed,
                feasible_first=True):
    """Return (rows, cols, provenance); raise ValueError on any mismatch."""
    with np.load(path, allow_pickle=False) as f:
        meta = {key: f[key] for key in f.files}
    expect = dict(matrix=matrix, matrix_sha256=matrix_sha256, explicit_zeros=explicit_zeros,
                  method=method, k=k, seed=seed, feasible_first=feasible_first)
    for key, value in expect.items():
        stored = meta[key].item()
        if stored != value:
            raise ValueError(f'{path}: {key}={stored!r}, expected {value!r}')
    if tuple(meta['shape']) != tuple(shape):
        raise ValueError(f'{path}: shape {tuple(meta["shape"])}, expected {tuple(shape)}')
    rows, cols = meta['rows'], meta['cols']
    for axis, labels, n in (('rows', rows, shape[0]), ('cols', cols, shape[1])):
        if labels.dtype != np.int32 or labels.ndim != 1 or labels.size != n:
            raise ValueError(f'{path}: {axis} labels must be int32 of length {n}')
        if n and (labels.min() < 0 or labels.max() >= n):
            raise ValueError(f'{path}: {axis} labels out of range')
        if n and np.bincount(labels).max() > k:
            raise ValueError(f'{path}: a {axis} group exceeds k={k}')
    if rm.digest_arrays(rows, cols) != str(meta['label_sha256']):
        raise ValueError(f'{path}: label hash mismatch (file corrupted or edited)')
    host = json.loads(str(meta['host']))
    return rows, cols, dict(cluster_ms=float(meta['cluster_ms']), label_sha256=str(meta['label_sha256']),
                            cluster_node=host.get('node'), cluster_cpu=host.get('cpu_model'),
                            cluster_cpus=host.get('cpus'), labels_path=str(path))


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return 'unknown'


def cluster_phase(args):
    import torch
    from torch_fast_spdmm.clustering import cluster_axes
    if torch.cuda.is_available():
        raise RuntimeError('cluster phase must run without a GPU (set CUDA_VISIBLE_DEVICES=)')
    args.labels.mkdir(parents=True, exist_ok=True)
    host = dict(node=socket.gethostname(), cpu_model=cpu_model(),
                cpus=len(os.sched_getaffinity(0)), slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    for spec in rm.resolve(args.matrices):
        csr, record = rm.load(spec, args.cache, args.explicit_zeros)
        for method in args.methods:
            for k in args.groups:
                path = label_path(args.labels, spec, args.explicit_zeros, method, k)
                expected = dict(matrix=spec.key, matrix_sha256=record['matrix_sha256'],
                                shape=csr.shape, explicit_zeros=args.explicit_zeros,
                                method=method, k=k, seed=args.seed)
                if path.exists():
                    load_labels(path, **expected)  # stale or corrupt labels must not be kept
                    print(f'keep {path.name}', flush=True)
                    continue
                print(f'cluster {spec.key} {method}_k{k}', flush=True)
                start = time.perf_counter()
                rows, cols = cluster_axes(csr, method=method, target_nr=k, target_nc=k,
                                          max_nr=k, max_nc=k, seed=args.seed, feasible_first=True)
                cluster_ms = 1000 * (time.perf_counter() - start)
                save_labels(path, rows, cols, feasible_first=True, cluster_ms=cluster_ms,
                            host=host, **expected)
                print(f'  {cluster_ms / 1000:.1f} s', flush=True)


# ---------------------------------------------------------------------------
# permutation and numerical reference (CPU, no GPU needed)
# ---------------------------------------------------------------------------

def permutation_from_labels(labels):
    """Group order, original order within each group: exactly BOO's row_ids/col_ids."""
    return np.argsort(np.asarray(labels), kind='stable').astype(np.int64)


def reorder(csr, pr, pc):
    """A' with A'[i, j] = A[pr[i], pc[j]]; rows and columns permuted independently."""
    out = csr[pr][:, pc].tocsr()
    out.sort_indices()
    out.indptr = out.indptr.astype(np.int32)
    out.indices = out.indices.astype(np.int32)
    return out


class Reference:
    """float64 CPU reference (scipy, never the GPU code path), stored as float32.

    Per-row deterministic bound for any float32 summation order of k_i products:
    |y - exact| <= gamma_{k_i+1} (|A||X|)_i; plus float32 rounding of the stored
    reference. A dropped, duplicated or misplaced term exceeds this whenever it is
    larger than ~2 k_i u of the row's absolute sum.
    """

    def __init__(self, csr, x, chunk=64):
        a64 = csr.astype(np.float64)
        abs_a = abs(a64)
        n_rows, features = csr.shape[0], x.shape[1]
        self.ref = np.empty((n_rows, features), np.float32)
        self.absref = np.empty((n_rows, features), np.float32)
        for j in range(0, features, chunk):
            xc = np.asarray(x[:, j:j + chunk], dtype=np.float64)  # chunked: never upcast all of X
            self.ref[:, j:j + chunk] = a64 @ xc
            self.absref[:, j:j + chunk] = abs_a @ np.abs(xc)
        k = np.diff(csr.indptr).astype(np.float64)
        self.tol = (2.0 * (k + 1) + 2.0) * U32  # gamma bound + reference rounding

    def check(self, y, row_perm=None, chunk=64):
        """y: (rows, F) tensor or array. row_perm: y[i] corresponds to original row row_perm[i]."""
        max_abs = max_scaled = 0.
        violations = 0
        for j in range(0, self.ref.shape[1], chunk):
            yc = y[:, j:j + chunk]
            yc = yc.cpu().numpy() if hasattr(yc, 'cpu') else np.asarray(yc)
            ref, absref, tol = self.ref[:, j:j + chunk], self.absref[:, j:j + chunk], self.tol
            if row_perm is not None:
                ref, absref, tol = ref[row_perm], absref[row_perm], tol[row_perm]
            err = np.abs(yc.astype(np.float64) - ref)
            bound = tol[:, None] * absref
            bad = ~(err <= bound)  # NaN counts as a violation
            violations += int(bad.sum())
            max_abs = max(max_abs, float(np.nanmax(err)) if err.size else 0.)
            nz = absref > 0
            if nz.any():
                max_scaled = max(max_scaled, float(np.nanmax(err[nz] / absref[nz])))
        return dict(max_abs_error=max_abs, max_scaled_error=max_scaled, bound_violations=violations)


# ---------------------------------------------------------------------------
# GPU measurement
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    config: dict
    run: Callable          # untimed call; returns the output tensor
    timed: Callable        # returns elapsed ms for one call
    row_perm: Optional[np.ndarray] = None  # None: output in original row order
    validate: bool = True


class Unsupported(Exception):
    pass


def _event_timer(torch, fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()  # initialize event resources outside measurement
    end.record()
    end.synchronize()

    def timed():
        start.record()
        y = fn()
        end.record()
        end.synchronize()
        del y
        return start.elapsed_time(end)
    return timed


def _headroom(torch, need):
    free, _ = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return free + cached >= need


def _to_torch_csr(torch, csr):
    return torch.sparse_csr_tensor(torch.from_numpy(csr.indptr), torch.from_numpy(csr.indices),
                                   torch.from_numpy(csr.data), size=csr.shape).cuda()


def _boo_name(fpt):
    return 'boo' if fpt == 1 else f'boo_fpt{fpt}'


def build_cell(torch, core, spmm, csr, x_cpu, labeling, args, limit):
    """Stage one (matrix, F, labeling) cell. Returns (variants, unsupported rows, prep costs).

    Memory policy: essentials first (X, A, BOO, A', X' buffer); then optional
    variants in priority order, each kept only if one complete call's transient
    output memory (Y' + Y) still fits afterwards. Skipped variants are reported.
    """
    from benchmark.bench_clustering import feature_tile
    rows, cols = csr.shape
    features = x_cpu.shape[1]
    out_bytes = rows * features * 4
    transient = 2 * out_bytes + (64 << 20)
    variants, unsupported, prep = [], [], {}
    base = dict(threads=args.threads, launch_mode='auto', small_block_policy='remainder',
                small_block_threshold=args.threshold, nz_order='rowmajor', feasible_first=True,
                dtype='float32', index_dtype='int32')

    def skip(config, reason):
        unsupported.append(dict(config, supported=False, unsupported_reason=reason))

    x = x_cpu.cuda()
    a = _to_torch_csr(torch, csr)
    torch.cuda.synchronize()
    variants.append(Variant(dict(config='torch_csr', algorithm='torch_csr', scope='full_call',
                                 coords='original', includes_permutation=False),
                            lambda: torch.sparse.mm(a, x),
                            _event_timer(torch, lambda: torch.sparse.mm(a, x))))

    boo = labeling['boo']
    p, arena, fits = feature_tile(boo, features, limit)
    boo_config = dict(base, **labeling['config'], algorithm='boo', p=p, arena_bytes=arena,
                      shared_limit_bytes=limit)
    boo_cpp = None
    if not fits:
        skip(dict(boo_config, config='boo', scope='full_call', coords='original'),
             f'shared-memory arena {arena} B exceeds device limit {limit} B even at p={p}')
    else:
        start = time.perf_counter()
        boo.to('cuda')
        torch.cuda.synchronize()
        prep['boo_staging_ms'] = 1000 * (time.perf_counter() - start)
        for fpt in args.boo_fpt:
            call = (lambda fpt=fpt: spmm.boo_matmul(boo, x, threads=args.threads, p=p, row_seg=True,
                                                    csr_mode=0, launch_mode='auto',
                                                    features_per_thread=fpt))
            variants.append(Variant(dict(boo_config, config=_boo_name(fpt), features_per_thread=fpt,
                                         scope='full_call', coords='original',
                                         includes_permutation=False),
                                    call, _event_timer(torch, call)))
        meta = boo._gpu_meta
        boo_cpp = (boo.rows, boo.cols, boo.max_nr, boo.max_nc, boo.max_nnz_blk,
                   meta.row_grp_ptr, meta.row_ids, meta.col_grp_ptr, meta.col_ids,
                   meta.blk_row_grp, meta.blk_col_grp, meta.blk_nz_ptr, meta.blk_shift,
                   meta.nz_idx, meta.nz_val, meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
                   meta.tail_row, meta.tail_col, meta.tail_val, x, args.threads, p, 0, True, False)

    pr, pc = labeling['pr'], labeling['pc']
    permuted = labeling['permuted']
    if permuted:
        start = time.perf_counter()
        a_perm = _to_torch_csr(torch, labeling['csr_perm'])
        pr_inv = torch.from_numpy(np.argsort(pr)).cuda()
        pc_t = torch.from_numpy(pc).cuda()
        x_perm = x.index_select(0, pc_t)  # resident X' (steady state) and gather target
        torch.cuda.synchronize()
        prep['perm_staging_ms'] = 1000 * (time.perf_counter() - start)
        perm_base = dict(labeling['config'], algorithm='torch_csr_perm')

        def gather_torch():
            torch.index_select(x, 0, pc_t, out=x_perm)
            return torch.sparse.mm(a_perm, x_perm).index_select(0, pr_inv)
        steady = lambda: torch.sparse.mm(a_perm, x_perm)
        variants.append(Variant(dict(perm_base, config='torch_perm_steady', scope='full_call',
                                     coords='reordered', includes_permutation=False),
                                steady, _event_timer(torch, steady), row_perm=pr))
        variants.append(Variant(dict(perm_base, config='torch_perm_gather', scope='full_call',
                                     coords='original', includes_permutation=True),
                                gather_torch, _event_timer(torch, gather_torch)))
        gather_x = lambda: torch.index_select(x, 0, pc_t, out=x_perm)
        variants.append(Variant(dict(perm_base, algorithm='permutation', config='perm_gather_x',
                                     scope='permutation_only', coords='reordered',
                                     includes_permutation=True),
                                gather_x, _event_timer(torch, gather_x), validate=False))

    # Optional variants in priority order.
    optional = []
    for i, alg in enumerate(args.direct_algorithms):
        optional.append(('direct', alg, False))
        if permuted:
            optional.append(('direct', alg, True))
        if i == 0 and boo_cpp is not None:
            optional.append(('boo_cpp', None, None))
    if permuted:
        optional.append(('restore', None, None))

    for kind, alg, on_perm in optional:
        if kind == 'boo_cpp':
            config = dict(boo_config, config='boo_cpp', scope='cpp_only', coords='original',
                          includes_permutation=False, launch_mode='legacy_forkjoin')
            if not _headroom(torch, transient + out_bytes):
                skip(config, 'skipped: GPU memory budget (needs one call of transient headroom)')
                continue
            try:
                timed = core.BOOTimedSpMM(*boo_cpp)
            except torch.cuda.OutOfMemoryError:
                skip(config, 'GPU out of memory at construction')
                continue
            # One harness, process-wide FPT applied before each call (outside its C++ events).
            for fpt in args.boo_fpt:
                def run(fpt=fpt, timed=timed):
                    core.set_boo_rowseg_fpt(fpt)
                    return timed.run()

                def timed_run(fpt=fpt, timed=timed):
                    core.set_boo_rowseg_fpt(fpt)
                    return timed.timed_run()
                variants.append(Variant(dict(config, config=_boo_name(fpt) + '_cpp',
                                             features_per_thread=fpt), run, timed_run))
            continue
        if kind == 'restore':
            config = dict(labeling['config'], algorithm='permutation', config='perm_restore_y',
                          scope='permutation_only', coords='original', includes_permutation=True)
            if not _headroom(torch, transient + out_bytes):
                skip(config, 'skipped: GPU memory budget')
                continue
            y_perm = torch.randn(rows, features, device='cuda')
            restore = lambda: y_perm.index_select(0, pr_inv)
            variants.append(Variant(config, restore, _event_timer(torch, restore), validate=False))
            continue
        prefix = 'direct_perm' if on_perm else 'direct'
        dconf = dict(labeling['config'] if on_perm else {}, algorithm='cusparse_direct',
                     direct_algorithm=alg, alpha=1., beta=0., dense_layout='row_major')
        names = ([(f'{prefix}_{alg}', 'cpp_only', 'reordered' if on_perm else 'original', False),
                  (f'{prefix}_{alg}_steady_api' if on_perm else f'{prefix}_{alg}_api', 'full_call',
                   'reordered' if on_perm else 'original', False)]
                 + ([(f'{prefix}_{alg}_gather_api', 'full_call', 'original', True)] if on_perm else []))
        configs = [dict(dconf, config=n, scope=s, coords=c, includes_permutation=g) for n, s, c, g in names]
        if not _headroom(torch, transient + out_bytes):
            for config in configs:
                skip(config, 'skipped: GPU memory budget (needs one call of transient headroom)')
            continue
        target_a, target_x = (a_perm, x_perm) if on_perm else (a, x)
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            plan = core.DirectSpMM(target_a.crow_indices(), target_a.col_indices(),
                                   target_a.values(), target_x, alg)
            torch.cuda.synchronize()
            build_ms = 1000 * (time.perf_counter() - start)
        except torch.cuda.OutOfMemoryError:
            for config in configs:
                skip(config, 'GPU out of memory at plan construction')
            continue
        info = plan.info()
        if not info['supported']:
            for config in configs:
                skip(dict(config, **info), info['unsupported_reason'])
            continue
        if not _headroom(torch, transient):
            del plan
            for config in configs:
                skip(config, 'skipped: GPU memory budget after plan workspace')
            continue
        for config in configs:
            config.update(plan_build_ms=build_ms, **info)
        row_perm = pr if on_perm else None
        variants.append(Variant(configs[0], plan.run, plan.timed_run, row_perm=row_perm))
        variants.append(Variant(configs[1], plan.run, _event_timer(torch, plan.run), row_perm=row_perm))
        if on_perm:
            def gather_direct(plan=plan):
                torch.index_select(x, 0, pc_t, out=x_perm)
                return plan.run().index_select(0, pr_inv)
            variants.append(Variant(configs[2], gather_direct, _event_timer(torch, gather_direct)))
    return variants, unsupported, prep


def measure(torch, variants, reference, warmup, repeats, seed):
    from benchmark.bench_spmm import round_orders
    unsupported = []
    kept = []
    for v in variants:
        if not v.validate:
            kept.append((v, dict(max_abs_error=None, max_scaled_error=None, bound_violations=None,
                                 correctness='not_applicable')))
            continue
        try:
            y = v.run()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
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
    for repeat, order in enumerate(round_orders(len(kept), repeats, seed)):
        for position, index in enumerate(order):
            v = kept[index][0]
            ms = v.timed()
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError(f"invalid elapsed time for {v.config['config']}: {ms}")
            samples.append(dict(config=v.config['config'], repeat=repeat + 1, position=position,
                                time_ms=ms))
    summaries = []
    for v, check in kept:
        values = [s['time_ms'] for s in samples if s['config'] == v.config['config']]
        summaries.append(dict(v.config, **check, supported=True, unsupported_reason='',
                              median_ms=statistics.median(values), mean_ms=statistics.mean(values),
                              min_ms=min(values), max_ms=max(values),
                              p25_ms=float(np.percentile(values, 25)),
                              p75_ms=float(np.percentile(values, 75)),
                              repeats=repeats, warmup=warmup))
    _speedups(summaries)
    return summaries, samples, unsupported


def _speedups(summaries):
    """Paired, matched-scope ratios within one cell (baseline_ms / variant_ms)."""
    def best(predicate):
        times = [s['median_ms'] for s in summaries if predicate(s)]
        return min(times) if times else None
    torch_ms = best(lambda s: s['config'] == 'torch_csr')
    direct_api = best(lambda s: s['algorithm'] == 'cusparse_direct' and s['scope'] == 'full_call'
                      and not s['config'].startswith('direct_perm'))
    direct_cpp = best(lambda s: s['algorithm'] == 'cusparse_direct' and s['scope'] == 'cpp_only'
                      and not s['config'].startswith('direct_perm'))
    for s in summaries:
        s['speedup_vs_torch_csr'] = torch_ms / s['median_ms'] if torch_ms and s['scope'] == 'full_call' else None
        baseline = direct_api if s['scope'] == 'full_call' else direct_cpp if s['scope'] == 'cpp_only' else None
        s['speedup_vs_best_direct_same_scope'] = baseline / s['median_ms'] if baseline else None


def append_csv(path, rows, columns):
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='raise')
        if not exists:
            writer.writeheader()
        writer.writerows([{c: r.get(c) for c in columns} for r in rows])


def matmul_phase(args):
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    from benchmark.bench_spmm import source_metadata
    if not torch.cuda.is_available():
        raise RuntimeError('matmul phase requires a GPU node')
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = source_metadata(Path(__file__).resolve().parents[3])
    metadata.update(arguments={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    status='running', completed_cells=0, matrices={},
                    timing=('interleaved seeded-rotation order across all variants of one '
                            '(matrix, F, labeling) cell; scope column names the interval'),
                    validation=('float64 scipy reference on CPU; per-row bound (2(k_i+1)+2)u*(|A||X|)_i; '
                                'reordered outputs compared to reference rows pr'),
                    boo_mapping='BOO consumes original-coordinate X/Y; row_ids/col_ids apply the '
                                'label permutation inside the kernel',
                    x_perm_buffer=('torch_perm_steady and direct_perm_*_steady read the resident X\' '
                                   'that *_gather variants rewrite in place with identical content'))
    meta_path = args.out / 'metadata.json'
    rm._write_json(meta_path, metadata)
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    try:
        with torch.inference_mode():
            for spec in rm.resolve(args.matrices):
                start = time.perf_counter()
                csr, record = rm.load(spec, args.cache, args.explicit_zeros)
                record['load_ms'] = 1000 * (time.perf_counter() - start)
                metadata['matrices'][spec.key] = record
                labelings, seen = [], {}
                for method in args.methods:
                    for k in args.groups:
                        name = f'{method}_k{k}'
                        rows, cols, provenance = load_labels(
                            label_path(args.labels, spec, args.explicit_zeros, method, k),
                            matrix=spec.key, matrix_sha256=record['matrix_sha256'], shape=csr.shape,
                            explicit_zeros=args.explicit_zeros, method=method, k=k, seed=args.seed)
                        config = dict(labeling=name, cluster=method, k=k, **provenance)
                        if provenance['label_sha256'] in seen:
                            dup = dict(config, config='*', duplicate_of=seen[provenance['label_sha256']],
                                       supported=False, unsupported_reason='duplicate labels; not re-measured')
                            labelings.append(dict(config=config, duplicate=dup))
                            continue
                        seen[provenance['label_sha256']] = name
                        start = time.perf_counter()
                        boo = spmm.csr_to_boo(csr, row_labels=rows, col_labels=cols,
                                              small_block_policy='remainder',
                                              small_block_threshold=args.threshold, nz_order='rowmajor')
                        build_ms = 1000 * (time.perf_counter() - start)
                        pr, pc = permutation_from_labels(rows), permutation_from_labels(cols)
                        np.testing.assert_array_equal(np.asarray(boo.row_ids), pr)
                        np.testing.assert_array_equal(np.asarray(boo.col_ids), pc)
                        permuted = not (np.array_equal(pr, np.arange(csr.shape[0]))
                                        and np.array_equal(pc, np.arange(csr.shape[1])))
                        start = time.perf_counter()
                        csr_perm = reorder(csr, pr, pc) if permuted else None
                        reorder_ms = 1000 * (time.perf_counter() - start)
                        info = spmm.boo_info(boo, p=64, threads=args.threads, row_seg=True)
                        rsz, csz = np.bincount(rows), np.bincount(cols)
                        config.update(boo_build_ms=build_ms, perm_build_ms=reorder_ms,
                                      permutation_identity=not permuted,
                                      num_blocks=info.num_blocks, compressed_fraction=info.compressed_fraction,
                                      remainder_nnz=info.remainder_nnz, x_loads=info.x_loads,
                                      y_writes=info.y_writes, max_nnz_blk=int(boo.max_nnz_blk),
                                      num_row_groups=info.num_row_groups, num_col_groups=info.num_col_groups,
                                      row_group_mean=float(rsz.mean()), row_group_max=int(rsz.max()),
                                      col_group_mean=float(csz.mean()), col_group_max=int(csz.max()),
                                      execution_path=('boo+csr' if info.remainder_nnz else 'boo')
                                      if info.compressed_nnz else 'csr_only')
                        labelings.append(dict(config=config, boo=boo, pr=pr, pc=pc, permuted=permuted,
                                              csr_perm=csr_perm))
                for features in args.features:
                    x_cpu = torch.randn(csr.shape[1], features,
                                        generator=torch.Generator().manual_seed(args.seed))
                    start = time.perf_counter()
                    reference = Reference(csr, x_cpu.numpy())
                    reference_ms = 1000 * (time.perf_counter() - start)
                    common = dict(matrix=spec.key, rows=csr.shape[0], cols=csr.shape[1],
                                  stored_entries=csr.nnz, numerical_nnz=record['numerical_nnz'],
                                  features=features, seed=args.seed, matrix_sha256=record['matrix_sha256'],
                                  x_sha256=rm.digest_arrays(x_cpu.numpy()), explicit_zeros=args.explicit_zeros,
                                  reference_ms=reference_ms)
                    for labeling in labelings:
                        if 'duplicate' in labeling:
                            append_csv_rows(args.out, [dict(common, **labeling['duplicate'])], [])
                            continue
                        cell = f"{spec.name} F={features} {labeling['config']['labeling']}"
                        print(f'cell {cell}', flush=True)
                        variants, unsupported, prep = build_cell(torch, core, spmm, csr, x_cpu,
                                                                 labeling, args, limit)
                        summaries, samples, more = measure(torch, variants, reference,
                                                           args.warmup, args.repeats, args.seed)
                        extra = dict(common, labeling=labeling['config']['labeling'], **prep)
                        rows_out = [dict(extra, **s) for s in summaries + unsupported + more]
                        append_csv_rows(args.out, rows_out,
                                        [dict(extra, labeling=labeling['config']['labeling'], **s)
                                         for s in samples])
                        for row in rows_out:
                            if row.get('median_ms') is None:
                                print(f"  {row['config']:34s} UNSUPPORTED: {row['unsupported_reason']}",
                                      flush=True)
                            else:
                                print(f"  {row['config']:34s} {row['scope']:16s} {row['median_ms']:9.4f} ms "
                                      f"IQR [{row['p25_ms']:.4f}, {row['p75_ms']:.4f}] "
                                      f"vs direct(same scope) {_fmt(row['speedup_vs_best_direct_same_scope'])}",
                                      flush=True)
                        del variants
                        if getattr(labeling['boo'], '_gpu_meta', None) is not None:
                            labeling['boo'].to('cpu')
                        torch.cuda.empty_cache()
                        metadata['completed_cells'] += 1
                        rm._write_json(meta_path, metadata)
                    del reference, x_cpu
                del labelings, csr
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        rm._write_json(meta_path, metadata)
    print(f'BENCH_REAL_DONE {args.out}', flush=True)


def _fmt(value):
    return f'{value:.3f}x' if value else '-'


SUMMARY_FIELDS = (
    'matrix', 'rows', 'cols', 'stored_entries', 'numerical_nnz', 'features', 'labeling', 'cluster', 'k',
    'config', 'algorithm', 'direct_algorithm', 'alpha', 'beta', 'dense_layout', 'scope', 'coords', 'includes_permutation',
    'supported', 'unsupported_reason', 'duplicate_of', 'correctness', 'max_abs_error',
    'max_scaled_error', 'bound_violations', 'median_ms', 'mean_ms', 'min_ms', 'max_ms', 'p25_ms',
    'p75_ms', 'repeats', 'warmup', 'speedup_vs_torch_csr', 'speedup_vs_best_direct_same_scope',
    'p', 'threads', 'features_per_thread', 'launch_mode', 'arena_bytes', 'shared_limit_bytes', 'small_block_policy',
    'small_block_threshold', 'nz_order', 'feasible_first', 'execution_path', 'num_blocks',
    'compressed_fraction', 'remainder_nnz', 'x_loads', 'y_writes', 'max_nnz_blk', 'num_row_groups',
    'num_col_groups', 'row_group_mean', 'row_group_max', 'col_group_mean', 'col_group_max',
    'permutation_identity', 'cluster_ms', 'boo_build_ms', 'boo_staging_ms', 'perm_build_ms',
    'perm_staging_ms', 'plan_build_ms', 'workspace_bytes', 'preprocess_status', 'cusparse_version',
    'cusparse_header_version',
    'reference_ms', 'label_sha256', 'labels_path', 'cluster_node', 'cluster_cpu', 'cluster_cpus',
    'matrix_sha256', 'x_sha256', 'seed', 'explicit_zeros', 'dtype', 'index_dtype')
SAMPLE_FIELDS = ('matrix', 'features', 'labeling', 'config', 'repeat', 'position', 'time_ms',
                 'matrix_sha256', 'x_sha256')


def append_csv_rows(out, summaries, samples):
    for row in summaries:
        unknown = set(row) - set(SUMMARY_FIELDS)
        if unknown:
            raise KeyError(f'unrecorded summary fields: {sorted(unknown)}')
    if summaries:
        append_csv(out / 'summary.csv', summaries, list(SUMMARY_FIELDS))
    if samples:
        append_csv(out / 'samples.csv', samples, list(SAMPLE_FIELDS))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(out):
    """Per cell: medians at matched scopes. Ratios >1 mean faster than the baseline."""
    with open(Path(out) / 'summary.csv') as f:
        rows = list(csv.DictReader(f))
    cells = {}
    for r in rows:
        cells.setdefault((r['matrix'], int(r['features']), r['labeling']), []).append(r)

    def best(group, predicate):
        hits = [(float(r['median_ms']), r['config']) for r in group if r['median_ms'] and predicate(r)]
        return min(hits) if hits else (None, '-')

    def ms(value):
        return f'{value:.4f}' if value is not None else '-'

    print('| matrix | F | labels | torch | direct api (best) | BOO | perm steady api (best) | '
          'perm gather api (best) | direct cpp | perm cpp | BOO cpp | BOO/direct api | BOO/perm gather |')
    print('|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for (matrix, features, labeling), group in sorted(cells.items()):
        if any(r['duplicate_of'] for r in group):
            print(f"| {matrix} | {features} | {labeling} | duplicate of {group[0]['duplicate_of']} |")
            continue
        api = lambda r: r['scope'] == 'full_call'
        torch_ms = best(group, lambda r: r['config'] == 'torch_csr')[0]
        d_api = best(group, lambda r: api(r) and r['config'].startswith('direct_') and
                     not r['config'].startswith('direct_perm'))
        boo = best(group, lambda r: r['config'] == 'boo')[0]
        p_steady = best(group, lambda r: api(r) and r['coords'] == 'reordered')
        p_gather = best(group, lambda r: api(r) and r['includes_permutation'] == 'True')
        d_cpp = best(group, lambda r: r['scope'] == 'cpp_only' and r['config'].startswith('direct_')
                     and not r['config'].startswith('direct_perm'))
        p_cpp = best(group, lambda r: r['scope'] == 'cpp_only' and r['config'].startswith('direct_perm'))
        b_cpp = best(group, lambda r: r['config'] == 'boo_cpp')[0]
        ratio = lambda a, b: f'{a / b:.3f}' if a and b else '-'
        print(f'| {matrix} | {features} | {labeling} | {ms(torch_ms)} | {ms(d_api[0])} | {ms(boo)} | '
              f'{ms(p_steady[0])} | {ms(p_gather[0])} | {ms(d_cpp[0])} | {ms(p_cpp[0])} | {ms(b_cpp)} | '
              f'{ratio(d_api[0], boo)} | {ratio(p_gather[0], boo)} |')
    fpt_rows = [r for r in rows if r['config'].startswith('boo_fpt') and r['median_ms']]
    if fpt_rows:
        print('\nFeatures-per-thread variants, paired with FPT=1 in the same cell '
              '(ratio = FPT1 ms / variant ms; >1 is faster):')
        print('| matrix | F | labels | variant | ms | FPT1 ms | ratio | variant / best direct (same scope) |')
        print('|---|---:|---|---|---:|---:|---:|---:|')
        for r in sorted(fpt_rows, key=lambda r: (r['matrix'], int(r['features']), r['labeling'], r['config'])):
            group = cells[(r['matrix'], int(r['features']), r['labeling'])]
            base_name = 'boo_cpp' if r['config'].endswith('_cpp') else 'boo'
            base = best(group, lambda b: b['config'] == base_name)[0]
            ms_v = float(r['median_ms'])
            vs = r['speedup_vs_best_direct_same_scope']
            print(f"| {r['matrix']} | {r['features']} | {r['labeling']} | {r['config']} | {ms(ms_v)} | "
                  f"{ms(base)} | {base / ms_v:.3f} | {float(vs):.3f} |" if base and vs else
                  f"| {r['matrix']} | {r['features']} | {r['labeling']} | {r['config']} | {ms(ms_v)} | - | - | - |")
    bad = [r for r in rows if r['supported'] == 'False' and not r['duplicate_of']]
    if bad:
        print('\nUnsupported / skipped:')
        for r in bad:
            print(f"- {r['matrix']} F={r['features']} {r['labeling']} {r['config']}: {r['unsupported_reason']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=['cluster', 'matmul', 'report'], required=True)
    parser.add_argument('--cache', type=Path, default=rm.default_cache())
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--matrices', nargs='+', default=[s.key for s in rm.PANEL])
    parser.add_argument('--explicit-zeros', choices=rm.EXPLICIT_ZERO_POLICIES, default='keep')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--groups', nargs='+', type=int, default=[32, 64])
    parser.add_argument('--features', nargs='+', type=int, default=[32, 128, 512])
    parser.add_argument('--direct-algorithms', nargs='+', choices=DIRECT_PRIORITY,
                        default=list(DIRECT_PRIORITY), help='in memory-priority order')
    parser.add_argument('--threads', type=int, choices=[128, 256, 512], default=256)
    parser.add_argument('--threshold', type=int, default=4, help='BOO remainder threshold')
    parser.add_argument('--boo-fpt', nargs='+', type=int, choices=[1, 2, 4], default=[1],
                        help='row-segmented features per thread; 1 (the original kernel) is '
                             'always measured as the paired baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args(argv)
    if args.phase == 'report':
        if args.out is None:
            parser.error('--phase report requires --out')
        return report(args.out)
    if args.labels is None or (args.phase == 'matmul' and args.out is None):
        parser.error('--labels is required; --phase matmul also requires --out')
    if (min(args.groups + args.features) < 1 or max(args.groups) > 256 or args.repeats < 1
            or args.warmup < 0 or args.threshold < 0 or not 0 <= args.seed < 2**32):
        parser.error('positive groups (<=256) and features, repeats >=1, warmup/threshold >=0, uint32 seed')
    if 1 not in args.boo_fpt:
        args.boo_fpt = [1] + args.boo_fpt
    for values in (args.matrices, args.methods, args.groups, args.features, args.direct_algorithms,
                   args.boo_fpt):
        if len(set(values)) != len(values):
            parser.error('sweep lists must not contain duplicates')
    rm.resolve(args.matrices)
    return cluster_phase(args) if args.phase == 'cluster' else matmul_phase(args)


if __name__ == '__main__':
    main()
