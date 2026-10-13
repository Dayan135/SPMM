"""Paired PBR / BOO / PyTorch CSR and optional direct cuSPARSE benchmark.

Run from tests/torch_fast_spdmm on a GPU node after rebuilding the extension:
    python -m benchmark.bench_spmm --out results/spmm

Each configuration receives five warmup calls and twenty individually timed
calls by default. All variants use identical A and X in one process. Conversion,
staging, validation and logging are outside the timed interval. Times are CUDA
event elapsed milliseconds for a complete matmul call, including output
initialization and stream joins. With --direct-cusparse, direct plans instead
reuse output and workspace and time only cusparseSpMM using C++ CUDA events.
With --symmetric (implies --direct-cusparse), every method also gets a
matching C++-only timing (PBRTimedSpMM/BOOTimedSpMM for PBR/BOO, obj.run()
routed through the normal Python loop for direct) so cuSPARSE and PBR/BOO
are compared at the same measurement scope in both directions, plus a
diagnostic upper-bound variant (Y memset removed, atomicAdd -> plain store;
output incorrect by construction, not a real timing) for PBR/BOO.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import time

import numpy as np
import torch
import torch_fast_spdmm as spmm
import torch_fast_spdmm_core as core
import wandb

from benchmark.bench_4way import DEFAULT_PROFILES
from benchmark.profile_pbr import _MATRIX_PROFILES, build_matrix


PBR_CONFIGS = ['2:2', '2:3', '4:2', '4:4', '4:8', '8:2', '8:8', '8:12']
DIRECT_ALGORITHMS = ['DEFAULT', 'CSR_ALG1', 'CSR_ALG2', 'CSR_ALG3']


def pbr_config(value):
    try:
        block, minimum = map(int, value.split(':'))
        if block not in (2, 4, 8) or not 1 <= minimum <= block * block:
            raise ValueError
        return block, minimum
    except ValueError:
        raise argparse.ArgumentTypeError('Expected block:min_nnz, block in {2,4,8}, '
                                         '1 <= min_nnz <= block²') from None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes', type=int, nargs='+', default=[16384, 65536, 262144])
    p.add_argument('--features', type=int, nargs='+', default=[128, 512])
    p.add_argument('--profiles', nargs='+', choices=list(_MATRIX_PROFILES), default=DEFAULT_PROFILES)
    p.add_argument('--pbr-configs', type=pbr_config, nargs='+',
                   default=[pbr_config(s) for s in PBR_CONFIGS])
    p.add_argument('--pbr-smem', type=int, nargs='+', choices=[512, 1024, 2048], default=[2048])
    p.add_argument('--boo-clusters', nargs='+', choices=['hem', 'identity'], default=['hem'])
    p.add_argument('--boo-k', type=int, nargs='+', default=[8, 16, 32, 64],
                   help='Target rows and columns per group (not number of clusters)')
    p.add_argument('--boo-threads', type=int, choices=[128, 256, 512], default=256)
    p.add_argument('--boo-p', type=int, choices=[8, 16, 32, 64, 128], default=64)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--direct-cusparse', action='store_true',
                   help='Add four persistent cusparseSpMM plans; time only SpMM in C++ CUDA events')
    p.add_argument('--symmetric', action='store_true',
                   help='Implies --direct-cusparse. Add an API-level (Python full-call) timing '
                       'for direct cuSPARSE, and a C++-only timing (PBRTimedSpMM/BOOTimedSpMM, '
                       'matching DirectSpMM.timed_run()\'s scope) plus a diagnostic upper-bound '
                       '(Y memset removed, Step-C atomicAdd -> plain store; output incorrect by '
                       'construction) for every PBR/BOO config already in the sweep.')
    p.add_argument('--out', type=Path, default=Path('results/spmm'))
    p.add_argument('--project', default='torch-fast-spdmm')
    p.add_argument('--entity', default=None)
    p.add_argument('--name', default=None)
    p.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    a = p.parse_args(argv)
    if min(a.sizes + a.features + a.boo_k) <= 0 or a.repeats < 1 or a.warmup < 0:
        p.error('Sizes, features, k and repeats must be positive; warmup must be nonnegative')
    if not 0 <= a.seed < 2**32 or max(a.boo_k) > 256:
        p.error('seed must fit uint32; BOO k must be <= 256 for packed coordinates')
    for values in (a.sizes, a.features, a.profiles, a.pbr_configs, a.pbr_smem,
                   a.boo_clusters, a.boo_k):
        if len(set(values)) != len(values):
            p.error('Sweep lists must not contain duplicates')
    return a


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def source_metadata(root):
    """Fingerprint current source, including uncommitted and new source files."""
    h = hashlib.sha256()
    for directory in ('torch_fast_spdmm', 'tests/torch_fast_spdmm'):
        for path in sorted((root / directory).rglob('*')):
            if (not path.is_file() or path.suffix not in ('.py', '.cu', '.cpp', '.hpp', '.h')
                    or any(part in ('build', '__pycache__', 'wandb') for part in path.parts)):
                continue
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    snapshot = root / 'benchmark_source.json'
    if snapshot.exists():
        origin = json.loads(snapshot.read_text())
    else:
        def git(*args):
            result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else 'unknown'
        origin = {'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short')}
    binary = Path(core.__file__).resolve()
    # CPU-only nodes (offline clustering phase) record no GPU fields.
    gpu = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {'gpu': props.name, 'gpu_memory_bytes': props.total_memory,
               'gpu_capability': list(torch.cuda.get_device_capability(0))}
    return {
        **origin, 'source_sha256': h.hexdigest(),
        'extension_path': str(binary), 'extension_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'python_package_path': str(Path(spmm.__file__).resolve()),
        'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
        **gpu,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'node': socket.gethostname(),
        'timing': 'CUDA event elapsed ms per complete matmul call; resident inputs',
        'baseline': 'torch.sparse.mm with CUDA CSR, float32 values, int32 indices',
    }


def configurations(a):
    result = [dict(algorithm='cusparse_csr', config='cusparse_csr')]
    for block, minimum in a.pbr_configs:
        for smem in a.pbr_smem:
            result.append(dict(algorithm='pbr', config=f'pbr_b{block}_min{minimum}_sm{smem}',
                               block_size=block, min_nnz=minimum, smem_size=smem, kernel='block'))
    for cluster in a.boo_clusters:
        for k in a.boo_k:
            result.append(dict(algorithm='boo', config=f'boo_{cluster}_k{k}', cluster=cluster,
                               k=k, threads=a.boo_threads, p=a.boo_p, kernel='rowseg',
                               small_block_policy='remainder', small_block_threshold=4,
                               hem_feasible_first=False, launch_mode='legacy'))
    return result


def build_variants(mat, configs, seed):
    variants = []
    cache = {}
    for config in configs:
        print(f"  prepare {config['config']}", flush=True)
        nr, nc = config.get('target_nr', config.get('k')), config.get('target_nc', config.get('k'))
        threshold = config.get('small_block_threshold', 4)
        key = (config['algorithm'], config.get('block_size'), config.get('min_nnz'),
               config.get('cluster'), nr, nc, threshold, config.get('hem_feasible_first', False))
        if key in cache:
            stats, obj = cache[key]
            variants.append((dict(config, **stats), obj))
            continue
        start = time.perf_counter()
        if config['algorithm'] == 'cusparse_csr':
            obj = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr.copy()),
                                          torch.from_numpy(mat.indices.copy()),
                                          torch.from_numpy(mat.data.copy()), size=mat.shape)
            stats = dict(compressed_fraction=None, remainder_nnz=None, num_blocks=None)
        elif config['algorithm'] == 'pbr':
            obj = spmm.csr_to_pbr(mat, block_rows=config['block_size'],
                                  block_cols=config['block_size'], min_nnz_per_block=config['min_nnz'])
            stats = dict(compressed_fraction=obj.compressed_nnz() / max(mat.nnz, 1),
                         remainder_nnz=int(obj.remainder_nnz()), num_blocks=int(obj.accounted_blocks()))
        else:
            obj = spmm.csr_to_boo(mat, cluster=config['cluster'], target_nr=nr,
                                  target_nc=nc, seed=seed, small_block_policy='remainder',
                                  small_block_threshold=threshold, nz_order='rowmajor',
                                  hem_feasible_first=config.get('hem_feasible_first', False))
            stats = dict(compressed_fraction=obj.compressed_nnz() / max(mat.nnz, 1),
                         remainder_nnz=int(obj.remainder_nnz()), num_blocks=len(obj.blk_row_grp),
                         max_nr=int(obj.max_nr), max_nc=int(obj.max_nc),
                         max_nnz_blk=int(obj.max_nnz_blk))
        build_ms = (time.perf_counter() - start) * 1000
        torch.cuda.synchronize()
        start = time.perf_counter()
        obj = obj.to('cuda')
        torch.cuda.synchronize()
        staging_ms = (time.perf_counter() - start) * 1000
        stats.update(build_ms=build_ms, staging_ms=staging_ms)
        cache[key] = stats, obj
        variants.append((dict(config, **stats), obj))
    planned = []
    for config, obj in variants:
        if config.get('execution_plan', False):
            plan = spmm.prepare_boo_cuda(obj, bucketed=config['bucketed'],
                                         precomputed_rows=config['precomputed_rows'], p=config['p'])
            config.update(plan_build_ms=plan.build_ms, plan_staging_ms=plan.staging_ms,
                          plan_extra_device_bytes=plan.extra_device_bytes,
                          plan_num_buckets=len(plan.buckets) // 5,
                          plan_buckets=json.dumps(plan.buckets))
            obj = (obj, plan)
        planned.append((config, obj))
    return planned


def matmul(config, obj, x):
    # cusparseSpMM_only (DirectSpMM) and cpp_only (PBRTimedSpMM/BOOTimedSpMM)
    # both expose .run() for the untimed/correctness-check call; timed_run()
    # is called directly from the sampling loop, never through here.
    if config.get('timing_scope') in ('cusparseSpMM_only', 'cpp_only'):
        return obj.run()
    if config['algorithm'] == 'cusparse_direct':
        return obj.run()
    if config['algorithm'] == 'cusparse_csr':
        return torch.sparse.mm(obj, x)
    if config['algorithm'] == 'pbr':
        return spmm.pbr_matmul(obj, x, kernel='block', smem_size=config['smem_size'], csr_mode=0)
    plan = None
    if config.get('execution_plan', False):
        obj, plan = obj
    fpt_args = ({'features_per_thread': config['features_per_thread']}
                if 'features_per_thread' in config else {})
    return spmm.boo_matmul(obj, x, threads=config['threads'], p=config['p'], row_seg=True, csr_mode=0,
                           launch_mode=config.get('launch_mode', 'legacy'), execution_plan=plan,
                           **fpt_args)


def round_orders(count, repeats, seed):
    """Rotate a seeded order so each variant moves through timing positions."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return [order[r % count:] + order[:r % count] for r in range(repeats)]


class _BOOFPTTimed:
    """Apply FPT before every C++ entry, including interleaved timing rounds.

    BOOTimedSpMM does not own FPT; the experimental launch selector is global.
    Setting it only at plan creation would silently time the wrong variant.
    """
    def __init__(self, plan, fpt):
        self.plan, self.fpt = plan, fpt

    def run(self):
        core.set_boo_rowseg_fpt(self.fpt)
        return self.plan.run()

    def timed_run(self):
        core.set_boo_rowseg_fpt(self.fpt)
        return self.plan.timed_run()


def _symmetric_variants(variants, x, diagnostics=True):
    """For every PBR/BOO config already in the sweep, build a C++-only timed
    entry (PBRTimedSpMM/BOOTimedSpMM, same scope as DirectSpMM.timed_run())
    and a diagnostic upper-bound entry (Y memset removed, Step-C atomicAdd ->
    plain store) wrapping the SAME already-staged GPU object -- no extra
    format construction. Skips execution-plan (bucketed) BOO configs: the
    timed harness only supports the plain three-way fork/join. Set diagnostics
    False for panels that should only execute correctness-checkable variants."""
    extra = []
    for config, obj in variants:
        if config['algorithm'] == 'pbr':
            meta = obj._gpu_meta
            args = (obj.cols, obj.rows, obj.block_rows, obj.block_cols,
                    meta.codes, meta.coords, meta.offsets, meta.data,
                    meta.rem_indptr, meta.rem_col_ind, meta.rem_vals, x,
                    config.get('csr_mode', 0), config['smem_size'])
            timed = core.PBRTimedSpMM(*args, False)
            extra.append((dict(config, config=config['config'] + '_cpp',
                               timing_scope='cpp_only'), timed))
            if diagnostics:
                diag = core.PBRTimedSpMM(*args, True)
                extra.append((dict(config, algorithm='pbr_diagnostic', config=config['config'] + '_diag',
                                   timing_scope='diagnostic_upper_bound', execution_path='diagnostic'), diag))
        elif config['algorithm'] == 'boo':
            if config.get('execution_plan', False):
                continue
            meta = obj._gpu_meta
            args = (obj.rows, obj.cols, obj.max_nr, obj.max_nc, obj.max_nnz_blk,
                    meta.row_grp_ptr, meta.row_ids, meta.col_grp_ptr, meta.col_ids,
                    meta.blk_row_grp, meta.blk_col_grp, meta.blk_nz_ptr, meta.blk_shift,
                    meta.nz_idx, meta.nz_val, meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
                    meta.tail_row, meta.tail_col, meta.tail_val, x,
                    config['threads'], config['p'], config.get('csr_mode', 0), True)
            timed = core.BOOTimedSpMM(*args, False)
            if 'features_per_thread' in config:
                timed = _BOOFPTTimed(timed, config['features_per_thread'])
            extra.append((dict(config, config=config['config'] + '_cpp',
                               timing_scope='cpp_only'), timed))
            if diagnostics:
                diag = core.BOOTimedSpMM(*args, True)
                extra.append((dict(config, algorithm='boo_diagnostic', features_per_thread=1,
                                   config=config['config'] + '_diag',
                                   timing_scope='diagnostic_upper_bound', execution_path='diagnostic'), diag))
    return extra


def measure_cell(variants, mat, x_cpu, warmup, repeats, seed, direct_algorithms=(), symmetric=False,
                 diagnostics=True):
    x = x_cpu.to('cuda')
    variants = [(dict(config), obj) for config, obj in variants]
    for config, obj in variants:
        config['timing_scope'] = 'full_call'
        if config['algorithm'] in ('pbr', 'boo'):
            family = 'block' if config['algorithm'] == 'pbr' else 'boo'
            config['execution_path'] = ('csr_only' if config['compressed_fraction'] == 0 else
                                         f'{family}+csr' if config['remainder_nnz'] else family)
        else:
            config['execution_path'] = 'torch.sparse.mm'
    if symmetric:
        variants.extend(_symmetric_variants(variants, x, diagnostics=diagnostics))
    if direct_algorithms:
        csr = next(obj for config, obj in variants if config['algorithm'] == 'cusparse_csr')
        for algorithm in direct_algorithms:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            plan = core.DirectSpMM(csr.crow_indices(), csr.col_indices(), csr.values(), x, algorithm)
            torch.cuda.synchronize()
            base = dict(algorithm='cusparse_direct', direct_algorithm=algorithm,
                        execution_path='cusparseSpMM', alpha=1., beta=0., dense_layout='row_major',
                        compressed_fraction=None, remainder_nnz=None, num_blocks=None,
                        plan_build_ms=1000*(time.perf_counter()-t0), **plan.info())
            variants.append((dict(base, config=f'direct_{algorithm}', timing_scope='cusparseSpMM_only'), plan))
            if symmetric:
                variants.append((dict(base, config=f'direct_{algorithm}_api', timing_scope='full_call'), plan))
    # An independent reference from the original sparse matrix; never densify A.
    reference = mat.astype(np.float64) @ x_cpu.numpy().astype(np.float64)
    validation = {}
    for config, obj in variants:
        if not config.get('supported', True) or config.get('timing_scope') == 'diagnostic_upper_bound':
            # Diagnostic variants' output is incorrect by construction (no Y
            # zero, atomicAdd -> plain store) -- never checked against the
            # reference, and .run() throws for them by design.
            validation[config['config']] = None
            continue
        try:
            y = matmul(config, obj, x)
        except RuntimeError as exc:
            if config['algorithm'] != 'cusparse_direct' or 'CUSPARSE_STATUS_NOT_SUPPORTED' not in str(exc):
                raise
            config.update(supported=False, unsupported_reason=str(exc))
            validation[config['config']] = None
            continue
        actual = y.cpu().numpy().astype(np.float64)
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4,
                                   err_msg=config['config'])
        validation[config['config']] = float(np.max(np.abs(actual - reference)))
        del y, actual
    del reference
    # No conversions, checks, host transfers or logging during warmup/timing.
    for config, obj in variants:
        if not config.get('supported', True):
            continue
        diagnostic = config.get('timing_scope') == 'diagnostic_upper_bound'
        for _ in range(warmup):
            obj.timed_run() if diagnostic else matmul(config, obj, x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Initialize event resources before collecting samples.
    start.record()
    end.record()
    end.synchronize()
    samples = []
    for repeat, order in enumerate(round_orders(len(variants), repeats, seed)):
        for position, index in enumerate(order):
            config, obj = variants[index]
            if not config.get('supported', True):
                continue
            if config.get('timing_scope') in ('cusparseSpMM_only', 'cpp_only', 'diagnostic_upper_bound'):
                ms = obj.timed_run()
            else:
                start.record()
                y = matmul(config, obj, x)
                end.record()
                end.synchronize()
                ms = start.elapsed_time(end)
                del y
            if not np.isfinite(ms) or ms <= 0:
                raise RuntimeError(f"Invalid elapsed time for {config['config']}: {ms}")
            samples.append(dict(config=config['config'], algorithm=config['algorithm'],
                                repeat=repeat + 1, position=position, time_ms=ms))
    summaries = []
    baseline = statistics.median(s['time_ms'] for s in samples if s['algorithm'] == 'cusparse_csr')
    # In --symmetric mode this mixes cusparseSpMM_only and full_call direct
    # entries into one convenience baseline (min across both scopes) --
    # loose but harmless, since report_symmetric.py computes its two-level
    # comparison directly from each row's own timing_scope rather than
    # trusting this field.
    direct_medians = [statistics.median([s['time_ms'] for s in samples if s['config'] == c['config']])
                      for c, _ in variants if c['algorithm'] == 'cusparse_direct' and c.get('supported', True)]
    direct_baseline = min(direct_medians) if direct_medians else None
    for config, _ in variants:
        values = [s['time_ms'] for s in samples if s['config'] == config['config']]
        if not values:
            summaries.append(dict(config, time_ms=None, median_ms=None, mean_ms=None, min_ms=None,
                                  max_ms=None, std_ms=None, p25_ms=None, p75_ms=None, repeats=0,
                                  warmup=0, max_abs_error=None, correctness='not_supported',
                                  speedup_vs_cusparse=None, speedup_vs_direct=None))
            continue
        median = statistics.median(values)
        summaries.append(dict(config, time_ms=median, median_ms=median,
                              mean_ms=statistics.mean(values), min_ms=min(values), max_ms=max(values),
                              std_ms=statistics.stdev(values) if len(values) > 1 else 0.0,
                              p25_ms=float(np.percentile(values, 25)), p75_ms=float(np.percentile(values, 75)),
                              repeats=repeats, warmup=warmup, max_abs_error=validation[config['config']],
                              correctness='passed', speedup_vs_cusparse=baseline / median,
                              speedup_vs_direct=direct_baseline / median if direct_baseline else None))
    return summaries, samples


def append_csv(path, rows, columns):
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def table(rows):
    columns = sorted({k for r in rows for k in r})
    return wandb.Table(columns=columns, data=[[r.get(c) for c in columns] for r in rows])


def main(argv=None, configuration_factory=configurations):
    a = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; build and run on the remote GPU machine')
    a.out.mkdir(parents=True, exist_ok=True)
    if any((a.out / f).exists() for f in ('metadata.json', 'summary.csv', 'samples.csv')):
        raise SystemExit(f'Results already exist in {a.out}; choose a new --out directory')
    metadata = source_metadata(Path(__file__).resolve().parents[3])
    configs = configuration_factory(a)
    metadata['benchmark_configurations'] = configs
    metadata['direct_algorithms'] = DIRECT_ALGORITHMS if (a.direct_cusparse or a.symmetric) else []
    metadata['direct_timing'] = ('C++ CUDA events bracket only cusparseSpMM; alpha=1 beta=0, row-major X/Y; '
                                'per-algorithm descriptors, output/workspace reused; bufferSize and preprocess once outside timing')
    metadata['symmetric'] = a.symmetric
    if a.symmetric:
        metadata['symmetric_timing'] = (
            'Every PBR/BOO config gets three timing_scope rows: full_call (Python-side event '
            'wrap around pbr_matmul/boo_matmul, includes Python/pybind overhead -- the original '
            'scope), cpp_only (PBRTimedSpMM/BOOTimedSpMM, C++ events around only the Y memset + '
            'kernel(s) + join, same scope as direct cuSPARSE\'s cusparseSpMM_only), and '
            'diagnostic_upper_bound (cpp_only but no Y memset, block kernel only, Step-C '
            'atomicAdd -> plain store -- output is INCORRECT by construction, never validated, '
            'not a real timing, upper-bound diagnostic only). Direct cuSPARSE gets both '
            'cusparseSpMM_only (existing) and full_call (routed through the same Python-side '
            'event wrap as everyone else, i.e. obj.run() instead of obj.timed_run()) rows.')
    args_dict = dict(vars(a), out=str(a.out))
    run = wandb.init(project=a.project, entity=a.entity, name=a.name, mode=a.wandb_mode,
                     config=dict(args_dict, **metadata), dir=str(a.out))
    metadata.update(arguments=args_dict, wandb_run_id=run.id, wandb_url=run.url,
                    status='running', completed_cells=0)
    (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    summary_columns = None
    all_summaries = []
    failed = True
    try:
        with torch.inference_mode():
            for size in a.sizes:
                for profile in a.profiles:
                    print(f'Preparing {profile}, requested N={size}', flush=True)
                    mat = build_matrix(size, profile, seed=a.seed).astype(np.float32).tocsr()
                    mat.sum_duplicates()
                    mat.sort_indices()
                    if max(*mat.shape, mat.nnz) > np.iinfo(np.int32).max:
                        raise ValueError('Benchmark requires int32-compatible CSR')
                    mat.indptr = mat.indptr.astype(np.int32)
                    mat.indices = mat.indices.astype(np.int32)
                    matrix_hash = digest_arrays(np.asarray(mat.shape), mat.indptr, mat.indices, mat.data)
                    variants = build_variants(mat, configs, a.seed)
                    for features in a.features:
                        generator = torch.Generator().manual_seed(a.seed)
                        x_cpu = torch.randn(mat.shape[1], features, generator=generator, dtype=torch.float32)
                        common = dict(profile=profile, requested_size=size, rows=mat.shape[0], cols=mat.shape[1],
                                      features=features, nnz=int(mat.nnz), dtype='float32', index_dtype='int32',
                                      seed=a.seed, matrix_sha256=matrix_hash, x_sha256=digest_arrays(x_cpu.numpy()),
                                      source_sha256=metadata['source_sha256'], wandb_run_id=run.id)
                        summaries, samples = measure_cell(
                            variants, mat, x_cpu, a.warmup, a.repeats, a.seed,
                            DIRECT_ALGORITHMS if (a.direct_cusparse or a.symmetric) else (),
                            symmetric=a.symmetric)
                        summaries = [dict(common, **s) for s in summaries]
                        samples = [dict(common, **s) for s in samples]
                        if summary_columns is None:
                            summary_columns = sorted({k for r in summaries for k in r})
                        append_csv(a.out / 'summary.csv', summaries, summary_columns)
                        append_csv(a.out / 'samples.csv', samples, list(samples[0]))
                        all_summaries.extend(summaries)
                        cell = f'{profile}_{size}_f{features}'
                        # Small per-cell tables avoid W&B table row truncation.
                        run.log({f'summary/{cell}': table(summaries), f'samples/{cell}': table(samples)})
                        for row in summaries:
                            if row['median_ms'] is None:
                                print(f"{cell} {row['config']}: {row.get('unsupported_reason')}", flush=True)
                                continue
                            print(f"{cell:42s} {row['config']:25s} {row['median_ms']:10.5f} ms "
                                  f"IQR [{row['p25_ms']:.5f}, {row['p75_ms']:.5f}] "
                                  f"speedup {row['speedup_vs_cusparse']:.3f}x", flush=True)
                        metadata['completed_cells'] += 1
                        (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
                        del x_cpu
                    del variants, mat
                    torch.cuda.empty_cache()
        failed = False
    except BaseException as exc:
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        metadata['status'] = 'failed' if failed else 'complete'
        (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        try:
            # Bound every table; full CSVs are always uploaded as artifacts.
            for page in range(0, len(all_summaries), 5000):
                run.log({f'benchmarks_{page // 5000}': table(all_summaries[page:page + 5000])})
            artifact = wandb.Artifact(f'spmm-{run.id}', type='benchmark-results')
            for name in ('metadata.json', 'summary.csv', 'samples.csv'):
                if (a.out / name).exists():
                    artifact.add_file(str(a.out / name), name=name)
            run.log_artifact(artifact)
            run.summary['status'] = metadata['status']
            run.summary['completed_cells'] = metadata['completed_cells']
        finally:
            run.finish(exit_code=1 if failed else 0)
    print(f"SPMM_BENCHMARK_DONE: {a.out} W&B: {metadata['wandb_url']}", flush=True)


if __name__ == '__main__':
    main()
