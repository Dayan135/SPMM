"""C++-scope-only benchmark across the 37 density-band matrix configurations
(density_profiles.py): direct cuSPARSE (DEFAULT/CSR_ALG1-3) vs PBR (8-config
grid) vs BOO (hem k in {8,16,32,64}), all timed with C++-only CUDA events
(DirectSpMM.timed_run() / PBRTimedSpMM.timed_run() / BOOTimedSpMM.timed_run()
-- no Python-side full-call timing, no diagnostic upper bound. Does not
import or modify bench_spmm.py's measure_cell (which always adds full_call +
diagnostic entries in --symmetric mode) -- reuses only its variant-
construction helpers (build_variants, _symmetric_variants) and runs its own
minimal cpp_only-only warmup/correctness/timing loop.

Run from tests/torch_fast_spdmm:
    python -m benchmark.bench_density_bands --out results/density-bands
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
import torch_fast_spdmm_core as core
import wandb

from benchmark.bench_spmm import (
    PBR_CONFIGS, DIRECT_ALGORITHMS, pbr_config, digest_arrays, source_metadata,
    configurations, build_variants, _symmetric_variants, append_csv, table,
)
from benchmark.density_profiles import DENSITY_PROFILES, BANDS, BAND_TARGET, build_density_matrix


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes', type=int, nargs='+', default=[262144, 1048576])
    p.add_argument('--features', type=int, nargs='+', default=[128, 512])
    p.add_argument('--profiles', nargs='+', choices=list(DENSITY_PROFILES),
                   default=list(DENSITY_PROFILES))
    p.add_argument('--pbr-configs', type=pbr_config, nargs='+',
                   default=[pbr_config(s) for s in PBR_CONFIGS])
    p.add_argument('--pbr-smem', type=int, nargs='+', choices=[512, 1024, 2048], default=[2048])
    p.add_argument('--boo-k', type=int, nargs='+', default=[8, 16, 32, 64])
    p.add_argument('--boo-threads', type=int, choices=[128, 256, 512], default=256)
    p.add_argument('--boo-p', type=int, choices=[8, 16, 32, 64, 128], default=64)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=Path('results/density-bands'))
    p.add_argument('--project', default='torch-fast-spdmm')
    p.add_argument('--entity', default=None)
    p.add_argument('--name', default=None)
    p.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    a = p.parse_args(argv)
    a.boo_clusters = ['hem']  # configurations() expects this; density profiles don't sweep it
    if min(a.sizes + a.features + a.boo_k) <= 0 or a.repeats < 1 or a.warmup < 0:
        p.error('Sizes, features, k and repeats must be positive; warmup must be nonnegative')
    return a


def round_orders(count, repeats, seed):
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return [order[r % count:] + order[:r % count] for r in range(repeats)]


def cpp_only_variants(mat, configs, seed, x):
    """PBR/BOO cpp_only entries (via _symmetric_variants, filtered to drop its
    diagnostic entries) plus direct cuSPARSE cusparseSpMM_only entries --
    exactly the 4+8+4=16 configs/workload this benchmark needs, nothing else.
    """
    base_variants = build_variants(mat, [c for c in configs if c['algorithm'] != 'cusparse_csr'], seed)
    for config, _ in base_variants:
        if config['algorithm'] in ('pbr', 'boo'):
            family = 'block' if config['algorithm'] == 'pbr' else 'boo'
            config['execution_path'] = ('csr_only' if config['compressed_fraction'] == 0 else
                                         f"{family}+csr" if config['remainder_nnz'] else family)
    extra = _symmetric_variants(base_variants, x)
    cpp_only = [(c, o) for c, o in extra if c['timing_scope'] == 'cpp_only']

    csr = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr.copy()),
                                  torch.from_numpy(mat.indices.copy()),
                                  torch.from_numpy(mat.data.copy()), size=mat.shape).cuda()
    direct = []
    for algorithm in DIRECT_ALGORITHMS:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        plan = core.DirectSpMM(csr.crow_indices(), csr.col_indices(), csr.values(), x, algorithm)
        torch.cuda.synchronize()
        config = dict(algorithm='cusparse_direct', config=f'direct_{algorithm}',
                      direct_algorithm=algorithm, timing_scope='cusparseSpMM_only',
                      execution_path='cusparseSpMM', compressed_fraction=None, remainder_nnz=None,
                      num_blocks=None, plan_build_ms=1000 * (time.perf_counter() - t0), **plan.info())
        direct.append((config, plan))
    return cpp_only + direct


def measure_cell(mat, configs, x_cpu, warmup, repeats, seed):
    x = x_cpu.to('cuda')
    variants = cpp_only_variants(mat, configs, seed, x)

    reference = mat.astype(np.float64) @ x_cpu.numpy().astype(np.float64)
    validation = {}
    for config, obj in variants:
        if not config.get('supported', True):
            validation[config['config']] = None
            continue
        try:
            y = obj.run()
        except RuntimeError as exc:
            if config['algorithm'] != 'cusparse_direct' or 'CUSPARSE_STATUS_NOT_SUPPORTED' not in str(exc):
                raise
            config.update(supported=False, unsupported_reason=str(exc))
            validation[config['config']] = None
            continue
        actual = y.cpu().numpy().astype(np.float64)
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4, err_msg=config['config'])
        validation[config['config']] = float(np.max(np.abs(actual - reference)))
        del y, actual
    del reference

    for config, obj in variants:
        if not config.get('supported', True):
            continue
        for _ in range(warmup):
            obj.timed_run()
    torch.cuda.synchronize()

    samples = []
    for repeat, order in enumerate(round_orders(len(variants), repeats, seed)):
        for position, index in enumerate(order):
            config, obj = variants[index]
            if not config.get('supported', True):
                continue
            ms = obj.timed_run()
            if not np.isfinite(ms) or ms <= 0:
                raise RuntimeError(f"Invalid elapsed time for {config['config']}: {ms}")
            samples.append(dict(config=config['config'], algorithm=config['algorithm'],
                                repeat=repeat + 1, position=position, time_ms=ms))

    summaries = []
    direct_medians = [statistics.median([s['time_ms'] for s in samples if s['config'] == c['config']])
                      for c, _ in variants if c['algorithm'] == 'cusparse_direct' and c.get('supported', True)]
    direct_baseline = min(direct_medians) if direct_medians else None
    for config, _ in variants:
        values = [s['time_ms'] for s in samples if s['config'] == config['config']]
        if not values:
            summaries.append(dict(config, time_ms=None, median_ms=None, mean_ms=None, min_ms=None,
                                  max_ms=None, std_ms=None, p25_ms=None, p75_ms=None, repeats=0,
                                  warmup=0, max_abs_error=None, correctness='not_supported',
                                  speedup_vs_direct=None))
            continue
        median = statistics.median(values)
        summaries.append(dict(config, time_ms=median, median_ms=median,
                              mean_ms=statistics.mean(values), min_ms=min(values), max_ms=max(values),
                              std_ms=statistics.stdev(values) if len(values) > 1 else 0.0,
                              p25_ms=float(np.percentile(values, 25)), p75_ms=float(np.percentile(values, 75)),
                              repeats=repeats, warmup=warmup, max_abs_error=validation[config['config']],
                              correctness='passed',
                              speedup_vs_direct=direct_baseline / median if direct_baseline else None))
    return summaries, samples


def main(argv=None):
    a = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; build and run on the remote GPU machine')
    a.out.mkdir(parents=True, exist_ok=True)
    if any((a.out / f).exists() for f in ('metadata.json', 'summary.csv', 'samples.csv')):
        raise SystemExit(f'Results already exist in {a.out}; choose a new --out directory')
    metadata = source_metadata(Path(__file__).resolve().parents[3])
    metadata.update(direct_algorithms=DIRECT_ALGORITHMS, bands=BANDS, band_target=BAND_TARGET,
                    scope='cpp_only (direct cusparseSpMM_only, PBRTimedSpMM/BOOTimedSpMM cpp_only) '
                          '-- no full_call, no diagnostic')
    args_dict = dict(vars(a), out=str(a.out), sizes=list(a.sizes), features=list(a.features),
                     profiles=list(a.profiles))
    run = wandb.init(project=a.project, entity=a.entity, name=a.name, mode=a.wandb_mode,
                     config=dict(args_dict, **metadata), dir=str(a.out))
    metadata.update(arguments=args_dict, wandb_run_id=run.id, wandb_url=run.url,
                    status='running', completed_cells=0)
    (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    configs = configurations(a)
    summary_columns = None
    all_summaries = []
    failed = True
    try:
        with torch.inference_mode():
            for size in a.sizes:
                for profile in a.profiles:
                    print(f'Preparing {profile}, requested N={size}', flush=True)
                    mat, band, note = build_density_matrix(size, profile, seed=a.seed)
                    if max(*mat.shape, mat.nnz) > np.iinfo(np.int32).max:
                        raise ValueError('Benchmark requires int32-compatible CSR')
                    mat.indptr = mat.indptr.astype(np.int32)
                    mat.indices = mat.indices.astype(np.int32)
                    actual_nnz_per_row = mat.nnz / mat.shape[0]
                    target = BAND_TARGET.get(band)
                    lo, hi = BANDS[band]
                    in_band = ('dofs' in profile) or (lo <= actual_nnz_per_row <= hi)
                    matrix_hash = digest_arrays(np.asarray(mat.shape), mat.indptr, mat.indices, mat.data)
                    for features in a.features:
                        generator = torch.Generator().manual_seed(a.seed)
                        x_cpu = torch.randn(mat.shape[1], features, generator=generator, dtype=torch.float32)
                        common = dict(profile=profile, band=band, target_nnz_per_row=target,
                                     actual_nnz_per_row=actual_nnz_per_row, in_band=in_band, note=note,
                                     requested_size=size, rows=mat.shape[0], cols=mat.shape[1],
                                     features=features, nnz=int(mat.nnz), dtype='float32', index_dtype='int32',
                                     seed=a.seed, matrix_sha256=matrix_hash, x_sha256=digest_arrays(x_cpu.numpy()),
                                     source_sha256=metadata['source_sha256'], wandb_run_id=run.id)
                        summaries, samples = measure_cell(mat, configs, x_cpu, a.warmup, a.repeats, a.seed)
                        summaries = [dict(common, **s) for s in summaries]
                        samples = [dict(common, **s) for s in samples]
                        if summary_columns is None:
                            summary_columns = sorted({k for r in summaries for k in r})
                        append_csv(a.out / 'summary.csv', summaries, summary_columns)
                        append_csv(a.out / 'samples.csv', samples, list(samples[0]))
                        all_summaries.extend(summaries)
                        cell = f'{profile}_{size}_f{features}'
                        run.log({f'summary/{cell}': table(summaries)})
                        for row in summaries:
                            if row['median_ms'] is None:
                                continue
                            print(f"{cell:36s} {row['config']:25s} {row['median_ms']:10.5f} ms "
                                  f"speedup {row['speedup_vs_direct']:.3f}x nnz/row={actual_nnz_per_row:.2f} "
                                  f"band={band}{'' if in_band else ' OUT_OF_BAND'}", flush=True)
                        metadata['completed_cells'] += 1
                        (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
                        del x_cpu
                    del mat
                    torch.cuda.empty_cache()
        failed = False
    except BaseException as exc:
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        metadata['status'] = 'failed' if failed else 'complete'
        (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        try:
            artifact = wandb.Artifact(f'density-bands-{run.id}', type='benchmark-results')
            for name in ('metadata.json', 'summary.csv', 'samples.csv'):
                if (a.out / name).exists():
                    artifact.add_file(str(a.out / name), name=name)
            run.log_artifact(artifact)
            run.summary['status'] = metadata['status']
            run.summary['completed_cells'] = metadata['completed_cells']
        finally:
            run.finish(exit_code=1 if failed else 0)
    print(f"DENSITY_BANDS_DONE: {a.out} W&B: {metadata['wandb_url']}", flush=True)


if __name__ == '__main__':
    main()
