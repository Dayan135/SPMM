"""Full FPT sweep using the real-panel harness for both native and generated CSR.

The adapter only registers a generated matrix in this process; it never edits
bench_real or real_matrices. All timings retain that harness's validation,
matched-coordinate baselines, memory accounting and saved-label checks.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

METHODS = ('identity', 'hem', 'graclus', 'metis')
REAL = ('Boeing/pwtk', 'Schmid/thermal2', 'AMD/G2_circuit',
        'Williams/webbase-1M', 'SNAP/roadNet-CA', 'LPnetlib/lp_cre_b')
GENERATED = ('fem2d', 'fem2d_permuted', 'diag_dd0.017', 'diag_dd0.033',
             'diag_dd0.05', 'diag_b0.1', 'diag_b0.3', 'diag_b1.0',
             'diag_skewed', 'pattern_diag_stripe')


def tasks():
    inputs = [dict(kind='real', matrix=name) for name in REAL]
    inputs += [dict(kind='generated', profile=p, size=n)
               for n in (262144, 1048576) for p in GENERATED]
    return [dict(item, method=m) for item in inputs for m in METHODS]


def register_generated(task, seed=0):
    import numpy as np
    from benchmark import real_matrices as rm
    from benchmark.bench_clustering import load_matrix
    profile, size = task['profile'], task['size']
    spec = rm.MatrixSpec('generated', f'{profile}_n{size}', 'generated control')
    rm.CATALOG[spec.key] = spec
    original_load = rm.load

    def load(selected, cache, explicit_zeros='keep'):
        if selected.key != spec.key:
            return original_load(selected, cache, explicit_zeros)
        if explicit_zeros != 'keep':
            raise ValueError('This sweep preserves structural zeros')
        csr = load_matrix(size, profile, seed).astype(np.float32)
        csr.sum_duplicates()
        csr.sort_indices()
        csr.indptr = csr.indptr.astype(np.int32)
        csr.indices = csr.indices.astype(np.int32)
        return csr, dict(matrix=spec.key, matrix_sha256=rm.matrix_hash(csr),
                         numerical_nnz=int(np.count_nonzero(csr.data)),
                         requested_size=size, profile=profile, seed=seed,
                         source_kind='generated', explicit_zeros_policy='keep')
    rm.load = load
    return spec.key


def run(root, task, phase, cache, index, *, smoke=False):
    from benchmark import bench_real as br
    key = task['matrix'] if task['kind'] == 'real' else register_generated(task)
    args = SimpleNamespace(
        cache=cache, labels=root / 'labels', out=root / 'results' / f'task_{index:03d}',
        matrices=[key], explicit_zeros='keep', methods=[task['method']], groups=[32, 64],
        features=[32, 128, 512] if not smoke else [35],
        direct_algorithms=list(br.DIRECT_PRIORITY), threads=256, threshold=4,
        boo_fpt=[1, 2, 4], seed=0, warmup=5 if not smoke else 1,
        repeats=20 if not smoke else 2)
    if phase == 'cluster':
        br.cluster_phase(args)
    else:
        br.matmul_phase(args)
        br.report(args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['manifest', 'cluster', 'matmul', 'smoke'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cache', type=Path)
    p.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    a = p.parse_args()
    if a.phase == 'manifest':
        a.root.mkdir(parents=True, exist_ok=True)
        (a.root / 'sweep.json').write_text(json.dumps(dict(tasks=tasks(), features=[32,128,512],
            groups=[32,64], fpt=[1,2,4], seed=0, warmup=5, repeats=20,
            expected_boo_full_call_combinations=1872), indent=2) + '\n')
    elif a.phase == 'smoke':
        # CPU labels are prepared by the caller with CUDA_VISIBLE_DEVICES empty.
        phase = os.environ['SWEEP_SMOKE_PHASE']
        for i, method in enumerate(METHODS):
            run(a.root, dict(kind='generated', profile='fem2d', size=1024, method=method),
                phase, a.cache, i, smoke=True)
    else:
        manifest = json.loads((a.root / 'sweep.json').read_text())
        run(a.root, manifest['tasks'][a.index], a.phase, a.cache, a.index)


if __name__ == '__main__':
    main()
