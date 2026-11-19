"""Governed synthetic_controls diagnostic panel: setup, provenance, refs, and label cache.

Isolated under <AGENT_ROOT>/suite/controls/ so it never mixes with the promotion suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import scipy.sparse as sp

from benchmark import real_matrices as rm
from benchmark import universal_suite as suite
from benchmark.bench_real import Reference
from benchmark.matrices import (
    generate_diag_blocks_plus_noise,
    generate_fem_2d,
    generate_permuted_fem_2d,
)

PANEL_NAME = 'synthetic_controls'
CONTROL_SIZE = 262144
CONTROL_FEATURES = (128, 512)
CONTROL_SEED = 0
CONTROL_K = 32

CONTROL_SPECS = {
    'fem2d': {
        'diagnostic_role': 'regression_negative_control',
        'size': CONTROL_SIZE,
        'seed': CONTROL_SEED,
        'label_method': 'hem',
        'k': CONTROL_K,
        'generator': 'generate_fem_2d',
        'generator_params': {
            'system_size': CONTROL_SIZE,
            'dofs_per_node': 4,
            'coupling': 'anisotropic',
            'noise_density': 0.0,
            'seed': CONTROL_SEED,
        },
    },
    'fem2d_permuted': {
        'diagnostic_role': 'positive_ordering_locality_control',
        'size': CONTROL_SIZE,
        'seed': CONTROL_SEED,
        'label_method': 'hem',
        'k': CONTROL_K,
        'generator': 'generate_permuted_fem_2d',
        'generator_params': {
            'system_size': CONTROL_SIZE,
            'dofs_per_node': 4,
            'coupling': 'anisotropic',
            'noise_density': 0.0,
            'seed': CONTROL_SEED,
        },
    },
    'diag_b1.0': {
        'diagnostic_role': 'ideal_dense_block_ceiling_control',
        'size': CONTROL_SIZE,
        'seed': CONTROL_SEED,
        'label_method': 'identity',
        'k': CONTROL_K,
        'generator': 'generate_diag_blocks_plus_noise',
        'generator_params': {
            'system_size': CONTROL_SIZE,
            'block_size': 64,
            'diag_density': 1.0,
            'off_diag_density': 0.0,
            'seed': CONTROL_SEED,
        },
    },
}
CONTROL_PROFILES = ('fem2d', 'fem2d_permuted', 'diag_b1.0')


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_save_csr(path: Path, csr: sp.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    sp.save_npz(tmp, csr)
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npy')
    np.save(tmp, arr)
    os.replace(tmp, path)


def _generator_src_sha256() -> str:
    import benchmark.matrices as bm_matrices
    h = hashlib.sha256()
    for p in (Path(bm_matrices.__file__).resolve(), Path(__file__).resolve()):
        h.update(p.read_bytes())
    return h.hexdigest()


def generate_control_matrix(profile: str) -> sp.csr_matrix:
    if profile not in CONTROL_SPECS:
        raise ValueError(f'unknown synthetic_controls profile {profile!r}')
    spec = CONTROL_SPECS[profile]
    params = spec['generator_params']
    if profile == 'fem2d':
        mat = generate_fem_2d(**params)
    elif profile == 'fem2d_permuted':
        mat = generate_permuted_fem_2d(**params)
    else:
        mat = generate_diag_blocks_plus_noise(**params)
    csr = mat.astype(np.float32).tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    if max(*csr.shape, csr.nnz) > np.iinfo(np.int32).max:
        raise ValueError(f'{profile}: CSR exceeds int32 limits')
    csr.indices = csr.indices.astype(np.int32)
    csr.indptr = csr.indptr.astype(np.int32)
    return csr


def _matrix_path(root: Path, profile: str) -> Path:
    return root / 'generated' / f'{profile}.npz'


def load_matrix(root: Path, profile: str, manifest: dict | None = None) -> tuple[sp.csr_matrix, dict]:
    if profile not in CONTROL_SPECS:
        raise ValueError(f'unknown synthetic_controls profile {profile!r}')
    root = Path(root)
    if manifest is None:
        manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('panel') != PANEL_NAME:
        raise ValueError(f'manifest panel mismatch: expected {PANEL_NAME!r}, got {manifest.get("panel")!r}')
    path = _matrix_path(root, profile)
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; run setup_controls')
    csr = sp.load_npz(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    expected_hash = manifest['matrices'][profile]['csr_hash']
    actual_hash = rm.matrix_hash(csr)
    if actual_hash != expected_hash or meta.get('csr_hash') != expected_hash:
        raise ValueError(f'{profile}: matrix hash mismatch (manifest={expected_hash}, actual={actual_hash})')
    return csr, meta


def _label_key(profile: str, method: str, k: int, clustering_src_hash: str) -> str:
    return f'{profile}_{method}_k{k}_{clustering_src_hash[:16]}'


def label_hash(row_labels: np.ndarray, col_labels: np.ndarray) -> str:
    return rm.digest_arrays(np.asarray(row_labels), np.asarray(col_labels))


def _validate_labels(row_labels: np.ndarray, col_labels: np.ndarray, shape: tuple[int, int], k: int) -> None:
    for labels, length in zip((row_labels, col_labels), shape):
        if labels.ndim != 1 or len(labels) != length or labels.dtype.kind not in 'iu':
            raise ValueError('cached labels have invalid shape or dtype')
        if labels.size:
            if labels.min() < 0 or labels.max() >= length:
                raise ValueError('cached labels out of range')
            counts = np.bincount(labels)
            if (counts == 0).any() or counts.max() > k:
                raise ValueError('cached labels have empty groups or exceed k')


def load_or_build_labels(csr: sp.csr_matrix, profile: str, root: Path, manifest: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch_fast_spdmm.clustering as clustering

    root = Path(root)
    spec = CONTROL_SPECS[profile]
    method = spec['label_method']
    k = spec['k']
    clustering_src_hash = rm.file_sha256(clustering.__file__)
    key = _label_key(profile, method, k, clustering_src_hash)
    npz_path = root / 'labels' / f'{key}.npz'
    meta_path = root / 'labels' / f'{key}.json'
    csr_hash = manifest['matrices'][profile]['csr_hash']

    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get('panel') == PANEL_NAME and meta.get('profile') == profile
                and meta.get('csr_hash') == csr_hash and meta.get('label_method') == method
                and meta.get('k') == k and meta.get('clustering_src_sha256') == clustering_src_hash):
            cached = np.load(npz_path)
            row_labels = cached['row_labels']
            col_labels = cached['col_labels']
            lhash = label_hash(row_labels, col_labels)
            if str(cached['csr_hash']) == csr_hash and lhash == meta.get('label_hash'):
                _validate_labels(row_labels, col_labels, csr.shape, k)
                return row_labels, col_labels, {
                    'cached': True,
                    'cache_path': str(npz_path),
                    'label_hash': lhash,
                    'label_method': method,
                    'k': k,
                    'clustering_src_sha256': clustering_src_hash,
                    'cluster_ms': meta.get('cluster_ms'),
                }

    t0 = time.perf_counter()
    row_labels, col_labels = clustering.cluster_axes(
        csr, method=method, target_nr=k, target_nc=k,
        max_nr=k, max_nc=k, max_block_nnz=None, seed=0,
        symmetric=None, feasible_first=True,
    )
    cluster_ms = 1000.0 * (time.perf_counter() - t0)
    _validate_labels(row_labels, col_labels, csr.shape, k)
    lhash = label_hash(row_labels, col_labels)
    _atomic_save_npz(npz_path, row_labels=row_labels, col_labels=col_labels,
                     csr_hash=np.array(csr_hash))
    meta = {
        'panel': PANEL_NAME,
        'profile': profile,
        'csr_hash': csr_hash,
        'label_method': method,
        'k': k,
        'label_hash': lhash,
        'clustering_src_sha256': clustering_src_hash,
        'cluster_ms': round(cluster_ms, 3),
        'npz_sha256': rm.file_sha256(npz_path),
    }
    _atomic_write_text(meta_path, json.dumps(meta, indent=2, sort_keys=True) + '\n')
    return row_labels, col_labels, {
        'cached': False,
        'cache_path': str(npz_path),
        'label_hash': lhash,
        'label_method': method,
        'k': k,
        'clustering_src_sha256': clustering_src_hash,
        'cluster_ms': meta['cluster_ms'],
    }


def prepare(root: Path) -> dict:
    root = Path(root)
    (root / 'generated').mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    manifest = {
        'panel': PANEL_NAME,
        'gating': False,
        'promotion_eligible': False,
        'version': 1,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'python_version': platform.python_version(),
        'matrices': {},
    }
    gen_src_hash = _generator_src_sha256()
    for profile in CONTROL_PROFILES:
        spec = CONTROL_SPECS[profile]
        npz_path = _matrix_path(root, profile)
        meta_path = npz_path.with_suffix('.json')
        csr = None
        if npz_path.exists() and meta_path.exists():
            candidate_csr = sp.load_npz(npz_path)
            candidate_meta = json.loads(meta_path.read_text())
            if (candidate_meta.get('generator') == spec['generator']
                    and candidate_meta.get('generator_params') == spec['generator_params']
                    and candidate_meta.get('generator_src_sha256') == gen_src_hash
                    and candidate_meta.get('numpy_version') == np.__version__
                    and candidate_meta.get('scipy_version') == scipy.__version__
                    and candidate_meta.get('csr_hash') == rm.matrix_hash(candidate_csr)):
                csr = candidate_csr
        if csr is None:
            csr = generate_control_matrix(profile)
            _atomic_save_csr(npz_path, csr)
        csr_hash = rm.matrix_hash(csr)
        cols = np.bincount(csr.indices, minlength=csr.shape[1])
        record = {
            'id': profile,
            'diagnostic_role': spec['diagnostic_role'],
            'generator': spec['generator'],
            'generator_params': spec['generator_params'],
            'generator_src_sha256': gen_src_hash,
            'seed': spec['seed'],
            'label_method': spec['label_method'],
            'k': spec['k'],
            'canonical_shape': [int(csr.shape[0]), int(csr.shape[1])],
            'canonical_nnz': int(csr.nnz),
            'nonempty_cols': int(np.count_nonzero(cols)),
            'data_dtype': str(csr.data.dtype),
            'index_dtype': str(csr.indices.dtype),
            'csr_hash': csr_hash,
            'numpy_version': np.__version__,
            'scipy_version': scipy.__version__,
        }
        _atomic_write_text(meta_path, json.dumps(record, indent=2, sort_keys=True) + '\n')
        manifest['matrices'][profile] = record
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def refs(root: Path, features: tuple[int, ...] = CONTROL_FEATURES) -> dict:
    root = Path(root)
    out = root / 'refs'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('panel') != PANEL_NAME:
        raise ValueError(f'invalid controls manifest panel: {manifest.get("panel")!r}')
    written = {}
    for profile in CONTROL_PROFILES:
        csr, _ = load_matrix(root, profile, manifest)
        csr_hash = manifest['matrices'][profile]['csr_hash']
        for f in features:
            stem = out / f'{profile}_F{f}'
            meta_path = Path(f'{stem}.json')
            x = suite.make_x(csr.shape[1], f)
            xsha = suite.x_hash(x)
            if meta_path.exists():
                existing = json.loads(meta_path.read_text())
                if (existing.get('matrix_sha256') == csr_hash
                        and existing.get('x_sha256') == xsha
                        and existing.get('features') == f
                        and all(Path(f'{stem}.{p}.npy').exists()
                                and rm.file_sha256(f'{stem}.{p}.npy') == existing['sha256'][p]
                                for p in ('ref', 'absref', 'tol'))):
                    written[f'{profile}_F{f}'] = existing
                    continue
            ref = Reference(csr, x)
            hashes = {}
            for part in ('ref', 'absref', 'tol'):
                part_path = Path(f'{stem}.{part}.npy')
                _atomic_save_npy(part_path, getattr(ref, part))
                hashes[part] = rm.file_sha256(part_path)
            meta = {
                'panel': PANEL_NAME,
                'profile': profile,
                'matrix_sha256': csr_hash,
                'x_sha256': xsha,
                'features': f,
                'sha256': hashes,
            }
            _atomic_write_text(meta_path, json.dumps(meta, indent=2, sort_keys=True) + '\n')
            written[f'{profile}_F{f}'] = meta
    return written


def labels(root: Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('panel') != PANEL_NAME:
        raise ValueError(f'invalid controls manifest panel: {manifest.get("panel")!r}')
    info = {}
    for profile in CONTROL_PROFILES:
        csr, _ = load_matrix(root, profile, manifest)
        _, _, meta = load_or_build_labels(csr, profile, root, manifest)
        info[profile] = meta
    return info


def _validate_manifest_contract(manifest: dict) -> str | None:
    if not isinstance(manifest, dict) or manifest.get('panel') != PANEL_NAME:
        return f'unexpected panel {manifest.get("panel")!r}'
    if manifest.get('gating') is not False or manifest.get('promotion_eligible') is not False:
        return 'manifest must have gating=false and promotion_eligible=false'
    if manifest.get('version') != 1:
        return f'unexpected manifest version {manifest.get("version")!r}'
    if manifest.get('numpy_version') != np.__version__ or manifest.get('scipy_version') != scipy.__version__:
        return 'manifest numpy/scipy version is stale for current environment'
    matrices = manifest.get('matrices')
    if not isinstance(matrices, dict) or set(matrices.keys()) != set(CONTROL_PROFILES):
        return 'manifest profiles do not match fixed control panel'
    gen_src_hash = _generator_src_sha256()
    for profile in CONTROL_PROFILES:
        rec = matrices[profile]
        spec = CONTROL_SPECS[profile]
        if rec.get('id') != profile or rec.get('diagnostic_role') != spec['diagnostic_role']:
            return f'{profile}: id/diagnostic_role mismatch'
        if rec.get('generator') != spec['generator'] or rec.get('generator_params') != spec['generator_params']:
            return f'{profile}: generator/params mismatch'
        if rec.get('seed') != spec['seed'] or rec.get('label_method') != spec['label_method'] or rec.get('k') != spec['k']:
            return f'{profile}: seed/label_method/k mismatch'
        if rec.get('canonical_shape') != [CONTROL_SIZE, CONTROL_SIZE]:
            return f'{profile}: unexpected shape {rec.get("canonical_shape")}'
        if rec.get('data_dtype') != 'float32' or rec.get('index_dtype') != 'int32':
            return f'{profile}: unexpected dtypes'
        if rec.get('generator_src_sha256') != gen_src_hash:
            return f'{profile}: stale generator_src_sha256'
        if rec.get('numpy_version') != np.__version__ or rec.get('scipy_version') != scipy.__version__:
            return f'{profile}: stale numpy/scipy version'
    return None


def status(root: Path, *, check_labels: bool = True) -> dict:
    root = Path(root)
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        return {'panel': PANEL_NAME, 'ready': False, 'reason': 'controls manifest.json missing', 'root': str(root)}
    manifest = json.loads(manifest_path.read_text())
    contract_err = _validate_manifest_contract(manifest)
    if contract_err:
        return {'panel': PANEL_NAME, 'ready': False, 'reason': contract_err, 'root': str(root)}
    missing_matrices, missing_refs, missing_labels = [], [], []
    for profile in CONTROL_PROFILES:
        if profile not in manifest.get('matrices', {}):
            missing_matrices.append(profile)
            continue
        try:
            csr, _ = load_matrix(root, profile, manifest)
        except Exception as exc:
            missing_matrices.append(f'{profile}:{exc}')
            continue
        csr_hash = manifest['matrices'][profile]['csr_hash']
        for f in CONTROL_FEATURES:
            stem = root / 'refs' / f'{profile}_F{f}'
            meta_path = Path(f'{stem}.json')
            if not meta_path.exists():
                missing_refs.append(f'{profile}_F{f}')
                continue
            ref_meta = json.loads(meta_path.read_text())
            if ref_meta.get('matrix_sha256') != csr_hash or ref_meta.get('features') != f:
                missing_refs.append(f'{profile}_F{f}:hash_mismatch')
                continue
            for part in ('ref', 'absref', 'tol'):
                p = Path(f'{stem}.{part}.npy')
                if not p.exists() or rm.file_sha256(p) != ref_meta.get('sha256', {}).get(part):
                    missing_refs.append(f'{profile}_F{f}.{part}')
                    break
        if check_labels:
            import torch_fast_spdmm.clustering as clustering
            spec = CONTROL_SPECS[profile]
            c_hash = rm.file_sha256(clustering.__file__)
            key = _label_key(profile, spec['label_method'], spec['k'], c_hash)
            lmeta = root / 'labels' / f'{key}.json'
            lnpz = root / 'labels' / f'{key}.npz'
            if not (lmeta.exists() and lnpz.exists()):
                missing_labels.append(profile)
            else:
                meta = json.loads(lmeta.read_text())
                if meta.get('csr_hash') != csr_hash or meta.get('clustering_src_sha256') != c_hash:
                    missing_labels.append(f'{profile}:stale')
    ready = not (missing_matrices or missing_refs or missing_labels)
    return {
        'panel': PANEL_NAME,
        'ready': ready,
        'root': str(root),
        'manifest_sha256': rm.file_sha256(manifest_path),
        'missing_matrices': missing_matrices,
        'missing_refs': missing_refs,
        'missing_labels': missing_labels,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'refs', 'labels', 'status'])
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.stage == 'prepare':
        prepare(args.root)
    elif args.stage == 'refs':
        refs(args.root)
    elif args.stage == 'labels':
        labels(args.root)
    else:
        print(json.dumps(status(args.root, check_labels=True), indent=2))


if __name__ == '__main__':
    main()
