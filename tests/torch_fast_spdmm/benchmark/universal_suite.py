"""Canonical eleven-matrix suite and node-independent float32 references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import scipy.sparse as sp

from benchmark import real_matrices as rm
from benchmark.bench_real import Reference

SPECS = {
    'web-Google': ('SNAP', 916428, 916428, 5105039),
    'soc-Pokec': ('SNAP', 1632803, 1632803, 30622564),
    'roadNet-CA': ('SNAP', 1971281, 1971281, 5533214),
    'cant': ('Williams', 62451, 62451, 4007383),
    'pdb1HYS': ('Williams', 36417, 36417, 4344765),
    'ASIC_680k': ('Sandia', 682862, 682862, 2638997),
    'amazon0312': ('SNAP', 400727, 400727, 3200440),
    'rail4284': ('Mittelmann', 4284, 1096894, 11284032),
    'uniform16': (None, 262144, 262144, 4194304),
    'blockdiag32': (None, 262144, 262144, 8388608),
    'pruned16x16': (None, 4096, 16384, 4194304),
}
TUNE = ('cant', 'web-Google', 'roadNet-CA', 'uniform16', 'blockdiag32')
HELD = ('soc-Pokec', 'pdb1HYS', 'ASIC_680k', 'amazon0312', 'rail4284', 'pruned16x16')
SUBSETS = {'tune': TUNE, 'held': HELD, 'smoke': ('cant', 'roadNet-CA', 'uniform16'),
           'all': TUNE + HELD}
FEATURES = (32, 64, 128, 256, 512, 33, 65, 129)
X_SEED = 20261117


def expand(names):
    result = []
    for name in names:
        for item in SUBSETS.get(name, (name,)):
            if item not in SPECS:
                raise ValueError(f'unknown matrix {item}')
            if item not in result:
                result.append(item)
    return result


def _csr(indptr, indices, shape):
    csr = sp.csr_matrix((np.ones(len(indices), np.float32),
                         np.asarray(indices, np.int32), np.asarray(indptr, np.int32)),
                        shape=shape)
    csr.sort_indices()
    return csr


def generated(name):
    if name == 'uniform16':
        rng = np.random.Generator(np.random.PCG64(20261117))
        indptr = np.arange(0, 262144 * 16 + 1, 16, dtype=np.int32)
        indices = np.empty(262144 * 16, np.int32)
        for row in range(262144):
            indices[row*16:(row+1)*16] = np.sort(rng.choice(262144, 16, replace=False))
        return _csr(indptr, indices, (262144, 262144))
    if name == 'blockdiag32':
        indptr = np.arange(0, 262144 * 32 + 1, 32, dtype=np.int32)
        indices = (np.arange(262144, dtype=np.int32)[:, None] // 32 * 32 +
                   np.arange(32, dtype=np.int32)[None, :]).reshape(-1)
        return _csr(indptr, indices, (262144, 262144))
    if name == 'pruned16x16':
        rng = np.random.Generator(np.random.PCG64(20261118))
        indices = np.empty(4096 * 1024, np.int32)
        tile_offsets = np.arange(16, dtype=np.int32)
        for block_row in range(256):
            blocks = np.sort(rng.choice(1024, 64, replace=False)).astype(np.int32)
            row_cols = (blocks[:, None] * 16 + tile_offsets).reshape(-1)
            for local_row in range(16):
                row = block_row * 16 + local_row
                indices[row*1024:(row+1)*1024] = row_cols
        indptr = np.arange(0, 4096 * 1024 + 1, 1024, dtype=np.int32)
        return _csr(indptr, indices, (4096, 16384))
    raise ValueError(name)


def make_x(ncols, features, seed=X_SEED):
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(
        (ncols, features), dtype=np.float32)


def x_hash(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).hexdigest()


def _generated_path(root, name):
    return root / 'generated' / f'{name}.npz'


def load_matrix(root, name, cache):
    if SPECS[name][0] is None:
        path = _generated_path(Path(root), name)
        if not path.exists():
            raise FileNotFoundError(f'{path} missing; run setup_suite prepare')
        return sp.load_npz(path), json.loads(path.with_suffix('.json').read_text())
    spec = rm.resolve([name])[0]
    return rm.load(spec, cache, 'drop')


def _distribution(csr):
    rows = np.diff(csr.indptr).astype(np.float64)
    cols = np.bincount(csr.indices, minlength=csr.shape[1]).astype(np.float64)
    return {
        'row_nnz_median': float(np.median(rows)),
        'row_nnz_p95': float(np.percentile(rows, 95)),
        'row_nnz_p99': float(np.percentile(rows, 99)),
        'row_nnz_max': int(rows.max(initial=0)),
        'row_nnz_cv': float(rows.std() / rows.mean()) if rows.mean() else 0.,
        'empty_rows': int(np.count_nonzero(rows == 0)),
        'empty_cols': int(np.count_nonzero(cols == 0)),
        'col_degree_max': int(cols.max(initial=0)),
        'col_degree_cv': float(cols.std() / cols.mean()) if cols.mean() else 0.,
        'nonempty_cols': int(np.count_nonzero(cols)),
    }


def _format_stats(csr):
    import torch_fast_spdmm as spmm  # CPU node has the snapshot extension
    pbr = spmm.csr_to_pbr(csr, block_rows=8, block_cols=8, min_nnz_per_block=4)
    pbr_blocks = len(pbr.block_codes)
    pbr_nnz = len(pbr.block_data)
    pbr_origins = np.asarray(pbr.block_coords).reshape(-1, 2)[:, 0] if pbr_blocks else np.empty(0)
    pbr_touches = np.zeros(csr.shape[0], np.int32)
    codes = np.asarray(pbr.block_codes, dtype=np.uint64)
    origins = pbr_origins.astype(np.int64)
    for local in range(8):
        rows = origins + local
        hit = (((codes >> np.uint64(8*local)) & np.uint64(255)) != 0) & (rows < csr.shape[0])
        np.add.at(pbr_touches, rows[hit], 1)
    boo = spmm.csr_to_boo(csr, cluster='hem', target_nr=32, target_nc=32)
    boo_nnz = len(boo.nz_val)
    grp = np.asarray(boo.blk_row_grp, dtype=np.int32)
    touches = np.zeros(csr.shape[0], np.int32)
    # Each BOO array property returns a fresh copy: read every array once, never in a loop.
    ptr, ids = np.asarray(boo.row_grp_ptr), np.asarray(boo.row_ids)
    cptr, bcg = np.asarray(boo.col_grp_ptr), np.asarray(boo.blk_col_grp, dtype=np.int64)
    counts = np.diff(np.asarray(boo.blk_nz_ptr, dtype=np.int64))
    block_index = np.repeat(np.arange(len(grp), dtype=np.int64), counts)
    local_row = np.right_shift(np.asarray(boo.nz_idx, dtype=np.uint16),
                               np.asarray(boo.blk_shift, dtype=np.uint8)[block_index])
    output_row = ids[ptr[grp[block_index]] + local_row]
    pairs = np.unique(block_index * csr.shape[0] + output_row)
    touches += np.bincount(pairs % csr.shape[0], minlength=csr.shape[0]).astype(np.int32)
    return {
        'pbr': {'useful_block_occupancy': pbr_nnz/(pbr_blocks*64) if pbr_blocks else 0.,
                'allocated_bytes_per_useful_nnz': (8*pbr_blocks + 8*pbr_blocks +
                    4*(pbr_blocks+1) + 4*pbr_nnz) / pbr_nnz if pbr_nnz else 0.,
                'remainder_fraction': len(pbr.remainder_col_ind)/csr.nnz,
                'groups_touching_row_mean': float(pbr_touches.mean()),
                'groups_touching_row_max': int(pbr_touches.max(initial=0))},
        'boo': {'useful_block_occupancy': boo_nnz/int(np.sum(
                    np.diff(ptr.astype(np.int64))[grp] * np.diff(cptr.astype(np.int64))[bcg]))
                    if boo_nnz else 0.,
                'allocated_bytes_per_useful_nnz': (2*boo_nnz+4*boo_nnz+
                    4*len(boo.blk_nz_ptr)+9*len(boo.blk_row_grp))/boo_nnz if boo_nnz else 0.,
                'remainder_fraction': len(boo.rem_col_ind)/csr.nnz,
                'groups_touching_row_mean': float(touches.mean()),
                'groups_touching_row_max': int(touches.max(initial=0))},
    }


def download(root, cache, names=None):
    import urllib.error
    for name in expand(names or ['all']):
        group, rows, cols, nnz = SPECS[name]
        if group is None:
            continue
        spec = rm.resolve([name])[0]
        try:
            rm.download(spec, cache)
        except urllib.error.HTTPError as exc:
            # The SuiteSparse file host has refused every file (403) since 2026-11-18.
            # SNAP-group matrices fall back to the SNAP edge list; source.json records it.
            if name not in rm.SNAP_EDGELISTS:
                raise
            print(f'{name}: SuiteSparse HTTP {exc.code}; using SNAP edge list', flush=True)
            rm.download_snap(spec, cache, (rows, cols), nnz)


def prepare(root, cache, names=None):
    root = Path(root)
    (root / 'generated').mkdir(parents=True, exist_ok=True)
    path = root / 'manifest.json'
    # Incremental: matrices prepared earlier keep their records.
    previous = json.loads(path.read_text())['matrices'] if path.exists() else {}
    manifest = {'version': 1, 'numpy_version': np.__version__,
                'scipy_version': scipy.__version__, 'python_version': platform.python_version(),
                'matrices': previous}
    for name in expand(names or ['all']):
        group, rows, cols, nnz = SPECS[name]
        if group is None:
            path = _generated_path(root, name)
            if path.exists():
                csr = sp.load_npz(path)
            else:
                csr = generated(name)
                sp.save_npz(path, csr)
            record = {'generator': name, 'seed': 20261118 if name == 'pruned16x16' else 20261117,
                      'numpy_version': np.__version__, 'published_shape': [rows, cols],
                      'published_nnz': nnz, 'archive_sha256': None, 'member_path': None,
                      'original_header': None, 'url': None, 'explicit_zeros_removed': 0}
            path.with_suffix('.json').write_text(json.dumps(record))
        else:
            spec = rm.resolve([name])[0]
            record = rm.prepare(spec, cache, 'drop')
            csr, _ = rm.load(spec, cache, 'drop')
            source = json.loads((rm.matrix_dir(cache, spec) / 'source.json').read_text())
            record.update(published_shape=[rows, cols], published_nnz=nnz,
                          member_path=source['mtx_member'],
                          original_header={'field': record['mm_field'],
                                           'symmetry': record['mm_symmetry'],
                                           'file_entries': record['file_entries']})
        if csr.shape != (rows, cols) or csr.nnz != nnz:
            raise ValueError(f'{name}: canonical {csr.shape} nnz={csr.nnz}, expected {(rows,cols)} nnz={nnz}')
        record.update(id=name, canonical_shape=[rows, cols], canonical_nnz=int(csr.nnz),
                      data_dtype=str(csr.data.dtype), index_dtype=str(csr.indices.dtype),
                      csr_hash=rm.matrix_hash(csr), numpy_version=np.__version__,
                      scipy_version=scipy.__version__, **_distribution(csr), **_format_stats(csr))
        manifest['matrices'][name] = record
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def refs(root, cache, features=FEATURES, names=None):
    root = Path(root)
    out = root / 'refs'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name in expand(names or ['all']):
        csr, _ = load_matrix(root, name, cache)
        for f in features:
            stem = out / f'{name}_F{f}'
            if stem.with_suffix('.json').exists():
                continue
            x = make_x(csr.shape[1], f)
            ref = Reference(csr, x)
            hashes = {}
            for part in ('ref', 'absref', 'tol'):
                path = Path(f'{stem}.{part}.npy')
                np.save(path, getattr(ref, part))
                hashes[part] = rm.file_sha256(path)
            stem.with_suffix('.json').write_text(json.dumps({
                'matrix_sha256': manifest['matrices'][name]['csr_hash'],
                'x_sha256': x_hash(x), 'features': f, 'sha256': hashes}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['download', 'prepare', 'refs'])
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=rm.default_cache())
    ap.add_argument('--matrices', nargs='*', default=None,
                    help='matrix ids or subset names (default: all)')
    args = ap.parse_args(argv)
    if args.stage == 'download': download(args.root, args.cache, args.matrices)
    elif args.stage == 'prepare': prepare(args.root, args.cache, args.matrices)
    else: refs(args.root, args.cache, names=args.matrices)


if __name__ == '__main__':
    main()
