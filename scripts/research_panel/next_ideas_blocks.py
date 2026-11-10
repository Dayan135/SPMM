"""CPU-only per-block statistics for evaluating next kernel ideas (experiments/2026-11-11_next_ideas).

  CUDA_VISIBLE_DEVICES= python scripts/research_panel/next_ideas_blocks.py --run RUN --source-labels L \
      --cache CACHE --gate-dir GATE --task I --out OUT

Rebuilds the real BOO structures exactly as kcp_gate.py does (same labels, same build), checks each
boo_digest against the k_c x P gate CSV, and writes OUT/<input>__<shape>.npz with
  per block:  n_r, n_c, nnz, active_cols (distinct local columns), rows_nonempty
  per row:    row_len (rows of every block in block-row order, empty rows included), row_ptr
plus OUT/<input>.json with provenance.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT / 'torch_fast_spdmm'), str(ROOT),
                str(ROOT / 'scripts' / 'research_panel')]
import shape_gate as sg  # noqa: E402
from benchmark import nnzflat_common as nc  # noqa: E402

NAMES = ('fem2d_permuted', 'roadnet', 'diag_dd0.05', 'diag_b1.0', 'fem2d')
SHAPES = [(32, 32), (64, 32), (128, 32), (128, 64), (256, 32), (256, 64)]


def block_stats(b):
    rgp = np.asarray(b.row_grp_ptr, np.int64)
    cgp = np.asarray(b.col_grp_ptr, np.int64)
    brg = np.asarray(b.blk_row_grp, np.int64)
    bcg = np.asarray(b.blk_col_grp, np.int64)
    bnp = np.asarray(b.blk_nz_ptr, np.int64)
    shift = np.asarray(b.blk_shift, np.int64)
    idx = np.asarray(b.nz_idx, np.int64)
    nb = len(brg)
    n_r = rgp[brg + 1] - rgp[brg]
    n_c = cgp[bcg + 1] - cgp[bcg]
    nnz = np.diff(bnp)
    blk = np.repeat(np.arange(nb), nnz)
    sh = shift[blk]
    lrow = idx >> sh
    lcol = idx & ((np.int64(1) << sh) - 1)
    active = np.bincount(blk[np.unique(blk * 65536 + lcol, return_index=True)[1]], minlength=nb)
    row_key = blk * 65536 + lrow
    uk, counts = np.unique(row_key, return_counts=True)
    rows_nonempty = np.bincount(uk // 65536, minlength=nb)
    row_ptr = np.concatenate([[0], np.cumsum(n_r)])
    row_len = np.zeros(row_ptr[-1], np.int32)
    row_len[row_ptr[uk // 65536] + uk % 65536] = counts
    assert row_len.sum() == idx.size and (lrow < n_r[blk]).all() and (lcol < n_c[blk]).all()
    return dict(n_r=n_r.astype(np.int32), n_c=n_c.astype(np.int32), nnz=nnz.astype(np.int32),
                active_cols=active.astype(np.int32), rows_nonempty=rows_nonempty.astype(np.int32),
                row_len=row_len, row_ptr=row_ptr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--source-labels', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--gate-dir', type=Path, required=True)
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    import torch
    assert not torch.cuda.is_available(), 'CPU-only'
    name = NAMES[a.task]
    inp = nc.INPUTS[name]
    key = inp.key
    prefix = key.replace('/', '__', 1)
    digests = {r['shape']: r['boo_digest'] for r in csv.DictReader(open(a.gate_dir / f'{name}.csv'))
               if r.get('status') == 'ok'}
    csr, record = sg.load_csr(key, a.cache)
    labels = {k: sg.labels_for(key, prefix, csr, record, inp.method, k, a.source_labels, a.run)
              for k in sorted({k for s in SHAPES for k in s})}
    a.out.mkdir(parents=True, exist_ok=True)
    meta = dict(input=name, matrix=key, matrix_sha256=record['matrix_sha256'], host=sg.host(), shapes={})
    t0 = time.time()
    for kr, kc in SHAPES:
        shape = f'{kr}x{kc}'
        rl, _, rp = labels[kr]
        _, cl, cp = labels[kc]
        b = nc.build_boo(csr, rl, cl)
        dig = nc.boo_digest(b)
        if digests.get(shape) and dig != digests[shape]:
            raise SystemExit(f'{name} {shape}: boo_digest {dig} != gate {digests[shape]}')
        st = block_stats(b)
        np.savez_compressed(a.out / f'{name}__{shape}.npz', **st)
        meta['shapes'][shape] = dict(boo_digest=dig, gate_digest_checked=bool(digests.get(shape)),
                                     row_label_sha256=rp['label_sha256'], col_label_sha256=cp['label_sha256'],
                                     blocks=int(len(st['nnz'])), nnz=int(st['nnz'].sum()),
                                     max_nnz=int(st['nnz'].max()), max_nr=int(st['n_r'].max()),
                                     max_nc=int(st['n_c'].max()), max_active=int(st['active_cols'].max()))
        print(name, shape, meta['shapes'][shape], flush=True)
        del b
    meta['seconds'] = time.time() - t0
    (a.out / f'{name}.json').write_text(json.dumps(meta, indent=1) + '\n')
    print('NEXT_IDEAS_BLOCKS_DONE', name)


if __name__ == '__main__':
    main()
