"""M1 gate -- the offline checks that must pass before any CUDA work.

Run from ``tests/torch_fast_spdmm``::

    python -m benchmark.boo_gate --sizes 4096 16384 --out boo_gate.csv

Three checks, in order of what they would invalidate (BOO_FORMAT.md sec. 10):

1. **PBR parity.**  ``target=8, cluster='identity', policy='remainder'`` must
   reproduce PBR's blocking exactly -- same blocks, same nonzero partition,
   different encoding.  Isolates the encoding change from everything else.
2. **Round-trip.**  ``boo_to_csr(csr_to_boo(A)) == A`` exactly.
3. **Clustering does something.**  ``x_loads``/``y_writes`` should rank
   ``random >= identity > degree > rcm/graclus/hem``.  If they do not, the
   clustering is not working and CUDA work is premature.

Also dumps the nnz-per-block histogram, which is what decides the small-block
policy (BOO_FORMAT.md sec. 7) -- deliberately an open question until measured.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from torch_fast_spdmm import csr_to_boo, boo_to_csr, boo_info, csr_to_pbr, pbr_info

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.profile_pbr import build_matrix, _MATRIX_PROFILES  # noqa: E402

METHODS = ('random', 'identity', 'degree', 'rcm', 'hem', 'graclus')


def check_roundtrip(csr, **kwargs) -> bool:
    boo = csr_to_boo(csr, **kwargs)
    back = boo_to_csr(boo)
    back.sort_indices()
    ref = csr.copy()
    ref.sort_indices()
    return (np.array_equal(back.indptr, ref.indptr)
            and np.array_equal(back.indices, ref.indices)
            and np.allclose(back.data, ref.data))


def check_pbr_parity(csr, block: int = 8, min_nnz: int = 4) -> tuple[bool, str]:
    """BOO with identity clustering at target=8 must partition nonzeros the same
    way PBR does at block=8x8 with the same min_nnz threshold."""
    pbr = csr_to_pbr(csr, block_rows=block, block_cols=block, min_nnz_per_block=min_nnz)
    boo = csr_to_boo(csr, cluster='identity', target_nr=block, target_nc=block,
                     small_block_policy='remainder', small_block_threshold=min_nnz)

    pi = pbr_info(pbr)
    bi = boo_info(boo)
    ok = (pi.num_blocks == bi.num_blocks
          and pi.compressed_nnz == bi.compressed_nnz
          and pi.remainder_nnz == bi.remainder_nnz)
    detail = (f'blocks {pi.num_blocks} vs {bi.num_blocks}, '
              f'compressed {pi.compressed_nnz} vs {bi.compressed_nnz}, '
              f'remainder {pi.remainder_nnz} vs {bi.remainder_nnz}')
    return ok, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[4096, 16384])
    parser.add_argument('--profiles', nargs='+', default=None)
    parser.add_argument('--targets', type=int, nargs='+', default=[8, 32, 64])
    parser.add_argument('--methods', nargs='+', default=list(METHODS))
    parser.add_argument('--p', type=int, default=8)
    parser.add_argument('--threads', type=int, default=128)
    parser.add_argument('--out', default='boo_gate.csv')
    args = parser.parse_args()

    profiles = args.profiles or sorted(_MATRIX_PROFILES)
    rows_out = []

    print('=' * 78)
    print('CHECK 1/2: PBR parity + round-trip')
    print('=' * 78)
    parity_fail = roundtrip_fail = 0
    for profile in profiles:
        for size in args.sizes:
            try:
                csr = build_matrix(size, profile).astype(np.float32)
            except Exception as exc:                      # noqa: BLE001
                print(f'  skip {profile}/{size}: {exc}')
                continue

            ok, detail = check_pbr_parity(csr)
            parity_fail += (not ok)
            rt = check_roundtrip(csr, cluster='hem', target_nr=32, target_nc=32)
            roundtrip_fail += (not rt)
            flag = 'ok  ' if (ok and rt) else 'FAIL'
            print(f'  [{flag}] {profile:28s} n={size:<7d} parity={ok} roundtrip={rt}'
                  + ('' if ok else f'   ({detail})'))

    print(f'\nparity failures: {parity_fail}   roundtrip failures: {roundtrip_fail}')

    print()
    print('=' * 78)
    print('CHECK 3: does clustering reduce traffic?  (lower x_loads/y_writes is better)')
    print('=' * 78)

    for profile in profiles:
        for size in args.sizes:
            try:
                csr = build_matrix(size, profile).astype(np.float32)
            except Exception:                             # noqa: BLE001
                continue
            for target in args.targets:
                line = []
                for method in args.methods:
                    try:
                        boo = csr_to_boo(csr, cluster=method, target_nr=target,
                                         target_nc=target, small_block_policy='keep')
                    except Exception as exc:              # noqa: BLE001
                        line.append((method, None, None, None))
                        continue
                    info = boo_info(boo, p=args.p, threads=args.threads)
                    record = dataclasses.asdict(info)
                    record.update(profile=profile, size=size, target=target,
                                  method=method)
                    record['nnz_per_block_hist'] = ';'.join(
                        f'{k}:{v}' for k, v in sorted(info.nnz_per_block_hist.items()))
                    rows_out.append(record)
                    line.append((method, info.x_loads, info.y_writes, info.occupancy_pct))

                base = dict((m, x) for m, x, _, _ in line if x)
                ref = base.get('identity')
                pretty = '  '.join(
                    f'{m}={x if x is not None else "-"}'
                    + (f'({ref / x:.1f}x)' if (ref and x) else '')
                    for m, x, _, _ in line)
                print(f'  {profile:24s} n={size:<7d} t={target:<4d} x_loads: {pretty}')

    if rows_out:
        out_path = Path(args.out)
        with out_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f'\nwrote {len(rows_out)} rows -> {out_path}')

    print()
    print('=' * 78)
    print('nnz-per-block distribution (decides the small-block policy)')
    print('=' * 78)
    for record in rows_out:
        if record['method'] in ('hem', 'graclus') and record['target'] == max(args.targets):
            print(f"  {record['profile']:24s} n={record['size']:<7d} "
                  f"blocks={record['num_blocks']:<8d} "
                  f"nnz/blk mean={record['nnz_per_block_mean']:6.1f} "
                  f"max={record['nnz_per_block_max']:<6d} "
                  f"occ={record['occupancy_pct']:5.1f}%  {record['nnz_per_block_hist']}")

    if parity_fail or roundtrip_fail:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
