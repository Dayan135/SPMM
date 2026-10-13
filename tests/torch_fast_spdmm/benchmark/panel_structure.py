"""CPU structural measurements for the profiling panel; no inferred DRAM traffic."""
from __future__ import annotations

import numpy as np


def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return dict(count=0, mean=0., p50=0., p90=0., p99=0., maximum=0.)
    return dict(count=int(values.size), mean=float(values.mean()),
                **dict(zip(('p50', 'p90', 'p99'), map(float, np.percentile(values, [50, 90, 99])))),
                maximum=float(values.max()))


def csr_structure(mat):
    rows = np.diff(mat.indptr)
    columns = np.bincount(mat.indices, minlength=mat.shape[1])
    return dict(shape=list(mat.shape), nnz=int(mat.nnz),
                row_degree=distribution(rows), column_degree=distribution(columns),
                empty_rows=int(np.count_nonzero(rows == 0)),
                empty_columns=int(np.count_nonzero(columns == 0)),
                csr_bytes=int(mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes),
                hub_column_nnz_fraction=float(columns[columns > 1024].sum() / max(mat.nnz, 1)))


def boo_structure(boo):
    """Count active support exactly, including group members unused by a block.

    Structural active-row counts bound output updates: numerical cancellation
    and the kernel's zero-sum skip can further reduce the actual atomic count.
    Byte models exclude values/remainder unless the field explicitly says so.
    """
    rp, cp = np.asarray(boo.row_grp_ptr), np.asarray(boo.col_grp_ptr)
    br, bc = np.asarray(boo.blk_row_grp), np.asarray(boo.blk_col_grp)
    ptr = np.asarray(boo.blk_nz_ptr)
    idx = np.asarray(boo.nz_idx, dtype=np.uint16)
    shifts = np.asarray(boo.blk_shift)
    nr, nc = np.diff(rp)[br], np.diff(cp)[bc]
    nz = np.diff(ptr)
    active_rows = np.empty(len(br), dtype=np.int64)
    active_cols = np.empty(len(br), dtype=np.int64)
    imbalance = np.empty(len(br), dtype=np.float64)
    fanout = np.zeros(int(boo.rows), dtype=np.int64)
    ids = np.asarray(boo.row_ids)
    for b in range(len(br)):
        local = idx[ptr[b]:ptr[b + 1]].astype(np.uint32)
        row, counts = np.unique(local >> int(shifts[b]), return_counts=True)
        col = np.unique(local & ((1 << int(shifts[b])) - 1))
        active_rows[b], active_cols[b] = len(row), len(col)
        # Include empty local rows in the imbalance statistic.
        mean = float(nz[b]) / int(nr[b])
        second_moment = float(np.square(counts.astype(np.float64)).sum()) / int(nr[b])
        imbalance[b] = np.sqrt(max(0., second_moment - mean * mean)) / mean if mean else 0.
        fanout[ids[rp[br[b]] + row]] += 1
    names = ('row_grp_ptr', 'row_ids', 'col_grp_ptr', 'col_ids', 'blk_row_grp',
             'blk_col_grp', 'blk_nz_ptr', 'blk_shift', 'nz_idx', 'nz_val',
             'rem_indptr', 'rem_col_ind', 'rem_vals', 'tail_row', 'tail_col', 'tail_val')
    return dict(num_blocks=len(br), block_nnz=distribution(nz),
                row_group_size=distribution(np.diff(rp)), col_group_size=distribution(np.diff(cp)),
                block_density=distribution(nz / np.maximum(nr.astype(np.int64) * nc, 1)),
                active_rows=distribution(active_rows), active_columns=distribution(active_cols),
                within_block_row_cv=distribution(imbalance), output_block_fanout=distribution(fanout),
                logical_x_rows=int(nc.sum()), active_x_rows=int(active_cols.sum()),
                potential_y_rows=int(nr.sum()), active_y_rows=int(active_rows.sum()),
                unused_x_fraction=float(1 - active_cols.sum() / max(int(nc.sum()), 1)),
                format_bytes=sum(np.asarray(getattr(boo, name)).nbytes for name in names),
                coordinate_bytes=int(2 * nz.sum()),
                uint8_csr_structure_bytes=int(nz.sum() + 2 * (nr.astype(np.int64) + 1).sum()
                                                + 4 * (len(br) + 1)),
                uint8_csr_model_eligible=bool(np.all(nr <= 64) and np.all(nc <= 64)),
                uint8_csr_model_note='uint8 columns + uint16 row offsets + int32 table pointers; no alignment',
                retained_nnz=int(nz.sum()), remainder_nnz=int(boo.remainder_nnz()))
