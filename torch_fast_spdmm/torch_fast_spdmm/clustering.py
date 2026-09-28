"""Row/column clustering for the BOO (Blocked COO) format.

BOO blocks are the non-empty ``(row group x col group)`` intersections of a
clustered grid.  This module decides that grid: it maps each row to a row-group
label and each column to a column-group label.  See ``BOO_FORMAT.md`` sec. 5.

Every method returns the same thing -- an ``int32`` label array per axis -- so
they are interchangeable at the call site and the offline comparison in the M1
gate is apples-to-apples.

The similarity weight used by the graph methods is

    G[i, j] = |col(i) ^ col(j)|          (shared-column count)

which is not a proxy: merging rows ``i`` and ``j`` needs
``deg(i) + deg(j) - |col(i) ^ col(j)|`` columns instead of ``deg(i) + deg(j)``,
so the saving *is* the edge weight and heavy-edge matching on ``G`` is literally
greedy on the true objective (total X traffic).
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

__all__ = ['cluster_axes', 'cluster_axis', 'CLUSTER_METHODS']

CLUSTER_METHODS = ('identity', 'random', 'degree', 'rcm', 'hem', 'graclus', 'metis')

# Columns above this degree are dropped when building the similarity graph.
# They connect nearly everything and carry no clustering signal, and without
# the guard `A @ A.T` costs sum_j deg(j)^2 -- which OOMs on dense profiles at
# N=262144.  Graph construction only; the columns stay in the matrix.
DEFAULT_HUB_DEGREE = 1024

# Keep only the K heaviest neighbours per row.  Turns the similarity graph from
# O(sum_j deg(j)^2) into O(K * rows).
DEFAULT_TOPK = 32

DEFAULT_CHUNK_ROWS = 4096


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pattern(mat: sp.spmatrix) -> sp.csr_matrix:
    """Binary CSR copy -- values discarded, only the sparsity pattern matters."""
    out = sp.csr_matrix(mat, copy=True)
    out.data = np.ones_like(out.data, dtype=np.int32)
    out.sum_duplicates()
    out.data[:] = 1
    return out


def is_pattern_symmetric(mat: sp.spmatrix) -> bool:
    """True if the matrix is square and its sparsity pattern equals its transpose's.

    When this holds, `A` is a graph and one vertex clustering serves both axes,
    which is both cheaper and structurally the right model (BOO_FORMAT sec. 5).
    """
    if mat.shape[0] != mat.shape[1]:
        return False
    pat = _pattern(mat)
    return (pat != pat.T).nnz == 0


def _tile_by_permutation(perm: np.ndarray, n: int, target: int) -> np.ndarray:
    """Assign contiguous runs of `target` vertices, in `perm` order, to one label."""
    labels = np.empty(n, dtype=np.int64)
    labels[perm] = np.arange(n, dtype=np.int64) // max(target, 1)
    return labels


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """Remap arbitrary label values onto 0..n_groups-1, preserving grouping."""
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def _enforce_caps(labels: np.ndarray, row_nnz: np.ndarray,
                  max_size: Optional[int], max_nnz: Optional[int]) -> np.ndarray:
    """Split any group that exceeds the size or nnz cap into conforming chunks.

    Applied to every method's output, so the builder's invariants hold no matter
    which clustering produced the labels -- `graclus` and `metis` in particular
    have no notion of our caps.  Members keep their relative order, so a split
    group stays contiguous in the original ordering.
    """
    if max_size is None and max_nnz is None:
        return labels

    labels = _relabel_consecutive(labels)
    order = np.argsort(labels, kind='stable')
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    group_starts = np.r_[boundaries, sorted_labels.size]

    out = np.empty_like(labels)
    next_label = 0
    for g in range(boundaries.size):
        members = order[group_starts[g]:group_starts[g + 1]]
        cur_size = 0
        cur_nnz = 0
        for member in members:
            member_nnz = int(row_nnz[member])
            size_full = max_size is not None and cur_size + 1 > max_size
            nnz_full = (max_nnz is not None and cur_size > 0
                        and cur_nnz + member_nnz > max_nnz)
            if cur_size > 0 and (size_full or nnz_full):
                next_label += 1
                cur_size = 0
                cur_nnz = 0
            out[member] = next_label
            cur_size += 1
            cur_nnz += member_nnz
        next_label += 1

    return out


# ---------------------------------------------------------------------------
# similarity graph
# ---------------------------------------------------------------------------

def _similarity_graph(pat: sp.csr_matrix, hub_degree: int, topk: Optional[int],
                      chunk_rows: int = DEFAULT_CHUNK_ROWS) -> sp.csr_matrix:
    """G[i,j] = |col(i) ^ col(j)|, hub-pruned, top-k capped, symmetrized.

    `pat` is (n x m) binary; the result is (n x n).  Works for rectangular input
    -- `pat @ pat.T` is square and symmetric for any `pat`, which is why the
    caller must NOT symmetrize the matrix itself.
    """
    n, m = pat.shape
    if n <= 1:
        return sp.csr_matrix((n, n), dtype=np.int64)

    col_degree = np.bincount(pat.indices, minlength=m)
    keep = col_degree <= hub_degree
    if keep.all():
        pruned = sp.csr_matrix((pat.data.astype(np.int64), pat.indices, pat.indptr),
                               shape=pat.shape)
    else:
        mask = keep[pat.indices]
        entry_rows = np.repeat(np.arange(n), np.diff(pat.indptr))
        kept_per_row = np.bincount(entry_rows[mask], minlength=n)
        pruned = sp.csr_matrix(
            (pat.data[mask].astype(np.int64), pat.indices[mask],
             np.r_[0, np.cumsum(kept_per_row)]),
            shape=pat.shape)

    transposed = pruned.T.tocsr()

    rows_all, cols_all, data_all = [], [], []
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = (pruned[start:stop] @ transposed).tocoo()
        if block.nnz == 0:
            continue
        r, c, d = block.row, block.col, block.data
        off_diagonal = c != (r + start)
        r, c, d = r[off_diagonal], c[off_diagonal], d[off_diagonal]
        if topk is not None and r.size:
            # rows asc, weight desc, col asc -- the col key makes ties deterministic
            order = np.lexsort((c, -d, r))
            sorted_rows = r[order]
            starts = np.searchsorted(sorted_rows, np.arange(stop - start))
            rank = np.arange(sorted_rows.size) - starts[sorted_rows]
            order = order[rank < topk]
            r, c, d = r[order], c[order], d[order]
        rows_all.append(r + start)
        cols_all.append(c)
        data_all.append(d)

    if not rows_all:
        return sp.csr_matrix((n, n), dtype=np.int64)

    graph = sp.coo_matrix(
        (np.concatenate(data_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(n, n)).tocsr()
    graph.sum_duplicates()
    # Top-k is one-sided; symmetrize so a heavy edge survives if either end kept it.
    return graph.maximum(graph.T)


# ---------------------------------------------------------------------------
# heavy-edge matching (the dependency-free Graclus-style coarsener)
# ---------------------------------------------------------------------------

def _pair_hash(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Deterministic, symmetric hash of an unordered vertex pair (splitmix64).

    Symmetry matters: `i` and `j` must agree on the value or they can never
    recognise each other as mutual bests.  Determinism matters because the whole
    module is required to be reproducible -- there is no RNG anywhere in `hem`.
    """
    lo = np.minimum(a, b).astype(np.uint64)
    hi = np.maximum(a, b).astype(np.uint64)
    h = lo * np.uint64(0x9E3779B97F4A7C15) ^ hi * np.uint64(0xC2B2AE3D27D4EB4F)
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return (h >> np.uint64(1)).astype(np.int64)   # drop the sign bit


def _mutual_max_pairs(graph: sp.csr_matrix, active: np.ndarray,
                      size: np.ndarray, nnz: np.ndarray,
                      max_size: Optional[int], max_nnz: Optional[int],
                      feasible_first: bool = False,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """One round of locally-dominant (mutual-max) matching.

    Sequential greedy HEM is inherently serial; mutual-max is the standard
    parallel-friendly substitute -- a maximal matching with weight >= 1/2 of the
    maximum -- and it is expressible as pure array ops, which is what keeps this
    vectorized and free of any RNG.

    Ties are broken by a symmetric per-PAIR hash, not by vertex index.  Index
    tie-breaking collapses on cliques: when every edge of a dense group has the
    same weight, every member picks the same lowest-index vertex, so only ONE
    mutual pair forms per group per round and coarsening crawls.  A pair hash
    makes each vertex prefer a different partner while still being symmetric --
    both endpoints compute the same value, which is what mutuality requires.
    """
    n = graph.shape[0]
    coo = graph.tocoo()
    if coo.nnz == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    live = active[coo.row] & active[coo.col]
    # A preferred but infeasible edge must not hide a feasible alternative.
    # Keep the legacy ordering selectable for paired performance experiments.
    if feasible_first:
        if max_size is not None:
            live &= (size[coo.row] + size[coo.col]) <= max_size
        if max_nnz is not None:
            live &= (nnz[coo.row] + nnz[coo.col]) <= max_nnz
    r, c, d = coo.row[live], coo.col[live], coo.data[live]
    if r.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    order = np.lexsort((-_pair_hash(r, c), -d, r))
    r, c = r[order], c[order]
    first = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])

    best = np.full(n, -1, dtype=np.int64)
    best[r[first]] = c[first]

    index = np.arange(n, dtype=np.int64)
    partner = np.where(best >= 0, best, index)
    mutual = (best >= 0) & (best[partner] == index)

    left = index[mutual & (index < partner)]
    right = partner[left]

    if left.size == 0:
        return left, right

    ok = np.ones(left.size, dtype=bool)
    if max_size is not None:
        ok &= (size[left] + size[right]) <= max_size
    if max_nnz is not None:
        ok &= (nnz[left] + nnz[right]) <= max_nnz
    return left[ok], right[ok]


def _hem_labels(pat: sp.csr_matrix, target: int, max_size: Optional[int],
                max_nnz: Optional[int], hub_degree: int, topk: Optional[int],
                exact_recompute: bool = True, residual_rounds: int = 3,
                feasible_first: bool = False, hierarchy: Optional[list] = None) -> np.ndarray:
    """Multilevel HEM, optionally recording original-vertex labels at every level.

    ``hierarchy`` receives identity followed by each successful contraction,
    finest to coarsest. Recording does not alter matching or returned labels.
    """
    n = pat.shape[0]
    labels = np.arange(n, dtype=np.int64)
    if hierarchy is not None:
        hierarchy.append(labels.copy())
    if n <= 1 or target <= 1:
        return labels

    size = np.ones(n, dtype=np.int64)
    nnz = np.diff(pat.indptr).astype(np.int64)
    level_pat = pat
    level_graph = None

    # Coarsen to CONVERGENCE, not for log2(target) levels.  A level only halves
    # the vertex count if the matching is perfect, and mutual-max is not: on a
    # clique of 16 it matches ~60% of the possible pairs per level.  Stopping at
    # log2(target) therefore leaves groups far short of the target.  The size and
    # nnz caps bound growth, so extra levels cannot overshoot -- they just let
    # slow-converging regions catch up.  The loop exits when a level matches
    # nothing; `max_levels` is a safety net, not the intended stopping rule.
    max_levels = 8 * max(1, int(np.ceil(np.log2(target)))) + 16
    for _ in range(max_levels):
        n_cur = level_pat.shape[0]
        if n_cur <= 1:
            break

        if exact_recompute or level_graph is None:
            # Supernode column supports combine by UNION, not by sum, so the
            # usual P G P^T shortcut over-counts.  Preprocessing is not
            # time-bounded and there are only ~log2(target) levels, so recompute.
            level_graph = _similarity_graph(level_pat, hub_degree, topk)
        if level_graph.nnz == 0:
            break

        active = np.ones(n_cur, dtype=bool)
        pair_left, pair_right = [], []
        for _round in range(residual_rounds):
            left, right = _mutual_max_pairs(level_graph, active, size, nnz,
                                            max_size, max_nnz, feasible_first)
            if left.size == 0:
                break
            pair_left.append(left)
            pair_right.append(right)
            active[left] = False
            active[right] = False

        if not pair_left:
            break
        left = np.concatenate(pair_left)
        right = np.concatenate(pair_right)

        parent = np.arange(n_cur, dtype=np.int64)
        parent[right] = left            # pairs are disjoint, so no chains form
        roots = np.flatnonzero(parent == np.arange(n_cur))
        remap = np.full(n_cur, -1, dtype=np.int64)
        remap[roots] = np.arange(roots.size, dtype=np.int64)
        new_id = remap[parent]
        n_new = roots.size

        size = np.bincount(new_id, weights=size, minlength=n_new).astype(np.int64)
        nnz = np.bincount(new_id, weights=nnz, minlength=n_new).astype(np.int64)
        labels = new_id[labels]
        if hierarchy is not None:
            hierarchy.append(labels.copy())

        contraction = sp.csr_matrix(
            (np.ones(n_cur, dtype=np.int64), (new_id, np.arange(n_cur))),
            shape=(n_new, n_cur))
        level_pat = contraction @ level_pat
        level_pat.data[:] = 1           # binarize: support is a union, not a count
        level_graph = None if exact_recompute else level_graph

    return labels


# ---------------------------------------------------------------------------
# packaged-library methods
# ---------------------------------------------------------------------------

def _graclus_labels(pat: sp.csr_matrix, target: int, hub_degree: int,
                    topk: Optional[int]) -> np.ndarray:
    import torch
    from torch_cluster import graclus_cluster

    n = pat.shape[0]
    labels = np.arange(n, dtype=np.int64)
    level_pat = pat

    # Same convergence rule as `_hem_labels`: graclus halves at best, so a fixed
    # log2(target) level count undershoots.  Caps are applied afterwards by
    # `_enforce_caps`, so the stopping rule here is mean group size.
    max_levels = 8 * max(1, int(np.ceil(np.log2(max(target, 1))))) + 16
    for _ in range(max_levels):
        if level_pat.shape[0] <= 1 or n / level_pat.shape[0] >= target:
            break
        graph = _similarity_graph(level_pat, hub_degree, topk).tocoo()
        if graph.nnz == 0:
            break
        cluster = graclus_cluster(
            torch.from_numpy(graph.row.astype(np.int64)),
            torch.from_numpy(graph.col.astype(np.int64)),
            torch.from_numpy(graph.data.astype(np.float32)),
            num_nodes=level_pat.shape[0],
        ).numpy()
        new_id = _relabel_consecutive(cluster)
        if int(new_id.max()) + 1 == level_pat.shape[0]:
            break                       # no progress this level
        labels = new_id[labels]

        n_new = int(new_id.max()) + 1
        contraction = sp.csr_matrix(
            (np.ones(new_id.size, dtype=np.int64), (new_id, np.arange(new_id.size))),
            shape=(n_new, new_id.size))
        level_pat = contraction @ level_pat
        level_pat.data[:] = 1

    return labels


def _metis_labels(pat: sp.csr_matrix, target: int, hub_degree: int,
                  topk: Optional[int]) -> np.ndarray:
    import torch
    import torch_sparse  # noqa: F401  -- registers torch.ops.torch_sparse

    n = pat.shape[0]
    nparts = max(1, int(np.ceil(n / max(target, 1))))
    if nparts <= 1:
        return np.zeros(n, dtype=np.int64)

    graph = _similarity_graph(pat, hub_degree, topk).tocsr()
    cluster = torch.ops.torch_sparse.partition(
        torch.from_numpy(graph.indptr.astype(np.int64)),
        torch.from_numpy(graph.indices.astype(np.int64)),
        torch.from_numpy(graph.data.astype(np.int64)),
        nparts,
        False,
    ).numpy()
    return cluster.astype(np.int64)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def cluster_axis(mat: sp.spmatrix, method: str = 'hem', target: int = 32,
                 max_size: Optional[int] = None, max_nnz: Optional[int] = None,
                 seed: int = 0, hub_degree: int = DEFAULT_HUB_DEGREE,
                 topk: Optional[int] = DEFAULT_TOPK,
                 exact_recompute: bool = True, feasible_first: bool = False,
                 hem_hierarchy: Optional[list] = None) -> np.ndarray:
    """Cluster the ROWS of `mat` into groups of about `target` rows.

    To cluster columns instead, pass ``mat.T``.  Returns an int32 label array of
    length ``mat.shape[0]``; caps are enforced on the result regardless of method.
    Pass an empty list as ``hem_hierarchy`` with method='hem' to record each
    matching level, finest to coarsest, in original vertex coordinates. Final
    cap enforcement/relabeling is represented by the returned labels separately.
    """
    if hem_hierarchy is not None and (method != 'hem' or len(hem_hierarchy)):
        raise ValueError('hem_hierarchy must be an empty list with method=hem')
    if method not in CLUSTER_METHODS:
        raise ValueError(f'unknown clustering method {method!r}; '
                         f'expected one of {CLUSTER_METHODS}')

    pat = _pattern(mat)
    n = pat.shape[0]
    row_nnz = np.diff(pat.indptr).astype(np.int64)

    if n == 0:
        if hem_hierarchy is not None:
            hem_hierarchy.append(np.zeros(0, dtype=np.int64))
        return np.zeros(0, dtype=np.int32)

    if method == 'identity':
        labels = np.arange(n, dtype=np.int64) // max(target, 1)
    elif method == 'random':
        labels = _tile_by_permutation(
            np.random.default_rng(seed).permutation(n), n, target)
    elif method == 'degree':
        labels = _tile_by_permutation(
            np.argsort(row_nnz, kind='stable'), n, target)
    elif method == 'rcm':
        if pat.shape[0] != pat.shape[1]:
            raise ValueError("cluster method 'rcm' requires a square matrix")
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        symmetric = (pat + pat.T).tocsr()
        symmetric.data[:] = 1
        perm = reverse_cuthill_mckee(symmetric, symmetric_mode=True).astype(np.int64)
        labels = _tile_by_permutation(perm, n, target)
    elif method == 'graclus':
        try:
            labels = _graclus_labels(pat, target, hub_degree, topk)
        except ImportError:
            warnings.warn("torch_cluster unavailable; falling back to 'hem'", RuntimeWarning)
            labels = _hem_labels(pat, target, max_size, max_nnz, hub_degree, topk,
                                 exact_recompute, feasible_first=feasible_first)
    elif method == 'metis':
        try:
            labels = _metis_labels(pat, target, hub_degree, topk)
        except (ImportError, RuntimeError, AttributeError):
            warnings.warn("torch_sparse METIS unavailable; falling back to 'hem'",
                          RuntimeWarning)
            labels = _hem_labels(pat, target, max_size, max_nnz, hub_degree, topk,
                                 exact_recompute, feasible_first=feasible_first)
    else:  # 'hem'
        labels = _hem_labels(pat, target, max_size, max_nnz, hub_degree, topk,
                             exact_recompute, feasible_first=feasible_first,
                             hierarchy=hem_hierarchy)

    labels = _enforce_caps(labels, row_nnz, max_size, max_nnz)
    return _relabel_consecutive(labels).astype(np.int32)


def cluster_axes(mat: sp.spmatrix, method: str = 'hem',
                 target_nr: int = 32, target_nc: int = 32,
                 max_nr: Optional[int] = None, max_nc: Optional[int] = None,
                 max_block_nnz: Optional[int] = None, seed: int = 0,
                 symmetric: Optional[bool] = None,
                 hub_degree: int = DEFAULT_HUB_DEGREE,
                 topk: Optional[int] = DEFAULT_TOPK,
                 exact_recompute: bool = True,
                 feasible_first: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster both axes, returning ``(row_labels, col_labels)``.

    When the pattern is symmetric and the two targets agree, `A` is a graph and a
    single vertex clustering is used for both axes -- cheaper, and the right
    model (BOO_FORMAT sec. 5).  Otherwise rows come from ``A A^T`` and columns
    from ``A^T A``.  Pass `symmetric` to override the detection.
    """
    csr = sp.csr_matrix(mat)

    if symmetric is None:
        symmetric = (target_nr == target_nc) and is_pattern_symmetric(csr)

    row_labels = cluster_axis(csr, method=method, target=target_nr,
                              max_size=max_nr, max_nnz=max_block_nnz, seed=seed,
                              hub_degree=hub_degree, topk=topk,
                              exact_recompute=exact_recompute, feasible_first=feasible_first)

    if symmetric:
        return row_labels, row_labels.copy()

    col_labels = cluster_axis(csr.T.tocsr(), method=method, target=target_nc,
                              max_size=max_nc, max_nnz=max_block_nnz, seed=seed,
                              hub_degree=hub_degree, topk=topk,
                              exact_recompute=exact_recompute, feasible_first=feasible_first)
    return row_labels, col_labels
