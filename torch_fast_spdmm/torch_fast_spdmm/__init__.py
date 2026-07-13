import types
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import torch
import torch_fast_spdmm_core as core

spdmm_csr_naive = core.spdmm_csr_naive

pbr_batched_matmul_cpu = core.pbr_batched_matmul_cpu


def _pin_to_device(arr, device, dtype=None):
    return torch.from_numpy(np.asarray(arr, dtype=dtype)).pin_memory().to(device, non_blocking=True)


def _reorder_pbr_arrays(block_codes_np, block_offsets_np, block_data_np, order):
    """Permute a PBR block array (codes/offsets/data) by `order`, recomputing
    block_offsets (a CSR-style prefix sum into block_data) from the permuted
    codes' popcounts. Shared by colgroup's col-sort and the tile kernel's
    per-shape tile-sort -- both need the same "reorder blocks, repack data" step."""
    if len(order) == 0:
        return block_codes_np.astype(np.int64), block_offsets_np, block_data_np

    codes_out = block_codes_np[order].astype(np.int64)
    popcounts = np.array([bin(int(c)).count('1') for c in block_codes_np], dtype=np.int64)
    lengths_out = popcounts[order]
    offsets_out = np.concatenate([[0], np.cumsum(lengths_out[:-1])]).astype(block_offsets_np.dtype)

    data_out = np.empty(len(block_data_np), dtype=block_data_np.dtype)
    for k, orig_i in enumerate(order):
        src = int(block_offsets_np[orig_i])
        n = int(popcounts[orig_i])
        dst = int(offsets_out[k])
        data_out[dst:dst + n] = block_data_np[src:src + n]

    return codes_out, offsets_out, data_out


def _pbr_to_method(self, device='cuda'):
    target_device = 'cpu' if device is None else str(torch.device(device))

    existing_meta = getattr(self, '_gpu_meta', None)
    current_device = str(existing_meta.codes.device) if existing_meta else 'cpu'

    if current_device == target_device:
        return self

    if target_device != 'cpu':
        # Build col-sorted permutation of PBR blocks for the colgroup kernel.
        # Col-major order: sort by (col_origin, row_origin) so consecutive blocks
        # in the same column band can share smem_X across col_origin runs.
        block_codes_np   = np.asarray(self.block_codes)
        block_coords_np  = np.asarray(self.block_coords)
        block_offsets_np = np.asarray(self.block_offsets)
        block_data_np    = np.asarray(self.block_data)

        n_blocks = len(block_codes_np)
        if n_blocks > 0:
            rows_arr   = block_coords_np[0::2]
            cols_arr   = block_coords_np[1::2]
            col_order  = np.lexsort((rows_arr, cols_arr))  # primary: col, secondary: row

            col_codes_np, col_offsets_np, col_data_np = _reorder_pbr_arrays(
                block_codes_np, block_offsets_np, block_data_np, col_order)
            col_coords_np = block_coords_np.reshape(-1, 2)[col_order].flatten()
        else:
            col_codes_np   = block_codes_np.astype(np.int64)
            col_coords_np  = block_coords_np
            col_offsets_np = block_offsets_np
            col_data_np    = block_data_np

        self._gpu_meta = types.SimpleNamespace(
            codes       = _pin_to_device(self.block_codes,      device, dtype=np.int64),
            coords      = _pin_to_device(self.block_coords,     device),
            offsets     = _pin_to_device(self.block_offsets,    device),
            data        = _pin_to_device(self.block_data,       device),
            col_codes   = _pin_to_device(col_codes_np,          device, dtype=np.int64),
            col_coords  = _pin_to_device(col_coords_np,         device),
            col_offsets = _pin_to_device(col_offsets_np,        device),
            col_data    = _pin_to_device(col_data_np,           device),
            rem_indptr  = _pin_to_device(self.remainder_indptr,  device),
            rem_col_ind = _pin_to_device(self.remainder_col_ind, device),
            rem_vals    = _pin_to_device(self.remainder_data,    device),
            tile_cache  = {},
        )
        torch.cuda.synchronize(device)
    else:
        self._gpu_meta = None

    return self


def _pbr_repr(self):
    return (
        f"PBRMatrix({self.rows}x{self.cols}, "
        f"blocks={self.accounted_blocks()}, "
        f"compressed_nnz={self.compressed_nnz()}, "
        f"remainder_nnz={self.remainder_nnz()}, "
        f"total_nnz={self.total_nnz})"
    )


# --- Inject the method into the C++ classes ---
for cls_name in ["PBRMatrixInt64Float", "PBRMatrixInt32Float",
                 "PBRMatrixInt64Double", "PBRMatrixInt32Double"]:
    if hasattr(core, cls_name):
        setattr(getattr(core, cls_name), "to", _pbr_to_method)
        setattr(getattr(core, cls_name), "__repr__", _pbr_repr)
        
    
def _build_cuda_dispatch():
    return {
        (torch.int32, torch.float32): core.pbr_spmm_cuda_i32_f32,
        (torch.int64, torch.float32): core.pbr_spmm_cuda_i64_f32,
        (torch.int32, torch.float64): core.pbr_spmm_cuda_i32_f64,
        (torch.int64, torch.float64): core.pbr_spmm_cuda_i64_f64,
    }

def _build_chunk_cuda_dispatch():
    return {
        (torch.int32, torch.float32): core.pbr_chunk_spmm_cuda_i32_f32,
        (torch.int64, torch.float32): core.pbr_chunk_spmm_cuda_i64_f32,
        (torch.int32, torch.float64): core.pbr_chunk_spmm_cuda_i32_f64,
        (torch.int64, torch.float64): core.pbr_chunk_spmm_cuda_i64_f64,
    }

def _build_rowgroup_cuda_dispatch():
    return {
        (torch.int32, torch.float32): core.pbr_rowgroup_spmm_cuda_i32_f32,
        (torch.int64, torch.float32): core.pbr_rowgroup_spmm_cuda_i64_f32,
        (torch.int32, torch.float64): core.pbr_rowgroup_spmm_cuda_i32_f64,
        (torch.int64, torch.float64): core.pbr_rowgroup_spmm_cuda_i64_f64,
    }

def _build_colgroup_cuda_dispatch():
    return {
        (torch.int32, torch.float32): core.pbr_colgroup_spmm_cuda_i32_f32,
        (torch.int64, torch.float32): core.pbr_colgroup_spmm_cuda_i64_f32,
        (torch.int32, torch.float64): core.pbr_colgroup_spmm_cuda_i32_f64,
        (torch.int64, torch.float64): core.pbr_colgroup_spmm_cuda_i64_f64,
    }

_CUDA_KERNEL_DISPATCH = None
_CHUNK_CUDA_KERNEL_DISPATCH = None
_ROWGROUP_CUDA_KERNEL_DISPATCH = None
_COLGROUP_CUDA_KERNEL_DISPATCH = None


def pbr_batched_matmul_cuda(pbr_mat, x: torch.Tensor, csr_mode: int = 0, smem_size: int = 2048) -> torch.Tensor:
    """
    Python wrapper that routes the GPU tensors to the correct PyBind11 CUDA function.
    csr_mode:  0=auto, 1=force_fast, 2=force_fallback
    smem_size: shared-memory tile size (floats) for the PBR block kernel; must be
               one of 512, 1024, 2048.
    """
    meta = getattr(pbr_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")

    global _CUDA_KERNEL_DISPATCH
    if _CUDA_KERNEL_DISPATCH is None:
        _CUDA_KERNEL_DISPATCH = _build_cuda_dispatch()

    kernel = _CUDA_KERNEL_DISPATCH[(meta.coords.dtype, x.dtype)]
    return kernel(
        pbr_mat.cols,
        pbr_mat.rows,
        pbr_mat.block_rows,
        pbr_mat.block_cols,
        meta.codes,
        meta.coords,
        meta.offsets,
        meta.data,
        meta.rem_indptr,
        meta.rem_col_ind,
        meta.rem_vals,
        x,
        csr_mode,
        smem_size,
    )


def pbr_batched_matmul_chunk_cuda(pbr_mat, x: torch.Tensor, chunk_size: int = 8,
                                   csr_mode: int = 0, smem_size: int = 2048) -> torch.Tensor:
    """
    Diagnostic-only variant: identical per-block body to the plain 'block' kernel,
    but CHUNK PBR blocks are processed per CUDA block via a #pragma-unrolled loop
    instead of one -- no smem reuse across the loop, no extra sync/flush logic.
    Isolates the CTA-count/occupancy effect from everything rowgroup/colgroup/tile
    also change. See WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md.
    chunk_size: PBR blocks processed per CUDA block; dispatched to {2,4,8,16,32,64}.
    csr_mode:  0=auto, 1=force_fast, 2=force_fallback (for the CSR remainder)
    smem_size: 512 / 1024 / 2048 floats (same per-block smem split as 'block')
    """
    meta = getattr(pbr_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")

    global _CHUNK_CUDA_KERNEL_DISPATCH
    if _CHUNK_CUDA_KERNEL_DISPATCH is None:
        _CHUNK_CUDA_KERNEL_DISPATCH = _build_chunk_cuda_dispatch()

    kernel = _CHUNK_CUDA_KERNEL_DISPATCH[(meta.coords.dtype, x.dtype)]
    return kernel(
        pbr_mat.cols,
        pbr_mat.rows,
        pbr_mat.block_rows,
        pbr_mat.block_cols,
        chunk_size,
        meta.codes,
        meta.coords,
        meta.offsets,
        meta.data,
        meta.rem_indptr,
        meta.rem_col_ind,
        meta.rem_vals,
        x,
        csr_mode,
        smem_size,
    )


def pbr_batched_matmul_rowgroup_cuda(pbr_mat, x: torch.Tensor, csr_mode: int = 0, smem_size: int = 2048) -> torch.Tensor:
    """
    Row-group fusion variant: one CUDA block per block-count-balanced group.
    Partitions PBR blocks uniformly by count; no CPU scan needed.
    Writes Y via atomicAdd (safe for concurrent CSR on a separate stream).
    csr_mode:  0=auto, 1=force_fast, 2=force_fallback (for the CSR remainder)
    smem_size: 512 / 1024 / 2048 floats; determines p = smem_size / (block_rows + block_cols)
    """
    meta = getattr(pbr_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")

    global _ROWGROUP_CUDA_KERNEL_DISPATCH
    if _ROWGROUP_CUDA_KERNEL_DISPATCH is None:
        _ROWGROUP_CUDA_KERNEL_DISPATCH = _build_rowgroup_cuda_dispatch()

    num_pbr_blocks = int(meta.codes.shape[0])
    if num_pbr_blocks > 0:
        device = meta.codes.device
        sm_count  = torch.cuda.get_device_properties(device).multi_processor_count
        bps       = core.pbr_rowgroup_blocks_per_sm(pbr_mat.block_rows, smem_size)
        target_groups = sm_count * bps
        group_size = max(1, (num_pbr_blocks + target_groups - 1) // target_groups)
        num_groups = (num_pbr_blocks + group_size - 1) // group_size
    else:
        num_groups = 0
        group_size = 1

    kernel = _ROWGROUP_CUDA_KERNEL_DISPATCH[(meta.coords.dtype, x.dtype)]
    return kernel(
        pbr_mat.cols,
        pbr_mat.rows,
        pbr_mat.block_rows,
        pbr_mat.block_cols,
        num_groups,
        group_size,
        meta.codes,
        meta.coords,
        meta.offsets,
        meta.data,
        meta.rem_indptr,
        meta.rem_col_ind,
        meta.rem_vals,
        x,
        csr_mode,
        smem_size,
    )


def pbr_batched_matmul_colgroup_cuda(pbr_mat, x: torch.Tensor, csr_mode: int = 0, smem_size: int = 2048) -> torch.Tensor:
    """
    Column-group fusion variant: PBR blocks pre-sorted by (col_origin, row_origin).
    Groups share smem_X across all blocks with the same col_origin, amortising
    the global X load for column-heavy sparsity patterns.
    csr_mode:  0=auto, 1=force_fast, 2=force_fallback (for the CSR remainder)
    smem_size: 512 / 1024 / 2048 floats; determines p = smem_size / (block_rows + block_cols)
    """
    meta = getattr(pbr_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")

    global _COLGROUP_CUDA_KERNEL_DISPATCH
    if _COLGROUP_CUDA_KERNEL_DISPATCH is None:
        _COLGROUP_CUDA_KERNEL_DISPATCH = _build_colgroup_cuda_dispatch()

    num_pbr_blocks = int(meta.col_codes.shape[0])
    if num_pbr_blocks > 0:
        device = meta.col_codes.device
        sm_count      = torch.cuda.get_device_properties(device).multi_processor_count
        bps           = core.pbr_colgroup_blocks_per_sm(pbr_mat.block_rows, smem_size)
        target_groups = sm_count * bps
        group_size    = max(1, (num_pbr_blocks + target_groups - 1) // target_groups)
        num_groups    = (num_pbr_blocks + group_size - 1) // group_size
    else:
        num_groups = 0
        group_size = 1

    kernel = _COLGROUP_CUDA_KERNEL_DISPATCH[(meta.col_coords.dtype, x.dtype)]
    return kernel(
        pbr_mat.cols,
        pbr_mat.rows,
        pbr_mat.block_rows,
        pbr_mat.block_cols,
        num_groups,
        group_size,
        meta.col_codes,
        meta.col_coords,
        meta.col_offsets,
        meta.col_data,
        meta.rem_indptr,
        meta.rem_col_ind,
        meta.rem_vals,
        x,
        csr_mode,
        smem_size,
    )


# ===========================================================================
# Tile ("block of blocks") kernel: pluggable PBR-block-to-tile grouping
# ===========================================================================
#
# A tile is a group of up to tile_rows x tile_cols PBR blocks sharing at most
# tile_rows distinct row-bands and tile_cols distinct col-bands, processed by
# one CUDA block with both smem_X and smem_Y resident (and flushed once) for
# the whole tile -- rowgroup/colgroup only ever reuse one of the two axes.
#
# Grouping strategy is decoupled from the kernel: any strategy function below
# only needs to produce a TileMeta (see _empty_tile_meta) satisfying this
# contract, and the CUDA kernel/bindings are entirely agnostic to how it was
# built. To add a strategy: write a builder with the signature
# (rows_arr, cols_arr, block_rows, block_cols, tile_rows, tile_cols) -> TileMeta
# and register it in _TILE_STRATEGIES.

def _empty_tile_meta():
    return types.SimpleNamespace(
        tile_block_order=np.empty(0, dtype=np.int64),
        tile_offsets=np.zeros(1, dtype=np.int64),
        tile_row_origins=np.empty(0, dtype=np.int64),
        tile_col_origins=np.empty(0, dtype=np.int64),
        tile_row_local=np.empty(0, dtype=np.uint8),
        tile_col_local=np.empty(0, dtype=np.uint8),
    )


def _finalize_tile_meta(rows_arr, cols_arr, tile_of_block, tile_rows, tile_cols):
    """Turn an arbitrary per-block tile-id labeling into the TileMeta contract:
    group blocks by tile_of_block, then dedupe each tile's row/col origins into
    local slots. Not vectorized (a per-tile np.unique) -- fine for 'greedy',
    which is already O(n_blocks x capacity) and Python-side; 'grid' has a
    fully-vectorized fast path instead since its slot assignment is arithmetic.
    """
    n_blocks = len(rows_arr)
    order = np.argsort(tile_of_block, kind='stable')
    sorted_tile_id = tile_of_block[order]

    change = np.empty(n_blocks, dtype=bool)
    change[0] = True
    change[1:] = sorted_tile_id[1:] != sorted_tile_id[:-1]
    tile_starts = np.flatnonzero(change)
    num_tiles = len(tile_starts)
    tile_offsets = np.append(tile_starts, n_blocks).astype(np.int64)

    tile_row_origins = np.full(num_tiles * tile_rows, -1, dtype=np.int64)
    tile_col_origins = np.full(num_tiles * tile_cols, -1, dtype=np.int64)
    tile_row_local = np.empty(n_blocks, dtype=np.uint8)
    tile_col_local = np.empty(n_blocks, dtype=np.uint8)

    sorted_rows = rows_arr[order]
    sorted_cols = cols_arr[order]

    for t in range(num_tiles):
        lo, hi = tile_offsets[t], tile_offsets[t + 1]
        uniq_r, r_local = np.unique(sorted_rows[lo:hi], return_inverse=True)
        uniq_c, c_local = np.unique(sorted_cols[lo:hi], return_inverse=True)
        if len(uniq_r) > tile_rows or len(uniq_c) > tile_cols:
            raise ValueError(
                f"tile {t} exceeds capacity: {len(uniq_r)} rows / {len(uniq_c)} cols "
                f"> capacity {tile_rows}x{tile_cols} -- tile_strategy bug")
        tile_row_origins[t * tile_rows: t * tile_rows + len(uniq_r)] = uniq_r
        tile_col_origins[t * tile_cols: t * tile_cols + len(uniq_c)] = uniq_c
        tile_row_local[lo:hi] = r_local.astype(np.uint8)
        tile_col_local[lo:hi] = c_local.astype(np.uint8)

    return types.SimpleNamespace(
        tile_block_order=order.astype(np.int64),
        tile_offsets=tile_offsets,
        tile_row_origins=tile_row_origins,
        tile_col_origins=tile_col_origins,
        tile_row_local=tile_row_local,
        tile_col_local=tile_col_local,
    )


def _build_tile_meta_grid(rows_arr, cols_arr, block_rows, block_cols, tile_rows, tile_cols):
    """Fixed grid partition: bucket PBR blocks by (row_band // tile_rows, col_band
    // tile_cols). Cheap (one lexsort, fully vectorized), but only catches reuse
    among blocks that already fall in the same fixed row/col-band window --
    e.g. a scattered pattern with populated bands {0, 2} under tile_rows=2 lands
    band 0 and band 2 in different tiles even though a 2-row tile could hold both.
    """
    n_blocks = len(rows_arr)
    if n_blocks == 0:
        return _empty_tile_meta()

    row_band = rows_arr // block_rows
    col_band = cols_arr // block_cols
    tile_row_id = row_band // tile_rows
    tile_col_id = col_band // tile_cols
    row_local_full = (row_band % tile_rows).astype(np.uint8)
    col_local_full = (col_band % tile_cols).astype(np.uint8)

    order = np.lexsort((tile_col_id, tile_row_id))
    s_tile_row = tile_row_id[order]
    s_tile_col = tile_col_id[order]

    change = np.empty(n_blocks, dtype=bool)
    change[0] = True
    change[1:] = (s_tile_row[1:] != s_tile_row[:-1]) | (s_tile_col[1:] != s_tile_col[:-1])
    tile_starts = np.flatnonzero(change)
    num_tiles = len(tile_starts)
    tile_offsets = np.append(tile_starts, n_blocks).astype(np.int64)

    tile_row_local = row_local_full[order]
    tile_col_local = col_local_full[order]

    tile_row_origins = np.full(num_tiles * tile_rows, -1, dtype=np.int64)
    tile_col_origins = np.full(num_tiles * tile_cols, -1, dtype=np.int64)

    tile_idx_per_block = np.repeat(np.arange(num_tiles), np.diff(tile_offsets))
    tile_row_origins[tile_idx_per_block * tile_rows + tile_row_local] = row_band[order] * block_rows
    tile_col_origins[tile_idx_per_block * tile_cols + tile_col_local] = col_band[order] * block_cols

    return types.SimpleNamespace(
        tile_block_order=order.astype(np.int64),
        tile_offsets=tile_offsets,
        tile_row_origins=tile_row_origins,
        tile_col_origins=tile_col_origins,
        tile_row_local=tile_row_local,
        tile_col_local=tile_col_local,
    )


def _build_tile_meta_greedy(rows_arr, cols_arr, block_rows, block_cols, tile_rows, tile_cols):
    """Seed-and-grow density-maximizing clustering: repeatedly seed a tile from
    an unassigned block, then greedily add whichever row-band or col-band
    (within capacity) covers the most additional unassigned blocks, until full
    or no beneficial candidate remains. Catches scattered/permuted reuse a
    fixed grid partition misses (e.g. populated row-bands {0, 2} both land in
    one tile if that's where the density is), at the cost of not being a
    single vectorized pass -- O(n_blocks x capacity) with plain Python sets.
    Not globally optimal (max-edge-biclique-cover is NP-hard); a bounded local
    heuristic is enough to beat a fixed grid on non-grid-aligned sparsity.
    """
    n_blocks = len(rows_arr)
    if n_blocks == 0:
        return _empty_tile_meta()

    row_band = (rows_arr // block_rows).astype(np.int64)
    col_band = (cols_arr // block_cols).astype(np.int64)

    blocks_by_row = defaultdict(set)
    blocks_by_col = defaultdict(set)
    for i in range(n_blocks):
        blocks_by_row[int(row_band[i])].add(i)
        blocks_by_col[int(col_band[i])].add(i)

    unassigned = set(range(n_blocks))
    tile_of_block = np.empty(n_blocks, dtype=np.int64)
    next_tile_id = 0

    for seed in range(n_blocks):
        if seed not in unassigned:
            continue

        row_set = {int(row_band[seed])}
        col_set = {int(col_band[seed])}

        while len(row_set) < tile_rows or len(col_set) < tile_cols:
            best_gain, best_kind, best_val = 0, None, None

            if len(row_set) < tile_rows:
                candidate_rows = set()
                for c in col_set:
                    candidate_rows.update(int(row_band[i]) for i in blocks_by_col[c] if i in unassigned)
                candidate_rows -= row_set
                for r in candidate_rows:
                    gain = sum(1 for i in blocks_by_row[r]
                               if i in unassigned and int(col_band[i]) in col_set)
                    if gain > best_gain:
                        best_gain, best_kind, best_val = gain, 'row', r

            if len(col_set) < tile_cols:
                candidate_cols = set()
                for r in row_set:
                    candidate_cols.update(int(col_band[i]) for i in blocks_by_row[r] if i in unassigned)
                candidate_cols -= col_set
                for c in candidate_cols:
                    gain = sum(1 for i in blocks_by_col[c]
                               if i in unassigned and int(row_band[i]) in row_set)
                    if gain > best_gain:
                        best_gain, best_kind, best_val = gain, 'col', c

            if best_kind is None:
                break
            elif best_kind == 'row':
                row_set.add(best_val)
            else:
                col_set.add(best_val)

        covered = {i for r in row_set for i in blocks_by_row[r]
                   if i in unassigned and int(col_band[i]) in col_set}
        if not covered:
            covered = {seed}

        for i in covered:
            tile_of_block[i] = next_tile_id
            unassigned.discard(i)
        next_tile_id += 1

    return _finalize_tile_meta(rows_arr, cols_arr, tile_of_block, tile_rows, tile_cols)


_TILE_STRATEGIES = {
    'grid':   _build_tile_meta_grid,
    'greedy': _build_tile_meta_greedy,
}


def _build_tile_meta(pbr_mat, tile_rows, tile_cols, tile_strategy, device):
    cache = pbr_mat._gpu_meta.tile_cache
    key = (tile_rows, tile_cols, tile_strategy)
    if key in cache:
        return cache[key]

    block_codes_np   = np.asarray(pbr_mat.block_codes)
    block_coords_np  = np.asarray(pbr_mat.block_coords)
    block_offsets_np = np.asarray(pbr_mat.block_offsets)
    block_data_np    = np.asarray(pbr_mat.block_data)
    n_blocks = len(block_codes_np)

    if n_blocks == 0:
        tile_meta = _empty_tile_meta()
    else:
        rows_arr = block_coords_np[0::2].astype(np.int64)
        cols_arr = block_coords_np[1::2].astype(np.int64)
        builder = _TILE_STRATEGIES[tile_strategy]
        tile_meta = builder(rows_arr, cols_arr, pbr_mat.block_rows, pbr_mat.block_cols, tile_rows, tile_cols)

    tile_block_codes_np, tile_block_offsets_np, tile_block_data_np = _reorder_pbr_arrays(
        block_codes_np, block_offsets_np, block_data_np, tile_meta.tile_block_order)

    index_dtype = block_offsets_np.dtype
    gpu_meta = types.SimpleNamespace(
        block_codes      = _pin_to_device(tile_block_codes_np,      device, dtype=np.int64),
        block_offsets    = _pin_to_device(tile_block_offsets_np,    device),
        block_data       = _pin_to_device(tile_block_data_np,       device),
        tile_offsets     = _pin_to_device(tile_meta.tile_offsets,     device, dtype=index_dtype),
        tile_row_origins = _pin_to_device(tile_meta.tile_row_origins, device, dtype=index_dtype),
        tile_col_origins = _pin_to_device(tile_meta.tile_col_origins, device, dtype=index_dtype),
        tile_row_local   = _pin_to_device(tile_meta.tile_row_local,   device, dtype=np.uint8),
        tile_col_local   = _pin_to_device(tile_meta.tile_col_local,   device, dtype=np.uint8),
    )
    cache[key] = gpu_meta
    return gpu_meta


def _build_tile_cuda_dispatch():
    return {
        (torch.int32, torch.float32): core.pbr_tile_spmm_cuda_i32_f32,
        (torch.int64, torch.float32): core.pbr_tile_spmm_cuda_i64_f32,
        (torch.int32, torch.float64): core.pbr_tile_spmm_cuda_i32_f64,
        (torch.int64, torch.float64): core.pbr_tile_spmm_cuda_i64_f64,
    }

_TILE_CUDA_KERNEL_DISPATCH = None


def pbr_batched_matmul_tile_cuda(pbr_mat, x: torch.Tensor, tile_rows: int = 2, tile_cols: int = 2,
                                  tile_strategy: str = 'grid', csr_mode: int = 0,
                                  smem_size: int = 2048) -> torch.Tensor:
    """
    Block-of-blocks (2D tile) fusion variant: one CUDA block per tile of up to
    tile_rows x tile_cols PBR blocks. Both smem_X and smem_Y are cached for the
    whole tile and flushed once at the end -- rowgroup/colgroup only reuse one
    of the two axes. tile_strategy picks how PBR blocks are grouped into tiles
    ('grid': fixed row/col-band partition; 'greedy': density-maximizing
    clustering -- see _TILE_STRATEGIES). Grid size is fully data-determined
    (one CUDA block per non-empty tile); no group_size/occupancy tuning needed.
    csr_mode:  0=auto, 1=force_fast, 2=force_fallback (for the CSR remainder)
    smem_size: 512 / 1024 / 2048 floats; p = smem_size / (tile_rows*block_rows + tile_cols*block_cols)
    """
    meta = getattr(pbr_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")
    if tile_strategy not in _TILE_STRATEGIES:
        raise ValueError(f"Unknown tile_strategy {tile_strategy!r}; choose from {list(_TILE_STRATEGIES)}")

    tile_meta = _build_tile_meta(pbr_mat, tile_rows, tile_cols, tile_strategy, meta.codes.device)

    global _TILE_CUDA_KERNEL_DISPATCH
    if _TILE_CUDA_KERNEL_DISPATCH is None:
        _TILE_CUDA_KERNEL_DISPATCH = _build_tile_cuda_dispatch()

    kernel = _TILE_CUDA_KERNEL_DISPATCH[(tile_meta.block_offsets.dtype, x.dtype)]
    return kernel(
        pbr_mat.cols,
        pbr_mat.rows,
        pbr_mat.block_rows,
        pbr_mat.block_cols,
        tile_rows,
        tile_cols,
        tile_meta.block_codes,
        tile_meta.block_offsets,
        tile_meta.block_data,
        tile_meta.tile_offsets,
        tile_meta.tile_row_origins,
        tile_meta.tile_col_origins,
        tile_meta.tile_row_local,
        tile_meta.tile_col_local,
        meta.rem_indptr,
        meta.rem_col_ind,
        meta.rem_vals,
        x,
        csr_mode,
        smem_size,
    )


def pbr_matmul(pbr_mat, x: torch.Tensor, csr_mode: int = 0, smem_size: int = 2048,
               kernel: str = 'block', tile_rows: int = 2, tile_cols: int = 2,
               tile_strategy: str = 'grid', chunk_size: int = 8) -> torch.Tensor:
    """
    Unified entry point for PBR MatMul.
    Dispatches to the correct backend based on x.device.
    csr_mode:  0=auto, 1=force_fast, 2=force_fallback (GPU only)
    smem_size: shared-memory tile size in floats; one of 512, 1024, 2048 (GPU only)
    kernel:    'block'    — one CUDA block per PBR block; atomicAdd to Y
               'rowgroup' — one CUDA block per block-count-balanced group; atomicAdd to Y
               'colgroup' — same as rowgroup but col-sorted; amortises smem_X per col band
               'tile'     — one CUDA block per tile_rows x tile_cols group of PBR blocks;
                            reuses both smem_X and smem_Y for the whole tile (see
                            pbr_batched_matmul_tile_cuda). tile_rows/tile_cols/tile_strategy
                            are only used when kernel='tile'.
               'chunk'    — diagnostic-only: identical per-block body to 'block', but
                            chunk_size PBR blocks per CUDA block via a #pragma-unrolled
                            loop, no smem reuse across the loop. Isolates the CTA-count
                            effect from rowgroup/colgroup/tile's added reuse machinery
                            (see WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md). chunk_size is only
                            used when kernel='chunk'.
    """
    if x.dim() not in (2, 3):
        raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

    if x.is_cuda:
        meta = getattr(pbr_mat, '_gpu_meta', None)
        if meta is None:
            raise RuntimeError("[pbr_matmul] X is on GPU but PBRMatrix is not. Call .to('cuda') first.")
        if kernel == 'rowgroup':
            y = pbr_batched_matmul_rowgroup_cuda(pbr_mat, x, csr_mode=csr_mode, smem_size=smem_size)
        elif kernel == 'colgroup':
            y = pbr_batched_matmul_colgroup_cuda(pbr_mat, x, csr_mode=csr_mode, smem_size=smem_size)
        elif kernel == 'tile':
            y = pbr_batched_matmul_tile_cuda(pbr_mat, x, tile_rows=tile_rows, tile_cols=tile_cols,
                                              tile_strategy=tile_strategy, csr_mode=csr_mode,
                                              smem_size=smem_size)
        elif kernel == 'chunk':
            y = pbr_batched_matmul_chunk_cuda(pbr_mat, x, chunk_size=chunk_size, csr_mode=csr_mode,
                                               smem_size=smem_size)
        else:
            y = pbr_batched_matmul_cuda(pbr_mat, x, csr_mode=csr_mode, smem_size=smem_size)
    else:
        y = core.pbr_batched_matmul_cpu(pbr_mat, x)

    return y.squeeze(0) if x.dim() == 2 else y


def csr_to_pbr(mat: sp.csr_matrix | torch.Tensor,
               block_rows: int = 8, block_cols: int = 8,
               min_nnz_per_block: int = 4):
    if isinstance(mat, torch.Tensor):
        if mat.device.type != 'cpu':
            raise ValueError('Input tensor must be on CPU')

        return core.csr_to_pbr(mat.crow_indices().numpy(),
                               mat.col_indices().numpy(),
                               mat.values().cpu().numpy(),
                               mat.shape[0], mat.shape[1],
                               block_rows, block_cols,
                               min_nnz_per_block)
    elif isinstance(mat, sp.csr_matrix):
        return core.csr_to_pbr(mat.indptr, mat.indices, mat.data,
                               mat.shape[0], mat.shape[1],
                               block_rows, block_cols,
                               min_nnz_per_block)
    else:
        raise TypeError(f'Unsupported type {type(mat)}')


_PBR_TYPE_MAP = {
    'PBRMatrixInt64Float':  (np.int64,  np.float32),
    'PBRMatrixInt32Float':  (np.int32,  np.float32),
    'PBRMatrixInt64Double': (np.int64,  np.float64),
    'PBRMatrixInt32Double': (np.int32,  np.float64),
}

def pbr_to_csr(pbr_mat) -> sp.csr_matrix:
    index_dtype, scalar_dtype = _PBR_TYPE_MAP[type(pbr_mat).__name__]
    indptr  = np.zeros(pbr_mat.rows + 1, dtype=index_dtype)
    indices = np.zeros(pbr_mat.total_nnz, dtype=index_dtype)
    data    = np.zeros(pbr_mat.total_nnz, dtype=scalar_dtype)
    core.pbr_to_csr(pbr_mat, indptr, indices, data)
    return sp.csr_matrix((data, indices, indptr), shape=(pbr_mat.rows, pbr_mat.cols))


def pbr_analyze_csr(mat: sp.csr_matrix | torch.Tensor,
                    block_rows: int = 8, block_cols: int = 8,
                    min_nnz_coverage=1000):
    if isinstance(mat, torch.Tensor):
        if mat.device.type != 'cpu':
            raise ValueError('Input tensor must be on CPU')

        return core.pbr_analyze_csr(mat.crow_indices().numpy(),
                                    mat.col_indices().numpy(),
                                    mat.shape[0], mat.shape[1], mat._nnz(),
                                    block_rows, block_cols,
                                    min_nnz_coverage)
    elif isinstance(mat, sp.csr_matrix):
        return core.pbr_analyze_csr(mat.indptr, mat.indices,
                                    mat.shape[0], mat.shape[1], mat.nnz,
                                    block_rows, block_cols,
                                    min_nnz_coverage)
    else:
        raise TypeError(f'Unsupported type {type(mat)}')


@dataclass(frozen=True)
class BlockFillStats:  # nnz count per compressed block (popcount of its 64-bit code)
    min: int
    max: int
    mean: float
    max_possible: int


@dataclass(frozen=True)
class BlockStripeStats:  # number of compressed blocks per horizontal row-stripe
    min: int
    max: int
    mean: float


@dataclass(frozen=True)
class PBRInfo:  # full structural summary of a built PBRMatrix
    shape: tuple
    block_size: tuple
    total_nnz: int
    compressed_nnz: int
    remainder_nnz: int
    compressed_fraction: float
    num_blocks: int
    num_unique_patterns: int
    avg_reuse: float
    block_fill: BlockFillStats
    blocks_per_stripe: BlockStripeStats


def pbr_info(pbr_mat) -> PBRInfo:
    """
    Returns a PBRInfo dataclass summarising a built PBRMatrix.

    Derived entirely from the already-exposed fields on pbr_mat — no rebuild
    or C++ call needed.  Useful for quick debugging and comparing the effect
    of different block sizes / min_nnz_per_block thresholds.
    Use dataclasses.asdict(info) to convert to a plain dict.
    """
    codes = np.asarray(pbr_mat.block_codes, dtype=np.uint64)
    num_blocks      = len(codes)
    total_nnz       = int(pbr_mat.total_nnz)
    compressed_nnz  = int(pbr_mat.compressed_nnz())
    remainder_nnz   = int(pbr_mat.remainder_nnz())

    if num_blocks > 0:
        popcounts = np.array([bin(int(c)).count('1') for c in codes], dtype=np.int32)
        unique_codes, _ = np.unique(codes, return_counts=True)
        num_unique = int(len(unique_codes))

        coords_flat = np.asarray(pbr_mat.block_coords)
        block_rows_abs = coords_flat.reshape(-1, 2)[:, 0]
        _, stripe_counts = np.unique(block_rows_abs // pbr_mat.block_rows, return_counts=True)

        block_fill = BlockFillStats(
            min=int(popcounts.min()),
            max=int(popcounts.max()),
            mean=float(popcounts.mean()),
            max_possible=pbr_mat.block_rows * pbr_mat.block_cols,
        )
        blocks_per_stripe = BlockStripeStats(
            min=int(stripe_counts.min()),
            max=int(stripe_counts.max()),
            mean=float(stripe_counts.mean()),
        )
    else:
        num_unique        = 0
        block_fill        = BlockFillStats(min=0, max=0, mean=0.0,
                                           max_possible=pbr_mat.block_rows * pbr_mat.block_cols)
        blocks_per_stripe = BlockStripeStats(min=0, max=0, mean=0.0)

    return PBRInfo(
        shape=(pbr_mat.rows, pbr_mat.cols),
        block_size=(pbr_mat.block_rows, pbr_mat.block_cols),
        total_nnz=total_nnz,
        compressed_nnz=compressed_nnz,
        remainder_nnz=remainder_nnz,
        compressed_fraction=compressed_nnz / total_nnz if total_nnz > 0 else 0.0,
        num_blocks=num_blocks,
        num_unique_patterns=num_unique,
        avg_reuse=num_blocks / num_unique if num_unique > 0 else 0.0,
        block_fill=block_fill,
        blocks_per_stripe=blocks_per_stripe,
    )


__all__ = [
    'spdmm_csr_naive',

    'csr_to_pbr',
    'pbr_to_csr',
    'pbr_analyze_csr',
    'pbr_info',
    'PBRInfo',
    'BlockFillStats',
    'BlockStripeStats',
    'pbr_matmul',
    'pbr_batched_matmul_cpu',
    'pbr_batched_matmul_cuda',
    'pbr_batched_matmul_chunk_cuda',
    'pbr_batched_matmul_rowgroup_cuda',
    'pbr_batched_matmul_colgroup_cuda',
    'pbr_batched_matmul_tile_cuda',
]