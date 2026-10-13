"""Shared case loading and launch accounting for the BOO P x FPT x slab experiments.

Cases reuse the unified full-sweep labels (benchmark.bench_real label schema,
hash-validated against the canonical matrix) and the real-panel loader. Generated
inputs are registered through scripts/research_panel/full_sweep.register_generated,
so both families share bench_real's float64 reference and explicit-zero policy
(keep). Nothing here clusters: a missing or mismatched label file is an error.

The row-segmented launcher's FPT and slab width are process-wide. Every BOO entry
point created here applies both immediately before the call, so interleaved
variants can never inherit another variant's setting.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import numpy as np

from benchmark import bench_real as br
from benchmark import real_matrices as rm

DEFAULT_SLAB = 512


@dataclass(frozen=True)
class Case:
    name: str
    kind: str                 # 'real' | 'generated'
    source: str               # real: catalog key; generated: profile
    labeling: str             # e.g. identity_k32
    size: Optional[int] = None
    role: str = 'tuning'      # 'tuning' | 'scale' | 'heldout'

    @property
    def method(self):
        return self.labeling.rsplit('_k', 1)[0]

    @property
    def k(self):
        return int(self.labeling.rsplit('_k', 1)[1])


CASES = {c.name: c for c in (
    Case('pwtk', 'real', 'Boeing/pwtk', 'identity_k32'),
    Case('pwtk_hem', 'real', 'Boeing/pwtk', 'hem_k32'),
    Case('webbase', 'real', 'Williams/webbase-1M', 'identity_k32'),
    Case('roadnet', 'real', 'SNAP/roadNet-CA', 'metis_k64'),
    Case('diag_b1.0', 'generated', 'diag_b1.0', 'identity_k32', 262144),
    Case('fem2d', 'generated', 'fem2d', 'hem_k32', 262144),
    Case('fem2d_permuted', 'generated', 'fem2d_permuted', 'hem_k32', 262144),
    # Scale controls (labels from the full sweep, never re-clustered).
    Case('diag_b1.0_1m', 'generated', 'diag_b1.0', 'identity_k32', 1048576, 'scale'),
    Case('fem2d_1m', 'generated', 'fem2d', 'hem_k32', 1048576, 'scale'),
    Case('fem2d_permuted_1m', 'generated', 'fem2d_permuted', 'hem_k32', 1048576, 'scale'),
    # Held out of all tuning: sparse noisy diagonal, the full sweep's F512 labeling.
    Case('diag_dd0.05', 'generated', 'diag_dd0.05', 'identity_k64', 262144, 'heldout'),
)}
TUNING_CASES = ('pwtk', 'webbase', 'roadnet', 'diag_b1.0', 'fem2d', 'fem2d_permuted')


def resolve_case(case):
    """Register a generated input in this process if needed; return its MatrixSpec."""
    if case.kind == 'real':
        spec, = rm.resolve([case.source])
        return spec
    from scripts.research_panel.full_sweep import register_generated
    key = f'generated/{case.source}_n{case.size}'
    if key not in rm.CATALOG:
        register_generated(dict(profile=case.source, size=case.size))
    return rm.CATALOG[key]


def load_case(case, cache, labels, *, threshold=4, seed=0, explicit_zeros='keep'):
    """Canonical CSR + saved labels + BOO (rowmajor, remainder policy). No clustering."""
    import torch_fast_spdmm as spmm
    spec = resolve_case(case)
    csr, record = rm.load(spec, cache, explicit_zeros)
    rows, cols, provenance = br.load_labels(
        br.label_path(labels, spec, explicit_zeros, case.method, case.k),
        matrix=spec.key, matrix_sha256=record['matrix_sha256'], shape=csr.shape,
        explicit_zeros=explicit_zeros, method=case.method, k=case.k, seed=seed)
    boo = spmm.csr_to_boo(csr, row_labels=rows, col_labels=cols, small_block_policy='remainder',
                          small_block_threshold=threshold, nz_order='rowmajor')
    np.testing.assert_array_equal(np.asarray(boo.row_ids), br.permutation_from_labels(rows))
    np.testing.assert_array_equal(np.asarray(boo.col_ids), br.permutation_from_labels(cols))
    info = dict(case=case.name, role=case.role, matrix=spec.key, labeling=case.labeling,
                cluster=case.method, k=case.k, rows=csr.shape[0], cols=csr.shape[1],
                stored_entries=int(csr.nnz), numerical_nnz=record['numerical_nnz'],
                matrix_sha256=record['matrix_sha256'], label_sha256=provenance['label_sha256'],
                labels_path=provenance['labels_path'], threshold=threshold, seed=seed,
                explicit_zeros=explicit_zeros, num_blocks=int(len(boo.blk_row_grp)),
                max_nr=int(boo.max_nr), max_nc=int(boo.max_nc), max_nnz_blk=int(boo.max_nnz_blk),
                compressed_nnz=int(boo.compressed_nnz()), remainder_nnz=int(boo.remainder_nnz()))
    return csr, boo, info


def apply_settings(core, fpt, slab):
    core.set_boo_feat_block(slab)
    core.set_boo_rowseg_fpt(fpt)


def launch_info(core, boo, features, threads, p, fpt, slab, limit):
    """Exact launch the auto (no-plan) path selects: arena from the global maxima.

    Returns (info, unsupported_reason). Infeasible requests are reported, never
    silently replaced by another P/slab.
    """
    if slab < p or slab % p:
        return None, f'slab {slab} is not a multiple of requested P{p}'
    apply_settings(core, fpt, slab)
    try:
        info = dict(core.boo_rowseg_launch_info(int(boo.max_nr), int(boo.max_nc),
                                                int(boo.max_nnz_blk), features, threads, p))
    except (ValueError, RuntimeError) as exc:
        return None, f'launcher rejects: {exc}'
    finally:
        apply_settings(core, 1, DEFAULT_SLAB)
    if info['effective_p'] > slab or slab % info['effective_p']:
        return info, f"effective P{info['effective_p']} does not tile slab {slab}"
    if info['arena_bytes'] > limit:
        return info, f"rowseg arena {info['arena_bytes']} B exceeds opt-in limit {limit} B"
    if info['max_active_blocks_per_sm'] < 1:
        return info, 'occupancy query reports 0 resident CTAs'
    return info, None


KERNEL_RE = re.compile(r'boo_spmm_rowseg_kernel<float, (\d+), (\d+), (\w+), (\d+), (\d+)>')


def expected_kernel(info):
    p = info['effective_p']
    return (info['threads'], p.bit_length() - 1, info['features_per_thread'], info['feat_block'])


def launched_kernels(torch, fn):
    """Names of CUDA kernels one call launches (torch.profiler/Kineto), or None."""
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            y = fn()
            torch.cuda.synchronize()
        del y
        return [e.name for e in prof.events() if str(e.device_type).endswith('CUDA')]
    except Exception as exc:  # profiler unavailable: record, do not guess
        return [f'UNVERIFIED: {type(exc).__name__}: {exc}']


def verify_launched(names, info):
    """'verified' if exactly the expected rowseg specialization ran; else a reason."""
    if names and names[0].startswith('UNVERIFIED'):
        return names[0]
    found = set()
    for name in names:
        m = KERNEL_RE.search(name)
        if m:
            t, pl, diag, fpt, slab = m.groups()
            if diag not in ('false', '0'):
                return f'diagnostic raw-store kernel launched: {name}'
            found.add((int(t), int(pl), int(fpt), int(slab)))
    want = expected_kernel(info)
    if found == {want}:
        return 'verified'
    if not found:  # Kineto occasionally drops a capture in long processes
        return f'UNVERIFIED: profiler captured no rowseg event ({len(names)} CUDA events)'
    return f'expected rowseg<T,PL,FPT,slab>={want}, launched {sorted(found)}'


def verify_call(torch, fn, info, attempts=8):
    status = ''
    for _ in range(attempts):
        status = verify_launched(launched_kernels(torch, fn), info)
        if not status.startswith('UNVERIFIED'):
            return status
    return status


def warp_row_efficiency(boo, lanes):
    """Fraction of Step-B lane-iterations doing useful work when a warp spans rows.

    A warp covers 32 consecutive (row, lane) items, i.e. max(1, 32 // lanes)
    consecutive local rows; it iterates for the longest of them. Returns
    sum(row nnz) / sum(group max * rows per group) over all blocks, plus the
    fraction of nonzeros in multi-row warps. Rows with no nonzero in the block
    are included (they idle). lanes >= 32 gives 1.0 (no cross-row divergence).
    """
    g = max(1, 32 // lanes)
    rp, br_, ptr = np.asarray(boo.row_grp_ptr), np.asarray(boo.blk_row_grp), np.asarray(boo.blk_nz_ptr)
    idx = np.asarray(boo.nz_idx, dtype=np.uint16).astype(np.uint32)
    shifts = np.asarray(boo.blk_shift).astype(np.uint32)
    nr = np.diff(rp)[br_]
    nblk = len(br_)
    if g == 1 or not nblk:
        return dict(rows_per_warp=g, lane_efficiency=1.0)
    blk_of_nz = np.repeat(np.arange(nblk), np.diff(ptr))
    local_row = idx >> shifts[blk_of_nz]
    # Global (block, local row) slot ids, with each block padded to a multiple of g.
    padded = ((nr + g - 1) // g) * g
    base = np.concatenate([[0], np.cumsum(padded)[:-1]])
    lengths = np.bincount(base[blk_of_nz] + local_row, minlength=int(padded.sum()))
    # Padded slots beyond n_r are not executed: mark them so they don't count.
    valid = np.zeros(int(padded.sum()), dtype=bool)
    nr64 = nr.astype(np.int64)
    within = np.arange(int(nr64.sum())) - np.repeat(np.cumsum(nr64) - nr64, nr64)
    valid[np.repeat(base, nr64) + within] = True
    groups = lengths.reshape(-1, g)
    vgroups = valid.reshape(-1, g)
    worst = groups.max(axis=1)
    issued = (worst[:, None] * vgroups).sum()
    useful = groups.sum()
    return dict(rows_per_warp=g, lane_efficiency=float(useful / issued) if issued else 1.0)
