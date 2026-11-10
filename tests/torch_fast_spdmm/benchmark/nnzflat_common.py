"""Inputs, artifacts and launch accounting for the BOO nnzflat FPT1/2 experiment.

See experiments/2026-10-14_nnzflat/README.md. CPU preparation
(scripts/research_panel/nnzflat_prep.py) writes hash-keyed artifacts; GPU jobs
(benchmark.bench_nnzflat) only load and verify them:

  artifacts/inputs/<input>.npz       canonical CSR (generated + synthetic inputs)
  artifacts/labels/*.npz             bench_real label schema, hash-validated
  artifacts/boo/<input>__<labeling>.json   BOO digest, structure, work models
  artifacts/refs/<input>__F<F>.{ref,absref,tol}.npy + .json   float64 CPU reference

Real inputs are loaded from the canonical SuiteSparse cache (hash-verified by
benchmark.real_matrices). BOO itself is rebuilt from the saved labels on the GPU
node (it has no deserialization path) and must reproduce the CPU-job digest.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Optional

import numpy as np
import scipy.sparse as sp

from benchmark import bench_real as br
from benchmark import real_matrices as rm

PS = (8, 16, 32, 64, 128)
THREADS = (128, 256, 512)
FPTS = (1, 2)
SLAB = 512
THRESHOLD = 4
SEED = 0
RTX3090_OPTIN = 101376  # CPU-side estimate only; GPU jobs use the device attribute


@dataclass(frozen=True)
class Input:
    name: str
    kind: str              # 'real' | 'generated' | 'synthetic'
    source: str            # real: catalog key; generated: profile; synthetic: pattern
    method: str            # frozen labeling method
    size: Optional[int] = None
    role: str = ''

    @property
    def key(self):
        if self.kind == 'real':
            return self.source
        if self.kind == 'generated':
            return f'generated/{self.source}_n{self.size}'
        return f'synthetic/{self.source}_n{self.size}'


INPUTS = {i.name: i for i in (
    # Stage 1: controlled mechanism panel (contiguous labels coarsened from 16x16 blocks).
    Input('syn_balanced', 'synthetic', 'balanced', 'contiguous', 16384, 'balanced local rows'),
    Input('syn_skewed', 'synthetic', 'skewed', 'contiguous', 16384, 'skewed local rows'),
    # Stage 2: application screen, frozen labeling methods.
    Input('pwtk', 'real', 'Boeing/pwtk', 'identity', None, 'strong reuse / instruction pressure'),
    Input('webbase', 'real', 'Williams/webbase-1M', 'identity', None, 'sparse irregular / atomic control'),
    Input('roadnet', 'real', 'SNAP/roadNet-CA', 'metis', None, 'sparse blocks / remainder control'),
    Input('fem2d', 'generated', 'fem2d', 'hem', 262144, 'naturally ordered FEM'),
    Input('fem2d_permuted', 'generated', 'fem2d_permuted', 'hem', 262144, 'FEM, locality destroyed'),
    Input('diag_b1.0', 'generated', 'diag_b1.0', 'identity', 262144, 'dense-block / smem limit control'),
    Input('diag_dd0.05', 'generated', 'diag_dd0.05', 'identity', 262144, 'sparse noisy diagonal'),
    Input('diag_skewed', 'generated', 'diag_skewed', 'identity', 262144, 'skew / remainder stress'),
    # Stage 3 held-out scale checks.
    Input('fem2d_1m', 'generated', 'fem2d', 'hem', 1048576, 'scale check'),
    Input('fem2d_permuted_1m', 'generated', 'fem2d_permuted', 'hem', 1048576, 'scale check'),
    Input('diag_b1.0_1m', 'generated', 'diag_b1.0', 'identity', 1048576, 'scale check'),
    Input('diag_dd0.05_1m', 'generated', 'diag_dd0.05', 'identity', 1048576, 'scale check'),
    Input('diag_skewed_1m', 'generated', 'diag_skewed', 'identity', 1048576, 'scale check'),
)}
STAGE1_INPUTS = ('syn_balanced', 'syn_skewed')
STAGE1_K = (16, 32, 64, 128)
STAGE2_INPUTS = ('pwtk', 'webbase', 'roadnet', 'fem2d', 'fem2d_permuted', 'diag_b1.0',
                 'diag_dd0.05', 'diag_skewed')
STAGE2_K = (32, 64, 128)
SCALE_INPUTS = {'fem2d': 'fem2d_1m', 'fem2d_permuted': 'fem2d_permuted_1m',
                'diag_b1.0': 'diag_b1.0_1m', 'diag_dd0.05': 'diag_dd0.05_1m',
                'diag_skewed': 'diag_skewed_1m'}
SCREEN_F = (128, 512)


def spec_of(inp):
    if inp.kind == 'real':
        spec, = rm.resolve([inp.source])
        return spec
    group, name = inp.key.split('/')
    return rm.MatrixSpec(group, name, inp.role)


# ---------------------------------------------------------------------------
# Stage 1 synthetic patterns
# ---------------------------------------------------------------------------

BASE = 16            # base block edge
BAND = 2             # nonzero base blocks at |bi - bj| <= BAND
NNZ_PER_BASE = 64    # every nonzero base block holds exactly this many entries
SKEWED_LENGTHS = np.array([16, 16, 16, 2, 2, 2] + [1] * 10)  # sums to 64, max/mean 4x


def synthetic(pattern, n=16384, seed=SEED):
    """Blocked band with equal nnz per 16x16 base block; only local row lengths differ.

    balanced: every local row of a base block holds 4 entries.
    skewed:   local row lengths are SKEWED_LENGTHS (16,16,16,2,2,2,1x10), with the
              heavy rows chosen once per base block-row, so a heavy global row is
              heavy in every base block it touches (skew survives coarsening).
    Values are i.i.d. uniform(-1, 1) from the same stream for both patterns.
    No duplicates, no explicit zeros (values are redrawn away from 0).
    """
    if pattern not in ('balanced', 'skewed'):
        raise ValueError(pattern)
    rng = np.random.default_rng(seed)
    nb = n // BASE
    rows, cols = [], []
    for bi in range(nb):
        if pattern == 'balanced':
            lengths = np.full(BASE, NNZ_PER_BASE // BASE)
        else:
            lengths = SKEWED_LENGTHS[rng.permutation(BASE)]
        for bj in range(max(0, bi - BAND), min(nb, bi + BAND + 1)):
            for r, length in enumerate(lengths):
                c = rng.choice(BASE, size=int(length), replace=False)
                rows.append(np.full(int(length), bi * BASE + r))
                cols.append(bj * BASE + c)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    vrng = np.random.default_rng(seed + 1)
    v = vrng.uniform(-1.0, 1.0, r.size)
    v = np.where(np.abs(v) < 1e-3, 0.5, v).astype(np.float32)
    csr = sp.csr_matrix((v, (r, c)), shape=(n, n))
    if csr.nnz != r.size:
        raise AssertionError('synthetic generator produced duplicate coordinates')
    csr.sort_indices()
    csr.indptr = csr.indptr.astype(np.int32)
    csr.indices = csr.indices.astype(np.int32)
    return csr


def contiguous_labels(n, k):
    return (np.arange(n, dtype=np.int64) // k).astype(np.int32)


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def art(root):
    return Path(root) / 'artifacts'


def input_path(root, name):
    return art(root) / 'inputs' / f'{name}.npz'


def save_csr(path, csr, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, indptr=csr.indptr, indices=csr.indices, data=csr.data,
             shape=np.asarray(csr.shape, np.int64), record=json.dumps(record, sort_keys=True))
    tmp.replace(path)


def load_input(root, name, cache):
    """Canonical CSR + record. Real: SuiteSparse cache. Others: CPU-job artifact."""
    inp = INPUTS[name]
    if inp.kind == 'real':
        return rm.load(spec_of(inp), cache, 'keep')
    path = input_path(root, name)
    with np.load(path, allow_pickle=False) as f:
        record = json.loads(str(f['record']))
        csr = sp.csr_matrix((f['data'], f['indices'], f['indptr']), shape=tuple(f['shape']))
    csr.has_sorted_indices = True
    if rm.matrix_hash(csr) != record['matrix_sha256']:
        raise ValueError(f'{path}: content hash does not match its record')
    return csr, record


def labeling_name(inp, k):
    return f'{inp.method}_k{k}'


def label_file(root, inp, k):
    return br.label_path(art(root) / 'labels', spec_of(inp), 'keep', inp.method, k)


def load_input_labels(root, inp, k, csr, record):
    return br.load_labels(label_file(root, inp, k), matrix=inp.key,
                          matrix_sha256=record['matrix_sha256'], shape=csr.shape,
                          explicit_zeros='keep', method=inp.method, k=k, seed=SEED,
                          feasible_first=inp.method != 'contiguous')


def build_boo(csr, rows, cols):
    import torch_fast_spdmm as spmm
    boo = spmm.csr_to_boo(csr, row_labels=rows, col_labels=cols, small_block_policy='remainder',
                          small_block_threshold=THRESHOLD, nz_order='rowmajor')
    np.testing.assert_array_equal(np.asarray(boo.row_ids), br.permutation_from_labels(rows))
    np.testing.assert_array_equal(np.asarray(boo.col_ids), br.permutation_from_labels(cols))
    return boo


BOO_ARRAYS = ('row_grp_ptr', 'row_ids', 'col_grp_ptr', 'col_ids', 'blk_row_grp', 'blk_col_grp',
              'blk_nz_ptr', 'blk_shift', 'nz_idx', 'nz_val', 'rem_indptr', 'rem_col_ind', 'rem_vals',
              'tail_row', 'tail_col', 'tail_val')


def boo_digest(boo):
    return rm.digest_arrays(np.asarray([boo.max_nr, boo.max_nc, boo.max_nnz_blk], np.int64),
                            *(np.asarray(getattr(boo, a)) for a in BOO_ARRAYS))


def boo_artifact_path(root, name, labeling):
    return art(root) / 'boo' / f'{name}__{labeling}.json'


def ref_stem(root, name, features):
    return art(root) / 'refs' / f'{name}__F{features}'


def make_x(cols, features, seed=SEED):
    import torch
    return torch.randn(cols, features, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# structure and Step-B work models (CPU, from the built format)
# ---------------------------------------------------------------------------

def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def boo_structure(boo):
    """Block/row nnz distributions, within-block skew, active vs staged columns."""
    rgp = np.diff(np.asarray(boo.row_grp_ptr, np.int64))
    cgp = np.diff(np.asarray(boo.col_grp_ptr, np.int64))
    brg = np.asarray(boo.blk_row_grp, np.int64)
    bcg = np.asarray(boo.blk_col_grp, np.int64)
    ptr = np.asarray(boo.blk_nz_ptr, np.int64)
    nb = len(brg)
    total = int(boo.total_nnz)
    out = dict(num_blocks=nb, total_nnz=total, compressed_nnz=int(boo.compressed_nnz()),
               remainder_nnz=int(boo.remainder_nnz()), tail_nnz=int(boo.tail_nnz()),
               max_nr=int(boo.max_nr), max_nc=int(boo.max_nc), max_nnz_blk=int(boo.max_nnz_blk),
               num_row_groups=int(len(rgp)), num_col_groups=int(len(cgp)),
               row_group_mean=float(rgp.mean()) if len(rgp) else 0.0,
               row_group_max=int(rgp.max()) if len(rgp) else 0,
               col_group_mean=float(cgp.mean()) if len(cgp) else 0.0,
               col_group_max=int(cgp.max()) if len(cgp) else 0)
    out['remainder_fraction'] = out['remainder_nnz'] / total if total else 0.0
    if not nb:
        return out
    nr, nc, nz = rgp[brg], cgp[bcg], np.diff(ptr)
    idx = np.asarray(boo.nz_idx, dtype=np.uint16).astype(np.int64)
    shifts = np.asarray(boo.blk_shift).astype(np.int64)
    blk = np.repeat(np.arange(nb), nz)
    lr = idx >> shifts[blk]
    lc = idx & ((1 << shifts[blk]) - 1)
    row_base = np.concatenate([[0], np.cumsum(nr)[:-1]])
    row_len = np.bincount(row_base[blk] + lr, minlength=int(nr.sum()))
    nonempty = row_len > 0
    row_blk = np.repeat(np.arange(nb), nr)
    max_row = np.zeros(nb, np.int64)
    np.maximum.at(max_row, row_blk, row_len)
    active_rows = np.bincount(row_blk[nonempty], minlength=nb)
    col_base = np.concatenate([[0], np.cumsum(nc)[:-1]])
    col_used = np.zeros(int(nc.sum()), bool)
    col_used[col_base[blk] + lc] = True
    active_cols = np.bincount(np.repeat(np.arange(nb), nc)[col_used], minlength=nb)
    skew = max_row / np.maximum(nz / np.maximum(active_rows, 1), 1e-12)
    w = nz / nz.sum()
    out.update(
        nnz_per_block_mean=float(nz.mean()), nnz_per_block_p50=_pct(nz, 50),
        nnz_per_block_p95=_pct(nz, 95), nnz_per_block_max=int(nz.max()),
        block_rows_mean=float(nr.mean()), block_cols_mean=float(nc.mean()),
        row_nnz_in_block_p50=_pct(row_len[nonempty], 50),
        row_nnz_in_block_p95=_pct(row_len[nonempty], 95),
        row_nnz_in_block_max=int(row_len.max()),
        empty_block_row_fraction=float(1 - nonempty.mean()),
        block_skew_p50=_pct(skew, 50), block_skew_p95=_pct(skew, 95), block_skew_max=float(skew.max()),
        block_skew_nnz_weighted=float((w * skew).sum()),
        active_col_fraction=float(active_cols.sum() / nc.sum()),
        block_density_mean=float((nz / (nr * nc)).mean()),
        x_loads=int(nc.sum()), y_writes=int(nr.sum()))
    return out


def work_model(boo, features, p, threads, fpt):
    """Step-B work for nnzflat: per block z nonzeros, z*P/FPT items per full tile.

    Returns tile passes, work items, max loop trips per thread per tile, active
    thread fraction of the first iteration (nnz-weighted and block mean), barriers
    per CTA, shared atomics, and executed padding items from feature tails.
    """
    nz = np.diff(np.asarray(boo.blk_nz_ptr, np.int64))
    slabs = -(-features // SLAB)
    tiles = sum(-(-min(SLAB, features - s * SLAB) // p) for s in range(slabs))
    padded = tiles * p
    lanes = p // fpt
    items_tile = nz * lanes
    trips = -(-items_tile // threads)
    active = np.minimum(1.0, items_tile / threads)
    w = nz / max(nz.sum(), 1)
    return dict(tile_passes=int(tiles), padded_features=int(padded),
                padding_items=int(nz.sum() * (padded - features) // fpt),
                stepb_items=int(nz.sum() * padded // fpt),
                stepb_max_trips_p50=_pct(trips, 50), stepb_max_trips_max=int(trips.max()) if len(nz) else 0,
                active_fraction_mean=float(active.mean()) if len(nz) else 0.0,
                active_fraction_nnz_weighted=float((w * active).sum()),
                barriers_per_cta=int(1 + 3 * tiles), shared_atomics=int(nz.sum() * padded),
                ctas=int(len(nz) * slabs))


def arena_nnzflat(boo, p, scalar_bytes=4):
    nr, nc, z = int(boo.max_nr), int(boo.max_nc), int(boo.max_nnz_blk)
    return z * (scalar_bytes + 2) + 4 * (nr + nc) + scalar_bytes * p * (nr + nc) + 16


def arena_rowseg(boo, p, scalar_bytes=4):
    nr, nc, z = int(boo.max_nr), int(boo.max_nc), int(boo.max_nnz_blk)
    return z * (scalar_bytes + 2) + 4 * (nr + nc) + 4 * (nr + 1) + scalar_bytes * p * nc + 16


# ---------------------------------------------------------------------------
# arms, launch accounting, kernel verification (GPU)
# ---------------------------------------------------------------------------

def arm_name(kernel, p, threads, fpt, impl='template', suffix=''):
    if kernel == 'nnzflat':
        tag = 'nfref' if impl == 'reference' else 'nf'
        return f'{tag}_p{p}_t{threads}_f{fpt}{suffix}'
    return f'rs_p{p}_t{threads}_f{fpt}{suffix}'


ARM_RE = re.compile(r'^(nf|nfref|rs)_p(\d+)_t(\d+)_f(\d+)(_dup)?$')


def parse_arm(name):
    m = ARM_RE.match(name)
    if not m:
        raise ValueError(name)
    tag, p, t, f, dup = m.groups()
    return dict(kernel='rowseg' if tag == 'rs' else 'nnzflat',
                impl='reference' if tag == 'nfref' else 'template' if tag == 'nf' else '',
                p=int(p), threads=int(t), fpt=int(f), duplicate=bool(dup))


def apply_arm(core, arm):
    """Set every process-wide BOO setting the arm depends on (outside timing)."""
    core.set_boo_feat_block(SLAB)
    if arm['kernel'] == 'rowseg':
        core.set_boo_rowseg_fpt(arm['fpt'])
        core.set_boo_nnzflat_fpt(1)
        core.set_boo_nnzflat_reference(True)
    else:
        core.set_boo_rowseg_fpt(1)
        core.set_boo_nnzflat_fpt(arm['fpt'])
        core.set_boo_nnzflat_reference(arm['impl'] == 'reference')


def reset_settings(core):
    core.set_boo_feat_block(SLAB)
    core.set_boo_rowseg_fpt(1)
    core.set_boo_nnzflat_fpt(1)
    core.set_boo_nnzflat_reference(True)


def launch_info(core, boo, features, arm, limit, batch=1):
    """(info, reason) for exactly what the launcher selects; never substitutes."""
    apply_arm(core, arm)
    try:
        if arm['kernel'] == 'rowseg':
            info = dict(core.boo_rowseg_launch_info(int(boo.max_nr), int(boo.max_nc),
                                                    int(boo.max_nnz_blk), features,
                                                    arm['threads'], arm['p']))
            info.update(grid_x=int(boo.num_blocks()), grid_y=batch, reference=False)
        else:
            info = dict(core.boo_nnzflat_launch_info(int(boo.num_blocks()), batch, int(boo.max_nr),
                                                     int(boo.max_nc), int(boo.max_nnz_blk), features,
                                                     arm['threads'], arm['p'], 4))
    except (ValueError, RuntimeError) as exc:
        return None, f'launcher rejects: {exc}'
    finally:
        reset_settings(core)
    if info['features_per_thread'] != arm['fpt'] or info['threads'] != arm['threads']:
        return info, f"launcher selected FPT{info['features_per_thread']}/T{info['threads']}"
    if info['effective_p'] != arm['p']:
        return info, f"effective P{info['effective_p']} differs from requested P{arm['p']}"
    if info['feat_block'] != SLAB:
        return info, f"slab {info['feat_block']} differs from {SLAB}"
    if info['arena_bytes'] > limit:
        return info, (f"{arm['kernel']} arena {info['arena_bytes']} B exceeds the device "
                      f"opt-in limit {limit} B")
    if info['max_active_blocks_per_sm'] < 1:
        return info, 'occupancy query reports 0 resident CTAs/SM'
    return info, None


NF_RE = re.compile(r'boo_spmm_nnzflat_kernel<float, (\d+), (\d+)>')
NFT_RE = re.compile(r'boo_spmm_nnzflat_fpt_kernel<float, (\d+), (\d+), (\d+)>')
RS_RE = re.compile(r'boo_spmm_rowseg_kernel<float, (\d+), (\d+), (\w+), (\d+), (\d+)>')


def expected_kernel(arm):
    pl = arm['p'].bit_length() - 1
    if arm['kernel'] == 'rowseg':
        return f"boo_spmm_rowseg_kernel<float, {arm['threads']}, {pl}, false, {arm['fpt']}, {SLAB}>"
    if arm['impl'] == 'reference':
        return f"boo_spmm_nnzflat_kernel<float, {arm['threads']}, {pl}>"
    return f"boo_spmm_nnzflat_fpt_kernel<float, {arm['threads']}, {pl}, {arm['fpt']}>"


def normalize_kernel_name(name):
    """Kineto prints template arguments as `256, false`; nsys/CUPTI demangling as
    `(int)256, (bool)0`. Map both to the Kineto spelling used by expected_kernel."""
    name = re.sub(r'\(bool\)0\b', 'false', name)
    name = re.sub(r'\(bool\)1\b', 'true', name)
    return re.sub(r'\((?:int|unsigned int|unsigned)\)', '', name)


def classify(name):
    name = normalize_kernel_name(name)
    for kind, rx in (('nf', NF_RE), ('nft', NFT_RE), ('rs', RS_RE)):
        m = rx.search(name)
        if m:
            return m.group(0)
    return None


def launched_kernels(torch, fn):
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            y = fn()
            torch.cuda.synchronize()
        del y
        return [e.name for e in prof.events() if str(e.device_type).endswith('CUDA')]
    except Exception as exc:
        return [f'UNVERIFIED: {type(exc).__name__}: {exc}']


def verify_call(torch, fn, arm, attempts=8, expect_main=True):
    """'verified' iff exactly the expected BOO main kernel ran (CSR/tail legs allowed)."""
    want = expected_kernel(arm)
    status = ''
    for _ in range(attempts):
        names = launched_kernels(torch, fn)
        if names and names[0].startswith('UNVERIFIED'):
            status = names[0]
            continue
        found = {k for k in map(classify, names) if k}
        if found == {want}:
            return 'verified', sorted(found)
        if not found and not expect_main:
            return 'verified-no-main-kernel', []
        if not found:
            status = f'UNVERIFIED: no BOO main kernel captured ({len(names)} CUDA events)'
            continue
        return f'expected {want}, launched {sorted(found)}', sorted(found)
    return status, []


def sha256_file(path):
    return rm.file_sha256(path)


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()
