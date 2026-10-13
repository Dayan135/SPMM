"""Loader, label-validation and permutation tests for the real-matrix benchmark.

Run remotely from tests/torch_fast_spdmm:  python -m pytest benchmark/test_real_matrices.py -v
The end-to-end matmul test needs a GPU; everything else is CPU-only.
"""

import csv
import io
import json
import tarfile

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from benchmark import bench_real as br
from benchmark import real_matrices as rm


def write_mtx(path, body, header='matrix coordinate real general'):
    path.write_text(f'%%MatrixMarket {header}\n% comment\n{body}')
    return path


def load_text(tmp_path, body, header='matrix coordinate real general', policy='keep'):
    h, r, c, v = rm.read_mtx(write_mtx(tmp_path / 'a.mtx', body, header))
    return rm.canonicalize(h, r, c, v, policy)


# ---------------------------------------------------------------------------
# Matrix Market parsing and canonicalization
# ---------------------------------------------------------------------------

def test_general_rectangular_duplicates_and_zeros(tmp_path):
    # (1,1) twice -> summed; (2,3) explicit zero; (1,2)+(1,2) cancel to zero.
    csr, rec = load_text(tmp_path, '2 4 5\n1 1 1.5\n1 1 2.0\n2 3 0\n1 2 3\n1 2 -3\n')
    assert csr.shape == (2, 4)
    assert csr.indptr.dtype == csr.indices.dtype == np.int32 and csr.dtype == np.float32
    assert csr.toarray().tolist() == [[3.5, 0, 0, 0], [0, 0, 0, 0]]
    assert rec['file_entries'] == 5 and rec['expanded_entries'] == 5
    assert rec['duplicates_summed'] == 2
    assert rec['file_explicit_zeros'] == 1 and rec['zeros_after_duplicate_sum'] == 2
    assert rec['stored_entries'] == 3 and rec['numerical_nnz'] == 1  # zeros kept as slots
    dropped, rec_drop = load_text(tmp_path, '2 4 5\n1 1 1.5\n1 1 2.0\n2 3 0\n1 2 3\n1 2 -3\n',
                                  policy='drop')
    assert dropped.nnz == 1 and rec_drop['explicit_zeros_removed'] == 2
    np.testing.assert_array_equal(dropped.toarray(), csr.toarray())


def test_directed_general_matrix_is_not_symmetrized(tmp_path):
    csr, rec = load_text(tmp_path, '3 3 2\n1 2 1\n3 1 2\n')
    assert (csr != csr.T).nnz > 0
    assert csr[0, 1] == 1 and csr[1, 0] == 0
    assert rm.structure_stats(csr)['pattern_symmetric'] is False


def test_symmetric_expansion_mirrors_off_diagonal_only(tmp_path):
    csr, rec = load_text(tmp_path, '3 3 3\n1 1 4\n2 1 -1\n3 2 0\n', 'matrix coordinate real symmetric')
    expected = np.array([[4, -1, 0], [-1, 0, 0], [0, 0, 0]], np.float32)
    np.testing.assert_array_equal(csr.toarray(), expected)
    assert rec['expanded_entries'] == 5 and csr.nnz == 5  # explicit zero mirrored and kept
    stats = rm.structure_stats(csr)
    assert stats['pattern_symmetric'] and stats['numerically_symmetric']


def test_symmetric_file_with_upper_entry_raises(tmp_path):
    with pytest.raises(ValueError, match='upper-triangle'):
        load_text(tmp_path, '2 2 1\n1 2 1\n', 'matrix coordinate real symmetric')


def test_skew_symmetric_negates_and_forbids_diagonal(tmp_path):
    csr, _ = load_text(tmp_path, '2 2 1\n2 1 5\n', 'matrix coordinate real skew-symmetric')
    np.testing.assert_array_equal(csr.toarray(), [[0, -5], [5, 0]])
    with pytest.raises(ValueError, match='skew'):
        load_text(tmp_path, '2 2 1\n1 1 5\n', 'matrix coordinate real skew-symmetric')


def test_pattern_symmetric_becomes_binary(tmp_path):
    csr, rec = load_text(tmp_path, '3 3 2\n2 1\n3 3\n', 'matrix coordinate pattern symmetric')
    np.testing.assert_array_equal(csr.toarray(), [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert rec['mm_field'] == 'pattern' and rec['numerical_nnz'] == 3


@pytest.mark.parametrize('body,match', [
    ('2 2 3\n1 1 1\n2 2 1\n', 'expected 3 entries'),
    ('2 2 1\n3 1 1\n', 'out of range'),
    ('2 2 1\n1.5 1 1\n', 'non-integer'),
])
def test_malformed_files_raise(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_text(tmp_path, body)


def test_unsupported_headers_raise(tmp_path):
    for header in ('matrix array real general', 'matrix coordinate complex general',
                   'matrix coordinate complex hermitian'):
        with pytest.raises(ValueError):
            load_text(tmp_path, '1 1 1\n1 1 1\n', header)


def test_float32_underflow_is_counted_and_kept(tmp_path):
    csr, rec = load_text(tmp_path, '1 2 2\n1 1 1e-50\n1 2 1\n')
    assert rec['float32_underflow_zeros'] == 1 and csr.nnz == 2 and rec['numerical_nnz'] == 1


def test_digest_matches_bench_spmm():
    pytest.importorskip('torch_fast_spdmm_core')
    from benchmark.bench_spmm import digest_arrays
    a, b = np.arange(7, dtype=np.int32), np.linspace(0, 1, 5, dtype=np.float32)
    assert rm.digest_arrays(a, b) == digest_arrays(a, b)


# ---------------------------------------------------------------------------
# download / prepare / load cache
# ---------------------------------------------------------------------------

def make_archive(tmp_path, spec, mtx_text):
    archive = tmp_path / 'src.tar.gz'
    payload = mtx_text.encode()
    with tarfile.open(archive, 'w:gz') as tar:
        for name, data in ((f'{spec.name}/{spec.name}.mtx', payload),
                           (f'{spec.name}/{spec.name}_b.mtx', b'aux')):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


SMALL = '%%MatrixMarket matrix coordinate real general\n3 5 4\n1 1 1\n1 5 2\n3 2 0\n3 2 1\n'


def test_download_prepare_load_roundtrip(tmp_path):
    spec = rm.MatrixSpec('Test', 'tiny', 'test')
    cache = tmp_path / 'cache'
    archive = make_archive(tmp_path, spec, SMALL)
    source = rm.download(spec, cache, url=archive.as_uri())
    assert source['archive_sha256'] == rm.file_sha256(archive)
    assert source['mtx_member'] == 'tiny/tiny.mtx' and len(source['archive_members']) == 2
    assert rm.download(spec, cache, url='file:///nonexistent') == source  # cached, no refetch
    stats = tmp_path / 'ssstats.csv'
    stats.write_text('1\n2026-01-01\nTest,tiny,3,5,3,1,0,0,0,0,0,test,4\n')
    record = rm.prepare(spec, cache, stats_path=stats)
    assert record['stored_entries'] == 3 and record['numerical_nnz'] == 3
    assert record['duplicates_summed'] == 1 and record['catalog_nnz_matches_numerical']
    assert record['mtx_sha256'] == source['mtx_sha256']
    csr, loaded = rm.load(spec, cache)
    assert loaded == record and csr.shape == (3, 5)
    assert rm.prepare(spec, cache) == record  # reused, not reparsed
    with pytest.raises(FileNotFoundError):
        rm.load(spec, cache, 'drop')


def test_cached_archive_tamper_is_detected(tmp_path):
    spec = rm.MatrixSpec('Test', 'tiny', 'test')
    cache = tmp_path / 'cache'
    rm.download(spec, cache, url=make_archive(tmp_path, spec, SMALL).as_uri())
    (rm.matrix_dir(cache, spec) / 'tiny.mtx').write_text(SMALL.replace('1 5 2', '1 5 3'))
    with pytest.raises(ValueError, match='hash differs'):
        rm.download(spec, cache)


def test_catalog_shape_mismatch_raises(tmp_path):
    spec = rm.MatrixSpec('Test', 'tiny', 'test')
    cache = tmp_path / 'cache'
    rm.download(spec, cache, url=make_archive(tmp_path, spec, SMALL).as_uri())
    stats = tmp_path / 'ssstats.csv'
    stats.write_text('1\n2026-01-01\nTest,tiny,5,3,3\n')
    with pytest.raises(ValueError, match='disagrees'):
        rm.prepare(spec, cache, stats_path=stats)


def test_canonical_cache_tamper_is_detected(tmp_path):
    spec = rm.MatrixSpec('Test', 'tiny', 'test')
    cache = tmp_path / 'cache'
    rm.download(spec, cache, url=make_archive(tmp_path, spec, SMALL).as_uri())
    rm.prepare(spec, cache)
    path = rm.canonical_path(cache, spec, 'keep')
    with np.load(path) as f:
        fields = {k: f[k] for k in f.files}
    fields['data'] = fields['data'] * 2
    np.savez(path, **fields)
    with pytest.raises(ValueError, match='content hash'):
        rm.load(spec, cache)


def test_panel_resolution():
    assert [s.key for s in rm.resolve(['lp_cre_b', 'SNAP/roadNet-CA'])] == \
        ['LPnetlib/lp_cre_b', 'SNAP/roadNet-CA']
    assert len(rm.PANEL) == 6
    with pytest.raises(ValueError):
        rm.resolve(['nope'])


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def random_csr(rows, cols, density, seed):
    a = sp.random(rows, cols, density=density, format='csr', dtype=np.float32,
                  random_state=np.random.default_rng(seed))
    a.data = a.data * 2 - 1
    a.sort_indices()
    a.indptr, a.indices = a.indptr.astype(np.int32), a.indices.astype(np.int32)
    return a


LABEL_KW = dict(matrix='Test/tiny', explicit_zeros='keep', method='hem', k=4, seed=0)


@pytest.fixture
def saved(tmp_path):
    csr = random_csr(10, 13, 0.3, 0)
    rows, cols = np.arange(10) // 4, np.arange(13) // 4
    sha = rm.matrix_hash(csr)
    path = tmp_path / 'labels.npz'
    br.save_labels(path, rows, cols, matrix_sha256=sha, shape=csr.shape, feasible_first=True,
                   cluster_ms=1.5, host=dict(node='n', cpu_model='c', cpus=2), **LABEL_KW)
    return path, csr, sha, rows, cols


def test_labels_roundtrip(saved):
    path, csr, sha, rows, cols = saved
    r, c, prov = br.load_labels(path, matrix_sha256=sha, shape=csr.shape, **LABEL_KW)
    np.testing.assert_array_equal(r, rows)
    np.testing.assert_array_equal(c, cols)
    assert prov['cluster_ms'] == 1.5 and prov['cluster_node'] == 'n'


@pytest.mark.parametrize('field,value', [
    ('matrix_sha256', '0' * 64), ('matrix', 'Other/x'), ('explicit_zeros', 'drop'),
    ('method', 'identity'), ('k', 8), ('seed', 1), ('shape', (10, 14)), ('feasible_first', False)])
def test_labels_reject_mismatched_provenance(saved, field, value):
    path, csr, sha, _, _ = saved
    kw = dict(LABEL_KW, matrix_sha256=sha, shape=csr.shape)
    kw[field] = value
    with pytest.raises(ValueError):
        br.load_labels(path, **kw)


def resave(path, **changes):
    with np.load(path) as f:
        fields = {k: f[k] for k in f.files}
    fields.update(changes)
    np.savez(path, **fields)


def test_labels_reject_edited_labels(saved):
    path, csr, sha, rows, _ = saved
    edited = rows.astype(np.int32).copy()
    edited[0], edited[4] = 1, 0  # same group sizes, still valid: only the hash can catch it
    resave(path, rows=edited)
    with pytest.raises(ValueError, match='label hash'):
        br.load_labels(path, matrix_sha256=sha, shape=csr.shape, **LABEL_KW)


@pytest.mark.parametrize('rows,match', [
    (np.zeros(10, np.int32), 'exceeds k'),
    (np.full(10, 10, np.int32), 'out of range'),
    (np.zeros(9, np.int32), 'length'),
    (np.zeros(10, np.int64), 'int32'),
])
def test_labels_reject_invalid_arrays(saved, rows, match):
    path, csr, sha, _, cols = saved
    resave(path, rows=rows, label_sha256=rm.digest_arrays(rows, cols.astype(np.int32)))
    with pytest.raises(ValueError, match=match):
        br.load_labels(path, matrix_sha256=sha, shape=csr.shape, **LABEL_KW)


# ---------------------------------------------------------------------------
# permutations and reference validation
# ---------------------------------------------------------------------------

def test_permutation_reorder_gather_restore():
    rng = np.random.default_rng(1)
    a = random_csr(37, 53, 0.1, 2)
    rows, cols = rng.integers(0, 6, 37), rng.integers(0, 9, 53)
    pr, pc = br.permutation_from_labels(rows), br.permutation_from_labels(cols)
    assert np.all(np.diff(rows[pr]) >= 0) and np.all(np.diff(cols[pc]) >= 0)
    for labels, p in ((rows, pr), (cols, pc)):  # original order inside each group
        for g in np.unique(labels):
            members = p[labels[p] == g]
            assert np.all(np.diff(members) > 0)
    ap = br.reorder(a, pr, pc)
    np.testing.assert_array_equal(ap.toarray(), a.toarray()[pr][:, pc])
    assert ap.indices.dtype == np.int32 and ap.has_sorted_indices
    x = rng.standard_normal((53, 5)).astype(np.float32)
    y_perm = ap.astype(np.float64) @ x[pc].astype(np.float64)        # steady state
    y = a.astype(np.float64) @ x.astype(np.float64)
    np.testing.assert_allclose(y_perm, y[pr])
    np.testing.assert_allclose(y_perm[np.argsort(pr)], y)             # restore


def test_identity_labels_give_identity_permutation():
    labels = np.arange(100) // 32
    np.testing.assert_array_equal(br.permutation_from_labels(labels), np.arange(100))


def test_permutation_matches_boo_ids():
    spmm = pytest.importorskip('torch_fast_spdmm')
    rng = np.random.default_rng(3)
    a = random_csr(70, 45, 0.15, 4)
    rows, cols = rng.integers(0, 5, 70), rng.integers(0, 4, 45)
    boo = spmm.csr_to_boo(a, row_labels=rows, col_labels=cols, small_block_policy='remainder',
                          small_block_threshold=4, nz_order='rowmajor')
    np.testing.assert_array_equal(np.asarray(boo.row_ids), br.permutation_from_labels(rows))
    np.testing.assert_array_equal(np.asarray(boo.col_ids), br.permutation_from_labels(cols))


def test_reference_accepts_float32_and_rejects_errors():
    rng = np.random.default_rng(5)
    a = random_csr(40, 30, 0.2, 6)
    a = sp.vstack([a, sp.csr_matrix((1, 30), dtype=np.float32)]).tocsr()  # empty last row
    x = rng.standard_normal((30, 70)).astype(np.float32)  # F not a chunk multiple
    ref = br.Reference(a, x)
    y = (a @ x).astype(np.float32)
    good = ref.check(torch.from_numpy(y))
    assert good['bound_violations'] == 0
    row = int(np.argmax(np.diff(a.indptr)))
    missing = y.copy()  # drop one term of the densest row
    j = a.indices[a.indptr[row]]
    missing[row] -= a.data[a.indptr[row]] * x[j]
    assert ref.check(missing)['bound_violations'] > 0
    nonzero_empty = y.copy()
    nonzero_empty[-1, 0] = 1e-30  # empty rows must be exactly zero
    assert ref.check(nonzero_empty)['bound_violations'] == 1
    nan = y.copy()
    nan[0, 0] = np.nan
    assert ref.check(nan)['bound_violations'] >= 1
    pr = rng.permutation(41)
    assert ref.check(y[pr], row_perm=pr)['bound_violations'] == 0
    assert ref.check(y[pr])['bound_violations'] > 0  # wrong coordinates are caught


# ---------------------------------------------------------------------------
# end-to-end GPU cell on a tiny rectangular matrix
# ---------------------------------------------------------------------------

def tiny_case(tmp_path, monkeypatch):
    """A small rectangular matrix with empty rows, cached and clustered like the panel."""
    from torch_fast_spdmm.clustering import cluster_axes
    spec = rm.MatrixSpec('Test', 'rect', 'test')
    monkeypatch.setattr(rm, 'PANEL', (spec,))
    monkeypatch.setattr(rm, 'CATALOG', {spec.key: spec})
    a = random_csr(300, 700, 0.02, 7)
    a = sp.vstack([a, sp.csr_matrix((5, 700), dtype=np.float32)]).tocsr()  # empty rows
    coo = a.tocoo()
    body = ''.join(f'{i + 1} {j + 1} {v!r}\n' for i, j, v in zip(coo.row, coo.col, coo.data.tolist()))
    cache = tmp_path / 'cache'
    archive = make_archive(tmp_path, spec, f'%%MatrixMarket matrix coordinate real general\n'
                                          f'{a.shape[0]} {a.shape[1]} {coo.nnz}\n{body}')
    rm.download(spec, cache, url=archive.as_uri())
    record = rm.prepare(spec, cache)
    csr, _ = rm.load(spec, cache)
    labels = tmp_path / 'labels'
    labels.mkdir()
    for method in ('identity', 'hem'):
        rows, cols = cluster_axes(csr, method=method, target_nr=32, target_nc=32, max_nr=32,
                                  max_nc=32, feasible_first=True)
        br.save_labels(br.label_path(labels, spec, 'keep', method, 32), rows, cols,
                       matrix=spec.key, matrix_sha256=record['matrix_sha256'], shape=csr.shape,
                       explicit_zeros='keep', method=method, k=32, seed=0, feasible_first=True,
                       cluster_ms=0., host={})
    return spec, cache, labels


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_matmul_phase_end_to_end(tmp_path, monkeypatch):
    spec, cache, labels = tiny_case(tmp_path, monkeypatch)
    out = tmp_path / 'out'
    br.main(['--phase', 'matmul', '--cache', str(cache), '--labels', str(labels), '--out', str(out),
             '--matrices', spec.key, '--groups', '32', '--features', '32', '72',
             '--direct-algorithms', 'DEFAULT', 'CSR_ALG2', '--boo-fpt', '2', '4',
             '--repeats', '3', '--warmup', '1'])
    meta = json.loads((out / 'metadata.json').read_text())
    assert meta['status'] == 'complete' and meta['completed_cells'] == 4
    with open(out / 'summary.csv') as f:
        summary = list(csv.DictReader(f))
    for row in summary:
        if row['supported'] == 'True' and row['correctness'] != 'not_applicable':
            assert row['correctness'] == 'passed' and row['bound_violations'] == '0', row['config']
        if row['supported'] == 'False':
            assert row['unsupported_reason'], row['config']
    configs = {(r['labeling'], r['features'], r['config']) for r in summary if r['supported'] == 'True'}
    for f in ('32', '72'):
        for name in ('torch_csr', 'boo', 'boo_cpp', 'direct_DEFAULT', 'direct_DEFAULT_api',
                     'boo_fpt2', 'boo_fpt4', 'boo_fpt2_cpp', 'boo_fpt4_cpp'):
            assert ('identity_k32', f, name) in configs and ('hem_k32', f, name) in configs
        for name in ('torch_perm_steady', 'torch_perm_gather', 'perm_gather_x', 'perm_restore_y',
                     'direct_perm_DEFAULT', 'direct_perm_DEFAULT_steady_api',
                     'direct_perm_DEFAULT_gather_api'):
            assert ('hem_k32', f, name) in configs
            assert ('identity_k32', f, name) not in configs  # identity permutation: omitted
    perm_rows = [r for r in summary if r['labeling'] == 'hem_k32']
    assert all(r['permutation_identity'] == 'False' for r in perm_rows if r['config'] == 'boo')
    with open(out / 'samples.csv') as f:
        samples = list(csv.DictReader(f))
    assert all(float(s['time_ms']) > 0 for s in samples)
    fpts = {r['config']: r['features_per_thread'] for r in summary
            if r['config'].startswith('boo') and r['supported'] == 'True'}
    assert fpts['boo'] == fpts['boo_cpp'] == '1' and fpts['boo_fpt4_cpp'] == '4'
    br.report(out)  # smoke: the report reads what matmul wrote


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_profile_real_driver(tmp_path, monkeypatch):
    from benchmark import profile_real
    spec, cache, labels = tiny_case(tmp_path, monkeypatch)
    common = ['--matrix', spec.key, '--labeling', 'hem_k32', '--features', '72',
              '--cache', str(cache), '--labels', str(labels), '--warmup', '1']
    for method in ('boo', 'direct', 'direct_perm', 'torch'):
        out = tmp_path / method
        profile_real.main(common + ['--method', method, '--out', str(out)])
        details = json.loads((out / 'profile.json').read_text())
        assert details['correctness'] == 'passed' and details['bound_violations'] == 0
        assert details['method'] == method and details['profiled_calls'] == 1
    profile_real.main(common + ['--structure', '--out', str(tmp_path / 'structure')])
    details = json.loads((tmp_path / 'structure' / 'structure.json').read_text())
    boo = details['boo']
    assert boo['active_x_rows'] <= boo['logical_x_rows'] and boo['retained_nnz'] > 0
    assert boo['retained_nnz'] + boo['remainder_nnz'] == details['csr']['nnz']
    with pytest.raises(FileNotFoundError):  # no saved labels for this labeling: never clusters
        profile_real.main(common[:2] + ['--labeling', 'hem_k31'] + common[4:] +
                          ['--out', str(tmp_path / 'bad')])
