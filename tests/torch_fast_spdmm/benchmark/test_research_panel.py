import numpy as np
import pytest
from scipy import sparse
import torch
import torch_fast_spdmm as spmm

from benchmark.panel_structure import boo_structure, csr_structure
from benchmark.research_panel import matched_ratios, parse_args, select_direct_algorithm, validate_labels
from benchmark.bench_spmm import _BOOFPTTimed, measure_cell


def example():
    return sparse.csr_matrix((np.array([1., -1., 2.], np.float32),
                              ([0, 0, 2], [0, 1, 1])), shape=(3, 4))


def test_active_support_does_not_count_unused_group_members():
    mat = example()
    boo = spmm.csr_to_boo(mat, row_labels=np.zeros(3, np.int32),
                          col_labels=np.zeros(4, np.int32), small_block_threshold=0)
    stats = boo_structure(boo)
    assert stats['logical_x_rows'] == 4
    assert stats['active_x_rows'] == 2
    assert stats['potential_y_rows'] == 3
    assert stats['active_y_rows'] == 2
    assert stats['unused_x_fraction'] == .5
    assert stats['coordinate_bytes'] == 6
    assert stats['uint8_csr_structure_bytes'] == 19
    assert stats['output_block_fanout']['maximum'] == 1
    csr = csr_structure(mat)
    assert csr['empty_rows'] == 1
    assert csr['empty_columns'] == 2


def test_structure_handles_all_remainder():
    mat = example()
    boo = spmm.csr_to_boo(mat, cluster='identity', target_nr=1, target_nc=1,
                          small_block_threshold=4)
    stats = boo_structure(boo)
    assert stats['num_blocks'] == stats['active_x_rows'] == stats['active_y_rows'] == 0
    assert stats['remainder_nnz'] == mat.nnz


@pytest.mark.parametrize('rows,cols,shape,k', [
    (np.array([0, 0]), np.array([0]), (3, 1), 2),
    (np.array([-1, 0]), np.array([0]), (2, 1), 2),
    (np.array([0., 0.]), np.array([0]), (2, 1), 2),
    (np.array([0, 2, 2]), np.array([0]), (3, 1), 2),
    (np.array([0, 0, 0]), np.array([0]), (3, 1), 2),
])
def test_saved_labels_reject_wrong_shape_dtype_or_groups(rows, cols, shape, k):
    with pytest.raises(ValueError):
        validate_labels(rows, cols, shape, k)


def test_ratios_match_timing_scopes():
    rows = [dict(algorithm='cusparse_direct', timing_scope='full_call', median_ms=10.),
            dict(algorithm='cusparse_direct', timing_scope='cusparseSpMM_only', median_ms=2.),
            dict(algorithm='boo', timing_scope='full_call', median_ms=5.),
            dict(algorithm='boo', timing_scope='cpp_only', median_ms=4.),
            dict(algorithm='cusparse_csr', timing_scope='full_call', median_ms=12.)]
    matched_ratios(rows)
    assert rows[2]['speedup_vs_direct'] == 2.
    assert rows[3]['speedup_vs_direct'] == .5
    assert rows[3]['speedup_vs_cusparse'] is None
    assert rows[3]['launch_mode'] == 'preallocated_legacy_fork_join'


def test_profile_requires_one_workload():
    with pytest.raises(SystemExit):
        parse_args(['--phase', 'profile', '--labels', '/tmp/labels', '--out', '/tmp/profile'])


def test_direct_selection_requires_matching_input_and_scope(tmp_path):
    path = tmp_path / 'summary.csv'
    path.write_text('algorithm,timing_scope,correctness,matrix_sha256,x_sha256,median_ms,direct_algorithm\n'
                    'cusparse_direct,full_call,passed,A,X,3,DEFAULT\n'
                    'cusparse_direct,full_call,passed,A,X,2,CSR_ALG2\n'
                    'cusparse_direct,cusparseSpMM_only,passed,A,X,1,CSR_ALG1\n'
                    'cusparse_direct,full_call,passed,A,Y,0.1,CSR_ALG3\n')
    assert select_direct_algorithm(path, 'A', 'X') == 'CSR_ALG2'
    with pytest.raises(ValueError, match='exact A/X'):
        select_direct_algorithm(path, 'B', 'X')


def test_cpp_fpt_is_reapplied_for_every_interleaved_entry(monkeypatch):
    import benchmark.bench_spmm as harness
    current = [None]
    monkeypatch.setattr(harness.core, 'set_boo_rowseg_fpt', lambda value: current.__setitem__(0, value))
    class Plan:
        def run(self):
            return current[0]
        timed_run = run
    two, four = _BOOFPTTimed(Plan(), 2), _BOOFPTTimed(Plan(), 4)
    assert two.run() == 2
    assert four.timed_run() == 4
    assert two.timed_run() == 2
    assert four.run() == 4


def test_fpt_timing_requires_paired_baseline():
    with pytest.raises(SystemExit):
        parse_args(['--phase', 'benchmark', '--labels', '/tmp/L', '--out', '/tmp/O', '--boo-fpt', '2', '4'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_matched_measurement_omits_incorrect_diagnostics():
    mat = example()
    mat.indptr = mat.indptr.astype(np.int32)
    mat.indices = mat.indices.astype(np.int32)
    csr = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr), torch.from_numpy(mat.indices),
                                  torch.from_numpy(mat.data), size=mat.shape).cuda()
    boo = spmm.csr_to_boo(mat, cluster='identity', target_nr=2, target_nc=2,
                          small_block_threshold=0).to('cuda')
    config = dict(algorithm='boo', config='boo', threads=256, p=32, launch_mode='auto',
                  compressed_fraction=1., remainder_nnz=0, num_blocks=len(boo.blk_row_grp))
    x = torch.randn(4, 32, generator=torch.Generator().manual_seed(0))
    rows, samples = measure_cell([(dict(algorithm='cusparse_csr', config='csr'), csr), (config, boo)],
                                 mat, x, 1, 2, 0, direct_algorithms=['DEFAULT'],
                                 symmetric=True, diagnostics=False)
    assert len(rows) == 5
    assert len(samples) == 10
    assert all(row['correctness'] == 'passed' for row in rows)
    assert not any('diagnostic' in row['algorithm'] for row in rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_interleaved_fpt_api_and_cpp_variants():
    mat = example()
    mat.indptr, mat.indices = mat.indptr.astype(np.int32), mat.indices.astype(np.int32)
    csr = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr), torch.from_numpy(mat.indices),
                                  torch.from_numpy(mat.data), size=mat.shape).cuda()
    boo = spmm.csr_to_boo(mat, cluster='identity', target_nr=2, target_nc=2,
                          small_block_threshold=0).to('cuda')
    variants = [(dict(algorithm='cusparse_csr', config='csr'), csr)]
    for fpt in [1, 2, 4]:
        variants.append((dict(algorithm='boo', config=f'boo_fpt{fpt}', threads=256, p=32,
                              launch_mode='auto', features_per_thread=fpt, compressed_fraction=1.,
                              remainder_nnz=0, num_blocks=len(boo.blk_row_grp)), boo))
    x = torch.randn(4, 35, generator=torch.Generator().manual_seed(0))
    rows, samples = measure_cell(variants, mat, x, 1, 2, 0, symmetric=True, diagnostics=False)
    assert len(rows) == 7 and len(samples) == 14
    assert all(r['correctness'] == 'passed' for r in rows)
    assert {r['features_per_thread'] for r in rows if r['algorithm'] == 'boo'} == {1, 2, 4}
