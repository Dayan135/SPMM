"""Extract named counters from the nnzflat Nsight Compute raw exports and nsys summaries.

  python nnzflat_profile_report.py PROFILE_DIR OUT_CSV

One row per profiled configuration (BOO main kernel; a CSR remainder kernel, if
captured, is reported in separate *_csr columns). Counter names are the exact
ncu raw-page names; derived ratios state their numerators and denominators.
Profiler durations are diagnostic; unprofiled bench_nnzflat timings decide.
"""
import csv
import json
from pathlib import Path
import re
import sys

COUNTERS = {
    'duration_ns': 'gpu__time_duration.sum',
    'grid_size': 'launch__grid_size',
    'block_size': 'launch__block_size',
    'registers_per_thread': 'launch__registers_per_thread',
    'smem_dynamic_bytes': 'launch__shared_mem_per_block_dynamic',
    'occ_limit_shared_mem_blocks': 'launch__occupancy_limit_shared_mem',
    'occ_limit_registers_blocks': 'launch__occupancy_limit_registers',
    'occ_limit_warps_blocks': 'launch__occupancy_limit_warps',
    'theoretical_occupancy_pct': 'sm__maximum_warps_per_active_cycle_pct',
    'achieved_occupancy_pct': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'waves_per_sm': 'launch__waves_per_multiprocessor',
    'sm_throughput_pct': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'warp_inst_executed': 'smsp__inst_executed.sum',
    'threads_per_inst': 'smsp__thread_inst_executed_per_inst_executed.ratio',
    'shared_ld_inst': 'smsp__sass_inst_executed_op_shared_ld.sum',
    'shared_st_inst': 'smsp__sass_inst_executed_op_shared_st.sum',
    'global_ld_inst': 'smsp__sass_inst_executed_op_global_ld.sum',
    'shared_atom_wavefronts': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum',
    'shared_ld_wavefronts': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    'shared_st_wavefronts': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    'bank_conflicts_shared_atom': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum',
    'bank_conflicts_shared_ld': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'bank_conflicts_shared_st': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'global_red_requests_l1': 'l1tex__t_requests_pipe_lsu_mem_global_op_red.sum',
    'global_red_sectors_l1': 'l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum',
    'l2_red_requests': 'lts__t_requests_srcunit_tex_op_red.sum',
    'dram_read_bytes': 'dram__bytes_read.sum',
    'dram_write_bytes': 'dram__bytes_write.sum',
    'ffma_thread_inst': 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
    'fmul_thread_inst': 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
    'fadd_thread_inst': 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
}
STALLS = ('barrier', 'short_scoreboard', 'long_scoreboard', 'mio_throttle', 'lg_throttle',
          'math_pipe_throttle', 'wait', 'not_selected', 'selected', 'membar', 'branch_resolving',
          'dispatch_stall', 'drain', 'no_instruction', 'imc_miss', 'tex_throttle', 'sleeping')
STALL_FMT = 'smsp__average_warps_issue_stalled_{}_per_issue_active.ratio'


SCALE = {'ns': 1.0, 'us': 1e3, 'ms': 1e6, 's': 1e9, 'nsecond': 1.0, 'usecond': 1e3, 'msecond': 1e6,
         'second': 1e9, 'byte': 1.0, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9}


def num(v, unit=''):
    """Value in base units: durations in ns, sizes in bytes (ncu mixes unit prefixes per row)."""
    try:
        x = float(str(v).replace(',', ''))
    except ValueError:
        return None
    return x * SCALE.get(unit, 1.0)


def rows_of(path):
    """Header, then (units row) per kernel: `ncu --page raw` writes units on row 1 when all
    kernels share them; per-row unit prefixes are resolved with the details page instead."""
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = [r for r in rows[2:] if len(r) == len(header)]
    return header, rows[1], data


def main(directory, out):
    directory = Path(directory)
    out_rows = []
    for raw in sorted(directory.glob('*_raw.csv')):
        stem = raw.name[:-len('_raw.csv')]
        header, units, data = rows_of(raw)
        idx = {h: i for i, h in enumerate(header)}
        rec = dict(config=stem)
        m = re.match(r'(.+)_k(\d+)_F(\d+)_(.+)$', stem)
        rec.update(input=m.group(1), k=int(m.group(2)), features=int(m.group(3)), arm=m.group(4))
        for r in data:
            name = r[idx['Kernel Name']]
            suffix = '' if 'boo_spmm' in name else '_csr'
            rec['kernel' + suffix] = name.split('(')[0]
            for key, metric in COUNTERS.items():
                if metric in idx:
                    rec[key + suffix] = num(r[idx[metric]], units[idx[metric]])
            for s in STALLS:
                metric = STALL_FMT.format(s)
                if metric in idx:
                    rec[f'stall_{s}' + suffix] = num(r[idx[metric]])
        if rec.get('warp_inst_executed'):
            rec['shared_atom_wavefronts_per_warp_inst'] = (rec.get('shared_atom_wavefronts') or 0) / rec['warp_inst_executed']
        meta = directory / f'{stem}_ncu_meta' / 'profile.json'
        if meta.exists():
            p = json.loads(meta.read_text())
            for key in ('tile_passes', 'stepb_items', 'stepb_max_trips_p50', 'active_fraction_nnz_weighted',
                        'shared_atomics', 'barriers_per_cta', 'arena_bytes', 'max_active_blocks_per_sm',
                        'num_blocks', 'block_skew_nnz_weighted', 'expected_kernel'):
                rec[key] = p.get(key)
        kern = directory / f'{stem}_nsys_cuda_gpu_kern_sum.csv'
        if kern.exists():
            with open(kern) as f:
                for r in csv.DictReader(f):
                    tag = 'main' if 'boo_spmm' in r.get('Name', '') else 'csr' if 'csr' in r.get('Name', '').lower() else 'other'
                    rec[f'nsys_{tag}_median_ns'] = num(r.get('Med (ns)'))
                    rec[f'nsys_{tag}_instances'] = num(r.get('Instances'))
        out_rows.append(rec)
    fields = sorted({k for r in out_rows for k in r}, key=lambda k: (k not in ('config', 'input', 'k', 'features', 'arm'), k))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f'{len(out_rows)} configurations -> {out}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
