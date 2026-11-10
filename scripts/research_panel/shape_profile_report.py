"""Summarize the matched-shape Nsight Compute profiles (shape_job.sh profile).

  python scripts/research_panel/shape_profile_report.py PROFILE_DIR LIST OUT_DIR

PROFILE_DIR holds <label>_raw.csv and <label>_meta/profile.json; LIST is profile_confirm.txt.
Writes OUT_DIR/profile_kernels.csv (one row per captured kernel) and profile_summary.csv (one row
per label: total kernel time, the dominant kernel's counters, and the SM-wide shared memory in use =
CTAs/SM x (dynamic + static + 1 KB reserved) / 100 KB on GA102). Counter names are ncu raw-page
names (nnzflat_profile_report.COUNTERS). Durations are profiler-side and diagnostic only.
"""
from pathlib import Path
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nnzflat_profile_report as npr  # noqa: E402

COUNTERS = dict(npr.COUNTERS,
                smem_static_bytes='launch__shared_mem_per_block_static',
                smem_config_bytes='launch__shared_mem_config_size',
                occ_limit_blocks='launch__occupancy_limit_blocks',
                l1_hit_pct='l1tex__t_sector_hit_rate.pct',
                l2_hit_pct='lts__t_sector_hit_rate.pct',
                l2_bytes='lts__t_bytes.sum',
                dram_pct='dram__throughput.avg.pct_of_peak_sustained_elapsed',
                l1_pct='l1tex__throughput.avg.pct_of_peak_sustained_active',
                l2_pct='lts__throughput.avg.pct_of_peak_sustained_elapsed',
                issue_active_pct='sm__inst_issued.avg.pct_of_peak_sustained_active',
                global_ld_sectors='l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
                global_ld_requests='l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
                l2_read_sectors='lts__t_sectors_srcunit_tex_op_read.sum',
                smem_allocated_bytes='launch__shared_mem_per_block_allocated')
SMEM_SM, RESERVED = 102400, 1024


def kernels(raw, label):
    header, units, data = npr.rows_of(raw)
    idx = {h: i for i, h in enumerate(header)}
    out = []
    for r in data:
        rec = dict(label=label, kernel=r[idx['Kernel Name']].split('(')[0][:90])
        for key, metric in COUNTERS.items():
            if metric in idx:
                rec[key] = npr.num(r[idx[metric]], units[idx[metric]].split('/')[0])
        for s in npr.STALLS:
            metric = npr.STALL_FMT.format(s)
            if metric in idx:
                rec[f'stall_{s}'] = npr.num(r[idx[metric]])
        out.append(rec)
    return out


def main(directory, listing, out):
    directory, out = Path(directory), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = [l.split() for l in Path(listing).read_text().splitlines() if l.strip() and not l.startswith('#')]
    rows, summ = [], []
    for label, cell, arm in labels:
        raw = directory / f'{label}_raw.csv'
        if not raw.exists():
            summ.append(dict(label=label, cell=cell, arm=arm, status='missing'))
            continue
        ks = kernels(raw, label)
        rows += ks
        k = pd.DataFrame(ks)
        main_k = k.loc[k.duration_ns.idxmax()]
        cta = main_k.get('theoretical_occupancy_pct')
        per = main_k.get('smem_allocated_bytes')  # dynamic + static + driver-reserved, as allocated
        ctas = min(v for v in (main_k.get('occ_limit_shared_mem_blocks'), main_k.get('occ_limit_registers_blocks'),
                               main_k.get('occ_limit_warps_blocks'), main_k.get('occ_limit_blocks')) if v == v and v)
        meta = directory / f'{label}_meta' / 'profile.json'
        p = json.loads(meta.read_text()) if meta.exists() else {}
        rec = dict(label=label, cell=cell, arm=arm, status='ok', kernels=len(k),
                   total_us=k.duration_ns.sum() / 1e3, main_kernel=main_k.kernel, main_us=main_k.duration_ns / 1e3,
                   ctas_per_sm=ctas, smem_per_cta_kb=per / 1024, smem_sm_pct=100 * ctas * per / SMEM_SM,
                   theoretical_occ_pct=cta, achieved_occ_pct=main_k.get('achieved_occupancy_pct'),
                   correctness=(p.get('check') or {}).get('bound_violations'))
        for c in ('block_size', 'grid_size', 'registers_per_thread', 'waves_per_sm', 'sm_throughput_pct',
                  'issue_active_pct', 'dram_read_bytes', 'dram_write_bytes', 'l2_bytes', 'l1_hit_pct',
                  'l2_hit_pct', 'dram_pct', 'l1_pct', 'l2_pct', 'shared_ld_wavefronts', 'shared_st_wavefronts',
                  'bank_conflicts_shared_ld', 'bank_conflicts_shared_st', 'global_red_requests_l1',
                  'global_red_sectors_l1', 'global_ld_sectors', 'global_ld_requests', 'l2_read_sectors',
                  'warp_inst_executed'):
            rec[c] = main_k.get(c)
        for s in npr.STALLS:
            rec[f'stall_{s}'] = main_k.get(f'stall_{s}')
        summ.append(rec)
    pd.DataFrame(rows).to_csv(out / 'profile_kernels.csv', index=False)
    s = pd.DataFrame(summ)
    s.to_csv(out / 'profile_summary.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    show = ['label', 'total_us', 'main_us', 'block_size', 'ctas_per_sm', 'smem_per_cta_kb', 'smem_sm_pct',
            'theoretical_occ_pct', 'achieved_occ_pct', 'waves_per_sm', 'issue_active_pct', 'dram_read_bytes',
            'l2_hit_pct']
    print(s[[c for c in show if c in s]].round(2).to_string(index=False))
    stalls = ['label'] + [f'stall_{x}' for x in ('long_scoreboard', 'barrier', 'short_scoreboard', 'mio_throttle',
                                                  'lg_throttle', 'wait', 'math_pipe_throttle', 'not_selected')]
    pairs(s, out)
    print('\nstalls per issued instruction (dominant kernel):')
    print(s[[c for c in stalls if c in s]].round(2).to_string(index=False))


PAIR_COLS = ['total_us', 'global_ld_sectors', 'shared_st_wavefronts', 'shared_ld_wavefronts', 'l2_read_sectors',
             'dram_read_bytes', 'global_red_sectors_l1', 'warp_inst_executed', 'achieved_occ_pct']


def pairs(s, out):
    """winner / 32x32 ratio per counter, for labels <stem>_win and <stem>_32."""
    s = s.set_index('label')
    rows = []
    for w in [l for l in s.index if l.endswith('_win')]:
        c = w[:-4] + '_32'
        if c in s.index and s.loc[w, 'status'] == s.loc[c, 'status'] == 'ok':
            rows.append(dict(pair=w[:-4], **{k: s.loc[w, k] / s.loc[c, k] for k in PAIR_COLS if k in s}))
    p = pd.DataFrame(rows)
    p.to_csv(out / 'profile_pairs.csv', index=False)
    print('\nwinner / 32x32 (dominant kernel; total_us sums all captured kernels):')
    print(p.round(3).to_string(index=False))


if __name__ == '__main__':
    main(*sys.argv[1:4])
