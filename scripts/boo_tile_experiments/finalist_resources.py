"""Resource/occupancy table for finalist BOO variants (RTX 3090, sm_86).

  python finalist_resources.py <out.csv> <finalists P:FPT:slab ...> -- <launch.csv ...> [-- <ncu_wide.csv ...>]

From bench_tiles launch.csv (the launcher's own report: dynamic arena bytes,
registers/thread, local bytes/thread, cudaOccupancyMaxActiveBlocksPerMultiprocessor),
per-limit resident-CTA counts are derived with sm_86 rules:
  registers: 65536 regs/SM, allocation per warp rounded up to 256 registers
  shared:    100 KiB carveout max (102400 B), +1 KiB driver reservation per CTA
  warps:     48 warps/SM;  hardware: 16 CTAs/SM
The binding limit(s) are the minimum; the computed minimum is checked against the
CUDA occupancy API value. Theoretical occupancy = CTAs * warps/CTA / 48.
Achieved occupancy and spill instructions come only from ncu (Occupancy section,
local ld/st counters) where a finalist was profiled; otherwise left blank.
"""
import csv
import math
import sys

SM_REGS, SM_SMEM, RESERVED, SM_WARPS, HW_CTAS = 65536, 102400, 1024, 48, 16


def limits(threads, regs, arena):
    warps = threads // 32
    per_warp = math.ceil(regs * 32 / 256) * 256
    out = dict(registers=SM_REGS // (per_warp * warps) if regs else HW_CTAS,
               shared_memory=SM_SMEM // (arena + RESERVED),
               warps=SM_WARPS // warps, hardware=HW_CTAS)
    return out


def main(argv):
    out, rest = argv[0], argv[1:]
    groups, cur = [[]], 0
    for a in rest:
        if a == '--':
            groups.append([])
        else:
            groups[-1].append(a)
    finalists = {tuple(int(v) for v in f.split(':')) for f in groups[0]}
    launch = [r for path in groups[1] for r in csv.DictReader(open(path))]
    ncu = [r for path in (groups[2] if len(groups) > 2 else []) for r in csv.DictReader(open(path))]
    rows, seen = [], set()
    for r in launch:
        if r['supported'] != 'True':
            continue
        req = (int(r['p']), int(r['features_per_thread_requested']), int(r['feat_block_requested']))
        k = (r['case'], r['features'], req)
        if req not in finalists or k in seen:
            continue
        seen.add(k)
        threads, regs, arena = int(r['threads']), int(r['registers_per_thread']), int(r['arena_bytes'])
        lim = limits(threads, regs, arena)
        ctas = min(lim.values())
        api = int(r['max_active_blocks_per_sm'])
        row = dict(case=r['case'], features=r['features'], config=f'P{req[0]}/FPT{req[1]}/slab{req[2]}',
                   effective_p=r['effective_p'], grid_z=r['grid_z'], num_blocks=r['num_blocks'],
                   dynamic_smem_bytes=arena, registers_per_thread=regs,
                   local_bytes_per_thread=r['local_bytes_per_thread'],
                   **{f'ctas_limit_{n}': v for n, v in lim.items()},
                   binding_limits='+'.join(n for n, v in lim.items() if v == ctas),
                   theoretical_ctas_per_sm=ctas, occupancy_api_ctas_per_sm=api,
                   api_matches=ctas == api,
                   theoretical_occupancy_pct=round(100 * ctas * threads / 32 / SM_WARPS, 1))
        stem = f"{r['case']}_F{r['features']}_boo_p{req[0]}_fpt{req[1]}_s{req[2]}"
        prof = [n for n in ncu if n['stem'] == stem and n['role'] == 'boo_main']
        if prof:
            n = prof[0]
            row.update(ncu_achieved_occupancy_pct=n.get('Achieved Occupancy'),
                       ncu_theoretical_occupancy_pct=n.get('Theoretical Occupancy'),
                       ncu_block_limit_registers=n.get('Block Limit Registers'),
                       ncu_block_limit_shared_mem=n.get('Block Limit Shared Mem'),
                       ncu_block_limit_warps=n.get('Block Limit Warps'),
                       ncu_local_ld_insts=n.get('smsp__sass_inst_executed_op_local_ld.sum'),
                       ncu_local_st_insts=n.get('smsp__sass_inst_executed_op_local_st.sum'))
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['case'], int(r['features']), r['config'])))
    print(f'{len(rows)} finalist rows -> {out}; API mismatches: '
          f"{sum(not r['api_matches'] for r in rows)}")


if __name__ == '__main__':
    main(sys.argv[1:])
