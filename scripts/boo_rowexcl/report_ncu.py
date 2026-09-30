"""Extract every NCU kernel event and the requested main-kernel comparison."""
import argparse
import csv
import json
import math
from pathlib import Path

DURATION='gpu__time_duration.sum'
OCCUPANCY='sm__warps_active.avg.pct_of_peak_sustained_active'
STALL='smsp__warp_issue_stalled_no_instruction_per_warp_active.pct'


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures',type=Path); p.add_argument('--out',type=Path,default=Path('.'))
    a=p.parse_args()
    events=[]; summaries=[]; metadata=[]
    for path in sorted(a.captures.glob('*.ncu.csv')):
        stem=path.name.removesuffix('.ncu.csv')
        meta=json.loads((a.captures/f'{stem}.json').read_text()); metadata.append(meta)
        args=meta['arguments']; profile=args['profile']; mode=args['mode']
        with path.open() as f:
            reader=csv.reader(f)
            for header in reader:
                if 'Metric Name' in header and 'Kernel Name' in header: break
            else: raise ValueError(f'No NCU CSV header in {path}')
            kernels={}
            for values in reader:
                if len(values)!=len(header): continue
                r=dict(zip(header,values))
                if r['Metric Name'] not in (DURATION,OCCUPANCY,STALL): continue
                k=r['ID']; d=kernels.setdefault(k,dict(name=r['Kernel Name'],grid=r['Grid Size'],block=r['Block Size']))
                d[r['Metric Name']]=(float(r['Metric Value'].replace(',','')),r['Metric Unit'])
        assert kernels,path
        selected=[]
        for k,d in kernels.items():
            assert all(metric in d for metric in (DURATION,OCCUPANCY,STALL)),(path,k)
            assert all(math.isfinite(d[m][0]) for m in (DURATION,OCCUPANCY,STALL)),(path,k)
            number,unit=d[DURATION]
            ms=number*{'ns':1e-6,'us':1e-3,'ms':1.,'s':1000.}[unit]
            main_kernel=('boo_rowexcl_kernel' in d['name'] if mode=='rowexcl' else 'boo_spmm_rowseg_kernel' in d['name'])
            event=dict(profile=profile,mode=mode,requested_size=args['size'],features=args['features'],
                       k=args['k'],slab=args['slab'] if mode=='rowexcl' else 64,
                       slurm_job_id=meta['slurm_job_id'],node=meta['node'],source_commit=meta['git_commit'],
                       source_sha256=meta['source_sha256'],extension_sha256=meta['extension_sha256'],
                       kernel_id=k,kernel_name=d['name'],grid=d['grid'],block=d['block'],main_kernel=main_kernel,
                       duration_ms=ms,achieved_occupancy_pct=d[OCCUPANCY][0],no_instruction_pct=d[STALL][0])
            events.append(event)
            if main_kernel:selected.append(event)
        assert selected,(profile,mode)
        duration=sum(r['duration_ms'] for r in selected)
        summaries.append(dict(profile=profile,mode=mode,k=args['k'],slab=args['slab'] if mode=='rowexcl' else 64,
            kernel_count=len(selected),duration_ms=duration,
            achieved_occupancy_pct=sum(r['achieved_occupancy_pct']*r['duration_ms'] for r in selected)/duration,
            no_instruction_pct=sum(r['no_instruction_pct']*r['duration_ms'] for r in selected)/duration,
            theoretical_resident_ctas=meta['plan'].get('resident_ctas_per_sm',''),
            registers_per_thread=meta['plan'].get('registers_per_thread',''),
            shared_bytes=meta['plan'].get('smem_bytes','')))
    assert len(summaries)==6 and {(r['profile'],r['mode']) for r in summaries}=={
        (p,m) for p in ('fem2d','fem2d_permuted','diag_b1.0') for m in ('rowexcl','boo')}
    for profile in ('fem2d','fem2d_permuted','diag_b1.0'):
        pair=[m for m in metadata if m['arguments']['profile']==profile]
        assert len(pair)==2 and len({m['arguments']['k'] for m in pair})==1
        for field in ('matrix_sha256','x_sha256','source_sha256','extension_sha256'):
            assert len({m[field] for m in pair})==1,(profile,field)
    for m in metadata:
        for field in ('python_package_path','extension_path'):
            assert m[field].startswith('/home/dayanb/research/boo-rowexcl-20260929/'),m[field]
    write(a.out/'BOO_ROWEXCL_ncu_kernels.csv',events)
    write(a.out/'BOO_ROWEXCL_ncu_summary.csv',summaries)
    (a.out/'BOO_ROWEXCL_ncu_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['## Nsight Compute: occupancy and instruction-fetch stalls','',
           'One warmed call per mode/profile at N=262,144, F=512. Row-exclusive uses its Phase A best C++ configuration. Current BOO uses the same HEM target and hence the same grouping; its feature tile stays at p=64. This isolates execution-mode resource differences. All launched kernels, including output initialization and CSR remainder work, are retained in the [kernel CSV](BOO_ROWEXCL_ncu_kernels.csv). The table compares the row-exclusive kernel with the current BOO row-segmented kernel; if a mode has multiple such launches, percentages are duration-weighted descriptive averages. These profiled durations are diagnostic, not the unprofiled benchmark timings. Caches and clocks are left uncontrolled; raw logs retain profiler warnings.', '',
           '| Profile | Mode | k / feature slab | Achieved occupancy % | no_instruction % | Main-kernel ms |','|---|---|---|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['profile']} | {r['mode']} | {r['k']} / {r['slab']} | {r['achieved_occupancy_pct']:.2f} | {r['no_instruction_pct']:.2f} | {r['duration_ms']:.5f} |")
    lines += ['', 'The `no_instruction` counter describes warps waiting for instruction fetch or an instruction-cache miss; it is not by itself proof of too few runnable CTAs. Compare it with achieved occupancy, the recorded resident-CTA resource bound, CTA nnz distribution, and unprofiled timings. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).', '',
              '[Per-capture provenance and plan statistics](BOO_ROWEXCL_ncu_metadata.json); [summary CSV](BOO_ROWEXCL_ncu_summary.csv).', '']
    (a.out/'BOO_ROWEXCL_NCU.md').write_text('\n'.join(lines))
    print(f'Extracted {len(events)} kernel events from six captures.')


if __name__=='__main__':main()
