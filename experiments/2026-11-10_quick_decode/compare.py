"""A/B compare for the quick decode edit. Usage: python compare.py LOCAL_DIR (holds results/ and profile/)."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path('/Users/dayan/local/research/SPMM-boo')
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT), str(ROOT / 'scripts' / 'research_panel')]
import shape_report as sr  # noqa: E402
import shape_profile_report as spr  # noqa: E402

pd.set_option('display.width', 220, 'display.max_columns', 30)
D = Path(sys.argv[1])
d = sr.arm_tag(sr.load(D / 'results', 'quick_decode'))
bad = d[d.status != 'measured']
print(f'arms: {len(d)}, not measured: {len(bad)}; correctness failed: {int((d.correctness == "failed").sum())}; '
      f'launch: {d.launch_verified.str[:10].value_counts().to_dict()}')
if len(bad):
    print(bad[['alloc', 'arm', 'status', 'reason']].to_string(index=False))
m = d[d.status == 'measured'].copy()
m['version'] = m.alloc.str.extract(r'alloc([AB])r')[0]
m['round'] = m.alloc.str.extract(r'r(\d)$')[0].astype(int)
p = m.pivot_table(index=['input', 'k', 'tag', 'round'], columns='version', values='median_ms').reset_index()
p['A_over_B'] = p.A / p.B
s = p.groupby(['input', 'k', 'tag']).agg(A_ms=('A', 'median'), B_ms=('B', 'median'), min=('A_over_B', 'min'),
                                          median=('A_over_B', 'median'), max=('A_over_B', 'max')).reset_index()
s['verdict'] = ['WIN' if lo > 1.03 else ('REGRESSION' if hi < 0.97 else 'WASH') for lo, hi in zip(s['min'], s['max'])]
print('\nA = original, B = decoded once per CTA. A/B > 1 means the edit is faster (3 rounds, alternating order)')
print(s.round(4).to_string(index=False))
dup = m[m.suffix == 'dup'].merge(m[m.suffix == ''][['alloc', 'input', 'k', 'base', 'median_ms']],
                                 on=['alloc', 'input', 'k', 'base'], suffixes=('', '_base'))
print(f"\n_dup / base (noise floor): {(dup.median_ms / dup.median_ms_base).describe()[['min', '50%', 'max']].round(4).to_dict()}")
s.to_csv(D / 'ab_timing.csv', index=False)

rows = []
for v in ('A', 'B'):
    for raw in sorted((D / 'profile' / v).glob('*_raw.csv')):
        k = pd.DataFrame(spr.kernels(raw, raw.name[:-8]))
        mk = k.loc[k.duration_ns.idxmax()]
        rows.append(dict(version=v, label=raw.name[:-8], us=mk.duration_ns / 1e3, inst_M=mk.warp_inst_executed / 1e6,
                         shld_M=mk.shared_ld_wavefronts / 1e6, issue_pct=mk.issue_active_pct,
                         occ=mk.achieved_occupancy_pct, regs=mk.registers_per_thread,
                         smem_kb=mk.smem_allocated_bytes / 1024,
                         **{s: mk.get(f'stall_{s}') for s in ('long_scoreboard', 'barrier', 'short_scoreboard',
                                                              'mio_throttle', 'wait', 'not_selected')}))
if rows:
    pr = pd.DataFrame(rows)
    w = pr.pivot(index='label', columns='version')
    out = pd.DataFrame({c: [f'{a:.2f} -> {b:.2f}' for a, b in zip(w[(c, 'A')], w[(c, 'B')])]
                        for c in pr.columns if c not in ('version', 'label')}, index=w.index)
    print('\nncu, dominant kernel, A -> B (durations diagnostic; stalls per issued instruction)')
    print(out.to_string())
    pr.to_csv(D / 'ab_profile.csv', index=False)
