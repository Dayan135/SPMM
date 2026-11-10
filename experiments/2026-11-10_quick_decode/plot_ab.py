"""Plot the quick-decode A/B: original / edited time per arm, range over 3 alternating rounds.

  conda run -n fem python experiments/2026-11-10_quick_decode/plot_ab.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
INK, INK2, GRID, SURFACE, BAND = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#f0efec'
COLORS = {'fem2d_permuted': '#2a78d6', 'roadnet': '#eb6834'}   # validated categorical slots 1-2
DUP_LO, DUP_HI = 0.9921, 1.0112                                  # _dup / base range in this run

s = pd.read_csv(HERE / 'ab_timing.csv')
s = s[~s.tag.str.endswith('_dup')].copy()
s['label'] = s.k + '  ' + s.tag.str.replace('rs_', '').str.replace('_f4', '').str.replace('_', ' ')
order = ['32x32  p64 t256'] + [f'128x64  p{p} t256' for p in (8, 16, 32, 64, 128)] + ['256x64  p64 t512']
y = {lab: i for i, lab in enumerate(reversed(order))}

fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axvspan(0.97, 1.03, color=BAND, zorder=0)
ax.axvspan(DUP_LO, DUP_HI, color='#e4e3df', zorder=0)
ax.axvline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
for j, (inp, color) in enumerate(COLORS.items()):
    g = s[s.input == inp]
    off = 0.14 if j == 0 else -0.14
    for r in g.itertuples():
        yy = y[r.label] + off
        ax.plot([r.min, r.max], [yy, yy], color=color, lw=2, solid_capstyle='round', zorder=2)
        ax.scatter(r.median, yy, s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    last = g.set_index('label').loc[order[-1]]
    ax.annotate(inp, (last['max'], y[order[-1]] + off), textcoords='offset points', xytext=(8, 0),
                va='center', fontsize=9, color=INK)
ax.set_yticks(range(len(order)), list(reversed(order)), fontsize=9)
ax.set_xlim(0.965, 1.035)
ax.set_xlabel('original time / edited time   (> 1: the edit is faster)')
ax.text(0.9705, len(order) - 0.45, 'no-difference band (±3%)', fontsize=8, color=INK2, va='center')
ax.text(DUP_LO + 0.0004, -0.75, 'duplicate-control noise', fontsize=8, color=INK2, va='center')
ax.set_ylim(-1.1, len(order) - 0.2)
ax.grid(axis='x', color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
ax.tick_params(axis='y', length=0)
fig.suptitle('Decoding the column offset once per CTA does not help', x=0.01, ha='left', fontsize=12.5,
             weight='bold', color=INK)
fig.text(0.01, 0.905, 'BOO rowseg, F512, one GPU. Dot: median of 3 alternating A/B rounds; line: min to max.\n'
         'Warp instructions rose 3.6% (fem2d_permuted) and 5% (roadnet).', fontsize=8.5, color=INK2, va='top')
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(HERE / 'ab_timing.png', dpi=160, facecolor=SURFACE)
print(HERE / 'ab_timing.png')
