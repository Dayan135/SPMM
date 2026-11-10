"""Plot: lowering P restores CTAs/SM but slows the kernel (block-shape screen + kp data).

  conda run -n fem python scripts/research_panel/plot_p_occupancy.py \
      --raw experiments/2026-11-06_block_shape/analysis/raw_gpu --out experiments/2026-11-06_block_shape/analysis/confirm

For each case where the frozen winner loses a CTA to shared memory, plots threads/SM against speedup
(best 32x32 in the same allocation / arm, minimum over allocations, arms on >= 2 GPUs). Filled points
are T=256 at each P (best FPT), joined in P order; the hollow point is the frozen P with T=512, the
other way to refill the SM. Writes p_occupancy.png and p_occupancy.csv.
"""
import argparse
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT), str(ROOT / 'scripts' / 'research_panel')]
import shape_report as sr  # noqa: E402

# (input, F, shape, frozen P) -- the confirmation winners that lose a CTA/SM to shared memory
CASES = [('diag_b1.0', 128, '64x32', 128), ('diag_dd0.05', 512, '128x64', 64), ('diag_dd0.05', 128, '64x64', 64)]
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']   # validated categorical slots 1-3 (light surface)
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def grid(raw):
    with tempfile.TemporaryDirectory() as tmp:
        d = pd.concat([sr.load(raw / 'results', 'screen'), sr.load(raw / 'results', 'kp')], ignore_index=True)
        d = sr.accept(d, raw / 'launch_audit', 'screen', Path(tmp))
    d = d[d.suffix == '']
    ref = d[d.k == '32x32'].groupby(['alloc', 'input', 'features']).median_ms.min().rename('ref')
    d = d.join(ref, on=['alloc', 'input', 'features'])
    d['sp'] = d.ref / d.median_ms
    s = (d.groupby(['input', 'features', 'k', 'p', 'threads', 'fpt'])
         .agg(speedup=('sp', 'min'), gpus=('gpu_uuid', 'nunique'), cta=('max_active_blocks_per_sm', 'first'),
              arena=('arena_bytes', 'first')).reset_index())
    s = s[s.gpus >= 2]
    s['threads_pct'] = 100 * s.cta * s.threads / 1536
    return s.loc[s.groupby(['input', 'features', 'k', 'p', 'threads']).speedup.idxmax()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    s = grid(a.raw)
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey='row', facecolor=SURFACE,
                             gridspec_kw=dict(height_ratios=[1, 1.5]))
    rows = []
    ps = [32, 64, 128]
    for j, ((inp, f, k, p0), color) in enumerate(zip(CASES, COLORS)):
        top, bot = axes[0, j], axes[1, j]
        c = s[(s.input == inp) & (s.features == f) & (s.k == k)]
        t = c[c.threads == 256].set_index('p').reindex(ps)
        alt = c[(c.threads == 512) & (c.p == p0)]
        rows += [t.reset_index().assign(series='T256'), alt.assign(series='T512_frozen_P')]
        x = range(len(ps))
        top.bar(x, t.cta, width=0.55, color=color, zorder=2)
        for i, r in enumerate(t.itertuples()):
            top.text(i, r.cta + 0.15, f'{r.cta:.0f} CTAs\n{r.threads_pct:.0f}% threads', ha='center',
                     va='bottom', fontsize=8.5, color=INK)
        top.set_title(f'{inp}, F{f}, block {k}', fontsize=10.5, color=INK, loc='left')
        bot.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        bot.plot(x, t.speedup, color=color, lw=2, zorder=2)
        for i, r in enumerate(t.itertuples()):
            frozen = ps[i] == p0
            bot.scatter(i, r.speedup, s=120 if frozen else 60, color=color, zorder=3, edgecolor=SURFACE, lw=2)
            if frozen:
                bot.annotate(f'frozen pick\n{r.speedup:.3f}', (i, r.speedup), textcoords='offset points',
                             xytext=(-6, 9), ha='right', va='bottom', fontsize=8.5, color=INK, weight='bold')
            else:
                bot.annotate(f'{r.speedup:.3f}', (i, r.speedup), textcoords='offset points', xytext=(0, -10),
                             ha='center', va='top', fontsize=8.5, color=INK)
        for r in alt.itertuples():
            i = ps.index(r.p) + 0.28
            bot.scatter(i, r.speedup, s=70, facecolor=SURFACE, edgecolor=color, lw=2, zorder=3)
            bot.annotate(f'T512: {r.speedup:.3f}\n{r.cta:.0f} CTAs, 100%', (i, r.speedup),
                         textcoords='offset points', xytext=(7, 0), va='center', fontsize=8, color=INK2)
        for ax in (top, bot):
            ax.set_facecolor(SURFACE)
            ax.grid(axis='y', color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        bot.set_xticks(list(x), [f'P{p}' for p in ps])
        bot.set_xlim(-0.5, 2.9)
        bot.set_xlabel('feature-tile width P  (T = 256)')
    axes[0, 0].set_ylabel('CTAs per SM')
    axes[0, 0].set_ylim(0, 8.5)
    axes[1, 0].set_ylabel('speedup vs tuned 32x32\n(min over GPUs)')
    axes[1, 0].set_ylim(0.8, 1.16)
    axes[1, 0].text(-0.45, 1.003, 'tuned 32x32', color=INK2, fontsize=8, va='bottom')
    fig.suptitle('Lowering P restores CTAs/SM, but the kernel gets slower', x=0.01, ha='left', fontsize=12.5,
                 color=INK, weight='bold')
    fig.text(0.01, 0.925, 'The three confirmation winners that lose a CTA to shared memory. Filled: T=256, best FPT '
             'per P. Hollow: the frozen P with T=512, which also refills the SM. Screen + kp data, >= 2 GPUs.',
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / 'p_occupancy.png', dpi=160, facecolor=SURFACE)
    cols = ['input', 'features', 'k', 'series', 'p', 'threads', 'fpt', 'cta', 'threads_pct', 'arena', 'speedup', 'gpus']
    pd.concat(rows)[cols].round(4).to_csv(a.out / 'p_occupancy.csv', index=False)
    print(f"-> {a.out / 'p_occupancy.png'}")


if __name__ == '__main__':
    main()
