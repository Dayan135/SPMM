"""Report bench_spmm results, retaining configuration choices and millisecond units.

This is an offline report/plot command, safe to run locally without CUDA:
    python report_spmm.py results/spmm/summary.csv --plot

The original summary.csv contains every configuration. This report explicitly
selects the fastest measured configuration per family, separately per workload.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


ALGORITHMS = ('cusparse_csr', 'pbr', 'boo')
LABELS = ('cuSPARSE CSR', 'PBR (best measured config)', 'BOO (best measured config)')


def load_cells(path):
    cells = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            key = row['profile'], int(row['requested_size']), int(row['features'])
            cells[key].append(row)
    if not cells:
        raise ValueError('No completed workloads')
    result = {}
    for key, rows in cells.items():
        if any(r['correctness'] != 'passed' for r in rows):
            raise ValueError(f'Unvalidated result in {key}')
        for field in ('matrix_sha256', 'x_sha256', 'source_sha256', 'rows', 'cols'):
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f'Mismatched {field} in {key}')
        if len({r['config'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate configurations in {key}')
        groups = {a: [r for r in rows if r['algorithm'] == a] for a in ALGORITHMS}
        if len(groups['cusparse_csr']) != 1 or any(not g for g in groups.values()):
            raise ValueError(f'Incomplete algorithm comparison in {key}')
        result[key] = {a: min(group, key=lambda r: float(r['median_ms'])) for a, group in groups.items()}
    return result


def report(path, plot=False):
    cells = load_cells(path)
    selected = []
    lines = ['# PBR / BOO / cuSPARSE CSR results', '',
             'Times are medians in milliseconds. PBR and BOO are the **best measured configuration** '
             'in each workload, selected retrospectively; all configurations remain in `summary.csv`.', '',
             '| Profile | Requested N | Features | CSR ms | PBR ms | PBR config | BOO ms | BOO config |',
             '|---|---:|---:|---:|---:|---|---:|---|']
    for key in sorted(cells, key=lambda k: (k[1], k[2], k[0])):
        group = cells[key]
        csr, pbr, boo = [group[a] for a in ALGORITHMS]
        row = dict(profile=key[0], requested_size=key[1], rows=csr['rows'], cols=csr['cols'],
                   features=key[2], csr_ms=float(csr['median_ms']),
                   pbr_ms=float(pbr['median_ms']), pbr_config=pbr['config'],
                   boo_ms=float(boo['median_ms']), boo_config=boo['config'])
        selected.append(row)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {row['csr_ms']:.5f} | "
                     f"{row['pbr_ms']:.5f} | {row['pbr_config']} | {row['boo_ms']:.5f} | {row['boo_config']} |")
    meta_path = path.parent / 'metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        lines[2:2] = [f"Run status: **{meta['status']}**. GPU: {meta['gpu']}. "
                       f"[W&B run]({meta['wandb_url']}).", '']
    if plot:
        lines[2:2] = ['![Median matmul timings](comparison.png)', '']
    (path.parent / 'comparison.md').write_text('\n'.join(lines) + '\n')
    with (path.parent / 'best_measured.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        sizes = sorted({k[1] for k in cells})
        widths = sorted({k[2] for k in cells})
        fig, axes = plt.subplots(len(sizes), len(widths), squeeze=False,
                                 figsize=(9 * len(widths), 7 * len(sizes)))
        colors = ('#72757e', '#2878b5', '#e58b25')
        for si, size in enumerate(sizes):
            for fi, width in enumerate(widths):
                ax = axes[si, fi]
                profiles = sorted(k[0] for k in cells if k[1:] == (size, width))
                y = np.arange(len(profiles))
                for ai, algorithm in enumerate(ALGORITHMS):
                    values = [cells[p, size, width][algorithm] for p in profiles]
                    medians = np.array([float(v['median_ms']) for v in values])
                    low = np.array([float(v['p25_ms']) for v in values])
                    high = np.array([float(v['p75_ms']) for v in values])
                    ax.errorbar(medians, y + (ai - 1) * 0.25, fmt='o', markersize=4,
                                color=colors[ai], label=LABELS[ai],
                                xerr=np.array([medians - low, high - medians]),
                                elinewidth=1, capsize=2)
                ax.set_yticks(y, profiles)
                ax.invert_yaxis()
                ax.set_xscale('log')
                ax.set_xlabel('Median matmul time (ms, log scale; quartile error bars)')
                ax.set_title(f'Requested N={size:,}, features={width}')
                ax.grid(axis='x', alpha=0.2)
                ax.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        height = fig.get_figheight()
        fig.legend(handles, labels, loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, 1 - 0.4 / height))
        fig.suptitle('PBR / BOO / cuSPARSE CSR — best measured configurations',
                     y=1 - 0.1 / height)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / height))
        fig.savefig(path.parent / 'comparison.png', dpi=160)
        plt.close(fig)
    print(f'Wrote comparison.md and best_measured.csv for {len(cells)} workloads in {path.parent}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    report(args.summary, args.plot)
