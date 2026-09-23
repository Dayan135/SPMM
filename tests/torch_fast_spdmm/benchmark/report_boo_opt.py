"""Offline analysis of paired BOO experiments; no extension or GPU required.

python report_boo_opt.py results/boo_optimization_20260923 --plot
Input subdirectories: first, parameters, plans. Missing phases are marked pending.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def number(row, key):
    return float(row.get(key) or 0)


def cells(path):
    groups = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            assert row['correctness'] == 'passed'
            assert int(row['repeats']) == 20 and int(row['warmup']) == 5
            key = row['profile'], int(row['requested_size']), int(row['features'])
            assert row['config'] not in groups[key]
            groups[key][row['config']] = row
    for group in groups.values():
        for field in ('matrix_sha256', 'x_sha256', 'source_sha256'):
            assert len({r[field] for r in group.values()}) == 1
    return groups


def analyze(root, plot=False):
    pairs, selected = [], []
    lines = ['# BOO optimization measurements', '',
             'All elapsed times are milliseconds. Every configuration has 5 warmups and 20 retained '
             'measurements; comparisons use medians. Twenty repeats do not eliminate outliers. '
             'Quartiles and all samples remain available in each phase directory.', '',
             'Speedups compare configurations in the same workload and process. Ratios above 1 favor '
             'the candidate. “Separated IQR” is descriptive, not a significance test. '
             'Best measured configurations are retrospective selections, not a validated selector.', '']

    def add(phase, key, label, baseline, candidate, k=''):
        before, after = number(baseline, 'median_ms'), number(candidate, 'median_ms')
        setup_before = sum(number(baseline, f) for f in ('build_ms', 'staging_ms', 'plan_build_ms', 'plan_staging_ms'))
        setup_after = sum(number(candidate, f) for f in ('build_ms', 'staging_ms', 'plan_build_ms', 'plan_staging_ms'))
        additional_setup = max(0, setup_after - setup_before)
        pairs.append(dict(phase=phase, profile=key[0], requested_size=key[1], features=key[2],
                          experiment=label, k=k, baseline_config=baseline['config'],
                          candidate_config=candidate['config'], baseline_ms=before, candidate_ms=after,
                          baseline_p25_ms=number(baseline, 'p25_ms'), baseline_p75_ms=number(baseline, 'p75_ms'),
                          candidate_p25_ms=number(candidate, 'p25_ms'), candidate_p75_ms=number(candidate, 'p75_ms'),
                          speedup=before / after, saved_ms=before - after,
                          separated_iqr=number(candidate, 'p75_ms') < number(baseline, 'p25_ms'),
                          baseline_setup_ms=setup_before, candidate_setup_ms=setup_after,
                          additional_setup_ms=additional_setup,
                          amortization_calls=math.ceil(additional_setup / (before - after)) if before > after else '',
                          plan_extra_device_bytes=candidate.get('plan_extra_device_bytes', ''),
                          remainder_nnz=candidate.get('remainder_nnz', '')))

    for phase in ('first', 'parameters', 'plans'):
        path = root / phase / 'summary.csv'
        if not path.exists():
            lines.extend([f'## {phase}: pending', ''])
            continue
        meta = json.loads((path.parent / 'metadata.json').read_text())
        grouped = cells(path)
        lines.extend([f'## {phase}', '',
                      f"Status: **{meta['status']}**; {len(grouped)} workloads. "
                      f"[W&B run]({meta['wandb_url']}). Source SHA256 `{meta['source_sha256']}`.", ''])
        for key, group in grouped.items():
            if phase == 'first':
                for k in (8, 16, 32, 64):
                    stem = f'boo_{k}x{k}_min4_t256_p64_feasible'
                    old = group[stem + '0_legacy']
                    add(phase, key, 'feasible partners', old, group[stem + '1_legacy'], k)
                    add(phase, key, 'auto launch', old, group[stem + '0_auto'], k)
                    add(phase, key, 'both', old, group[stem + '1_auto'], k)
                baseline, candidate = group['boo_32x32_min4_t256_p64_feasible0_legacy'], group['boo_32x32_min4_t256_p64_feasible1_auto']
            elif phase == 'parameters':
                baseline = group['boo_original_k32']
                candidate = min((r for r in group.values() if r['algorithm'] == 'boo' and r is not baseline),
                                key=lambda r: number(r, 'median_ms'))
                add(phase, key, 'best measured vs original', baseline, candidate)
                fixed = group['boo_32x32_min4_t256_p64_feasible1_auto']
                add(phase, key, 'best measured vs fixed candidate', fixed, candidate)
            else:
                for k in (32, 64):
                    auto, plan = group[f'boo_k{k}_auto'], group[f'boo_k{k}_bucket0_rows0']
                    add(phase, key, 'plan wrapper', auto, plan, k)
                    for name, suffix in (('buckets', 'bucket1_rows0'), ('row boundaries', 'bucket0_rows1'),
                                         ('buckets and boundaries', 'bucket1_rows1')):
                        add(phase, key, name + ' vs plan', plan, group[f'boo_k{k}_{suffix}'], k)
                        add(phase, key, name + ' vs auto', auto, group[f'boo_k{k}_{suffix}'], k)
                baseline, candidate = group['boo_k32_auto'], group['boo_k32_bucket1_rows1']
            csr = group['cusparse_csr']
            pbr = min((r for r in group.values() if r['algorithm'] == 'pbr'), key=lambda r: number(r, 'median_ms'))
            selected.append(dict(phase=phase, profile=key[0], requested_size=key[1], features=key[2],
                                 baseline_ms=number(baseline, 'median_ms'), candidate_ms=number(candidate, 'median_ms'),
                                 csr_ms=number(csr, 'median_ms'), pbr_best_ms=number(pbr, 'median_ms'),
                                 candidate_config=candidate['config']))
        subsets = defaultdict(list)
        for row in pairs:
            if row['phase'] == phase:
                subsets[row['experiment']].append(row)
        lines.extend(['| Experiment | Pairs | Geometric speedup | Median wins | Separated IQR wins |',
                      '|---|---:|---:|---:|---:|'])
        for experiment, subset in subsets.items():
            gm = math.exp(statistics.mean(math.log(r['speedup']) for r in subset))
            lines.append(f"| {experiment} | {len(subset)} | {gm:.4f}x | "
                         f"{sum(r['speedup'] > 1 for r in subset)} | {sum(r['separated_iqr'] for r in subset)} |")
        lines.extend(['', 'Individual workload medians (fixed k=32 comparison for first/plans; '
                      'best measured grid configuration for parameters):', '',
                      '| Profile | N | F | Baseline ms | Candidate ms | CSR ms | Best PBR ms | Candidate |',
                      '|---|---:|---:|---:|---:|---:|---:|---|'])
        for row in selected:
            if row['phase'] == phase:
                lines.append(f"| {row['profile']} | {row['requested_size']} | {row['features']} | "
                             f"{row['baseline_ms']:.5f} | {row['candidate_ms']:.5f} | {row['csr_ms']:.5f} | "
                             f"{row['pbr_best_ms']:.5f} | {row['candidate_config']} |")
        lines.append('')
    for name, rows in (('paired_comparisons.csv', pairs), ('selected_comparisons.csv', selected)):
        if rows:
            with (root / name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if plot and selected:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        phases = sorted({r['phase'] for r in selected})
        fig, axes = plt.subplots(1, len(phases), figsize=(8 * len(phases), 12), squeeze=False)
        for ax, phase in zip(axes[0], phases):
            subset = [r for r in selected if r['phase'] == phase]
            for i, row in enumerate(subset):
                ax.plot([row['baseline_ms'], row['candidate_ms']], [i, i], color='#bdbdbd', linewidth=1)
            for field, label, color in (('baseline_ms', 'Baseline', '#777777'), ('candidate_ms', 'Candidate', '#2878b5')):
                ax.scatter([r[field] for r in subset], range(len(subset)), s=18, label=label, color=color)
            ax.set_yticks(range(len(subset)), [f"{r['profile']} N={r['requested_size']} F={r['features']}" for r in subset], fontsize=8)
            ax.invert_yaxis()
            ax.set_xscale('log')
            ax.set_xlabel('Median complete matmul time (ms, log scale)')
            ax.set_title(phase + (' — retrospective best' if phase == 'parameters' else ' — fixed k=32'))
            ax.grid(axis='x', alpha=0.2)
            ax.legend()
        fig.tight_layout()
        fig.savefig(root / 'optimization.png', dpi=160)
        plt.close(fig)
        lines[2:2] = ['![Paired median timings](optimization.png)', '']
    (root / 'optimization_report.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote {len(pairs)} paired comparisons and {len(selected)} selected workloads to {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    analyze(args.root, args.plot)
