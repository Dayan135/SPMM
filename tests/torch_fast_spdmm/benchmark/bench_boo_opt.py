"""Paired BOO ablations and joint parameter sweeps; uses bench_spmm's timing/logging.

python -m benchmark.bench_boo_opt --phase first --out results/boo-first
python -m benchmark.bench_boo_opt --phase parameters --out results/boo-parameters
"""

import argparse
import itertools

from benchmark import bench_spmm


def make_configs(a, phase):
    baseline = bench_spmm.configurations(a)
    result = [r for r in baseline if r['algorithm'] != 'boo']
    if phase == 'plans':
        for k in (32, 64):
            base = dict(algorithm='boo', cluster='hem', target_nr=k, target_nc=k,
                        threads=256, p=64, small_block_threshold=4,
                        small_block_policy='remainder', hem_feasible_first=True,
                        launch_mode='auto', phase=phase)
            result.append(dict(base, config=f'boo_k{k}_auto'))
            for bucketed, precomputed in itertools.product([False, True], repeat=2):
                result.append(dict(base, config=f'boo_k{k}_bucket{int(bucketed)}_rows{int(precomputed)}',
                                   execution_plan=True, bucketed=bucketed, precomputed_rows=precomputed))
        result.append(dict(base, config='boo_original_k32', target_nr=32, target_nc=32,
                           hem_feasible_first=False, launch_mode='legacy'))
        return result
    if phase == 'first':
        grid = itertools.product([(k, k) for k in a.boo_k], [4], [256], [64],
                                 [False, True], ['legacy', 'auto'])
    else:
        grid = itertools.product([(16, 16), (32, 32), (64, 64), (16, 64), (64, 16), (32, 64)],
                                 [4, 16], [128, 256, 512], [32, 64, 128], [True], ['auto'])
        # Same-build legacy baseline, retained regardless of the swept grid.
        result.append(dict(algorithm='boo', config='boo_original_k32', cluster='hem', k=32,
                           target_nr=32, target_nc=32, threads=256, p=64,
                           small_block_threshold=4, small_block_policy='remainder',
                           hem_feasible_first=False, launch_mode='legacy', phase=phase))
    for (nr, nc), threshold, threads, p, feasible, launch in grid:
        name = f'boo_{nr}x{nc}_min{threshold}_t{threads}_p{p}_feasible{int(feasible)}_{launch}'
        result.append(dict(algorithm='boo', config=name, cluster='hem', target_nr=nr,
                           target_nc=nc, threads=threads, p=p, small_block_threshold=threshold,
                           small_block_policy='remainder', hem_feasible_first=feasible,
                           launch_mode=launch, phase=phase))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phase', choices=['first', 'parameters', 'plans'], required=True)
    selected, remaining = parser.parse_known_args()
    bench_spmm.main(remaining, configuration_factory=lambda a: make_configs(a, selected.phase))
