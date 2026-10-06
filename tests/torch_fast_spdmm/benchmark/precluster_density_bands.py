"""CPU-only pre-clustering for bench_density_bands.py's BOO configs. Runs
locally (no GPU, no Slurm) -- pure numpy/scipy, same pattern as this
session's earlier clustering-cost calibration. Saves row_labels/col_labels
per (profile, k) to --out, so the GPU-timing job can load them and skip
HEM clustering entirely (the actual CPU-bound cost that triggered the
cluster's low-GPU-utilization auto-cancellation policy on two of three
replicate jobs).

Clustering is seed-deterministic and shared across all 3 replicates (same
seed=0 matrix every time) -- run this once, use the output for all three.

Usage (run locally, NOT on the cluster):
    conda run -n fem python precluster_density_bands.py \\
        --sizes 1048576 --profiles <names...> --out precluster_cache
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import time

import numpy as np

_HERE = Path(__file__).parent
_clustering_spec = importlib.util.spec_from_file_location(
    "clustering", str(_HERE.parent.parent.parent / "torch_fast_spdmm" / "torch_fast_spdmm" / "clustering.py"))
clustering = importlib.util.module_from_spec(_clustering_spec)
_clustering_spec.loader.exec_module(clustering)

_density_spec = importlib.util.spec_from_file_location("density_profiles", str(_HERE / "density_profiles.py"))
density_profiles = importlib.util.module_from_spec(_density_spec)
_density_spec.loader.exec_module(density_profiles)

# Must match bench_spmm.py's configurations()'s BOO config exactly --
# hem_feasible_first=False there (the "original" baseline BOO config kept
# for comparability with earlier benchmarks), NOT csr_to_boo's current
# default (True). Getting this wrong produces a different cluster grid than
# the real timing run would, silently invalidating the whole point of
# precomputing it.
BOO_KS = (8, 16, 32, 64)
HEM_FEASIBLE_FIRST = False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes', type=int, nargs='+', required=True)
    p.add_argument('--profiles', nargs='+', choices=list(density_profiles.DENSITY_PROFILES), required=True)
    p.add_argument('--ks', type=int, nargs='+', default=list(BOO_KS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    for size in a.sizes:
        for profile in a.profiles:
            mat, band, note = density_profiles.build_density_matrix(size, profile, seed=a.seed)
            print(f'{profile} @ N={size} (actual N={mat.shape[0]}, nnz={mat.nnz}, band={band})', flush=True)
            for k in a.ks:
                out_path = a.out / f'{profile.replace("@", "_at_")}_N{mat.shape[0]}_k{k}.npz'
                if out_path.exists():
                    print(f'  k={k}: already cached, skipping', flush=True)
                    continue
                t0 = time.perf_counter()
                row_labels, col_labels = clustering.cluster_axes(
                    mat, method='hem', target_nr=k, target_nc=k,
                    max_nr=k, max_nc=k, max_block_nnz=None, seed=a.seed,
                    symmetric=None, feasible_first=HEM_FEASIBLE_FIRST)
                dt = time.perf_counter() - t0
                np.savez(out_path, row_labels=row_labels.astype(np.int64),
                        col_labels=col_labels.astype(np.int64))
                print(f'  k={k}: {dt:.1f}s -> {out_path.name}', flush=True)


if __name__ == '__main__':
    main()
