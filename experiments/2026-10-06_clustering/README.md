# Clustering backends — 2026-10-06

The [experiment report](CLUSTERING_EXPERIMENT.md) compares identity, HEM,
METIS, and Graclus groupings for BOO. It separates offline grouping cost from
GPU multiplication time.

- Driver: [`bench_clustering.py`](../../tests/torch_fast_spdmm/benchmark/bench_clustering.py)
- Plotter: [`plot_clustering.py`](../../tests/torch_fast_spdmm/benchmark/plot_clustering.py)
- Figures: [initial sweep](plots/initial/) and [larger sizes](plots/xl/)
- Results: [initial](../../tests/torch_fast_spdmm/benchmark/results/clustering_20261006/)
  and [large sweep](../../tests/torch_fast_spdmm/benchmark/results/clustering_20261006_xl/)
- Saved labels and samples: [archive](../../RESULTS_ARCHIVE.md)
