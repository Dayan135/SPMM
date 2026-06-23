"""Fetch SpMM benchmark tables from W&B and render comparison plots.

The benchmark suite logs one ``wandb.Table`` named ``benchmarks`` per run
(one run per system size). This module pulls those tables, concatenates them
into a tidy DataFrame, and renders:

  * per-size facet grids  — rows=profile, cols=features, bars=algorithm (log y)
  * a speedup view        — best-PBR vs cusparse per (profile, size, features)

Everything is split into small reusable functions so individual pieces
(``fetch_results``, ``facet_grid``, ``speedup_heatmap``) can be imported and
reused from notebooks or other scripts.

CLI:
    python -m benchmark.plot_benchmarks --project private-dayan/torch-fast-spdmm \
        --outdir plots [--runs ede9b48_16384 ede9b48_65536 ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

TABLE_KEY = "benchmarks"
# Baselines first, then PBR variants — keeps bar/column order stable across cells.
_BASELINE_ORDER = ["cusparse", "csr_naive"]


# ── Data ────────────────────────────────────────────────────────────────────

def _table_to_df(run) -> pd.DataFrame:
    """Download the logged ``benchmarks`` table from one run as a DataFrame."""
    ref = run.summary.get(TABLE_KEY)
    if ref is None:
        return pd.DataFrame()
    path = ref["path"] if isinstance(ref, dict) else ref.get("path")
    local = run.file(path).download(replace=True, root=f"/tmp/wandb_tables/{run.id}")
    with open(local.name) as fh:
        payload = json.load(fh)
    df = pd.DataFrame(payload["data"], columns=payload["columns"])
    df["run"] = run.name
    return df


def fetch_results(project: str, runs: list[str] | None = None) -> pd.DataFrame:
    """Concatenate the benchmark tables of the given project into one DataFrame.

    ``runs`` optionally restricts to specific run names; otherwise every run in
    the project is pulled.
    """
    api = wandb.Api()
    frames = []
    for run in api.runs(project):
        if runs and run.name not in runs:
            continue
        df = _table_to_df(run)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise SystemExit(f"no '{TABLE_KEY}' tables found in {project}")
    return pd.concat(frames, ignore_index=True)


def _algo_order(algos) -> list[str]:
    """Baselines first (fixed order), then PBR variants sorted by name."""
    present = list(dict.fromkeys(algos))
    baselines = [a for a in _BASELINE_ORDER if a in present]
    pbr = sorted(a for a in present if a not in _BASELINE_ORDER)
    return baselines + pbr


# ── Plots ───────────────────────────────────────────────────────────────────

def facet_grid(df: pd.DataFrame, system_size: int, out_path: Path) -> Path:
    """Bar grid for one size: rows=profile, cols=features, bars=algorithm."""
    sub = df[df["system_size"] == system_size]
    profiles = sorted(sub["profile"].unique())
    feats = sorted(sub["features"].unique())
    algos = _algo_order(sub["algorithm"])
    colors = {a: c for a, c in zip(algos, plt.cm.tab10.colors)}

    fig, axes = plt.subplots(
        len(profiles), len(feats),
        figsize=(3.2 * len(feats), 2.4 * len(profiles)),
        squeeze=False, sharey=True,
    )
    for i, profile in enumerate(profiles):
        for j, feat in enumerate(feats):
            ax = axes[i][j]
            cell = sub[(sub["profile"] == profile) & (sub["features"] == feat)]
            times = cell.set_index("algorithm")["time_ms"]
            present = [a for a in algos if a in times.index]
            ax.bar(range(len(present)), [times[a] for a in present],
                   color=[colors[a] for a in present])
            ax.set_yscale("log")
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels(present, rotation=90, fontsize=6)
            if j == 0:
                ax.set_ylabel(profile, fontsize=8)
            if i == 0:
                ax.set_title(f"features={feat}", fontsize=9)
    fig.suptitle(f"SpMM time_ms — system_size={system_size}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def speedup_heatmap(df: pd.DataFrame, out_path: Path, baseline: str = "cusparse") -> Path:
    """Heatmap of best-PBR speedup over ``baseline`` per (profile, features), per size."""
    pbr = df[df["algorithm"].str.startswith("pbr_")]
    base = df[df["algorithm"] == baseline]
    best = pbr.groupby(["system_size", "profile", "features"])["time_ms"].min()
    base = base.set_index(["system_size", "profile", "features"])["time_ms"]
    speedup = (base / best).rename("speedup").reset_index()

    sizes = sorted(speedup["system_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.5 * len(sizes), 4.5), squeeze=False)
    for k, size in enumerate(sizes):
        ax = axes[0][k]
        piv = speedup[speedup["system_size"] == size].pivot(
            index="profile", columns="features", values="speedup")
        im = ax.imshow(piv.values, cmap="RdYlGn", norm=matplotlib.colors.LogNorm(
            vmin=max(piv.values.min(), 1e-2), vmax=max(piv.values.max(), 1.0)))
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_title(f"size={size}", fontsize=10)
        ax.set_xlabel("features")
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                v = piv.values[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"best-PBR speedup over {baseline} (>1 = PBR faster)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="private-dayan/torch-fast-spdmm")
    p.add_argument("--outdir", default="plots")
    p.add_argument("--runs", nargs="*", default=None,
                   help="restrict to these run names (default: all)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = fetch_results(args.project, args.runs)
    written = []
    for size in sorted(df["system_size"].unique()):
        written.append(facet_grid(df, size, outdir / f"facet_size{size}.png"))
    written.append(speedup_heatmap(df, outdir / "speedup.png"))

    print(f"rows={len(df)}  sizes={sorted(df['system_size'].unique())}")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
