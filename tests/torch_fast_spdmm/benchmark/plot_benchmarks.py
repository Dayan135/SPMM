"""Fetch SpMM benchmark tables from W&B and render comparison plots.

The benchmark suite logs one ``wandb.Table`` named ``benchmarks`` per run
(one run per system size). This module pulls those tables, concatenates them
into a tidy DataFrame, and renders:

  * per-size facet grids    — rows=profile, cols=features, bars=best-PBR + baselines (log y)
  * speedup heatmap         — best-PBR vs cusparse per (profile, size, features)
  * smem sweep (per size)   — time_ms vs smem_size, lines=block_cfg, dashes=baselines

CLI:
    python -m benchmark.plot_benchmarks --project private-dayan/torch-fast-spdmm \
        --outdir plots/smem_colsort [--runs <run_name> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

TABLE_KEY = "benchmarks"
_BASELINE_ORDER = ["cusparse", "csr_naive"]
_BASE_STYLES = {"cusparse": ("--", "C0"), "csr_naive": (":", "C1")}


# ── Data ────────────────────────────────────────────────────────────────────

def _table_to_df(run) -> pd.DataFrame:
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


def _block_cfg_label(row) -> str:
    return f"{int(row['block_rows'])}x{int(row['block_cols'])}_min{int(row['min_nnz'])}"


def _has_smem_sweep(df: pd.DataFrame) -> bool:
    """True if PBR rows carry multiple smem_size values."""
    if "smem_size" not in df.columns:
        return False
    pbr = df[df["algorithm"].str.startswith("pbr_")]
    return pbr["smem_size"].nunique() > 1


def _best_smem(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse PBR smem variants to the best (min time_ms) per block config.

    Baseline rows pass through unchanged. Used to keep facet_grid readable
    when multiple smem_sizes are present.
    """
    if not _has_smem_sweep(df):
        return df

    pbr = df[df["algorithm"].str.startswith("pbr_")].copy()
    base = df[~df["algorithm"].str.startswith("pbr_")]

    pbr["_cfg"] = pbr.apply(_block_cfg_label, axis=1)
    group = ["commit", "profile", "system_size", "features", "dtype", "_cfg"]
    best = pbr.loc[pbr.groupby(group)["time_ms"].idxmin()].copy()
    best["algorithm"] = "pbr_" + best["_cfg"]
    best = best.drop(columns=["_cfg"])

    return pd.concat([base, best], ignore_index=True)


def _algo_order(algos) -> list[str]:
    present = list(dict.fromkeys(algos))
    baselines = [a for a in _BASELINE_ORDER if a in present]
    pbr = sorted(a for a in present if a not in _BASELINE_ORDER)
    return baselines + pbr


# ── Plots ────────────────────────────────────────────────────────────────────

def facet_grid(df: pd.DataFrame, system_size: int, out_path: Path) -> Path:
    """Bar grid for one size: rows=profile, cols=features, bars=algorithm.

    When multiple smem_sizes are present the PBR bars show the best (min)
    smem_size per block config so the chart stays readable.
    """
    df = _best_smem(df)
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
            times = cell.groupby("algorithm")["time_ms"].min()
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
    fig.suptitle(f"SpMM time_ms — size={system_size} (PBR=best smem)", fontsize=12)
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
            vmin=max(float(np.nanmin(piv.values)), 1e-2),
            vmax=max(float(np.nanmax(piv.values)), 1.0)))
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


def smem_sweep(df: pd.DataFrame, system_size: int, out_path: Path) -> Path:
    """Line plot of time_ms vs smem_size per block config, for one system_size.

    rows=profile, cols=features.
    Colored lines = PBR block configs (6 combos).
    Horizontal dashed lines = baselines (cusparse, csr_naive).
    y-axis log-scaled so baselines and PBR are directly comparable.
    """
    sub = df[df["system_size"] == system_size].copy()
    pbr = sub[sub["algorithm"].str.startswith("pbr_")].copy()
    base = sub[sub["algorithm"].isin(_BASELINE_ORDER)]

    if pbr.empty or not _has_smem_sweep(sub):
        return out_path

    pbr["block_cfg"] = pbr.apply(_block_cfg_label, axis=1)
    smem_vals = sorted(pbr["smem_size"].dropna().unique())
    cfgs = sorted(pbr["block_cfg"].unique())
    profiles = sorted(sub["profile"].unique())
    feats = sorted(sub["features"].unique())

    cfg_colors = {c: col for c, col in zip(cfgs, plt.cm.tab10.colors)}

    fig, axes = plt.subplots(
        len(profiles), len(feats),
        figsize=(3.5 * len(feats), 2.8 * len(profiles)),
        squeeze=False, sharey="row",
    )

    for i, profile in enumerate(profiles):
        for j, feat in enumerate(feats):
            ax = axes[i][j]
            cell_pbr = pbr[(pbr["profile"] == profile) & (pbr["features"] == feat)]
            cell_base = base[(base["profile"] == profile) & (base["features"] == feat)]

            for cfg in cfgs:
                row = (cell_pbr[cell_pbr["block_cfg"] == cfg]
                       .groupby("smem_size")["time_ms"].mean()
                       .reindex(smem_vals))
                if row.isna().all():
                    continue
                ax.plot(smem_vals, row.values, marker="o", color=cfg_colors[cfg],
                        linewidth=1.4, markersize=4, label=cfg)

            for algo, (ls, color) in _BASE_STYLES.items():
                brow = cell_base[cell_base["algorithm"] == algo]["time_ms"]
                if not brow.empty:
                    ax.axhline(brow.mean(), linestyle=ls, color=color,
                               linewidth=1.2, label=algo, alpha=0.85)

            ax.set_yscale("log")
            ax.set_xticks(smem_vals)
            ax.set_xticklabels([str(int(s)) for s in smem_vals], fontsize=7)
            if j == 0:
                ax.set_ylabel(profile, fontsize=7)
            if i == 0:
                ax.set_title(f"features={feat}", fontsize=9)
            if i == len(profiles) - 1:
                ax.set_xlabel("smem_size (floats)", fontsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=min(len(by_label), 8),
               fontsize=7, bbox_to_anchor=(0.5, 0))
    fig.suptitle(f"time_ms vs smem_size  —  system_size={system_size}", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
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
    sizes = sorted(df["system_size"].unique())
    written = []

    for size in sizes:
        written.append(facet_grid(df, size, outdir / f"facet_size{size}.png"))

    written.append(speedup_heatmap(df, outdir / "speedup.png"))

    if _has_smem_sweep(df):
        for size in sizes:
            written.append(smem_sweep(df, size, outdir / f"smem_sweep_size{size}.png"))

    print(f"rows={len(df)}  sizes={sizes}")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
