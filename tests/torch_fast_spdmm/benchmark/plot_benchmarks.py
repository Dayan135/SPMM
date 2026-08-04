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
import itertools
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
    # W&B allows multiple runs to share the same name (e.g. a one-off sanity
    # check re-run under the same name as a later full sweep) -- keep only the
    # most recently created run per name so its data isn't double-counted
    # (duplicate rows break the profile/features pivot in speedup_heatmap etc).
    by_name: dict[str, "wandb.apis.public.Run"] = {}
    for run in api.runs(project):
        if runs and run.name not in runs:
            continue
        existing = by_name.get(run.name)
        if existing is None or run.created_at > existing.created_at:
            by_name[run.name] = run

    frames = []
    for run in by_name.values():
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
    when multiple smem_sizes are present. When multiple `kernel` values are
    present (e.g. zero_idle + nnzflat), kernel is kept in the grouping key so
    variants aren't silently collapsed into one "best of all kernels" bar --
    harmless no-op when there's only ever been one kernel value, as in this
    project's history until the zero_idle/nnzflat sweep.
    """
    if not _has_smem_sweep(df):
        return df

    pbr = df[df["algorithm"].str.startswith("pbr_")].copy()
    base = df[~df["algorithm"].str.startswith("pbr_")]

    pbr["_cfg"] = pbr.apply(_block_cfg_label, axis=1)
    group = ["commit", "profile", "system_size", "features", "dtype", "_cfg"]
    multi_kernel = "kernel" in pbr.columns and pbr["kernel"].nunique() > 1
    if multi_kernel:
        group = group + ["kernel"]

    best = pbr.loc[pbr.groupby(group)["time_ms"].idxmin()].copy()
    if multi_kernel:
        best["algorithm"] = "pbr_" + best["_cfg"] + "_" + best["kernel"].astype(str)
    else:
        best["algorithm"] = "pbr_" + best["_cfg"]
    best = best.drop(columns=["_cfg"])

    return pd.concat([base, best], ignore_index=True)


def _algo_order(algos) -> list[str]:
    present = list(dict.fromkeys(algos))
    baselines = [a for a in _BASELINE_ORDER if a in present]
    pbr = sorted(a for a in present if a not in _BASELINE_ORDER)
    return baselines + pbr


def _color_map(labels) -> dict:
    """label -> color, cycling through tab20 (not zip-truncating like tab10
    does past 10 entries) -- needed now that per-kernel algorithm/cfg labels
    (e.g. 8 block configs x 2 kernels = 16) can exceed tab10's palette size."""
    return {label: color for label, color in zip(labels, itertools.cycle(plt.cm.tab20.colors))}


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
    colors = _color_map(algos)

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


def _best_pbr_by_kernel(df: pd.DataFrame, kernel: str) -> pd.Series:
    """Best (min time_ms) PBR time per (system_size, profile, features) for one kernel.

    Selects across all block-config / smem variants for the given kernel
    (currently just 'block'), so each cell reports the best that kernel can do.
    """
    pbr = df[df["algorithm"].str.startswith("pbr_") & (df["kernel"] == kernel)]
    return pbr.groupby(["system_size", "profile", "features"])["time_ms"].min()


def _annotated_heatmap(ax, piv, fmt="{:.2f}", center=1.0):
    """Render a log-norm RdYlGn heatmap with per-cell text labels."""
    vals = piv.values.astype(float)
    finite = vals[np.isfinite(vals)]
    vmin = max(float(np.nanmin(finite)), 1e-2) if finite.size else 1e-2
    vmax = max(float(np.nanmax(finite)), 1.0) if finite.size else 1.0
    im = ax.imshow(vals, cmap="RdYlGn",
                   norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            v = vals[r, c]
            if np.isfinite(v):
                ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=7)
    return im


def kernel_speedup_heatmap(df: pd.DataFrame, challenger: str, out_path: Path,
                            reference: str = "block") -> Path:
    """Heatmap of best-challenger speedup over best-reference per (profile, features), per size.

    speedup = best_reference_time / best_challenger_time  (>1 = challenger faster).
    Both kernels are best-of across all block-config/smem variants.
    """
    ref = _best_pbr_by_kernel(df, reference)
    other = _best_pbr_by_kernel(df, challenger)
    speedup = (ref / other).rename("speedup").reset_index().dropna()

    if speedup.empty:
        return out_path

    sizes = sorted(speedup["system_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.5 * len(sizes), 4.5), squeeze=False)
    for k, size in enumerate(sizes):
        ax = axes[0][k]
        piv = speedup[speedup["system_size"] == size].pivot(
            index="profile", columns="features", values="speedup")
        im = _annotated_heatmap(ax, piv)
        ax.set_title(f"size={size}", fontsize=10)
        ax.set_xlabel("features")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{challenger} speedup over {reference} kernel (>1 = {challenger} faster)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def kernel_vs_cusparse_heatmap(df: pd.DataFrame, kernel: str, out_path: Path) -> Path:
    """Heatmap of best-PBR(kernel) speedup over cusparse per (profile, features), per size."""
    pbr = _best_pbr_by_kernel(df, kernel)
    base = (df[df["algorithm"] == "cusparse"]
            .set_index(["system_size", "profile", "features"])["time_ms"])
    speedup = (base / pbr).rename("speedup").reset_index().dropna()

    if speedup.empty:
        return out_path

    sizes = sorted(speedup["system_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.5 * len(sizes), 4.5), squeeze=False)
    for k, size in enumerate(sizes):
        ax = axes[0][k]
        piv = speedup[speedup["system_size"] == size].pivot(
            index="profile", columns="features", values="speedup")
        im = _annotated_heatmap(ax, piv)
        ax.set_title(f"size={size}", fontsize=10)
        ax.set_xlabel("features")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"best-PBR ({kernel}) speedup over cusparse (>1 = PBR faster)", fontsize=12)
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
    # Fold kernel into the line label too when multiple kernels are present --
    # otherwise a (block_cfg, smem_size) group spanning e.g. zero_idle AND
    # nnzflat rows gets silently averaged together into one misleading line.
    multi_kernel = "kernel" in pbr.columns and pbr["kernel"].nunique() > 1
    pbr["cfg_label"] = pbr["block_cfg"] + ("_" + pbr["kernel"].astype(str) if multi_kernel else "")
    smem_vals = sorted(pbr["smem_size"].dropna().unique())
    cfgs = sorted(pbr["cfg_label"].unique())
    profiles = sorted(sub["profile"].unique())
    feats = sorted(sub["features"].unique())

    cfg_colors = _color_map(cfgs)

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
                row = (cell_pbr[cell_pbr["cfg_label"] == cfg]
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
    p.add_argument("--reference-kernel", default="block",
                   help="kernel treated as the comparison baseline for the kernel-vs-kernel "
                        "heatmaps (default 'block'; use e.g. 'zero_idle' for a sweep that has "
                        "no 'block'-labeled data, such as the zero_idle/nnzflat comparison)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = fetch_results(args.project, args.runs)
    sizes = sorted(df["system_size"].unique())
    written = []

    def _write_kernel_heatmaps(sub_df: pd.DataFrame, sub_outdir: Path) -> None:
        if "kernel" not in sub_df.columns:
            return
        kernels = set(sub_df["kernel"].unique())
        ref = args.reference_kernel
        written.append(kernel_vs_cusparse_heatmap(sub_df, ref, sub_outdir / f"{ref}_vs_cusparse.png"))
        for challenger in kernels - {ref, "baseline"}:
            written.append(kernel_speedup_heatmap(
                sub_df, challenger, sub_outdir / f"{challenger}_vs_{ref}.png", reference=ref))
            written.append(kernel_vs_cusparse_heatmap(
                sub_df, challenger, sub_outdir / f"{challenger}_vs_cusparse.png"))

    # One subfolder per smem tier with the FULL (uncollapsed) breakdown for that
    # tier -- not a "best smem" summary. Each slice keeps every baseline row
    # (cusparse/csr_naive have no smem_size, so they'd otherwise be dropped by
    # an exact-match filter) alongside just that tier's PBR rows.
    smem_values = sorted(df["smem_size"].dropna().unique()) if "smem_size" in df.columns else []

    if smem_values:
        for smem in smem_values:
            smem_dir = outdir / f"smem{int(smem)}"
            smem_dir.mkdir(parents=True, exist_ok=True)
            sub = df[(df["smem_size"] == smem) | df["smem_size"].isna()]

            for size in sizes:
                written.append(facet_grid(sub, size, smem_dir / f"facet_size{size}.png"))
            written.append(speedup_heatmap(sub, smem_dir / "speedup.png"))
            _write_kernel_heatmaps(sub, smem_dir)
    else:
        for size in sizes:
            written.append(facet_grid(df, size, outdir / f"facet_size{size}.png"))
        written.append(speedup_heatmap(df, outdir / "speedup.png"))
        _write_kernel_heatmaps(df, outdir)

    # smem_sweep is inherently a cross-smem comparison chart -- keep it at the
    # top level on the full (unfiltered) data, not split per tier.
    if _has_smem_sweep(df):
        for size in sizes:
            written.append(smem_sweep(df, size, outdir / f"smem_sweep_size{size}.png"))

    print(f"rows={len(df)}  sizes={sizes}  smem_values={smem_values}")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
