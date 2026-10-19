#!/usr/bin/env python3
"""Create paper-ready visual summaries for the RP/GADF structure experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


METRIC_LABELS = {
    "weighted_bin_jaccard_pct": "Weighted regional Jaccard (%)",
    "boundary_f1_pct": "Boundary F1 (%)",
    "same_bin_spatial_overlap_pct": "Same-bin spatial overlap (%)",
    "distribution_overlap_pct": "Distribution overlap (%)",
}

COMPARISON_LABELS = {
    "observed": "Matched RP/GADF",
    "same_label_random": "Same-activity random",
    "global_random": "Global random",
}

COMPARISON_COLORS = {
    "observed": "#1f77b4",
    "same_label_random": "#d62728",
    "global_random": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RP/GADF structure experiment figures.")
    parser.add_argument(
        "--pair-metrics",
        type=Path,
        default=Path("results/rp_gadf_structure_color_bin_sweep/bins_5/pair_metrics.csv"),
        help="Pair-metric CSV for the histogram figure.",
    )
    parser.add_argument(
        "--compact-table",
        type=Path,
        default=Path("results/rp_gadf_structure_color_bin_sweep/compact_claim_table.csv"),
        help="Consolidated bin-sweep table.",
    )
    parser.add_argument(
        "--visual-root",
        type=Path,
        default=Path("results/rp_gadf_structure_overlap_all"),
        help="Root containing per-sample visual folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rp_gadf_structure_figures"),
        help="Directory where figures are written.",
    )
    return parser.parse_args()


def find_example_panel(visual_root: Path) -> Path:
    preferred = visual_root / "wisdm_1600_Walking_seg0_0" / "gray" / "comparison_panel.png"
    if preferred.exists():
        return preferred
    panels = sorted(visual_root.glob("*/gray/comparison_panel.png"))
    if not panels:
        raise FileNotFoundError(f"No comparison_panel.png found under {visual_root}")
    return panels[0]


def save_metric_histograms(pair_metrics: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        "weighted_bin_jaccard_pct",
        "boundary_f1_pct",
        "same_bin_spatial_overlap_pct",
        "distribution_overlap_pct",
    ]
    comparisons = ["observed", "same_label_random", "global_random"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), dpi=180)
    axes = axes.ravel()
    bins = np.linspace(0, 100, 41)

    for ax, metric in zip(axes, metrics, strict=True):
        for comparison in comparisons:
            values = pair_metrics.loc[pair_metrics["comparison"] == comparison, metric].to_numpy()
            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.38 if comparison != "global_random" else 0.22,
                color=COMPARISON_COLORS[comparison],
                label=COMPARISON_LABELS[comparison],
                edgecolor="none",
            )
            ax.axvline(
                values.mean(),
                color=COMPARISON_COLORS[comparison],
                linewidth=1.6,
                linestyle="-" if comparison != "global_random" else "--",
            )
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18, linewidth=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("RP/GADF matched windows vs random pairings (5 visual bins)", y=0.99, fontsize=13)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_bin_sweep(compact: pd.DataFrame, output_path: Path) -> None:
    wj = compact[
        (compact["metric"] == "weighted_bin_jaccard_pct")
        & (compact["group"].isin(["gray", "xyz", "all_channels"]))
    ].copy()
    dist = compact[
        (compact["metric"] == "distribution_overlap_pct")
        & (compact["group"] == "all_channels")
    ].copy()
    boundary = compact[
        (compact["metric"] == "boundary_f1_pct")
        & (compact["group"] == "all_channels")
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180)

    ax = axes[0]
    all_wj = wj[wj["group"] == "all_channels"].sort_values("bins")
    ax.plot(all_wj["bins"], all_wj["observed_mean"], marker="o", label="Matched")
    ax.plot(all_wj["bins"], all_wj["same_label_null_mean"], marker="o", label="Same-activity random")
    ax.set_title("Regional overlap")
    ax.set_xlabel("Visual intensity bins")
    ax.set_ylabel("Weighted Jaccard (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    for group, label in [("gray", "gray"), ("xyz", "x/y/z"), ("all_channels", "all")]:
        sub = wj[wj["group"] == group].sort_values("bins")
        ax.plot(sub["bins"], sub["factor_mean"], marker="o", label=label)
    ax.set_title("Observed/null overlap factor")
    ax.set_xlabel("Visual intensity bins")
    ax.set_ylabel("Factor vs same-activity random")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[2]
    dist = dist.sort_values("bins")
    boundary = boundary.sort_values("bins")
    ax.plot(dist["bins"], dist["factor_mean"], marker="o", label="Distribution factor")
    ax.plot(boundary["bins"], boundary["factor_mean"], marker="o", label="Boundary F1 factor")
    ax.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_title("Secondary controls")
    ax.set_xlabel("Visual intensity bins")
    ax.set_ylabel("Observed/null factor")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle("RP/GADF structural-overlap robustness across visual quantisation", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_collage(example_panel: Path, pair_metrics: pd.DataFrame, compact: pd.DataFrame, output_path: Path) -> None:
    panel = Image.open(example_panel).convert("RGB")
    metrics = ["weighted_bin_jaccard_pct", "boundary_f1_pct"]
    comparisons = ["observed", "same_label_random"]
    hist_bins = np.linspace(0, 100, 36)

    fig = plt.figure(figsize=(12, 10), dpi=180)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1], hspace=0.42, wspace=0.28)

    ax_panel = fig.add_subplot(grid[0, :])
    ax_panel.imshow(panel)
    ax_panel.set_title("Example 5-bin regional comparison: RP, aligned GADF, and same-bin overlap")
    ax_panel.axis("off")

    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(grid[1, idx])
        for comparison in comparisons:
            values = pair_metrics.loc[pair_metrics["comparison"] == comparison, metric].to_numpy()
            ax.hist(
                values,
                bins=hist_bins,
                density=True,
                alpha=0.45,
                color=COMPARISON_COLORS[comparison],
                label=COMPARISON_LABELS[comparison],
                edgecolor="none",
            )
            ax.axvline(values.mean(), color=COMPARISON_COLORS[comparison], linewidth=1.8)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlim(0, 100)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18, linewidth=0.6)
        if idx == 0:
            ax.legend(frameon=False)

    wj = compact[
        (compact["metric"] == "weighted_bin_jaccard_pct")
        & (compact["group"] == "all_channels")
    ].sort_values("bins")
    dist = compact[
        (compact["metric"] == "distribution_overlap_pct")
        & (compact["group"] == "all_channels")
    ].sort_values("bins")

    ax = fig.add_subplot(grid[2, 0])
    ax.plot(wj["bins"], wj["observed_mean"], marker="o", label="Matched")
    ax.plot(wj["bins"], wj["same_label_null_mean"], marker="o", label="Same-activity random")
    ax.set_title("Weighted regional Jaccard across visual bins")
    ax.set_xlabel("Visual intensity bins")
    ax.set_ylabel("Weighted Jaccard (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[2, 1])
    ax.plot(wj["bins"], wj["factor_mean"], marker="o", label="Regional factor")
    ax.plot(dist["bins"], dist["factor_mean"], marker="o", label="Distribution-only factor")
    ax.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_title("Observed/null factor")
    ax.set_xlabel("Visual intensity bins")
    ax.set_ylabel("Factor")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle("RP/GADF structure analysis: visual regions, random baseline, and bin robustness", y=0.995)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_metrics = pd.read_csv(args.pair_metrics)
    compact = pd.read_csv(args.compact_table)
    example_panel = find_example_panel(args.visual_root)

    save_metric_histograms(pair_metrics, args.output_dir / "metric_histograms_bins5.png")
    save_bin_sweep(compact, args.output_dir / "bin_sweep_summary.png")
    save_collage(example_panel, pair_metrics, compact, args.output_dir / "rp_gadf_structure_collage.png")

    print(f"Example panel: {example_panel}")
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
