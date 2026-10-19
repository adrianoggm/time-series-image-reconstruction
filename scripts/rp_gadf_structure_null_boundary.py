#!/usr/bin/env python3
"""Null-corrected RP/GADF regional overlap and boundary alignment.

This script tests whether RP and GADF regions align more than expected from
their marginal colour distributions by comparing matched windows against
random GADF windows, both within the same activity and globally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rp_gadf_structure_overlap import (
    CHANNELS,
    align_images,
    channel_values,
    iter_wisdm_sample_images,
    quantize,
)


PAIR_FIELDS = [
    "sample_id",
    "label",
    "channel",
    "comparison",
    "paired_gadf_sample_id",
    "same_bin_spatial_overlap_pct",
    "weighted_bin_jaccard_pct",
    "distribution_overlap_pct",
    "boundary_f1_pct",
    "boundary_precision_pct",
    "boundary_recall_pct",
    "rp_boundary_density_pct",
    "gadf_boundary_density_pct",
]

EFFECT_FIELDS = [
    "channel",
    "metric",
    "observed_mean",
    "same_label_null_mean",
    "global_null_mean",
    "observed_minus_same_label_null_mean",
    "observed_minus_global_null_mean",
    "observed_minus_same_label_null_median",
    "observed_gt_same_label_null_pct",
    "wilcoxon_same_label_p",
    "n_samples",
]

AGGREGATE_FIELDS = [
    "comparison",
    "channel",
    "n_samples",
    "same_bin_spatial_overlap_pct_mean",
    "same_bin_spatial_overlap_pct_std",
    "weighted_bin_jaccard_pct_mean",
    "weighted_bin_jaccard_pct_std",
    "distribution_overlap_pct_mean",
    "distribution_overlap_pct_std",
    "boundary_f1_pct_mean",
    "boundary_f1_pct_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare observed RP/GADF structure against random-pair baselines."
    )
    parser.add_argument("--bins", type=int, default=5, help="Number of quantized intensity ranges.")
    parser.add_argument("--time-steps", type=int, default=128, help="Window length.")
    parser.add_argument("--stride", type=int, default=128, help="Window stride.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/WISDM"), help="WISDM data directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rp_gadf_structure_null_boundary"),
        help="Directory where CSV outputs are written.",
    )
    parser.add_argument(
        "--alignment",
        choices=["crop-gadf", "resize-gadf"],
        default="crop-gadf",
        help="How to align GADF 128x128 to RP 127x127.",
    )
    parser.add_argument("--boundary-radius", type=int, default=1, help="Pixel tolerance for boundary matching.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for null pairings.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    return parser.parse_args()


def weighted_bin_jaccard(rp_q: np.ndarray, gadf_q: np.ndarray, bins: int) -> float:
    intersection_sum = 0
    union_sum = 0
    for bin_idx in range(bins):
        rp_mask = rp_q == bin_idx
        gadf_mask = gadf_q == bin_idx
        intersection_sum += int(np.logical_and(rp_mask, gadf_mask).sum())
        union_sum += int(np.logical_or(rp_mask, gadf_mask).sum())
    return 100.0 * intersection_sum / union_sum if union_sum else 0.0


def distribution_overlap(rp_q: np.ndarray, gadf_q: np.ndarray, bins: int) -> float:
    rp_hist = np.bincount(rp_q.ravel(), minlength=bins)
    gadf_hist = np.bincount(gadf_q.ravel(), minlength=bins)
    return 100.0 * np.minimum(rp_hist, gadf_hist).sum() / rp_q.size


def boundary_map(q: np.ndarray) -> np.ndarray:
    boundary = np.zeros(q.shape, dtype=bool)
    horizontal = q[:, 1:] != q[:, :-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    vertical = q[1:, :] != q[:-1, :]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def boundary_scores(rp_q: np.ndarray, gadf_q: np.ndarray, radius: int) -> tuple[float, float, float, float, float]:
    rp_boundary = boundary_map(rp_q)
    gadf_boundary = boundary_map(gadf_q)
    rp_count = int(rp_boundary.sum())
    gadf_count = int(gadf_boundary.sum())

    if rp_count == 0 and gadf_count == 0:
        return 100.0, 100.0, 100.0, 0.0, 0.0
    if rp_count == 0 or gadf_count == 0:
        return 0.0, 0.0, 0.0, 100.0 * rp_count / rp_q.size, 100.0 * gadf_count / gadf_q.size

    if radius > 0:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        rp_match_target = binary_dilation(gadf_boundary, structure=structure)
        gadf_match_target = binary_dilation(rp_boundary, structure=structure)
    else:
        rp_match_target = gadf_boundary
        gadf_match_target = rp_boundary

    precision = 100.0 * np.logical_and(rp_boundary, rp_match_target).sum() / rp_count
    recall = 100.0 * np.logical_and(gadf_boundary, gadf_match_target).sum() / gadf_count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        float(f1),
        float(precision),
        float(recall),
        100.0 * rp_count / rp_q.size,
        100.0 * gadf_count / gadf_q.size,
    )


def pair_metrics(rp_q: np.ndarray, gadf_q: np.ndarray, bins: int, boundary_radius: int) -> dict[str, float]:
    boundary_f1, boundary_precision, boundary_recall, rp_density, gadf_density = boundary_scores(
        rp_q,
        gadf_q,
        boundary_radius,
    )
    return {
        "same_bin_spatial_overlap_pct": 100.0 * float((rp_q == gadf_q).mean()),
        "weighted_bin_jaccard_pct": weighted_bin_jaccard(rp_q, gadf_q, bins),
        "distribution_overlap_pct": distribution_overlap(rp_q, gadf_q, bins),
        "boundary_f1_pct": boundary_f1,
        "boundary_precision_pct": boundary_precision,
        "boundary_recall_pct": boundary_recall,
        "rp_boundary_density_pct": rp_density,
        "gadf_boundary_density_pct": gadf_density,
    }


def collect_quantized_samples(args: argparse.Namespace) -> dict[str, dict[str, list[dict[str, object]]]]:
    samples: dict[str, dict[str, list[dict[str, object]]]] = {}
    channels = ["gray", "x", "y", "z"]
    for idx, sample in enumerate(
        iter_wisdm_sample_images(args.data_dir, args.time_steps, args.stride, args.limit),
        start=1,
    ):
        rp_aligned, gadf_aligned = align_images(sample.rp, sample.gadf, args.alignment)
        label_bucket = samples.setdefault(sample.label, {channel: [] for channel in channels})
        for channel in channels:
            rp_q = quantize(channel_values(rp_aligned, channel), args.bins)
            gadf_q = quantize(channel_values(gadf_aligned, channel), args.bins)
            label_bucket[channel].append(
                {
                    "sample_id": sample.sample_id,
                    "rp_q": rp_q,
                    "gadf_q": gadf_q,
                }
            )
        if idx % 500 == 0:
            print(f"Encoded and quantized {idx} windows...")
    return samples


def random_other_index(rng: np.random.Generator, n: int, current_idx: int) -> int:
    if n <= 1:
        return current_idx
    candidate = int(rng.integers(0, n - 1))
    return candidate + 1 if candidate >= current_idx else candidate


def global_lookup(
    samples: dict[str, dict[str, list[dict[str, object]]]],
    channel: str,
) -> list[tuple[str, int, dict[str, object]]]:
    items: list[tuple[str, int, dict[str, object]]] = []
    for label, channel_data in samples.items():
        for idx, sample in enumerate(channel_data[channel]):
            items.append((label, idx, sample))
    return items


def write_pair_rows(
    samples: dict[str, dict[str, list[dict[str, object]]]],
    output_path: Path,
    bins: int,
    boundary_radius: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    global_by_channel = {channel: global_lookup(samples, channel) for channel in ["gray", "x", "y", "z"]}
    global_pos_by_channel = {
        channel: {(label, idx): pos for pos, (label, idx, _sample) in enumerate(items)}
        for channel, items in global_by_channel.items()
    }

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        writer.writeheader()

        written = 0
        for label, channel_data in sorted(samples.items()):
            for channel, label_samples in channel_data.items():
                global_items = global_by_channel[channel]
                for idx, sample in enumerate(label_samples):
                    rp_q = sample["rp_q"]

                    pairings: list[tuple[str, str, np.ndarray]] = [
                        ("observed", str(sample["sample_id"]), sample["gadf_q"]),
                    ]

                    same_label_idx = random_other_index(rng, len(label_samples), idx)
                    same_label_sample = label_samples[same_label_idx]
                    pairings.append(
                        (
                            "same_label_random",
                            str(same_label_sample["sample_id"]),
                            same_label_sample["gadf_q"],
                        )
                    )

                    global_position = global_pos_by_channel[channel][(label, idx)]
                    global_idx = random_other_index(rng, len(global_items), global_position)
                    _global_label, _global_sample_idx, global_sample = global_items[global_idx]
                    pairings.append(
                        (
                            "global_random",
                            str(global_sample["sample_id"]),
                            global_sample["gadf_q"],
                        )
                    )

                    for comparison, paired_id, gadf_q in pairings:
                        metrics = pair_metrics(rp_q, gadf_q, bins, boundary_radius)
                        writer.writerow(
                            {
                                "sample_id": sample["sample_id"],
                                "label": label,
                                "channel": channel,
                                "comparison": comparison,
                                "paired_gadf_sample_id": paired_id,
                                **metrics,
                            }
                        )
                        written += 1

                    if written and written % 15000 == 0:
                        print(f"Wrote {written} pair metric rows...")


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(pair_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = [
        "same_bin_spatial_overlap_pct",
        "weighted_bin_jaccard_pct",
        "distribution_overlap_pct",
        "boundary_f1_pct",
    ]
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in pair_rows:
        groups.setdefault((row["comparison"], row["channel"]), []).append(row)

    aggregate: list[dict[str, object]] = []
    for (comparison, channel), rows in sorted(groups.items()):
        out: dict[str, object] = {"comparison": comparison, "channel": channel, "n_samples": len(rows)}
        for metric in metrics:
            values = np.array([float(row[metric]) for row in rows], dtype=float)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=0))
        aggregate.append(out)
    return aggregate


def effect_rows(pair_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = [
        "same_bin_spatial_overlap_pct",
        "weighted_bin_jaccard_pct",
        "distribution_overlap_pct",
        "boundary_f1_pct",
    ]
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in pair_rows:
        key = (row["sample_id"], row["channel"])
        by_key.setdefault(key, {})[row["comparison"]] = row

    rows_out: list[dict[str, object]] = []
    for channel in ["gray", "x", "y", "z"]:
        channel_keys = [key for key in by_key if key[1] == channel]
        for metric in metrics:
            observed = np.array([float(by_key[key]["observed"][metric]) for key in channel_keys], dtype=float)
            same_null = np.array(
                [float(by_key[key]["same_label_random"][metric]) for key in channel_keys],
                dtype=float,
            )
            global_null = np.array(
                [float(by_key[key]["global_random"][metric]) for key in channel_keys],
                dtype=float,
            )
            diff_same = observed - same_null
            diff_global = observed - global_null
            try:
                p_value = float(wilcoxon(diff_same, alternative="greater", zero_method="wilcox").pvalue)
            except ValueError:
                p_value = math.nan
            rows_out.append(
                {
                    "channel": channel,
                    "metric": metric,
                    "observed_mean": float(observed.mean()),
                    "same_label_null_mean": float(same_null.mean()),
                    "global_null_mean": float(global_null.mean()),
                    "observed_minus_same_label_null_mean": float(diff_same.mean()),
                    "observed_minus_global_null_mean": float(diff_global.mean()),
                    "observed_minus_same_label_null_median": float(np.median(diff_same)),
                    "observed_gt_same_label_null_pct": 100.0 * float((diff_same > 0).mean()),
                    "wilcoxon_same_label_p": p_value,
                    "n_samples": len(channel_keys),
                }
            )
    return rows_out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_quantized_samples(args)
    total_samples = sum(len(channel_data["gray"]) for channel_data in samples.values())
    if total_samples == 0:
        raise SystemExit("No WISDM windows were found.")

    pair_metrics_path = args.output_dir / "pair_metrics.csv"
    write_pair_rows(samples, pair_metrics_path, args.bins, args.boundary_radius, args.seed)

    pair_rows = read_metric_rows(pair_metrics_path)
    write_csv(args.output_dir / "aggregate_by_comparison.csv", aggregate_rows(pair_rows), AGGREGATE_FIELDS)
    write_csv(args.output_dir / "effect_summary.csv", effect_rows(pair_rows), EFFECT_FIELDS)

    config = {
        "samples_processed": total_samples,
        "bins": args.bins,
        "alignment": args.alignment,
        "boundary_radius": args.boundary_radius,
        "seed": args.seed,
        "time_steps": args.time_steps,
        "stride": args.stride,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "comparisons": {
            "observed": "RP_i vs GADF_i",
            "same_label_random": "RP_i vs GADF_j from the same activity, j != i",
            "global_random": "RP_i vs GADF_j from any activity, j != i",
        },
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Processed {total_samples} WISDM windows")
    print(f"Wrote pair metrics to {pair_metrics_path}")


if __name__ == "__main__":
    main()
