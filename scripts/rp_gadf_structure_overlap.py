#!/usr/bin/env python3
"""Compare RP and GADF image structure with discrete intensity ranges.

The experiment creates clearer visual maps by quantizing each RP/GADF image
into a small number of intensity bins, then measures how much the bin areas
overlap after aligning the matrix sizes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ACTIVITIES = {
    "A": "Walking",
    "B": "Jogging",
    "C": "Stairs",
    "D": "Sitting",
    "E": "Standing",
}

CHANNELS = {
    "x": 0,
    "y": 1,
    "z": 2,
}

SUMMARY_FIELDS = [
    "sample_id",
    "label",
    "source",
    "channel",
    "rp_shape",
    "gadf_shape",
    "aligned_shape",
    "bins",
    "alignment",
    "same_bin_spatial_overlap_pct",
    "mean_bin_jaccard_pct",
    "weighted_bin_jaccard_pct",
    "distribution_overlap_pct",
]

BIN_FIELDS = [
    "sample_id",
    "label",
    "source",
    "channel",
    "bin",
    "bin_start_inclusive",
    "bin_end_exclusive",
    "pixels_total",
    "rp_area_pixels",
    "gadf_area_pixels",
    "intersection_pixels",
    "union_pixels",
    "intersection_total_pct",
    "jaccard_pct",
    "rp_area_covered_pct",
    "gadf_area_covered_pct",
]

AGGREGATE_METRICS = [
    "same_bin_spatial_overlap_pct",
    "mean_bin_jaccard_pct",
    "weighted_bin_jaccard_pct",
    "distribution_overlap_pct",
]

QUANT_PALETTE = np.array(
    [
        [49, 54, 149],
        [69, 117, 180],
        [116, 173, 209],
        [253, 174, 97],
        [215, 48, 39],
    ],
    dtype=np.uint8,
)

OVERLAY_PALETTE = np.array(
    [
        [18, 18, 18],      # neither
        [220, 60, 50],     # RP only
        [60, 120, 220],    # GADF only
        [250, 220, 70],    # overlap
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class SampleImages:
    sample_id: str
    label: str
    source: str
    rp: np.ndarray
    gadf: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize RP/GADF images and measure spatial overlap by intensity range."
    )
    parser.add_argument("--samples", type=int, default=8, help="Number of paired samples to process.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every non-overlapping WISDM watch-accelerometer window.",
    )
    parser.add_argument(
        "--visual-limit",
        type=int,
        default=12,
        help="Maximum number of per-sample visual folders to save in --all mode.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride for generated WISDM pairs. Defaults to 128 in --all mode and 512 otherwise.",
    )
    parser.add_argument("--bins", type=int, default=5, help="Number of intensity ranges.")
    parser.add_argument(
        "--source",
        choices=["auto", "doc-images", "wisdm"],
        default="auto",
        help="Use existing doc image pairs, generate WISDM pairs, or combine both.",
    )
    parser.add_argument("--time-steps", type=int, default=128, help="Window length for generated WISDM pairs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/WISDM"), help="WISDM data directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rp_gadf_structure_overlap"),
        help="Directory where images and CSV metrics are written.",
    )
    parser.add_argument(
        "--alignment",
        choices=["crop-gadf", "resize-gadf"],
        default="crop-gadf",
        help="How to align GADF 128x128 to RP 127x127 before comparing masks.",
    )
    return parser.parse_args()


class RunningAggregate:
    def __init__(self, group_fields: tuple[str, ...] = ("channel",)) -> None:
        self._group_fields = group_fields
        self._stats: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}

    def update(self, row: dict[str, object]) -> None:
        key = tuple(str(row[field]) for field in self._group_fields)
        channel_stats = self._stats.setdefault(key, {})
        for metric in AGGREGATE_METRICS:
            value = float(row[metric])
            stat = channel_stats.setdefault(metric, {"n": 0.0, "sum": 0.0, "sumsq": 0.0})
            stat["n"] += 1.0
            stat["sum"] += value
            stat["sumsq"] += value * value

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for key in sorted(self._stats):
            channel_stats = self._stats[key]
            first_metric = channel_stats[AGGREGATE_METRICS[0]]
            n = int(first_metric["n"])
            row: dict[str, object] = {
                field: value for field, value in zip(self._group_fields, key, strict=True)
            }
            row["n_samples"] = n
            for metric in AGGREGATE_METRICS:
                stat = channel_stats[metric]
                mean = stat["sum"] / stat["n"] if stat["n"] else 0.0
                variance = stat["sumsq"] / stat["n"] - mean * mean if stat["n"] else 0.0
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = math.sqrt(max(variance, 0.0))
            rows.append(row)
        return rows


def normalize_zero_one(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if math.isclose(max_value, min_value):
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - min_value) / (max_value - min_value), 0.0, 1.0)


def signed_rp_axis(values: np.ndarray) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    states = np.column_stack([axis[:-1], axis[1:]])
    diff = states[:, None, :] - states[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    denom = dist * math.sqrt(2.0)
    cosine = np.divide(
        diff[:, :, 0] + diff[:, :, 1],
        denom,
        out=np.zeros_like(dist),
        where=denom > 0.0,
    )
    sign = np.where(cosine >= math.sqrt(2.0) / 2.0, 1.0, -1.0)
    return sign * dist


def gadf_axis(values: np.ndarray) -> np.ndarray:
    axis = normalize_zero_one(np.asarray(values, dtype=float))
    phi = np.arccos(axis)
    return np.sin(phi[None, :] - phi[:, None])


def encode_rp(window: np.ndarray) -> np.ndarray:
    channels = [normalize_zero_one(signed_rp_axis(window[:, idx])) for idx in range(3)]
    return np.rint(np.stack(channels, axis=2) * 255.0).astype(np.uint8)


def encode_gadf(window: np.ndarray) -> np.ndarray:
    channels = [normalize_zero_one(gadf_axis(window[:, idx])) for idx in range(3)]
    return np.rint(np.stack(channels, axis=2) * 255.0).astype(np.uint8)


def sanitize_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("_") or "sample"


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def doc_image_samples() -> list[SampleImages]:
    pairs = [
        ("doc_0x0", "doc_image", Path("doc/images/0x0.png"), Path("doc/images/0x0gadf.png")),
        ("doc_1600x0", "doc_image", Path("doc/images/1600xtrp0.png"), Path("doc/images/1600x0tgadf.png")),
    ]
    samples: list[SampleImages] = []
    for sample_id, label, rp_path, gadf_path in pairs:
        if rp_path.exists() and gadf_path.exists():
            samples.append(
                SampleImages(
                    sample_id=sample_id,
                    label=label,
                    source="doc-images",
                    rp=read_rgb(rp_path),
                    gadf=read_rgb(gadf_path),
                )
            )
    return samples


def iter_wisdm_windows(data_dir: Path, time_steps: int, stride: int) -> Iterable[tuple[str, str, np.ndarray]]:
    raw_dir = data_dir / "wisdm-dataset" / "raw" / "watch" / "accel"
    if not raw_dir.exists():
        return

    for raw_file in sorted(raw_dir.glob("data_*_accel_watch.txt")):
        subject_match = re.search(r"data_(\d+)_", raw_file.name)
        subject = subject_match.group(1) if subject_match else "unknown"
        df = pd.read_csv(
            raw_file,
            header=None,
            names=["user_id", "activity", "timestamp", "x_axis", "y_axis", "z_axis"],
        )
        df["z_axis"] = df["z_axis"].astype(str).str.replace(";", "", regex=False).astype(float)
        df = df.dropna(axis=0, how="any")

        segment_idx = 0
        current_activity: str | None = None
        current_values: list[list[float]] = []

        def emit_segment(activity_code: str | None, values: list[list[float]], seg_idx: int):
            if activity_code is None or len(values) < time_steps:
                return
            label = ACTIVITIES[activity_code]
            arr = np.asarray(values, dtype=float)
            for offset in range(0, len(arr) - time_steps + 1, stride):
                window = arr[offset : offset + time_steps]
                sample_id = f"wisdm_{subject}_{sanitize_name(label)}_seg{seg_idx}_{offset}"
                yield sample_id, label, window

        for row in df.itertuples(index=False):
            activity_code = str(row.activity)
            if activity_code not in ACTIVITIES:
                yield from emit_segment(current_activity, current_values, segment_idx)
                if current_activity is not None:
                    segment_idx += 1
                current_activity = None
                current_values = []
                continue
            if current_activity != activity_code:
                yield from emit_segment(current_activity, current_values, segment_idx)
                if current_activity is not None:
                    segment_idx += 1
                current_activity = activity_code
                current_values = []
            current_values.append([float(row.x_axis), float(row.y_axis), float(row.z_axis)])

        yield from emit_segment(current_activity, current_values, segment_idx)


def iter_wisdm_sample_images(
    data_dir: Path,
    time_steps: int,
    stride: int,
    limit: int | None = None,
) -> Iterable[SampleImages]:
    for idx, (sample_id, label, window) in enumerate(iter_wisdm_windows(data_dir, time_steps, stride)):
        if limit is not None and idx >= limit:
            break
        yield SampleImages(
            sample_id=sample_id,
            label=label,
            source="wisdm-generated",
            rp=encode_rp(window),
            gadf=encode_gadf(window),
        )


def wisdm_samples(data_dir: Path, time_steps: int, stride: int, limit: int) -> list[SampleImages]:
    samples: list[SampleImages] = []
    for sample in iter_wisdm_sample_images(data_dir, time_steps, stride, limit):
        samples.append(sample)
        if len(samples) >= limit:
            break
    return samples


def choose_samples(args: argparse.Namespace) -> list[SampleImages]:
    stride = args.stride if args.stride is not None else args.time_steps * 4
    samples: list[SampleImages] = []
    if args.source in {"auto", "doc-images"}:
        samples.extend(doc_image_samples())
    if args.source in {"auto", "wisdm"} and len(samples) < args.samples:
        samples.extend(wisdm_samples(args.data_dir, args.time_steps, stride, args.samples - len(samples)))
    deduped: list[SampleImages] = []
    seen: set[str] = set()
    for sample in samples:
        digest = hashlib.sha1(sample.rp.tobytes() + sample.gadf.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        deduped.append(sample)
    if args.source in {"auto", "wisdm"} and len(deduped) < args.samples:
        extra = wisdm_samples(args.data_dir, args.time_steps, stride, args.samples * 2)
        for sample in extra:
            digest = hashlib.sha1(sample.rp.tobytes() + sample.gadf.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            deduped.append(sample)
            if len(deduped) >= args.samples:
                break
    return deduped[: args.samples]


def align_images(rp: np.ndarray, gadf: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if rp.shape[:2] == gadf.shape[:2]:
        return rp, gadf

    if mode == "resize-gadf":
        size = (rp.shape[1], rp.shape[0])
        gadf_img = Image.fromarray(gadf).resize(size, resample=Image.Resampling.NEAREST)
        return rp, np.asarray(gadf_img, dtype=np.uint8)

    height = min(rp.shape[0], gadf.shape[0])
    width = min(rp.shape[1], gadf.shape[1])
    return rp[:height, :width], gadf[:height, :width]


def channel_values(image: np.ndarray, channel: str) -> np.ndarray:
    if channel == "gray":
        return image.astype(float).mean(axis=2)
    return image[:, :, CHANNELS[channel]].astype(float)


def quantize(values: np.ndarray, bins: int) -> np.ndarray:
    scaled = np.floor(np.clip(values, 0, 255) * bins / 256.0).astype(np.int16)
    return np.clip(scaled, 0, bins - 1).astype(np.uint8)


def palette_for_bins(bins: int) -> np.ndarray:
    if bins == len(QUANT_PALETTE):
        return QUANT_PALETTE
    anchors = np.linspace(0, len(QUANT_PALETTE) - 1, bins)
    return QUANT_PALETTE[np.rint(anchors).astype(int)]


def save_quantized(path: Path, quantized: np.ndarray, bins: int) -> None:
    palette = palette_for_bins(bins)
    Image.fromarray(palette[quantized]).save(path)


def save_overlay(path: Path, rp_mask: np.ndarray, gadf_mask: np.ndarray) -> None:
    overlay = np.zeros(rp_mask.shape, dtype=np.uint8)
    overlay[rp_mask] = 1
    overlay[gadf_mask] = 2
    overlay[rp_mask & gadf_mask] = 3
    Image.fromarray(OVERLAY_PALETTE[overlay]).save(path)


def save_panel(path: Path, rp_q: np.ndarray, gadf_q: np.ndarray, bins: int) -> None:
    palette = palette_for_bins(bins)
    rp_img = Image.fromarray(palette[rp_q])
    gadf_img = Image.fromarray(palette[gadf_q])
    match = np.zeros((*rp_q.shape, 3), dtype=np.uint8)
    same = rp_q == gadf_q
    match[same] = palette[rp_q[same]]
    match_img = Image.fromarray(match)

    pad = 12
    label_h = 22
    width = rp_img.width * 3 + pad * 4
    height = rp_img.height + label_h + pad * 2
    panel = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    labels = [("RP", rp_img), ("GADF aligned", gadf_img), ("same bin", match_img)]
    x = pad
    for label, img in labels:
        draw.text((x, pad), label, fill=(20, 20, 20))
        panel.paste(img, (x, pad + label_h))
        x += img.width + pad
    panel.save(path)


def save_legend(path: Path, bins: int) -> None:
    palette = palette_for_bins(bins)
    row_h = 26
    swatch = 34
    width = 430
    height = row_h * (bins + 5) + 16
    image = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    y = 8
    draw.text((8, y), "Quantized intensity ranges", fill=(20, 20, 20))
    y += row_h
    for bin_idx, color in enumerate(palette):
        draw.rectangle((8, y, 8 + swatch, y + 18), fill=tuple(int(v) for v in color))
        start = 256 * bin_idx / bins
        end = 256 * (bin_idx + 1) / bins
        draw.text((52, y + 2), f"bin {bin_idx}: [{start:.1f}, {end:.1f})", fill=(20, 20, 20))
        y += row_h

    y += 6
    draw.text((8, y), "Per-bin overlap maps", fill=(20, 20, 20))
    y += row_h
    overlay_labels = ["neither", "RP only", "GADF only", "overlap"]
    for idx, label in enumerate(overlay_labels):
        color = OVERLAY_PALETTE[idx]
        draw.rectangle((8, y, 8 + swatch, y + 18), fill=tuple(int(v) for v in color))
        draw.text((52, y + 2), label, fill=(20, 20, 20))
        y += row_h

    image.save(path)


def save_sample_inputs(sample_dir: Path, rp: np.ndarray, gadf: np.ndarray, gadf_aligned: np.ndarray) -> None:
    Image.fromarray(rp).save(sample_dir / "rp_original.png")
    Image.fromarray(gadf).save(sample_dir / "gadf_original.png")
    Image.fromarray(gadf_aligned).save(sample_dir / "gadf_aligned.png")


def measure_sample(
    sample: SampleImages,
    output_dir: Path,
    bins: int,
    alignment: str,
    save_visuals: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rp_aligned, gadf_aligned = align_images(sample.rp, sample.gadf, alignment)
    sample_dir = output_dir / sanitize_name(sample.sample_id)
    if save_visuals:
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_sample_inputs(sample_dir, rp_aligned, sample.gadf, gadf_aligned)

    summary_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    total_pixels = int(rp_aligned.shape[0] * rp_aligned.shape[1])

    for channel in ["gray", "x", "y", "z"]:
        rp_q = quantize(channel_values(rp_aligned, channel), bins)
        gadf_q = quantize(channel_values(gadf_aligned, channel), bins)

        if save_visuals:
            channel_dir = sample_dir / channel
            channel_dir.mkdir(parents=True, exist_ok=True)
            save_quantized(channel_dir / "rp_quantized.png", rp_q, bins)
            save_quantized(channel_dir / "gadf_quantized.png", gadf_q, bins)
            save_panel(channel_dir / "comparison_panel.png", rp_q, gadf_q, bins)

        intersections = 0
        union_sum = 0
        jaccards: list[float] = []
        hist_rp = np.bincount(rp_q.ravel(), minlength=bins)
        hist_gadf = np.bincount(gadf_q.ravel(), minlength=bins)

        for bin_idx in range(bins):
            rp_mask = rp_q == bin_idx
            gadf_mask = gadf_q == bin_idx
            intersection = int(np.logical_and(rp_mask, gadf_mask).sum())
            union = int(np.logical_or(rp_mask, gadf_mask).sum())
            rp_area = int(rp_mask.sum())
            gadf_area = int(gadf_mask.sum())
            intersections += intersection
            union_sum += union
            if union:
                jaccards.append(intersection / union)

            if save_visuals:
                save_overlay(channel_dir / f"overlap_bin_{bin_idx}.png", rp_mask, gadf_mask)
            bin_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "source": sample.source,
                    "channel": channel,
                    "bin": bin_idx,
                    "bin_start_inclusive": round(256 * bin_idx / bins, 3),
                    "bin_end_exclusive": round(256 * (bin_idx + 1) / bins, 3),
                    "pixels_total": total_pixels,
                    "rp_area_pixels": rp_area,
                    "gadf_area_pixels": gadf_area,
                    "intersection_pixels": intersection,
                    "union_pixels": union,
                    "intersection_total_pct": 100.0 * intersection / total_pixels,
                    "jaccard_pct": 100.0 * intersection / union if union else "",
                    "rp_area_covered_pct": 100.0 * intersection / rp_area if rp_area else "",
                    "gadf_area_covered_pct": 100.0 * intersection / gadf_area if gadf_area else "",
                }
            )

        same_bin_pct = 100.0 * intersections / total_pixels
        weighted_jaccard_pct = 100.0 * intersections / union_sum if union_sum else 0.0
        distribution_overlap_pct = 100.0 * np.minimum(hist_rp, hist_gadf).sum() / total_pixels
        summary_rows.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "source": sample.source,
                "channel": channel,
                "rp_shape": f"{sample.rp.shape[0]}x{sample.rp.shape[1]}",
                "gadf_shape": f"{sample.gadf.shape[0]}x{sample.gadf.shape[1]}",
                "aligned_shape": f"{rp_aligned.shape[0]}x{rp_aligned.shape[1]}",
                "bins": bins,
                "alignment": alignment,
                "same_bin_spatial_overlap_pct": same_bin_pct,
                "mean_bin_jaccard_pct": 100.0 * float(np.mean(jaccards)) if jaccards else 0.0,
                "weighted_bin_jaccard_pct": weighted_jaccard_pct,
                "distribution_overlap_pct": distribution_overlap_pct,
            }
        )

    return summary_rows, bin_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    channels = sorted({str(row["channel"]) for row in rows})
    metrics = [
        "same_bin_spatial_overlap_pct",
        "mean_bin_jaccard_pct",
        "weighted_bin_jaccard_pct",
        "distribution_overlap_pct",
    ]
    aggregate_rows: list[dict[str, object]] = []
    for channel in channels:
        channel_rows = [row for row in rows if row["channel"] == channel]
        out: dict[str, object] = {"channel": channel, "n_samples": len(channel_rows)}
        for metric in metrics:
            values = np.array([float(row[metric]) for row in channel_rows], dtype=float)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=0))
        aggregate_rows.append(out)
    return aggregate_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stride = args.stride if args.stride is not None else (args.time_steps if args.all else args.time_steps * 4)
    if args.all:
        sample_iter: Iterable[SampleImages] = iter_wisdm_sample_images(
            args.data_dir,
            args.time_steps,
            stride,
            limit=None,
        )
    else:
        sample_iter = choose_samples(args)

    aggregate = RunningAggregate(("channel",))
    aggregate_by_label = RunningAggregate(("label", "channel"))
    processed = 0
    summary_path = args.output_dir / "summary.csv"
    bin_path = args.output_dir / "bin_overlap.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as summary_fh, bin_path.open(
        "w", newline="", encoding="utf-8"
    ) as bin_fh:
        summary_writer = csv.DictWriter(summary_fh, fieldnames=SUMMARY_FIELDS)
        bin_writer = csv.DictWriter(bin_fh, fieldnames=BIN_FIELDS)
        summary_writer.writeheader()
        bin_writer.writeheader()

        for processed, sample in enumerate(sample_iter, start=1):
            save_visuals = (not args.all) or processed <= args.visual_limit
            summary_rows, bin_rows = measure_sample(
                sample,
                args.output_dir,
                args.bins,
                args.alignment,
                save_visuals=save_visuals,
            )
            summary_writer.writerows(summary_rows)
            bin_writer.writerows(bin_rows)
            for row in summary_rows:
                aggregate.update(row)
                aggregate_by_label.update(row)

            if args.all and processed % 500 == 0:
                print(f"Processed {processed} WISDM windows...")

    if processed == 0:
        raise SystemExit(
            "No RP/GADF samples found. Check doc/images or data/WISDM/wisdm-dataset/raw/watch/accel."
        )

    write_csv(args.output_dir / "aggregate_summary.csv", aggregate.rows())
    write_csv(args.output_dir / "aggregate_by_label.csv", aggregate_by_label.rows())
    save_legend(args.output_dir / "legend.png", args.bins)

    config = {
        "samples_requested": "all" if args.all else args.samples,
        "samples_processed": processed,
        "bins": args.bins,
        "source": args.source,
        "alignment": args.alignment,
        "stride": stride,
        "visual_limit": args.visual_limit if args.all else processed,
        "time_steps": args.time_steps,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "notes": [
            "RP is 127x127 for N=128 because it uses adjacent states.",
            "GADF is 128x128 and is aligned by the selected alignment rule before mask overlap.",
            "same_bin_spatial_overlap_pct measures pixels where RP and GADF fall in the same intensity range.",
            "distribution_overlap_pct ignores position and compares only how much area each range occupies.",
            "In --all mode, CSV metrics are written for every WISDM window, while per-sample PNGs are limited by --visual-limit.",
        ],
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Processed {processed} samples")
    print(f"Wrote results to {args.output_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
