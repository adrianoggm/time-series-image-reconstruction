#!/usr/bin/env python3
"""Controlled RP geometry ablation for the current production decoder.

Runs four variants over LOTO fold-0 test windows and reports
pointwise fidelity (MAE, median MAE, MSE, Pearson) plus geometry diagnostics
(E_RP, E_overlap, EVR2, MDS stress) and per-window wall-clock. The adopted,
unsigned and legacy-1D arms all use the same metadata-free diagonal inversion;
the signal-range arm deliberately supplies the incorrect signal amplitude range
to isolate the affine-shift failure. The 2-D arms use closed-form orientation.

    python -m scripts._ablation_rp_2d --n 120
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import DEFAULT_DATA_DIR
from src.rp_decoders import (
    align_states,
    closed_form_orientation_candidates,
    decode_window,
    distance_matrix,
    embed_2d,
    orientation_candidates,
    overlap_error_abs,
)

VARIANTS = {
    "Adopted metadata-free + signed": dict(
        source="direct", dims=2, recover="overlap", signed=True,
        rp_mode="diagonal", orientation_method="closed_form"),
    "Incorrect signal-range inversion": dict(
        source="direct", dims=2, recover="overlap", signed=True,
        rp_mode="signal", orientation_method="closed_form"),
    "Without signed selection": dict(
        source="direct", dims=2, recover="overlap", signed=False,
        rp_mode="diagonal", orientation_method="closed_form"),
    "Legacy Jaccard 1-D": dict(
        source="jaccard", dims=1, recover="append", signed=True,
        rp_mode="diagonal", orientation_method="closed_form"),
}
PERCENTILE_RHO = 0.60


def _minmax(window):
    return [[float(np.max(window[:, c])) for c in range(3)],
            [float(np.min(window[:, c])) for c in range(3)]]


def _pearson(a, b):
    a, b = a.ravel(), b.ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _overlap_for_window(img, minmax, cfg):
    """E_overlap (L1) for a 2-D variant; N/A (nan) for 1-D variants.

    # E_overlap here is the MIN-overlap-orientation residual — a source-level, sign-agnostic
    # diagnostic (identical within a (source, rho) pair); it is NOT the overlap of the
    # signed-selected reconstruction.
    """
    if cfg["dims"] != 2:
        return np.nan
    maxs, mins = minmax
    vals = []
    for c in range(3):
        png = img[:, :, c].astype(float)
        if cfg["rp_mode"] == "diagonal":
            R_obs = png - float(np.median(np.diag(png)))
        else:
            R_obs = np.interp(png, (0, 255), (mins[c], maxs[c]))
        Y, _ = embed_2d(distance_matrix(R_obs, source=cfg["source"], rho=PERCENTILE_RHO))
        if cfg["orientation_method"] == "closed_form":
            Qs = closed_form_orientation_candidates(Y)
        else:
            Qs = orientation_candidates()
        cands = [align_states(Y, Q) for Q in Qs]
        a, b, _ = min(cands, key=lambda ab: ab[2])
        vals.append(overlap_error_abs(a, b))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    base = f"{DEFAULT_DATA_DIR}tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/"
    plots = f"{DEFAULT_DATA_DIR}plots/recurrence_plot/sampling_loto/3-fold/fold-0/test/"
    orig = np.load(base + "validation_data.npy")
    y_val = np.load(base + "y_validation_data.npy")
    sj_val = np.load(base + "sj_validation_data.npy")
    m = min(args.n, len(orig))

    acc = {k: dict(mae=[], mse=[], pear=[], e_rp=[], e_ov=[], evr2=[], stress=[], t=[]) for k in VARIANTS}

    for i in range(m):
        label = int(np.argmax(y_val[i]))
        sj = sj_val[i][0]
        img = cv2.imread(f"{plots}{label}/{sj}x{i}.png")
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mm = _minmax(orig[i])
        for name, cfg in VARIANTS.items():
            # Time_ms below measures only the decode_window call; it EXCLUDES the
            # diagnostic-only _overlap_for_window recomputation done later in this loop,
            # so cross-row Time_ms is not strictly apples-to-apples.
            t0 = time.perf_counter()
            rec, metrics = decode_window(img, mm, source=cfg["source"], dims=cfg["dims"],
                                         recover=cfg["recover"], signed=cfg["signed"],
                                         rho=PERCENTILE_RHO,
                                         rp_range="diagonal" if cfg["rp_mode"] == "diagonal" else None,
                                         orientation_method=cfg["orientation_method"])
            dt = time.perf_counter() - t0
            a = acc[name]
            a["mae"].append(float(np.mean(np.abs(orig[i] - rec))))
            a["mse"].append(float(np.mean((orig[i] - rec) ** 2)))
            a["pear"].append(_pearson(orig[i], rec))
            a["e_rp"].append(metrics["e_rp"])
            a["e_ov"].append(_overlap_for_window(img, mm, cfg))
            a["evr2"].append(metrics["evr2"])
            a["stress"].append(metrics["stress"])
            a["t"].append(dt)

    def agg(v):
        v = np.asarray(v, dtype=float)
        return float(np.nanmean(v)) if v.size and not np.all(np.isnan(v)) else float("nan")

    rows = {}
    for name, a in acc.items():
        rows[name] = dict(
            MAE=agg(a["mae"]), MedianMAE=float(np.nanmedian(a["mae"])) if a["mae"] else float("nan"),
            MSE=agg(a["mse"]), Pearson=agg(a["pear"]), E_RP=agg(a["e_rp"]),
            E_overlap=agg(a["e_ov"]), EVR2=agg(a["evr2"]), Stress=agg(a["stress"]),
            Time_ms=1000.0 * agg(a["t"]))

    out_dir = Path("results/_stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    design = dict(n_windows=m, rho=PERCENTILE_RHO,
                  png_inversion="diagonal metadata-free except explicit signal-range control",
                  orientation="closed_form for all 2-D arms",
                  sampling="loto/3-fold/fold-0/test")
    json_rows = {
        name: {key: (None if np.isnan(value) else value) for key, value in row.items()}
        for name, row in rows.items()
    }
    (out_dir / "rp_geometry_ablation.json").write_text(
        json.dumps(dict(design=design, variants=json_rows), indent=2))

    cols = ["MAE", "MedianMAE", "MSE", "Pearson", "E_RP", "E_overlap", "EVR2", "Stress", "Time_ms"]
    lines = [f"# RP geometry ablation (LOTO fold-0, {m} windows)", "",
             "The adopted, unsigned and legacy-1D arms use metadata-free diagonal centring; "
             "the signal-range arm is the explicit incorrect-inversion control. "
             "All 2-D arms use closed-form orientation.", "",
             "| Method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name, r in rows.items():
        cells = [name] + [("N/A" if np.isnan(r[c]) else f"{r[c]:.4f}") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("> E_overlap is the min-overlap-orientation residual (source-level, "
                 "sign-agnostic — identical within a source); Time_ms excludes this "
                 "diagnostic recomputation.")
    md = "\n".join(lines) + "\n"
    (out_dir / "rp_geometry_ablation.md").write_text(md)

    tex = [r"\begin{tabular}{lccc}", r"\toprule",
           r"Variant & MAE (m/s$^2$) & Pearson & $\mathrm{EVR}_2$ \\",
           r"\midrule"]
    for name, row in rows.items():
        tex.append(f"{name} & {row['MAE']:.3f} & {row['Pearson']:.3f} & "
                   f"{row['EVR2']:.3f}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "rp_geometry_ablation.tex").write_text("\n".join(tex) + "\n")
    print(md)


if __name__ == "__main__":
    main()
