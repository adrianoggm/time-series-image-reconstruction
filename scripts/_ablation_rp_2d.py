#!/usr/bin/env python3
"""RP decoder ablation: which RP decoder reconstructs best?

Runs the variant matrix over LOTO fold-0 test windows at rho=0.60 and reports
pointwise fidelity (MAE, median MAE, MSE, Pearson) plus geometry diagnostics
(E_RP, E_overlap, EVR2, MDS stress) and per-window wall-clock. Reconstruction
only — classifier accuracy is a separate, optional phase (see plan Task 6).

    python -m scripts._ablation_rp_2d --n 120
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import DEFAULT_DATA_DIR
from src.recurrence_plots import varRP2
from src.rp_decoders import (
    align_states,
    decode_window,
    distance_matrix,
    embed_2d,
    orientation_candidates,
    overlap_error_abs,
)

VARIANTS = {
    "RP-Jaccard-1D": dict(source="jaccard", dims=1, recover="append", signed=True),
    "RP-GEO-1D":     dict(source="geo",     dims=1, recover="append", signed=True),
    "RP-GEO-2D":     dict(source="geo",     dims=2, recover="overlap", signed=False),
    "RP-DIRECT-2D":  dict(source="direct",  dims=2, recover="overlap", signed=False),
    "RP-DIRECT-2D-S":dict(source="direct",  dims=2, recover="overlap", signed=True),
}
PERCENTILE_RHO = 0.60


def _minmax(window):
    return [[float(np.max(window[:, c])) for c in range(3)],
            [float(np.min(window[:, c])) for c in range(3)]]


def _rp_range(window):
    """Per-channel (R_min, R_max) of the signed RP, matching NormalizeMatrix's
    encode convention (global max, min over NONZERO entries). This is the
    metadata a corrected encoder would persist so the PNG can be inverted to the
    RP's own distance range instead of the signal amplitude range."""
    out = []
    for c in range(3):
        R = varRP2(window[:, c].astype(float), TIME_STEPS=window.shape[0])
        nz = R[R != 0]
        out.append((float(nz.min()), float(R.max())))
    return out


def _pearson(a, b):
    a, b = a.ravel(), b.ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _overlap_for_window(img, minmax, cfg, rp_range=None):
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
        lo, hi = (mins[c], maxs[c]) if rp_range is None else (rp_range[c][0], rp_range[c][1])
        R_obs = np.interp(img[:, :, c].astype(float), (0, 255), (lo, hi))
        Y, _ = embed_2d(distance_matrix(R_obs, source=cfg["source"], rho=PERCENTILE_RHO))
        cands = [align_states(Y, Q) for Q in orientation_candidates()]
        a, b, _ = min(cands, key=lambda ab: ab[2])
        vals.append(overlap_error_abs(a, b))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--scale", choices=("signal", "rp"), default="signal",
                    help="input-decode range: 'signal' (legacy, affinely distorts distances) "
                         "or 'rp' (invert PNG with the RP matrix's own range — the corrected metadata)")
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
        rr = _rp_range(orig[i]) if args.scale == "rp" else None
        for name, cfg in VARIANTS.items():
            # Time_ms below measures only the decode_window call; it EXCLUDES the
            # diagnostic-only _overlap_for_window recomputation done later in this loop,
            # so cross-row Time_ms is not strictly apples-to-apples.
            t0 = time.perf_counter()
            rec, metrics = decode_window(img, mm, source=cfg["source"], dims=cfg["dims"],
                                         recover=cfg["recover"], signed=cfg["signed"],
                                         rho=PERCENTILE_RHO, rp_range=rr)
            dt = time.perf_counter() - t0
            a = acc[name]
            a["mae"].append(float(np.mean(np.abs(orig[i] - rec))))
            a["mse"].append(float(np.mean((orig[i] - rec) ** 2)))
            a["pear"].append(_pearson(orig[i], rec))
            a["e_rp"].append(metrics["e_rp"])
            a["e_ov"].append(_overlap_for_window(img, mm, cfg, rp_range=rr))
            a["evr2"].append(metrics["evr2"])
            a["stress"].append(metrics["stress"])
            a["t"].append(dt)

    def agg(v):
        v = np.asarray(v, dtype=float)
        return float(np.nanmean(v)) if v.size else float("nan")

    rows = {}
    for name, a in acc.items():
        rows[name] = dict(
            MAE=agg(a["mae"]), MedianMAE=float(np.nanmedian(a["mae"])) if a["mae"] else float("nan"),
            MSE=agg(a["mse"]), Pearson=agg(a["pear"]), E_RP=agg(a["e_rp"]),
            E_overlap=agg(a["e_ov"]), EVR2=agg(a["evr2"]), Stress=agg(a["stress"]),
            Time_ms=1000.0 * agg(a["t"]))

    out_dir = Path("results/_stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.scale == "signal" else "_rpscale"
    (out_dir / f"rp_2d_ablation{suffix}.json").write_text(
        json.dumps(dict(n_windows=m, rho=PERCENTILE_RHO, scale=args.scale, variants=rows), indent=2))

    cols = ["MAE", "MedianMAE", "MSE", "Pearson", "E_RP", "E_overlap", "EVR2", "Stress", "Time_ms"]
    lines = [f"# RP 2-D decoder ablation (LOTO fold-0, {m} windows, rho=0.60, scale={args.scale})", "",
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
    (out_dir / f"rp_2d_ablation{suffix}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
