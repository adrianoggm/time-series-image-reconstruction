#!/usr/bin/env python3
"""GADF pivot diagnostics + n_pivots sensitivity (reviewer follow-up).

Two questions the multi-pivot GADF decoder leaves open:

  (1) Are the arcsin arguments the decoder actually reads really in the gentle
      near-linear regime?  The claim |G^D_pj| <~ 0.70 and |G^D_ep| ~ 0.70 is
      EMPIRICAL (the pivot criterion selects rows near the circular-mean angle
      delta, which need not sit at pi/2).  We measure, on the PNG matrix the
      decoder receives, the max / 99th-percentile of |G^D_pj| and |G^D_ep| and
      how far the selected pivots (and delta) sit from pi/2.

  (2) Does n_pivots matter?  Sweep n_p in {1,3,5,7,9}; report PNG MAE and cost.

    # real WISDM windows (WSL venv):
    python -m scripts._gadf_pivot_stats --n 1500
    # local smoke, no dataset:
    python -m scripts._gadf_pivot_stats --synthetic --n 60
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.GADF import (
    GADF_to_TS,
    gadf_anchor,
    gramian_angular_diff_field,
    normalize_series07,
    normalize_to_zero_one,
)

ALPHA = 0.7
N_PIVOT_GRID = [1, 3, 5, 7, 9]
HEADLINE_NP = 5
STATS_DIR = Path(__file__).resolve().parents[1] / "results" / "_stats"


def _png_matrix(gaf):
    """The pipeline's store/read: normalize -> uint8 -> interp back to [-1,1]."""
    png = np.round(normalize_to_zero_one(gaf) * 255).astype(np.uint8)
    return np.interp(png.astype(float), (0, 255), (-1, 1))


def _pivots(mat, n_pivots):
    row_means = np.mean(mat, axis=1)
    e = int(np.argmax(np.abs(row_means)))
    order = np.argsort(np.abs(row_means))
    piv = order[:max(1, min(n_pivots, mat.shape[0]))]
    return e, piv


def load_windows(args):
    if args.synthetic:
        rng = np.random.default_rng(0)
        t = np.linspace(0, 6 * np.pi, 128)
        out = []
        for _ in range(args.n):
            ch = [amp * np.sin(t + ph) + 0.3 * rng.standard_normal(128)
                  for amp, ph in zip(rng.uniform(1, 20, 3), rng.uniform(0, 3, 3))]
            out.append(np.stack(ch, axis=1))
        return np.asarray(out)
    from src import activity_data as act
    X, _, _ = act.load_numpy_datasets("WISDM", args.data_folder, USE_RECONSTRUCTED_DATA=False)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(args.n, len(X)), replace=False)
    return np.asarray(X)[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    if args.data_folder is None:
        from src.config import DEFAULT_DATA_DIR
        args.data_folder = DEFAULT_DATA_DIR

    windows = load_windows(args)
    n_ax = windows.shape[2]

    g_pj, g_ep, piv_dist, delta_dist = [], [], [], []
    axes = []
    for w in windows:
        for k in range(n_ax):
            x = np.asarray(w[:, k], dtype=float)
            if np.max(x) - np.min(x) < 1e-9:
                continue
            axes.append(x)
            gaf = gramian_angular_diff_field(x, alpha=ALPHA)
            mat = _png_matrix(gaf)                      # what the decoder reads
            e, piv = _pivots(mat, 5)
            g_ep.append(np.abs(mat[e][piv]))            # 5 anchor scalars
            g_pj.append(np.abs(mat[piv].ravel()))       # 5*N pivot-row entries
            phi_true = np.arccos(normalize_series07(x, ALPHA))
            piv_dist.append(np.abs(phi_true[piv] - np.pi / 2))
            c, s = np.cos(phi_true).mean(), np.sin(phi_true).mean()
            delta = np.arctan2(s, c)
            delta_dist.append(abs(delta - np.pi / 2))

    g_pj = np.concatenate(g_pj)
    g_ep = np.concatenate(g_ep)
    piv_dist = np.concatenate(piv_dist)
    delta_dist = np.asarray(delta_dist)
    n_ser = len(axes)

    print(f"windows={len(windows)} axes={n_ax} series={n_ser} alpha={ALPHA}\n")
    print("=== (1) arcsin-argument magnitudes on the PNG matrix the decoder reads ===")
    print(f"{'quantity':28} {'mean':>9} {'p99':>9} {'max':>9}")
    for name, arr in [("|G^D_pj| (pivot rows)", g_pj),
                      ("|G^D_ep| (anchor scalars)", g_ep)]:
        print(f"{name:28} {arr.mean():9.4f} {np.percentile(arr, 99):9.4f} {arr.max():9.4f}")
    print(f"{'arcsin|.| for sin=0.70':28} {'':>9} {'':>9} {'0.70 -> 0.775 rad'}")
    print(f"\n=== pivot / circular-mean angle vs pi/2 (rad) ===")
    print(f"{'quantity':28} {'mean':>9} {'p99':>9} {'max':>9}")
    print(f"{'|phi_p - pi/2|':28} {piv_dist.mean():9.4f} {np.percentile(piv_dist,99):9.4f} {piv_dist.max():9.4f}")
    print(f"{'|delta - pi/2|':28} {delta_dist.mean():9.4f} {np.percentile(delta_dist,99):9.4f} {delta_dist.max():9.4f}")
    print(f"admissible half-spread arccos(-a)-pi/2 = {np.arccos(-ALPHA) - np.pi/2:.4f} rad; "
          f"sin of it = {np.sin(np.arccos(-ALPHA) - np.pi/2):.4f}")

    print(f"\n=== (2) n_pivots sensitivity (PNG round-trip, physical units) ===")
    print(f"{'n_p':>4} {'MAE_png':>10} {'time_ms/axis':>13}")
    sweep = {}
    for n_p in N_PIVOT_GRID:
        errs, t0 = [], time.perf_counter()
        for x in axes:
            gaf = gramian_angular_diff_field(x, alpha=ALPHA)
            mat = _png_matrix(gaf)
            d = [[float(np.max(x))], [float(np.min(x))]]
            xr = GADF_to_TS(mat, 0, d, n_pivots=n_p, alpha=ALPHA)
            errs.append(np.mean(np.abs(x - xr)))
        dt = (time.perf_counter() - t0) * 1000.0 / max(1, len(axes))
        sweep[str(n_p)] = {"mae_png": float(np.mean(errs)), "time_ms_per_axis": float(dt)}
        print(f"{n_p:>4} {np.mean(errs):10.4f} {dt:13.3f}")

    if args.synthetic:
        print("\n(synthetic run: nothing persisted)")
        return

    # Persist, so the supplementary pivot table is traceable to an artefact
    # rather than to a console transcript.
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": {"n_windows": int(len(windows)), "n_axis_series": int(n_ser),
                   "sampling": "full WISDM window set, drawn without replacement",
                   "seed": 0, "alpha": ALPHA, "n_pivot_grid": N_PIVOT_GRID,
                   "headline_n_pivots": HEADLINE_NP,
                   "storage": "per-window min-max -> uint8 -> [-1,1] (the pipeline round trip)"},
        "arcsin_arguments": {
            "abs_G_pj": {"mean": float(g_pj.mean()), "p99": float(np.percentile(g_pj, 99)),
                         "max": float(g_pj.max())},
            "abs_G_ep": {"mean": float(g_ep.mean()), "p99": float(np.percentile(g_ep, 99)),
                         "max": float(g_ep.max())},
            "pivot_angle_from_half_pi": {"mean": float(piv_dist.mean()),
                                         "p99": float(np.percentile(piv_dist, 99)),
                                         "max": float(piv_dist.max())},
        },
        "pivot_sweep": sweep,
    }
    out_json = STATS_DIR / "gadf_pivot_sweep.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cells = " & ".join(
        (f"$\\mathbf{{{sweep[str(n)]['mae_png']:.4f}}}$" if n == HEADLINE_NP
         else f"${sweep[str(n)]['mae_png']:.4f}$")
        for n in N_PIVOT_GRID)
    times = " & ".join(f"${sweep[str(n)]['time_ms_per_axis']:.2f}$" for n in N_PIVOT_GRID)
    tex = "\n".join([
        r"\begin{tabular}{c" + "c" * len(N_PIVOT_GRID) + "}",
        r"\toprule",
        "$n_p$ & " + " & ".join(str(n) for n in N_PIVOT_GRID) + r" \\",
        r"\midrule",
        "MAE (8-bit PNG) & " + cells + r" \\",
        "Time (ms/axis)  & " + times + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    out_tex = STATS_DIR / "gadf_pivot_sweep.tex"
    out_tex.write_text(tex + "\n", encoding="utf-8")
    print(f"\nWROTE {out_json}\nWROTE {out_tex}")


if __name__ == "__main__":
    main()
