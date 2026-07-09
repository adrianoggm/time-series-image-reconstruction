"""Grid search for the RP recurrence-rate parameter rho.

Rho is the percentile threshold used in ``CreateCostMatrix`` when rebuilding the
weighted graph during RP reconstruction (paper: recurrence-rate parameter,
rho=0.75 -> 75th percentile). It only affects *reconstruction*, not the encoded
images, so the search reuses the RP PNGs already on disk and just re-reconstructs
each validation window under every candidate rho.

For each rho we compute the same global pointwise metrics behind the paper's RP
error table (``src.calculate_ts_errors.pointwise_global``, on the LOTO fold-0
validation set by default), so the search is comparable 1:1 with the reported
numbers. Emits a CSV + JSON table and an "error vs rho" figure to justify the
chosen value.

    python -m scripts.search_rp_rho --sampling loto --fold 0 --n-folds 3
    python -m scripts.search_rp_rho --grid 20,40,60,75,90 --metric mae --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")  # headless (WSL) — write figures, never open a window
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import src.recurrence_plots as rpts
from src.calculate_ts_errors import pointwise_global
from src.config import DEFAULT_DATA_DIR

# Higher-is-better only for Pearson; the error metrics are minimised.
_METRIC_IDX = {"mae": 0, "mre": 1, "mse": 2, "pearson": 3}
_MAXIMISE = {"pearson"}
INCUMBENT = 75.0  # rho currently baked into the pipeline / paper


def _minmax(window):
    """Per-channel [maxes, mins] for one window, mirroring dictionary_val in the pipeline."""
    return [
        [float(np.max(window[:, c])) for c in range(3)],
        [float(np.min(window[:, c])) for c in range(3)],
    ]


def _recon_window(i, img_path, minmax, percentile):
    """Reconstruct a single window from its PNG under one rho. Worker (picklable)."""
    import cv2  # imported in-worker so the pool doesn't pickle the module handle

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"missing RP image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Single-entry dict keyed by i — Reconstruct_RP indexes dictionary[valor].
    rp, _ = rpts.Reconstruct_RP(img, {i: minmax}, i, percentile=percentile)
    return i, np.asarray(rp)


def _align(rec, target_shape):
    """Normalise (N, C, T) -> (N, T, C) to match the originals; identity if already aligned."""
    return rec if rec.shape == target_shape else np.transpose(rec, (0, 2, 1))


def _evaluate(tasks, orig, n, percentile):
    """Reconstruct the whole fold under one rho and score it with the paper metric."""
    # ponytail: pool recreated per rho — setup overhead is noise next to
    # N Dijkstra+MDS reconstructions; not worth one giant futures list.
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(_recon_window, i, path, mm, percentile) for (i, path, mm) in tasks]
        res = [f.result() for f in tqdm(futs, total=n, desc=f"rho={percentile / 100:.3f}")]
    res.sort(key=lambda t: t[0])
    rec = _align(np.array([r for _, r in res]), orig.shape)
    mae, mre, mse, pe = pointwise_global(orig, rec)
    row = {"rho": percentile / 100, "percentile": percentile, "mae": mae, "mre": mre,
           "mse": mse, "rmse": float(np.sqrt(mse)), "pearson": pe}
    print(f"  rho={percentile / 100:.3f}  MAE={mae:.4f}  RMSE={np.sqrt(mse):.4f}  Pearson={pe:.4f}")
    return row


def _golden(tasks, orig, n, key, bounds_str):
    """scipy bounded scalar minimiser (Brent / golden-section) over rho.

    The optimiser picks each rho to probe — deterministic, ~10-15 evaluations to
    converge on a smooth unimodal objective, no hand-picked points. Returns
    ``(rows_sorted_by_rho, best_row)``; every probed rho is recorded for the trace.
    """
    from scipy.optimize import minimize_scalar

    lo, hi = (float(x) for x in bounds_str.split(","))
    sign = -1.0 if key in _MAXIMISE else 1.0
    memo = {}

    def objective(rho_frac):
        p = round(float(rho_frac) * 100, 1)   # 0.1-percentile grid -> cache repeat probes
        if p not in memo:
            memo[p] = _evaluate(tasks, orig, n, p)
        return sign * memo[p][key]

    res = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                          options={"xatol": 0.01, "maxiter": 25})
    # Score the incumbent so the improvement figure is self-contained.
    if INCUMBENT not in memo:
        memo[INCUMBENT] = _evaluate(tasks, orig, n, INCUMBENT)

    best_p = round(float(res.x) * 100, 1)
    if best_p not in memo:
        memo[best_p] = _evaluate(tasks, orig, n, best_p)
    rows = sorted(memo.values(), key=lambda r: r["percentile"])
    print(f"\ngolden-section converged at rho={best_p / 100:.3f} after {len(memo)} evaluations")
    return rows, memo[best_p]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampling", default="loto")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--method", default="grid", choices=("grid", "golden"),
                    help="grid = exhaustive sweep of --grid; golden = scipy bounded scalar minimiser")
    ap.add_argument("--grid", default="10,20,30,40,50,60,70,75,80,90",
                    help="grid mode: comma-separated percentiles (0-100); 75 is added if absent")
    ap.add_argument("--bounds", default="0.55,0.80",
                    help="golden mode: rho search interval, e.g. 0.55,0.80")
    ap.add_argument("--metric", default="mae", choices=list(_METRIC_IDX))
    ap.add_argument("--limit", type=int, default=None,
                    help="reconstruct only the first N windows (smoke test)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = args.data_dir if args.data_dir.endswith(("/", "\\")) else args.data_dir + "/"
    fold_test = (f"{data_dir}tseries/recurrence_plot/sampling_{args.sampling}/"
                 f"{args.n_folds}-fold/fold-{args.fold}/test/")
    plots_dir = (f"{data_dir}plots/recurrence_plot/sampling_{args.sampling}/"
                 f"{args.n_folds}-fold/fold-{args.fold}/test/")

    orig = np.load(fold_test + "validation_data.npy")            # (N, T, C)
    y_val = np.load(fold_test + "y_validation_data.npy")         # (N, n_classes)
    sj_val = np.load(fold_test + "sj_validation_data.npy")       # (N, 1)

    n = len(orig) if args.limit is None else min(args.limit, len(orig))
    orig = orig[:n]

    # Precompute per-window (image path, minmax) once — identical across rho.
    tasks = []
    for i in range(n):
        label = int(np.argmax(y_val[i]))
        sj = sj_val[i][0]
        tasks.append((i, f"{plots_dir}{label}/{sj}x{i}.png", _minmax(orig[i])))

    key = args.metric
    print(f"Windows: {n} | method: {args.method} | optimising: {key.upper()}")

    if args.method == "grid":
        grid = sorted({float(g) for g in args.grid.split(",")} | {INCUMBENT})
        print(f"grid (percentiles): {grid}")
        rows = [_evaluate(tasks, orig, n, p) for p in grid]
        best = (max if key in _MAXIMISE else min)(rows, key=lambda r: r[key])
    else:
        print(f"bounds: {args.bounds}")
        rows, best = _golden(tasks, orig, n, key, args.bounds)
    base = next(r for r in rows if r["percentile"] == INCUMBENT)
    delta = base[key] - best[key] if key not in _MAXIMISE else best[key] - base[key]
    pct = (delta / base[key] * 100) if base[key] else 0.0

    print(f"\nBest rho by {key.upper()}: {best['rho']:.2f} "
          f"({key}={best[key]:.4f}) vs incumbent 0.75 ({key}={base[key]:.4f}) "
          f"-> {pct:+.1f}% {'improvement' if delta > 0 else 'change'}")

    out_dir = args.out or f"{data_dir}../../results/_rho_search"
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"rho_search_{args.sampling}-fold-{args.fold}-{args.n_folds}f_{args.method}"

    with open(os.path.join(out_dir, stem + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, stem + ".json"), "w") as fh:
        json.dump({"metric": key, "best": best, "incumbent": base,
                   "improvement_pct": pct, "rows": rows}, fh, indent=2)

    _plot(rows, best, os.path.join(out_dir, stem + ".png"),
          f"RP reconstruction error vs rho ({args.sampling} fold-{args.fold})")
    print(f"\nWrote {stem}.{{csv,json,png}} to {out_dir}")


def _plot(rows, best, path, title):
    x = [r["rho"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [r["mae"] for r in rows], "o-", label="MAE")
    ax.plot(x, [r["rmse"] for r in rows], "s-", label="RMSE")
    ax.axvline(INCUMBENT / 100, ls="--", color="grey", label="rho=0.75 (current)")
    ax.axvline(best["rho"], ls=":", color="red", label=f"best={best['rho']:.2f}")
    ax.set_xlabel(r"recurrence rate $\rho$")
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
