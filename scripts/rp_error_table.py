"""Per-dimension pointwise error table for an RP reconstruction, in the exact
format of the paper's tab:recurrence_plot (Dim 0/1/2 + Global Mean, mean $\\pm$ std).

Window-level definition: compute one scalar per (window, dim) — MAE/MRE/MSE/Pearson
averaged over the T time steps — then report mean and std over windows (per dim),
and over all window x dim pairs (global). This reproduces the global column
verified by scripts/_verify_error_tables.py.

    # validate the generator against the backed-up rho=0.75 array:
    python -m scripts.rp_error_table --npy X_all_rec_rho075.npy
    # generate at the new rho=0.60 (default X_all_rec.npy):
    python -m scripts.rp_error_table
"""

from __future__ import annotations

import argparse

import numpy as np

from src.config import DEFAULT_DATA_DIR


def _align(rec, target):
    return rec if rec.shape == target else np.transpose(rec, (0, 2, 1))


def per_window_per_dim(orig, rec):
    """Return dict of (N, C) arrays: one scalar per window per dim."""
    diff = orig - rec
    absd = np.abs(diff)
    safe = np.where(np.abs(orig) < 1.0, 1.0, np.abs(orig))
    oc = orig - orig.mean(axis=1, keepdims=True)
    rc = rec - rec.mean(axis=1, keepdims=True)
    num = (oc * rc).sum(axis=1)
    den = np.linalg.norm(oc, axis=1) * np.linalg.norm(rc, axis=1)
    return {
        "Mean Absolute Error": absd.mean(axis=1),
        "Mean Relative Error": (absd / safe).mean(axis=1),
        "Mean Squared Error": (diff ** 2).mean(axis=1),
        "Pearson Correlation Coefficient": np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampling", default="loto")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--npy", default="X_all_rec.npy",
                    help="reconstruction file name inside the fold test dir")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    data_dir = args.data_dir if args.data_dir.endswith(("/", "\\")) else args.data_dir + "/"
    base = (f"{data_dir}tseries/recurrence_plot/sampling_{args.sampling}/"
            f"{args.n_folds}-fold/fold-{args.fold}/test/")
    orig = np.load(base + "validation_data.npy")
    rec = _align(np.load(base + args.npy), orig.shape)

    metrics = per_window_per_dim(orig, rec)
    ndec = {"Mean Absolute Error": 2, "Mean Relative Error": 2,
            "Mean Squared Error": 2, "Pearson Correlation Coefficient": 5}

    print(f"# {args.sampling} fold-{args.fold} | {args.npy} | N={len(orig)}")
    print(f"{'Metric':<34} {'Dim 0':>16} {'Dim 1':>16} {'Dim 2':>16} {'Global Mean':>16}")
    for name, M in metrics.items():
        d = ndec[name]
        cells = []
        for c in range(3):
            cells.append(f"${M[:, c].mean():.{d}f} \\pm {M[:, c].std():.{d}f}$")
        cells.append(f"${M.mean():.{d}f} \\pm {M.std():.{d}f}$")
        # LaTeX row + aligned console view
        print(f"{name} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
