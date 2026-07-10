"""Emit the per-dimension pointwise error-table rows for each encoder on LOTO
fold-0, matching the paper's error tables (MAE/MRE/MSE/Pearson, mean $\\pm$ std
over windows; Dim 0/1/2 + Global Mean). GASF uses 4 decimals, the others 2;
Pearson always 5.

    python -m scripts.emit_error_tables
"""

from __future__ import annotations

import numpy as np

from src.config import DEFAULT_DATA_DIR

FOLD = "sampling_loto/3-fold/fold-0/test"
ORIGP = f"{DEFAULT_DATA_DIR}tseries/recurrence_plot/{FOLD}/validation_data.npy"

# label -> (tseries subdir, X_all file, decimals for MAE/MRE/MSE)
ENCS = {
    "tab:recurrence_plot": ("recurrence_plot", "X_all_rec.npy", 2),
    "tab:markov_fields":   ("MTF",             "X_all_mtf.npy", 2),
    "tab:grammian_fields": ("GAF",             "X_all_gaf.npy", 4),
    "tab:gadf_fields":     ("GADF",            "X_all_gadf.npy", 2),
}


def _align(rec, target):
    return rec if rec.shape == target else np.transpose(rec, (0, 2, 1))


def _pwpd(orig, rec):
    diff = orig - rec
    absd = np.abs(diff)
    safe = np.where(np.abs(orig) < 1.0, 1.0, np.abs(orig))
    oc = orig - orig.mean(axis=1, keepdims=True)
    rc = rec - rec.mean(axis=1, keepdims=True)
    num = (oc * rc).sum(axis=1)
    den = np.linalg.norm(oc, axis=1) * np.linalg.norm(rc, axis=1)
    pe = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    return {
        "Mean Absolute Error": absd.mean(axis=1),
        "Mean Relative Error": (absd / safe).mean(axis=1),
        "Mean Squared Error": (diff ** 2).mean(axis=1),
        "Pearson Correlation Coefficient": pe,
    }


def emit(label):
    subdir, npy, d = ENCS[label]
    orig = np.load(ORIGP)
    rec = _align(np.load(f"{DEFAULT_DATA_DIR}tseries/{subdir}/{FOLD}/{npy}"), orig.shape)
    M = _pwpd(orig, rec)
    rows = []
    for name, arr in M.items():
        dec = 5 if name.startswith("Pearson") else d
        cells = [f"${arr[:, c].mean():.{dec}f} \\pm {arr[:, c].std():.{dec}f}$" for c in range(3)]
        cells.append(f"${arr.mean():.{dec}f} \\pm {arr.std():.{dec}f}$")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def main():
    for label in ENCS:
        print(f"%%%%% {label} %%%%%")
        print(emit(label))
        print()


if __name__ == "__main__":
    main()
