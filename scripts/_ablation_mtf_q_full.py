"""Full MTF Q-ablation across LOTO 3 folds.

For each Q in a configured set, this script:
  1. Loads the cached LOTO test windows (3 folds, 8172 windows total)
  2. Re-encodes every window with MarkovTransitionField(n_bins=Q),
     applying the uint8 PNG round-trip to keep the input distribution
     identical to what the production pipeline sees from disk
  3. Decodes each (window, axis) with MTF_to_TS_v2(n_bins=Q)
  4. Computes pointwise MAE/Pearson on the full pooled sample
  5. Loads the cached per-fold classifier models (fit ONCE on originals,
     unchanged across encoder runs) and predicts on every (Q, fold,
     classifier) triple to obtain the downstream-accuracy curve

Output: a tidy table written to results/_stats/mtf_q_ablation.csv plus a LaTeX block
ready to paste into the paper.
"""
from __future__ import annotations

import json
import os
from itertools import product

import joblib
import numpy as np
from PIL import Image  # only to anchor the uint8 round-trip semantics
from pyts.image import MarkovTransitionField
from tensorflow.keras.models import load_model

from src.MTF_v2 import MTF_to_TS_v2
from src.config import DEFAULT_DATA_DIR
from src.models import model_path
from src.paths import windowed_npy

SAMPLING = "loto"
N_FOLDS = 3
FOLDS = (0, 1, 2)
Q_VALUES = (2, 4, 6, 8, 10, 16)

CLASSIFIERS = [
    ("knn",          True),
    ("svc",          True),
    ("randomforest", True),
    ("lstm",         False),
    ("gru",          False),
    ("cnn",          False),
]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _to_labels(y_cat):
    out = np.full(len(y_cat), -1)
    for i in range(len(y_cat)):
        for j in range(y_cat.shape[1]):
            if 0.9 <= y_cat[i, j] <= 1.1:
                out[i] = j
    return out


def _load_model(clf, fold):
    p = model_path(clf, SAMPLING, fold, N_FOLDS)
    if not p.exists():
        raise FileNotFoundError(f"missing trained model: {p}")
    if p.suffix == ".h5":
        if clf == "cnn":
            from scripts.cnnvalidation import statistical_features_native
            return load_model(p, custom_objects={
                "statistical_features_native": statistical_features_native})
        return load_model(p)
    return joblib.load(p)


def _predict(model, X, flatten):
    if flatten:
        X = X.reshape(X.shape[0], -1)
    out = model.predict(X)
    return out.argmax(axis=-1) if (out.ndim == 2) else out.astype(int)


def _encode_window_axis(x_axis_1d: np.ndarray, q: int) -> np.ndarray:
    """Encode a single 1-D axis vector through pyts MarkovTransitionField,
    apply the uint8 round-trip that the disk pipeline applies, and return a
    float matrix in [0, 1] ready for the v2 decoder."""
    T = len(x_axis_1d)
    m = MarkovTransitionField(image_size=T, overlapping=True, n_bins=q)
    img = m.fit_transform(x_axis_1d.reshape(1, -1)).reshape(T, T)  # [0, 1]
    # uint8 round-trip
    img_u8 = np.round(np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    img_back = img_u8.astype(np.float64) / 255.0
    return img_back


def _reconstruct_window(window_3axis: np.ndarray, q: int) -> np.ndarray:
    """For one (T, 3) window, encode + decode every axis and return the (T, 3)
    reconstruction in physical scale (using per-axis window min/max as the
    rest of the pipeline does)."""
    T = window_3axis.shape[0]
    rec = np.empty_like(window_3axis, dtype=np.float64)
    maxes = [window_3axis[:, k].max() for k in range(3)]
    mins = [window_3axis[:, k].min() for k in range(3)]
    for k in range(3):
        img = _encode_window_axis(window_3axis[:, k], q)
        rec[:, k] = MTF_to_TS_v2(img, dictionary=[maxes, mins], axis=k, n_bins=q)
    return rec


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def main():
    print("Loading windows and labels for each fold...")
    fold_data = {}
    for f in FOLDS:
        Xv = np.load(windowed_npy("validation_data", SAMPLING, f, N_FOLDS, DEFAULT_DATA_DIR))
        yc = np.load(windowed_npy("y_validation_data", SAMPLING, f, N_FOLDS, DEFAULT_DATA_DIR))
        fold_data[f] = (Xv, _to_labels(yc))
        print(f"  fold-{f}: {len(yc)} windows")

    rows = []  # one dict per (Q, classifier)
    pointwise = {}  # one row per Q with MAE/Pearson

    for q in Q_VALUES:
        print(f"\n=== Q = {q} ===")
        # 1. Reconstruct every window of every fold
        recon_per_fold = {}
        for f in FOLDS:
            Xv, _ = fold_data[f]
            print(f"  encoding+decoding {len(Xv)} windows for fold-{f}...", flush=True)
            rec = np.empty_like(Xv, dtype=np.float64)
            for i in range(len(Xv)):
                rec[i] = _reconstruct_window(Xv[i], q)
            recon_per_fold[f] = rec

        # 2. Pointwise pooled
        all_orig = np.concatenate([fold_data[f][0] for f in FOLDS], axis=0)
        all_rec  = np.concatenate([recon_per_fold[f]   for f in FOLDS], axis=0)
        diff = all_orig - all_rec
        mae = float(np.abs(diff).mean())
        mae_std = float(np.abs(diff).std())
        oc = all_orig - all_orig.mean(axis=1, keepdims=True)
        rc = all_rec  - all_rec.mean(axis=1, keepdims=True)
        num = (oc * rc).sum(axis=1)
        den = np.linalg.norm(oc, axis=1) * np.linalg.norm(rc, axis=1)
        pe_per = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0).mean(axis=1)
        pearson = float(pe_per.mean())
        pearson_std = float(pe_per.std())
        pointwise[q] = (mae, mae_std, pearson, pearson_std)
        print(f"  pointwise: MAE={mae:.3f}±{mae_std:.3f}  Pearson={pearson:.3f}±{pearson_std:.3f}")

        # 3. Classifier accuracies per fold then mean
        for clf, flatten in CLASSIFIERS:
            accs = []
            for f in FOLDS:
                _, y = fold_data[f]
                m = _load_model(clf, f)
                pred = _predict(m, recon_per_fold[f], flatten)
                accs.append(float((pred == y).mean()))
            mean = float(np.mean(accs))
            std = float(np.std(accs))
            rows.append({"q": q, "classifier": clf,
                         "accs": accs, "mean": mean, "std": std})
            print(f"  {clf:14}: fold accs={[round(a,3) for a in accs]}  mean={mean:.3f}±{std:.3f}")

    # 4. Persist + emit LaTeX
    print("\n=== summary table (LaTeX) ===")
    print(r"\begin{tabular}{rcccc|cccccc}")
    print(r"\toprule")
    print(r"$Q$ & MAE & Pear. & & "
          r"& KNN & SVM & RF & LSTM & GRU & CNN \\")
    print(r"\midrule")
    by_q = {q: {} for q in Q_VALUES}
    for r in rows:
        by_q[r["q"]][r["classifier"]] = r["mean"]
    for q in Q_VALUES:
        mae, _, pear, _ = pointwise[q]
        cells = [f"{q}", f"{mae:.2f}", f"{pear:+.3f}", "&", ""]
        for clf in ("knn", "svc", "randomforest", "lstm", "gru", "cnn"):
            cells.append(f"{by_q[q][clf]:.3f}")
        line = " & ".join(cells[:3]) + " & & " + " & ".join(cells[5:]) + r" \\"
        print(line)
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # CSV
    import csv
    out_csv = "results/_stats/mtf_q_ablation.csv"
    os.makedirs("results/_stats", exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Q", "MAE", "MAE_std", "Pearson", "Pearson_std",
                    "knn", "svc", "randomforest", "lstm", "gru", "cnn"])
        for q in Q_VALUES:
            mae, mae_std, pe, pe_std = pointwise[q]
            row = [q, f"{mae:.4f}", f"{mae_std:.4f}", f"{pe:.4f}", f"{pe_std:.4f}"]
            for clf in ("knn", "svc", "randomforest", "lstm", "gru", "cnn"):
                row.append(f"{by_q[q][clf]:.4f}")
            w.writerow(row)
    print(f"\nCSV written to {out_csv}")


if __name__ == "__main__":
    main()
