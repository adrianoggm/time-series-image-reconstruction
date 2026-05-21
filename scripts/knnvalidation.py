#!/usr/bin/env python3
"""kNN baseline on raw and reconstructed WISDM windows.

Examples
--------
    python -m scripts.knnvalidation --stype GADF
    python -m scripts.knnvalidation --stype GAF --sampling loto --fold 0
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import src.ts_plots as plt
from src.config import DEFAULT_DATA_DIR
from src.paths import ALL_STYPES, xall_path, windowed_npy
from src.results import save_run_metrics

CLASSIFIER_NAME = "knn"


def transformary(y):
    _y = np.ones(len(y)) * -1
    for i in range(len(y)):
        for j in range(5):
            if y[i, j] <= 1.1 and y[i, j] >= 0.9:
                _y[i] = j
    return _y


def _load_reconstruction(stype: str, sampling: str, fold: int, n_folds: int, dataset_folder: str) -> np.ndarray:
    arr = np.load(xall_path(stype, sampling, fold, n_folds, "test", dataset_folder))
    # RP arrays are already stored as (N, time, 3); the other encodings
    # are saved as (N, 3, time) and need a transpose to match.
    if stype == "RP":
        return arr
    return np.transpose(arr, (0, 2, 1))


def run(
    stype: str,
    sampling: str = "loto",
    fold: int = 0,
    n_folds: int = 3,
    dataset_folder: str = DEFAULT_DATA_DIR,
    save_results: bool = True,
    n_neighbors: int = 5,
):
    """Train kNN on the originals, score on originals + reconstructions of `stype`."""
    # Window arrays are written under the recurrence_plot tree regardless of stype.
    X_train = np.load(windowed_npy("training_data",   sampling, fold, n_folds, dataset_folder))
    y_train = np.load(windowed_npy("y_training_data", sampling, fold, n_folds, dataset_folder))
    X_val   = np.load(windowed_npy("validation_data", sampling, fold, n_folds, dataset_folder))
    y_val   = np.load(windowed_npy("y_validation_data", sampling, fold, n_folds, dataset_folder))

    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    y_train_trans = transformary(y_train)
    y_val_trans   = transformary(y_val)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_flat, y_train_trans)

    pred_val = model.predict(X_val_flat)
    original_report = classification_report(y_val_trans, pred_val, output_dict=True, zero_division=0)
    print(f"\n=== {CLASSIFIER_NAME} on original validation_data ({stype}) ===")
    plt.plot_cm(y_val_trans, pred_val)
    print(classification_report(y_val_trans, pred_val, zero_division=0))

    X_recon = _load_reconstruction(stype, sampling, fold, n_folds, dataset_folder)
    X_recon_flat = X_recon.reshape(X_recon.shape[0], -1)
    pred_recon = model.predict(X_recon_flat)
    reconstructed_report = classification_report(y_val_trans, pred_recon, output_dict=True, zero_division=0)
    print(f"\n=== {CLASSIFIER_NAME} on {stype} reconstructions ===")
    plt.plot_cm(y_val_trans, pred_recon)
    print(classification_report(y_val_trans, pred_recon, zero_division=0))

    if save_results:
        out = save_run_metrics(
            stype=stype, classifier=CLASSIFIER_NAME,
            sampling=sampling, fold=fold, n_folds=n_folds,
            original_report=original_report,
            reconstructed_report=reconstructed_report,
            extra={"n_neighbors": n_neighbors},
        )
        print(f"\nSaved metrics to {out}")

    return original_report, reconstructed_report


def build_argparser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stype", default="GAF", choices=ALL_STYPES, help="encoding whose reconstructions to evaluate")
    p.add_argument("--sampling", default="loto", choices=["loto", "loso"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--data-folder", default=DEFAULT_DATA_DIR)
    p.add_argument("--no-save-results", action="store_true", help="skip writing the JSON metrics file")
    p.add_argument("--n-neighbors", type=int, default=5)
    return p


def main():
    args = build_argparser().parse_args()
    run(
        stype=args.stype, sampling=args.sampling, fold=args.fold, n_folds=args.n_folds,
        dataset_folder=args.data_folder, save_results=not args.no_save_results,
        n_neighbors=args.n_neighbors,
    )


if __name__ == "__main__":
    main()
