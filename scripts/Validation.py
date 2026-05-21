#!/usr/bin/env python3
"""SVM baseline on raw and reconstructed WISDM windows."""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import src.ts_plots as plt
from src.config import DEFAULT_DATA_DIR
from src.paths import ALL_STYPES, xall_path, windowed_npy
from src.results import save_run_metrics

CLASSIFIER_NAME = "svc"


def transformary(y):
    _y = np.ones(len(y)) * -1
    for i in range(len(y)):
        for j in range(5):
            if y[i, j] <= 1.1 and y[i, j] >= 0.9:
                _y[i] = j
    return _y


def _load_reconstruction_flat(stype, sampling, fold, n_folds, dataset_folder):
    arr = np.load(xall_path(stype, sampling, fold, n_folds, "test", dataset_folder))
    arr = arr if stype == "RP" else np.transpose(arr, (0, 2, 1))
    return arr.reshape(arr.shape[0], -1)


def run(
    stype: str,
    sampling: str = "loto",
    fold: int = 0,
    n_folds: int = 3,
    dataset_folder: str = DEFAULT_DATA_DIR,
    save_results: bool = True,
    model_path: str = "modelo_svcX.joblib",
    train_new: bool = False,
    C: float = 1.0,
    kernel: str = "rbf",
):
    X_train = np.load(windowed_npy("training_data",   sampling, fold, n_folds, dataset_folder))
    y_train = np.load(windowed_npy("y_training_data", sampling, fold, n_folds, dataset_folder))
    X_val   = np.load(windowed_npy("validation_data", sampling, fold, n_folds, dataset_folder))
    y_val   = np.load(windowed_npy("y_validation_data", sampling, fold, n_folds, dataset_folder))

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    y_train_trans = transformary(y_train)
    y_val_trans   = transformary(y_val)

    if train_new or not os.path.exists(model_path):
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train_flat, y_train_trans)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    pred_val = model.predict(X_val_flat)
    original_report = classification_report(y_val_trans, pred_val, output_dict=True, zero_division=0)
    print(f"\n=== {CLASSIFIER_NAME} on original validation_data ({stype}) ===")
    plt.plot_cm(y_val_trans, pred_val)
    print(classification_report(y_val_trans, pred_val, zero_division=0))

    X_recon_flat = _load_reconstruction_flat(stype, sampling, fold, n_folds, dataset_folder)
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
            extra={"model_path": model_path, "C": C, "kernel": kernel, "train_new": train_new},
        )
        print(f"\nSaved metrics to {out}")
    return original_report, reconstructed_report


def build_argparser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stype", default="GAF", choices=ALL_STYPES)
    p.add_argument("--sampling", default="loto", choices=["loto", "loso"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--data-folder", default=DEFAULT_DATA_DIR)
    p.add_argument("--no-save-results", action="store_true")
    p.add_argument("--model-path", default="modelo_svcX.joblib")
    p.add_argument("--train-new", action="store_true")
    p.add_argument("--svm-C", type=float, default=1.0, dest="C")
    p.add_argument("--kernel", default="rbf")
    return p


def main():
    args = build_argparser().parse_args()
    run(
        stype=args.stype, sampling=args.sampling, fold=args.fold, n_folds=args.n_folds,
        dataset_folder=args.data_folder, save_results=not args.no_save_results,
        model_path=args.model_path, train_new=args.train_new,
        C=args.C, kernel=args.kernel,
    )


if __name__ == "__main__":
    main()
