#!/usr/bin/env python3
"""GRU baseline on raw and reconstructed WISDM windows."""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import src.ts_plots as plt
from src.config import DEFAULT_DATA_DIR
from src.paths import ALL_STYPES, xall_path, windowed_npy
from src.results import save_run_metrics

CLASSIFIER_NAME = "gru"


def transformary(y):
    _y = np.ones(len(y)) * -1
    for i in range(len(y)):
        for j in range(5):
            if y[i, j] <= 1.1 and y[i, j] >= 0.9:
                _y[i] = j
    return _y


def create_gru_model(input_shape, num_classes=5):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _load_reconstruction(stype, sampling, fold, n_folds, dataset_folder):
    arr = np.load(xall_path(stype, sampling, fold, n_folds, "test", dataset_folder))
    return arr if stype == "RP" else np.transpose(arr, (0, 2, 1))


def run(
    stype: str,
    sampling: str = "loto",
    fold: int = 0,
    n_folds: int = 3,
    dataset_folder: str = DEFAULT_DATA_DIR,
    save_results: bool = True,
    model_path: str = "modelo_gruX.h5",
    train_new: bool = False,
    epochs: int = 20,
    batch_size: int = 64,
):
    X_train = np.load(windowed_npy("training_data",   sampling, fold, n_folds, dataset_folder))
    y_train = np.load(windowed_npy("y_training_data", sampling, fold, n_folds, dataset_folder))
    X_val   = np.load(windowed_npy("validation_data", sampling, fold, n_folds, dataset_folder))
    y_val   = np.load(windowed_npy("y_validation_data", sampling, fold, n_folds, dataset_folder))

    y_train_trans = transformary(y_train)
    y_val_trans   = transformary(y_val)

    if train_new or not os.path.exists(model_path):
        model = create_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train_trans, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_val, y_val_trans))
        model.save(model_path)
    else:
        model = load_model(model_path)

    pred_val = model.predict(X_val).argmax(axis=-1)
    original_report = classification_report(y_val_trans, pred_val, output_dict=True, zero_division=0)
    print(f"\n=== {CLASSIFIER_NAME} on original validation_data ({stype}) ===")
    plt.plot_cm(y_val_trans, pred_val)
    print(classification_report(y_val_trans, pred_val, zero_division=0))

    X_recon = _load_reconstruction(stype, sampling, fold, n_folds, dataset_folder)
    pred_recon = model.predict(X_recon).argmax(axis=-1)
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
            extra={"model_path": model_path, "train_new": train_new},
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
    p.add_argument("--model-path", default="modelo_gruX.h5")
    p.add_argument("--train-new", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    return p


def main():
    args = build_argparser().parse_args()
    run(
        stype=args.stype, sampling=args.sampling, fold=args.fold, n_folds=args.n_folds,
        dataset_folder=args.data_folder, save_results=not args.no_save_results,
        model_path=args.model_path, train_new=args.train_new,
        epochs=args.epochs, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
