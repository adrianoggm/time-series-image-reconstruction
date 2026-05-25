#!/usr/bin/env python3
"""GRU baseline on raw and reconstructed WISDM windows."""

from __future__ import annotations

import argparse

from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from src.config import DEFAULT_DATA_DIR
from src.evaluator import ClassifierAdapter, evaluate_run
from src.paths import ALL_STYPES

CLASSIFIER_NAME = "gru"


def _build(input_shape, num_classes=5):
    m = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    m.compile(optimizer=Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def _adapter(epochs: int, batch_size: int) -> ClassifierAdapter:
    return ClassifierAdapter(
        name=CLASSIFIER_NAME,
        build=lambda shape: _build(shape),
        fit=lambda m, X, y: (m.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2), m)[1],
        predict_labels=lambda m, X: m.predict(X, verbose=0).argmax(axis=-1),
        save=lambda m, p: m.save(p),
        load=lambda p: load_model(p),
        flatten_for_model=False,
    )


def run(stype, sampling="loto", fold=0, n_folds=3, dataset_folder=DEFAULT_DATA_DIR,
        force=False, train_new=False, epochs=20, batch_size=64):
    return evaluate_run(
        adapter=_adapter(epochs, batch_size), stype=stype, sampling=sampling,
        fold=fold, n_folds=n_folds, dataset_folder=dataset_folder,
        force=force, train_new=train_new,
        extra={"epochs": epochs, "batch_size": batch_size},
    )


def build_argparser(prog=None):
    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stype", default="GAF", choices=ALL_STYPES)
    p.add_argument("--sampling", default="loto", choices=["loto", "loso"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--data-folder", default=DEFAULT_DATA_DIR)
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-new", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    return p


def main():
    a = build_argparser().parse_args()
    run(stype=a.stype, sampling=a.sampling, fold=a.fold, n_folds=a.n_folds,
        dataset_folder=a.data_folder, force=a.force, train_new=a.train_new,
        epochs=a.epochs, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
