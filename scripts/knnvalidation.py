#!/usr/bin/env python3
"""kNN baseline on raw and reconstructed WISDM windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.config import DEFAULT_DATA_DIR
from src.evaluator import ClassifierAdapter, evaluate_run
from src.paths import ALL_STYPES

CLASSIFIER_NAME = "knn"


def _adapter(n_neighbors: int) -> ClassifierAdapter:
    return ClassifierAdapter(
        name=CLASSIFIER_NAME,
        build=lambda _shape: KNeighborsClassifier(n_neighbors=n_neighbors),
        fit=lambda m, X, y: m.fit(X, y),
        predict_labels=lambda m, X: m.predict(X),
        save=lambda m, p: joblib.dump(m, p),
        load=lambda p: joblib.load(p),
        flatten_for_model=True,
    )


def run(stype, sampling="loto", fold=0, n_folds=3, dataset_folder=DEFAULT_DATA_DIR,
        force=False, train_new=False, n_neighbors=5):
    return evaluate_run(
        adapter=_adapter(n_neighbors), stype=stype, sampling=sampling,
        fold=fold, n_folds=n_folds, dataset_folder=dataset_folder,
        force=force, train_new=train_new, extra={"n_neighbors": n_neighbors},
    )


def build_argparser(prog=None):
    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stype", default="GAF", choices=ALL_STYPES)
    p.add_argument("--sampling", default="loto", choices=["loto", "loso"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--data-folder", default=DEFAULT_DATA_DIR)
    p.add_argument("--force", action="store_true", help="re-run even if results JSON is cached")
    p.add_argument("--train-new", action="store_true", help="refit the classifier instead of loading the cached model")
    p.add_argument("--n-neighbors", type=int, default=5)
    return p


def main():
    a = build_argparser().parse_args()
    run(stype=a.stype, sampling=a.sampling, fold=a.fold, n_folds=a.n_folds,
        dataset_folder=a.data_folder, force=a.force, train_new=a.train_new,
        n_neighbors=a.n_neighbors)


if __name__ == "__main__":
    main()
