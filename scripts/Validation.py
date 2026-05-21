#!/usr/bin/env python3
"""SVM baseline on raw and reconstructed WISDM windows."""

from __future__ import annotations

import argparse

import joblib
from sklearn.svm import SVC

from src.config import DEFAULT_DATA_DIR
from src.evaluator import ClassifierAdapter, evaluate_run
from src.paths import ALL_STYPES

CLASSIFIER_NAME = "svc"


def _adapter(C: float, kernel: str) -> ClassifierAdapter:
    return ClassifierAdapter(
        name=CLASSIFIER_NAME,
        build=lambda _shape: SVC(C=C, kernel=kernel),
        fit=lambda m, X, y: m.fit(X, y),
        predict_labels=lambda m, X: m.predict(X),
        save=lambda m, p: joblib.dump(m, p),
        load=lambda p: joblib.load(p),
        flatten_for_model=True,
    )


def run(stype, sampling="loto", fold=0, n_folds=3, dataset_folder=DEFAULT_DATA_DIR,
        force=False, train_new=False, C=1.0, kernel="rbf"):
    return evaluate_run(
        adapter=_adapter(C, kernel), stype=stype, sampling=sampling,
        fold=fold, n_folds=n_folds, dataset_folder=dataset_folder,
        force=force, train_new=train_new, extra={"C": C, "kernel": kernel},
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
    p.add_argument("--svm-C", type=float, default=1.0, dest="C")
    p.add_argument("--kernel", default="rbf")
    return p


def main():
    a = build_argparser().parse_args()
    run(stype=a.stype, sampling=a.sampling, fold=a.fold, n_folds=a.n_folds,
        dataset_folder=a.data_folder, force=a.force, train_new=a.train_new,
        C=a.C, kernel=a.kernel)


if __name__ == "__main__":
    main()
