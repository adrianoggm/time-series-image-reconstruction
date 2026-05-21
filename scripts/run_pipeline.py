#!/usr/bin/env python3
"""End-to-end pipeline for one encoding: encode → reconstruct → validate.

Calls each script's ``run(...)`` function directly (no subprocess) so the
shared interpreter is reused and we keep Python tracebacks readable.

Examples
--------
    # Full pipeline on GADF, LOTO, default fold/folds
    python -m scripts.run_pipeline --stype GADF

    # Just classifiers (skip generating images and reconstructing)
    python -m scripts.run_pipeline --stype GAF --skip-encode --skip-reconstruct

    # Re-do everything except training (use cached classifier models)
    python -m scripts.run_pipeline --stype MTF --classifiers knn,svc,randomforest
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

from src.config import DEFAULT_DATA_DIR
from src.paths import ALL_STYPES

# We import lazily inside helpers so a missing optional dep (e.g. tensorflow
# in a CPU-only debug shell) doesn't break the orchestrator's --help.

ALL_CLASSIFIERS = ["knn", "lstm", "cnn", "gru", "randomforest", "svc"]


def _run_encode(stype, sampling, n_folds, dataset_folder, create_numpies):
    from src import activity_data as act
    from src import generate_images as gen
    from keras.utils import to_categorical

    print(f"\n>>> ENCODE  stype={stype} sampling={sampling} n_folds={n_folds}")
    TIME_STEPS, _ = act.get_time_setup(DATASET_NAME="WISDM")
    if create_numpies:
        X, y, sj = act.create_all_numpy_datasets("WISDM", dataset_folder,
                                                 COL_SELECTED_IDXS=list(range(3, 6)))
        y = to_categorical(y, dtype="uint8")
    else:
        X, y, sj = act.load_numpy_datasets("WISDM", dataset_folder, USE_RECONSTRUCTED_DATA=False)
    gen.generate_all_images(X, y, sj, dataset_folder, TIME_STEPS, n_folds, sampling, stype)


def _run_reconstruct(stype, sampling, n_folds, dataset_folder):
    from src import activity_data as act
    from src import reconstruct_tseries as ts

    print(f"\n>>> RECONSTRUCT stype={stype} sampling={sampling} n_folds={n_folds}")
    TIME_STEPS, _ = act.get_time_setup(DATASET_NAME="WISDM")
    X, y, sj = act.load_numpy_datasets("WISDM", dataset_folder, USE_RECONSTRUCTED_DATA=False)
    ts.generate_all_time_series(X, y, sj, dataset_folder, TIME_STEPS, n_folds, sampling, stype)


# Each entry is the module path that exposes a `run(stype, ...)` function.
_CLASSIFIER_MODULES: dict[str, str] = {
    "knn":           "scripts.knnvalidation",
    "lstm":          "scripts.lstmvalidation",
    "cnn":           "scripts.cnnvalidation",
    "gru":           "scripts.gruvalidation",
    "randomforest":  "scripts.randomforestvalidation",
    "svc":           "scripts.Validation",
}


def _run_classifier(clf: str, stype, sampling, fold, n_folds, dataset_folder, train_new):
    import importlib
    mod_path = _CLASSIFIER_MODULES[clf]
    mod = importlib.import_module(mod_path)
    print(f"\n>>> VALIDATE  classifier={clf}  stype={stype}  fold={fold}")
    kwargs = dict(stype=stype, sampling=sampling, fold=fold, n_folds=n_folds,
                  dataset_folder=dataset_folder, save_results=True)
    # `train_new` is supported by every validator that trains a model (all
    # except `knn`, which always fits).
    if "train_new" in mod.run.__code__.co_varnames:
        kwargs["train_new"] = train_new
    mod.run(**kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stype", required=True, choices=ALL_STYPES)
    p.add_argument("--sampling", default="loto", choices=["loto", "loso"])
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--fold", type=int, default=0, help="fold used for the classifier evaluation step")
    p.add_argument("--data-folder", default=DEFAULT_DATA_DIR)
    p.add_argument("--skip-encode", action="store_true", help="reuse existing PNG images")
    p.add_argument("--skip-reconstruct", action="store_true", help="reuse existing tseries/.../X_all_*.npy")
    p.add_argument("--skip-validate", action="store_true", help="skip the classification step")
    p.add_argument("--classifiers", default=",".join(ALL_CLASSIFIERS),
                   help="comma-separated subset of: " + ",".join(ALL_CLASSIFIERS))
    p.add_argument("--create-numpies", action="store_true",
                   help="rebuild the windowed numpy cache from raw WISDM files (first run only)")
    p.add_argument("--train-new", action="store_true",
                   help="force retraining of every classifier that caches a model on disk")
    args = p.parse_args()

    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    bad = [c for c in classifiers if c not in ALL_CLASSIFIERS]
    if bad:
        p.error(f"unknown classifier(s): {bad}; choose from {ALL_CLASSIFIERS}")

    t0 = time.time()
    if not args.skip_encode:
        _run_encode(args.stype, args.sampling, args.n_folds, args.data_folder, args.create_numpies)
    if not args.skip_reconstruct:
        _run_reconstruct(args.stype, args.sampling, args.n_folds, args.data_folder)
    if not args.skip_validate:
        for clf in classifiers:
            _run_classifier(clf, args.stype, args.sampling, args.fold, args.n_folds,
                            args.data_folder, args.train_new)
    print(f"\n>>> pipeline finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
