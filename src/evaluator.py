"""Shared evaluation pipeline for every validator script.

Each validator only needs to plug in three primitives — how to build the
classifier, how to fit it, and how to predict — and ``evaluate_run`` handles
the cross-cutting concerns:

* **E1**: locate the cached model under ``models/<classifier>.<ext>`` and
  load it if present.
* **E2**: cache the *original* classification_report under
  ``results/_originals/<classifier>/...`` so it is reused across stypes.
* **E3**: short-circuit if ``results/<stype>/<classifier>/fold-N.json``
  already exists, unless ``force=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from sklearn.metrics import classification_report

from src.models import (load_original_report, model_path,
                        save_original_report)
from src.paths import results_dir, xall_path, windowed_npy
from src.results import save_run_metrics


class _ModelLike(Protocol):
    """A minimal classifier interface the evaluator depends on."""
    def fit(self, X, y): ...
    def predict(self, X): ...


@dataclass
class ClassifierAdapter:
    """Strategy bundle plugged in by each concrete validator.

    Attributes
    ----------
    name : canonical short name (used to derive cache filenames).
    build : creates a fresh, untrained model.
    fit : trains a model in-place (or returns a new fitted one).
    predict_labels : returns int labels for the supplied X.
    save / load : how to persist the model under ``models/<name>.<ext>``.
    flatten_for_model : whether to reshape X from (N, T, C) to (N, T*C).
    """
    name: str
    build: Callable[[tuple[int, int]], Any]
    fit: Callable[[Any, np.ndarray, np.ndarray], Any]
    predict_labels: Callable[[Any, np.ndarray], np.ndarray]
    save: Callable[[Any, Path], None]
    load: Callable[[Path], Any]
    flatten_for_model: bool = False


def _to_int_labels(y_cat: np.ndarray) -> np.ndarray:
    """Convert one-hot-ish y to int labels with the legacy 0.9..1.1 tolerance."""
    out = np.full(len(y_cat), -1)
    for i in range(len(y_cat)):
        for j in range(y_cat.shape[1]):
            if 0.9 <= y_cat[i, j] <= 1.1:
                out[i] = j
    return out


def _load_reconstruction(stype: str, sampling: str, fold: int, n_folds: int,
                         dataset_folder: str, flatten: bool) -> np.ndarray:
    """Load the aggregate reconstruction array for `stype` and align its axes
    to the (N, T, C) convention every model in this repo expects.

    All four encodings now persist their reconstructions as (N, C, T) with
    T == TIME_STEPS (typically 129) — the old RP-specific (N, C, 128) shape
    has been retired (see `src.recurrence_plots.fix_rotationscale`). Caching
    a legacy file? Re-run reconstruction once with the new code.
    """
    arr = np.load(xall_path(stype, sampling, fold, n_folds, "test", dataset_folder))
    # Stored as (N, C, T) → expose as (N, T, C). Detect axis order from shape
    # rather than hard-coding per-stype, so legacy mixed shapes still work.
    if arr.ndim == 3 and arr.shape[1] in (1, 3) and arr.shape[2] != arr.shape[1]:
        arr = np.transpose(arr, (0, 2, 1))
    if flatten:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def evaluate_run(
    *,
    adapter: ClassifierAdapter,
    stype: str,
    sampling: str,
    fold: int,
    n_folds: int,
    dataset_folder: str,
    force: bool = False,
    train_new: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full evaluation flow for one (encoder, classifier, fold).

    Returns the persisted payload as a dict.
    """
    # E3 — skip if the final JSON is already on disk
    out_path = results_dir(stype, adapter.name, sampling=sampling) / f"fold-{fold}.json"
    if out_path.exists() and not force:
        print(f"[skip] {stype}/{adapter.name}/fold-{fold} already exists at {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    # Validation labels are needed in both branches
    y_val_cat = np.load(windowed_npy("y_validation_data", sampling, fold, n_folds, dataset_folder))
    y_val = _to_int_labels(y_val_cat)

    # E2 — try the cached original report
    original_report = None if (force or train_new) else load_original_report(
        adapter.name, sampling, fold, n_folds
    )
    mp = model_path(adapter.name, sampling, fold, n_folds)

    if original_report is None or not mp.exists() or train_new:
        # Need to fit (model and/or original report missing or invalidated)
        X_train = np.load(windowed_npy("training_data",   sampling, fold, n_folds, dataset_folder))
        y_train_cat = np.load(windowed_npy("y_training_data", sampling, fold, n_folds, dataset_folder))
        X_val   = np.load(windowed_npy("validation_data", sampling, fold, n_folds, dataset_folder))

        y_train = _to_int_labels(y_train_cat)

        if adapter.flatten_for_model:
            X_train_for_fit = X_train.reshape(X_train.shape[0], -1)
            X_val_for_pred  = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_for_fit = X_train
            X_val_for_pred  = X_val

        if mp.exists() and not train_new:
            model = adapter.load(mp)
        else:
            print(f"[train] {adapter.name}: fitting on originals (will cache to {mp})")
            model = adapter.build((X_train.shape[1], X_train.shape[2]))
            model = adapter.fit(model, X_train_for_fit, y_train) or model
            adapter.save(model, mp)

        pred_val = adapter.predict_labels(model, X_val_for_pred)
        original_report = classification_report(y_val, pred_val, output_dict=True, zero_division=0)
        save_original_report(adapter.name, sampling, fold, n_folds, original_report)
    else:
        print(f"[cache] {adapter.name}: reusing cached model + original report")
        model = adapter.load(mp)

    # Always evaluate on reconstructions
    X_recon = _load_reconstruction(stype, sampling, fold, n_folds, dataset_folder,
                                   adapter.flatten_for_model)
    pred_recon = adapter.predict_labels(model, X_recon)
    reconstructed_report = classification_report(y_val, pred_recon, output_dict=True, zero_division=0)

    payload_path = save_run_metrics(
        stype=stype, classifier=adapter.name,
        sampling=sampling, fold=fold, n_folds=n_folds,
        original_report=original_report,
        reconstructed_report=reconstructed_report,
        extra=extra,
    )
    acc_o = original_report.get("accuracy")
    acc_r = reconstructed_report.get("accuracy")
    print(f"[done] {stype}/{adapter.name}/fold-{fold}: orig={acc_o:.3f} rec={acc_r:.3f}  → {payload_path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


__all__ = ["ClassifierAdapter", "evaluate_run"]
