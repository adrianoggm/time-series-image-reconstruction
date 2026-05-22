"""Central registry for trained-classifier model caches.

Every validator stores its trained model under
``<repo_root>/models/<classifier>.<ext>`` so the cache survives changes to
``data/`` and so the path is identical for every encoding (the same trained
model is evaluated against every stype).

The matching ``results/_originals/<classifier>/fold-<n>.json`` cache stores
the validator's report on the *original* validation_data — that report is
independent of the encoding being studied, so we keep it once and reuse it
across stypes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT

MODELS_DIR: Path = REPO_ROOT / "models"

# Canonical filename suffix for each classifier (Keras → .h5, sklearn → .joblib).
_EXT: dict[str, str] = {
    "knn":           "joblib",   # cheap to refit, but we still cache for consistency
    "lstm":          "h5",
    "gru":           "h5",
    "cnn":           "h5",
    "randomforest":  "joblib",
    "svc":           "joblib",
}


def model_path(classifier: str,
               sampling: str | None = None,
               fold: int | None = None,
               n_folds: int | None = None) -> Path:
    """Return the cached-model path for one ``(classifier, sampling, fold)`` triple.

    The model trained on fold k of LOTO 3-fold is **not** the same model as the
    one trained on fold k+1: the split picks different subjects/trials. To
    avoid silently reusing fold-0's model when evaluating fold-1, the filename
    embeds the split identifier when one is provided:

        models/<clf>__<sampling>-fold-<k>-<n>f.<ext>     (preferred)
        models/<clf>.<ext>                                (legacy, no split info)

    The legacy path is kept for backward compatibility but new callers should
    always pass ``sampling``, ``fold``, and ``n_folds``.
    """
    if classifier not in _EXT:
        raise ValueError(f"Unknown classifier {classifier!r}. Known: {list(_EXT)}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ext = _EXT[classifier]
    if sampling is None or fold is None or n_folds is None:
        return MODELS_DIR / f"{classifier}.{ext}"
    return MODELS_DIR / f"{classifier}__{sampling}-fold-{fold}-{n_folds}f.{ext}"


def original_report_path(classifier: str, sampling: str, fold: int, n_folds: int) -> Path:
    """Per-classifier cache for the report on the raw validation_data set."""
    base = REPO_ROOT / "results" / "_originals" / classifier
    base.mkdir(parents=True, exist_ok=True)
    return base / f"fold-{fold}-{sampling}-{n_folds}f.json"


def load_original_report(classifier: str, sampling: str, fold: int, n_folds: int) -> dict[str, Any] | None:
    p = original_report_path(classifier, sampling, fold, n_folds)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_original_report(classifier: str, sampling: str, fold: int, n_folds: int,
                         report: dict[str, Any]) -> Path:
    p = original_report_path(classifier, sampling, fold, n_folds)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


__all__ = [
    "MODELS_DIR",
    "model_path",
    "original_report_path",
    "load_original_report",
    "save_original_report",
]
