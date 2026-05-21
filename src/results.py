"""Persist per-run validator metrics as structured JSON.

Every validator (`scripts/knnvalidation.py`, ..., `scripts/Validation.py`)
used to print ``classification_report`` to stdout. That made comparisons
across encodings/classifiers tedious. This module persists the same metrics
as JSON under ``results/<stype>/<classifier>/fold-<n>.json`` so they can be
diff-ed, plotted, or fed into ``scripts.summarize_results``.

JSON layout::

    {
      "stype": "GADF",
      "classifier": "knn",
      "sampling": "loto",
      "fold": 0,
      "n_folds": 3,
      "timestamp": "2026-05-21T13:42:00",
      "evaluations": {
        "original":      { ... classification_report dict ... },
        "reconstructed": { ... classification_report dict ... }
      }
    }
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.paths import results_dir


def save_run_metrics(
    *,
    stype: str,
    classifier: str,
    sampling: str,
    fold: int,
    n_folds: int,
    original_report: dict[str, Any],
    reconstructed_report: dict[str, Any],
    extra: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Write a per-run JSON report and return its path.

    Pass ``output_dict=True`` to ``sklearn.metrics.classification_report``
    so the input dicts are already JSON-serialisable.
    """
    target_dir = results_dir(stype, classifier, repo_root=repo_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"fold-{fold}.json"

    payload: dict[str, Any] = {
        "stype": stype,
        "classifier": classifier,
        "sampling": sampling,
        "fold": fold,
        "n_folds": n_folds,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "evaluations": {
            "original": original_report,
            "reconstructed": reconstructed_report,
        },
    }
    if extra:
        payload["extra"] = extra

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


__all__ = ["save_run_metrics"]
