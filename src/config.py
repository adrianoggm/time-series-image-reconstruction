"""Central path/configuration resolution for the repo.

The original code hardcoded absolute paths from three different developer
machines (e.g. ``/home/adriano/Escritorio/...``,
``C:/Users/adria/OneDrive/Documentos/...``). This module replaces those with
a single source of truth: an environment variable with a sensible default
relative to the repository root.

Resolution order for ``DEFAULT_DATA_DIR``:
  1. ``$TFG_DATA_DIR`` if set.
  2. ``./data/WISDM/`` relative to the repo root.

The path is normalised to always end with a trailing separator because
existing code paths concatenate with literals like
``f"{dataset_folder}wisdm-dataset/raw/..."``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def _with_trailing_sep(path: Path) -> str:
    s = str(path)
    return s if s.endswith(("/", os.sep)) else s + os.sep


def get_data_dir() -> str:
    """Return the WISDM data directory as a string with trailing separator.

    Honours the ``TFG_DATA_DIR`` environment variable; otherwise falls back
    to ``<repo_root>/data/WISDM/``.
    """
    env = os.environ.get("TFG_DATA_DIR")
    base = Path(env) if env else REPO_ROOT / "data" / "WISDM"
    return _with_trailing_sep(base)


DEFAULT_DATA_DIR: str = get_data_dir()
DEFAULT_DATASET_NAME: str = os.environ.get("TFG_DATASET_NAME", "WISDM")
