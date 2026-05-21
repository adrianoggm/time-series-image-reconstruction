"""Central path / filename mapping for every encoding.

Before this module, each validator and reconstruction script duplicated
`if stype == "RP": rec_path = ... elif stype == "GAF": ...` blocks pointing
at different subfolders and filename conventions. This module collapses
those tables into one source of truth so that adding a new encoding (e.g.
GADF) requires editing only this file.

All helpers accept `dataset_folder` as a string with trailing separator
(the same convention as `src.config.DEFAULT_DATA_DIR`) and return
:class:`pathlib.Path` objects so callers can compose them safely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.config import DEFAULT_DATA_DIR

# Canonical encoding identifiers used everywhere (CLI, JSON results, paths).
SupportedStype = Literal["RP", "GAF", "GADF", "MTF"]
ALL_STYPES: tuple[str, ...] = ("RP", "GAF", "GADF", "MTF")

# Subfolder under `data/WISDM/plots/` and `data/WISDM/tseries/` where each
# encoding stores its artefacts. RP has historically used a verbose name.
STYPE_FOLDER: dict[str, str] = {
    "RP":   "recurrence_plot",
    "GAF":  "GAF",
    "GADF": "GADF",
    "MTF":  "MTF",
}

# Filename of the aggregated reconstruction array (`np.save` target inside
# `tseries/<encoder>/.../test/`).
STYPE_XALL_BASENAME: dict[str, str] = {
    "RP":   "X_all_rec.npy",
    "GAF":  "X_all_gaf.npy",
    "GADF": "X_all_gadf.npy",
    "MTF":  "X_all_mtf.npy",
}

DataType = Literal["train", "test"]
Sampling = Literal["loto", "loso"]


def _validate_stype(stype: str) -> None:
    if stype not in STYPE_FOLDER:
        raise ValueError(
            f"Unknown stype {stype!r}. Expected one of: {', '.join(ALL_STYPES)}"
        )


def _root(dataset_folder: str | None = None) -> Path:
    return Path(dataset_folder if dataset_folder is not None else DEFAULT_DATA_DIR)


def plots_dir(
    stype: str,
    sampling: Sampling,
    fold: int,
    n_folds: int = 3,
    data_type: DataType = "train",
    label: int | str | None = None,
    dataset_folder: str | None = None,
) -> Path:
    """Folder where the PNG images for `stype` live for a given fold."""
    _validate_stype(stype)
    base = (
        _root(dataset_folder)
        / "plots"
        / STYPE_FOLDER[stype]
        / f"sampling_{sampling}"
        / f"{n_folds}-fold"
        / f"fold-{fold}"
        / data_type
    )
    if label is not None:
        base = base / str(label)
    return base


def tseries_dir(
    stype: str,
    sampling: Sampling,
    fold: int,
    n_folds: int = 3,
    data_type: DataType = "test",
    dataset_folder: str | None = None,
) -> Path:
    """Folder where the reconstructed `.npy` time series for `stype` live."""
    _validate_stype(stype)
    return (
        _root(dataset_folder)
        / "tseries"
        / STYPE_FOLDER[stype]
        / f"sampling_{sampling}"
        / f"{n_folds}-fold"
        / f"fold-{fold}"
        / data_type
    )


def xall_path(
    stype: str,
    sampling: Sampling,
    fold: int,
    n_folds: int = 3,
    data_type: DataType = "test",
    dataset_folder: str | None = None,
) -> Path:
    """Path of the aggregated reconstruction `X_all_<stype>.npy`."""
    return tseries_dir(stype, sampling, fold, n_folds, data_type, dataset_folder) / STYPE_XALL_BASENAME[stype]


def windowed_npy(
    name: Literal[
        "training_data", "y_training_data", "sj_training_data",
        "validation_data", "y_validation_data", "sj_validation_data",
    ],
    sampling: Sampling,
    fold: int,
    n_folds: int = 3,
    dataset_folder: str | None = None,
) -> Path:
    """Path of the per-fold raw window arrays.

    These live under `tseries/recurrence_plot/...` because the original
    `generate_all_time_series` writes them there, regardless of which
    encoding the rest of the pipeline is exercising.
    """
    data_type: DataType = "train" if name.endswith("training_data") else "test"
    return (
        _root(dataset_folder)
        / "tseries"
        / STYPE_FOLDER["RP"]
        / f"sampling_{sampling}"
        / f"{n_folds}-fold"
        / f"fold-{fold}"
        / data_type
        / f"{name}.npy"
    )


def results_dir(
    stype: str,
    classifier: str,
    repo_root: Path | None = None,
) -> Path:
    """Where validators drop their per-run JSON reports."""
    _validate_stype(stype)
    from src.config import REPO_ROOT
    root = repo_root if repo_root is not None else REPO_ROOT
    return root / "results" / stype / classifier


__all__ = [
    "SupportedStype",
    "ALL_STYPES",
    "STYPE_FOLDER",
    "STYPE_XALL_BASENAME",
    "plots_dir",
    "tseries_dir",
    "xall_path",
    "windowed_npy",
    "results_dir",
]
