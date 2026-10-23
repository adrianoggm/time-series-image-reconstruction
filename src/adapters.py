"""Concrete dataset adapters for the interface in ``src/datasets.py``.

Importing this module registers them; ``src.datasets.get`` does it on demand.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_DATA_DIR
from src.datasets import DatasetAdapter, DatasetSpec, Recording, files_fingerprint, register

WISDM_ACTIVITIES = ("Walking", "Jogging", "Stairs", "Sitting", "Standing")


@register
class WISDMAdapter(DatasetAdapter):
    """WISDM v2 smartwatch accelerometer, five activities, 51 subjects, 20 Hz.

    Reproduces ``src/activity_data.py`` exactly: rows in CSV order, one
    recording per maximal run of constant (subject, activity), timestamps
    ignored, and the integer label of an activity is its order of first
    appearance in the filtered CSV.
    """

    spec = DatasetSpec(name="wisdm", fs=20.0, channels=("x", "y", "z"), units=("m/s^2",) * 3,
                       classes=WISDM_ACTIVITIES)

    def __init__(self, root: str | Path = DEFAULT_DATA_DIR):
        self.root = str(root)

    def fingerprint(self) -> str:
        return files_fingerprint([Path(self.root) / "wisdm-watch-acc-dataset.csv"])

    def recordings(self) -> Iterator[Recording]:
        df = pd.read_csv(Path(self.root) / "wisdm-watch-acc-dataset.csv")
        df = df[df.activity.isin(WISDM_ACTIVITIES)]
        order = list(df.activity.unique())
        if tuple(order) != WISDM_ACTIVITIES:
            raise ValueError(f"activity order {order} differs from the spec; labels would be permuted")
        X = df[["x_axis", "y_axis", "z_axis"]].to_numpy()
        y = df.activity.map({a: i for i, a in enumerate(order)}).to_numpy()
        sj = df.user_id.to_numpy()
        change = np.flatnonzero((sj[1:] != sj[:-1]) | (y[1:] != y[:-1])) + 1
        bounds = np.concatenate([[0], change, [len(df)]])
        seen: dict = {}
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            key = (int(sj[a]), int(y[a]))
            seen[key] = seen.get(key, -1) + 1
            yield Recording(data=X[a:b], group=str(key[0]), rec_id=f"{key[0]}/{order[key[1]]}/{seen[key]}",
                            labels=y[a:b])
