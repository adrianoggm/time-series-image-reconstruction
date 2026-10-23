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


# PAMAP2 activity ids (readme.pdf). 0 = transient between activities, never a class.
PAMAP2_PROTOCOL = {1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running", 6: "cycling",
                   7: "nordic_walking", 12: "ascending_stairs", 13: "descending_stairs", 16: "vacuum_cleaning",
                   17: "ironing", 24: "rope_jumping"}
PAMAP2_OPTIONAL = {9: "watching_tv", 10: "computer_work", 11: "car_driving", 18: "folding_laundry",
                   19: "house_cleaning", 20: "playing_soccer"}
# Column layout of the 54-column .dat rows: 0 timestamp (s), 1 activity id, 2 heart rate, then 17 columns
# per IMU (hand 3-19, chest 20-36, ankle 37-53): temperature, acc +-16 g (3), acc +-6 g (3), gyro, mag, orientation.
PAMAP2_ACC16 = {"hand": 4, "chest": 21, "ankle": 38}


def _pamap2_spec(include_optional: bool) -> DatasetSpec:
    acts = PAMAP2_PROTOCOL | (PAMAP2_OPTIONAL if include_optional else {})
    return DatasetSpec(name="pamap2", fs=100.0,
                       channels=tuple(f"{imu}_acc_{ax}" for imu in PAMAP2_ACC16 for ax in "xyz"),
                       units=("m/s^2",) * 9, classes=tuple(acts[k] for k in sorted(acts)))


@register
class PAMAP2Adapter(DatasetAdapter):
    """PAMAP2 (UCI 231): 9 subjects, three IMUs at 100 Hz; the +-16 g accelerometer of each IMU.

    One recording per maximal run of a constant non-zero activity inside a
    subject file, so transients (activity 0) are excluded and windows never span
    two activities. Heart rate (about 9 Hz) and the saturating +-6 g
    accelerometer are not used. Isolated NaN gaps from wireless drop-outs of up
    to ``max_gap`` samples are filled by linear interpolation; longer gaps stay
    NaN, so ``make_windows`` drops and counts the windows that contain them.
    """

    spec = _pamap2_spec(include_optional=False)

    def __init__(self, root: str | Path = "~/datasets/PAMAP2/PAMAP2_Dataset", include_optional: bool = False,
                 max_gap: int = 10):
        self.root = str(Path(root).expanduser())
        self.include_optional, self.max_gap = include_optional, max_gap
        self.spec = _pamap2_spec(include_optional)

    def _files(self) -> list[Path]:
        dirs = ["Protocol"] + (["Optional"] if self.include_optional else [])
        return sorted(p for d in dirs for p in (Path(self.root) / d).glob("subject*.dat"))

    def fingerprint(self) -> str:
        return files_fingerprint(self._files())

    def recordings(self) -> Iterator[Recording]:
        label_of = {aid: i for i, aid in enumerate(sorted(PAMAP2_PROTOCOL | (PAMAP2_OPTIONAL if self.include_optional else {})))}
        cols = [c + k for c in PAMAP2_ACC16.values() for k in range(3)]
        for path in self._files():
            raw = np.loadtxt(path)
            t, act = raw[:, 0], raw[:, 1].astype(int)
            X = pd.DataFrame(raw[:, cols]).interpolate(limit=self.max_gap, limit_area="inside").to_numpy()
            subject = path.stem.removeprefix("subject")
            change = np.flatnonzero(act[1:] != act[:-1]) + 1
            bounds = np.concatenate([[0], change, [len(act)]])
            for a, b in zip(bounds[:-1], bounds[1:], strict=True):
                if act[a] not in label_of:          # 0 = transient (and optional ids when excluded)
                    continue
                yield Recording(data=X[a:b], group=subject, rec_id=f"{subject}/{path.parent.name}/{a}",
                                labels=np.full(b - a, label_of[act[a]]), t=t[a:b])


@register
class ESC50Adapter(DatasetAdapter):
    """ESC-50 (Piczak, 2015): 2000 five-second clips, 50 environmental-sound classes, CC BY-NC 3.0.

    One recording per clip, mono 16-bit PCM scaled to [-1, 1]. The group is the
    Freesound source recording (``src_file``), so clips cut from the same
    recording never straddle a split. Native rate 44.1 kHz; ``target_fs``
    resamples with SciPy's polyphase filter (anti-aliased), since a 128-sample
    window at 44.1 kHz spans only 2.9 ms. The dataset's own 5 folds are in the
    ``fold`` column of meta/esc50.csv if a run must match published splits.
    """

    NATIVE_FS = 44100

    def __init__(self, root: str | Path = "~/datasets/ESC-50/ESC-50-master", target_fs: int | None = None):
        self.root = str(Path(root).expanduser())
        self.target_fs = target_fs
        meta = pd.read_csv(Path(self.root) / "meta" / "esc50.csv")
        classes = meta.drop_duplicates("target").sort_values("target").category.tolist()
        self.spec = DatasetSpec(name="esc50", fs=float(target_fs or self.NATIVE_FS), channels=("audio",),
                                units=("a.u.",), classes=tuple(classes))
        self._meta = meta

    spec = DatasetSpec(name="esc50", fs=44100.0, channels=("audio",), units=("a.u.",))

    def fingerprint(self) -> str:
        return files_fingerprint([Path(self.root) / "meta" / "esc50.csv"])

    def recordings(self) -> Iterator[Recording]:
        from math import gcd

        from scipy.io import wavfile
        from scipy.signal import resample_poly

        for row in self._meta.itertuples():
            fs, pcm = wavfile.read(Path(self.root) / "audio" / row.filename)
            if fs != self.NATIVE_FS or pcm.ndim != 1:
                raise ValueError(f"{row.filename}: expected mono {self.NATIVE_FS} Hz, got {fs} Hz {pcm.shape}")
            x = pcm.astype(float) / 32768.0
            if self.target_fs and self.target_fs != fs:
                g = gcd(self.target_fs, fs)
                x = resample_poly(x, self.target_fs // g, fs // g)
            yield Recording(data=x[:, None], group=str(row.src_file), rec_id=row.filename,
                            labels=np.full(len(x), row.target))
