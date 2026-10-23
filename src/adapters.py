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
    ``esc10=True`` keeps the dataset's official 10-class ESC-10 subset (400 clips).
    """

    NATIVE_FS = 44100

    def __init__(self, root: str | Path = "~/datasets/ESC-50/ESC-50-master", target_fs: int | None = None,
                 esc10: bool = False):
        self.root = str(Path(root).expanduser())
        self.target_fs = target_fs
        self.esc10 = esc10
        meta = pd.read_csv(Path(self.root) / "meta" / "esc50.csv")
        if esc10:
            meta = meta[meta.esc10].copy()
        classes = meta.drop_duplicates("target").sort_values("target").category.tolist()
        meta["label"] = meta.target.map({t: i for i, t in enumerate(sorted(meta.target.unique()))})
        self.spec = DatasetSpec(name="esc10" if esc10 else "esc50", fs=float(target_fs or self.NATIVE_FS), channels=("audio",),
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
                            labels=np.full(len(x), row.label))


PTBXL_SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


@register
class PTBXLAdapter(DatasetAdapter):
    """PTB-XL 1.0.3 (Wagner et al., 2020): 21 799 ten-second 12-lead ECGs from 18 869 patients, CC BY 4.0.

    Multilabel over the five diagnostic superclasses, aggregated from
    ``scp_codes`` through ``scp_statements.csv`` (rows with diagnostic == 1);
    records with no diagnostic statement get an all-zero vector and are counted.
    Signals are in mV, read from the 16-bit WFDB records with each lead's gain
    and baseline. The group is the patient, so one patient's ECGs never straddle
    a split. ``fs`` picks records100 (100 Hz) or records500 (500 Hz). The
    dataset's own stratified folds are in the ``strat_fold`` column.
    """

    spec = DatasetSpec(name="ptbxl", fs=100.0, channels=("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3",
                                                         "V4", "V5", "V6"), units=("mV",) * 12, task="multilabel",
                       classes=PTBXL_SUPERCLASSES)

    def __init__(self, root: str | Path = "~/datasets/PTB-XL", fs: int = 100, max_records: int | None = None,
                 seed: int = 0):
        if fs not in (100, 500):
            raise ValueError("PTB-XL ships 100 Hz and 500 Hz records only")
        self.root, self.fs, self.max_records, self.seed = str(Path(root).expanduser()), fs, max_records, seed
        self.spec = DatasetSpec(name="ptbxl", fs=float(fs), channels=PTBXLAdapter.spec.channels, units=("mV",) * 12,
                                task="multilabel", classes=PTBXL_SUPERCLASSES)

    def fingerprint(self) -> str:
        return files_fingerprint([Path(self.root) / "ptbxl_database.csv", Path(self.root) / "scp_statements.csv"])

    def _labels(self, db: pd.DataFrame) -> np.ndarray:
        import ast

        stm = pd.read_csv(Path(self.root) / "scp_statements.csv", index_col=0)
        diag = stm[stm.diagnostic == 1].diagnostic_class.to_dict()
        y = np.zeros((len(db), len(PTBXL_SUPERCLASSES)), dtype=np.int8)
        for i, codes in enumerate(db.scp_codes):
            for code in ast.literal_eval(codes):
                if code in diag:
                    y[i, PTBXL_SUPERCLASSES.index(diag[code])] = 1
        return y

    def recordings(self) -> Iterator[Recording]:
        db = pd.read_csv(Path(self.root) / "ptbxl_database.csv")
        if self.max_records:
            db = db.sample(n=min(self.max_records, len(db)), random_state=self.seed).sort_values("ecg_id")
        y = self._labels(db)
        col = "filename_lr" if self.fs == 100 else "filename_hr"
        for i, row in enumerate(db.itertuples()):
            base = Path(self.root) / getattr(row, col)
            lines = (base.with_suffix(".hea")).read_text().splitlines()
            n_sig, n_samp = int(lines[0].split()[1]), int(lines[0].split()[3])
            gain = np.array([float(ln.split()[2].split("(")[0].split("/")[0]) for ln in lines[1:1 + n_sig]])
            base_ = np.array([float(ln.split()[2].split("(")[1].split(")")[0]) if "(" in ln.split()[2] else 0.0
                              for ln in lines[1:1 + n_sig]])
            raw = np.fromfile(base.with_suffix(".dat"), dtype="<i2").reshape(n_samp, n_sig)
            yield Recording(data=(raw - base_) / gain, group=str(int(row.patient_id)), rec_id=str(row.ecg_id),
                            labels=np.repeat(y[i][None, :], n_samp, axis=0))


@register
class SwarmMAGAdapter(DatasetAdapter):
    """ESA Swarm MAGx_LR_1B: vector magnetic field B_NEC (nT) at 1 Hz, downloaded through VirES.

    One recording per satellite and day (scripts/_swarm_download.py writes
    ``<root>/<A|B|C>/<YYYY-MM-DD>.npz``). Unlabelled. ``Flags_B`` is a sum of
    bits (Swarm Handbook, SW_MAGx_LR_1B): 1 ASM off, 2 outlier/gap/temperature,
    4 suspicious VFM samples, 8 ASM-VFM discrepancy, 16 VFM gap; 255 means no
    B_NEC was generated. Samples with 255 or with any bit in ``bad_flag_bits``
    (default 2 and 16) become NaN, so ``make_windows`` drops and counts their
    windows. Bit 1 is kept: it is set on almost every Swarm C sample (ASM off)
    while the vector field is still delivered. Timestamps expose any gap. The
    group is the day by default, or the satellite with ``group_by="satellite"``.
    """

    spec = DatasetSpec(name="swarm", fs=1.0, channels=("B_N", "B_E", "B_C"), units=("nT",) * 3, task="none")

    def __init__(self, root: str | Path = "~/datasets/SWARM", satellites: str = "ABC", group_by: str = "day",
                 bad_flag_bits: tuple[int, ...] = (2, 16)):
        if group_by not in ("day", "satellite"):
            raise ValueError("group_by must be 'day' or 'satellite'")
        self.root, self.satellites = str(Path(root).expanduser()), satellites
        self.group_by, self.bad_flag_bits = group_by, tuple(bad_flag_bits)

    def _files(self) -> list[Path]:
        return sorted(p for s in self.satellites for p in (Path(self.root) / s).glob("*.npz"))

    def fingerprint(self) -> str:
        return files_fingerprint(self._files())

    def recordings(self) -> Iterator[Recording]:
        for f in self._files():
            z = np.load(f)
            B = z["B_NEC"].astype(float)
            flags = z["flags"].astype(int)
            bad = flags == 255
            for bit in self.bad_flag_bits:
                bad |= (flags != 255) & (flags & bit != 0)
            B[bad] = np.nan
            sat, day = f.parent.name, f.stem
            yield Recording(data=B, group=sat if self.group_by == "satellite" else day, rec_id=f"{sat}/{day}",
                            t=(z["t"] - z["t"][0]) / 1e9)


@register
class STEADAdapter(DatasetAdapter):
    """STEAD (Mousavi et al., 2019): 60-s three-component (E, N, Z) seismograms at 100 Hz, CC BY 4.0.

    Binary task: local earthquake vs noise. The two downloaded chunks hold
    235 426 noise and 200 000 earthquake traces, so a class-balanced random
    subset of ``n_per_class`` traces per class is drawn with ``seed``. The group
    is the earthquake (``source_id``) — one event is recorded by several
    stations — and the station for noise traces, so neither an event nor a noise
    station straddles a split.
    """

    spec = DatasetSpec(name="stead", fs=100.0, channels=("E", "N", "Z"), units=("counts",) * 3,
                       classes=("noise", "earthquake_local"))

    def __init__(self, root: str | Path = "~/datasets/STEAD", chunks: tuple[str, ...] = ("chunk1", "chunk2"),
                 n_per_class: int = 1000, seed: int = 0):
        self.root, self.chunks = str(Path(root).expanduser()), tuple(chunks)
        self.n_per_class, self.seed = n_per_class, seed

    def fingerprint(self) -> str:
        return files_fingerprint([Path(self.root) / f"{c}.csv" for c in self.chunks])

    def recordings(self) -> Iterator[Recording]:
        import h5py

        meta = pd.concat([pd.read_csv(Path(self.root) / f"{c}.csv", low_memory=False).assign(_chunk=c)
                          for c in self.chunks], ignore_index=True)
        picks = [g.sample(n=min(self.n_per_class, len(g)), random_state=self.seed)
                 for _, g in meta.groupby("trace_category")]
        sel = pd.concat(picks).sort_values(["_chunk", "trace_name"])
        for chunk, rows in sel.groupby("_chunk"):
            with h5py.File(Path(self.root) / f"{chunk}.hdf5", "r") as h:
                for row in rows.itertuples():
                    x = np.asarray(h["data"][row.trace_name], dtype=float)
                    label = self.spec.classes.index(row.trace_category)
                    group = (f"event:{row.source_id}" if row.trace_category == "earthquake_local"
                             else f"station:{row.network_code}.{row.receiver_code}")
                    yield Recording(data=x, group=group, rec_id=row.trace_name, labels=np.full(len(x), label))


EMG_GESTURES = ("rest", "fist", "wrist_flexion", "wrist_extension", "radial_deviation", "ulnar_deviation",
                "extended_palm")


@register
class EMGGesturesAdapter(DatasetAdapter):
    """UCI "EMG Data for Gestures" (Krilova et al., DOI 10.24432/C5ZP5C, CC BY 4.0): 8-channel Myo armband sEMG.

    36 subjects, two series each, static gestures of 3 s separated by 3 s pauses.
    Label 0 (unmarked pauses) is excluded like PAMAP2's transients; label 7
    (extended palm) is excluded unless ``include_palm`` because only a few
    subjects performed it. The logged time column advances in ~1 ms steps by
    repeating the last value, while the signal itself changes about 100 times
    per second (measured 91-111 Hz across files; UCI states no rate), so each
    file is resampled with sample-and-hold onto a uniform ``fs`` grid (100 Hz
    by default). One recording per gesture run; the group is the subject.
    """

    spec = DatasetSpec(name="emg_gestures", fs=100.0, channels=tuple(f"ch{k}" for k in range(1, 9)),
                       units=("V",) * 8, classes=EMG_GESTURES[:6])

    def __init__(self, root: str | Path = "~/datasets/EMG-Gestures/EMG_data_for_gestures-master", fs: float = 100.0,
                 include_palm: bool = False):
        self.root, self.fs, self.include_palm = str(Path(root).expanduser()), float(fs), include_palm
        classes = EMG_GESTURES if include_palm else EMG_GESTURES[:6]
        self.spec = DatasetSpec(name="emg_gestures", fs=self.fs, channels=EMGGesturesAdapter.spec.channels,
                                units=("V",) * 8, classes=classes)

    def _files(self) -> list[Path]:
        return sorted(Path(self.root).glob("*/*.txt"))

    def fingerprint(self) -> str:
        return files_fingerprint(self._files())

    def recordings(self) -> Iterator[Recording]:
        n_classes = len(self.spec.classes)
        for f in self._files():
            df = pd.read_csv(f, sep="\t", on_bad_lines="skip").dropna()     # one truncated row in the release
            t = df["time"].to_numpy(dtype=float) / 1000.0
            X, lab = df.iloc[:, 1:9].to_numpy(dtype=float), df["class"].to_numpy(dtype=int)
            grid = np.arange(t[0], t[-1], 1.0 / self.fs)
            idx = np.searchsorted(t, grid, side="right") - 1             # sample-and-hold onto the grid
            Xg, yg = X[idx], lab[idx]
            change = np.flatnonzero(yg[1:] != yg[:-1]) + 1
            bounds = np.concatenate([[0], change, [len(yg)]])
            subject = f.parent.name
            for a, b in zip(bounds[:-1], bounds[1:], strict=True):
                if not 1 <= yg[a] <= n_classes:                         # 0 = pause; 7 unless include_palm
                    continue
                yield Recording(data=Xg[a:b], group=subject, rec_id=f"{subject}/{f.stem}/{a}",
                                labels=np.full(b - a, yg[a] - 1), t=grid[a:b])
