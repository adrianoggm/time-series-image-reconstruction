"""Dataset-agnostic interface: recordings in, traceable windows and leak-free folds out.

Every dataset is reduced to a stream of :class:`Recording` objects, each a
``(T, C)`` array from one acquisition (file, session, event, station pass) with
the group it belongs to (subject, event, station). Everything downstream —
windowing, validation, splitting, caching — only sees that contract, so adding a
dataset means writing one adapter and registering it:

    @register
    class MyAdapter(DatasetAdapter):
        spec = DatasetSpec(name="mine", fs=100.0, channels=("x", "y"), units=("g", "g"))
        def recordings(self):
            yield Recording(data=..., labels=..., group="s01", rec_id="s01/run1")

    w = make_windows(get("mine"), length=128)
    for train, test in group_kfold(w, n_folds=3): ...

Windows never cross a label change or a timestamp gap, carry the recording and
start index they came from, and keep a log of every transformation applied.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

UNLABELLED = -1
GAP_FACTOR = 1.5  # a step longer than GAP_FACTOR / fs is a discontinuity
CACHE_VERSION = 1  # bump when make_windows changes what it produces


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    fs: float                       # Hz
    channels: tuple[str, ...]
    units: tuple[str, ...]
    task: str = "multiclass"        # "multiclass" | "multilabel" | "none"
    classes: tuple[str, ...] = ()   # label names, index = integer label


@dataclass(frozen=True)
class Recording:
    data: np.ndarray                    # (T, C)
    group: str                          # unit a split must never break: subject, event, station
    rec_id: str                         # unique acquisition id
    # Per-sample labels: (T,) int class index for "multiclass", (T, K) 0/1 multi-hot for
    # "multilabel" (K = len(spec.classes); repeat a per-record vector over T). None if unlabelled.
    labels: np.ndarray | None = None
    t: np.ndarray | None = None         # (T,) seconds; None means uniform sampling at spec.fs


class DatasetAdapter:
    spec: DatasetSpec

    def recordings(self) -> Iterator[Recording]:
        raise NotImplementedError

    def config(self) -> dict:
        """Everything that changes the produced recordings; part of the cache key."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def fingerprint(self) -> str:
        """Identity of the raw source data, so the cache notices a re-download. Override for file-backed
        adapters, usually with ``files_fingerprint(self.root.rglob(...))``."""
        return ""


def files_fingerprint(paths) -> str:
    """Cheap source identity: name, size and mtime of every file (no content hashing)."""
    h = hashlib.sha256()
    for p in sorted(Path(p) for p in paths):
        st = p.stat()
        h.update(f"{p.as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    REGISTRY[cls.spec.name] = cls
    return cls


def get(name: str, **kwargs) -> DatasetAdapter:
    import src.adapters  # noqa: F401  (registers the real datasets; imported late to avoid a cycle)

    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def validate(rec: Recording, spec: DatasetSpec) -> dict:
    """Raise on structural errors; count data-quality problems so they can be logged, not hidden."""
    if rec.data.ndim != 2 or rec.data.shape[1] != len(spec.channels):
        raise ValueError(f"{rec.rec_id}: data shape {rec.data.shape}, expected (T, {len(spec.channels)})")
    n = rec.data.shape[0]
    if rec.t is not None and rec.t.shape != (n,):
        raise ValueError(f"{rec.rec_id}: t shape {rec.t.shape}, expected ({n},)")
    if rec.labels is not None:
        if spec.task == "none":
            raise ValueError(f"{rec.rec_id}: labels given for a dataset declared task='none'")
        expected = (n, len(spec.classes)) if spec.task == "multilabel" else (n,)
        if rec.labels.shape != expected:
            raise ValueError(f"{rec.rec_id}: labels shape {rec.labels.shape}, expected {expected}")
        if spec.task == "multilabel" and not np.isin(rec.labels, (0, 1)).all():
            raise ValueError(f"{rec.rec_id}: multilabel labels must be 0/1")
    report = {"samples": n, "nan": int(np.isnan(rec.data).sum()), "inf": int(np.isinf(rec.data).sum()),
              "non_monotonic_t": 0, "gaps": 0}
    if rec.t is not None:
        dt = np.diff(rec.t)
        report["non_monotonic_t"] = int((dt <= 0).sum())
        report["gaps"] = int((dt > GAP_FACTOR / spec.fs).sum())
    return report


def _runs(rec: Recording, spec: DatasetSpec) -> list[tuple[int, int]]:
    """Maximal [start, end) spans with a constant label and no timestamp break."""
    n = rec.data.shape[0]
    breaks = np.zeros(n, dtype=bool)
    breaks[0] = True
    if rec.labels is not None:
        breaks[1:] |= (rec.labels[1:] != rec.labels[:-1]).reshape(n - 1, -1).any(axis=1)
    if rec.t is not None:
        dt = np.diff(rec.t)
        breaks[1:] |= (dt <= 0) | (dt > GAP_FACTOR / spec.fs)
    starts = np.flatnonzero(breaks)
    return list(zip(starts, np.append(starts[1:], n), strict=True))


@dataclass
class Windows:
    X: np.ndarray           # (N, T, C)
    y: np.ndarray           # (N,) int (UNLABELLED when unknown), or (N, K) multi-hot for multilabel
    groups: np.ndarray      # (N,) str
    rec_id: np.ndarray      # (N,) str — together with `start`, where each window came from
    start: np.ndarray       # (N,) int sample offset inside its recording
    meta: dict = field(default_factory=dict)


def make_windows(adapter: DatasetAdapter, length: int, step: int | None = None,
                 drop_nonfinite: bool = True, dtype=None) -> Windows:
    """Segment every recording into windows of ``length`` samples every ``step`` (default: no overlap).

    Tails shorter than ``length`` are dropped. Windows with NaN/Inf are dropped
    when ``drop_nonfinite`` and counted in ``meta["dropped"]``. ``dtype`` (e.g.
    ``np.float32``) halves memory for the large datasets; None keeps the source dtype.
    """
    spec, step = adapter.spec, step or length
    X, y, groups, rec_ids, starts, reports = [], [], [], [], [], {}
    dropped = {"nonfinite": 0, "tail_samples": 0}
    for rec in adapter.recordings():
        if rec.rec_id in reports:
            raise ValueError(f"duplicate rec_id {rec.rec_id!r}")
        reports[rec.rec_id] = validate(rec, spec)
        for a, b in _runs(rec, spec):
            last = a
            for s in range(a, b - length + 1, step):
                win = rec.data[s:s + length]
                last = s + length
                if drop_nonfinite and not np.isfinite(win).all():
                    dropped["nonfinite"] += 1
                    continue
                X.append(win if dtype is None else win.astype(dtype))
                y.append(UNLABELLED if rec.labels is None else rec.labels[s])
                groups.append(rec.group)
                rec_ids.append(rec.rec_id)
                starts.append(s)
            dropped["tail_samples"] += b - last
    if not X:
        raise ValueError(f"{spec.name}: no window of length {length} survived")
    transforms = [f"segment(length={length}, step={step}, break_on=label|gap)"]
    if drop_nonfinite:
        transforms.append("drop_nonfinite")
    if dtype is not None:
        transforms.append(f"cast({np.dtype(dtype).name})")
    meta = {"dataset": spec.name, "fs": spec.fs, "channels": list(spec.channels), "units": list(spec.units),
            "task": spec.task, "classes": list(spec.classes), "length": length, "step": step,
            "duration_s": length / spec.fs, "adapter_config": adapter.config(), "transforms": transforms,
            "dropped": dropped, "validation": reports}
    return Windows(np.stack(X), np.asarray(y), np.array(groups), np.array(rec_ids), np.array(starts), meta)


def split_units(w: Windows, by: str = "group") -> np.ndarray:
    """The unit a split keeps whole: ``"group"`` (subject/event: LOSO) or ``"trial"`` (group x label: LOTO)."""
    if by == "group":
        return w.groups
    if by == "trial":
        if w.y.ndim != 1:
            raise ValueError("trial units need one class label per window")
        # Zero-padded label keeps lexicographic order equal to (group, label) order, which is what
        # StratifiedGroupKFold's assignment depends on.
        return np.char.add(np.char.add(w.groups.astype(str), "|"), np.char.zfill(w.y.astype(str), 4))
    raise ValueError(f"unknown split unit {by!r}")


def group_kfold(w: Windows, n_folds: int, by: str = "group") -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic folds that never put one unit (see ``split_units``) on both sides.

    Stratified by class for multiclass data; plain group folds for multilabel
    and unlabelled data. No shuffling, so the same windows always give the same
    folds (as the current WISDM pipeline does).
    """
    # ponytail: multilabel folds are not label-stratified; add iterative stratification if PTB-XL folds come out skewed.
    units = split_units(w, by)
    stratify = w.meta.get("task") == "multiclass" and w.y.ndim == 1 and (w.y != UNLABELLED).all()
    splitter = StratifiedGroupKFold(n_splits=n_folds) if stratify else GroupKFold(n_splits=n_folds)
    folds = list(splitter.split(w.X, w.y if stratify else None, groups=units))
    for train, test in folds:
        leaked = set(units[train]) & set(units[test])
        if leaked:
            raise AssertionError(f"{by} units on both sides of a fold: {sorted(leaked)[:5]}")
    return folds


def cached_windows(adapter: DatasetAdapter, length: int, step: int | None = None,
                   cache_dir: str | Path = "data/_windows", dtype=None) -> Windows:
    """``make_windows`` memoised on disk, keyed by spec, adapter config, source fingerprint and parameters."""
    key_src = json.dumps({"v": CACHE_VERSION, "spec": repr(adapter.spec), "config": adapter.config(),
                          "source": adapter.fingerprint(), "length": length, "step": step or length,
                          "dtype": None if dtype is None else np.dtype(dtype).name}, sort_keys=True, default=str)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{adapter.spec.name}-{key}"
    if (path.with_suffix(".npz")).exists():
        z = np.load(path.with_suffix(".npz"), allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text())
        return Windows(z["X"], z["y"], z["groups"], z["rec_id"], z["start"], meta | {"cache": str(path)})
    w = make_windows(adapter, length, step, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), X=w.X, y=w.y, groups=w.groups, rec_id=w.rec_id, start=w.start)
    path.with_suffix(".json").write_text(json.dumps(w.meta | {"cache_key": key_src}, indent=1, default=str))
    return w


@register
class SyntheticAdapter(DatasetAdapter):
    """Multichannel test dataset: per-class sinusoids, one recording per subject.

    Deliberately carries one NaN and one timestamp gap per subject so the
    validation and segmentation paths are exercised.
    """

    spec = DatasetSpec(name="synthetic", fs=50.0, channels=("c0", "c1", "c2", "c3", "c4"),
                       units=("a.u.",) * 5, classes=("slow", "medium", "fast"))

    def __init__(self, n_subjects: int = 6, seconds_per_class: float = 20.0, seed: int = 0):
        self.n_subjects, self.seconds_per_class, self.seed = n_subjects, seconds_per_class, seed

    def recordings(self) -> Iterator[Recording]:
        rng = np.random.default_rng(self.seed)
        fs, n_ch = self.spec.fs, len(self.spec.channels)
        per = int(self.seconds_per_class * fs)
        for s in range(self.n_subjects):
            labels = np.repeat(np.arange(len(self.spec.classes)), per)
            t = np.arange(labels.size) / fs
            freq = (1.0 + 2.0 * labels)[:, None] * (1 + 0.1 * np.arange(n_ch))[None, :]
            data = np.sin(2 * np.pi * freq * t[:, None] + s) + 0.05 * rng.standard_normal((labels.size, n_ch))
            data[per // 2, 0] = np.nan                  # one corrupted sample
            t[per + per // 2:] += 1.0                   # one 1-second dropout inside class 1
            yield Recording(data=data, group=f"subject{s:02d}", rec_id=f"subject{s:02d}/run0", labels=labels, t=t)
