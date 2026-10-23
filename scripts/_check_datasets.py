"""Gate for the dataset interface (src/datasets.py) on the synthetic multichannel dataset.

    python -m scripts._check_datasets
"""
import tempfile
from pathlib import Path

import numpy as np

from src.datasets import (
    REGISTRY,
    UNLABELLED,
    DatasetAdapter,
    DatasetSpec,
    Recording,
    cached_windows,
    files_fingerprint,
    get,
    group_kfold,
    make_windows,
    validate,
)

LENGTH = 128

adapter = get("synthetic")
spec = adapter.spec
assert "synthetic" in REGISTRY
recs = {r.rec_id: r for r in adapter.recordings()}
print(f"[ok] registry: {spec.name}, {len(recs)} recordings, {len(spec.channels)} channels @ {spec.fs} Hz")

# Contract [T, C] + metadata.
w = make_windows(adapter, LENGTH)
assert w.X.shape[1:] == (LENGTH, len(spec.channels)), w.X.shape
assert len({len(a) for a in (w.X, w.y, w.groups, w.rec_id, w.start)}) == 1
assert w.meta["fs"] == spec.fs and w.meta["units"] == list(spec.units) and w.meta["duration_s"] == LENGTH / spec.fs
print(f"[ok] contract: X {w.X.shape}, {w.meta['duration_s']} s windows, units {w.meta['units'][0]}")

# Validation counts problems instead of hiding them; structural errors raise.
rep = w.meta["validation"]["subject00/run0"]
assert rep["nan"] == 1 and rep["gaps"] == 1 and rep["non_monotonic_t"] == 0, rep
for bad in (Recording(np.zeros((10, 2)), "g", "bad"), Recording(np.zeros((10, 5)), "g", "bad", labels=np.zeros(9))):
    try:
        validate(bad, spec)
        raise SystemExit("validate accepted a structurally broken recording")
    except ValueError:
        pass
print(f"[ok] validation: {rep}")

# Traceability: every window is exactly the slice it claims to be, inside one label run.
for i in range(len(w.X)):
    rec = recs[w.rec_id[i]]
    sl = slice(w.start[i], w.start[i] + LENGTH)
    assert np.array_equal(w.X[i], rec.data[sl])
    assert (rec.labels[sl] == w.y[i]).all(), "window crosses a label change"
    assert (np.diff(rec.t[sl]) < 1.5 / spec.fs).all(), "window crosses a timestamp gap"
assert w.meta["dropped"]["nonfinite"] == len(recs), w.meta["dropped"]
assert not np.isnan(w.X).any() and (w.y != UNLABELLED).all()
print(f"[ok] traceability + segmentation: {len(w.X)} windows, dropped {w.meta['dropped']}")

# Configurable windowing: 50 % overlap gives roughly twice as many windows.
w_half = make_windows(adapter, LENGTH, step=LENGTH // 2)
assert len(w.X) < len(w_half.X) <= 2 * len(w.X) + 2 * len(recs) * len(spec.classes), (len(w.X), len(w_half.X))
assert "segment(length=128, step=64, break_on=label|gap)" in w_half.meta["transforms"]
print(f"[ok] windowing: step {LENGTH} -> {len(w.X)} windows, step {LENGTH // 2} -> {len(w_half.X)}")

# Deterministic, leak-free folds (group_kfold itself asserts no group straddles a fold).
folds = group_kfold(w, n_folds=3)
again = group_kfold(make_windows(adapter, LENGTH), n_folds=3)
assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again, strict=True)), "folds are not deterministic"
assert np.array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(len(w.X)))
print(f"[ok] splits: 3 folds, test groups {[sorted(set(w.groups[te])) for _, te in folds]}")

# Cache round-trip.
with tempfile.TemporaryDirectory() as d:
    first = cached_windows(adapter, LENGTH, cache_dir=d)
    second = cached_windows(adapter, LENGTH, cache_dir=d)
    other = cached_windows(get("synthetic", seed=1), LENGTH, cache_dir=d)
    assert "cache" in second.meta and "cache" not in first.meta
    assert all(np.array_equal(getattr(first, k), getattr(second, k)) for k in ("X", "y", "groups", "rec_id", "start"))
    assert not np.array_equal(first.X, other.X), "a different adapter config must not hit the same cache entry"
print("[ok] cache: hit on identical config, miss on a different seed")


# Multilabel (PTB-XL-like: one multi-hot vector per record) and unlabelled (Swarm-like) datasets.
class _Multilabel(DatasetAdapter):
    spec = DatasetSpec(name="ml", fs=100.0, channels=("I", "II"), units=("mV", "mV"), task="multilabel",
                       classes=("NORM", "MI", "STTC"))

    def recordings(self):
        rng = np.random.default_rng(0)
        for r in range(9):
            hot = np.array([r % 3 == 0, r % 2 == 0, r % 5 == 0], dtype=np.int8)
            yield Recording(rng.standard_normal((1000, 2)), f"patient{r // 3}", f"rec{r}", labels=np.tile(hot, (1000, 1)))


class _Unlabelled(DatasetAdapter):
    spec = DatasetSpec(name="ul", fs=1.0, channels=("B_N", "B_E", "B_C"), units=("nT",) * 3, task="none")

    def __init__(self, root):
        self.root = root

    def recordings(self):
        for p in sorted(Path(self.root).glob("*.npy")):
            yield Recording(np.load(p), p.stem.split("_")[0], p.stem)

    def fingerprint(self):
        return files_fingerprint(Path(self.root).glob("*.npy"))


ml = make_windows(_Multilabel(), 250, dtype=np.float32)
assert ml.X.dtype == np.float32 and ml.y.shape == (len(ml.X), 3) and "cast(float32)" in ml.meta["transforms"]
for train, test in group_kfold(ml, n_folds=3):
    assert not set(ml.groups[train]) & set(ml.groups[test])
try:
    validate(Recording(np.zeros((10, 2)), "g", "bad", labels=np.full((10, 3), 2)), _Multilabel.spec)
    raise SystemExit("validate accepted non-binary multilabel targets")
except ValueError:
    pass
print(f"[ok] multilabel: X {ml.X.shape} {ml.X.dtype}, y {ml.y.shape}, group folds")

with tempfile.TemporaryDirectory() as d:
    for s in range(4):
        np.save(Path(d) / f"pass{s}_A.npy", np.random.default_rng(s).standard_normal((600, 3)))
    ul_adapter = _Unlabelled(d)
    ul = make_windows(ul_adapter, 128)
    assert (ul.y == UNLABELLED).all() and len(group_kfold(ul, n_folds=2)) == 2
    cache = Path(d) / "cache"
    before = cached_windows(ul_adapter, 128, cache_dir=cache)
    np.save(Path(d) / "pass0_A.npy", np.zeros((700, 3)))            # source re-downloaded/changed
    after = cached_windows(ul_adapter, 128, cache_dir=cache)
    assert "cache" not in after.meta and not np.array_equal(before.X, after.X), "stale cache after source change"
print(f"[ok] unlabelled: {len(ul.X)} windows, GroupKFold; cache invalidated when a source file changes")
print("all checks passed")
