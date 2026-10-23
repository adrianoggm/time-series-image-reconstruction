"""Gate for the dataset interface (src/datasets.py) on the synthetic multichannel dataset.

    python -m scripts._check_datasets
"""
import tempfile

import numpy as np

from src.datasets import REGISTRY, UNLABELLED, Recording, cached_windows, get, group_kfold, make_windows, validate

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
print("all checks passed")
