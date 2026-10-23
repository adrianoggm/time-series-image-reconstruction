"""Gate + data report for the ESC-50 adapter.

    python -m scripts._check_esc50
"""
from collections import Counter

import numpy as np

from src.datasets import get, group_kfold, make_windows

native = get("esc50")
recs = list(native.recordings())
assert len(recs) == 2000 and all(r.data.shape == (220500, 1) for r in recs)
assert len(native.spec.classes) == 50 and Counter(int(r.labels[0]) for r in recs) == Counter({k: 40 for k in range(50)})
amp = max(float(np.abs(r.data).max()) for r in recs)
assert amp <= 1.0
groups = {r.group for r in recs}
print(f"[ok] 2000 clips x 5 s at 44.1 kHz, 50 classes x 40, {len(groups)} source recordings, max |x| {amp:.3f}")

for fs in (8000, 2000):
    a = get("esc50", target_fs=fs)
    w = make_windows(a, 128, dtype=np.float32)
    assert w.X.shape[1:] == (128, 1) and w.meta["fs"] == fs
    folds = group_kfold(w, n_folds=5)
    sizes = [len(te) for _, te in folds]
    flat = int((np.ptp(w.X[:, :, 0], axis=1) == 0).sum())   # digital-silence padding: nothing to reconstruct
    print(f"[ok] target_fs={fs}: {len(w.X)} windows of {1000 * 128 / fs:.0f} ms, dropped {w.meta['dropped']}, "
          f"constant (silent) windows {flat} ({100 * flat / len(w.X):.1f}%), source-grouped 5-fold test sizes {sizes}")
print("all checks passed")
