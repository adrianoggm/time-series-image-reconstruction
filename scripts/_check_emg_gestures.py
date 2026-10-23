"""Gate + data report for the UCI EMG Data for Gestures adapter.

    python -m scripts._check_emg_gestures
"""
from collections import Counter

import numpy as np

from src.datasets import get, group_kfold, make_windows

a = get("emg_gestures")
recs = list(a.recordings())
assert all(r.data.shape[1] == 8 for r in recs)
subjects = {r.group for r in recs}
assert len(subjects) == 36, len(subjects)
per_class = Counter(int(r.labels[0]) for r in recs)
assert set(per_class) == set(range(6)), per_class
dur = np.array([len(r.data) / a.spec.fs for r in recs])
print(f"[ok] {len(recs)} gesture runs from {len(subjects)} subjects, runs per class "
      f"{dict((a.spec.classes[k], v) for k, v in sorted(per_class.items()))}, run length median {np.median(dur):.2f} s")

for length in (128, 64):
    w = make_windows(a, length, dtype=np.float32)
    for by in ("group", "trial"):
        folds = group_kfold(w, n_folds=3, by=by)
        print(f"[ok] {length}-sample windows ({length / a.spec.fs:.2f} s): {len(w.X)} windows, dropped {w.meta['dropped']}, "
              f"{by}-out test folds {[len(te) for _, te in folds]}")
palm = get("emg_gestures", include_palm=True)
assert len(palm.spec.classes) == 7
print("all checks passed")
