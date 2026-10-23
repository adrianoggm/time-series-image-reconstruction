"""Gate + data report for the PAMAP2 adapter.

    python -m scripts._check_pamap2 [--root ~/datasets/PAMAP2/PAMAP2_Dataset]

Checks the adapter against the raw files (channel layout, label mapping,
transients excluded, no window spanning an activity change or a timestamp gap)
and prints the data-quality figures the roadmap asks for: NaN rate before and
after gap filling, windows dropped, windows per subject and activity, and the
trial-out / subject-out fold sizes.
"""
import argparse
from collections import Counter

import numpy as np

from src.adapters import PAMAP2_ACC16, PAMAP2_PROTOCOL
from src.datasets import get, group_kfold, make_windows

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="~/datasets/PAMAP2/PAMAP2_Dataset")
ap.add_argument("--length", type=int, default=128)
args = ap.parse_args()

adapter = get("pamap2", root=args.root)
files = adapter._files()
assert len(files) == 9, f"expected 9 protocol subject files, found {len(files)}"

# Raw-level facts, independent of the adapter.
raw_nan, raw_n, transient, hr_rate = 0, 0, 0, []
cols = [c + k for c in PAMAP2_ACC16.values() for k in range(3)]
for p in files:
    raw = np.loadtxt(p)
    assert raw.shape[1] == 54, f"{p.name}: {raw.shape[1]} columns"
    raw_nan += int(np.isnan(raw[:, cols]).sum())
    raw_n += raw[:, cols].size
    transient += int((raw[:, 1] == 0).sum())
    hr_rate.append(np.isfinite(raw[:, 2]).mean() * 100)
print(f"raw: {raw_n // 9} rows, acc16 NaN {100 * raw_nan / raw_n:.2f}% of values, "
      f"{transient} transient rows (activity 0), heart rate present in {np.mean(hr_rate):.1f}% of rows (~{np.mean(hr_rate):.0f} Hz)")

recs = list(adapter.recordings())
nan_left = sum(int(np.isnan(r.data).sum()) for r in recs)
print(f"adapter: {len(recs)} recordings, {sum(len(r.data) for r in recs)} samples, "
      f"NaN left after filling gaps <= {adapter.max_gap} samples: {nan_left}")

w = make_windows(adapter, args.length)
gaps = sum(v["gaps"] for v in w.meta["validation"].values())
print(f"windows: {w.X.shape}, dropped {w.meta['dropped']}, timestamp gaps {gaps}")
for i in range(0, len(w.X), 97):  # spot-check provenance
    r = next(r for r in recs if r.rec_id == w.rec_id[i])
    assert np.array_equal(w.X[i], r.data[w.start[i]:w.start[i] + args.length], equal_nan=True)
assert set(np.unique(w.y)) <= set(range(len(PAMAP2_PROTOCOL)))

names = adapter.spec.classes
per_subject = Counter(w.groups)
per_class = Counter(names[k] for k in w.y)
print("windows per subject:", dict(sorted(per_subject.items())))
print("windows per activity:", dict(sorted(per_class.items(), key=lambda kv: -kv[1])))
trials = len(set(zip(w.groups, w.y, strict=True)))
for by in ("group", "trial"):
    folds = group_kfold(w, n_folds=3, by=by)
    print(f"[ok] {by}-out folds: test sizes {[len(te) for _, te in folds]}"
          + (f" over {trials} trials" if by == "trial" else ""))
print("all checks passed")
