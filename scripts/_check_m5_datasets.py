"""Gate + data report for the PTB-XL, Swarm and STEAD adapters.

    python -m scripts._check_m5_datasets [ptbxl,swarm,stead]
"""
import sys
from collections import Counter

import numpy as np

from src.datasets import get, group_kfold, make_windows

which = (sys.argv[1] if len(sys.argv) > 1 else "ptbxl,swarm,stead").split(",")


def report(name, adapter, length, by="group"):
    w = make_windows(adapter, length, dtype=np.float32)
    folds = group_kfold(w, n_folds=3, by=by)          # asserts no group straddles a fold
    flat = int((np.ptp(w.X, axis=1) == 0).all(axis=1).sum())
    print(f"[ok] {name}: X {w.X.shape} at {w.meta['fs']} Hz ({w.meta['duration_s']:.2f} s), "
          f"{len(set(w.groups))} groups, dropped {w.meta['dropped']}, all-flat windows {flat}, "
          f"test folds {[len(te) for _, te in folds]}")
    return w


if "ptbxl" in which:
    a = get("ptbxl", max_records=600)
    recs = list(a.recordings())
    assert all(r.data.shape == (1000, 12) for r in recs)
    amp = np.percentile(np.abs(np.concatenate([r.data for r in recs])), 99)
    assert 0.1 < amp < 10, f"99th percentile |ECG| {amp:.2f} mV is implausible"
    y = np.array([r.labels[0] for r in recs])
    print(f"[ok] ptbxl records: 600 x (1000, 12) mV, 99th pct |x| {amp:.2f} mV, label counts "
          f"{dict(zip(a.spec.classes, y.sum(0).tolist(), strict=True))}, unlabelled {int((y.sum(1) == 0).sum())}")
    hr = list(get("ptbxl", fs=500, max_records=3).recordings())
    assert all(r.data.shape == (5000, 12) for r in hr)
    w = report("ptbxl 100 Hz", a, 128)
    assert w.y.shape[1] == 5

if "swarm" in which:
    a = get("swarm")
    w = report("swarm", a, 128)
    assert (w.y == -1).all()
    gaps = sum(v["gaps"] for v in w.meta["validation"].values())
    nan = sum(v["nan"] for v in w.meta["validation"].values())
    print(f"     swarm: {len(w.meta['validation'])} satellite-days, flagged samples set to NaN {nan // 3}, gaps {gaps}, "
          f"|B| range {np.linalg.norm(w.X, axis=2).min():.0f}-{np.linalg.norm(w.X, axis=2).max():.0f} nT")
    report("swarm by satellite", get("swarm", group_by="satellite"), 128)

if "stead" in which:
    a = get("stead", n_per_class=300)
    recs = list(a.recordings())
    assert len(recs) == 600 and all(r.data.shape == (6000, 3) for r in recs)
    cats = Counter(int(r.labels[0]) for r in recs)
    events = len({r.group for r in recs if r.group.startswith("event:")})
    print(f"[ok] stead traces: {dict((a.spec.classes[k], v) for k, v in cats.items())}, {events} distinct events")
    report("stead", a, 128)
print("all checks passed")
