"""Gate: WISDMAdapter + src/datasets.py reproduce the legacy WISDM windows and folds exactly.

    python -m scripts._check_wisdm_parity

Compares against ``numpies/train/windowed/{X,y,sj}_train.npy`` (written by
``src/activity_data.py``) and against the StratifiedGroupKFold folds the
pipeline builds from ``activity_data.fold_groups``.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.activity_data import fold_groups
from src.config import DEFAULT_DATA_DIR
from src.datasets import get, group_kfold, make_windows

base = f"{DEFAULT_DATA_DIR}numpies/train/windowed/"
X_old, y_old, sj_old = (np.load(f"{base}{k}_train.npy") for k in ("X", "y", "sj"))

w = make_windows(get("wisdm"), length=128)
print(f"adapter: {w.X.shape}, legacy: {X_old.shape}; dropped {w.meta['dropped']}")
assert w.X.shape == X_old.shape, "window count differs"
assert np.array_equal(w.X, X_old), f"max |dX| = {np.abs(w.X - X_old).max()}"
assert np.array_equal(w.y, y_old.ravel()) and np.array_equal(w.groups.astype(int), sj_old.ravel())
print("[ok] windows, labels and subjects identical to the legacy arrays")

for sampling, by in (("loso", "group"), ("loto", "trial")):
    legacy = list(StratifiedGroupKFold(n_splits=3).split(X_old, y_old.ravel(), groups=fold_groups(sj_old, y_old, sampling)))
    new = group_kfold(w, n_folds=3, by=by)
    for k, ((_, te_old), (_, te_new)) in enumerate(zip(legacy, new, strict=True)):
        assert np.array_equal(te_old, te_new), f"{sampling} fold {k} differs"
    print(f"[ok] {sampling} folds identical (test sizes {[len(te) for _, te in new]})")
print("all checks passed")
