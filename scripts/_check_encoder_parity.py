"""Gate: the vectorised encoder helpers are bit-identical to the original per-pixel loops.

Covers varRP, NormalizeMatrix, the GASF/GADF tabulate kernels and the RGB stacking,
on synthetic windows and on real WISDM and PAMAP2 windows.

    python -m scripts._check_encoder_parity
"""
import math
import time

import numpy as np

import src.GADF as gadf
import src.GAF as gaf
import src.MTF as mtf
import src.recurrence_plots as rec
from scripts._check_baseline import synthetic_windows


# ---- the original implementations, verbatim in behaviour --------------------------------------
def ref_normalize(_r):
    dimR = _r.shape[0]
    _max = max(max(_r[i]) for i in range(dimR))
    _min = min(min([v for v in _r[i] if v != 0.0]) for i in range(dimR))
    out = np.zeros((dimR, dimR))
    for i in range(dimR):
        for j in range(dimR):
            out[i][j] = (_r[i][j] - _min) / (_max - _min)
    return out


def ref_tabulate(x, y, f):
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))


def ref_rgb(X, Y, Z):
    n = X.shape[0]
    img = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            img[i][j] = [X[i][j], Y[i][j], Z[i][j]]
    return img


def same(a, b):
    return a.shape == b.shape and np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))


windows = list(synthetic_windows())
try:
    from src.datasets import get, make_windows

    for name in ("wisdm", "pamap2"):
        w = make_windows(get(name), 128)
        for i in np.random.default_rng(0).choice(len(w.X), 6, replace=False):
            windows.append(w.X[i][:, :3].astype(float))
except Exception as exc:  # datasets not available on this machine: synthetic windows only
    print(f"[warn] real windows skipped: {exc}")

for k, win in enumerate(windows):
    for c, dim in enumerate("xyz"):
        R = rec.varRP(win, dim, 128)
        assert same(R, rec._varRP2_loop(win[:, c])), (k, dim)
        assert same(rec.NormalizeMatrix(R), ref_normalize(R)), (k, dim)
        phi = np.arccos(gaf.normalize_to_zero_one(win[:, c].reshape(1, -1)))
        for mod in (gaf, gadf):
            assert same(mod.tabulate(phi, phi, mod.cos_sum), ref_tabulate(phi, phi, lambda a, b: math.cos(a + b)))
            assert same(mod.tabulate(phi, phi, mod.sin_diff), ref_tabulate(phi, phi, lambda a, b: math.sin(a - b)))
    mats = [rec.varRP(win, d, 128) for d in "xyz"]
    for fn in (rec.RGBfromRPMatrix_of_XYZ, gaf.RGBfromGAFMatrix_of_XYZ, gadf.RGBfromGADFMatrix_of_XYZ,
               mtf.RGBfromMTFMatrix_of_XYZ):
        assert same(fn(*mats), ref_rgb(*mats)), (k, fn.__name__)
print(f"[ok] bit-identical to the loop implementations on {len(windows)} windows x 3 axes")

win = windows[3]
for name, enc in (("RP", rec.SavevarRP_XYZ), ("GAF", gaf.SavevarGAF_XYZ), ("GADF", gadf.SavevarGADF_XYZ),
                  ("MTF", mtf.SavevarMTF_XYZ)):
    enc(win, 0, 0, "x", normalized=1, saveImage=False, TIME_STEPS=128)          # warm-up (pyts JIT)
    t0 = time.perf_counter()
    for _ in range(5):
        enc(win, 0, 0, "x", normalized=1, saveImage=False, TIME_STEPS=128)
    print(f"[info] {name:4} encode {1000 * (time.perf_counter() - t0) / 5:7.1f} ms per window")
print("all checks passed")
