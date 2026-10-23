"""Gate for src/XIRP.py.

    python -m scripts._check_xirp
"""
import numpy as np

from scripts._check_baseline import minmax, synthetic_windows
from src.XIRP import decode_window, encode_window, from_positive, invert, to_positive, xirp

rng = np.random.default_rng(0)

# The paper's definition and its two inversions are exact in floating point.
x = 1.0 + rng.random(128)
R = xirp(x)
assert np.allclose(np.diag(R), x) and np.isclose(R[3, 7], np.log(x[3] / x[7]))
for method in ("diag", "im"):
    err = np.abs(invert(R, method) - x).max()
    assert err < 1e-12, (method, err)
print("[ok] float round trip exact (diag and inversion-by-mean)")

# Scale invariance claimed by the paper: the off-diagonal ignores a positive rescaling.
off = ~np.eye(128, dtype=bool)
assert np.allclose(xirp(3.7 * x)[off], R[off])
try:
    xirp(x - 1.5)
    raise SystemExit("xirp accepted a non-positive series")
except ValueError:
    pass
print("[ok] off-diagonal scale invariant; non-positive input rejected")

# Our positivity shift is an exact affine bijection.
w = rng.standard_normal(128)
assert np.allclose(from_positive(to_positive(w, w.min(), w.max()), w.min(), w.max()), w)

# 8-bit storage: metadata-free decode on the M0 synthetic windows.
for k, win in enumerate(synthetic_windows()):
    mm = minmax(win)
    img = encode_window(win, mm)
    assert img.dtype == np.uint8 and img.shape == (128, 128, 3)
    rng_ax = win.max(0) - win.min(0)
    res = {m: float((np.abs(decode_window(img, mm, m) - win).mean(0) / rng_ax).mean()) for m in ("diag", "im")}
    print(f"[ok] window {k}: 8-bit NMAE diag={res['diag']:.5f}  im={res['im']:.5f}")
    assert max(res.values()) < 0.01, res
print("all checks passed")
