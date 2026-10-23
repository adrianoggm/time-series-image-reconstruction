"""Gate: the vectorised varRP2 is bit-identical to the original double loop.

    python -m scripts._check_varrp2
"""
import time

import numpy as np

from scripts._check_baseline import synthetic_windows
from src.recurrence_plots import _varRP2_loop, varRP2

rng = np.random.default_rng(0)
cases = [rng.standard_normal(128), np.cumsum(rng.standard_normal(128)), np.zeros(128), np.ones(16) * 3.0,
         np.round(rng.standard_normal(128), 1), rng.integers(-3, 4, 64).astype(float), 1e-12 * rng.standard_normal(32)]
cases += [w[:, c] for w in synthetic_windows() for c in range(3)]
for i, x in enumerate(cases):
    a, b = _varRP2_loop(x, TIME_STEPS=len(x)), varRP2(x, TIME_STEPS=len(x))
    assert a.shape == b.shape and np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b)), i
print(f"[ok] bit-identical on {len(cases)} series (values and signs of zero)")

x = cases[1]
t0 = time.perf_counter()
_varRP2_loop(x)
t1 = time.perf_counter()
for _ in range(20):
    varRP2(x)
t2 = time.perf_counter()
print(f"[info] T=128: loop {1000 * (t1 - t0):.1f} ms, vectorised {1000 * (t2 - t1) / 20:.2f} ms")
print("all checks passed")
