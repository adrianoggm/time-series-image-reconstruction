"""Self-check for the 2-D RP decoder family (src/rp_decoders.py).

Assert-based (no pytest). Run:
    python -m scripts._check_rp_decoders
"""
import numpy as np

from src.rp_decoders import (
    distance_matrix,
    embed_2d,
    evr2,
    mds_stress,
    symmetrized_magnitude,
)


def _abs_corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return abs(float(a @ b) / d) if d > 0 else 0.0


# --- Task 1: distance sources, embedding, geometry metrics ---
t = np.linspace(0.0, 4.0 * np.pi, 128)
x = np.sin(t) + 0.3 * np.sin(3.0 * t)
states = np.stack([x[:-1], x[1:]], axis=1)                       # (127, 2)
D_true = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=2)

# signed RP with the L2 magnitude and an arbitrary sign pattern (quantisation-free)
R_obs = D_true.copy()

D = symmetrized_magnitude(R_obs)
assert D.shape == (127, 127) and np.allclose(np.diag(D), 0.0)
assert np.allclose(D, D.T), "symmetrised magnitude must be symmetric"

for src in ("direct", "geo", "jaccard"):
    dm = distance_matrix(R_obs, source=src, rho=0.60)
    assert dm.shape == (127, 127), (src, dm.shape)

Y, ev = embed_2d(distance_matrix(R_obs, source="direct"))
assert Y.shape == (127, 2)
assert evr2(ev) > 0.99, f"EVR2 for a true 2-D config should be ~1, got {evr2(ev):.4f}"
assert mds_stress(distance_matrix(R_obs, source="direct"), Y) < 1e-6, "direct MDS stress must be ~0"
print("[ok] Task 1: distance sources + embedding + geometry metrics")
