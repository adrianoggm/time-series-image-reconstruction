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

asym = np.array([[0.0, 1.0, 4.0], [3.0, 0.0, 2.0], [6.0, 2.0, 0.0]])
Ds = symmetrized_magnitude(asym)
assert np.isclose(Ds[0, 1], 2.0) and np.isclose(Ds[0, 2], 5.0), "symmetrized_magnitude must average |R_ij| and |R_ji|"

for src in ("direct", "geo", "jaccard"):
    dm = distance_matrix(R_obs, source=src, rho=0.60)
    assert dm.shape == (127, 127), (src, dm.shape)

Y, ev = embed_2d(distance_matrix(R_obs, source="direct"))
assert Y.shape == (127, 2)
assert evr2(ev) > 0.99, f"EVR2 for a true 2-D config should be ~1, got {evr2(ev):.4f}"
assert mds_stress(distance_matrix(R_obs, source="direct"), Y) < 1e-6, "direct MDS stress must be ~0"
print("[ok] Task 1: distance sources + embedding + geometry metrics")

# --- Task 2: overlap alignment + reconstruction (geometric core, all sources) ---
from src.rp_decoders import (
    align_states,
    orientation_candidates,
    reconstruct_signal_from_states,
)

Qs = orientation_candidates(n_theta=720)
assert len(Qs) == 1440
assert all(np.allclose(Q.T @ Q, np.eye(2), atol=1e-9) for Q in Qs), "all Q must be orthogonal"

for src in ("direct", "geo"):
    Y, _ = embed_2d(distance_matrix(R_obs, source=src))
    best = min((align_states(Y, Q) for Q in Qs), key=lambda ab: ab[2])
    a, b, e_ov = best
    xhat = reconstruct_signal_from_states(a, b)
    assert xhat.shape == (128,), (src, xhat.shape)
    r = _abs_corr(xhat, x)
    assert r > 0.99, f"[{src}] geometric-core corr too low: {r:.4f}"
    assert e_ov < 1e-3, f"[{src}] overlap error should be ~0 on clean input: {e_ov:.2e}"
print("[ok] Task 2: overlap alignment + reconstruction")

# --- Task 3: signed selection + unified single-channel decoder ---
from src.rp_decoders import decode_channel, signed_rp_error

# Build a clean signed RP (physical units) for one channel from a known signal.
from src.recurrence_plots import varRP2
xc = 2.0 * np.sin(t) - 0.5 * np.cos(2.0 * t)      # length 128
R_signed = varRP2(xc, TIME_STEPS=128)              # (127,127) signed, physical units
minmax_c = (float(xc.max()), float(xc.min()))

for src in ("direct", "geo"):
    rec = decode_channel(R_signed, minmax_c, source=src, dims=2,
                         recover="overlap", signed=True)
    assert rec.shape == (128,), (src, rec.shape)
    r = _abs_corr(rec, xc)
    assert r > 0.95, f"[{src}] signed 2-D decode corr too low: {r:.4f}"

rec_unsigned = decode_channel(R_signed, minmax_c, source="direct", dims=2,
                              recover="overlap", signed=False)
assert rec_unsigned.shape == (128,) and np.all(np.isfinite(rec_unsigned))

# signed error must be small for the true signal, large for a shuffled one.
e_true = signed_rp_error(R_signed, np.interp(xc, (xc.min(), xc.max()), (xc.min(), xc.max())))
e_bad = signed_rp_error(R_signed, np.random.default_rng(0).permutation(xc))
assert e_true < e_bad, f"signed error not discriminative: {e_true:.3f} !< {e_bad:.3f}"

# 1-D legacy path still reachable through the unified entry.
rec1 = decode_channel(R_signed, minmax_c, source="jaccard", dims=1,
                      recover="append", signed=True)
assert rec1.shape == (128,)
print("[ok] Task 3: signed selection + unified decoder")

# --- Task 4: metrics + 3-channel window decode ---
from src.rp_decoders import decode_window, overlap_error_abs
from src.recurrence_plots import NormalizeMatrix

sig = {0: x, 1: np.cos(t), 2: 0.5 * np.sin(2.0 * t)}
chans, maxs, mins = [], [], []
for c in range(3):
    xs = sig[c]
    Rn = NormalizeMatrix(varRP2(xs, TIME_STEPS=128))
    chans.append(np.round(Rn * 255).astype(np.uint8))
    maxs.append(float(xs.max())); mins.append(float(xs.min()))
img = np.stack(chans, axis=2)                     # (127,127,3)
minmax = [maxs, mins]

rec, metrics = decode_window(img, minmax, source="direct", dims=2,
                             recover="overlap", signed=True)
assert rec.shape == (128, 3), rec.shape
assert np.all(np.isfinite(rec))
assert set(metrics) == {"e_rp", "evr2", "stress"}
assert 0.0 <= metrics["evr2"] <= 1.0

ab = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 9.0])  # last b intentionally off
assert abs(overlap_error_abs(*ab) - (abs(2-2)+abs(3-3))/2) < 1e-12
print("[ok] Task 4: metrics + 3-channel window decode")
print("all checks passed")
