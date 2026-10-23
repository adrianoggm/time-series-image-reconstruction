"""Gate for the classical Hirata (RPR) reconstruction (src/rp_hirata.py, roadmap #64).

    python -m scripts._check_rp_hirata
"""
import numpy as np

from scripts._check_baseline import synthetic_windows
from src.recurrence_plots import varRP2
from src.rp_decoders import decode_channel
from src.rp_hirata import binary_rp, decode_series, eps_for_rate, state_distances, states
from src.rp_legacy import decode_channel as legacy_channel

rng = np.random.default_rng(0)
x = np.cumsum(rng.standard_normal(128))
D = state_distances(x)
M = len(D)

# Eq. (1): strict threshold at the requested recurrence rate.
eps = eps_for_rate(D, 0.75)
B = binary_rp(D, eps)
rate = B[np.triu_indices(M, 1)].mean()
assert abs(rate - 0.75) < 0.01 and B.trace() == M, rate
print(f"[ok] binary RP: eps={eps:.4f}, off-diagonal recurrence rate {rate:.4f}")

# Eq. (2) and the shortest paths, against a reference loop + Floyd-Warshall.
W = np.full((M, M), np.inf)
np.fill_diagonal(W, 0.0)
for i, j in zip(*np.nonzero(B), strict=True):
    if i != j:
        Gi, Gj = np.nonzero(B[i])[0], np.nonzero(B[j])[0]
        W[i, j] = 1 - len(np.intersect1d(Gi, Gj)) / len(np.union1d(Gi, Gj))
for k in range(M):
    W = np.minimum(W, W[:, [k]] + W[[k], :])
Y, unreachable = states(B)
Jc = np.eye(M) - 1.0 / M
Wf = np.where(np.isfinite(W), W, 0.0)
vals, vecs = np.linalg.eigh(-0.5 * Jc @ np.square(Wf) @ Jc)
assert np.allclose(np.abs(Y[:, 0]), np.abs(vecs[:, -1] * np.sqrt(vals[-1])), atol=1e-8)
print(f"[ok] Eq. (2) graph + shortest paths + classical MDS match the reference (unreachable {unreachable})")

# The method recovers the state geometry up to isometry: the leading coordinate tracks x (up to sign).
corr = abs(np.corrcoef(Y[:, 0], x[:-1])[0, 1])
print(f"[info] |corr(leading MDS coordinate, x)| = {corr:.3f}")

print("float NMAE   Hirata(as is)  Hirata(oracle)  legacy(signed)  direct2ds(signed)")
for k, win in enumerate(synthetic_windows()):
    rng_ax = win.max(0) - win.min(0)
    cols = {"h": [], "ho": [], "l": [], "d": []}
    for c in range(3):
        xc = win[:, c]
        R = varRP2(xc, TIME_STEPS=128)
        mm = (xc.max(), xc.min())
        cols["h"].append(decode_series(xc, orientation="as_is")[0])
        cols["ho"].append(decode_series(xc, orientation="oracle")[0])
        cols["l"].append(legacy_channel(R, mm, scale_invariant=False)[0])
        cols["d"].append(decode_channel(R, mm, source="direct"))
    nm = {key: float((np.abs(np.stack(v, 1) - win).mean(0) / rng_ax).mean()) for key, v in cols.items()}
    assert all(np.isfinite(list(nm.values())))
    print(f"  window {k}:  {nm['h']:.4f}         {nm['ho']:.4f}          {nm['l']:.4f}          {nm['d']:.2e}")
print("all checks passed")
