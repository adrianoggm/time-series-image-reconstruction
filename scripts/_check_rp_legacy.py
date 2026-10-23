"""Gate for the faithful previous RP decoder (src/rp_legacy.py, roadmap #65).

    python -m scripts._check_rp_legacy
"""
import numpy as np

from scripts._check_baseline import minmax, synthetic_windows
from src.recurrence_plots import SavevarRP_XYZ, varRP2, weigthed_graphRP
from src.rp_decoders import decode_window, symmetrized_magnitude
from src.rp_legacy import RHO, decode_channel, geodesic_distances, jaccard_graph

rng = np.random.default_rng(0)
x = np.cumsum(rng.standard_normal(128))
R = varRP2(x, TIME_STEPS=128)
M = R.shape[0]

# Reference: the edge-only loop of the original code, with the manuscript's off-diagonal quantile.
D = symmetrized_magnitude(R)
C = (D <= np.quantile(D[np.triu_indices(M, 1)], RHO)).astype(int)
W_ref = np.full((M, M), np.inf)
np.fill_diagonal(W_ref, 0.0)
for i, j in zip(*np.nonzero(C), strict=True):
    if i != j:
        Gi, Gj = np.nonzero(C[i])[0], np.nonzero(C[j])[0]
        W_ref[i, j] = 1 - len(np.intersect1d(Gi, Gj)) / len(np.union1d(Gi, Gj))
G = jaccard_graph(R).toarray()
edges = np.isfinite(W_ref) & ~np.eye(M, dtype=bool)
assert np.allclose(G[edges], W_ref[edges]) and not G[~edges].any()
print(f"[ok] edge-only Jaccard graph matches the reference loop ({edges.sum()} edges, "
      f"{int((W_ref[edges] == 0).sum())} of weight 0)")

# Geodesics against an independent Floyd-Warshall that keeps zero-weight edges.
F = W_ref.copy()
for k in range(M):
    F = np.minimum(F, F[:, [k]] + F[[k], :])
SP, unreachable = geodesic_distances(R)
assert np.allclose(np.where(np.isfinite(F), F, 0.0), SP) and unreachable == int(np.isinf(F).sum())
print(f"[ok] Dijkstra geodesics match Floyd-Warshall (unreachable pairs: {unreachable})")

# The 2025 vectorised version weights every pair, so it is a different graph.
allpairs = weigthed_graphRP(R, 100 * RHO)
print(f"[info] vectorised 2025 graph: {(allpairs > 0).sum()} non-zero weights vs {edges.sum()} edges here")

# Deterministic (SMACOF is seeded) and finite.
rec1, diag1 = decode_channel(R, (x.max(), x.min()))
rec2, _ = decode_channel(R, (x.max(), x.min()))
assert np.array_equal(rec1, rec2) and np.isfinite(rec1).all() and len(rec1) == 128
print(f"[ok] deterministic, {len(rec1)} samples, diagnostics {diag1}")

# End-to-end on the M0 synthetic windows, same metadata-free input as the production decoder.
print("8-bit NMAE   jaccard_legacy  jaccard(2025)  direct2ds")
for k, win in enumerate(synthetic_windows()):
    mm = minmax(win)
    img = np.asarray(SavevarRP_XYZ(win, 0, k, "x", normalized=1, saveImage=False, TIME_STEPS=128))
    rng_ax = win.max(0) - win.min(0)

    def nmae(rec, win=win, rng_ax=rng_ax):
        return float((np.abs(rec - win).mean(0) / rng_ax).mean())

    legacy, _ = decode_window(img, mm, source="jaccard_legacy", dims=1, recover="append", rp_range="diagonal")
    old, _ = decode_window(img, mm, source="jaccard", dims=1, recover="append", rho=0.60, rp_range="diagonal")
    new, _ = decode_window(img, mm, source="direct", rp_range="diagonal")
    print(f"  window {k}:  {nmae(legacy):.4f}          {nmae(old):.4f}         {nmae(new):.4f}")
    assert np.isfinite(legacy).all()
print("all checks passed")
