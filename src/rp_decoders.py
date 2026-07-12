"""Two-dimensional signed-RP decoder family and reconstruction metrics.

All variants share this module. They differ only in the distance matrix handed
to MDS (``distance_matrix(source=...)``) and in the recovery mode (1-D
projection + mean-append, legacy; or 2-D O(2) overlap alignment). The current
pipeline decoder (recurrence_plots.reconstruct_time_series + fix_rotationscale)
is unaffected and remains the default; this module is opt-in and evaluated by
scripts/_ablation_rp_2d.py.

State convention: s_i = (x_i, x_{i+1}); RP is (N-1)x(N-1); reconstruction -> N.
"""
import numpy as np

from src.recurrence_plots import (
    calculate_shortest_path_matrix,
    weigthed_graphRP,
)

EPS = 1e-10


def symmetrized_magnitude(R_obs):
    """(M,M) signed RP -> symmetric magnitude matrix with zero diagonal.

    PNG quantisation can make R_ij and R_ji differ slightly; averaging the two
    magnitudes restores the symmetry the encoder guarantees in principle.
    """
    A = np.abs(np.asarray(R_obs, dtype=float))
    D = 0.5 * (A + A.T)
    np.fill_diagonal(D, 0.0)
    return D


def _continuous_geodesic(D, rho):
    """Continuous-weighted graph geodesic: keep edges below the rho-quantile of
    the off-diagonal magnitudes, weight each retained edge by its magnitude
    D_ij, then take all-pairs shortest paths. Mirrors the Jaccard path's
    graph->shortest-path structure but preserves the observed distances."""
    iu = np.triu_indices_from(D, k=1)
    tau = np.quantile(D[iu], rho)
    W = np.where(D <= tau, D, 0.0)   # 0 = no edge (dijkstra treats 0 as absent for sparse)
    np.fill_diagonal(W, 0.0)
    return calculate_shortest_path_matrix(W)


def distance_matrix(R_obs, source, rho=0.60):
    """Build the (M,M) matrix fed to MDS.

    source="direct" : symmetric magnitudes (== L2 state distances, up to noise)
    source="geo"    : continuous-weighted graph geodesic distances
    source="jaccard": legacy Jaccard-dissimilarity graph geodesic (baseline)
    """
    if source == "direct":
        return symmetrized_magnitude(R_obs)
    if source == "geo":
        return _continuous_geodesic(symmetrized_magnitude(R_obs), rho)
    if source == "jaccard":
        # weigthed_graphRP expects the signed RP and applies its own |.| + threshold.
        return calculate_shortest_path_matrix(weigthed_graphRP(R_obs, 100.0 * rho))
    raise ValueError(f"unknown source: {source!r}")


def embed_2d(dmat):
    """Classical MDS to 2-D. Returns (Y (M,2), positive eigenvalues descending)."""
    n = dmat.shape[0]
    dmat = np.where(np.isfinite(dmat), dmat, 0.0)  # ponytail: disconnected geo graphs yield inf geodesics; replace with 0 (matches reconstruct_time_series) to avoid NaN propagation
    D2 = np.square(dmat)
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ D2 @ J
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    pos = eigvals[eigvals > 0]
    L = np.clip(eigvals[:2], a_min=0.0, a_max=None)
    Y = eigvecs[:, :2] * np.sqrt(L)
    return Y, pos


def evr2(pos_eigvals):
    """Fraction of geometry explained by the two leading positive dimensions."""
    s = float(np.sum(pos_eigvals))
    if s <= 0:
        return 0.0
    return float(np.sum(pos_eigvals[:2]) / s)


def mds_stress(dmat, Y):
    """Kruskal stress-1 between input distances and embedded distances."""
    iu = np.triu_indices_from(dmat, k=1)
    d = dmat[iu]
    dh = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)[iu]
    denom = float(np.sum(d ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(np.sum((d - dh) ** 2) / denom))
