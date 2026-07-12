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
    fix_rotationscale,
    reconstruct_time_series,
    varRP2,
    weigthed_graphRP,
)

EPS = 1e-10
TIME_STEPS = 128


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


def orientation_candidates(n_theta=720):
    """The 2*n_theta matrices Q_{θ,η} = rot(θ)·diag(1,η), η∈{+1,-1}, spanning O(2)."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    Qs = []
    for eta in (1.0, -1.0):
        for th in thetas:
            c, s = np.cos(th), np.sin(th)
            Qs.append(np.array([[c, -s * eta], [s, c * eta]]))
    return Qs


def align_states(Y, Q):
    """Apply Q, correct the MDS translation offset along the overlap, and return
    aligned state coordinates (a, b) plus the normalised overlap error.

    a_i approximates x_i, b_i approximates x_{i+1}; the overlap constraint is
    b_i ≈ a_{i+1}. δ is the mean offset between the two, split symmetrically.
    """
    Z = Y @ Q                                   # (M, 2)
    d = Z[:-1, 1] - Z[1:, 0]                     # b_i - a_{i+1}, i=0..M-2
    delta = -float(np.mean(d)) if d.size else 0.0
    a = Z[:, 0] - 0.5 * delta
    b = Z[:, 1] + 0.5 * delta
    num = float(np.sum((b[:-1] - a[1:]) ** 2))
    den = float(np.sum(a ** 2 + b ** 2)) + EPS
    return a, b, num / den


def reconstruct_signal_from_states(a, b):
    """Overlap reconstruction: x1=a1; x_i=(b_{i-1}+a_i)/2; xN=b_{M}. Length M+1."""
    M = a.shape[0]
    x = np.empty(M + 1)
    x[0] = a[0]
    x[1:M] = 0.5 * (b[:-1] + a[1:])
    x[M] = b[M - 1]
    return x


def signed_rp_error(R_obs, x_scaled):
    """Normalised Frobenius distance between the observed signed RP and the
    signed RP regenerated from a candidate (physical-unit) reconstruction."""
    Rhat = varRP2(np.asarray(x_scaled, dtype=float), TIME_STEPS=len(x_scaled))
    num = float(np.linalg.norm(R_obs - Rhat))
    den = float(np.linalg.norm(R_obs)) + EPS
    return num / den


def _rescale(x, minmax_c):
    """Map a unit-free reconstruction onto the channel's physical [min,max]."""
    mx, mn = minmax_c
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < EPS:
        return np.full_like(x, 0.5 * (mn + mx))
    return np.interp(x, (lo, hi), (mn, mx))


def decode_channel(R_obs, minmax_c, source, dims=2, recover="overlap",
                   signed=True, rho=0.60, n_theta=720, top_k=8):
    """Unified single-channel RP decoder. Returns a length-TIME_STEPS channel in
    physical units, axis-compatible with recurrence_plots.Reconstruct_RP output.

    dims=1, recover="append": legacy 1-D projection + fix_rotationscale (baseline
      and RP-GEO-1D). dims=2, recover="overlap": 2-D O(2) alignment + overlap
      reconstruction; signed=True adds the signed-RP orientation selection.
    """
    dmat = distance_matrix(R_obs, source=source, rho=rho)

    if dims == 1:
        # Reuse the legacy 1-D machinery: project to the max-variance MDS axis,
        # then mean-append + (optional) signed flip. reconstruct_time_series does
        # its own double-centring on the passed matrix, so hand it `dmat`.
        proj = reconstruct_time_series(dmat, ep=0.0)          # (M,1)
        rec, _ = fix_rotationscale(R_obs, proj, 0, [ [minmax_c[0]], [minmax_c[1]] ],
                                   TIME_STEPS=TIME_STEPS,
                                   use_orientation=signed, append_value="mean")
        return np.asarray(rec)[:TIME_STEPS]

    # 2-D path.
    Y, _ = embed_2d(dmat)
    cands = []
    for Q in orientation_candidates(n_theta=n_theta):
        a, b, e_ov = align_states(Y, Q)
        cands.append((e_ov, a, b))
    cands.sort(key=lambda c: c[0])

    if not signed:
        _, a, b = cands[0]
        x = reconstruct_signal_from_states(a, b)
        return _rescale(x, minmax_c)[:TIME_STEPS]

    # Signed selection (lambda-free lexicographic): among the top_k lowest
    # overlap-error orientations, pick the one whose regenerated signed RP is
    # closest to the observed one.
    # ponytail: top_k prefilter caps the O(n_theta) signed-RP regenerations at
    # top_k per channel; widen top_k only if the winner is unstable across folds.
    best = None
    for e_ov, a, b in cands[:top_k]:
        x = _rescale(reconstruct_signal_from_states(a, b), minmax_c)[:TIME_STEPS]
        e_sig = signed_rp_error(R_obs, x)
        if best is None or e_sig < best[0]:
            best = (e_sig, x)
    return best[1]
