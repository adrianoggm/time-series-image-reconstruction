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
    source="jaccard_legacy": the faithful previous decoder's geodesics (src/rp_legacy.py)
    """
    if source == "direct":
        return symmetrized_magnitude(R_obs)
    if source == "geo":
        return _continuous_geodesic(symmetrized_magnitude(R_obs), rho)
    if source == "jaccard":
        # weigthed_graphRP expects the signed RP and applies its own |.| + threshold.
        return calculate_shortest_path_matrix(weigthed_graphRP(R_obs, 100.0 * rho))
    if source == "binary_geodesic":
        # A conventional binary RP has no distances; estimate them with Hirata's graph geodesics so the
        # 2-D decoder can run on it (src/rp_hirata.py). R_obs is the binary matrix itself.
        from src.rp_hirata import geodesics

        return geodesics(R_obs)[0]
    if source == "jaccard_legacy":
        # Faithful previous decoder (src/rp_legacy.py); always its own rho = 0.75, never this rho.
        from src.rp_legacy import geodesic_distances

        return geodesic_distances(R_obs)[0]
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


def closed_form_orientation_candidates(Y):
    """Return the four overlap-optimal transforms without an angular grid.

    For a fixed reflection parity ``eta``, the translation-corrected overlap
    residual has the form ``C @ [cos(theta), sin(theta)]``. Its squared norm is
    the quadratic form of the 2x2 matrix ``C.T @ C`` on the unit circle; the
    eigenvector for the smallest eigenvalue gives the globally optimal angle
    for this unnormalised least-squares objective. Both reflection
    parities and both global signs are retained for the signed-RP comparison.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"Y must have shape (n_states, 2), got {Y.shape}")

    Qs = []
    for eta in (1.0, -1.0):
        # Before translation correction,
        # d_i = c*(eta*y_i2 - y_{i+1,1})
        #     + s*(-eta*y_i1 - y_{i+1,2}).
        # Centring both coefficients is exactly delta=-mean(d).
        A = eta * Y[:-1, 1] - Y[1:, 0]
        B = -eta * Y[:-1, 0] - Y[1:, 1]
        C = np.column_stack((A, B))
        if C.size:
            C = C - np.mean(C, axis=0, keepdims=True)
            H = C.T @ C
        else:
            H = np.zeros((2, 2), dtype=float)
        H = 0.5 * (H + H.T)
        _, eigvecs = np.linalg.eigh(H)
        u = eigvecs[:, 0]
        for sign in (1.0, -1.0):
            c, s = sign * u
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


def signed_rp_error(R_obs, x_scaled, scale_invariant=False):
    """Normalised Frobenius distance between the observed signed RP and the
    signed RP regenerated from a candidate (physical-unit) reconstruction.

    ``scale_invariant`` normalises both matrices to unit Frobenius norm before
    comparing, so the orientation choice does not depend on the absolute scale
    of ``R_obs``. The metadata-free (diagonal-centred) mode needs this because
    there ``R_obs`` is known only up to a positive multiplicative constant.
    """
    Rhat = varRP2(np.asarray(x_scaled, dtype=float), TIME_STEPS=len(x_scaled))
    if scale_invariant:
        R_obs = R_obs / (float(np.linalg.norm(R_obs)) + EPS)
        Rhat = Rhat / (float(np.linalg.norm(Rhat)) + EPS)
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
                   signed=True, rho=0.60, n_theta=720, top_k=8,
                   scale_invariant=False, orientation_method="closed_form"):
    """Unified single-channel RP decoder. Returns a length-N channel in
    physical units, axis-compatible with recurrence_plots.Reconstruct_RP output.

    dims=1, recover="append": legacy 1-D projection + fix_rotationscale (baseline
      and RP-GEO-1D). dims=2, recover="overlap": 2-D O(2) alignment + overlap
      reconstruction; signed=True adds the signed-RP orientation selection.

    The output has N = M + 1 samples for an (M, M) matrix, so any window length works.
    """
    n_out = np.asarray(R_obs).shape[0] + 1
    # The decoder is scale-equivariant (the output is rescaled to the channel range), but EPS and the
    # flatness checks below are absolute. Bring matrices of tiny magnitude (e.g. a dead seismometer
    # channel spanning 1e-12) to O(1) first; ordinary inputs never take this branch and are unchanged.
    scale = float(np.max(np.abs(R_obs)))
    if 0.0 < scale < 1e-6:
        R_obs = np.asarray(R_obs, dtype=float) / scale
    if source == "jaccard_legacy":
        from src.rp_legacy import decode_channel as legacy_decode

        return legacy_decode(R_obs, minmax_c, signed=signed, scale_invariant=scale_invariant, n_out=n_out)[0]
    if (dims, recover) not in {(1, "append"), (2, "overlap")}:
        raise ValueError(f"unsupported (dims, recover) combo: ({dims}, {recover!r})")
    dmat = distance_matrix(R_obs, source=source, rho=rho)

    if dims == 1:
        # Reuse the legacy 1-D machinery: project to the max-variance MDS axis,
        # then mean-append + (optional) signed flip. reconstruct_time_series does
        # its own double-centring on the passed matrix, so hand it `dmat`.
        proj = reconstruct_time_series(dmat, ep=0.0)          # (M,1)
        rec, _ = fix_rotationscale(R_obs, proj, 0, [ [minmax_c[0]], [minmax_c[1]] ],
                                   TIME_STEPS=n_out,
                                   use_orientation=signed, append_value="mean")
        return np.asarray(rec)[:n_out]

    # 2-D path.
    Y, _ = embed_2d(dmat)
    if orientation_method == "closed_form":
        Qs = closed_form_orientation_candidates(Y)
    elif orientation_method == "grid":
        Qs = orientation_candidates(n_theta=n_theta)
    else:
        raise ValueError(
            "orientation_method must be 'closed_form' or 'grid', "
            f"got {orientation_method!r}"
        )
    cands = []
    for Q in Qs:
        a, b, e_ov = align_states(Y, Q)
        cands.append((e_ov, a, b))
    cands.sort(key=lambda c: c[0])

    if not signed:
        _, a, b = cands[0]
        x = reconstruct_signal_from_states(a, b)
        return _rescale(x, minmax_c)[:n_out]

    # Signed selection (lambda-free lexicographic): among the top_k lowest
    # overlap-error orientations, pick the one whose regenerated signed RP is
    # closest to the observed one. The closed-form path has only four
    # candidates; ``top_k`` remains useful for the optional grid ablation.
    # With scale_invariant the candidates are compared on the unit range and the calibration is applied only
    # to the winner, so the choice depends on the image alone (an exact tie, when no pair is sign-informative,
    # is no longer broken by the rounding of the physical-unit rescale).
    ref = (1.0, 0.0) if scale_invariant else minmax_c
    best = None
    for e_ov, a, b in cands[:top_k]:
        x = _rescale(reconstruct_signal_from_states(a, b), ref)[:n_out]
        e_sig = signed_rp_error(R_obs, x, scale_invariant=scale_invariant)
        if best is None or e_sig < best[0]:
            best = (e_sig, x)
    return _rescale(best[1], minmax_c) if scale_invariant else best[1]


def overlap_error_abs(a, b):
    """L1 overlap diagnostic: mean|b_i - a_{i+1}| over shared samples."""
    if a.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(b[:-1] - a[1:])))


def decode_window(img_rgb, minmax, source, dims=2, recover="overlap",
                  signed=True, rho=0.60, rp_range=None,
                  orientation_method="closed_form", with_metrics=True):
    """Decode a 3-channel RP image to (TIME_STEPS,3) plus geometry metrics.

    Metrics (averaged over channels): e_rp (signed-RP reproduction error),
    evr2 (fraction of geometry in 2 dims), stress (MDS stress-1).

    ``rp_range``: per-channel PNG inversion mode.
      * list of ``(R_min, R_max)``: invert the PNG back to the RP's OWN distance
        range (persisted metadata).
      * ``"diagonal"``: metadata-free. The RP diagonal is identically zero, so
        the constant diagonal pixel level marks where ``R=0`` sits in ``[0,255]``;
        centring on it (``R = png - median(diag(png))``) recovers the signed
        matrix up to a positive scale, which classical MDS and the final range
        rescale absorb. No per-window ``(R_min,R_max)`` is used. Orientation
        selection switches to the scale-invariant signed error.
      * ``None``: inverted with the signal amplitude range ``(mins, maxs)`` — the
        legacy convention, which affinely distorts the distances handed to MDS
        (see ``scripts/_check_rp_scale.py``).
    The OUTPUT is always rescaled to the signal range, since the reconstruction
    is an amplitude.

    ``with_metrics=False`` skips the diagnostics (a second distance matrix and
    MDS per channel); the reconstruction is the same and the metrics are NaN.
    """
    maxs, mins = minmax
    diagonal = isinstance(rp_range, str) and rp_range == "diagonal"
    chans, e_rp, evrs, stresses = [], [], [], []
    for c in range(3):
        png_c = img_rgb[:, :, c].astype(float)
        if diagonal:
            R_obs = png_c - float(np.median(np.diag(png_c)))
        else:
            lo, hi = (mins[c], maxs[c]) if rp_range is None else (rp_range[c][0], rp_range[c][1])
            R_obs = np.interp(png_c, (0, 255), (lo, hi))
        rec = decode_channel(R_obs, (maxs[c], mins[c]), source=source, dims=dims,
                             recover=recover, signed=signed, rho=rho,
                             scale_invariant=diagonal,
                             orientation_method=orientation_method)
        chans.append(rec)
        if not with_metrics:
            continue
        e_rp.append(signed_rp_error(R_obs, rec, scale_invariant=diagonal))
        dmat = distance_matrix(R_obs, source=source, rho=rho)
        Y, pos = embed_2d(dmat)
        evrs.append(evr2(pos))
        stresses.append(mds_stress(dmat, Y))
    rec = np.stack(chans, axis=1)                       # (TIME_STEPS, 3)
    metrics = dict(e_rp=float(np.mean(e_rp)) if e_rp else np.nan,
                   evr2=float(np.mean(evrs)) if evrs else np.nan,
                   stress=float(np.mean(stresses)) if stresses else np.nan)
    return rec, metrics


def _demo():
    """Self-check: metadata-free diagonal centring reproduces the metadata
    decoder to within the 8-bit quantisation floor on a synthetic window."""
    from src.GAF import normalize_to_zero_one
    t = np.linspace(0, 4 * np.pi, TIME_STEPS)
    win = np.stack([np.sin(t), np.cos(0.7 * t), np.sin(t) - np.cos(t)], axis=1)
    imgs, ranges = [], []
    for c in range(3):
        M = varRP2(win[:, c].astype(float), TIME_STEPS=TIME_STEPS)
        imgs.append(np.rint(np.clip(normalize_to_zero_one(M) * 255, 0, 255)))
        ranges.append((float(M.min()), float(M.max())))
    img = np.stack(imgs, axis=-1)
    mm = ([float(win[:, c].max()) for c in range(3)],
          [float(win[:, c].min()) for c in range(3)])
    rec_meta, _ = decode_window(img, mm, source="direct", rp_range=ranges)
    rec_free, _ = decode_window(img, mm, source="direct", rp_range="diagonal")
    rng = win.max(0) - win.min(0)
    nmae = float((np.abs(rec_meta - rec_free).mean(0) / rng).mean())
    assert nmae < 0.02, f"diagonal vs metadata NMAE too large: {nmae}"
    print(f"[rp_decoders] diagonal vs metadata agreement NMAE={nmae:.4f} (ok)")


if __name__ == "__main__":
    _demo()
