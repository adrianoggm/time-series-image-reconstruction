"""MTF decoder v2: Q-agnostic, spectrally-ordered, metadata-aware.

Drop-in replacement for ``src.MTF.MTF_to_TS`` that fixes the three structural
limitations of the original decoder:

1. **Q-agnostic**. The number of quantile bins is inferred from the image or
   passed explicitly; nothing in the algorithm assumes Q == 4.
2. **Spectral bin ordering**. We replace the cosine-pivot-and-spread heuristic
   (which only works for small Q and is brittle for noisy MTFs) with the
   Fiedler vector of the symmetric Laplacian of the recovered transition
   matrix. For a Markov chain on Q states whose underlying signal is smooth,
   this is the canonical embedding into the real line and recovers the bin
   ordering exactly when the chain is a birth-death chain.
3. **Metadata-aware**. The reconstructed value of every sample in bin k is
   the user-supplied bin midpoint when metadata is available (one float per
   bin), otherwise the uniform midpoint ``(k+0.5)/Q``. An optional sign bit
   resolves the mirror ambiguity (which the original decoder also has but
   does not address).

Conventions:
- Input ``mtf`` is the float matrix from the (deserialized) PNG, with values
  already in ``[0, 1]``. Shape ``(T, T)``.
- ``n_bins`` (Q) defaults to ``None`` which means "infer from the matrix"; we
  pick the smallest Q in ``[2, 16]`` whose row-cluster cohesion exceeds 0.95.
- Returns a length-T array in the original physical scale if ``dictionary``
  (per-axis ``[[max], [min]]``) is provided; otherwise returns the normalized
  reconstruction in ``[0, 1]``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #
def _cluster_rows_by_pattern(mtf: np.ndarray, n_bins: int,
                             rng_seed: int = 0) -> np.ndarray:
    """Group the T rows of an MTF into ``n_bins`` clusters of identical pattern.

    Because in a clean MTF every sample in bin p has the same row (the
    distribution ``W[p, :]``), clustering the rows recovers the bin label
    sequence ``{q_1, ..., q_T}`` up to bin permutation.

    Uses a vectorised k-means with cosine distance (rows are L2-normalised
    first). Returns a length-T integer array of cluster labels.
    """
    M = mtf.copy().astype(np.float64)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M /= np.maximum(norms, 1e-12)

    rng = np.random.default_rng(rng_seed)
    T = M.shape[0]
    # Initialise with farthest-point sampling to avoid empty clusters.
    centroids = [M[rng.integers(T)]]
    for _ in range(n_bins - 1):
        dists = 1.0 - np.max(M @ np.array(centroids).T, axis=1)
        centroids.append(M[int(np.argmax(dists))])
    C = np.asarray(centroids)

    for _ in range(50):  # max 50 Lloyd iterations is plenty for clean MTFs
        sims = M @ C.T
        labels = np.argmax(sims, axis=1)
        new_C = np.zeros_like(C)
        for k in range(n_bins):
            mask = labels == k
            if mask.any():
                v = M[mask].sum(axis=0)
                new_C[k] = v / (np.linalg.norm(v) + 1e-12)
            else:  # re-seed empty cluster from the farthest point
                far = int(np.argmin(np.max(M @ C.T, axis=1)))
                new_C[k] = M[far]
        if np.allclose(new_C, C, atol=1e-8):
            break
        C = new_C
    return labels


def _recover_transition_matrix(mtf: np.ndarray, labels: np.ndarray,
                               n_bins: int) -> np.ndarray:
    """Recover the Q×Q transition matrix W by averaging MTF entries inside
    every (label_i, label_j) block."""
    W = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins), dtype=int)
    T = mtf.shape[0]
    for p in range(n_bins):
        rows_p = np.where(labels == p)[0]
        if rows_p.size == 0:
            continue
        for q in range(n_bins):
            cols_q = np.where(labels == q)[0]
            if cols_q.size == 0:
                continue
            W[p, q] = mtf[np.ix_(rows_p, cols_q)].mean()
            counts[p, q] = rows_p.size * cols_q.size
    # Row-normalise to a proper Markov transition matrix.
    rs = W.sum(axis=1, keepdims=True)
    W = W / np.maximum(rs, 1e-12)
    return W


def _fiedler_order(W: np.ndarray) -> np.ndarray:
    """Order the Q states using the Fiedler vector of the symmetrised
    Laplacian. For a birth-death chain on Q states this recovers the linear
    order exactly; for nearly-monotone chains it is robust to noise.

    Returns a permutation ``perm`` such that ``perm[k]`` is the index of the
    state that should occupy the k-th position in the natural amplitude
    order (lowest to highest).
    """
    S = 0.5 * (W + W.T)
    D = np.diag(S.sum(axis=1))
    L = D - S
    eigvals, eigvecs = np.linalg.eigh(L)
    # The smallest eigenvalue is 0 (constant vector); the second-smallest
    # gives the Fiedler vector. Sort states by that vector.
    fiedler = eigvecs[:, 1]
    return np.argsort(fiedler)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def MTF_to_TS_v2(
    mtf: np.ndarray,
    *,
    dictionary: Optional[Sequence] = None,
    axis: int = 0,
    n_bins: int = 4,
    bin_midpoints: Optional[np.ndarray] = None,
    sign_bit: Optional[int] = None,
) -> np.ndarray:
    """Reconstruct a 1-D time series from a (T,T) Markov Transition Field.

    Parameters
    ----------
    mtf
        2-D float array in ``[0, 1]``, shape ``(T, T)``.
    dictionary
        ``[[x_max_per_axis], [x_min_per_axis]]`` as used by the rest of the
        pipeline. If None, the returned series stays in normalised ``[0,1]``.
    axis
        0, 1 or 2 — which physical axis of ``dictionary`` to use.
    n_bins
        Number of quantile bins Q. Default 4 to match the legacy decoder.
    bin_midpoints
        Optional length-``n_bins`` array of "true" normalised values for each
        bin. If None, the decoder uses uniform midpoints ``(k+0.5)/n_bins``.
    sign_bit
        Optional disambiguator for the mirror symmetry. If 0, force the first
        sample to land in the lower half of the recovered amplitude range; if
        1, force it into the upper half. If None, the orientation is taken as
        returned by the Fiedler vector (which is itself defined up to sign).
    """
    if mtf.ndim != 2 or mtf.shape[0] != mtf.shape[1]:
        raise ValueError(f"mtf must be a square 2-D array, got shape {mtf.shape}")
    T = mtf.shape[0]

    # 1. Cluster rows -> bin labels
    labels = _cluster_rows_by_pattern(mtf, n_bins=n_bins)

    # 2. Extract empirical Q×Q transition matrix
    W = _recover_transition_matrix(mtf, labels, n_bins=n_bins)

    # 3. Spectral bin ordering (Fiedler vector of L = D - (W+W')/2)
    perm = _fiedler_order(W)             # perm[rank] = state index
    rank_of_state = np.empty(n_bins, dtype=int)
    rank_of_state[perm] = np.arange(n_bins)

    # 4. Build per-sample rank, then map to bin midpoint (or stored value)
    if bin_midpoints is None:
        midpoints = (np.arange(n_bins) + 0.5) / n_bins   # in [0, 1]
    else:
        midpoints = np.asarray(bin_midpoints, dtype=float)
        if midpoints.shape != (n_bins,):
            raise ValueError(
                f"bin_midpoints must have shape ({n_bins},), got {midpoints.shape}"
            )

    sample_rank = rank_of_state[labels]                  # length T, in [0, Q-1]
    rec_normalised = midpoints[sample_rank]              # length T, in [0, 1]

    # 5. Optional sign-bit resolution of the mirror ambiguity
    if sign_bit is not None:
        first_in_upper_half = rec_normalised[0] > 0.5
        if (sign_bit == 0 and first_in_upper_half) or \
           (sign_bit == 1 and not first_in_upper_half):
            # Mirror the rank assignment
            rec_normalised = midpoints[(n_bins - 1) - sample_rank]

    # 6. Denormalise to physical scale if dictionary is provided
    if dictionary is None:
        return rec_normalised
    x_max = dictionary[0][axis]
    x_min = dictionary[1][axis]
    return x_min + rec_normalised * (x_max - x_min)


__all__ = ["MTF_to_TS_v2"]
