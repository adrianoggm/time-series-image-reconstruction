"""Our previous RP decoder, reproduced as specified — not optimised.

Specification: Steps 1-6 of the RP reconstruction in the first manuscript
(``git show 6a02ad1:doc/main.tex``), an adaptation of Hirata et al. (2008) to
the signed RP. Where the text is silent the code of that period is followed.
``docs/audit/m0_baseline.md`` lists how the later code drifted from it; this
module is the faithful version (roadmap #65). Choices, in order:

1. Threshold: tau is the rho-quantile (rho = 0.75) of the OFF-DIAGONAL state
   distances |R_ij|, and c_ij = 1 iff |R_ij| <= tau (so c_ii = 1).
2-3. Graph: an edge for every c_ij = 1, weighted with the Jaccard dissimilarity
   of the neighbour sets, w_ij = 1 - |G_i & G_j| / |G_i | G_j|. Pairs with
   c_ij = 0 have no edge. An edge whose weight is exactly 0 (identical
   neighbourhoods) is kept as an edge; the 2025 code lost it because a dense
   SciPy graph reads 0 as "no edge".
4. Dijkstra on that graph. Unreachable pairs (disconnected graph) are set to 0
   before MDS, as the code always did, and counted.
5. MDS, keeping the coordinate of the largest-variance direction. ``mds="smacof"``
   (default) is what the code ran: sklearn SMACOF in 2-D with random_state=1, then
   the column at the argmax of the coordinate covariance eigenvalues — an
   eigenvalue index used as a column index, reproduced as is. ``mds="classical"``
   is the literal reading of the text (leading eigenvector of classical MDS).
6. Append the mean (N-1 states -> N samples), rescale to the channel range, and
   keep x or its reflection, whichever regenerates a signed RP closer to the
   observed one. The comparison is scale-invariant so that the same input as the
   new decoder (the metadata-free, diagonal-centred RP) can be used.

The manuscript's Step 7 (cubic-spline resampling to N) is not applied: after
Step 6.1 the series already has N samples, so it would be the identity.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from src.rp_decoders import _rescale, signed_rp_error, symmetrized_magnitude

RHO = 0.75


def jaccard_graph(R_obs, rho: float = RHO) -> csr_matrix:
    D = symmetrized_magnitude(R_obs)
    iu = np.triu_indices_from(D, k=1)
    C = (D <= np.quantile(D[iu], rho)).astype(float)       # adjacency, self loops included
    inter = C @ C.T
    sizes = C.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    W = 1.0 - inter / union                                  # union >= 1 since c_ii = 1
    rows, cols = np.nonzero(C)
    keep = rows != cols
    return csr_matrix((W[rows[keep], cols[keep]], (rows[keep], cols[keep])), shape=D.shape)


def geodesic_distances(R_obs, rho: float = RHO) -> tuple[np.ndarray, int]:
    """Steps 1-4. Returns the symmetric geodesic matrix and the number of unreachable pairs."""
    G = jaccard_graph(R_obs, rho)
    # Explicit zeros in a CSR graph are edges for scipy.sparse.csgraph, so zero-weight edges survive.
    SP = shortest_path(G, method="D", directed=False)
    unreachable = int(np.isinf(SP).sum())
    return np.where(np.isfinite(SP), SP, 0.0), unreachable


def project_1d(dmat: np.ndarray, mds: str = "smacof") -> np.ndarray:
    """Step 5."""
    if mds == "classical":
        n = dmat.shape[0]
        J = np.eye(n) - 1.0 / n
        B = -0.5 * J @ np.square(dmat) @ J
        vals, vecs = np.linalg.eigh(0.5 * (B + B.T))
        return vecs[:, -1] * np.sqrt(max(vals[-1], 0.0))
    if mds == "smacof":
        from sklearn.manifold import MDS

        # SMACOF divides by a zero stress normaliser on degenerate inputs (e.g. impulse windows) and
        # floods stderr; callers check the output is finite.
        with np.errstate(divide="ignore", invalid="ignore"):
            Y = MDS(n_components=2, dissimilarity="precomputed", random_state=1).fit_transform(dmat)
        eigenvalues, _ = np.linalg.eig(np.cov(Y, rowvar=False))
        return Y[:, int(np.argmax(eigenvalues))]
    raise ValueError(f"unknown mds {mds!r}")


def decode_channel(R_obs, minmax_c, rho: float = RHO, mds: str = "smacof", signed: bool = True,
                   scale_invariant: bool = True, n_out: int | None = None) -> tuple[np.ndarray, dict]:
    """Steps 1-6 for one channel. Returns the N-sample reconstruction and a diagnostics dict."""
    dmat, unreachable = geodesic_distances(R_obs, rho)
    proj = project_1d(dmat, mds)
    x = _rescale(np.append(proj, proj.mean()), minmax_c)
    flipped = False
    if signed:
        mirrored = _rescale(-np.append(proj, proj.mean()), minmax_c)
        if signed_rp_error(R_obs, mirrored, scale_invariant) < signed_rp_error(R_obs, x, scale_invariant):
            x, flipped = mirrored, True
    n_out = n_out or len(x)
    return x[:n_out], {"unreachable_pairs": unreachable, "flipped": flipped}
