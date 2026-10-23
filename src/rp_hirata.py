"""Classical RP reconstruction of Hirata et al. (RPR), on the conventional binary RP.

Source of the procedure: Hirata, Horai and Aihara, Eur. Phys. J. Special Topics
164, 13-22 (2008), as restated by the same group in Hirata, Oda, Ohta and
Aihara, Sci. Rep. 6, 34982 (2016), Eqs. (1)-(2) and Figs. 2-3 (the 2008 text
itself was not accessible when this was written):

1. Binary recurrence plot R(i, j) = 1 if ||s_i - s_j|| < eps, else 0.
2. Weighted graph: an edge between i != j when R(i, j) = 1, with local distance
   d(i, j) = 1 - |G_i & G_j| / |G_i | G_j|, G_i = {k : R(i, k) = 1}.
3. Shortest distances between every pair of nodes (Dijkstra).
4. Multidimensional scaling (classical / principal coordinates, Gower 1966).

That yields the state cloud up to an isometry; it is the published algorithm
and ``states`` returns exactly that. Turning it into a series of N samples for
the metrics is OUR post-processing, kept separate in ``to_series``: the leading
MDS coordinate, the mean appended (N-1 states -> N samples) and min-max
rescaling to the channel range. A binary RP carries no sign, so the reflection
cannot be resolved: ``orientation="as_is"`` keeps the MDS output, and
``"oracle"`` picks the better of x and its reflection using the true signal —
an evaluation-only bound, never a reconstruction result.

eps is not prescribed by the method; ``eps_for_rate`` sets it from a target
recurrence rate so it can be matched to the other variants and swept.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def state_distances(x: np.ndarray) -> np.ndarray:
    s = np.stack([x[:-1], x[1:]], axis=1)
    return np.linalg.norm(s[:, None, :] - s[None, :, :], axis=-1)


def eps_for_rate(D: np.ndarray, rate: float) -> float:
    """Smallest eps whose off-diagonal recurrence rate (strict <) reaches ``rate``."""
    off = np.sort(D[np.triu_indices_from(D, k=1)])
    k = min(int(np.ceil(rate * off.size)), off.size - 1)
    return float(np.nextafter(off[k - 1], np.inf)) if k > 0 else float(off[0])


def binary_rp(D: np.ndarray, eps: float) -> np.ndarray:
    return (D < eps).astype(np.int8)                     # Eq. (1); diagonal is 1


def states(B: np.ndarray, dims: int = 2) -> tuple[np.ndarray, int]:
    """Steps 2-4 on a binary RP. Returns the MDS state coordinates and the unreachable-pair count."""
    C = B.astype(float)
    inter = C @ C.T
    sizes = C.sum(axis=1)
    W = 1.0 - inter / (sizes[:, None] + sizes[None, :] - inter)      # Eq. (2)
    rows, cols = np.nonzero(C)
    keep = rows != cols
    G = csr_matrix((W[rows[keep], cols[keep]], (rows[keep], cols[keep])), shape=C.shape)
    SP = shortest_path(G, method="D", directed=False)
    unreachable = int(np.isinf(SP).sum())
    SP = np.where(np.isfinite(SP), SP, 0.0)              # same convention as our other decoders; counted
    n = len(SP)
    J = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(-0.5 * J @ np.square(SP) @ J)
    order = np.argsort(vals)[::-1][:dims]
    return vecs[:, order] * np.sqrt(np.clip(vals[order], 0.0, None)), unreachable


def to_series(Y: np.ndarray, minmax_c, orientation: str = "as_is", x_true: np.ndarray | None = None) -> np.ndarray:
    """OUR post-processing (not part of RPR): leading coordinate -> N samples in physical units."""
    lead = Y[:, 0]
    mx, mn = minmax_c

    def scale(v):
        v = np.append(v, v.mean())
        lo, hi = v.min(), v.max()
        return np.full_like(v, 0.5 * (mn + mx)) if hi - lo < 1e-12 else mn + (v - lo) / (hi - lo) * (mx - mn)

    x = scale(lead)
    if orientation == "as_is":
        return x
    if orientation == "oracle":
        if x_true is None:
            raise ValueError("the oracle orientation needs the true signal")
        y = scale(-lead)
        return y if np.abs(y - x_true).mean() < np.abs(x - x_true).mean() else x
    raise ValueError(f"unknown orientation {orientation!r}")


def decode_series(x: np.ndarray, rate: float = 0.75, orientation: str = "as_is") -> tuple[np.ndarray, dict]:
    """Float-precision experiment: signal -> conventional binary RP -> RPR -> series."""
    D = state_distances(np.asarray(x, dtype=float))
    eps = eps_for_rate(D, rate)
    Y, unreachable = states(binary_rp(D, eps))
    rec = to_series(Y, (x.max(), x.min()), orientation, x_true=x)
    return rec, {"eps": eps, "unreachable_pairs": unreachable}
