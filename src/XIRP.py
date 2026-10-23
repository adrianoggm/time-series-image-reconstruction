"""Extended Intertemporal Return Plot (XIRP).

Hellermann and Lessmann (International Journal of Forecasting, 2026; preprint
arXiv:2112.08060) define, for a strictly positive series ``x`` of length S,

    R[i, j] = log(x_i / x_j)   for i != j
    R[i, i] = x_i

The off-diagonal holds scale-invariant log-returns, the diagonal the series
itself. They invert either by reading the diagonal, or by *inversion by mean*
(IM): every column j proposes x_i = x_j * exp(R[i, j]) and the proposals are
averaged.

Two choices here are ours, not the paper's:

* Positivity. The log needs x > 0 and the paper does not treat other series.
  Each channel is mapped with the per-window min-max calibration the other
  four encoders already share (``dictionary[k] = [maxs, mins]``) onto
  ``[c, 1 + c]`` (``c = 1`` by default) before encoding, and back after decoding.
* 8-bit storage without metadata. After that shift both parts have fixed,
  known ranges, the diagonal ``[c, 1 + c]`` and the off-diagonal
  ``[-L, L]`` with ``L = log((1 + c) / c)``, so each is mapped to [0, 255]
  by a fixed affine map and the image alone suffices to decode.
"""

from __future__ import annotations

import numpy as np

C = 1.0


def _bound(c: float) -> float:
    return float(np.log((1.0 + c) / c))


def xirp(x: np.ndarray) -> np.ndarray:
    """(S,) strictly positive series -> (S, S) XIRP, exactly as in the paper."""
    x = np.asarray(x, dtype=float)
    if (x <= 0).any():
        raise ValueError("XIRP needs a strictly positive series; shift it first (see to_positive)")
    lx = np.log(x)
    R = lx[:, None] - lx[None, :]
    np.fill_diagonal(R, x)
    return R


def invert(R: np.ndarray, method: str = "im") -> np.ndarray:
    """XIRP -> series. ``"diag"`` reads the diagonal; ``"im"`` is the paper's inversion by mean."""
    d = np.diag(R).astype(float)
    if method == "diag":
        return d
    if method == "im":
        off = R.astype(float).copy()
        np.fill_diagonal(off, 0.0)              # log(x_i / x_i) = 0, so column i proposes d_i itself
        return (np.exp(off) * d[None, :]).mean(axis=1)
    raise ValueError(f"unknown inversion {method!r}")


def to_positive(x: np.ndarray, lo: float, hi: float, c: float = C) -> np.ndarray:
    span = hi - lo
    return c + (np.asarray(x, dtype=float) - lo) / span if span > 0 else np.full(len(x), c + 0.5)


def from_positive(p: np.ndarray, lo: float, hi: float, c: float = C) -> np.ndarray:
    return lo + (np.asarray(p, dtype=float) - c) * (hi - lo)


def to_uint8(R: np.ndarray, c: float = C) -> np.ndarray:
    """Fixed-range 8-bit quantisation: diagonal from [c, 1+c], off-diagonal from [-L, L]."""
    L = _bound(c)
    q = (R + L) / (2 * L)
    np.fill_diagonal(q, np.diag(R) - c)
    return np.round(np.clip(q, 0.0, 1.0) * 255).astype(np.uint8)


def from_uint8(img: np.ndarray, c: float = C) -> np.ndarray:
    L = _bound(c)
    q = img.astype(float) / 255.0
    R = q * 2 * L - L
    np.fill_diagonal(R, np.diag(q) + c)
    return R


def encode_window(window: np.ndarray, minmax, c: float = C) -> np.ndarray:
    """(T, 3) window + ``[maxs, mins]`` calibration -> (T, T, 3) uint8 image, one channel per axis."""
    maxs, mins = minmax
    return np.stack([to_uint8(xirp(to_positive(window[:, k], mins[k], maxs[k], c)), c)
                     for k in range(window.shape[1])], axis=-1)


def decode_window(img: np.ndarray, minmax, method: str = "im", c: float = C) -> np.ndarray:
    """(T, T, 3) uint8 image + calibration -> (T, 3) reconstruction in physical units."""
    maxs, mins = minmax
    return np.stack([from_positive(invert(from_uint8(img[:, :, k], c), method), mins[k], maxs[k], c)
                     for k in range(img.shape[2])], axis=1)
