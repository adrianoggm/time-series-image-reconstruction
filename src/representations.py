"""One interface for every image representation: per channel, any length, float or 8-bit.

Each :class:`Representation` splits what the legacy ``Savevar*_XYZ`` /
``Reconstruct_*`` pairs did on fixed 3-channel RGB images into four steps on a
single channel:

    encode(x, mm)     series (T,) -> operator matrix in float (the representation itself)
    to_uint8(M)       operator -> stored 8-bit image, with the method's own normalisation
    from_uint8(img)   stored image -> matrix in the decoder's input domain
    decode(M, mm)     matrix -> series (T,) in physical units

``mm = (max, min)`` is the per-channel amplitude calibration all methods share
(the ``dictionary[k] = [maxs, mins]`` convention of the pipeline). Decoding the
float operator measures the algorithm; decoding ``from_uint8(to_uint8(...))``
adds 8-bit storage, so the two error sources can be reported separately.

For three channels the stacked 8-bit images and the reconstructions are
bit-identical to the legacy RGB pipeline (scripts/_check_representations.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Representation:
    name: str
    encode: Callable[[np.ndarray, tuple], np.ndarray]
    to_uint8: Callable[[np.ndarray], np.ndarray]
    from_uint8: Callable[[np.ndarray], np.ndarray]
    decode: Callable[[np.ndarray, tuple], np.ndarray]


def _u8(m01: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(m01) * 255).astype(np.uint8)


def _dictionary(mm):
    return [[mm[0]], [mm[1]]]


# ---- RP: signed RP of adjacent states, Direct-2D decoder (production) -------------------------
def _rp_encode(x, mm):
    from src.recurrence_plots import varRP2

    return varRP2(np.asarray(x, dtype=float), TIME_STEPS=len(x))


def _rp_to_uint8(R):
    from src.recurrence_plots import NormalizeMatrix

    return _u8(NormalizeMatrix(R))


def _rp_from_uint8(img):
    png = np.asarray(img, dtype=float)
    return png - float(np.median(np.diag(png)))      # metadata-free: the diagonal is R = 0


def _rp_decode(R, mm):
    from src.rp_decoders import decode_channel

    return decode_channel(R, mm, source="direct", dims=2, recover="overlap", signed=True, rho=0.75,
                          scale_invariant=True, orientation_method="closed_form")


# ---- GASF / GADF ----------------------------------------------------------------------------
def _gaf_encode(x, mm):
    from src.GAF import gramian_angular_field

    return gramian_angular_field(np.asarray(x, dtype=float).reshape(1, -1))


def _gadf_encode(x, mm):
    from src.GADF import gramian_angular_diff_field

    return gramian_angular_diff_field(np.asarray(x, dtype=float).reshape(1, -1))


def _gaf_to_uint8(G):
    from src.GAF import normalize_to_zero_one

    return _u8(normalize_to_zero_one(G))


def _signed_from_uint8(img):
    return np.interp(np.asarray(img, dtype=float), (0, 255), (-1, 1))


def _gaf_decode(G, mm):
    from src.GAF import GAF_to_TS

    return GAF_to_TS(G, 0, _dictionary(mm))


def _gadf_decode(G, mm):
    from src.GADF import GADF_to_TS

    return GADF_to_TS(G, 0, _dictionary(mm))


# ---- MTF (Q = 4, Fiedler decoder) -----------------------------------------------------------
def _mtf_encode(x, mm):
    from pyts.image import MarkovTransitionField

    T = len(x)
    return MarkovTransitionField(image_size=T, overlapping=True, n_bins=4).fit_transform(
        np.asarray(x, dtype=float).reshape(1, -1)).reshape(T, T)


def _mtf_from_uint8(img):
    return np.interp(np.asarray(img, dtype=float), (0, 255), (0, 1))


def _mtf_decode(M, mm):
    from src.MTF import MTF_to_TS

    return MTF_to_TS(M, _dictionary(mm), 0)


# ---- XIRP (Hellermann and Lessmann), inversion by mean ---------------------------------------
def _xirp_encode(x, mm):
    from src.XIRP import to_positive, xirp

    return xirp(to_positive(x, mm[1], mm[0]))


def _xirp_to_uint8(R):
    from src.XIRP import to_uint8

    return to_uint8(R)


def _xirp_from_uint8(img):
    from src.XIRP import from_uint8

    return from_uint8(img)


def _xirp_decode(R, mm):
    from src.XIRP import from_positive, invert

    return from_positive(invert(R, "im"), mm[1], mm[0])


REPRESENTATIONS: dict[str, Representation] = {
    r.name: r for r in (
        Representation("RP", _rp_encode, _rp_to_uint8, _rp_from_uint8, _rp_decode),
        Representation("GASF", _gaf_encode, _gaf_to_uint8, _signed_from_uint8, _gaf_decode),
        Representation("GADF", _gadf_encode, _gaf_to_uint8, _signed_from_uint8, _gadf_decode),
        Representation("MTF", _mtf_encode, _u8, _mtf_from_uint8, _mtf_decode),
        Representation("XIRP", _xirp_encode, _xirp_to_uint8, _xirp_from_uint8, _xirp_decode),
    )
}


def calibration(window: np.ndarray) -> list[tuple[float, float]]:
    """Per-channel (max, min) of a (T, C) window."""
    return [(float(window[:, c].max()), float(window[:, c].min())) for c in range(window.shape[1])]


def encode_window(name: str, window: np.ndarray, quantize: bool = True) -> np.ndarray:
    """(T, C) window -> (C, H, W) stack: uint8 images, or float operators when ``quantize=False``."""
    rep, mm = REPRESENTATIONS[name], calibration(window)
    mats = [rep.encode(window[:, c], mm[c]) for c in range(window.shape[1])]
    return np.stack([rep.to_uint8(m) for m in mats] if quantize else mats)


def decode_window(name: str, stack: np.ndarray, mm: list[tuple[float, float]]) -> np.ndarray:
    """(C, H, W) stack from ``encode_window`` + per-channel calibration -> (T, C) reconstruction."""
    rep = REPRESENTATIONS[name]
    quantized = stack.dtype == np.uint8
    return np.stack([rep.decode(rep.from_uint8(m) if quantized else m, mm[c]) for c, m in enumerate(stack)], axis=1)
