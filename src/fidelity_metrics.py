"""Fidelity + spectral reconstruction metrics for the validation redesign
(doc/VALIDATION_REDESIGN.md, Blocks 1 and 2a).

All functions take originals and reconstructions shaped ``(N, T, C)`` (windows,
timesteps, axes) and return per-window GLOBAL arrays ``(N,)`` obtained by
averaging over the three accelerometer axes, so callers can then take the
across-window mean / median / bootstrap CI.

Signal-fidelity metrics (range-normalised so RP/GASF/GADF/MTF are comparable
regardless of each window's amplitude):
  - NMAE  : MAE  / (per-window per-axis range)
  - NRMSE : RMSE / (per-window per-axis range)
  - Pearson
  - Delta-NMAE : MAE of the first difference divided by the original signal
    range (local-change fidelity). A raw Delta-MAE is also returned.
    This replaces MRE, whose |x|+eps denominator lets accelerometer
    zero-crossings dominate.

Spectral metrics (WISDM smartwatch stream, fs = 20 Hz; Welch PSD):
  - PSD-shape distance: ||Px/sum(Px) - Pr/sum(Pr)||_1
  - dominant-freq err : |argmax_f Px - argmax_f Pr|  (Hz)
  - energy error      : | sum Px - sum Pr | / (sum Px + eps)
  - band-shape diff   : per-band difference between normalised spectral mass
  - mean coherence    : optional, mean magnitude-squared coherence over freq
"""
from __future__ import annotations

import numpy as np
from scipy.signal import coherence, welch

FS = 20.0                       # WISDM smartwatch sampling rate (Hz)
NPERSEG = 64                    # Welch segment for N=128 windows (~3 segments, 50% overlap)
BANDS = ((0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 10.0))  # Hz
_EPS = 1e-12
RANGE_TOL = 1e-8              # m/s^2; below this, range-normalised error is undefined


def _range(x):
    """Per-window per-axis amplitude range (N, C), without artificial flooring."""
    return x.max(axis=1) - x.min(axis=1)


def _range_normalised_axis_mean(values: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Divide axis-level values by valid ranges and average valid axes.

    A range at or below ``RANGE_TOL`` makes a relative error numerically
    meaningless. Such axes are omitted instead of being divided by an epsilon;
    a window with no valid axis receives NaN and is excluded by the table
    generator's NaN-aware summaries.
    """
    valid = ranges > RANGE_TOL
    scaled = np.zeros_like(values, dtype=float)
    np.divide(values, ranges, out=scaled, where=valid)
    count = valid.sum(axis=1)
    return np.divide(scaled.sum(axis=1), count,
                     out=np.full(len(values), np.nan), where=count > 0)


def per_window_fidelity(orig: np.ndarray, rec: np.ndarray) -> dict:
    """Per-window global (N,) fidelity arrays."""
    diff = orig - rec
    rng = _range(orig)                                   # (N, C)
    mae_ax = np.abs(diff).mean(axis=1)                   # (N, C)
    rmse_ax = np.sqrt((diff ** 2).mean(axis=1))          # (N, C)
    d_o = np.diff(orig, axis=1)
    d_r = np.diff(rec, axis=1)
    dmae_ax = np.abs(d_o - d_r).mean(axis=1)             # (N, C)
    oc = orig - orig.mean(axis=1, keepdims=True)
    rc = rec - rec.mean(axis=1, keepdims=True)
    num = (oc * rc).sum(axis=1)
    den = np.linalg.norm(oc, axis=1) * np.linalg.norm(rc, axis=1)
    pear_ax = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    return {
        "NMAE":     _range_normalised_axis_mean(mae_ax, rng),
        "NRMSE":    _range_normalised_axis_mean(rmse_ax, rng),
        "Pearson":  pear_ax.mean(axis=1),
        "dMAE":     dmae_ax.mean(axis=1),                # raw units (m/s^2)
        "dNMAE":    _range_normalised_axis_mean(dmae_ax, rng),
    }


def _psd(sig2d: np.ndarray):
    """Welch PSD of each row of (M, T). Returns f (F,), Pxx (M, F)."""
    f, pxx = welch(sig2d, fs=FS, nperseg=min(NPERSEG, sig2d.shape[1]), axis=-1)
    return f, pxx


def per_window_spectral(orig: np.ndarray, rec: np.ndarray) -> dict:
    """Per-window global (N,) spectral arrays (averaged over axes)."""
    N, T, C = orig.shape
    o = np.transpose(orig, (0, 2, 1)).reshape(N * C, T)
    r = np.transpose(rec, (0, 2, 1)).reshape(N * C, T)
    f, po = _psd(o)
    _, pr = _psd(r)
    energy_o = po.sum(axis=1)
    energy_r = pr.sum(axis=1)
    po_shape = po / np.maximum(energy_o[:, None], _EPS)
    pr_shape = pr / np.maximum(energy_r[:, None], _EPS)
    psd_shape_dist = np.abs(po_shape - pr_shape).sum(axis=1)
    dom_err = np.abs(f[np.argmax(po, axis=1)] - f[np.argmax(pr, axis=1)])
    energy_err = np.abs(energy_o - energy_r) / np.maximum(energy_o, _EPS)
    # Difference in normalised spectral mass, averaged over the defined bands.
    band_shape_diff = np.zeros(N * C)
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        if not m.any():
            continue
        bx = po_shape[:, m].sum(axis=1)
        br = pr_shape[:, m].sum(axis=1)
        band_shape_diff += np.abs(bx - br)
    band_shape_diff /= len(BANDS)

    def _axavg(v):
        return v.reshape(N, C).mean(axis=1)

    return {
        "PSDShapeDist": _axavg(psd_shape_dist),
        "DomFreqErr": _axavg(dom_err),
        "EnergyErr": _axavg(energy_err),
        "BandShapeDiff": _axavg(band_shape_diff),
    }


def mean_coherence(orig: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """Optional: per-window global (N,) mean magnitude-squared coherence."""
    N, T, C = orig.shape
    o = np.transpose(orig, (0, 2, 1)).reshape(N * C, T)
    r = np.transpose(rec, (0, 2, 1)).reshape(N * C, T)
    nps = min(NPERSEG, T)
    out = np.empty(N * C)
    for i in range(N * C):
        _, cxy = coherence(o[i], r[i], fs=FS, nperseg=nps)
        out[i] = np.nanmean(cxy)
    return out.reshape(N, C).mean(axis=1)


def all_metrics(orig: np.ndarray, rec: np.ndarray) -> dict:
    """Union of fidelity + spectral per-window global arrays."""
    return {**per_window_fidelity(orig, rec), **per_window_spectral(orig, rec)}


def demo() -> None:
    """Self-check: identical signal -> perfect scores; degraded -> worse."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi * 3, 128)
    base = np.stack([np.sin(t), np.cos(2 * t), np.sin(0.5 * t)], axis=-1)  # (128, 3)
    orig = np.stack([base + 0.01 * rng.standard_normal(base.shape) for _ in range(20)])  # (20,128,3)

    # 1) Identical reconstruction -> near-perfect on every metric.
    m = all_metrics(orig, orig.copy())
    assert m["NMAE"].mean() < 1e-9, m["NMAE"].mean()
    assert m["NRMSE"].mean() < 1e-9
    assert m["dMAE"].mean() < 1e-9
    assert abs(m["Pearson"].mean() - 1.0) < 1e-9
    assert m["PSDShapeDist"].mean() < 1e-9, m["PSDShapeDist"].mean()
    assert m["DomFreqErr"].mean() < 1e-9

    # 2) Amplitude-scaled copy: spectral shape is preserved but energy changes.
    scaled = orig * 0.5
    ms = all_metrics(orig, scaled)
    assert ms["NMAE"].mean() > m["NMAE"].mean() + 1e-3
    assert ms["PSDShapeDist"].mean() < 1e-9
    assert ms["EnergyErr"].mean() > 0.1
    assert ms["Pearson"].mean() > 0.99          # shape preserved

    # 3) Frequency-shifted copy: dominant-frequency error must fire.
    t2 = np.linspace(0, 2 * np.pi * 6, 128)     # double the base frequency
    shifted = np.stack([np.stack([np.sin(t2), np.cos(2 * t2), np.sin(0.5 * t2)], -1)
                        for _ in range(20)])
    mf = all_metrics(orig, shifted)
    assert mf["DomFreqErr"].mean() > 0.2, mf["DomFreqErr"].mean()

    # 4) delta-MAE catches local-change corruption a global scale can hide.
    jitter = orig + 0.3 * rng.standard_normal(orig.shape)
    mj = all_metrics(orig, jitter)
    assert mj["dMAE"].mean() > m["dMAE"].mean() + 1e-2
    print("fidelity_metrics.demo OK")


if __name__ == "__main__":
    demo()
