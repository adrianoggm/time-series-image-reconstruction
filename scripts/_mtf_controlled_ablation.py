"""Controlled MTF decoder ablation: every variant reads the SAME 8-bit image.

Addresses the reviewer point that the archived MTF comparison fed the
cosine-pivot decoder an 8-bit image but the Fiedler variants the floating MTF
matrix, confounding ordering quality with storage precision. Here every decoder
variant reconstructs from the identical uint8 PNG round trip, on the same 100
stratified LOTO fold-0 windows (seed 0, Q=4, T=128). MAE is in physical units
(m/s^2); Pearson is averaged across the three axes.

Run in WSL:  cd /home/adrianogarcia/tfg && .venv/bin/python -m scripts._mtf_controlled_ablation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyts.image import MarkovTransitionField

from src.config import DEFAULT_DATA_DIR
from src.MTF import _legacy_MTF_to_TS
from src.MTF_v2 import MTF_to_TS_v2
from src.paths import windowed_npy

SAMPLING, FOLD, N_FOLDS, N_SAMPLES, Q = "loto", 0, 3, 100, 4
SEED = 0
# Repo-relative, so the artefact lands next to the other paper statistics
# instead of in a checkout-specific home directory.
OUT = Path(__file__).resolve().parents[1] / "results" / "_stats" / "mtf_decoder_ablation.json"


def encode_8bit(x_axis: np.ndarray, q: int) -> np.ndarray:
    """MTF-encode one axis and return the uint8 PNG round trip in [0,1].
    This identical array is the input to EVERY decoder variant."""
    T = len(x_axis)
    m = MarkovTransitionField(image_size=T, overlapping=True, n_bins=q)
    img = m.fit_transform(x_axis.reshape(1, -1)).reshape(T, T)   # [0,1]
    u8 = np.round(np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    return u8.astype(np.float64) / 255.0


def bin_midpoints(x_axis: np.ndarray, q: int) -> np.ndarray:
    xmin, xmax = x_axis.min(), x_axis.max()
    norm = (x_axis - xmin) / (xmax - xmin + 1e-12)
    b = np.quantile(norm, np.linspace(0, 1, q + 1))
    return np.array([0.5 * (b[k] + b[k + 1]) for k in range(q)])


def mae_pear(rec: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(rec - target)))
    den = np.std(rec) * np.std(target)
    r = float(np.corrcoef(rec, target)[0, 1]) if den > 1e-12 else 0.0
    return mae, r


def main() -> None:
    X = np.load(windowed_npy("validation_data", SAMPLING, FOLD, N_FOLDS, DEFAULT_DATA_DIR))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(X.shape[0], size=min(N_SAMPLES, X.shape[0]), replace=False)

    variants = {"cosine-pivot": [], "Fiedler": [],
                "Fiedler+midpoint": [], "Fiedler+midpoint+sign": []}
    for i in idx:
        w = X[i]
        mx = [w[:, j].max() for j in range(3)]
        mn = [w[:, j].min() for j in range(3)]
        for k in range(3):
            img = encode_8bit(w[:, k], Q)          # SAME 8-bit image for all variants
            tgt = w[:, k]
            variants["cosine-pivot"].append(
                mae_pear(_legacy_MTF_to_TS(img, [mx, mn], index=k, numstates=Q), tgt))
            variants["Fiedler"].append(
                mae_pear(MTF_to_TS_v2(img, dictionary=[mx, mn], axis=k, n_bins=Q), tgt))
            mids = bin_midpoints(w[:, k], Q)
            variants["Fiedler+midpoint"].append(
                mae_pear(MTF_to_TS_v2(img, dictionary=[mx, mn], axis=k, n_bins=Q,
                                      bin_midpoints=mids), tgt))
            sgn = 1 if w[0, k] > 0.5 * (w[:, k].min() + w[:, k].max()) else 0
            variants["Fiedler+midpoint+sign"].append(
                mae_pear(MTF_to_TS_v2(img, dictionary=[mx, mn], axis=k, n_bins=Q,
                                      bin_midpoints=mids, sign_bit=sgn), tgt))

    out = {}
    print(f"{'variant':<28}{'MAE':>10}{'Pearson':>10}   (same 8-bit image, N={len(idx)} windows, Q={Q})")
    print("-" * 60)
    for name, vals in variants.items():
        a = np.array(vals)
        out[name] = {"MAE": float(a[:, 0].mean()), "Pearson": float(a[:, 1].mean())}
        print(f"{name:<28}{a[:, 0].mean():>10.4f}{a[:, 1].mean():>+10.4f}")

    # sanity: adding info must not increase MAE (ordering -> +midpoint -> +sign monotone)
    seq = [out[n]["MAE"] for n in ("Fiedler", "Fiedler+midpoint", "Fiedler+midpoint+sign")]
    assert seq[0] >= seq[1] - 1e-6 >= 0 and seq[1] >= seq[2] - 1e-6, f"non-monotone metadata gain: {seq}"

    payload = {
        "design": {"sampling": SAMPLING, "fold": FOLD, "n_folds": N_FOLDS,
                   "n_windows": int(len(idx)), "seed": SEED, "n_bins": Q,
                   "window_indices": [int(i) for i in idx],
                   "note": "every variant decodes the identical uint8 PNG round trip"},
        "variants": out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
