"""Gate for src/representations.py.

1. Three channels reproduce the legacy RGB pipeline bit for bit (images and reconstructions).
2. Any length and any number of channels work.
3. On the float operator RP, GASF, GADF and XIRP are exact; 8-bit error is reported per method.

    python -m scripts._check_representations
"""
import numpy as np

import src.GADF as gadf
import src.GAF as gaf
import src.MTF as mtf
import src.recurrence_plots as rec
from scripts._check_baseline import minmax, synthetic_windows
from src.representations import REPRESENTATIONS, calibration, decode_window, encode_window

LEGACY = {"RP": (rec.SavevarRP_XYZ, lambda im, d: rec.Reconstruct_RP(im, d, 0, method="direct2ds")[0]),
          "GASF": (gaf.SavevarGAF_XYZ, lambda im, d: gaf.Reconstruct_GAF(im, d, 0)),
          "GADF": (gadf.SavevarGADF_XYZ, lambda im, d: gadf.Reconstruct_GADF(im, d, 0)),
          "MTF": (mtf.SavevarMTF_XYZ, lambda im, d: mtf.Reconstruct_MTF(im, d, 0))}

for k, win in enumerate(synthetic_windows()):
    for name, (enc, dec) in LEGACY.items():
        legacy_img = np.asarray(enc(win, 0, k, "x", normalized=1, saveImage=False, TIME_STEPS=128))
        stack = encode_window(name, win)
        assert np.array_equal(np.moveaxis(stack, 0, -1), legacy_img), (k, name, "image")
        legacy_rec = np.asarray(dec(legacy_img, {0: minmax(win)}), dtype=float)
        new_rec = decode_window(name, stack, calibration(win))
        assert np.array_equal(new_rec, legacy_rec.T), (k, name, "decode", np.abs(new_rec - legacy_rec.T).max())
print("[ok] 3-channel images and reconstructions bit-identical to the legacy pipeline (RP, GASF, GADF, MTF)")

rng = np.random.default_rng(0)
for T, C in ((64, 1), (200, 5), (256, 12)):
    t = np.arange(T)[:, None]
    win = np.sin(2 * np.pi * t * (1 + np.arange(C)) / T) + 0.1 * np.cumsum(rng.standard_normal((T, C)), 0)
    mm = calibration(win)
    rng_ax = win.max(0) - win.min(0)
    row = []
    for name in REPRESENTATIONS:
        img = encode_window(name, win)
        H = T - 1 if name == "RP" else T
        assert img.shape == (C, H, H) and img.dtype == np.uint8, (name, img.shape)
        nm = {}
        for cond, stack in (("float", encode_window(name, win, quantize=False)), ("8bit", img)):
            out = decode_window(name, stack, mm)
            assert out.shape == (T, C) and np.isfinite(out).all(), (name, cond, out.shape)
            nm[cond] = float((np.abs(out - win).mean(0) / rng_ax).mean())
        if name != "MTF":
            assert nm["float"] < 1e-6, (name, T, C, nm)
        row.append(f"{name} {nm['float']:.0e}/{nm['8bit']:.4f}")
    print(f"[ok] T={T} C={C}: NMAE float/8-bit  " + "  ".join(row))

# Scale equivariance: a channel of amplitude ~1e-12 (dead sensor) decodes as well as one of amplitude 1.
tiny = 1e-12 * synthetic_windows()[3]
for name in ("RP", "GASF", "GADF", "XIRP"):
    out = decode_window(name, encode_window(name, tiny, quantize=False), calibration(tiny))
    nm = float((np.abs(out - tiny).mean(0) / np.ptp(tiny, axis=0)).mean())
    assert nm < 1e-6, (name, nm)
print("[ok] amplitude 1e-12: RP, GASF, GADF and XIRP still exact in float")
print("all checks passed")
