"""Regression gate for every encoder/decoder as it stands at the M0 baseline.

Encodes a fixed set of synthetic triaxial windows with the production encoders
(RP, GASF, GADF, MTF), decodes them with every decoder variant the repo still
ships, and compares against ``results/_baseline/golden_decoders.npz``.

    python -m scripts._check_baseline            # compare, exit 1 on drift
    python -m scripts._check_baseline --update   # rewrite the golden file

Only refresh the golden file when a change of behaviour is intended, and say so
in the commit. The windows are synthetic so the gate needs no WISDM download.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

import src.GADF as gadf
import src.GAF as gaf
import src.MTF as mtf
import src.recurrence_plots as rec
from src.rp_decoders import decode_window

T = 128
GOLDEN = Path("results/_baseline/golden_decoders.npz")
ATOL = 1e-6  # float drift across BLAS builds; real behaviour changes are orders larger


def synthetic_windows():
    """Four deterministic (T, 3) windows: sinusoids, trend+noise, impulse, random walk."""
    rng = np.random.default_rng(20261023)
    t = np.linspace(0, 1, T)
    w = [
        np.stack([np.sin(2 * np.pi * 3 * t), 0.5 * np.cos(2 * np.pi * 5 * t), np.sin(2 * np.pi * 7 * t + 1)], 1),
        np.stack([4 * t + 0.1 * rng.standard_normal(T), -2 * t + 0.1 * rng.standard_normal(T),
                  0.3 * rng.standard_normal(T)], 1),
        np.zeros((T, 3)),
        np.cumsum(rng.standard_normal((T, 3)), 0),
    ]
    w[2][40, 0], w[2][80, 1], w[2][100, 2] = 5.0, -3.0, 2.0
    w[2] += 0.01 * rng.standard_normal((T, 3))
    return [9.81 * x for x in w]  # accelerometer-like amplitudes


def minmax(x):
    return [[float(v) for v in x.max(0)], [float(v) for v in x.min(0)]]


def run():
    out = {}
    windows = synthetic_windows()
    dictionary = {k: minmax(x) for k, x in enumerate(windows)}
    encoders = {"RP": rec.SavevarRP_XYZ, "GAF": gaf.SavevarGAF_XYZ, "GADF": gadf.SavevarGADF_XYZ,
                "MTF": mtf.SavevarMTF_XYZ}
    for k, x in enumerate(windows):
        out[f"w{k}/original"] = x
        imgs = {}
        for name, enc in encoders.items():
            imgs[name] = np.asarray(enc(x, 0, k, "x", normalized=1, saveImage=False, TIME_STEPS=T))
            out[f"w{k}/img_sha256/{name}"] = np.array(hashlib.sha256(imgs[name].tobytes()).hexdigest())

        # RP: production (Direct-2D, metadata-free) and the historical 1-D / 2-D graph paths.
        for method in ("direct2ds", "1d", "2d"):
            r, _ = rec.Reconstruct_RP(imgs["RP"], dictionary, k, method=method)
            out[f"w{k}/RP/{method}"] = np.asarray(r, dtype=float)
        # RP ablation sources at the paper's rho=0.60.
        for source, dims, recover in (("geo", 2, "overlap"), ("jaccard", 1, "append")):
            r, _ = decode_window(imgs["RP"], dictionary[k], source=source, dims=dims, recover=recover,
                                 signed=True, rho=0.60, rp_range="diagonal")
            out[f"w{k}/RP/{source}"] = np.asarray(r, dtype=float)
        out[f"w{k}/GAF"] = np.asarray(gaf.Reconstruct_GAF(imgs["GAF"], dictionary, k), dtype=float)
        out[f"w{k}/GADF"] = np.asarray(gadf.Reconstruct_GADF(imgs["GADF"], dictionary, k), dtype=float)
        out[f"w{k}/MTF"] = np.asarray(mtf.Reconstruct_MTF(imgs["MTF"], dictionary, k), dtype=float)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true", help="rewrite the golden file")
    args = ap.parse_args()

    current = run()
    if args.update:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, **current)
        print(f"wrote {GOLDEN} ({len(current)} arrays)")
        return 0

    golden = np.load(GOLDEN)
    failures = []
    for key in sorted(set(golden.files) | set(current)):
        if key not in golden.files or key not in current:
            failures.append(f"{key}: present on one side only")
            continue
        g, c = golden[key], current[key]
        if g.dtype.kind == "U":
            if str(g) != str(c):
                failures.append(f"{key}: encoder image changed")
        elif g.shape != c.shape:
            failures.append(f"{key}: shape {g.shape} -> {c.shape}")
        elif not np.allclose(g, c, rtol=0, atol=ATOL, equal_nan=True):
            failures.append(f"{key}: max |diff| = {np.nanmax(np.abs(g - c)):.3e}")
    for f in failures:
        print("DRIFT", f)
    print(f"{len(current) - len(failures)}/{len(current)} arrays match the M0 baseline")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
