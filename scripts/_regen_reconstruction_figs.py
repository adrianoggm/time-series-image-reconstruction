"""Regenerate the reconstruction-comparison figures the paper references.

For the synthetic signal (cos / sin / sin-cos) and for the WISDM window
1600x0 (Walking), encode → decode under each of the four encoders and
plot original vs reconstructed (all three channels overlaid) at N=128.

Output goes to `doc/images/` with the filenames the paper already references:

  Synthetic (one per encoder):
    Comparison_all_wideRP.png
    Comparison_all_wideGAF.png
    Comparison_all_wideGADF.png   (new)
    Comparison_all_widemtf.png

  WISDM 1600x0 (global, one per encoder):
    rp_sample_0_global.png
    gaf_sample_0_global.png
    gadf_sample_0_global.png      (new)
    mtf_sample_0_global.png

  WISDM 1600x0 per-axis for the four encoders (x/y/z):
    {rp,gaf,gadf,mtf}_sample_0_{x,y,z}_comp.png
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import src.GADF as gadf_mod
import src.GAF as gaf_mod
import src.MTF as mtf_mod
import src.recurrence_plots as rec_mod

N = 128
OUT = "/mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales/doc/images"

sns.set(style="whitegrid", context="paper", font="serif", rc={
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "lines.linewidth": 1.6,
})


def _read_encoded_png(path: str) -> np.ndarray:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _dictionary_for(signal: np.ndarray) -> dict:
    """Mimic the `dictionary` passed to the Reconstruct_* functions: per-window
    per-channel max/min used to undo the [0,1] normalisation."""
    maxs = [float(signal[:, c].max()) for c in range(3)]
    mins = [float(signal[:, c].min()) for c in range(3)]
    return {0: [maxs, mins]}


def encode_decode(signal: np.ndarray, encoder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode, decode and return (reconstructed_x, _y, _z) at length N."""
    d = _dictionary_for(signal)
    with tempfile.TemporaryDirectory() as tmp:
        slash = tmp + "/"
        if encoder == "RP":
            rec_mod.SavevarRP_XYZ(signal, "w", "", "", normalized=1,
                                  path=slash, saveImage=True, TIME_STEPS=N)
            img = _read_encoded_png(os.path.join(tmp, "w.png"))
            # Production DIRECT-2D-S decoder: metadata-free diagonal centring
            # (rp_range=None at the Reconstruct_RP boundary) and closed-form
            # orientation, exactly as used for the regenerated pipeline arrays.
            rec, _ = rec_mod.Reconstruct_RP(img, d, 0, method="direct2ds", rp_range=None)
            # rec.shape is (3, N)
        elif encoder == "GAF":
            gaf_mod.SavevarGAF_XYZ(signal, "w", "", "", normalized=1,
                                   path=slash, saveImage=True, TIME_STEPS=N)
            img = _read_encoded_png(os.path.join(tmp, "wgasf.png"))
            rec = gaf_mod.Reconstruct_GAF(img, d, 0)
        elif encoder == "GADF":
            gadf_mod.SavevarGADF_XYZ(signal, "w", "", "", normalized=1,
                                     path=slash, saveImage=True, TIME_STEPS=N)
            img = _read_encoded_png(os.path.join(tmp, "wgadf.png"))
            rec = gadf_mod.Reconstruct_GADF(img, d, 0)
        elif encoder == "MTF":
            mtf_mod.SavevarMTF_XYZ(signal, "w", "", "", normalized=1,
                                   path=slash, saveImage=True, TIME_STEPS=N)
            img = _read_encoded_png(os.path.join(tmp, "wmtf.png"))
            rec = mtf_mod.Reconstruct_MTF(img, d, 0)
        else:
            raise ValueError(encoder)
    rec = np.asarray(rec)  # (3, T) with T in {N-1, N}
    return rec


def _pad_to_N(arr: np.ndarray, N: int) -> np.ndarray:
    """Pad a (3, T<=N) reconstruction so per-axis plots line up with originals."""
    if arr.shape[1] == N:
        return arr
    pad = np.full((arr.shape[0], N - arr.shape[1]), np.nan)
    return np.hstack([arr, pad])


def plot_global(signal: np.ndarray, rec: np.ndarray, out_name: str, title: str) -> None:
    """Original vs reconstruction with all three channels overlaid."""
    rec = _pad_to_N(rec, signal.shape[0])
    palette = sns.color_palette("tab10", 6)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(signal[:, 0], label="Original x",      color=palette[0])
    ax.plot(rec[0],       label="Reconstructed x", color=palette[1], linestyle="--")
    ax.plot(signal[:, 1], label="Original y",      color=palette[2])
    ax.plot(rec[1],       label="Reconstructed y", color=palette[3], linestyle="--")
    ax.plot(signal[:, 2], label="Original z",      color=palette[4])
    ax.plot(rec[2],       label="Reconstructed z", color=palette[5], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Signal value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, ncol=1)
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(os.path.join(OUT, out_name), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_name}")


def plot_per_axis(signal: np.ndarray, rec: np.ndarray, axis: int,
                  out_name: str, title: str) -> None:
    rec = _pad_to_N(rec, signal.shape[0])
    axis_letter = "xyz"[axis]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(signal[:, axis], label="Original",      color="C0")
    ax.plot(rec[axis],       label="Reconstructed", color="C1", linestyle="--")
    ax.set_title(f"{title} — channel {axis_letter}")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Signal value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(os.path.join(OUT, out_name), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-rp", action="store_true",
                    help="regenerate only the RP synthetic, global and per-axis figures")
    args = ap.parse_args()

    # -------- Synthetic --------
    t = np.linspace(0.0, np.pi, N)
    synth = np.column_stack([np.cos(t), np.sin(t), np.sin(t) - np.cos(t)]).astype(np.float64)
    print("Synthetic reconstructions:")
    synth_specs = [
        ("RP",   "Comparison_all_wideRP.png",   "RP reconstruction (synthetic)"),
        ("GAF",  "Comparison_all_wideGAF.png",  "GASF reconstruction (synthetic)"),
        ("GADF", "Comparison_all_wideGADF.png", "GADF reconstruction (synthetic)"),
        ("MTF",  "Comparison_all_widemtf.png",  "MTF reconstruction (synthetic)"),
    ]
    if args.only_rp:
        synth_specs = [s for s in synth_specs if s[0] == "RP"]
    for enc, fname, title in synth_specs:
        rec = encode_decode(synth, enc)
        plot_global(synth, rec, fname, title)

    # -------- WISDM 1600x0 --------
    wisdm = np.load("/home/adrianogarcia/tfg/data/WISDM/numpies/train/windowed/0/1600/x.npy")[0]
    assert wisdm.shape == (N, 3)
    print("WISDM 1600x0 reconstructions (global):")
    encoders = [
        ("RP",   "rp",   "rp_sample_0_global.png",   "RP reconstruction (WISDM 1600x0)"),
        ("GAF",  "gaf",  "gaf_sample_0_global.png",  "GASF reconstruction (WISDM 1600x0)"),
        ("GADF", "gadf", "gadf_sample_0_global.png", "GADF reconstruction (WISDM 1600x0)"),
        ("MTF",  "mtf",  "mtf_sample_0_global.png",  "MTF reconstruction (WISDM 1600x0)"),
    ]
    if args.only_rp:
        encoders = [e for e in encoders if e[0] == "RP"]
    reconstructions = {}
    for enc, short, fname, title in encoders:
        rec = encode_decode(wisdm, enc)
        reconstructions[short] = rec
        plot_global(wisdm, rec, fname, title)

    # -------- WISDM per-axis (all four encoders so they're symmetric) --------
    print("WISDM 1600x0 reconstructions (per axis):")
    axis_specs = [("rp", "RP"), ("gaf", "GASF"),
                  ("gadf", "GADF"), ("mtf", "MTF")]
    if args.only_rp:
        axis_specs = [("rp", "RP")]
    for enc_short, title in axis_specs:
        rec = reconstructions[enc_short]
        for axis, letter in enumerate("xyz"):
            plot_per_axis(wisdm, rec, axis,
                          f"{enc_short}_sample_0_{letter}_comp.png",
                          f"{title} (WISDM 1600x0)")

    print("Done.")


if __name__ == "__main__":
    main()
