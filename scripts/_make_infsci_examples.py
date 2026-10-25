"""Encoded images and reconstructed series, one window per dataset, for the Information Sciences manuscript.

One channel of the first non-flat benchmark window (results/_benchmark/<dataset>_L128.npz) of
each of the seven datasets. Both figures use the same window and channel, so every
column of the image collage corresponds to one row of the reconstruction figure.

* encoding_examples.pdf: the stored 8-bit image of every encoding (binary RP at
  recurrence rate 0.75, unsigned RP, signed RP, GASF, GADF, MTF, XIRP), shown in grey
  levels exactly as stored (0 black, 255 white).
* reconstruction_examples.pdf: one panel per dataset and decoder (geometric decoder on the
  signed and on the unsigned RP, Hirata on the unsigned RP, MTF): the original as a grey
  band, the reconstruction on top and the panel's NMAE. GASF, GADF and XIRP overlap the
  original at this scale and are left out. Each colour is alone in its panel against grey.

    python -m scripts._make_infsci_examples        (needs the datasets: run in WSL)
Writes doc/infsci/images/{encoding,reconstruction}_examples.{pdf,png}.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATASETS = (("wisdm", "WISDM (accelerometer, x)"), ("pamap2", "PAMAP2 (hand accelerometer, x)"),
            ("esc50", "ESC-50 (audio)"), ("ptbxl", "PTB-XL (ECG, lead I)"),
            ("swarm", "Swarm (magnetometer, N)"), ("stead", "STEAD (seismogram, E)"),
            ("emg_gestures", "EMG (channel 1)"))
SHORT = {"wisdm": "WISDM", "pamap2": "PAMAP2", "esc50": "ESC-50", "ptbxl": "PTB-XL", "swarm": "Swarm",
         "stead": "STEAD", "emg_gestures": "EMG"}
ENCODINGS = (("BINARY", "Binary RP"), ("RP_UNSIGNED", "Unsigned RP"), ("RP", "Signed RP"), ("GASF", "GASF"),
             ("GADF", "GADF"), ("MTF", "MTF"), ("XIRP", "XIRP"))
COLUMNS = (("RP", "Signed RP, geometric (ours)", "#2a78d6"), ("RP_UNSIGNED", "Unsigned RP, geometric (ours)", "#e34948"),
           ("RP_HIRATA", "Unsigned RP, Hirata", "#e87ba4"), ("MTF", "MTF representative", "#008300"))
ORIGINAL = "#bdbdbd"          # the original is a light band; the reconstruction is drawn on top of it
INK, MUTED = "#0b0b0b", "#52514e"
IMAGES = Path("doc/infsci/images")


def windows():
    from scripts._benchmark import DATASETS as CFG
    from src.datasets import get, make_windows

    out = {}
    for name, _ in DATASETS:
        X = make_windows(get(name, **CFG[name]), 128, dtype=np.float32).X
        # first non-flat benchmark window whose first channel is not flat (ESC-50 has silent windows)
        idx = next(int(i) for i in np.load(Path("results/_benchmark") / f"{name}_L128.npz")["window_index"]
                   if np.ptp(X[i][:, 0]) > 1e-12)
        out[name] = X[idx].astype(float)                  # (T, C): the collage uses up to three channels
    return out


def stored_image(key, x, mm):
    from src.representations import REPRESENTATIONS
    from src.rp_hirata import binary_rp, eps_for_rate

    if key == "BINARY":                         # thresholded distances, stored losslessly as 0/255
        rep = REPRESENTATIONS["RP"]
        D = np.abs(rep.encode(x, mm))
        return 255 * binary_rp(D, eps_for_rate(D, 0.75)).astype(np.uint8)
    rep = REPRESENTATIONS[key]
    return rep.to_uint8(rep.encode(x, mm))


def save(fig, stem):
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / f"{stem}.pdf")
    fig.savefig(IMAGES / f"{stem}.png", dpi=150)
    print(IMAGES / f"{stem}.pdf")


def composite(key, window):
    """Stored images of the first three channels as R, G and B (each with its own calibration); a
    single-channel dataset stays grey. A flat channel contributes a black plane."""
    planes = []
    for c in range(min(3, window.shape[1])):
        x = window[:, c]
        mm = (float(x.max()), float(x.min()))
        planes.append(stored_image(key, x, mm) if np.ptp(x) > 1e-12 else None)
    shape = next(p.shape for p in planes if p is not None)
    planes = [np.zeros(shape, np.uint8) if p is None else p for p in planes]
    return planes[0] if len(planes) == 1 else np.stack(planes + [planes[0] * 0] * (3 - len(planes)), axis=-1)


def encoding_figure(xs):
    fig, axes = plt.subplots(len(ENCODINGS), len(DATASETS), figsize=(7.2, 7.6), constrained_layout=True)
    for col, (name, _) in enumerate(DATASETS):
        for row, (key, label) in enumerate(ENCODINGS):
            ax = axes[row, col]
            img = composite(key, xs[name])
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(MUTED)
                spine.set_linewidth(0.4)
            if row == 0:
                ax.set_title(SHORT[name], fontsize=7.5, color=INK, pad=3)
            if col == 0:
                ax.set_ylabel(label, fontsize=7.5, color=INK)
    save(fig, "encoding_examples")


def reconstruction_figure(xs):
    """One panel per dataset (row) and decoder (column): the original as a grey band, the reconstruction
    on top in the column's colour, and the panel's NMAE, so each technique is read on its own."""
    from src.representations import REPRESENTATIONS

    fig, axes = plt.subplots(len(DATASETS), len(COLUMNS), figsize=(7.2, 9.0), sharex=True, sharey="row",
                             constrained_layout=True)
    for row, (name, _) in enumerate(DATASETS):
        x = xs[name][:, 0]                                  # the red channel of the collage
        mm = (float(x.max()), float(x.min()))
        for col, (key, label, color) in enumerate(COLUMNS):
            ax = axes[row, col]
            rep = REPRESENTATIONS[key]
            r = rep.decode(rep.from_uint8(rep.to_uint8(rep.encode(x, mm))), mm)
            nmae = float(np.mean(np.abs(r - x)) / np.ptp(x))
            ax.plot(x, color=ORIGINAL, linewidth=2.4, solid_capstyle="round")
            ax.plot(r, color=color, linewidth=0.9)
            ax.text(1.0, 1.02, f"NMAE {nmae:.3f}", transform=ax.transAxes, fontsize=6, color=MUTED,
                    ha="right", va="bottom")                # above the frame, never over the data
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_color(MUTED)
            ax.spines["bottom"].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=5.5)
            if row == 0:
                ax.set_title(label, fontsize=7, color=INK, pad=12)
            if col == 0:
                ax.set_ylabel(SHORT[name], fontsize=7.5, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("Sample", color=INK, fontsize=7)
    handles = [plt.Line2D([], [], color=ORIGINAL, linewidth=2.4, label="Original"),
               plt.Line2D([], [], color=INK, linewidth=0.9, label="Reconstruction (colour of the column)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7)
    save(fig, "reconstruction_examples")


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    xs = windows()
    encoding_figure(xs)
    reconstruction_figure(xs)


if __name__ == "__main__":
    main()
