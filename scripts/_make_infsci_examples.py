"""Encoded images and reconstructed series, one window per dataset, for the Information Sciences manuscript.

One channel of the first benchmark window (results/_benchmark/<dataset>_L128.npz) of
each of the seven datasets. Both figures use the same window and channel, so every
column of the image collage corresponds to one row of the reconstruction figure.

* encoding_examples.pdf: the stored 8-bit image of every encoding (binary RP at
  recurrence rate 0.75, unsigned RP, signed RP, GASF, GADF, MTF, XIRP), shown in grey
  levels exactly as stored (0 black, 255 white).
* reconstruction_examples.pdf: original and decoded series. Left column: the geometric
  decoder on the signed and on the unsigned RP; right column: Hirata on the unsigned RP
  and MTF. GASF, GADF and XIRP overlap the original at this scale and are left out.
  Palettes validated per panel with the dataviz validator (blue/red, magenta/green;
  magenta gets a direct label for contrast).

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
LEFT = (("RP", "Signed RP (ours)", "#2a78d6", "-"), ("RP_UNSIGNED", "Unsigned RP (ours)", "#e34948", "--"))
RIGHT = (("RP_HIRATA", "Hirata, unsigned RP", "#e87ba4", "-"), ("MTF", "MTF", "#008300", "-"))
INK, MUTED = "#0b0b0b", "#52514e"
IMAGES = Path("doc/infsci/images")


def windows():
    from scripts._benchmark import DATASETS as CFG
    from src.datasets import get, make_windows

    out = {}
    for name, _ in DATASETS:
        idx = int(np.load(Path("results/_benchmark") / f"{name}_L128.npz")["window_index"][0])
        out[name] = make_windows(get(name, **CFG[name]), 128, dtype=np.float32).X[idx][:, 0].astype(float)
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


def encoding_figure(xs):
    fig, axes = plt.subplots(len(ENCODINGS), len(DATASETS), figsize=(7.2, 7.6), constrained_layout=True)
    for col, (name, _) in enumerate(DATASETS):
        x = xs[name]
        mm = (float(x.max()), float(x.min()))
        for row, (key, label) in enumerate(ENCODINGS):
            ax = axes[row, col]
            ax.imshow(stored_image(key, x, mm), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
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
    from src.representations import REPRESENTATIONS

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(7.2, 9.4), sharex=True, constrained_layout=True)
    for row, (name, title) in enumerate(DATASETS):
        x = xs[name]
        mm = (float(x.max()), float(x.min()))
        for col, series in enumerate((LEFT, RIGHT)):
            ax = axes[row, col]
            ax.plot(x, color=INK, linewidth=1.8, alpha=0.35, label="Original")
            for key, label, color, style in series:
                rep = REPRESENTATIONS[key]
                r = rep.decode(rep.from_uint8(rep.to_uint8(rep.encode(x, mm))), mm)
                ax.plot(r, color=color, linewidth=1.1, linestyle=style, label=label)
            ax.set_title(title, fontsize=7.5, loc="left", color=INK, pad=16 if row == 0 else 4)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_color(MUTED)
            ax.spines["bottom"].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=6)
            if row == 0:
                ax.legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
                          borderaxespad=0.2, handlelength=2.2)        # above the data, below the row title
    for ax in axes[-1]:
        ax.set_xlabel("Sample", color=INK)
    save(fig, "reconstruction_examples")


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    xs = windows()
    encoding_figure(xs)
    reconstruction_figure(xs)


if __name__ == "__main__":
    main()
