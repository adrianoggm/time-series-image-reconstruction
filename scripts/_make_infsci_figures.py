"""Figure 2 of the Information Sciences manuscript, from stored artefacts (#97).

(a) 8-bit NMAE against Gaussian noise on the stored image (results/_stats/robustness_multi.json,
    T = 128, mean over the seven datasets); (b) 8-bit NMAE against the sampling rate and
    (c) median sigma2/sigma1 of the RP states against the sampling rate (results/_stats/sampling_rate.json,
    PTB-XL, T = 128). Only the near-lossless representations are drawn: the others stay at
    0.09-0.34 throughout and are reported in the text. Palette: the dataviz reference categorical
    slots 1-4 (validated; the two low-contrast slots get distinct markers and direct labels).

    python -m scripts._make_infsci_figures
Writes doc/infsci/images/robustness_sampling.pdf.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = (("RP", "Signed RP (ours)", "#2a78d6", "o"), ("GASF", "GASF", "#eb6834", "s"),
          ("GADF", "GADF", "#1baf7a", "^"), ("XIRP", "XIRP", "#eda100", "D"))
INK, MUTED = "#0b0b0b", "#52514e"
OUT = Path("doc/infsci/images/robustness_sampling.pdf")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    rob = json.loads(Path("results/_stats/robustness_multi.json").read_text())
    sr = json.loads(Path("results/_stats/sampling_rate.json").read_text())
    sigmas = [float(s) for s in rob["_design"]["sigmas"]]
    blocks = [k for k in rob if k.endswith("/L128")]
    rates = [500, 250, 100, 50]

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    a, b, c = axes
    for key, label, color, marker in SERIES:
        y = [np.mean([rob[k][key]["nmae"][str(s)]["mean"] for k in blocks]) for s in sigmas]
        a.plot(sigmas, y, color=color, marker=marker, markersize=4, linewidth=1.6, label=label)
        if key != "GADF":                         # GASF and GADF end on the same point: one shared label
            a.annotate("GASF, GADF" if key == "GASF" else label, (sigmas[-1], y[-1]), xytext=(4, 0),
                       textcoords="offset points", fontsize=6.5, color=INK, va="center")
        z = [sr[f"{r}Hz/L128"][f"{key}/png8"]["nmae"]["mean"] for r in rates]
        b.plot(rates, z, color=color, marker=marker, markersize=4, linewidth=1.6, label=label)
    col = [sr[f"{r}Hz/L128"]["collinearity_median"] for r in rates]
    c.plot(rates, col, color=SERIES[0][2], marker="o", markersize=4, linewidth=1.6)
    c.axhline(0.1, color=MUTED, linewidth=0.8, linestyle="--")
    c.annotate(r"$\sigma_2/\sigma_1=0.1$", (50, 0.1), xytext=(0, 3), textcoords="offset points", fontsize=6.5,
               color=MUTED, ha="left")

    a.set(xlabel="Noise $\\sigma$ (intensity levels)", ylabel="8-bit NMAE", title="(a) Noise on the stored image")
    a.set_xlim(-0.5, 21)
    for ax in (b, c):
        ax.set_xscale("log")
        ax.set_xticks(rates)
        ax.set_xticklabels([str(r) for r in rates])
        ax.set_xlabel("Sampling rate (Hz)")
    b.set(ylabel="8-bit NMAE", title="(b) Sampling rate, PTB-XL")
    c.set(ylabel=r"Median $\sigma_2/\sigma_1$ of RP states", title="(c) State conditioning, PTB-XL")
    b.legend(fontsize=6.5, frameon=False, loc="upper left")
    for ax in axes:
        style(ax)
        ax.title.set_fontsize(8)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
