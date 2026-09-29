"""Build the two results figures of the main manuscript from stored artefacts.

Nothing is hardcoded: the curves come from results/_stats/robustness.json and
the intervals from results/_stats/stat_report_loso.json, the same files the
robustness and equivalence tables are generated from.

    python -m scripts._make_paper_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "results" / "_stats"
OUT = ROOT / "doc" / "images"

# Okabe-Ito, colourblind-safe and legible in greyscale print.
COLOUR = {"RP": "#0072B2", "GASF": "#E69F00", "GADF": "#009E73", "MTF": "#D55E00"}
MARKER = {"RP": "o", "GASF": "s", "GADF": "^", "MTF": "D"}
MARGIN = 0.01
CLASSIFIERS = [("knn", "KNN"), ("svc", "SVM"), ("randomforest", "RF"),
               ("lstm", "LSTM"), ("gru", "GRU"), ("cnn", "CNN")]
# JSON encoder key -> label used in the paper.
ENCODERS = [("RP", "RP"), ("GAF", "GASF"), ("GADF", "GADF"), ("MTF", "MTF")]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "figure.dpi": 200,
})


def robustness_curves() -> Path:
    """Mean NMAE against pixel-noise sigma, with subject-cluster 95% bands."""
    d = json.loads((STATS / "robustness.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    for key, label in ENCODERS:
        curve = d[label if label in d else key]["nmae_curve"]
        sig = np.array(sorted(float(s) for s in curve))
        mean = np.array([curve[f"{s}"]["mean"] for s in sig])
        lo = np.array([curve[f"{s}"]["ci95"][0] for s in sig])
        hi = np.array([curve[f"{s}"]["ci95"][1] for s in sig])
        ax.fill_between(sig, lo, hi, color=COLOUR[label], alpha=0.20, linewidth=0)
        ax.plot(sig, mean, color=COLOUR[label], marker=MARKER[label],
                markersize=4.5, linewidth=1.6, label=label)

    ax.set_yscale("log")
    ax.set_xlabel(r"Gaussian pixel-noise standard deviation $\sigma$ (out of 255)")
    ax.set_ylabel("Mean NMAE")
    ax.set_xticks([0, 2, 4, 8, 16])
    ax.grid(True, which="major", axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, ncol=4, loc="lower right", columnspacing=1.4)
    fig.tight_layout(pad=0.4)
    out = OUT / "robustness_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def equivalence_forest() -> Path:
    """Per-subject accuracy differences with simultaneous 95% intervals.

    Two panels because the scales differ by an order of magnitude: the primary
    RP/GASF/GADF family sits inside the margin, the MTF control family far
    outside it. A single axis would collapse the 18 primary intervals to a dot.
    """
    rows = json.loads((STATS / "stat_report_loso.json").read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in rows["subject_cluster_bootstrap_simultaneous_ci_accuracy"]}

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(6.4, 4.3), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # One row pitch for both panels: the left one sets the extent, the right one
    # centres its six rows inside it so the markers stay the same size apart.
    span = len(CLASSIFIERS) * 3 + 2 * 0.8
    ylim = (-span + 0.6, 0.8)

    def draw(ax, encoders, title, y0=0.0):
        labels, y = [], y0
        ticks, ticklabels = [], []
        for key, label in encoders:
            for clf_key, clf_label in CLASSIFIERS:
                r = rows[f"{clf_key}/Original-vs-{key}"]
                inside = r["equivalent_0.01"]
                ax.plot([r["ci_lo"], r["ci_hi"]], [y, y],
                        color=COLOUR[label], linewidth=1.5,
                        solid_capstyle="butt", zorder=3)
                ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], "|",
                        color=COLOUR[label], markersize=4, zorder=3)
                ax.plot(r["mean_delta_acc"], y, MARKER[label], color=COLOUR[label],
                        markersize=4.5, zorder=4,
                        markerfacecolor=COLOUR[label] if inside else "white")
                ticks.append(y)
                ticklabels.append(clf_label)
                y -= 1
            labels.append((y + len(CLASSIFIERS) / 2, label))
            y -= 0.8

        ax.axvspan(-MARGIN, MARGIN, color="0.88", zorder=0)
        ax.axvline(0, color="0.45", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=6)
        ax.tick_params(axis="y", length=0)
        for ypos, name in labels:
            ax.text(-0.30, ypos, name, transform=ax.get_yaxis_transform(),
                    ha="center", va="center", rotation=90, fontsize=8.5,
                    color=COLOUR[name], fontweight="bold")
        return labels

    draw(axl, ENCODERS[:3], "Primary family (18 comparisons)")
    axl.set_xlim(-0.0135, 0.0135)
    axl.set_xticks([-0.01, 0, 0.01])
    axl.set_xticklabels([r"$-0.01$", "0", r"$+0.01$"])

    # Centre the six control rows against the eighteen on the left.
    draw(axr, ENCODERS[3:], "MTF control family (6)",
         y0=(ylim[0] + ylim[1]) / 2 + (len(CLASSIFIERS) - 1) / 2)
    axr.set_xlim(-0.012, 0.155)
    axr.set_xticks([0, 0.05, 0.10, 0.15])

    fig.supxlabel(r"Per-subject $\Delta$accuracy (Original $-$ reconstruction)",
                  fontsize=9, y=0.02)
    fig.tight_layout(pad=0.4, w_pad=2.6, rect=(0, 0.035, 1, 1))
    out = OUT / "equivalence_forest.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (robustness_curves(), equivalence_forest()):
        print(f"WROTE {path}  ({path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
