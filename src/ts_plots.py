#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# Global seaborn style for scientific publication
sns.set(style="whitegrid", context="paper", font="serif", rc={
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.5,
    "grid.color": "0.8",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

def plot_time_series_g(original, reconstruction):
    w = original
    rp = reconstruction

    # Make the figure wider
    fig, ax = plt.subplots(figsize=(12, 4.5))
    palette = sns.color_palette("tab10", 6)
    ax.plot(w[:, 0], label='Original x', color=palette[0])
    ax.plot(rp[0], linestyle="--", label='Reconstructed x', color=palette[1])
    ax.plot(w[:, 1], label='Original y', color=palette[2])
    ax.plot(rp[1], linestyle="--", label='Reconstructed y', color=palette[3])
    ax.plot(w[:, 2], label='Original z', color=palette[4])
    ax.plot(rp[2], linestyle="--", label='Reconstructed z', color=palette[5])

    ax.set_title("Original vs. Reconstructed (all channels)")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Signal value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, ncol=1)
    ax.grid(True)
    # Adjust layout to make space for the legend on the right
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig("Comparison_all_wide.png", bbox_inches="tight")
    plt.close(fig)

def plot_time_series(original, reconstruction, dim):
    w = original
    rp = reconstruction

    # Single-channel original (wider)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(w[:, dim], color="C0")
    ax.set_title(f"Original — Channel {dim}")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Signal value")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"original_ch{dim}_wide.png", bbox_inches="tight")
    plt.close(fig)

    # Single-channel reconstruction (wider)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(rp[dim], color="C1")
    ax.set_title(f"Reconstruction — Channel {dim}")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Signal value")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"reconstruction_ch{dim}_wide.png", bbox_inches="tight")
    plt.close(fig)

    # Overlay comparison (wider)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(w[:, dim], label="Original", color="C0")
    ax.plot(rp[dim], linestyle="--", label="Reconstruction", color="C1")
    ax.set_title(f"Comparison — Channel {dim}")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Signal value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(f"comparison_ch{dim}_wide.png", bbox_inches="tight")
    plt.close(fig)

def plot_cm(labels, predictions):
    activity_names = ["Walking", "Jogging", "Stairs", "Sitting", "Standing"]
    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(cm, index=activity_names, columns=activity_names)

    # Slightly wider confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size":12})
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig("confusion_matrix_wide.png", bbox_inches="tight")
    plt.close(fig)

# Example usage (uncomment to run):
# w = np.random.randn(129,3)
# rp = [w[:,0]+0.1, w[:,1]+0.2, w[:,2]+0.3]
# plot_time_series_g(w, rp)
# plot_time_series(w, rp, dim=0)
# plot_cm(np.random.randint(0,5,100), np.random.randint(0,5,100))
