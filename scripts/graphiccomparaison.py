#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

# Map channel indices to axis labels
CHANNEL_NAMES = ['x', 'y', 'z']

def plot_global(original, reconstructed, sample_index, plot_type, save_folder):
    t = np.linspace(0, 1, original.shape[1])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    orig_colors = ["C0", "C2", "C4"]
    rec_colors  = ["C1", "C3", "C5"]
    for i, name in enumerate(CHANNEL_NAMES):
        ax.plot(t,
                original[sample_index, :, i],
                label=f"Original {name}",
                color=orig_colors[i])
        ax.plot(t,
                reconstructed[sample_index, :, i],
                linestyle="--",
                label=f"{plot_type} {name}",
                color=rec_colors[i])
    ax.set_title(f"{plot_type} Global Comparison — Sample {sample_index}")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02,1), frameon=False)
    ax.grid(True)
    fig.tight_layout(rect=[0,0,0.78,1])
    os.makedirs(save_folder, exist_ok=True)
    out = os.path.join(save_folder, f"{plot_type.lower()}_sample_{sample_index}_global.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved global comparison: {out}")

def plot_channel(original, reconstructed, sample_index, channel, plot_type, save_folder):
    t = np.linspace(0, 1, original.shape[1])
    name = CHANNEL_NAMES[channel]

    # Original (single)
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot(t, original[sample_index,:,channel], color="C0")
    ax.set_title(f"Original — Sample {sample_index}, {name}")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    fig.tight_layout()
    out1 = os.path.join(save_folder, f"{plot_type.lower()}_sample_{sample_index}_{name}_orig.png")
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved original channel: {out1}")

    # Reconstruction (single)
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot(t, reconstructed[sample_index,:,channel], color="C1")
    ax.set_title(f"{plot_type} Reconstruction — Sample {sample_index}, {name}")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    fig.tight_layout()
    out2 = os.path.join(save_folder, f"{plot_type.lower()}_sample_{sample_index}_{name}_rec.png")
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reconstructed channel: {out2}")

    # Overlay comparison
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot(t, original[sample_index,:,channel], label="Original", color="C0")
    ax.plot(t, reconstructed[sample_index,:,channel], linestyle="--", label="Reconstruction", color="C1")
    ax.set_title(f"{plot_type} Comparison — Sample {sample_index}, {name}")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02,1), frameon=False)
    ax.grid(True)
    fig.tight_layout(rect=[0,0,0.78,1])
    out3 = os.path.join(save_folder, f"{plot_type.lower()}_sample_{sample_index}_{name}_comp.png")
    fig.savefig(out3, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison channel: {out3}")
def main():
    # Hardcoded paths
    dataset_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM"
    images_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/images"

    parser = argparse.ArgumentParser(description="Plot global and per-channel comparisons for RP, GAF, or MTF")
    parser.add_argument('--type', choices=['RP','GAF','MTF'], required=True, help='Encoding type')
    parser.add_argument('--sample-index', type=int, required=True, help='Sample index to plot')
    args = parser.parse_args()

    # Load original
    train_path = os.path.join(dataset_folder, 'tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train/training_data.npy')
    X_train = np.load(train_path)  # (n_samples, 129, 3)

    # Load reconstructed
    if args.type == 'RP':
        rec_path = os.path.join(dataset_folder, 'tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/X_all_rec.npy')
    elif args.type == 'GAF':
        rec_path = os.path.join(dataset_folder, 'tseries/GAF/sampling_loto/3-fold/fold-0/train/X_all_gaf.npy')
    else:
        rec_path = os.path.join(dataset_folder, 'tseries/MTF/sampling_loto/3-fold/fold-0/train/X_all_mtf.npy')
    X_rec = np.load(rec_path)  # assume already (n_samples, 129, 3)
    if args.type == 'GAF' or args.type == 'MTF':
        X_rec = np.transpose(X_rec, (0,2,1))
    # Plot global and per-channel
    plot_global(X_train, X_rec, args.sample_index, args.type, images_folder)
    for ch in range(3):
        plot_channel(X_train, X_rec, args.sample_index, ch, args.type, images_folder)

if __name__ == '__main__':
    main()
 