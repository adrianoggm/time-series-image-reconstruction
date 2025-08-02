#!/usr/bin/env python3
import numpy as np
import src.recurrence_plots as rec
import src.GAF as gaf
import src.MTF as mtf
import src.ts_plots as plots
import cv2
import argparse

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--image-type', type=str, default="RP",
                   help='Which encoding to generate/reconstruct: RP, GAF, or MTF')
    p.add_argument('--dim', type=int, default=0,
                   help='Which channel to plot (0=cos, 1=sin, 2=sin(cos))')
    args = p.parse_args()

    # 1) Generate synthetic time series w of shape (128, 3)
    N = 128
    t = np.linspace(0, 2 * np.pi, N)
    w = np.stack([
        np.cos(t),           # dim 0: cos(t)
        np.sin(t),           # dim 1: sin(t) (could be changed to sin(t) if desired)
        np.sin(t)- np.cos(t)    # dim 2: sin(cos(t))
    ], axis=1)
    print("Generated w with shape:", w.shape)  # should be (128, 3)

    # 2) Build dictionary of extremes for reconstruction
    maximos = [w[:, i].max() for i in range(w.shape[1])]
    minimos = [w[:, i].min() for i in range(w.shape[1])]
    dictionary = {0: [maximos, minimos]}  # only one "sample" at index 0
    a = 0  # index in dictionary
    w=w*-1 #invertir la señal para que sea positiva
    # 3) Create & load the encoded image, reconstruct, and plot
    dim = args.dim
    if args.image_type.upper() == "MTF":
        mtf.SavevarMTF_XYZ(w, 0, 0, "x", normalized=1, path="./", TIME_STEPS=N)
        img = cv2.imread("./0x0mtf.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("MTF image shape:", img.shape)
        rp = mtf.Reconstruct_MTF(img, dictionary, a)

    elif args.image_type.upper() == "RP":
        rec.SavevarRP_XYZ(w, 0, 0, "x", normalized=1, path="./", TIME_STEPS=N)
        img = cv2.imread("./0x0.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("RP image shape:", img.shape)
        rp, _ = rec.Reconstruct_RP(img, dictionary, a)

    elif args.image_type.upper() == "GAF":
        gaf.SavevarGAF_XYZ(w, 0, 0, "x", normalized=1, path="./", TIME_STEPS=N)
        img = cv2.imread("./0x0gasf.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("GAF image shape:", img.shape)
        rp = gaf.Reconstruct_GAF(img, dictionary, a)

    else:
        raise ValueError(f"Unknown image type: {args.image_type}")

    # 4) Plot original vs. reconstructed for the selected channel
    plots.plot_time_series(w, rp, dim)
    plots.plot_time_series_g(w, rp)

if __name__ == '__main__':
    main()
