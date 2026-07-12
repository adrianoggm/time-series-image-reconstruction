"""Validate the scaling anomaly: RP images are normalised on save by the RP
matrix's own range but decoded with the SIGNAL's amplitude range, which
affinely distorts the distances handed to MDS (R_obs = a*R_true + b, b != 0).

For the `direct` decoder, compare three input-decode conditions on the same
LOTO fold-0 windows:
  signal-range : np.interp(img,(0,255),(x_min,x_max))    <- current (wrong)
  RP-range     : np.interp(img,(0,255),(R_min,R_max))    <- proposed fix (keeps PNG quant)
  RP-true      : varRP2(orig) directly                   <- upper bound (no quant)

Prediction: RP-range/RP-true recover EVR2 -> ~1, Stress -> ~0 and lower MAE.
The OUTPUT rescale still uses signal range (the reconstruction is an amplitude);
only the INPUT distance-matrix scaling is corrected.

    python -m scripts._check_rp_scale --n 10
"""
import argparse

import cv2
import numpy as np

from src.config import DEFAULT_DATA_DIR
from src.recurrence_plots import varRP2
from src.rp_decoders import decode_channel, distance_matrix, embed_2d, evr2, mds_stress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    base = f"{DEFAULT_DATA_DIR}tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/"
    plots = f"{DEFAULT_DATA_DIR}plots/recurrence_plot/sampling_loto/3-fold/fold-0/test/"
    orig = np.load(base + "validation_data.npy")
    y_val = np.load(base + "y_validation_data.npy")
    sj_val = np.load(base + "sj_validation_data.npy")
    m = min(args.n, len(orig))

    conds = ("signal-range", "RP-range", "RP-true")
    acc = {c: dict(evr2=[], stress=[], mae=[]) for c in conds}

    for i in range(m):
        label = int(np.argmax(y_val[i]))
        sj = sj_val[i][0]
        img = cv2.imread(f"{plots}{label}/{sj}x{i}.png")
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for c in range(3):
            xc = orig[i][:, c].astype(float)
            smin, smax = float(xc.min()), float(xc.max())
            Rc_true = varRP2(xc, TIME_STEPS=len(xc))          # (N-1,N-1) signed, physical
            nz = Rc_true[Rc_true != 0]
            Rmin, Rmax = float(nz.min()), float(Rc_true.max())  # NormalizeMatrix convention

            px = img[:, :, c].astype(float)
            candidates = {
                "signal-range": np.interp(px, (0, 255), (smin, smax)),
                "RP-range": np.interp(px, (0, 255), (Rmin, Rmax)),
                "RP-true": Rc_true,
            }
            for cond, R_obs in candidates.items():
                dm = distance_matrix(R_obs, source="direct")
                Y, pos = embed_2d(dm)
                acc[cond]["evr2"].append(evr2(pos))
                acc[cond]["stress"].append(mds_stress(dm, Y))
                # OUTPUT rescale uses signal range (correct); only INPUT scaling varies.
                rec = decode_channel(R_obs, (smax, smin), source="direct", dims=2,
                                     recover="overlap", signed=True)
                acc[cond]["mae"].append(float(np.mean(np.abs(xc - rec))))

    print(f"RP input-scaling check (direct decoder, LOTO fold-0, {m} windows)\n")
    print(f"{'condition':14} {'EVR2':>8} {'Stress':>8} {'MAE':>8}")
    print("-" * 42)
    for cond in conds:
        a = acc[cond]
        print(f"{cond:14} {np.mean(a['evr2']):8.4f} {np.mean(a['stress']):8.4f} {np.mean(a['mae']):8.4f}")


if __name__ == "__main__":
    main()
