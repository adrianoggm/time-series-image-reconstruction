"""Regenerate the RP reconstruction aggregate X_all_rec.npy at a chosen rho.

Backs up the current (rho=0.75) array to X_all_rec_rho075.npy the first time, then
overwrites X_all_rec.npy with the rho-reconstructed version so the downstream
classifiers / stats / error tables pick up the new operating point. Reconstructs
from the existing PNGs (rho does NOT change the encoded images) using the
PRODUCTION decoder (method="direct2ds": metadata-free, closed-form orientation),
so its output is byte-comparable with what scripts.run_pipeline writes. Note rho
is inert for that decoder — it only affects the legacy graph-geodesic path.

    python -m scripts.regen_rp_recon --rho 0.60 --samplings loto,loso --folds 0,1,2
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from src.config import DEFAULT_DATA_DIR
from scripts.search_rp_rho import _align, _minmax, _recon_window

BACKUP_NAME = "X_all_rec_rho075.npy"


def regen_one(data_dir, sampling, fold, n_folds, percentile):
    base = (f"{data_dir}tseries/recurrence_plot/sampling_{sampling}/"
            f"{n_folds}-fold/fold-{fold}/test/")
    plots = (f"{data_dir}plots/recurrence_plot/sampling_{sampling}/"
             f"{n_folds}-fold/fold-{fold}/test/")

    orig = np.load(base + "validation_data.npy")
    y_val = np.load(base + "y_validation_data.npy")
    sj_val = np.load(base + "sj_validation_data.npy")
    n = len(orig)
    tasks = [(i, f"{plots}{int(np.argmax(y_val[i]))}/{sj_val[i][0]}x{i}.png", _minmax(orig[i]))
             for i in range(n)]

    # Back up the incumbent reconstruction exactly once (never clobber the backup).
    xall = base + "X_all_rec.npy"
    backup = base + BACKUP_NAME
    import os
    if os.path.exists(xall) and not os.path.exists(backup):
        shutil.copy2(xall, backup)
        print(f"  backed up rho=0.75 -> {backup}")

    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(_recon_window, i, p, mm, percentile, "direct2ds") for (i, p, mm) in tasks]
        res = [f.result() for f in tqdm(futs, total=n, desc=f"{sampling} fold-{fold}")]
    res.sort(key=lambda t: t[0])
    rec = np.array([r for _, r in res])          # (N, C, T) — pipeline convention
    np.save(xall, rec)

    mae = float(np.abs(orig - _align(rec, orig.shape)).mean())
    print(f"  {sampling} fold-{fold}: saved {rec.shape}, mean MAE = {mae:.4f}")
    return mae


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rho", type=float, default=0.60)
    ap.add_argument("--samplings", default="loto,loso")
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    data_dir = args.data_dir if args.data_dir.endswith(("/", "\\")) else args.data_dir + "/"
    samplings = [s.strip() for s in args.samplings.split(",") if s.strip()]
    folds = [int(f) for f in args.folds.split(",") if f.strip() != ""]
    percentile = args.rho * 100

    print(f"Regenerating X_all_rec.npy at rho={args.rho} for {samplings} x folds {folds}")
    for s in samplings:
        for f in folds:
            regen_one(data_dir, s, f, args.n_folds, percentile)
    print("Done.")


if __name__ == "__main__":
    main()
