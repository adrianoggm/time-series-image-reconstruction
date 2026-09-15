#!/usr/bin/env bash
# Read-only probe: is the RP float-operator cycle residual consistent with a
# float-exact reconstruction? Decodes 3 windows straight from the native float
# operator (no PNG), then re-encodes and compares.
set -uo pipefail
cd /home/adrianogarcia/tfg
export OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
.venv/bin/python - <<'PY'
import numpy as np
from src.recurrence_plots import varRP2
from src.rp_decoders import decode_window
from src.GAF import normalize_to_zero_one

base = "data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/"
X = np.load(base + "validation_data.npy")
for wi in (1751, 48, 2632):
    win = X[wi].astype(float)
    mats = [varRP2(win[:, c], TIME_STEPS=128) for c in range(3)]
    # float path: the encoder's per-channel min-max map WITHOUT 8-bit rounding
    img = np.stack([normalize_to_zero_one(m) * 255 for m in mats], axis=-1)
    mm = ([float(win[:, c].max()) for c in range(3)],
          [float(win[:, c].min()) for c in range(3)])
    rec, _ = decode_window(img, mm, source="direct", rp_range="diagonal")
    rng = win.max(0) - win.min(0)
    nmae = float((np.abs(rec - win).mean(0) / rng).mean())
    num = sum(float(np.linalg.norm(varRP2(rec[:, c], TIME_STEPS=128) - mats[c])) for c in range(3))
    den = sum(float(np.linalg.norm(m)) for m in mats)
    print(f"win {wi}: NMAE={nmae:.3e}  cycle={num/den:.6f}")
    # per-channel detail
    for c in range(3):
        d = float(np.abs(rec[:, c] - win[:, c]).max())
        cy = float(np.linalg.norm(varRP2(rec[:, c], TIME_STEPS=128) - mats[c]) / np.linalg.norm(mats[c]))
        print(f"   ch{c}: max|dx|={d:.3e}  cycle={cy:.6f}")
PY
