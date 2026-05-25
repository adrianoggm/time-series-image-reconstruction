"""Quick sanity check on the GADF reconstructed time series."""
import numpy as np
from pathlib import Path

ROOT = Path.home() / "tfg/data/WISDM/tseries/GADF/sampling_loto/3-fold/fold-0/test"
X_all = np.load(ROOT / "X_all_gadf.npy")
print(f"X_all_gadf.npy shape: {X_all.shape}, dtype: {X_all.dtype}")
print(f"  min={X_all.min():.4f}  max={X_all.max():.4f}  mean={X_all.mean():.4f}")
print(f"  NaNs: {np.isnan(X_all).sum()}, Infs: {np.isinf(X_all).sum()}")

# load one sample and compare to the corresponding original window
ORIG = np.load(Path.home() / "tfg/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/validation_data.npy")
print(f"\nvalidation_data.npy (originals) shape: {ORIG.shape}")

# 1st sample
rec = X_all[0]
orig = ORIG[0]
print(f"\nSample 0:")
print(f"  rec  shape: {rec.shape}  dtype: {rec.dtype}")
print(f"  orig shape: {orig.shape}")

# rec layout is (3, N); orig is (N, 3). Align before comparing.
if rec.shape != orig.shape:
    rec_aligned = rec.T  # (N, 3)
else:
    rec_aligned = rec
print(f"  aligned rec shape: {rec_aligned.shape}")
print(f"  rec  range : [{rec_aligned.min():.3f}, {rec_aligned.max():.3f}]")
print(f"  orig range : [{orig.min():.3f}, {orig.max():.3f}]")
mae = float(np.mean(np.abs(rec_aligned - orig)))
rmse = float(np.sqrt(np.mean((rec_aligned - orig) ** 2)))
print(f"  MAE  = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")
