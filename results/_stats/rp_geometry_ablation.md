# RP geometry ablation (LOTO fold-0, 100 windows)

The adopted, unsigned and legacy-1D arms use metadata-free diagonal centring; the signal-range arm is the explicit incorrect-inversion control. All 2-D arms use closed-form orientation.

| Method | MAE | MedianMAE | MSE | Pearson | E_RP | E_overlap | EVR2 | Stress | Time_ms |
|---|---|---|---|---|---|---|---|---|---|
| Adopted metadata-free + signed | 0.0107 | 0.0076 | 0.0004 | 1.0000 | 0.0680 | 0.1469 | 0.9467 | 0.0079 | 358.4028 |
| Incorrect signal-range inversion | 1.6912 | 1.3660 | 10.2286 | 0.9309 | 1.0792 | 0.6885 | 0.2185 | 0.6301 | 359.3128 |
| Without signed selection | 2.9398 | 1.6498 | 50.8930 | 0.7451 | 1.0146 | 0.1469 | 0.9467 | 0.0079 | 80.2665 |
| Legacy Jaccard 1-D | 1.9352 | 1.6917 | 10.2978 | 0.9359 | 0.6729 | N/A | 0.7605 | 0.1813 | 253.9196 |

> E_overlap is the min-overlap-orientation residual (source-level, sign-agnostic — identical within a source); Time_ms excludes this diagnostic recomputation.
