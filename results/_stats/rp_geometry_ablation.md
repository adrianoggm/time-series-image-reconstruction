# RP geometry ablation (LOTO fold-0, 100 windows)

The adopted, unsigned and legacy-1D arms use metadata-free diagonal centring; the signal-range arm is the explicit incorrect-inversion control. All 2-D arms use closed-form orientation.

| Method | MAE | MedianMAE | MSE | Pearson | E_RP | E_overlap | EVR2 | Stress | Time_ms |
|---|---|---|---|---|---|---|---|---|---|
| Adopted metadata-free + signed | 0.0095 | 0.0078 | 0.0003 | 1.0000 | 0.0652 | 0.1424 | 0.9439 | 0.0084 | 337.4787 |
| Incorrect signal-range inversion | 1.2586 | 1.0572 | 7.0120 | 0.9406 | 1.0709 | 0.6909 | 0.2181 | 0.6203 | 334.5873 |
| Without signed selection | 2.0332 | 1.2765 | 27.7880 | 0.7895 | 0.8801 | 0.1424 | 0.9439 | 0.0084 | 76.9606 |
| Legacy Jaccard 1-D | 2.1596 | 1.8567 | 13.4026 | 0.9187 | 0.7110 | N/A | 0.7287 | 0.2053 | 240.4598 |

> E_overlap is the min-overlap-orientation residual (source-level, sign-agnostic — identical within a source); Time_ms excludes this diagnostic recomputation.
