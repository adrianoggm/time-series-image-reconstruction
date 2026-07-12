# RP 2-D decoder ablation (LOTO fold-0, 100 windows, rho=0.60, scale=rp)

| Method | MAE | MedianMAE | MSE | Pearson | E_RP | E_overlap | EVR2 | Stress | Time_ms |
|---|---|---|---|---|---|---|---|---|---|
| RP-Jaccard-1D | 2.1765 | 1.8946 | 13.5708 | 0.9180 | 1.1202 | N/A | 0.7301 | 0.2046 | 556.8817 |
| RP-GEO-1D | 1.0238 | 0.6557 | 4.1679 | 0.9699 | 0.6623 | N/A | 0.9215 | 0.0163 | 558.1569 |
| RP-GEO-2D | 2.0528 | 1.1000 | 25.0059 | 0.8071 | 0.8912 | 0.0895 | 0.9215 | 0.0163 | 463.1225 |
| RP-DIRECT-2D | 2.0326 | 0.9765 | 26.6167 | 0.7939 | 0.9300 | 0.0123 | 0.9471 | 0.0072 | 405.7373 |
| RP-DIRECT-2D-S | 0.0103 | 0.0070 | 0.0004 | 1.0000 | 0.0633 | 0.0123 | 0.9471 | 0.0072 | 945.4530 |

> E_overlap is the min-overlap-orientation residual (source-level, sign-agnostic — identical within a source); Time_ms excludes this diagnostic recomputation.
