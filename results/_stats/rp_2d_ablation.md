# RP 2-D decoder ablation (LOTO fold-0, 100 windows, rho=0.60)

| Method | MAE | MedianMAE | MSE | Pearson | E_RP | E_overlap | EVR2 | Stress | Time_ms |
|---|---|---|---|---|---|---|---|---|---|
| RP-Jaccard-1D | 2.0470 | 1.9414 | 11.6932 | 0.9153 | 1.6373 | N/A | 0.6781 | 0.2299 | 569.3184 |
| RP-GEO-1D | 1.8797 | 1.7597 | 11.3251 | 0.9147 | 1.2059 | N/A | 0.2493 | 0.4988 | 547.8830 |
| RP-GEO-2D | 2.8825 | 2.1051 | 30.9100 | 0.7624 | 1.3719 | 1.4057 | 0.2493 | 0.4988 | 442.9367 |
| RP-DIRECT-2D | 2.2646 | 1.6004 | 22.9165 | 0.8108 | 1.2972 | 0.6909 | 0.2181 | 0.6203 | 386.9596 |
| RP-DIRECT-2D-S | 1.0472 | 0.9495 | 4.4919 | 0.9586 | 1.0849 | 0.6909 | 0.2181 | 0.6203 | 960.9336 |

> E_overlap is the min-overlap-orientation residual (source-level, sign-agnostic — identical within a source); Time_ms excludes this diagnostic recomputation.
