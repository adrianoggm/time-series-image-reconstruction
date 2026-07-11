# Statistical significance report (LOTO 3-fold pooled)

All paired tests run on the pooled $N=8172$ test windows from
folds [0, 1, 2] (per-fold sizes [2724, 2724, 2724]); each window is one
paired (Original, Reconstruction) observation so pooling preserves
the McNemar/Wilcoxon validity. Raw p-values are corrected with
Holm--Bonferroni inside each family at $\alpha=0.05$.

## 1. Per-fold accuracy summary (mean +/- std across 3 folds)

| Classifier | Original | RP | MTF | GASF | GADF |
|---|---|---|---|---|---|
| knn | 0.642±0.001 [0.641,0.644] | 0.686±0.002 [0.685,0.689] | 0.559±0.005 [0.552,0.566] | 0.642±0.001 [0.641,0.644] | 0.632±0.007 [0.624,0.641] |
| svc | 0.801±0.005 [0.795,0.805] | 0.694±0.007 [0.684,0.699] | 0.697±0.005 [0.692,0.704] | 0.801±0.005 [0.795,0.805] | 0.721±0.001 [0.720,0.722] |
| randomforest | 0.891±0.001 [0.890,0.892] | 0.706±0.006 [0.697,0.712] | 0.757±0.002 [0.755,0.760] | 0.891±0.002 [0.890,0.894] | 0.794±0.007 [0.784,0.800] |
| lstm | 0.869±0.031 [0.825,0.895] | 0.714±0.009 [0.701,0.722] | 0.757±0.016 [0.737,0.775] | 0.868±0.030 [0.826,0.894] | 0.789±0.011 [0.776,0.803] |
| gru | 0.923±0.004 [0.919,0.930] | 0.749±0.006 [0.741,0.756] | 0.765±0.006 [0.757,0.769] | 0.923±0.004 [0.919,0.929] | 0.845±0.023 [0.813,0.863] |
| cnn | 0.907±0.005 [0.901,0.913] | 0.747±0.026 [0.714,0.778] | 0.773±0.010 [0.762,0.787] | 0.907±0.005 [0.900,0.913] | 0.830±0.022 [0.800,0.852] |

## 2. McNemar Original vs reconstruction (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/Original-vs-RP | 1.493e-26 | 1.194e-25 | yes |
| knn/Original-vs-GAF | 5.050e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | 2.228e-03 | 1.559e-02 | yes |
| knn/Original-vs-MTF | 2.562e-69 | 2.562e-68 | yes |
| svc/Original-vs-RP | 3.120e-109 | 4.993e-108 | yes |
| svc/Original-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| svc/Original-vs-GADF | 3.452e-56 | 3.106e-55 | yes |
| svc/Original-vs-MTF | 2.818e-89 | 3.381e-88 | yes |
| randomforest/Original-vs-RP | 3.249e-252 | 7.474e-251 | yes |
| randomforest/Original-vs-GAF | 9.035e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | 3.198e-103 | 4.477e-102 | yes |
| randomforest/Original-vs-MTF | 2.718e-173 | 4.893e-172 | yes |
| lstm/Original-vs-RP | 2.267e-185 | 4.535e-184 | yes |
| lstm/Original-vs-GAF | 8.137e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | 1.733e-79 | 1.906e-78 | yes |
| lstm/Original-vs-MTF | 2.794e-129 | 4.750e-128 | yes |
| gru/Original-vs-RP | 1.859e-254 | 4.461e-253 | yes |
| gru/Original-vs-GAF | 4.497e-01 | 1.000e+00 | no |
| gru/Original-vs-GADF | 1.507e-103 | 2.261e-102 | yes |
| gru/Original-vs-MTF | 3.008e-228 | 6.618e-227 | yes |
| cnn/Original-vs-RP | 2.509e-217 | 5.269e-216 | yes |
| cnn/Original-vs-GAF | 7.728e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | 2.260e-92 | 2.939e-91 | yes |
| cnn/Original-vs-MTF | 3.979e-180 | 7.561e-179 | yes |

## 3. McNemar pairwise between encoders (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/RP-vs-GAF | 3.001e-26 | 3.001e-25 | yes |
| knn/RP-vs-GADF | 1.595e-38 | 2.233e-37 | yes |
| knn/RP-vs-MTF | 2.938e-132 | 8.520e-131 | yes |
| knn/GAF-vs-GADF | 1.513e-03 | 3.026e-03 | yes |
| knn/GAF-vs-MTF | 1.093e-69 | 2.185e-68 | yes |
| knn/GADF-vs-MTF | 1.763e-56 | 2.997e-55 | yes |
| svc/RP-vs-GAF | 1.515e-109 | 4.091e-108 | yes |
| svc/RP-vs-GADF | 1.776e-06 | 7.105e-06 | yes |
| svc/RP-vs-MTF | 5.187e-01 | 5.187e-01 | no |
| svc/GAF-vs-GADF | 3.917e-56 | 6.267e-55 | yes |
| svc/GAF-vs-MTF | 3.837e-89 | 8.825e-88 | yes |
| svc/GADF-vs-MTF | 5.096e-10 | 3.058e-09 | yes |
| randomforest/RP-vs-GAF | 3.249e-252 | 1.137e-250 | yes |
| randomforest/RP-vs-GADF | 5.952e-50 | 8.928e-49 | yes |
| randomforest/RP-vs-MTF | 3.168e-29 | 3.485e-28 | yes |
| randomforest/GAF-vs-GADF | 1.253e-102 | 3.258e-101 | yes |
| randomforest/GAF-vs-MTF | 2.119e-172 | 6.357e-171 | yes |
| randomforest/GADF-vs-MTF | 2.822e-14 | 2.257e-13 | yes |
| lstm/RP-vs-GAF | 2.251e-185 | 7.204e-184 | yes |
| lstm/RP-vs-GADF | 4.897e-38 | 6.366e-37 | yes |
| lstm/RP-vs-MTF | 4.107e-19 | 3.696e-18 | yes |
| lstm/GAF-vs-GADF | 5.230e-79 | 1.151e-77 | yes |
| lstm/GAF-vs-MTF | 1.531e-128 | 4.287e-127 | yes |
| lstm/GADF-vs-MTF | 1.488e-11 | 1.041e-10 | yes |
| gru/RP-vs-GAF | 2.100e-254 | 7.561e-253 | yes |
| gru/RP-vs-GADF | 2.325e-72 | 4.882e-71 | yes |
| gru/RP-vs-MTF | 1.668e-04 | 5.005e-04 | yes |
| gru/GAF-vs-GADF | 2.944e-102 | 7.360e-101 | yes |
| gru/GAF-vs-MTF | 1.276e-227 | 4.338e-226 | yes |
| gru/GADF-vs-MTF | 1.348e-57 | 2.561e-56 | yes |
| cnn/RP-vs-GAF | 1.409e-217 | 4.650e-216 | yes |
| cnn/RP-vs-GADF | 1.831e-57 | 3.296e-56 | yes |
| cnn/RP-vs-MTF | 1.919e-08 | 9.597e-08 | yes |
| cnn/GAF-vs-GADF | 2.260e-92 | 5.425e-91 | yes |
| cnn/GAF-vs-MTF | 1.278e-180 | 3.962e-179 | yes |
| cnn/GADF-vs-MTF | 7.731e-35 | 9.277e-34 | yes |

## 4. Wilcoxon signed-rank on per-window MAE
| Comparison | effect r | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|---|
| MAE/RP-vs-GAF | +0.866 | <1e-300 | <1e-300 | yes |
| MAE/RP-vs-GADF | +0.859 | <1e-300 | <1e-300 | yes |
| MAE/RP-vs-MTF | -0.474 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-GADF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GADF-vs-MTF | -0.847 | <1e-300 | <1e-300 | yes |

## 5. Steiger's z: GASF vs GADF Pearson
- r(orig, GASF) = 0.99988
- r(orig, GADF) = 0.98327
- r(GASF, GADF) = 0.96797
- z = 474.874, p = <1e-300
