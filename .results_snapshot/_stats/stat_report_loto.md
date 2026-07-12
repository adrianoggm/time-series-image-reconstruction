# Statistical significance report (LOTO 3-fold pooled)

All paired tests run on the pooled $N=8172$ test windows from
folds [0, 1, 2] (per-fold sizes [2724, 2724, 2724]); each window is one
paired (Original, Reconstruction) observation so pooling preserves
the McNemar/Wilcoxon validity. Raw p-values are corrected with
Holm--Bonferroni inside each family at $\alpha=0.05$.

## 1. Per-fold accuracy summary (mean +/- std across 3 folds)

| Classifier | Original | RP | MTF | GASF | GADF |
|---|---|---|---|---|---|
| knn | 0.642±0.001 [0.641,0.644] | 0.686±0.002 [0.685,0.689] | 0.559±0.005 [0.552,0.566] | 0.642±0.001 [0.641,0.644] | 0.643±0.001 [0.641,0.645] |
| svc | 0.801±0.005 [0.795,0.805] | 0.694±0.007 [0.684,0.699] | 0.697±0.005 [0.692,0.704] | 0.801±0.005 [0.795,0.805] | 0.801±0.004 [0.796,0.805] |
| randomforest | 0.891±0.001 [0.890,0.892] | 0.706±0.006 [0.697,0.712] | 0.757±0.002 [0.755,0.760] | 0.891±0.002 [0.890,0.894] | 0.892±0.000 [0.891,0.892] |
| lstm | 0.869±0.031 [0.825,0.895] | 0.714±0.009 [0.701,0.722] | 0.757±0.016 [0.737,0.775] | 0.868±0.030 [0.826,0.894] | 0.869±0.031 [0.826,0.895] |
| gru | 0.923±0.004 [0.919,0.930] | 0.749±0.006 [0.741,0.756] | 0.765±0.006 [0.757,0.769] | 0.923±0.004 [0.919,0.929] | 0.923±0.005 [0.919,0.930] |
| cnn | 0.907±0.005 [0.901,0.913] | 0.747±0.026 [0.714,0.778] | 0.773±0.010 [0.762,0.787] | 0.907±0.005 [0.900,0.913] | 0.907±0.005 [0.901,0.912] |

## 2. McNemar Original vs reconstruction (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/Original-vs-RP | 1.493e-26 | 1.941e-25 | yes |
| knn/Original-vs-GAF | 5.050e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | 1.814e-01 | 1.000e+00 | no |
| knn/Original-vs-MTF | 2.562e-69 | 3.587e-68 | yes |
| svc/Original-vs-RP | 3.120e-109 | 4.993e-108 | yes |
| svc/Original-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| svc/Original-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| svc/Original-vs-MTF | 2.818e-89 | 4.227e-88 | yes |
| randomforest/Original-vs-RP | 3.249e-252 | 7.474e-251 | yes |
| randomforest/Original-vs-GAF | 9.035e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | 2.718e-173 | 4.893e-172 | yes |
| lstm/Original-vs-RP | 2.267e-185 | 4.535e-184 | yes |
| lstm/Original-vs-GAF | 8.137e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | 8.026e-01 | 1.000e+00 | no |
| lstm/Original-vs-MTF | 2.794e-129 | 4.750e-128 | yes |
| gru/Original-vs-RP | 1.859e-254 | 4.461e-253 | yes |
| gru/Original-vs-GAF | 4.497e-01 | 1.000e+00 | no |
| gru/Original-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-MTF | 3.008e-228 | 6.618e-227 | yes |
| cnn/Original-vs-RP | 2.509e-217 | 5.269e-216 | yes |
| cnn/Original-vs-GAF | 7.728e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | 7.518e-01 | 1.000e+00 | no |
| cnn/Original-vs-MTF | 3.979e-180 | 7.561e-179 | yes |

## 3. McNemar pairwise between encoders (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/RP-vs-GAF | 3.001e-26 | 3.602e-25 | yes |
| knn/RP-vs-GADF | 6.636e-26 | 7.300e-25 | yes |
| knn/RP-vs-MTF | 2.938e-132 | 6.464e-131 | yes |
| knn/GAF-vs-GADF | 6.056e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | 1.093e-69 | 1.530e-68 | yes |
| knn/GADF-vs-MTF | 3.053e-70 | 4.580e-69 | yes |
| svc/RP-vs-GAF | 1.515e-109 | 2.879e-108 | yes |
| svc/RP-vs-GADF | 2.856e-109 | 5.141e-108 | yes |
| svc/RP-vs-MTF | 5.187e-01 | 1.000e+00 | no |
| svc/GAF-vs-GADF | 8.597e-01 | 1.000e+00 | no |
| svc/GAF-vs-MTF | 3.837e-89 | 6.523e-88 | yes |
| svc/GADF-vs-MTF | 3.837e-89 | 6.523e-88 | yes |
| randomforest/RP-vs-GAF | 3.249e-252 | 1.072e-250 | yes |
| randomforest/RP-vs-GADF | 1.140e-252 | 3.875e-251 | yes |
| randomforest/RP-vs-MTF | 3.168e-29 | 4.119e-28 | yes |
| randomforest/GAF-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | 2.119e-172 | 4.874e-171 | yes |
| randomforest/GADF-vs-MTF | 2.872e-173 | 6.892e-172 | yes |
| lstm/RP-vs-GAF | 2.251e-185 | 6.078e-184 | yes |
| lstm/RP-vs-GADF | 5.870e-186 | 1.644e-184 | yes |
| lstm/RP-vs-MTF | 4.107e-19 | 4.107e-18 | yes |
| lstm/GAF-vs-GADF | 8.445e-01 | 1.000e+00 | no |
| lstm/GAF-vs-MTF | 1.531e-128 | 3.062e-127 | yes |
| lstm/GADF-vs-MTF | 1.844e-129 | 3.872e-128 | yes |
| gru/RP-vs-GAF | 2.100e-254 | 7.561e-253 | yes |
| gru/RP-vs-GADF | 3.013e-254 | 1.055e-252 | yes |
| gru/RP-vs-MTF | 1.668e-04 | 1.335e-03 | yes |
| gru/GAF-vs-GADF | 7.237e-01 | 1.000e+00 | no |
| gru/GAF-vs-MTF | 1.276e-227 | 3.955e-226 | yes |
| gru/GADF-vs-MTF | 6.881e-229 | 2.202e-227 | yes |
| cnn/RP-vs-GAF | 1.409e-217 | 4.227e-216 | yes |
| cnn/RP-vs-GADF | 2.509e-217 | 7.276e-216 | yes |
| cnn/RP-vs-MTF | 1.919e-08 | 1.727e-07 | yes |
| cnn/GAF-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| cnn/GAF-vs-MTF | 1.278e-180 | 3.323e-179 | yes |
| cnn/GADF-vs-MTF | 1.278e-180 | 3.323e-179 | yes |

## 4. Wilcoxon signed-rank on per-window MAE
| Comparison | effect r | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|---|
| MAE/RP-vs-GAF | +0.866 | <1e-300 | <1e-300 | yes |
| MAE/RP-vs-GADF | +0.866 | <1e-300 | <1e-300 | yes |
| MAE/RP-vs-MTF | -0.474 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-GADF | +0.457 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GADF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
