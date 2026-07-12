# Statistical significance report (LOTO 3-fold pooled)

All paired tests run on the pooled $N=8172$ test windows from
folds [0, 1, 2] (per-fold sizes [2724, 2724, 2724]); each window is one
paired (Original, Reconstruction) observation so pooling preserves
the McNemar/Wilcoxon validity. Raw p-values are corrected with
Holm--Bonferroni inside each family at $\alpha=0.05$.

## 1. Per-fold accuracy summary (mean +/- std across 3 folds)

| Classifier | Original | RP | MTF | GASF | GADF |
|---|---|---|---|---|---|
| knn | 0.642±0.001 [0.641,0.644] | 0.642±0.002 [0.641,0.644] | 0.559±0.005 [0.552,0.566] | 0.642±0.001 [0.641,0.644] | 0.643±0.001 [0.641,0.645] |
| svc | 0.801±0.005 [0.795,0.805] | 0.802±0.005 [0.796,0.806] | 0.697±0.005 [0.692,0.704] | 0.801±0.005 [0.795,0.805] | 0.801±0.004 [0.796,0.805] |
| randomforest | 0.891±0.001 [0.890,0.892] | 0.892±0.001 [0.891,0.894] | 0.757±0.002 [0.755,0.760] | 0.891±0.002 [0.890,0.894] | 0.892±0.000 [0.891,0.892] |
| lstm | 0.869±0.031 [0.825,0.895] | 0.869±0.031 [0.826,0.894] | 0.757±0.016 [0.737,0.775] | 0.868±0.030 [0.826,0.894] | 0.869±0.031 [0.826,0.895] |
| gru | 0.923±0.004 [0.919,0.930] | 0.923±0.005 [0.919,0.930] | 0.765±0.006 [0.757,0.769] | 0.923±0.004 [0.919,0.929] | 0.923±0.005 [0.919,0.930] |
| cnn | 0.901±0.006 [0.894,0.909] | 0.901±0.006 [0.893,0.909] | 0.758±0.020 [0.731,0.778] | 0.901±0.007 [0.892,0.909] | 0.901±0.007 [0.893,0.909] |

## 2. McNemar Original vs reconstruction (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/Original-vs-RP | 5.050e-01 | 1.000e+00 | no |
| knn/Original-vs-GAF | 5.050e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | 1.814e-01 | 1.000e+00 | no |
| knn/Original-vs-MTF | 2.562e-69 | 4.867e-68 | yes |
| svc/Original-vs-RP | 4.042e-01 | 1.000e+00 | no |
| svc/Original-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| svc/Original-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| svc/Original-vs-MTF | 2.818e-89 | 5.636e-88 | yes |
| randomforest/Original-vs-RP | 3.602e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GAF | 9.035e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | 2.718e-173 | 5.980e-172 | yes |
| lstm/Original-vs-RP | 8.026e-01 | 1.000e+00 | no |
| lstm/Original-vs-GAF | 8.137e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | 8.026e-01 | 1.000e+00 | no |
| lstm/Original-vs-MTF | 2.794e-129 | 5.868e-128 | yes |
| gru/Original-vs-RP | 4.497e-01 | 1.000e+00 | no |
| gru/Original-vs-GAF | 4.497e-01 | 1.000e+00 | no |
| gru/Original-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-MTF | 3.008e-228 | 7.220e-227 | yes |
| cnn/Original-vs-RP | 3.428e-01 | 1.000e+00 | no |
| cnn/Original-vs-GAF | 3.428e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | 2.888e-01 | 1.000e+00 | no |
| cnn/Original-vs-MTF | 4.066e-197 | 9.352e-196 | yes |

## 3. McNemar pairwise between encoders (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/RP-vs-GAF | 8.137e-01 | 1.000e+00 | no |
| knn/RP-vs-GADF | 6.276e-01 | 1.000e+00 | no |
| knn/RP-vs-MTF | 8.852e-70 | 1.770e-68 | yes |
| knn/GAF-vs-GADF | 6.056e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | 1.093e-69 | 2.076e-68 | yes |
| knn/GADF-vs-MTF | 3.053e-70 | 6.411e-69 | yes |
| svc/RP-vs-GAF | 5.839e-01 | 1.000e+00 | no |
| svc/RP-vs-GADF | 6.171e-01 | 1.000e+00 | no |
| svc/RP-vs-MTF | 4.605e-90 | 1.105e-88 | yes |
| svc/GAF-vs-GADF | 8.597e-01 | 1.000e+00 | no |
| svc/GAF-vs-MTF | 3.837e-89 | 8.825e-88 | yes |
| svc/GADF-vs-MTF | 3.837e-89 | 8.825e-88 | yes |
| randomforest/RP-vs-GAF | 4.568e-01 | 1.000e+00 | no |
| randomforest/RP-vs-GADF | 5.186e-01 | 1.000e+00 | no |
| randomforest/RP-vs-MTF | 6.422e-175 | 1.927e-173 | yes |
| randomforest/GAF-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | 2.119e-172 | 5.934e-171 | yes |
| randomforest/GADF-vs-MTF | 2.872e-173 | 8.328e-172 | yes |
| lstm/RP-vs-GAF | 8.445e-01 | 1.000e+00 | no |
| lstm/RP-vs-GADF | 8.445e-01 | 1.000e+00 | no |
| lstm/RP-vs-MTF | 4.230e-129 | 1.100e-127 | yes |
| lstm/GAF-vs-GADF | 8.445e-01 | 1.000e+00 | no |
| lstm/GAF-vs-MTF | 1.531e-128 | 3.828e-127 | yes |
| lstm/GADF-vs-MTF | 1.844e-129 | 4.978e-128 | yes |
| gru/RP-vs-GAF | 7.237e-01 | 1.000e+00 | no |
| gru/RP-vs-GADF | 7.237e-01 | 1.000e+00 | no |
| gru/RP-vs-MTF | 1.276e-227 | 4.465e-226 | yes |
| gru/GAF-vs-GADF | 7.237e-01 | 1.000e+00 | no |
| gru/GAF-vs-MTF | 1.276e-227 | 4.465e-226 | yes |
| gru/GADF-vs-MTF | 6.881e-229 | 2.477e-227 | yes |
| cnn/RP-vs-GAF | 9.896e-02 | 1.000e+00 | no |
| cnn/RP-vs-GADF | 7.893e-01 | 1.000e+00 | no |
| cnn/RP-vs-MTF | 6.109e-198 | 1.955e-196 | yes |
| cnn/GAF-vs-GADF | 4.331e-02 | 7.795e-01 | no |
| cnn/GAF-vs-MTF | 4.879e-196 | 1.513e-194 | yes |
| cnn/GADF-vs-MTF | 1.025e-198 | 3.384e-197 | yes |

## 4. Wilcoxon signed-rank on per-window MAE
| Comparison | effect r | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|---|
| MAE/RP-vs-GAF | -0.615 | <1e-300 | <1e-300 | yes |
| MAE/RP-vs-GADF | -0.223 | 1.208e-90 | 1.208e-90 | yes |
| MAE/RP-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-GADF | +0.457 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GADF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
