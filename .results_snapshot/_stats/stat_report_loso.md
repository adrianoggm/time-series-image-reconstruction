# Statistical significance report (LOSO 3-fold pooled)

All paired tests run on the pooled $N=8172$ test windows from
folds [0, 1, 2] (per-fold sizes [2645, 2788, 2739]); each window is one
paired (Original, Reconstruction) observation so pooling preserves
the McNemar/Wilcoxon validity. Raw p-values are corrected with
Holm--Bonferroni inside each family at $\alpha=0.05$.

## 1. Per-fold accuracy summary (mean +/- std across 3 folds)

| Classifier | Original | RP | MTF | GASF | GADF |
|---|---|---|---|---|---|
| knn | 0.504±0.035 [0.467,0.551] | 0.504±0.035 [0.466,0.551] | 0.465±0.039 [0.421,0.515] | 0.504±0.034 [0.468,0.550] | 0.505±0.034 [0.467,0.550] |
| svc | 0.738±0.020 [0.711,0.757] | 0.739±0.020 [0.711,0.757] | 0.665±0.026 [0.639,0.701] | 0.738±0.020 [0.710,0.757] | 0.738±0.020 [0.709,0.756] |
| randomforest | 0.786±0.039 [0.753,0.842] | 0.786±0.039 [0.751,0.840] | 0.689±0.026 [0.666,0.726] | 0.787±0.039 [0.754,0.841] | 0.787±0.039 [0.754,0.842] |
| lstm | 0.744±0.029 [0.708,0.779] | 0.744±0.028 [0.708,0.777] | 0.672±0.035 [0.633,0.719] | 0.744±0.029 [0.708,0.779] | 0.744±0.028 [0.709,0.778] |
| gru | 0.811±0.026 [0.791,0.848] | 0.811±0.026 [0.791,0.848] | 0.702±0.025 [0.671,0.732] | 0.811±0.026 [0.791,0.848] | 0.811±0.026 [0.791,0.847] |
| cnn | 0.794±0.050 [0.740,0.860] | 0.794±0.049 [0.740,0.859] | 0.684±0.031 [0.651,0.726] | 0.794±0.050 [0.739,0.860] | 0.794±0.050 [0.740,0.860] |

## 2. McNemar Original vs reconstruction (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/Original-vs-RP | 6.831e-01 | 1.000e+00 | no |
| knn/Original-vs-GAF | 7.237e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | 2.888e-01 | 1.000e+00 | no |
| knn/Original-vs-MTF | 6.655e-21 | 1.264e-19 | yes |
| svc/Original-vs-RP | 1.000e+00 | 1.000e+00 | no |
| svc/Original-vs-GAF | 6.171e-01 | 1.000e+00 | no |
| svc/Original-vs-GADF | 2.109e-01 | 1.000e+00 | no |
| svc/Original-vs-MTF | 1.859e-48 | 3.718e-47 | yes |
| randomforest/Original-vs-RP | 4.990e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GAF | 4.705e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | 6.650e-01 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | 2.608e-99 | 5.738e-98 | yes |
| lstm/Original-vs-RP | 5.791e-01 | 1.000e+00 | no |
| lstm/Original-vs-GAF | 6.831e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| lstm/Original-vs-MTF | 2.935e-54 | 6.163e-53 | yes |
| gru/Original-vs-RP | 6.831e-01 | 1.000e+00 | no |
| gru/Original-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-MTF | 4.104e-117 | 9.850e-116 | yes |
| cnn/Original-vs-RP | 5.050e-01 | 1.000e+00 | no |
| cnn/Original-vs-GAF | 7.728e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | 8.026e-01 | 1.000e+00 | no |
| cnn/Original-vs-MTF | 2.219e-116 | 5.104e-115 | yes |

## 3. McNemar pairwise between encoders (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/RP-vs-GAF | 7.728e-01 | 1.000e+00 | no |
| knn/RP-vs-GADF | 1.489e-01 | 1.000e+00 | no |
| knn/RP-vs-MTF | 1.241e-20 | 2.358e-19 | yes |
| knn/GAF-vs-GADF | 3.865e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | 6.655e-21 | 1.331e-19 | yes |
| knn/GADF-vs-MTF | 2.197e-21 | 4.614e-20 | yes |
| svc/RP-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| svc/RP-vs-GADF | 2.561e-01 | 1.000e+00 | no |
| svc/RP-vs-MTF | 2.028e-48 | 4.868e-47 | yes |
| svc/GAF-vs-GADF | 1.904e-01 | 1.000e+00 | no |
| svc/GAF-vs-MTF | 2.111e-48 | 4.868e-47 | yes |
| svc/GADF-vs-MTF | 1.447e-47 | 3.183e-46 | yes |
| randomforest/RP-vs-GAF | 1.696e-01 | 1.000e+00 | no |
| randomforest/RP-vs-GADF | 3.211e-01 | 1.000e+00 | no |
| randomforest/RP-vs-MTF | 1.049e-98 | 2.937e-97 | yes |
| randomforest/GAF-vs-GADF | 9.020e-01 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | 1.200e-100 | 3.600e-99 | yes |
| randomforest/GADF-vs-MTF | 2.709e-100 | 7.855e-99 | yes |
| lstm/RP-vs-GAF | 6.276e-01 | 1.000e+00 | no |
| lstm/RP-vs-GADF | 6.276e-01 | 1.000e+00 | no |
| lstm/RP-vs-MTF | 6.605e-54 | 1.651e-52 | yes |
| lstm/GAF-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| lstm/GAF-vs-MTF | 3.469e-54 | 9.019e-53 | yes |
| lstm/GADF-vs-MTF | 2.481e-54 | 6.700e-53 | yes |
| gru/RP-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| gru/RP-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| gru/RP-vs-MTF | 8.736e-118 | 3.145e-116 | yes |
| gru/GAF-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| gru/GAF-vs-MTF | 2.701e-117 | 9.183e-116 | yes |
| gru/GADF-vs-MTF | 1.329e-117 | 4.650e-116 | yes |
| cnn/RP-vs-GAF | 3.320e-01 | 1.000e+00 | no |
| cnn/RP-vs-GADF | 6.464e-01 | 1.000e+00 | no |
| cnn/RP-vs-MTF | 1.095e-115 | 3.394e-114 | yes |
| cnn/GAF-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| cnn/GAF-vs-MTF | 3.372e-117 | 1.113e-115 | yes |
| cnn/GADF-vs-MTF | 2.219e-116 | 7.101e-115 | yes |

## 4. Wilcoxon signed-rank on per-window MAE
| Comparison | effect r | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|---|
| MAE/RP-vs-GAF | -0.615 | <1e-300 | <1e-300 | yes |
| MAE/RP-vs-GADF | -0.223 | 1.208e-90 | 1.208e-90 | yes |
| MAE/RP-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-GADF | +0.457 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GADF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
