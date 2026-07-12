# Statistical significance report (LOSO 3-fold pooled)

All paired tests run on the pooled $N=8172$ test windows from
folds [0, 1, 2] (per-fold sizes [2645, 2788, 2739]); each window is one
paired (Original, Reconstruction) observation so pooling preserves
the McNemar/Wilcoxon validity. Raw p-values are corrected with
Holm--Bonferroni inside each family at $\alpha=0.05$.

## 1. Per-fold accuracy summary (mean +/- std across 3 folds)

| Classifier | Original | RP | MTF | GASF | GADF |
|---|---|---|---|---|---|
| knn | 0.504±0.035 [0.467,0.551] | 0.549±0.029 [0.524,0.589] | 0.465±0.039 [0.421,0.515] | 0.504±0.034 [0.468,0.550] | 0.505±0.034 [0.467,0.550] |
| svc | 0.738±0.020 [0.711,0.757] | 0.657±0.041 [0.610,0.710] | 0.665±0.026 [0.639,0.701] | 0.738±0.020 [0.710,0.757] | 0.738±0.020 [0.709,0.756] |
| randomforest | 0.786±0.039 [0.753,0.842] | 0.648±0.041 [0.616,0.706] | 0.689±0.026 [0.666,0.726] | 0.787±0.039 [0.754,0.841] | 0.787±0.039 [0.754,0.842] |
| lstm | 0.744±0.029 [0.708,0.779] | 0.641±0.030 [0.612,0.683] | 0.672±0.035 [0.633,0.719] | 0.744±0.029 [0.708,0.779] | 0.744±0.028 [0.709,0.778] |
| gru | 0.811±0.026 [0.791,0.848] | 0.702±0.030 [0.660,0.726] | 0.702±0.025 [0.671,0.732] | 0.811±0.026 [0.791,0.848] | 0.811±0.026 [0.791,0.847] |
| cnn | 0.796±0.036 [0.764,0.847] | 0.691±0.035 [0.654,0.738] | 0.686±0.035 [0.654,0.735] | 0.796±0.036 [0.764,0.847] | 0.796±0.036 [0.764,0.847] |

## 2. McNemar Original vs reconstruction (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/Original-vs-RP | 3.684e-29 | 5.157e-28 | yes |
| knn/Original-vs-GAF | 7.237e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | 2.888e-01 | 1.000e+00 | no |
| knn/Original-vs-MTF | 6.655e-21 | 8.652e-20 | yes |
| svc/Original-vs-RP | 8.652e-67 | 1.471e-65 | yes |
| svc/Original-vs-GAF | 6.171e-01 | 1.000e+00 | no |
| svc/Original-vs-GADF | 2.109e-01 | 1.000e+00 | no |
| svc/Original-vs-MTF | 1.859e-48 | 2.788e-47 | yes |
| randomforest/Original-vs-RP | 3.099e-166 | 7.436e-165 | yes |
| randomforest/Original-vs-GAF | 4.705e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | 6.650e-01 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | 2.608e-99 | 4.955e-98 | yes |
| lstm/Original-vs-RP | 1.071e-86 | 1.927e-85 | yes |
| lstm/Original-vs-GAF | 6.831e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| lstm/Original-vs-MTF | 2.935e-54 | 4.695e-53 | yes |
| gru/Original-vs-RP | 5.405e-111 | 1.081e-109 | yes |
| gru/Original-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-MTF | 4.104e-117 | 9.030e-116 | yes |
| cnn/Original-vs-RP | 1.207e-115 | 2.536e-114 | yes |
| cnn/Original-vs-GAF | 8.137e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | 7.518e-01 | 1.000e+00 | no |
| cnn/Original-vs-MTF | 2.258e-119 | 5.193e-118 | yes |

## 3. McNemar pairwise between encoders (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/RP-vs-GAF | 3.262e-29 | 4.893e-28 | yes |
| knn/RP-vs-GADF | 1.315e-28 | 1.841e-27 | yes |
| knn/RP-vs-MTF | 1.214e-71 | 2.672e-70 | yes |
| knn/GAF-vs-GADF | 3.865e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | 6.655e-21 | 7.986e-20 | yes |
| knn/GADF-vs-MTF | 2.197e-21 | 2.856e-20 | yes |
| svc/RP-vs-GAF | 1.054e-66 | 2.213e-65 | yes |
| svc/RP-vs-GADF | 1.742e-65 | 3.484e-64 | yes |
| svc/RP-vs-MTF | 9.135e-02 | 8.221e-01 | no |
| svc/GAF-vs-GADF | 1.904e-01 | 1.000e+00 | no |
| svc/GAF-vs-MTF | 2.111e-48 | 3.589e-47 | yes |
| svc/GADF-vs-MTF | 1.447e-47 | 2.315e-46 | yes |
| randomforest/RP-vs-GAF | 5.229e-167 | 1.830e-165 | yes |
| randomforest/RP-vs-GADF | 3.327e-167 | 1.198e-165 | yes |
| randomforest/RP-vs-MTF | 5.981e-20 | 6.579e-19 | yes |
| randomforest/GAF-vs-GADF | 9.020e-01 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | 1.200e-100 | 3.120e-99 | yes |
| randomforest/GADF-vs-MTF | 2.709e-100 | 6.772e-99 | yes |
| lstm/RP-vs-GAF | 6.937e-87 | 1.665e-85 | yes |
| lstm/RP-vs-GADF | 8.620e-87 | 1.983e-85 | yes |
| lstm/RP-vs-MTF | 2.801e-10 | 2.801e-09 | yes |
| lstm/GAF-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| lstm/GAF-vs-MTF | 3.469e-54 | 6.244e-53 | yes |
| lstm/GADF-vs-MTF | 2.481e-54 | 4.714e-53 | yes |
| gru/RP-vs-GAF | 4.957e-111 | 1.338e-109 | yes |
| gru/RP-vs-GADF | 9.940e-112 | 2.783e-110 | yes |
| gru/RP-vs-MTF | 9.335e-01 | 1.000e+00 | no |
| gru/GAF-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| gru/GAF-vs-MTF | 2.701e-117 | 8.373e-116 | yes |
| gru/GADF-vs-MTF | 1.329e-117 | 4.251e-116 | yes |
| cnn/RP-vs-GAF | 1.352e-115 | 3.920e-114 | yes |
| cnn/RP-vs-GADF | 9.360e-116 | 2.808e-114 | yes |
| cnn/RP-vs-MTF | 3.503e-01 | 1.000e+00 | no |
| cnn/GAF-vs-GADF | 8.312e-01 | 1.000e+00 | no |
| cnn/GAF-vs-MTF | 2.189e-118 | 7.223e-117 | yes |
| cnn/GADF-vs-MTF | 7.497e-119 | 2.549e-117 | yes |

## 4. Wilcoxon signed-rank on per-window MAE
| Comparison | effect r | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|---|
| MAE/RP-vs-GAF | +0.866 | <1e-300 | <1e-300 | yes |
| MAE/RP-vs-GADF | +0.866 | <1e-300 | <1e-300 | yes |
| MAE/RP-vs-MTF | -0.474 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-GADF | +0.457 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GADF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
