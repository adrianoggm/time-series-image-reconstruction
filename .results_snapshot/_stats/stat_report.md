# Statistical significance report (LOSO 3-fold pooled)

All paired tests run on the pooled $N=8172$ test windows from
folds [0, 1, 2] (per-fold sizes [2645, 2788, 2739]); each window is one
paired (Original, Reconstruction) observation so pooling preserves
the McNemar/Wilcoxon validity. Raw p-values are corrected with
Holm--Bonferroni inside each family at $\alpha=0.05$.

## 1. Per-fold accuracy summary (mean +/- std across 3 folds)

| Classifier | Original | RP | MTF | GASF | GADF |
|---|---|---|---|---|---|
| knn | 0.504±0.035 [0.467,0.551] | 0.549±0.029 [0.524,0.589] | 0.465±0.039 [0.421,0.515] | 0.504±0.034 [0.468,0.550] | 0.521±0.045 [0.474,0.581] |
| svc | 0.738±0.020 [0.711,0.757] | 0.657±0.041 [0.610,0.710] | 0.665±0.026 [0.639,0.701] | 0.738±0.020 [0.710,0.757] | 0.674±0.025 [0.646,0.708] |
| randomforest | 0.786±0.039 [0.753,0.842] | 0.648±0.041 [0.616,0.706] | 0.689±0.026 [0.666,0.726] | 0.787±0.039 [0.754,0.841] | 0.707±0.041 [0.670,0.763] |
| lstm | 0.744±0.029 [0.708,0.779] | 0.641±0.030 [0.612,0.683] | 0.672±0.035 [0.633,0.719] | 0.744±0.029 [0.708,0.779] | 0.664±0.027 [0.643,0.703] |
| gru | 0.811±0.026 [0.791,0.848] | 0.702±0.030 [0.660,0.726] | 0.702±0.025 [0.671,0.732] | 0.811±0.026 [0.791,0.848] | 0.739±0.038 [0.700,0.791] |
| cnn | 0.796±0.036 [0.764,0.847] | 0.691±0.035 [0.654,0.738] | 0.686±0.035 [0.654,0.735] | 0.796±0.036 [0.764,0.847] | 0.740±0.031 [0.704,0.780] |

## 2. McNemar Original vs reconstruction (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/Original-vs-RP | 3.684e-29 | 3.315e-28 | yes |
| knn/Original-vs-GAF | 7.237e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | 2.545e-07 | 1.782e-06 | yes |
| knn/Original-vs-MTF | 6.655e-21 | 5.324e-20 | yes |
| svc/Original-vs-RP | 8.652e-67 | 1.211e-65 | yes |
| svc/Original-vs-GAF | 6.171e-01 | 1.000e+00 | no |
| svc/Original-vs-GADF | 5.033e-37 | 5.033e-36 | yes |
| svc/Original-vs-MTF | 1.859e-48 | 2.231e-47 | yes |
| randomforest/Original-vs-RP | 3.099e-166 | 7.436e-165 | yes |
| randomforest/Original-vs-GAF | 4.705e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | 6.997e-70 | 1.050e-68 | yes |
| randomforest/Original-vs-MTF | 2.608e-99 | 4.955e-98 | yes |
| lstm/Original-vs-RP | 1.071e-86 | 1.927e-85 | yes |
| lstm/Original-vs-GAF | 6.831e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | 1.717e-72 | 2.747e-71 | yes |
| lstm/Original-vs-MTF | 2.935e-54 | 3.815e-53 | yes |
| gru/Original-vs-RP | 5.405e-111 | 1.081e-109 | yes |
| gru/Original-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-GADF | 5.332e-80 | 9.065e-79 | yes |
| gru/Original-vs-MTF | 4.104e-117 | 9.030e-116 | yes |
| cnn/Original-vs-RP | 1.207e-115 | 2.536e-114 | yes |
| cnn/Original-vs-GAF | 8.137e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | 3.126e-48 | 3.438e-47 | yes |
| cnn/Original-vs-MTF | 2.258e-119 | 5.193e-118 | yes |

## 3. McNemar pairwise between encoders (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/RP-vs-GAF | 3.262e-29 | 5.871e-28 | yes |
| knn/RP-vs-GADF | 3.657e-13 | 4.389e-12 | yes |
| knn/RP-vs-MTF | 1.214e-71 | 3.279e-70 | yes |
| knn/GAF-vs-GADF | 2.445e-07 | 2.201e-06 | yes |
| knn/GAF-vs-MTF | 6.655e-21 | 9.983e-20 | yes |
| knn/GADF-vs-MTF | 6.983e-40 | 1.466e-38 | yes |
| svc/RP-vs-GAF | 1.054e-66 | 2.634e-65 | yes |
| svc/RP-vs-GADF | 1.161e-03 | 6.966e-03 | yes |
| svc/RP-vs-MTF | 9.135e-02 | 3.654e-01 | no |
| svc/GAF-vs-GADF | 5.538e-37 | 1.108e-35 | yes |
| svc/GAF-vs-MTF | 2.111e-48 | 4.855e-47 | yes |
| svc/GADF-vs-MTF | 9.849e-03 | 4.924e-02 | yes |
| randomforest/RP-vs-GAF | 5.229e-167 | 1.883e-165 | yes |
| randomforest/RP-vs-GADF | 1.441e-24 | 2.450e-23 | yes |
| randomforest/RP-vs-MTF | 5.981e-20 | 8.373e-19 | yes |
| randomforest/GAF-vs-GADF | 7.749e-71 | 2.015e-69 | yes |
| randomforest/GAF-vs-MTF | 1.200e-100 | 3.720e-99 | yes |
| randomforest/GADF-vs-MTF | 2.516e-04 | 1.761e-03 | yes |
| lstm/RP-vs-GAF | 6.937e-87 | 2.081e-85 | yes |
| lstm/RP-vs-GADF | 1.368e-04 | 1.094e-03 | yes |
| lstm/RP-vs-MTF | 2.801e-10 | 2.801e-09 | yes |
| lstm/GAF-vs-GADF | 2.203e-72 | 6.169e-71 | yes |
| lstm/GAF-vs-MTF | 3.469e-54 | 8.325e-53 | yes |
| lstm/GADF-vs-MTF | 1.407e-01 | 4.222e-01 | no |
| gru/RP-vs-GAF | 4.957e-111 | 1.586e-109 | yes |
| gru/RP-vs-GADF | 5.523e-13 | 6.075e-12 | yes |
| gru/RP-vs-MTF | 9.335e-01 | 9.335e-01 | no |
| gru/GAF-vs-GADF | 2.384e-80 | 6.915e-79 | yes |
| gru/GAF-vs-MTF | 2.701e-117 | 9.183e-116 | yes |
| gru/GADF-vs-MTF | 3.406e-15 | 4.427e-14 | yes |
| cnn/RP-vs-GAF | 1.352e-115 | 4.460e-114 | yes |
| cnn/RP-vs-GADF | 9.797e-24 | 1.567e-22 | yes |
| cnn/RP-vs-MTF | 3.503e-01 | 7.006e-01 | no |
| cnn/GAF-vs-GADF | 9.756e-48 | 2.146e-46 | yes |
| cnn/GAF-vs-MTF | 2.189e-118 | 7.661e-117 | yes |
| cnn/GADF-vs-MTF | 8.740e-33 | 1.661e-31 | yes |

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
