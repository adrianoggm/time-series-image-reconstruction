# Statistical significance report (LOSO 3-fold pooled)

Window-level diagnostics use the pooled $N=8172$ test windows from
folds [0, 1, 2] (per-fold sizes [2645, 2788, 2739]). Pairing is preserved,
but windows from one subject are dependent; these tests are therefore
secondary. Section 5 aggregates by subject and provides the primary
equivalence inference. Raw p-values are corrected with Holm--Bonferroni
inside each stated family at $\alpha=0.05$.

## 1. Per-fold accuracy summary (mean +/- std across 3 folds)

| Classifier | Original | RP | MTF | GASF | GADF |
|---|---|---|---|---|---|
| knn | 0.504±0.035 [0.467,0.551] | 0.504±0.035 [0.467,0.551] | 0.465±0.039 [0.421,0.515] | 0.504±0.034 [0.468,0.550] | 0.505±0.034 [0.467,0.550] |
| svc | 0.738±0.020 [0.711,0.757] | 0.738±0.020 [0.711,0.756] | 0.665±0.026 [0.639,0.701] | 0.738±0.020 [0.710,0.757] | 0.738±0.020 [0.709,0.756] |
| randomforest | 0.786±0.039 [0.753,0.842] | 0.786±0.039 [0.752,0.841] | 0.689±0.026 [0.666,0.726] | 0.787±0.039 [0.754,0.841] | 0.787±0.039 [0.754,0.842] |
| lstm | 0.744±0.029 [0.708,0.779] | 0.744±0.028 [0.708,0.778] | 0.672±0.035 [0.633,0.719] | 0.744±0.029 [0.708,0.779] | 0.744±0.028 [0.709,0.778] |
| gru | 0.811±0.026 [0.791,0.848] | 0.811±0.026 [0.791,0.848] | 0.702±0.025 [0.671,0.732] | 0.811±0.026 [0.791,0.848] | 0.811±0.026 [0.791,0.847] |
| cnn | 0.796±0.036 [0.764,0.847] | 0.796±0.035 [0.764,0.845] | 0.686±0.035 [0.654,0.735] | 0.796±0.036 [0.764,0.847] | 0.796±0.036 [0.764,0.847] |

## 2. McNemar Original vs reconstruction (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/Original-vs-RP | 6.171e-01 | 1.000e+00 | no |
| knn/Original-vs-GAF | 7.237e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | 2.888e-01 | 1.000e+00 | no |
| knn/Original-vs-MTF | 6.655e-21 | 1.264e-19 | yes |
| svc/Original-vs-RP | 6.056e-01 | 1.000e+00 | no |
| svc/Original-vs-GAF | 6.171e-01 | 1.000e+00 | no |
| svc/Original-vs-GADF | 2.109e-01 | 1.000e+00 | no |
| svc/Original-vs-MTF | 1.859e-48 | 3.718e-47 | yes |
| randomforest/Original-vs-RP | 1.000e+00 | 1.000e+00 | no |
| randomforest/Original-vs-GAF | 4.705e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | 6.650e-01 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | 2.608e-99 | 5.738e-98 | yes |
| lstm/Original-vs-RP | 5.791e-01 | 1.000e+00 | no |
| lstm/Original-vs-GAF | 6.831e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| lstm/Original-vs-MTF | 2.935e-54 | 6.163e-53 | yes |
| gru/Original-vs-RP | 2.207e-01 | 1.000e+00 | no |
| gru/Original-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-MTF | 4.104e-117 | 9.440e-116 | yes |
| cnn/Original-vs-RP | 7.518e-01 | 1.000e+00 | no |
| cnn/Original-vs-GAF | 8.137e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | 7.518e-01 | 1.000e+00 | no |
| cnn/Original-vs-MTF | 2.258e-119 | 5.419e-118 | yes |

## 3. McNemar pairwise between encoders (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/RP-vs-GAF | 7.518e-01 | 1.000e+00 | no |
| knn/RP-vs-GADF | 3.428e-01 | 1.000e+00 | no |
| knn/RP-vs-MTF | 7.173e-21 | 1.363e-19 | yes |
| knn/GAF-vs-GADF | 3.865e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | 6.655e-21 | 1.331e-19 | yes |
| knn/GADF-vs-MTF | 2.197e-21 | 4.614e-20 | yes |
| svc/RP-vs-GAF | 6.464e-01 | 1.000e+00 | no |
| svc/RP-vs-GADF | 6.171e-01 | 1.000e+00 | no |
| svc/RP-vs-MTF | 6.548e-48 | 1.506e-46 | yes |
| svc/GAF-vs-GADF | 1.904e-01 | 1.000e+00 | no |
| svc/GAF-vs-MTF | 2.111e-48 | 5.066e-47 | yes |
| svc/GADF-vs-MTF | 1.447e-47 | 3.183e-46 | yes |
| randomforest/RP-vs-GAF | 4.185e-01 | 1.000e+00 | no |
| randomforest/RP-vs-GADF | 6.143e-01 | 1.000e+00 | no |
| randomforest/RP-vs-MTF | 2.846e-99 | 7.968e-98 | yes |
| randomforest/GAF-vs-GADF | 9.020e-01 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | 1.200e-100 | 3.600e-99 | yes |
| randomforest/GADF-vs-MTF | 2.709e-100 | 7.855e-99 | yes |
| lstm/RP-vs-GAF | 6.056e-01 | 1.000e+00 | no |
| lstm/RP-vs-GADF | 6.276e-01 | 1.000e+00 | no |
| lstm/RP-vs-MTF | 5.588e-54 | 1.397e-52 | yes |
| lstm/GAF-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| lstm/GAF-vs-MTF | 3.469e-54 | 9.019e-53 | yes |
| lstm/GADF-vs-MTF | 2.481e-54 | 6.700e-53 | yes |
| gru/RP-vs-GAF | 5.050e-01 | 1.000e+00 | no |
| gru/RP-vs-GADF | 5.791e-01 | 1.000e+00 | no |
| gru/RP-vs-MTF | 3.777e-118 | 1.246e-116 | yes |
| gru/GAF-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| gru/GAF-vs-MTF | 2.701e-117 | 8.373e-116 | yes |
| gru/GADF-vs-MTF | 1.329e-117 | 4.251e-116 | yes |
| cnn/RP-vs-GAF | 8.137e-01 | 1.000e+00 | no |
| cnn/RP-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| cnn/RP-vs-MTF | 2.189e-118 | 7.661e-117 | yes |
| cnn/GAF-vs-GADF | 8.312e-01 | 1.000e+00 | no |
| cnn/GAF-vs-MTF | 2.189e-118 | 7.661e-117 | yes |
| cnn/GADF-vs-MTF | 7.497e-119 | 2.699e-117 | yes |

## 4. Wilcoxon signed-rank on per-window MAE
| Comparison | effect r | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|---|
| MAE/RP-vs-GAF | -0.703 | <1e-300 | <1e-300 | yes |
| MAE/RP-vs-GADF | -0.265 | 5.162e-127 | 5.162e-127 | yes |
| MAE/RP-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-GADF | +0.457 | <1e-300 | <1e-300 | yes |
| MAE/GAF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |
| MAE/GADF-vs-MTF | -0.866 | <1e-300 | <1e-300 | yes |

## 5. Subject-clustered classification (51 subjects as units)

### 5a. Subject-clustered Wilcoxon on per-subject accuracy, Original vs reconstruction
| Comparison | effect r | p (raw) | p (Holm) | Reject H0 (difference)? |
|---|---|---|---|---|
| knn/Original-vs-RP | -0.091 | 8.501e-01 | 1.000e+00 | no |
| knn/Original-vs-GAF | +0.000 | 1.000e+00 | 1.000e+00 | no |
| knn/Original-vs-GADF | -0.447 | 2.282e-01 | 1.000e+00 | no |
| knn/Original-vs-MTF | +0.600 | 3.258e-05 | 6.190e-04 | yes |
| svc/Original-vs-RP | +0.192 | 4.905e-01 | 1.000e+00 | no |
| svc/Original-vs-GAF | +0.091 | 8.501e-01 | 1.000e+00 | no |
| svc/Original-vs-GADF | +0.339 | 1.575e-01 | 1.000e+00 | no |
| svc/Original-vs-MTF | +0.645 | 9.677e-06 | 1.935e-04 | yes |
| randomforest/Original-vs-RP | +0.163 | 4.517e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GAF | -0.072 | 7.116e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | +0.085 | 6.771e-01 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | +0.835 | 3.563e-09 | 7.838e-08 | yes |
| lstm/Original-vs-RP | +0.192 | 4.999e-01 | 1.000e+00 | no |
| lstm/Original-vs-GAF | -0.128 | 7.498e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | +0.000 | 1.000e+00 | 1.000e+00 | no |
| lstm/Original-vs-MTF | +0.707 | 4.450e-07 | 9.346e-06 | yes |
| gru/Original-vs-RP | -0.422 | 3.430e-01 | 1.000e+00 | no |
| gru/Original-vs-GAF | -0.463 | 4.227e-01 | 1.000e+00 | no |
| gru/Original-vs-GADF | -0.217 | 5.102e-01 | 1.000e+00 | no |
| gru/Original-vs-MTF | +0.853 | 1.106e-09 | 2.654e-08 | yes |
| cnn/Original-vs-RP | +0.274 | 3.749e-01 | 1.000e+00 | no |
| cnn/Original-vs-GAF | +0.073 | 7.739e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | +0.039 | 9.055e-01 | 1.000e+00 | no |
| cnn/Original-vs-MTF | +0.846 | 2.218e-09 | 5.101e-08 | yes |

### 5b. TOST equivalence on per-subject $\Delta$accuracy (margin $\pm0.01$)
Equivalence within the margin is declared when the Holm-adjusted TOST p < 0.05.
| Comparison | mean $\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |
|---|---|---|---|---|
| knn/Original-vs-RP | -0.0000 | 1.550e-38 | 3.564e-37 | yes |
| knn/Original-vs-GAF | -0.0001 | 1.897e-32 | 3.795e-31 | yes |
| knn/Original-vs-GADF | -0.0005 | 6.982e-28 | 1.047e-26 | yes |
| knn/Original-vs-MTF | +0.0474 | 9.999e-01 | 1.000e+00 | no |
| svc/Original-vs-RP | +0.0004 | 3.023e-27 | 3.963e-26 | yes |
| svc/Original-vs-GAF | +0.0000 | 2.439e-37 | 5.367e-36 | yes |
| svc/Original-vs-GADF | +0.0008 | 1.624e-23 | 1.624e-22 | yes |
| svc/Original-vs-MTF | +0.0651 | 1.000e+00 | 1.000e+00 | no |
| randomforest/Original-vs-RP | +0.0006 | 2.302e-17 | 2.072e-16 | yes |
| randomforest/Original-vs-GAF | -0.0006 | 2.545e-13 | 1.782e-12 | yes |
| randomforest/Original-vs-GADF | -0.0002 | 1.480e-14 | 1.184e-13 | yes |
| randomforest/Original-vs-MTF | +0.0910 | 1.000e+00 | 1.000e+00 | no |
| lstm/Original-vs-RP | +0.0005 | 5.906e-24 | 6.497e-23 | yes |
| lstm/Original-vs-GAF | -0.0000 | 3.740e-32 | 7.106e-31 | yes |
| lstm/Original-vs-GADF | +0.0001 | 2.831e-27 | 3.963e-26 | yes |
| lstm/Original-vs-MTF | +0.0697 | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-RP | -0.0004 | 1.990e-35 | 4.178e-34 | yes |
| gru/Original-vs-GAF | -0.0002 | 1.036e-40 | 2.486e-39 | yes |
| gru/Original-vs-GADF | -0.0003 | 6.813e-30 | 1.226e-28 | yes |
| gru/Original-vs-MTF | +0.1124 | 1.000e+00 | 1.000e+00 | no |
| cnn/Original-vs-RP | +0.0003 | 2.461e-29 | 4.184e-28 | yes |
| cnn/Original-vs-GAF | +0.0003 | 1.184e-25 | 1.420e-24 | yes |
| cnn/Original-vs-GADF | +0.0002 | 4.834e-28 | 7.735e-27 | yes |
| cnn/Original-vs-MTF | +0.1144 | 1.000e+00 | 1.000e+00 | no |

### 5c. Pairwise subject-clustered Wilcoxon on per-subject accuracy (between encoders)
| Comparison | effect r | p (raw) | p (Holm) | Reject H0 (difference)? |
|---|---|---|---|---|
| knn/RP-vs-GAF | +0.000 | 1.000e+00 | 1.000e+00 | no |
| knn/RP-vs-GADF | -0.421 | 2.223e-01 | 1.000e+00 | no |
| knn/RP-vs-MTF | +0.603 | 2.977e-05 | 6.253e-04 | yes |
| knn/GAF-vs-GADF | -0.276 | 4.028e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | +0.597 | 3.562e-05 | 6.768e-04 | yes |
| knn/GADF-vs-MTF | +0.603 | 2.983e-05 | 6.253e-04 | yes |
| svc/RP-vs-GAF | -0.110 | 6.624e-01 | 1.000e+00 | no |
| svc/RP-vs-GADF | +0.076 | 7.158e-01 | 1.000e+00 | no |
| svc/RP-vs-MTF | +0.632 | 9.529e-06 | 2.264e-04 | yes |
| svc/GAF-vs-GADF | +0.381 | 1.352e-01 | 1.000e+00 | no |
| svc/GAF-vs-MTF | +0.646 | 9.434e-06 | 2.264e-04 | yes |
| svc/GADF-vs-MTF | +0.637 | 1.027e-05 | 2.264e-04 | yes |
| randomforest/RP-vs-GAF | -0.191 | 3.378e-01 | 1.000e+00 | no |
| randomforest/RP-vs-GADF | -0.002 | 9.903e-01 | 1.000e+00 | no |
| randomforest/RP-vs-MTF | +0.838 | 4.372e-09 | 1.224e-07 | yes |
| randomforest/GAF-vs-GADF | +0.128 | 5.260e-01 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | +0.849 | 1.910e-09 | 6.302e-08 | yes |
| randomforest/GADF-vs-MTF | +0.837 | 3.252e-09 | 9.430e-08 | yes |
| lstm/RP-vs-GAF | -0.174 | 5.606e-01 | 1.000e+00 | no |
| lstm/RP-vs-GADF | -0.155 | 5.262e-01 | 1.000e+00 | no |
| lstm/RP-vs-MTF | +0.713 | 4.659e-07 | 1.165e-05 | yes |
| lstm/GAF-vs-GADF | +0.079 | 7.817e-01 | 1.000e+00 | no |
| lstm/GAF-vs-MTF | +0.709 | 4.034e-07 | 1.089e-05 | yes |
| lstm/GADF-vs-MTF | +0.707 | 4.344e-07 | 1.130e-05 | yes |
| gru/RP-vs-GAF | +0.000 | 1.000e+00 | 1.000e+00 | no |
| gru/RP-vs-GADF | +0.059 | 8.577e-01 | 1.000e+00 | no |
| gru/RP-vs-MTF | +0.854 | 1.075e-09 | 3.869e-08 | yes |
| gru/GAF-vs-GADF | -0.074 | 8.326e-01 | 1.000e+00 | no |
| gru/GAF-vs-MTF | +0.853 | 1.105e-09 | 3.869e-08 | yes |
| gru/GADF-vs-MTF | +0.850 | 1.279e-09 | 4.349e-08 | yes |
| cnn/RP-vs-GAF | -0.110 | 6.643e-01 | 1.000e+00 | no |
| cnn/RP-vs-GADF | -0.125 | 6.590e-01 | 1.000e+00 | no |
| cnn/RP-vs-MTF | +0.846 | 2.215e-09 | 7.089e-08 | yes |
| cnn/GAF-vs-GADF | +0.029 | 9.140e-01 | 1.000e+00 | no |
| cnn/GAF-vs-MTF | +0.845 | 2.284e-09 | 7.089e-08 | yes |
| cnn/GADF-vs-MTF | +0.846 | 2.220e-09 | 7.089e-08 | yes |

### 5d. Pairwise TOST equivalence between encoders (margin $\pm0.01$)
| Comparison | mean $\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |
|---|---|---|---|---|
| knn/RP-vs-GAF | -0.0000 | 2.956e-26 | 9.165e-25 | yes |
| knn/RP-vs-GADF | -0.0005 | 1.948e-25 | 5.843e-24 | yes |
| knn/RP-vs-MTF | +0.0474 | 9.999e-01 | 1.000e+00 | no |
| knn/GAF-vs-GADF | -0.0005 | 2.434e-25 | 7.059e-24 | yes |
| knn/GAF-vs-MTF | +0.0475 | 9.999e-01 | 1.000e+00 | no |
| knn/GADF-vs-MTF | +0.0479 | 9.999e-01 | 1.000e+00 | no |
| svc/RP-vs-GAF | -0.0004 | 9.669e-23 | 2.417e-21 | yes |
| svc/RP-vs-GADF | +0.0005 | 3.609e-18 | 7.939e-17 | yes |
| svc/RP-vs-MTF | +0.0647 | 1.000e+00 | 1.000e+00 | no |
| svc/GAF-vs-GADF | +0.0008 | 4.684e-25 | 1.311e-23 | yes |
| svc/GAF-vs-MTF | +0.0651 | 1.000e+00 | 1.000e+00 | no |
| svc/GADF-vs-MTF | +0.0642 | 1.000e+00 | 1.000e+00 | no |
| randomforest/RP-vs-GAF | -0.0011 | 1.428e-11 | 2.857e-10 | yes |
| randomforest/RP-vs-GADF | -0.0008 | 5.731e-11 | 1.089e-09 | yes |
| randomforest/RP-vs-MTF | +0.0905 | 1.000e+00 | 1.000e+00 | no |
| randomforest/GAF-vs-GADF | +0.0004 | 1.538e-12 | 3.229e-11 | yes |
| randomforest/GAF-vs-MTF | +0.0916 | 1.000e+00 | 1.000e+00 | no |
| randomforest/GADF-vs-MTF | +0.0913 | 1.000e+00 | 1.000e+00 | no |
| lstm/RP-vs-GAF | -0.0005 | 3.025e-21 | 6.958e-20 | yes |
| lstm/RP-vs-GADF | -0.0004 | 2.786e-22 | 6.687e-21 | yes |
| lstm/RP-vs-MTF | +0.0692 | 1.000e+00 | 1.000e+00 | no |
| lstm/GAF-vs-GADF | +0.0001 | 1.056e-26 | 3.484e-25 | yes |
| lstm/GAF-vs-MTF | +0.0698 | 1.000e+00 | 1.000e+00 | no |
| lstm/GADF-vs-MTF | +0.0696 | 1.000e+00 | 1.000e+00 | no |
| gru/RP-vs-GAF | +0.0002 | 8.787e-34 | 3.163e-32 | yes |
| gru/RP-vs-GADF | +0.0001 | 1.168e-28 | 3.970e-27 | yes |
| gru/RP-vs-MTF | +0.1128 | 1.000e+00 | 1.000e+00 | no |
| gru/GAF-vs-GADF | -0.0001 | 9.593e-32 | 3.358e-30 | yes |
| gru/GAF-vs-MTF | +0.1125 | 1.000e+00 | 1.000e+00 | no |
| gru/GADF-vs-MTF | +0.1126 | 1.000e+00 | 1.000e+00 | no |
| cnn/RP-vs-GAF | +0.0000 | 7.033e-25 | 1.899e-23 | yes |
| cnn/RP-vs-GADF | -0.0001 | 1.188e-26 | 3.802e-25 | yes |
| cnn/RP-vs-MTF | +0.1141 | 1.000e+00 | 1.000e+00 | no |
| cnn/GAF-vs-GADF | -0.0001 | 7.973e-25 | 2.073e-23 | yes |
| cnn/GAF-vs-MTF | +0.1141 | 1.000e+00 | 1.000e+00 | no |
| cnn/GADF-vs-MTF | +0.1142 | 1.000e+00 | 1.000e+00 | no |

### 5e. Simultaneous subject-cluster bootstrap 95% intervals on per-subject $\Delta$accuracy (Original - reconstruction)
The primary family contains 18 comparisons (six classifiers by RP/GASF/GADF); the six MTF negative-control comparisons form a separate family. Within each family the same subject resamples are used jointly and a max-deviation critical value controls family-wise coverage. Equivalence is declared when the whole simultaneous interval lies inside the margin. Primary margin $\pm0.01$; the less stringent $\pm0.02$ margin is a sensitivity analysis.
| Comparison | mean $\Delta$acc | simultaneous 95% interval | Equiv. $\pm$0.01 | Equiv. $\pm$0.02 |
|---|---|---|---|---|
| knn/Original-vs-RP | -0.0000 | [-0.0022, +0.0021] | yes | yes |
| knn/Original-vs-GAF | -0.0001 | [-0.0022, +0.0021] | yes | yes |
| knn/Original-vs-GADF | -0.0005 | [-0.0027, +0.0016] | yes | yes |
| svc/Original-vs-RP | +0.0004 | [-0.0018, +0.0025] | yes | yes |
| svc/Original-vs-GAF | +0.0000 | [-0.0022, +0.0022] | yes | yes |
| svc/Original-vs-GADF | +0.0008 | [-0.0013, +0.0030] | yes | yes |
| randomforest/Original-vs-RP | +0.0006 | [-0.0016, +0.0027] | yes | yes |
| randomforest/Original-vs-GAF | -0.0006 | [-0.0027, +0.0016] | yes | yes |
| randomforest/Original-vs-GADF | -0.0002 | [-0.0024, +0.0019] | yes | yes |
| lstm/Original-vs-RP | +0.0005 | [-0.0017, +0.0026] | yes | yes |
| lstm/Original-vs-GAF | -0.0000 | [-0.0022, +0.0021] | yes | yes |
| lstm/Original-vs-GADF | +0.0001 | [-0.0021, +0.0023] | yes | yes |
| gru/Original-vs-RP | -0.0004 | [-0.0025, +0.0018] | yes | yes |
| gru/Original-vs-GAF | -0.0002 | [-0.0023, +0.0020] | yes | yes |
| gru/Original-vs-GADF | -0.0003 | [-0.0024, +0.0019] | yes | yes |
| cnn/Original-vs-RP | +0.0003 | [-0.0019, +0.0024] | yes | yes |
| cnn/Original-vs-GAF | +0.0003 | [-0.0019, +0.0025] | yes | yes |
| cnn/Original-vs-GADF | +0.0002 | [-0.0020, +0.0023] | yes | yes |
| knn/Original-vs-MTF | +0.0474 | [+0.0212, +0.0736] | no | no |
| svc/Original-vs-MTF | +0.0651 | [+0.0389, +0.0913] | no | no |
| randomforest/Original-vs-MTF | +0.0910 | [+0.0649, +0.1172] | no | no |
| lstm/Original-vs-MTF | +0.0697 | [+0.0435, +0.0959] | no | no |
| gru/Original-vs-MTF | +0.1124 | [+0.0862, +0.1386] | no | no |
| cnn/Original-vs-MTF | +0.1144 | [+0.0882, +0.1405] | no | no |
