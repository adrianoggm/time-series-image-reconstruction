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

## 5. Subject-clustered classification (51 subjects as units)

### 5a. Subject-clustered Wilcoxon on per-subject accuracy, Original vs reconstruction
| Comparison | effect r | p (raw) | p (Holm) | Reject H0 (difference)? |
|---|---|---|---|---|
| knn/Original-vs-RP | +0.428 | 2.932e-01 | 1.000e+00 | no |
| knn/Original-vs-GAF | +0.000 | 1.000e+00 | 1.000e+00 | no |
| knn/Original-vs-GADF | -0.447 | 2.282e-01 | 1.000e+00 | no |
| knn/Original-vs-MTF | +0.600 | 3.258e-05 | 6.190e-04 | yes |
| svc/Original-vs-RP | -0.097 | 7.508e-01 | 1.000e+00 | no |
| svc/Original-vs-GAF | +0.091 | 8.501e-01 | 1.000e+00 | no |
| svc/Original-vs-GADF | +0.339 | 1.575e-01 | 1.000e+00 | no |
| svc/Original-vs-MTF | +0.645 | 9.677e-06 | 1.935e-04 | yes |
| randomforest/Original-vs-RP | +0.413 | 8.451e-02 | 1.000e+00 | no |
| randomforest/Original-vs-GAF | -0.072 | 7.116e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | +0.085 | 6.771e-01 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | +0.835 | 3.563e-09 | 7.838e-08 | yes |
| lstm/Original-vs-RP | +0.161 | 6.056e-01 | 1.000e+00 | no |
| lstm/Original-vs-GAF | -0.128 | 7.498e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | +0.000 | 1.000e+00 | 1.000e+00 | no |
| lstm/Original-vs-MTF | +0.707 | 4.450e-07 | 9.346e-06 | yes |
| gru/Original-vs-RP | -0.181 | 6.845e-01 | 1.000e+00 | no |
| gru/Original-vs-GAF | -0.463 | 4.227e-01 | 1.000e+00 | no |
| gru/Original-vs-GADF | -0.217 | 5.102e-01 | 1.000e+00 | no |
| gru/Original-vs-MTF | +0.853 | 1.106e-09 | 2.654e-08 | yes |
| cnn/Original-vs-RP | +0.297 | 3.979e-01 | 1.000e+00 | no |
| cnn/Original-vs-GAF | -0.210 | 5.037e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | -0.228 | 4.450e-01 | 1.000e+00 | no |
| cnn/Original-vs-MTF | +0.838 | 2.157e-09 | 4.961e-08 | yes |

### 5b. TOST equivalence on per-subject $\Delta$accuracy (margin $\pm0.01$)
Equivalence within the margin is declared when the Holm-adjusted TOST p < 0.05.
| Comparison | mean $\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |
|---|---|---|---|---|
| knn/Original-vs-RP | +0.0003 | 1.630e-32 | 3.423e-31 | yes |
| knn/Original-vs-GAF | -0.0001 | 1.897e-32 | 3.795e-31 | yes |
| knn/Original-vs-GADF | -0.0005 | 6.982e-28 | 1.047e-26 | yes |
| knn/Original-vs-MTF | +0.0474 | 9.999e-01 | 1.000e+00 | no |
| svc/Original-vs-RP | -0.0001 | 5.610e-29 | 9.537e-28 | yes |
| svc/Original-vs-GAF | +0.0000 | 2.439e-37 | 5.367e-36 | yes |
| svc/Original-vs-GADF | +0.0008 | 1.624e-23 | 1.787e-22 | yes |
| svc/Original-vs-MTF | +0.0651 | 1.000e+00 | 1.000e+00 | no |
| randomforest/Original-vs-RP | +0.0012 | 7.686e-18 | 6.917e-17 | yes |
| randomforest/Original-vs-GAF | -0.0006 | 2.545e-13 | 1.782e-12 | yes |
| randomforest/Original-vs-GADF | -0.0002 | 1.480e-14 | 1.184e-13 | yes |
| randomforest/Original-vs-MTF | +0.0910 | 1.000e+00 | 1.000e+00 | no |
| lstm/Original-vs-RP | +0.0004 | 6.574e-25 | 8.546e-24 | yes |
| lstm/Original-vs-GAF | -0.0000 | 3.740e-32 | 7.106e-31 | yes |
| lstm/Original-vs-GADF | +0.0001 | 2.831e-27 | 3.963e-26 | yes |
| lstm/Original-vs-MTF | +0.0697 | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-RP | -0.0002 | 1.073e-37 | 2.467e-36 | yes |
| gru/Original-vs-GAF | -0.0002 | 1.036e-40 | 2.486e-39 | yes |
| gru/Original-vs-GADF | -0.0003 | 6.813e-30 | 1.226e-28 | yes |
| gru/Original-vs-MTF | +0.1124 | 1.000e+00 | 1.000e+00 | no |
| cnn/Original-vs-RP | +0.0004 | 4.293e-28 | 6.869e-27 | yes |
| cnn/Original-vs-GAF | -0.0002 | 4.323e-22 | 4.323e-21 | yes |
| cnn/Original-vs-GADF | -0.0001 | 8.381e-25 | 1.006e-23 | yes |
| cnn/Original-vs-MTF | +0.1102 | 1.000e+00 | 1.000e+00 | no |

### 5c. Pairwise subject-clustered Wilcoxon on per-subject accuracy (between encoders)
| Comparison | effect r | p (raw) | p (Holm) | Reject H0 (difference)? |
|---|---|---|---|---|
| knn/RP-vs-GAF | -0.214 | 4.758e-01 | 1.000e+00 | no |
| knn/RP-vs-GADF | -0.612 | 5.125e-02 | 9.225e-01 | no |
| knn/RP-vs-MTF | +0.603 | 2.983e-05 | 6.263e-04 | yes |
| knn/GAF-vs-GADF | -0.276 | 4.028e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | +0.597 | 3.562e-05 | 6.768e-04 | yes |
| knn/GADF-vs-MTF | +0.603 | 2.983e-05 | 6.263e-04 | yes |
| svc/RP-vs-GAF | +0.109 | 6.703e-01 | 1.000e+00 | no |
| svc/RP-vs-GADF | +0.239 | 2.556e-01 | 1.000e+00 | no |
| svc/RP-vs-MTF | +0.647 | 9.199e-06 | 2.208e-04 | yes |
| svc/GAF-vs-GADF | +0.381 | 1.352e-01 | 1.000e+00 | no |
| svc/GAF-vs-MTF | +0.646 | 9.434e-06 | 2.208e-04 | yes |
| svc/GADF-vs-MTF | +0.637 | 1.027e-05 | 2.260e-04 | yes |
| randomforest/RP-vs-GAF | -0.339 | 8.299e-02 | 1.000e+00 | no |
| randomforest/RP-vs-GADF | -0.153 | 4.396e-01 | 1.000e+00 | no |
| randomforest/RP-vs-MTF | +0.838 | 4.505e-09 | 1.261e-07 | yes |
| randomforest/GAF-vs-GADF | +0.128 | 5.260e-01 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | +0.849 | 1.910e-09 | 6.302e-08 | yes |
| randomforest/GADF-vs-MTF | +0.837 | 3.252e-09 | 9.430e-08 | yes |
| lstm/RP-vs-GAF | -0.121 | 6.855e-01 | 1.000e+00 | no |
| lstm/RP-vs-GADF | -0.145 | 5.957e-01 | 1.000e+00 | no |
| lstm/RP-vs-MTF | +0.713 | 4.659e-07 | 1.165e-05 | yes |
| lstm/GAF-vs-GADF | +0.079 | 7.817e-01 | 1.000e+00 | no |
| lstm/GAF-vs-MTF | +0.709 | 4.034e-07 | 1.089e-05 | yes |
| lstm/GADF-vs-MTF | +0.707 | 4.344e-07 | 1.130e-05 | yes |
| gru/RP-vs-GAF | -0.124 | 7.250e-01 | 1.000e+00 | no |
| gru/RP-vs-GADF | -0.020 | 9.523e-01 | 1.000e+00 | no |
| gru/RP-vs-MTF | +0.854 | 1.075e-09 | 3.869e-08 | yes |
| gru/GAF-vs-GADF | -0.074 | 8.326e-01 | 1.000e+00 | no |
| gru/GAF-vs-MTF | +0.853 | 1.105e-09 | 3.869e-08 | yes |
| gru/GADF-vs-MTF | +0.850 | 1.279e-09 | 4.349e-08 | yes |
| cnn/RP-vs-GAF | -0.308 | 2.240e-01 | 1.000e+00 | no |
| cnn/RP-vs-GADF | -0.436 | 1.115e-01 | 1.000e+00 | no |
| cnn/RP-vs-MTF | +0.839 | 2.035e-09 | 6.513e-08 | yes |
| cnn/GAF-vs-GADF | -0.025 | 9.434e-01 | 1.000e+00 | no |
| cnn/GAF-vs-MTF | +0.837 | 2.283e-09 | 7.076e-08 | yes |
| cnn/GADF-vs-MTF | +0.837 | 2.284e-09 | 7.076e-08 | yes |

### 5d. Pairwise TOST equivalence between encoders (margin $\pm0.01$)
| Comparison | mean $\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |
|---|---|---|---|---|
| knn/RP-vs-GAF | -0.0003 | 7.640e-24 | 2.063e-22 | yes |
| knn/RP-vs-GADF | -0.0008 | 3.378e-23 | 8.445e-22 | yes |
| knn/RP-vs-MTF | +0.0471 | 9.999e-01 | 1.000e+00 | no |
| knn/GAF-vs-GADF | -0.0005 | 2.434e-25 | 7.546e-24 | yes |
| knn/GAF-vs-MTF | +0.0475 | 9.999e-01 | 1.000e+00 | no |
| knn/GADF-vs-MTF | +0.0479 | 9.999e-01 | 1.000e+00 | no |
| svc/RP-vs-GAF | +0.0001 | 2.479e-25 | 7.546e-24 | yes |
| svc/RP-vs-GADF | +0.0009 | 7.132e-18 | 1.569e-16 | yes |
| svc/RP-vs-MTF | +0.0652 | 1.000e+00 | 1.000e+00 | no |
| svc/GAF-vs-GADF | +0.0008 | 4.684e-25 | 1.358e-23 | yes |
| svc/GAF-vs-MTF | +0.0651 | 1.000e+00 | 1.000e+00 | no |
| svc/GADF-vs-MTF | +0.0642 | 1.000e+00 | 1.000e+00 | no |
| randomforest/RP-vs-GAF | -0.0017 | 3.263e-11 | 6.526e-10 | yes |
| randomforest/RP-vs-GADF | -0.0014 | 1.793e-09 | 3.406e-08 | yes |
| randomforest/RP-vs-MTF | +0.0899 | 1.000e+00 | 1.000e+00 | no |
| randomforest/GAF-vs-GADF | +0.0004 | 1.538e-12 | 3.229e-11 | yes |
| randomforest/GAF-vs-MTF | +0.0916 | 1.000e+00 | 1.000e+00 | no |
| randomforest/GADF-vs-MTF | +0.0913 | 1.000e+00 | 1.000e+00 | no |
| lstm/RP-vs-GAF | -0.0004 | 5.821e-21 | 1.339e-19 | yes |
| lstm/RP-vs-GADF | -0.0003 | 5.803e-22 | 1.393e-20 | yes |
| lstm/RP-vs-MTF | +0.0693 | 1.000e+00 | 1.000e+00 | no |
| lstm/GAF-vs-GADF | +0.0001 | 1.056e-26 | 3.378e-25 | yes |
| lstm/GAF-vs-MTF | +0.0698 | 1.000e+00 | 1.000e+00 | no |
| lstm/GADF-vs-MTF | +0.0696 | 1.000e+00 | 1.000e+00 | no |
| gru/RP-vs-GAF | +0.0000 | 4.263e-32 | 1.492e-30 | yes |
| gru/RP-vs-GADF | -0.0001 | 7.356e-28 | 2.427e-26 | yes |
| gru/RP-vs-MTF | +0.1126 | 1.000e+00 | 1.000e+00 | no |
| gru/GAF-vs-GADF | -0.0001 | 9.593e-32 | 3.262e-30 | yes |
| gru/GAF-vs-MTF | +0.1125 | 1.000e+00 | 1.000e+00 | no |
| gru/GADF-vs-MTF | +0.1126 | 1.000e+00 | 1.000e+00 | no |
| cnn/RP-vs-GAF | -0.0006 | 2.742e-23 | 7.129e-22 | yes |
| cnn/RP-vs-GADF | -0.0005 | 5.467e-25 | 1.531e-23 | yes |
| cnn/RP-vs-MTF | +0.1098 | 1.000e+00 | 1.000e+00 | no |
| cnn/GAF-vs-GADF | +0.0001 | 9.629e-33 | 3.467e-31 | yes |
| cnn/GAF-vs-MTF | +0.1105 | 1.000e+00 | 1.000e+00 | no |
| cnn/GADF-vs-MTF | +0.1104 | 1.000e+00 | 1.000e+00 | no |

### 5e. Simultaneous subject-cluster bootstrap 95% intervals on per-subject $\Delta$accuracy (Original - reconstruction)
The primary family contains 18 comparisons (six classifiers by RP/GASF/GADF); the six MTF negative-control comparisons form a separate family. Within each family the same subject resamples are used jointly and a max-deviation critical value controls family-wise coverage. Equivalence is declared when the whole simultaneous interval lies inside the margin. Primary margin $\pm0.01$; the less stringent $\pm0.02$ margin is a sensitivity analysis.
| Comparison | mean $\Delta$acc | simultaneous 95% interval | Equiv. $\pm$0.01 | Equiv. $\pm$0.02 |
|---|---|---|---|---|
| knn/Original-vs-RP | +0.0003 | [-0.0019, +0.0024] | yes | yes |
| knn/Original-vs-GAF | -0.0001 | [-0.0022, +0.0021] | yes | yes |
| knn/Original-vs-GADF | -0.0005 | [-0.0027, +0.0016] | yes | yes |
| svc/Original-vs-RP | -0.0001 | [-0.0022, +0.0020] | yes | yes |
| svc/Original-vs-GAF | +0.0000 | [-0.0021, +0.0021] | yes | yes |
| svc/Original-vs-GADF | +0.0008 | [-0.0013, +0.0030] | yes | yes |
| randomforest/Original-vs-RP | +0.0012 | [-0.0009, +0.0033] | yes | yes |
| randomforest/Original-vs-GAF | -0.0006 | [-0.0027, +0.0016] | yes | yes |
| randomforest/Original-vs-GADF | -0.0002 | [-0.0023, +0.0019] | yes | yes |
| lstm/Original-vs-RP | +0.0004 | [-0.0017, +0.0025] | yes | yes |
| lstm/Original-vs-GAF | -0.0000 | [-0.0022, +0.0021] | yes | yes |
| lstm/Original-vs-GADF | +0.0001 | [-0.0020, +0.0022] | yes | yes |
| gru/Original-vs-RP | -0.0002 | [-0.0023, +0.0019] | yes | yes |
| gru/Original-vs-GAF | -0.0002 | [-0.0023, +0.0020] | yes | yes |
| gru/Original-vs-GADF | -0.0003 | [-0.0024, +0.0019] | yes | yes |
| cnn/Original-vs-RP | +0.0004 | [-0.0017, +0.0025] | yes | yes |
| cnn/Original-vs-GAF | -0.0002 | [-0.0024, +0.0019] | yes | yes |
| cnn/Original-vs-GADF | -0.0001 | [-0.0023, +0.0020] | yes | yes |
| knn/Original-vs-MTF | +0.0474 | [+0.0209, +0.0739] | no | no |
| svc/Original-vs-MTF | +0.0651 | [+0.0385, +0.0916] | no | no |
| randomforest/Original-vs-MTF | +0.0910 | [+0.0645, +0.1176] | no | no |
| lstm/Original-vs-MTF | +0.0697 | [+0.0432, +0.0963] | no | no |
| gru/Original-vs-MTF | +0.1124 | [+0.0858, +0.1389] | no | no |
| cnn/Original-vs-MTF | +0.1102 | [+0.0837, +0.1368] | no | no |
