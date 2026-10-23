# Statistical significance report (LOTO 3-fold pooled)

Window-level diagnostics use the pooled $N=8172$ test windows from
folds [0, 1, 2] (per-fold sizes [2734, 2685, 2753]). Pairing is preserved,
but windows from one subject are dependent; these tests are therefore
secondary. Section 5 aggregates by subject and provides the primary
equivalence inference. Raw p-values are corrected with Holm--Bonferroni
inside each stated family at $\alpha=0.05$.

## 1. Per-fold accuracy summary (mean +/- std across 3 folds)

| Classifier | Original | RP | MTF | GASF | GADF |
|---|---|---|---|---|---|
| knn | 0.511±0.017 [0.490,0.531] | 0.511±0.017 [0.490,0.532] | 0.471±0.013 [0.452,0.481] | 0.511±0.018 [0.489,0.533] | 0.511±0.017 [0.490,0.532] |
| svc | 0.732±0.013 [0.719,0.751] | 0.732±0.014 [0.720,0.752] | 0.667±0.018 [0.651,0.693] | 0.733±0.013 [0.720,0.751] | 0.732±0.014 [0.720,0.752] |
| randomforest | 0.772±0.008 [0.764,0.783] | 0.773±0.007 [0.764,0.782] | 0.679±0.012 [0.665,0.693] | 0.773±0.008 [0.764,0.784] | 0.772±0.009 [0.762,0.785] |
| lstm | 0.767±0.015 [0.746,0.780] | 0.767±0.015 [0.746,0.781] | 0.701±0.013 [0.682,0.711] | 0.767±0.015 [0.746,0.781] | 0.766±0.015 [0.745,0.780] |
| gru | 0.825±0.012 [0.809,0.837] | 0.825±0.011 [0.809,0.836] | 0.708±0.032 [0.663,0.736] | 0.825±0.011 [0.810,0.836] | 0.825±0.011 [0.810,0.837] |
| cnn | 0.793±0.030 [0.751,0.815] | 0.793±0.030 [0.750,0.816] | 0.685±0.042 [0.628,0.726] | 0.793±0.029 [0.752,0.816] | 0.794±0.030 [0.751,0.816] |

## 2. McNemar Original vs reconstruction (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/Original-vs-RP | 1.000e+00 | 1.000e+00 | no |
| knn/Original-vs-GAF | 3.428e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| knn/Original-vs-MTF | 8.082e-21 | 1.536e-19 | yes |
| svc/Original-vs-RP | 7.103e-01 | 1.000e+00 | no |
| svc/Original-vs-GAF | 1.814e-01 | 1.000e+00 | no |
| svc/Original-vs-GADF | 4.414e-01 | 1.000e+00 | no |
| svc/Original-vs-MTF | 7.442e-40 | 1.488e-38 | yes |
| randomforest/Original-vs-RP | 8.828e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GAF | 2.976e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | 8.918e-01 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | 6.100e-91 | 1.342e-89 | yes |
| lstm/Original-vs-RP | 1.824e-01 | 1.000e+00 | no |
| lstm/Original-vs-GAF | 8.026e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | 1.213e-01 | 1.000e+00 | no |
| lstm/Original-vs-MTF | 2.359e-47 | 4.954e-46 | yes |
| gru/Original-vs-RP | 3.865e-01 | 1.000e+00 | no |
| gru/Original-vs-GAF | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-GADF | 7.237e-01 | 1.000e+00 | no |
| gru/Original-vs-MTF | 6.043e-137 | 1.450e-135 | yes |
| cnn/Original-vs-RP | 1.000e+00 | 1.000e+00 | no |
| cnn/Original-vs-GAF | 6.276e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | 3.428e-01 | 1.000e+00 | no |
| cnn/Original-vs-MTF | 5.997e-126 | 1.379e-124 | yes |

## 3. McNemar pairwise between encoders (Holm)
| Comparison | p (raw) | p (Holm) | Reject H0? |
|---|---|---|---|
| knn/RP-vs-GAF | 5.465e-01 | 1.000e+00 | no |
| knn/RP-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| knn/RP-vs-MTF | 6.383e-21 | 1.213e-19 | yes |
| knn/GAF-vs-GADF | 8.026e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | 3.136e-21 | 6.586e-20 | yes |
| knn/GADF-vs-MTF | 4.343e-21 | 8.686e-20 | yes |
| svc/RP-vs-GAF | 7.194e-01 | 1.000e+00 | no |
| svc/RP-vs-GADF | 8.897e-01 | 1.000e+00 | no |
| svc/RP-vs-MTF | 3.233e-40 | 7.112e-39 | yes |
| svc/GAF-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| svc/GAF-vs-MTF | 1.401e-40 | 3.222e-39 | yes |
| svc/GADF-vs-MTF | 1.324e-40 | 3.178e-39 | yes |
| randomforest/RP-vs-GAF | 4.884e-01 | 1.000e+00 | no |
| randomforest/RP-vs-GADF | 7.119e-01 | 1.000e+00 | no |
| randomforest/RP-vs-MTF | 2.773e-91 | 8.043e-90 | yes |
| randomforest/GAF-vs-GADF | 2.286e-01 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | 4.007e-93 | 1.202e-91 | yes |
| randomforest/GADF-vs-MTF | 2.385e-90 | 6.679e-89 | yes |
| lstm/RP-vs-GAF | 3.320e-01 | 1.000e+00 | no |
| lstm/RP-vs-GADF | 9.522e-03 | 1.714e-01 | no |
| lstm/RP-vs-MTF | 2.600e-48 | 7.019e-47 | yes |
| lstm/GAF-vs-GADF | 1.456e-01 | 1.000e+00 | no |
| lstm/GAF-vs-MTF | 2.359e-47 | 6.134e-46 | yes |
| lstm/GADF-vs-MTF | 3.353e-46 | 8.382e-45 | yes |
| gru/RP-vs-GAF | 6.056e-01 | 1.000e+00 | no |
| gru/RP-vs-GADF | 7.728e-01 | 1.000e+00 | no |
| gru/RP-vs-MTF | 3.501e-136 | 1.190e-134 | yes |
| gru/GAF-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| gru/GAF-vs-MTF | 2.178e-136 | 7.622e-135 | yes |
| gru/GADF-vs-MTF | 1.455e-136 | 5.237e-135 | yes |
| cnn/RP-vs-GAF | 5.403e-01 | 1.000e+00 | no |
| cnn/RP-vs-GADF | 3.588e-01 | 1.000e+00 | no |
| cnn/RP-vs-MTF | 1.128e-126 | 3.495e-125 | yes |
| cnn/GAF-vs-GADF | 1.000e+00 | 1.000e+00 | no |
| cnn/GAF-vs-MTF | 2.967e-127 | 9.496e-126 | yes |
| cnn/GADF-vs-MTF | 5.303e-128 | 1.750e-126 | yes |

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
| knn/Original-vs-RP | -0.287 | 4.299e-01 | 1.000e+00 | no |
| knn/Original-vs-GAF | -0.556 | 1.294e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | +0.000 | 1.000e+00 | 1.000e+00 | no |
| knn/Original-vs-MTF | +0.614 | 2.590e-05 | 4.920e-04 | yes |
| svc/Original-vs-RP | -0.008 | 9.748e-01 | 1.000e+00 | no |
| svc/Original-vs-GAF | -0.396 | 1.639e-01 | 1.000e+00 | no |
| svc/Original-vs-GADF | -0.042 | 8.550e-01 | 1.000e+00 | no |
| svc/Original-vs-MTF | +0.619 | 1.476e-05 | 2.952e-04 | yes |
| randomforest/Original-vs-RP | -0.010 | 9.602e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GAF | -0.329 | 7.678e-02 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | +0.049 | 7.981e-01 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | +0.853 | 3.452e-09 | 7.595e-08 | yes |
| lstm/Original-vs-RP | -0.543 | 1.360e-01 | 1.000e+00 | no |
| lstm/Original-vs-GAF | +0.013 | 9.575e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | +0.378 | 1.568e-01 | 1.000e+00 | no |
| lstm/Original-vs-MTF | +0.772 | 8.827e-08 | 1.854e-06 | yes |
| gru/Original-vs-RP | +0.385 | 1.678e-01 | 1.000e+00 | no |
| gru/Original-vs-GAF | +0.107 | 7.172e-01 | 1.000e+00 | no |
| gru/Original-vs-GADF | +0.000 | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-MTF | +0.867 | 8.762e-10 | 2.103e-08 | yes |
| cnn/Original-vs-RP | +0.039 | 8.740e-01 | 1.000e+00 | no |
| cnn/Original-vs-GAF | -0.051 | 8.388e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | -0.596 | 5.344e-02 | 9.618e-01 | no |
| cnn/Original-vs-MTF | +0.845 | 1.616e-09 | 3.716e-08 | yes |

### 5b. TOST equivalence on per-subject $\Delta$accuracy (margin $\pm0.01$)
Equivalence within the margin is declared when the Holm-adjusted TOST p < 0.05.
| Comparison | mean $\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |
|---|---|---|---|---|
| knn/Original-vs-RP | -0.0002 | 6.135e-32 | 1.288e-30 | yes |
| knn/Original-vs-GAF | -0.0006 | 1.649e-32 | 3.793e-31 | yes |
| knn/Original-vs-GADF | -0.0001 | 1.376e-27 | 2.476e-26 | yes |
| knn/Original-vs-MTF | +0.0496 | 9.999e-01 | 1.000e+00 | no |
| svc/Original-vs-RP | -0.0002 | 2.084e-21 | 2.500e-20 | yes |
| svc/Original-vs-GAF | -0.0008 | 9.673e-27 | 1.644e-25 | yes |
| svc/Original-vs-GADF | -0.0003 | 2.006e-17 | 1.806e-16 | yes |
| svc/Original-vs-MTF | +0.0594 | 1.000e+00 | 1.000e+00 | no |
| randomforest/Original-vs-RP | -0.0000 | 2.950e-18 | 2.950e-17 | yes |
| randomforest/Original-vs-GAF | -0.0011 | 1.663e-11 | 1.164e-10 | yes |
| randomforest/Original-vs-GADF | +0.0004 | 1.323e-16 | 1.058e-15 | yes |
| randomforest/Original-vs-MTF | +0.0900 | 1.000e+00 | 1.000e+00 | no |
| lstm/Original-vs-RP | -0.0006 | 3.271e-32 | 7.195e-31 | yes |
| lstm/Original-vs-GAF | +0.0001 | 1.623e-23 | 2.434e-22 | yes |
| lstm/Original-vs-GADF | +0.0011 | 2.390e-19 | 2.629e-18 | yes |
| lstm/Original-vs-MTF | +0.0678 | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-RP | +0.0005 | 2.772e-26 | 4.436e-25 | yes |
| gru/Original-vs-GAF | +0.0001 | 4.956e-28 | 9.416e-27 | yes |
| gru/Original-vs-GADF | +0.0001 | 1.526e-35 | 3.663e-34 | yes |
| gru/Original-vs-MTF | +0.1249 | 1.000e+00 | 1.000e+00 | no |
| cnn/Original-vs-RP | +0.0000 | 2.677e-22 | 3.481e-21 | yes |
| cnn/Original-vs-GAF | -0.0004 | 4.589e-23 | 6.424e-22 | yes |
| cnn/Original-vs-GADF | -0.0007 | 3.227e-28 | 6.455e-27 | yes |
| cnn/Original-vs-MTF | +0.1129 | 1.000e+00 | 1.000e+00 | no |

### 5c. Pairwise subject-clustered Wilcoxon on per-subject accuracy (between encoders)
| Comparison | effect r | p (raw) | p (Holm) | Reject H0 (difference)? |
|---|---|---|---|---|
| knn/RP-vs-GAF | -0.160 | 6.610e-01 | 1.000e+00 | no |
| knn/RP-vs-GADF | +0.197 | 5.465e-01 | 1.000e+00 | no |
| knn/RP-vs-MTF | +0.606 | 2.662e-05 | 5.090e-04 | yes |
| knn/GAF-vs-GADF | +0.387 | 2.106e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | +0.608 | 2.545e-05 | 5.090e-04 | yes |
| knn/GADF-vs-MTF | +0.612 | 2.219e-05 | 4.660e-04 | yes |
| svc/RP-vs-GAF | -0.154 | 5.497e-01 | 1.000e+00 | no |
| svc/RP-vs-GADF | -0.019 | 9.306e-01 | 1.000e+00 | no |
| svc/RP-vs-MTF | +0.614 | 1.727e-05 | 3.798e-04 | yes |
| svc/GAF-vs-GADF | +0.150 | 4.967e-01 | 1.000e+00 | no |
| svc/GAF-vs-MTF | +0.620 | 1.443e-05 | 3.318e-04 | yes |
| svc/GADF-vs-MTF | +0.632 | 9.556e-06 | 2.293e-04 | yes |
| randomforest/RP-vs-GAF | -0.173 | 3.410e-01 | 1.000e+00 | no |
| randomforest/RP-vs-GADF | +0.136 | 4.649e-01 | 1.000e+00 | no |
| randomforest/RP-vs-MTF | +0.850 | 2.697e-09 | 7.822e-08 | yes |
| randomforest/GAF-vs-GADF | +0.293 | 1.118e-01 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | +0.859 | 1.254e-09 | 4.139e-08 | yes |
| randomforest/GADF-vs-MTF | +0.853 | 3.344e-09 | 9.363e-08 | yes |
| lstm/RP-vs-GAF | +0.337 | 1.796e-01 | 1.000e+00 | no |
| lstm/RP-vs-GADF | +0.572 | 2.473e-02 | 4.451e-01 | no |
| lstm/RP-vs-MTF | +0.775 | 7.870e-08 | 2.125e-06 | yes |
| lstm/GAF-vs-GADF | +0.581 | 3.130e-02 | 5.322e-01 | no |
| lstm/GAF-vs-MTF | +0.769 | 9.879e-08 | 2.568e-06 | yes |
| lstm/GADF-vs-MTF | +0.768 | 1.045e-07 | 2.614e-06 | yes |
| gru/RP-vs-GAF | -0.336 | 2.982e-01 | 1.000e+00 | no |
| gru/RP-vs-GADF | -0.419 | 1.747e-01 | 1.000e+00 | no |
| gru/RP-vs-MTF | +0.867 | 8.776e-10 | 3.154e-08 | yes |
| gru/GAF-vs-GADF | -0.080 | 7.859e-01 | 1.000e+00 | no |
| gru/GAF-vs-MTF | +0.867 | 8.772e-10 | 3.154e-08 | yes |
| gru/GADF-vs-MTF | +0.867 | 8.762e-10 | 3.154e-08 | yes |
| cnn/RP-vs-GAF | -0.110 | 6.039e-01 | 1.000e+00 | no |
| cnn/RP-vs-GADF | -0.373 | 1.157e-01 | 1.000e+00 | no |
| cnn/RP-vs-MTF | +0.845 | 1.617e-09 | 5.169e-08 | yes |
| cnn/GAF-vs-GADF | -0.162 | 5.113e-01 | 1.000e+00 | no |
| cnn/GAF-vs-MTF | +0.845 | 1.617e-09 | 5.169e-08 | yes |
| cnn/GADF-vs-MTF | +0.845 | 1.615e-09 | 5.169e-08 | yes |

### 5d. Pairwise TOST equivalence between encoders (margin $\pm0.01$)
| Comparison | mean $\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |
|---|---|---|---|---|
| knn/RP-vs-GAF | -0.0004 | 1.091e-31 | 3.927e-30 | yes |
| knn/RP-vs-GADF | +0.0000 | 1.772e-26 | 5.669e-25 | yes |
| knn/RP-vs-MTF | +0.0498 | 9.999e-01 | 1.000e+00 | no |
| knn/GAF-vs-GADF | +0.0004 | 2.062e-25 | 6.392e-24 | yes |
| knn/GAF-vs-MTF | +0.0502 | 9.999e-01 | 1.000e+00 | no |
| knn/GADF-vs-MTF | +0.0498 | 9.999e-01 | 1.000e+00 | no |
| svc/RP-vs-GAF | -0.0005 | 1.103e-20 | 2.869e-19 | yes |
| svc/RP-vs-GADF | -0.0001 | 5.379e-14 | 1.183e-12 | yes |
| svc/RP-vs-MTF | +0.0596 | 1.000e+00 | 1.000e+00 | no |
| svc/GAF-vs-GADF | +0.0005 | 3.302e-18 | 7.924e-17 | yes |
| svc/GAF-vs-MTF | +0.0601 | 1.000e+00 | 1.000e+00 | no |
| svc/GADF-vs-MTF | +0.0597 | 1.000e+00 | 1.000e+00 | no |
| randomforest/RP-vs-GAF | -0.0011 | 2.850e-09 | 5.415e-08 | yes |
| randomforest/RP-vs-GADF | +0.0004 | 1.966e-13 | 4.129e-12 | yes |
| randomforest/RP-vs-MTF | +0.0900 | 1.000e+00 | 1.000e+00 | no |
| randomforest/GAF-vs-GADF | +0.0015 | 9.417e-12 | 1.883e-10 | yes |
| randomforest/GAF-vs-MTF | +0.0911 | 1.000e+00 | 1.000e+00 | no |
| randomforest/GADF-vs-MTF | +0.0896 | 1.000e+00 | 1.000e+00 | no |
| lstm/RP-vs-GAF | +0.0007 | 9.956e-23 | 2.788e-21 | yes |
| lstm/RP-vs-GADF | +0.0017 | 3.787e-17 | 8.710e-16 | yes |
| lstm/RP-vs-MTF | +0.0684 | 1.000e+00 | 1.000e+00 | no |
| lstm/GAF-vs-GADF | +0.0009 | 1.477e-24 | 4.430e-23 | yes |
| lstm/GAF-vs-MTF | +0.0677 | 1.000e+00 | 1.000e+00 | no |
| lstm/GADF-vs-MTF | +0.0667 | 1.000e+00 | 1.000e+00 | no |
| gru/RP-vs-GAF | -0.0004 | 1.255e-28 | 4.391e-27 | yes |
| gru/RP-vs-GADF | -0.0004 | 1.987e-28 | 6.756e-27 | yes |
| gru/RP-vs-MTF | +0.1244 | 1.000e+00 | 1.000e+00 | no |
| gru/GAF-vs-GADF | +0.0000 | 3.922e-28 | 1.294e-26 | yes |
| gru/GAF-vs-MTF | +0.1248 | 1.000e+00 | 1.000e+00 | no |
| gru/GADF-vs-MTF | +0.1248 | 1.000e+00 | 1.000e+00 | no |
| cnn/RP-vs-GAF | -0.0004 | 3.065e-19 | 7.662e-18 | yes |
| cnn/RP-vs-GADF | -0.0007 | 8.479e-23 | 2.459e-21 | yes |
| cnn/RP-vs-MTF | +0.1128 | 1.000e+00 | 1.000e+00 | no |
| cnn/GAF-vs-GADF | -0.0003 | 1.278e-21 | 3.449e-20 | yes |
| cnn/GAF-vs-MTF | +0.1133 | 1.000e+00 | 1.000e+00 | no |
| cnn/GADF-vs-MTF | +0.1135 | 1.000e+00 | 1.000e+00 | no |

### 5e. Simultaneous subject-cluster bootstrap 95% intervals on per-subject $\Delta$accuracy (Original - reconstruction)
The primary family contains 18 comparisons (six classifiers by RP/GASF/GADF); the six MTF negative-control comparisons form a separate family. Within each family the same subject resamples are used jointly and a max-deviation critical value controls family-wise coverage. Equivalence is declared when the whole simultaneous interval lies inside the margin. Primary margin $\pm0.01$; the less stringent $\pm0.02$ margin is a sensitivity analysis.
| Comparison | mean $\Delta$acc | simultaneous 95% interval | Equiv. $\pm$0.01 | Equiv. $\pm$0.02 |
|---|---|---|---|---|
| knn/Original-vs-RP | -0.0002 | [-0.0023, +0.0020] | yes | yes |
| knn/Original-vs-GAF | -0.0006 | [-0.0027, +0.0016] | yes | yes |
| knn/Original-vs-GADF | -0.0001 | [-0.0023, +0.0020] | yes | yes |
| svc/Original-vs-RP | -0.0002 | [-0.0024, +0.0019] | yes | yes |
| svc/Original-vs-GAF | -0.0008 | [-0.0029, +0.0014] | yes | yes |
| svc/Original-vs-GADF | -0.0003 | [-0.0025, +0.0019] | yes | yes |
| randomforest/Original-vs-RP | -0.0000 | [-0.0022, +0.0021] | yes | yes |
| randomforest/Original-vs-GAF | -0.0011 | [-0.0033, +0.0011] | yes | yes |
| randomforest/Original-vs-GADF | +0.0004 | [-0.0018, +0.0026] | yes | yes |
| lstm/Original-vs-RP | -0.0006 | [-0.0028, +0.0016] | yes | yes |
| lstm/Original-vs-GAF | +0.0001 | [-0.0021, +0.0023] | yes | yes |
| lstm/Original-vs-GADF | +0.0011 | [-0.0011, +0.0032] | yes | yes |
| gru/Original-vs-RP | +0.0005 | [-0.0017, +0.0027] | yes | yes |
| gru/Original-vs-GAF | +0.0001 | [-0.0021, +0.0023] | yes | yes |
| gru/Original-vs-GADF | +0.0001 | [-0.0021, +0.0023] | yes | yes |
| cnn/Original-vs-RP | +0.0000 | [-0.0021, +0.0022] | yes | yes |
| cnn/Original-vs-GAF | -0.0004 | [-0.0026, +0.0018] | yes | yes |
| cnn/Original-vs-GADF | -0.0007 | [-0.0028, +0.0015] | yes | yes |
| knn/Original-vs-MTF | +0.0496 | [+0.0235, +0.0757] | no | no |
| svc/Original-vs-MTF | +0.0594 | [+0.0333, +0.0855] | no | no |
| randomforest/Original-vs-MTF | +0.0900 | [+0.0639, +0.1161] | no | no |
| lstm/Original-vs-MTF | +0.0678 | [+0.0417, +0.0939] | no | no |
| gru/Original-vs-MTF | +0.1249 | [+0.0988, +0.1510] | no | no |
| cnn/Original-vs-MTF | +0.1129 | [+0.0868, +0.1390] | no | no |
