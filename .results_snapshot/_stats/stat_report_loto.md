# Statistical significance report (LOTO 3-fold pooled)

Window-level diagnostics use the pooled $N=8172$ test windows from
folds [0, 1, 2] (per-fold sizes [2724, 2724, 2724]). Pairing is preserved,
but windows from one subject are dependent; these tests are therefore
secondary. Section 5 aggregates by subject and provides the primary
equivalence inference. Raw p-values are corrected with Holm--Bonferroni
inside each stated family at $\alpha=0.05$.

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

## 5. Subject-clustered classification (51 subjects as units)

### 5a. Subject-clustered Wilcoxon on per-subject accuracy, Original vs reconstruction
| Comparison | effect r | p (raw) | p (Holm) | Reject H0 (difference)? |
|---|---|---|---|---|
| knn/Original-vs-RP | +0.074 | 8.326e-01 | 1.000e+00 | no |
| knn/Original-vs-GAF | -0.257 | 5.282e-01 | 1.000e+00 | no |
| knn/Original-vs-GADF | -0.241 | 4.190e-01 | 1.000e+00 | no |
| knn/Original-vs-MTF | +0.819 | 4.921e-09 | 9.842e-08 | yes |
| svc/Original-vs-RP | -0.081 | 7.488e-01 | 1.000e+00 | no |
| svc/Original-vs-GAF | -0.079 | 8.030e-01 | 1.000e+00 | no |
| svc/Original-vs-GADF | -0.159 | 4.759e-01 | 1.000e+00 | no |
| svc/Original-vs-MTF | +0.725 | 2.282e-07 | 4.335e-06 | yes |
| randomforest/Original-vs-RP | -0.349 | 1.283e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GAF | +0.048 | 8.182e-01 | 1.000e+00 | no |
| randomforest/Original-vs-GADF | -0.009 | 9.613e-01 | 1.000e+00 | no |
| randomforest/Original-vs-MTF | +0.870 | 5.267e-10 | 1.264e-08 | yes |
| lstm/Original-vs-RP | +0.000 | 1.000e+00 | 1.000e+00 | no |
| lstm/Original-vs-GAF | +0.129 | 5.910e-01 | 1.000e+00 | no |
| lstm/Original-vs-GADF | -0.039 | 8.868e-01 | 1.000e+00 | no |
| lstm/Original-vs-MTF | +0.867 | 5.927e-10 | 1.304e-08 | yes |
| gru/Original-vs-RP | +0.415 | 2.402e-01 | 1.000e+00 | no |
| gru/Original-vs-GAF | +0.724 | 9.929e-02 | 1.000e+00 | no |
| gru/Original-vs-GADF | +0.422 | 3.408e-01 | 1.000e+00 | no |
| gru/Original-vs-MTF | +0.870 | 5.277e-10 | 1.264e-08 | yes |
| cnn/Original-vs-RP | -0.419 | 1.794e-01 | 1.000e+00 | no |
| cnn/Original-vs-GAF | +0.290 | 3.464e-01 | 1.000e+00 | no |
| cnn/Original-vs-GADF | -0.607 | 1.052e-01 | 1.000e+00 | no |
| cnn/Original-vs-MTF | +0.870 | 1.137e-09 | 2.387e-08 | yes |

### 5b. TOST equivalence on per-subject $\Delta$accuracy (margin $\pm0.01$)
Equivalence within the margin is declared when the Holm-adjusted TOST p < 0.05.
| Comparison | mean $\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |
|---|---|---|---|---|
| knn/Original-vs-RP | -0.0001 | 2.750e-32 | 5.776e-31 | yes |
| knn/Original-vs-GAF | -0.0003 | 6.293e-35 | 1.447e-33 | yes |
| knn/Original-vs-GADF | -0.0006 | 3.207e-27 | 4.810e-26 | yes |
| knn/Original-vs-MTF | +0.0819 | 1.000e+00 | 1.000e+00 | no |
| svc/Original-vs-RP | -0.0004 | 1.132e-22 | 1.359e-21 | yes |
| svc/Original-vs-GAF | -0.0001 | 6.707e-30 | 1.274e-28 | yes |
| svc/Original-vs-GADF | -0.0003 | 3.708e-18 | 3.337e-17 | yes |
| svc/Original-vs-MTF | +0.0857 | 1.000e+00 | 1.000e+00 | no |
| randomforest/Original-vs-RP | -0.0009 | 4.566e-19 | 4.566e-18 | yes |
| randomforest/Original-vs-GAF | +0.0002 | 4.165e-12 | 2.916e-11 | yes |
| randomforest/Original-vs-GADF | -0.0001 | 6.246e-14 | 4.997e-13 | yes |
| randomforest/Original-vs-MTF | +0.1186 | 1.000e+00 | 1.000e+00 | no |
| lstm/Original-vs-RP | +0.0000 | 1.296e-24 | 1.685e-23 | yes |
| lstm/Original-vs-GAF | +0.0003 | 1.813e-22 | 1.994e-21 | yes |
| lstm/Original-vs-GADF | -0.0001 | 2.074e-26 | 2.903e-25 | yes |
| lstm/Original-vs-MTF | +0.1008 | 1.000e+00 | 1.000e+00 | no |
| gru/Original-vs-RP | +0.0004 | 5.915e-31 | 1.183e-29 | yes |
| gru/Original-vs-GAF | +0.0004 | 1.201e-34 | 2.642e-33 | yes |
| gru/Original-vs-GADF | +0.0002 | 1.030e-35 | 2.472e-34 | yes |
| gru/Original-vs-MTF | +0.1542 | 1.000e+00 | 1.000e+00 | no |
| cnn/Original-vs-RP | -0.0005 | 3.442e-28 | 6.196e-27 | yes |
| cnn/Original-vs-GAF | +0.0005 | 1.123e-27 | 1.909e-26 | yes |
| cnn/Original-vs-GADF | -0.0006 | 1.307e-27 | 2.091e-26 | yes |
| cnn/Original-vs-MTF | +0.1388 | 1.000e+00 | 1.000e+00 | no |

### 5c. Pairwise subject-clustered Wilcoxon on per-subject accuracy (between encoders)
| Comparison | effect r | p (raw) | p (Holm) | Reject H0 (difference)? |
|---|---|---|---|---|
| knn/RP-vs-GAF | -0.102 | 7.216e-01 | 1.000e+00 | no |
| knn/RP-vs-GADF | -0.396 | 1.663e-01 | 1.000e+00 | no |
| knn/RP-vs-MTF | +0.820 | 4.783e-09 | 1.052e-07 | yes |
| knn/GAF-vs-GADF | -0.147 | 6.196e-01 | 1.000e+00 | no |
| knn/GAF-vs-MTF | +0.822 | 4.398e-09 | 1.011e-07 | yes |
| knn/GADF-vs-MTF | +0.823 | 4.141e-09 | 9.939e-08 | yes |
| svc/RP-vs-GAF | +0.026 | 9.153e-01 | 1.000e+00 | no |
| svc/RP-vs-GADF | +0.006 | 9.756e-01 | 1.000e+00 | no |
| svc/RP-vs-MTF | +0.732 | 1.731e-07 | 3.635e-06 | yes |
| svc/GAF-vs-GADF | -0.144 | 5.045e-01 | 1.000e+00 | no |
| svc/GAF-vs-MTF | +0.727 | 2.063e-07 | 3.921e-06 | yes |
| svc/GADF-vs-MTF | +0.730 | 1.820e-07 | 3.640e-06 | yes |
| randomforest/RP-vs-GAF | +0.234 | 2.319e-01 | 1.000e+00 | no |
| randomforest/RP-vs-GADF | +0.123 | 5.112e-01 | 1.000e+00 | no |
| randomforest/RP-vs-MTF | +0.870 | 5.275e-10 | 1.868e-08 | yes |
| randomforest/GAF-vs-GADF | +0.002 | 9.908e-01 | 1.000e+00 | no |
| randomforest/GAF-vs-MTF | +0.870 | 5.188e-10 | 1.868e-08 | yes |
| randomforest/GADF-vs-MTF | +0.870 | 5.254e-10 | 1.868e-08 | yes |
| lstm/RP-vs-GAF | +0.051 | 8.208e-01 | 1.000e+00 | no |
| lstm/RP-vs-GADF | -0.128 | 5.822e-01 | 1.000e+00 | no |
| lstm/RP-vs-MTF | +0.867 | 5.913e-10 | 1.868e-08 | yes |
| lstm/GAF-vs-GADF | -0.155 | 4.686e-01 | 1.000e+00 | no |
| lstm/GAF-vs-MTF | +0.867 | 5.940e-10 | 1.868e-08 | yes |
| lstm/GADF-vs-MTF | +0.867 | 5.942e-10 | 1.868e-08 | yes |
| gru/RP-vs-GAF | +0.086 | 8.241e-01 | 1.000e+00 | no |
| gru/RP-vs-GADF | -0.025 | 9.416e-01 | 1.000e+00 | no |
| gru/RP-vs-MTF | +0.870 | 5.276e-10 | 1.868e-08 | yes |
| gru/GAF-vs-GADF | -0.171 | 6.716e-01 | 1.000e+00 | no |
| gru/GAF-vs-MTF | +0.870 | 5.276e-10 | 1.868e-08 | yes |
| gru/GADF-vs-MTF | +0.870 | 5.275e-10 | 1.868e-08 | yes |
| cnn/RP-vs-GAF | +0.427 | 8.103e-02 | 1.000e+00 | no |
| cnn/RP-vs-GADF | -0.078 | 7.741e-01 | 1.000e+00 | no |
| cnn/RP-vs-MTF | +0.870 | 7.759e-10 | 2.095e-08 | yes |
| cnn/GAF-vs-GADF | -0.632 | 5.278e-02 | 9.500e-01 | no |
| cnn/GAF-vs-MTF | +0.870 | 1.134e-09 | 2.949e-08 | yes |
| cnn/GADF-vs-MTF | +0.870 | 1.136e-09 | 2.949e-08 | yes |

### 5d. Pairwise TOST equivalence between encoders (margin $\pm0.01$)
| Comparison | mean $\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |
|---|---|---|---|---|
| knn/RP-vs-GAF | -0.0002 | 1.619e-26 | 5.018e-25 | yes |
| knn/RP-vs-GADF | -0.0005 | 1.184e-27 | 3.790e-26 | yes |
| knn/RP-vs-MTF | +0.0821 | 1.000e+00 | 1.000e+00 | no |
| knn/GAF-vs-GADF | -0.0003 | 3.997e-28 | 1.319e-26 | yes |
| knn/GAF-vs-MTF | +0.0823 | 1.000e+00 | 1.000e+00 | no |
| knn/GADF-vs-MTF | +0.0825 | 1.000e+00 | 1.000e+00 | no |
| svc/RP-vs-GAF | +0.0004 | 9.348e-22 | 2.524e-20 | yes |
| svc/RP-vs-GADF | +0.0001 | 1.214e-16 | 2.672e-15 | yes |
| svc/RP-vs-MTF | +0.0861 | 1.000e+00 | 1.000e+00 | no |
| svc/GAF-vs-GADF | -0.0003 | 1.087e-19 | 2.608e-18 | yes |
| svc/GAF-vs-MTF | +0.0858 | 1.000e+00 | 1.000e+00 | no |
| svc/GADF-vs-MTF | +0.0860 | 1.000e+00 | 1.000e+00 | no |
| randomforest/RP-vs-GAF | +0.0011 | 7.060e-12 | 1.412e-10 | yes |
| randomforest/RP-vs-GADF | +0.0008 | 1.794e-11 | 3.408e-10 | yes |
| randomforest/RP-vs-MTF | +0.1196 | 1.000e+00 | 1.000e+00 | no |
| randomforest/GAF-vs-GADF | -0.0003 | 1.436e-13 | 3.015e-12 | yes |
| randomforest/GAF-vs-MTF | +0.1185 | 1.000e+00 | 1.000e+00 | no |
| randomforest/GADF-vs-MTF | +0.1187 | 1.000e+00 | 1.000e+00 | no |
| lstm/RP-vs-GAF | +0.0003 | 2.656e-20 | 6.904e-19 | yes |
| lstm/RP-vs-GADF | -0.0001 | 2.954e-20 | 7.386e-19 | yes |
| lstm/RP-vs-MTF | +0.1008 | 1.000e+00 | 1.000e+00 | no |
| lstm/GAF-vs-GADF | -0.0004 | 2.711e-18 | 6.236e-17 | yes |
| lstm/GAF-vs-MTF | +0.1005 | 1.000e+00 | 1.000e+00 | no |
| lstm/GADF-vs-MTF | +0.1009 | 1.000e+00 | 1.000e+00 | no |
| gru/RP-vs-GAF | +0.0000 | 3.020e-33 | 1.057e-31 | yes |
| gru/RP-vs-GADF | -0.0002 | 2.164e-30 | 7.356e-29 | yes |
| gru/RP-vs-MTF | +0.1537 | 1.000e+00 | 1.000e+00 | no |
| gru/GAF-vs-GADF | -0.0002 | 1.113e-33 | 4.007e-32 | yes |
| gru/GAF-vs-MTF | +0.1537 | 1.000e+00 | 1.000e+00 | no |
| gru/GADF-vs-MTF | +0.1540 | 1.000e+00 | 1.000e+00 | no |
| cnn/RP-vs-GAF | +0.0011 | 2.390e-22 | 6.692e-21 | yes |
| cnn/RP-vs-GADF | -0.0001 | 1.281e-23 | 3.716e-22 | yes |
| cnn/RP-vs-MTF | +0.1393 | 1.000e+00 | 1.000e+00 | no |
| cnn/GAF-vs-GADF | -0.0011 | 5.666e-25 | 1.700e-23 | yes |
| cnn/GAF-vs-MTF | +0.1383 | 1.000e+00 | 1.000e+00 | no |
| cnn/GADF-vs-MTF | +0.1394 | 1.000e+00 | 1.000e+00 | no |

### 5e. Simultaneous subject-cluster bootstrap 95% intervals on per-subject $\Delta$accuracy (Original - reconstruction)
The primary family contains 18 comparisons (six classifiers by RP/GASF/GADF); the six MTF negative-control comparisons form a separate family. Within each family the same subject resamples are used jointly and a max-deviation critical value controls family-wise coverage. Equivalence is declared when the whole simultaneous interval lies inside the margin. Primary margin $\pm0.01$; the less stringent $\pm0.02$ margin is a sensitivity analysis.
| Comparison | mean $\Delta$acc | simultaneous 95% interval | Equiv. $\pm$0.01 | Equiv. $\pm$0.02 |
|---|---|---|---|---|
| knn/Original-vs-RP | -0.0001 | [-0.0024, +0.0022] | yes | yes |
| knn/Original-vs-GAF | -0.0003 | [-0.0026, +0.0020] | yes | yes |
| knn/Original-vs-GADF | -0.0006 | [-0.0029, +0.0017] | yes | yes |
| svc/Original-vs-RP | -0.0004 | [-0.0027, +0.0019] | yes | yes |
| svc/Original-vs-GAF | -0.0001 | [-0.0024, +0.0022] | yes | yes |
| svc/Original-vs-GADF | -0.0003 | [-0.0026, +0.0020] | yes | yes |
| randomforest/Original-vs-RP | -0.0009 | [-0.0032, +0.0014] | yes | yes |
| randomforest/Original-vs-GAF | +0.0002 | [-0.0021, +0.0025] | yes | yes |
| randomforest/Original-vs-GADF | -0.0001 | [-0.0024, +0.0022] | yes | yes |
| lstm/Original-vs-RP | +0.0000 | [-0.0023, +0.0023] | yes | yes |
| lstm/Original-vs-GAF | +0.0003 | [-0.0020, +0.0026] | yes | yes |
| lstm/Original-vs-GADF | -0.0001 | [-0.0024, +0.0022] | yes | yes |
| gru/Original-vs-RP | +0.0004 | [-0.0019, +0.0027] | yes | yes |
| gru/Original-vs-GAF | +0.0004 | [-0.0019, +0.0027] | yes | yes |
| gru/Original-vs-GADF | +0.0002 | [-0.0021, +0.0025] | yes | yes |
| cnn/Original-vs-RP | -0.0005 | [-0.0028, +0.0018] | yes | yes |
| cnn/Original-vs-GAF | +0.0005 | [-0.0018, +0.0028] | yes | yes |
| cnn/Original-vs-GADF | -0.0006 | [-0.0029, +0.0017] | yes | yes |
| knn/Original-vs-MTF | +0.0819 | [+0.0550, +0.1089] | no | no |
| svc/Original-vs-MTF | +0.0857 | [+0.0587, +0.1127] | no | no |
| randomforest/Original-vs-MTF | +0.1186 | [+0.0917, +0.1456] | no | no |
| lstm/Original-vs-MTF | +0.1008 | [+0.0738, +0.1278] | no | no |
| gru/Original-vs-MTF | +0.1542 | [+0.1272, +0.1812] | no | no |
| cnn/Original-vs-MTF | +0.1388 | [+0.1118, +0.1658] | no | no |
