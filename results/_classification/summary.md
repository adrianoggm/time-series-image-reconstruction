# M7 classification: classifiers trained on original windows, tested on reconstructions

Accuracy for multiclass data, macro-F1 for multilabel PTB-XL. Grouped 3-fold. Mark ✓: per-group accuracy difference within ±0.01 (95% cluster bootstrap).

## RF

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP |
|---|---|---|---|---|---|---|---|---|---|
| wisdm | 0.750 | 0.636 | 0.726 | 0.659 | 0.751 ✓ | 0.751 ✓ | 0.750 ✓ | 0.659 | 0.750 ✓ |
| pamap2 | 0.689 | 0.545 | 0.625 | 0.569 | 0.691 ✓ | 0.691 ✓ | 0.691 ✓ | 0.593 | 0.689 ✓ |
| esc50 | 0.073 | 0.061 | 0.070 ✓ | 0.073 ✓ | 0.074 ✓ | 0.074 ✓ | 0.073 ✓ | 0.072 | 0.074 ✓ |
| ptbxl | 0.154 | 0.003 | 0.023 | 0.004 | 0.153 | 0.153 | 0.157 | 0.001 | 0.155 |
| stead | 0.658 | 0.652 | 0.661 | 0.648 | 0.658 ✓ | 0.659 ✓ | 0.659 ✓ | 0.624 | 0.659 ✓ |
| emg_gestures | 0.770 | 0.667 | 0.656 | 0.739 | 0.763 | 0.774 | 0.768 | 0.674 | 0.768 |

## CNN

| Dataset | Original | Hirata · RP sin signo | Hirata · Signed RP | RP nueva · RP sin signo | RP nueva · Signed RP | GASF | GADF | MTF | XIRP |
|---|---|---|---|---|---|---|---|---|---|
| wisdm | 0.746 | 0.656 | 0.728 | 0.678 | 0.746 ✓ | 0.746 ✓ | 0.746 ✓ | 0.665 | 0.746 ✓ |
| pamap2 | 0.589 | 0.474 | 0.440 | 0.496 | 0.590 ✓ | 0.590 | 0.588 ✓ | 0.523 | 0.589 ✓ |
| esc50 | 0.104 | 0.081 | 0.086 | 0.098 | 0.104 ✓ | 0.104 ✓ | 0.103 ✓ | 0.085 | 0.104 ✓ |
| ptbxl | 0.589 | 0.216 | 0.308 | 0.227 | 0.590 | 0.590 | 0.590 | 0.171 | 0.589 |
| stead | 0.583 | 0.575 ✓ | 0.578 ✓ | 0.581 ✓ | 0.583 ✓ | 0.583 ✓ | 0.583 ✓ | 0.582 | 0.583 ✓ |
| emg_gestures | 0.806 | 0.527 | 0.599 | 0.635 | 0.807 ✓ | 0.806 ✓ | 0.806 ✓ | 0.664 | 0.806 ✓ |

