# RP baseline snapshot: grid-720 orientation + operator-range metadata

Frozen: 2026-07-29T06:46:36Z (UTC)
Windows checkout commit at freeze time: 5ac22f6d7ee23811c9396da5b1166f36ed2930fa
WSL working-copy commit: 0459dd7e0414c13864ec91ccfbdd3046dc8ef8cb
Python: Python 3.10.20
numpy: 1.26.4
opencv: 4.10.0

## Decoder settings that produced these artefacts

| setting | value |
|---|---|
| entry point | `recurrence_plots.Reconstruct_RP(method="direct2ds")` |
| distance source | `direct` (symmetrised magnitudes) |
| PNG inversion | operator-range metadata, per-window `(R_min,R_max)` from `_build_rp_range` |
| orientation search | angular grid, `n_theta=720` x 2 reflections = 1440 candidates |
| signed selection | top_k=8 lowest-overlap candidates, scale-DEPENDENT signed-RP error |
| translation | `delta = -mean(b_i - a_{i+1})`, split symmetrically |

## Artefact correspondence table

| role | grid720 baseline | closed-form (Phase 2 output) |
|---|---|---|
| test reconstructions | `.../fold-<k>/test/X_all_rec_grid720.npy` | `.../fold-<k>/test/X_all_rec.npy` |
| classification reports | `results/_baseline_grid720/RP/<clf>/<sampling>/fold-<k>.json` | `results/RP/<clf>/<sampling>/fold-<k>.json` |
| earlier rho=0.75 point | `.../test/X_all_rec_rho075.npy` | n/a (superseded) |

Do not merge grid720 and closed-form numbers into one table without labelling them.

## Array inventory at freeze time

| sampling | fold | shape | dtype | sha256 (first 16) |
|---|---|---|---|---|
| loto | 0 | (2724, 3, 128) | float64 | `9d74e12d0a7fe244` |
| loto | 1 | (2724, 3, 128) | float64 | `f674b04ab0ac8e13` |
| loto | 2 | (2724, 3, 128) | float64 | `a3efb8a57fc380de` |
| loso | 0 | (2645, 3, 128) | float64 | `9331599668df2fbd` |
| loso | 1 | (2788, 3, 128) | float64 | `96ac7f7a58e1285f` |
| loso | 2 | (2739, 3, 128) | float64 | `ecf9c96fb0e5f90c` |
