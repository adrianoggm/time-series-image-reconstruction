# RP translation-offset ablation after MDS

Date: 2026-07-28

## Mensaje breve para correo

Hemos comprobado si el pequeño error de reconstrucción de RP podía deberse a
que MDS deja una traslación diferente en sus dos coordenadas. Esa traslación ya
se corrige explícitamente en el decodificador mediante la condición de
solapamiento entre estados consecutivos.

La prueba confirma que la corrección funciona: si la quitamos, el error aumenta.
Por tanto, el error que queda no viene de la traslación. Su causa principal es
que la orientación se busca probando una rejilla de 720 ángulos, en lugar de
calcular el ángulo óptimo exacto. Al usar la solución exacta, el error en coma
flotante baja prácticamente a cero.

En resumen: la traslación ya está bien resuelta. La mejora estaba en el cálculo
de la orientación y ya se ha implementado: elimina el error de rejilla en coma
flotante, mejora el PNG limpio y no empeora de forma apreciable el caso con
ruido.

## Short conclusion

The small floating-point reconstruction error does **not** come from an
uncorrected relative translation between the two MDS coordinates. The current
production decoder already estimates and removes that translation using the
adjacent-state overlap condition.

Removing this correction makes the result worse. The remaining floating-point
error comes mainly from searching the orientation on a finite grid of 720
angles. Replacing that grid with the closed-form optimum reduces the error to
machine precision on the native-float test.

The closed-form orientation was implemented as the production default on
2026-07-28. The previous grid remains available through
`orientation_method="grid"` for regression checks and ablations.

## Current correction

After applying a candidate MDS orientation, the decoder computes

```text
d_i = second_coordinate(i) - first_coordinate(i + 1)
delta = -mean(d_i)
```

and splits `delta` symmetrically between the two coordinates. This is the
least-squares optimum for their relative translation. It is implemented in
`src/rp_decoders.py::align_states` and used by the production `direct2ds`
decoder.

Only a common global translation remains undetermined. That is unavoidable
because pairwise distances are invariant to a constant offset, and it is fixed
later by the known min--max amplitude range.

## Native-float ablation

The check used 50 non-overlapping smartwatch-accelerometer WISDM windows:
10 windows from each activity A--E and all three axes, for a total of 150 scalar
series. RP matrices were evaluated directly in floating point, without PNG
quantisation.

All variants retained the two MDS dimensions and used the signed RP to choose
the orientation.

| Variant | Mean NMAE | Median NMAE | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| 720-angle grid, without relative-translation correction | 0.0004450 | 0.0002764 | 0.0013893 | 0.0033394 |
| 720-angle grid, with the current correction | 0.0003760 | 0.0002471 | 0.0012833 | 0.0033394 |
| Current correction, with closed-form orientation | 8.40e-17 | 6.95e-17 | 1.89e-16 | 9.32e-16 |

The current translation correction improves the mean NMAE by approximately
15.5%. It also reduces the mean overlap residual from about `7.40e-5` to
`6.64e-6`, approximately one order of magnitude.

The observed mean NMAE with the current correction is also consistent with the
larger stored robustness result (`0.0003553` in
`.results_snapshot/_stats/robustness_summary.json`).

## Integrated decoder validation

After implementing the closed-form solution, the complete decoder was compared
with the previous 720-angle grid on 10 activity-balanced WISDM windows (two per
activity A--E), all three axes, for a total of 30 scalar series. PNG tests used
metadata-free diagonal centring. Perturbed PNGs used independent Gaussian pixel
noise with `sigma=8`, clipping and integer rerounding.

| Input | Grid mean NMAE | Closed-form mean NMAE |
|---|---:|---:|
| Native float RP | 0.0003149 | 1.08e-16 |
| Clean 8-bit PNG | 0.0011355 | 0.0008385 |
| Perturbed PNG, sigma=8 | 0.0170998 | 0.0171020 |

The exact orientation removes the native-float grid error, improves clean-PNG
NMAE by about 26%, and is effectively neutral under sigma-8 perturbation. The
median complete-decoder runtime fell from 188.4 to 86.0 ms per axis on the same
machine (about 2.2x faster). This total includes MDS and signed-RP candidate
selection, not only the orientation calculation.

## Interpretation

The translation hypothesis is therefore already handled and cannot explain the
remaining error. The dominant native-float limitation is the angular resolution
of the `n_theta=720` orientation scan (0.5 degrees per step).

For each reflection parity, the overlap residual can instead be written as a
two-variable quadratic form. The eigenvector associated with the smallest
eigenvalue of its 2x2 matrix gives the optimum orientation directly. In the
test above, this closed-form solution reduced the native-float error to numerical
round-off.

The implementation retains both reflection parities and both global signs, so
the signed-RP comparison still chooses between four exact candidates. The old
grid path remains selectable for future sensitivity analyses.

## Paper-facing confirmation (150 windows)

The final controlled comparison uses 150 activity-balanced fold-0 windows
(450 axis-series) and keeps the metadata-free diagonal inversion fixed in both
arms. Its main results are:

| Input | Grid-720 NMAE | Closed-form NMAE |
|---|---:|---:|
| Native float RP | 0.000352 | 9.1e-17 |
| Clean 8-bit PNG | 0.001201 | 0.001014 |
| PNG, sigma=8 | 0.017165 | 0.017094 |

The reported 0.60 versus 0.27 seconds per axis-series is total decode time,
including MDS, rather than orientation-only time. The overlap value returned by
`align_states` is a normalised diagnostic; the closed-form eigensystem minimises
its unnormalised quadratic numerator. The cycle residual does not improve in
every condition (0.0533 versus 0.0512 on clean PNG and 0.1627 versus 0.1596 at
sigma=8), so no cycle-residual improvement is claimed.

## Secondary implementation note

`src/recurrence_plots.py::reconstruct_from_embedding_2d` is an older optional
`method="2d"` path and does not apply the relative-translation correction. It is
not the reported production path: `method="direct2ds"` delegates to
`src/rp_decoders.py`, where the correction is active.
