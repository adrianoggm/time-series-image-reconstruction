# Information Sciences manuscript: Sipers and reproducibility audit (2026-10-26)

Scope: `doc/infsci/main.tex` and `doc/infsci/supplementary.tex`. Every check below can be rerun with the
command given; nothing here changes data, metrics or generated tables.

## 1. Sipers' frequency threshold

| Item | Finding |
|---|---|
| Threshold in the code | `TOL = 1e-4` in `src/rp_sipers.py`. |
| How it was first chosen | `scripts/_sipers_pilot.py` swept it on the first 100 benchmark windows per dataset at T = 128. Those windows are also evaluated, so that choice was not independent of the evaluation. |
| Independent calibration | `scripts/_sipers_calibration.py`: 100 windows per dataset at T = 128 outside both the fidelity sample and the classification sample (the latter redrawn exactly as `scripts/_classification.py` draws it), seed 1. |
| Result | The value was chosen exploratorily, then confirmed: both criteria choose `1e-4`: 8-bit NMAE with the reflection fixed by the original series, and with the reflection chosen from the signed image. The evaluation stands unchanged. |
| Limits | EMG has no calibration windows, because all 880 of its windows are evaluated. Calibration windows can come from the same recordings as evaluated windows. |
| Artefacts | `results/_stats/sipers_calibration.{json,md}`; supplementary Section S8. |

### Pairing of the comparisons

| Comparison | Paired by series | Paired by image |
|---|---|---|
| Hirata (unsigned/signed) vs geometric (unsigned/signed) | yes | yes: same conventional RP |
| Sipers (unsigned/signed) vs any other RP decoder, benchmark and classification | yes | no: Sipers reads the circular RP |
| Sipers vs geometric, supplementary Table S6 | yes | yes: same circular RP |

Table S6 was refreshed on the current ESC-50 sample (3000 windows). Before that, it used the earlier 500-window sample for ESC-50. The geometric decoder is now 1.8 to 4.8 times more accurate than Sipers' on the same signed circular image.

## 2. Hypotheses of Proposition 1

`scripts/_check_sign_identifiability.py` checks (a), (b) and (c) directly on all 61 079 channel series of the benchmark, with relative tolerance 1e-9. It excludes flat series with the benchmark's own rule (range below 1e-12). An earlier run excluded only exactly constant series and counted 61 083; the four extra near-flat series did not set any minimum margin.

- No series fails any hypothesis.
- Smallest margins: sigma2/sigma1 = 5.7e-6 for (a); 3.1e-6 for the (F) relation; 0.10 for the (R) relation.
- The sufficient rank-three criterion also holds for every series.

The discussion's statement that every benchmark series meets the hypotheses now rests on this check, not on the observed exact recovery. The full proof is in supplementary Section S13 and should still be reviewed by the co-author in algebra. The check is numerical evidence, not a proof.

## 3. Image-only decoding

Every decoder receives the stored image and the channel calibration (max, min) only: see the decode signatures in `src/representations.py`.

`scripts/_check_image_only.py` decodes the same 8-bit image with the true calibration and with an unrelated one. For all ten combinations, the second result equals the first after the affine change of range, with a maximum relative deviation of 2.5e-16. The calibration therefore enters only as the final map to physical units; it never chooses a reflection, branch or threshold.

**Tie between reflections (found and fixed 2026-10-26).** The first version of this check only used series with
sign-informative pairs. On the counterexample of the main text (no informative pair, so x and -x give the same
signed RP), the three signed decoders still returned both x and -x correctly. They chose the reflection after
mapping the candidates to physical units, so in an exact tie the rounding of the calibration decided. They now
compare the candidates on the unit range and apply the calibration to the winner only. The check now includes
that series and fails on the old code (deviation 1.0 for the signed RP). On 2 000 benchmark series the change
moves the reconstructions by at most 5.7e-14 of the range, and the M0 baseline still matches (52/52 arrays), so
no reported number changes.

The RP images are invariant to increasing affine maps of the series (the binary RP because its threshold follows a per-window recurrence rate, `eps_for_rate` in `src/representations.py`). The text now states that the RP decoders return the trajectory mapped onto [0, 1], and that the calibration is applied afterwards.

## 4. Calibrations, splits and tables

- **Normalisation per representation** (verified in `src/GAF.py`, `src/GADF.py`, `src/XIRP.py`, `src/recurrence_plots.py`):
  - GASF: [0, 1];
  - GADF: [-0.7, 0.7];
  - XIRP: [1, 2], with fixed storage ranges [1, 2] on the diagonal and [-log 2, log 2] off it;
  - MTF: quantile states;
  - RP: computed on the window as recorded. The signed RP is stored with a per-window min-max over the non-zero entries; the unsigned RP is divided by its maximum.
- **Splits:** `src/datasets.py::group_kfold` raises `AssertionError` if a group appears on both sides of a fold. `scripts/_classification.py` standardises with training-fold statistics only.
- **Tables and quoted numbers:** `scripts/_emit_infsci_tables.py` regenerates every table and number macro. Running it twice gives byte-identical files. Both documents compile without undefined references.

## 5. Numbers written by hand in Results, Discussion and Conclusions

All the others are design parameters (T, sigma, rates, durations).

| Claim | Value in the artefacts |
|---|---|
| Lossy combinations at T = 128 between about 0.03 and 0.34 | 0.028 (Swarm, Hirata signed) to 0.340 (PTB-XL, MTF) |
| Figure 4: they stay between 0.08 and 0.34 | robustness 0.085 to 0.293; sampling rate 0.083 to 0.344 |
| Utility: they lose 0.08 to 0.12 on average, up to 0.40 | 0.080 to 0.120; maximum 0.400 |
| Hirata's cost exponent about 2.6 | 2.59 (signed) and 2.64 (unsigned) |
| Geometric decoder about as costly as Sipers' at T = 256, half at T = 1024 | 25 ms vs 30 ms; 552 ms vs 1193 ms |

## 6. Open items for human review

- Check the sign convention against Lu and Tong's paper (pi/4 here).
- The algebra co-author should review the proof in supplementary Section S13.
- Closed: the code is cited by its GitHub URL only, with no Zenodo DOI (#137); the declaration of generative-AI use is the authors' text (#138).
- The manuscript is 39 pages, against a target of about 35. Moving the counterexamples and the conditioning paragraph to S13 would save about one page.
