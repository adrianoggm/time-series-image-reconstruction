# M0 baseline audit

Reference state of the code and results before the multi-dataset refactor
(milestone M0, issues #33–#40). Nothing in `src/` was changed to produce it.

Audited on 2026-10-23 at `master` = `ebf235d`. USS is out of scope.

## Gates

| Gate | What it pins | Command |
|---|---|---|
| `scripts/_check_baseline.py` | Every encoder image (sha256) and every decoder output on four synthetic windows → `results/_baseline/golden_decoders.npz` | `python -m scripts._check_baseline` |
| `scripts/_freeze_baseline.py` | sha256 + shape of the 216 fold JSONs (144 reported + 72 in `_originals/` and `_baseline_grid720/`), 36 `X_all_*.npy`, the window arrays and the 36 models → `results/_baseline/manifest.json` | `python -m scripts._freeze_baseline --check` (run in the tree that holds the data) |
| `scripts/_check_rp_decoders.py`, `_check_rp_production.py` | Direct-2D equations and the production entry point | already existed; both pass at `ebf235d` |

`_check_baseline` was run twice in the WSL venv (Python 3.10.20, numpy 1.26.4,
scipy 1.14.0, scikit-learn 1.5.1, pyts 0.13.0): 52/52 arrays identical. Its
tolerance is `atol=1e-6`; refresh the golden file with `--update` only for an
intended change and say so in the commit.

Normalised MAE of each decoder on the golden windows (sanity, not a result):

| Window | RP direct2ds | RP geo | RP jaccard (ρ=.60) | RP 1d (legacy) | RP 2d | GASF | GADF | MTF |
|---|---|---|---|---|---|---|---|---|
| sinusoids | 0.0015 | 0.0206 | 0.0683 | 0.0478 | 0.0317 | 0.0034 | 0.0008 | 0.408 |
| trend + noise | 0.0016 | 0.0101 | 0.1251 | 0.1804 | 0.1889 | 0.0018 | 0.0006 | 0.400 |
| impulses | 0.0025 | 0.5068 | 0.3111 | 0.0911 | 0.1000 | 0.0062 | 0.0225 | 0.495 |
| random walk | 0.0017 | 0.0134 | 0.1328 | 0.1184 | 0.1257 | 0.0016 | 0.0009 | 0.344 |

**Provenance caveat.** The results, arrays and models live in the WSL tree
(`/home/adrianogarcia/tfg`, commit `0459dd7` plus 53 locally modified files in
`src/`/`scripts/`), not in a clean checkout. The manifest therefore pins *what
exists*, not a clean code→result chain. The RP rows were regenerated with the
closed-form decoder (see `results/_stats/RP_CLOSED_FORM_PROVENANCE.md`).

## Pipeline (#33)

`generate_images` → `generate_time_series` → six validators via
`evaluate_run` (E1 model cache, E2 original-report cache, E3 skip-if-done; see
`CLAUDE.md`).

- WISDM watch accelerometer, 20 Hz, 3 channels; windows of `T = 128` samples
  (6.4 s), non-overlapping, never crossing a (subject, activity) run
  (`src/activity_data.py::create_dataset`).
- Folds: `StratifiedGroupKFold(n_splits=3)` without shuffling, so splits are
  deterministic; LOTO and LOSO groupings.
- Result matrix frozen: 4 encodings × 6 classifiers × 2 samplings × 3 folds = 144 JSONs.
- Classifier seed: `DEFAULT_SEED = 95` (`src/seeding.py`).

## Encoders and decoders (#34)

| Encoding | Encoder | PNG range | Decoder (production) | Metadata the decoder reads |
|---|---|---|---|---|
| RP | `recurrence_plots.SavevarRP_XYZ` (`varRP`, adjacent states `(x_i, x_{i+1})`, signed, per-channel min–max) | `NormalizeMatrix` → `[0,255]`, 127×127 | `Reconstruct_RP(method="direct2ds")` → `rp_decoders.decode_window(source="direct", rp_range="diagonal")` | per-window signal `(max, min)` for the output amplitude only |
| GASF | `GAF.SavevarGAF_XYZ` | `normalize_to_zero_one` → `[0,255]` | `GAF.Reconstruct_GAF` → `GAF_to_TS` | per-window signal `(max, min)` |
| GADF | `GADF.SavevarGADF_XYZ` (α = 0.7) | idem | `GADF.Reconstruct_GADF` → `GADF_to_TS` (5 pivots) | per-window signal `(max, min)` |
| MTF | `MTF.SavevarMTF_XYZ` (`pyts`, Q = 4 bins) | `[0,1]` → `[0,255]` | `MTF.Reconstruct_MTF` → `MTF_to_TS` (Fiedler ordering, `MTF_v2`) | per-window signal `(max, min)` |

RP variants still reachable:

| Variant | Entry point | Role today |
|---|---|---|
| Direct-2D | `rp_decoders`, `source="direct"`, closed-form orientation | production |
| GEO-2D | `rp_decoders`, `source="geo"` (continuous weights, ρ-quantile graph, Dijkstra, 2-D MDS) | ablation (roadmap 4.4 already exists) |
| Jaccard 1-D | `rp_decoders`, `source="jaccard", dims=1, recover="append"`, ρ = 0.60 | "Legacy Jaccard one-dimensional baseline" row of the supplement |
| Legacy 1-D / 2-D | `Reconstruct_RP(method="1d"/"2d")`, percentile 75, signal-range PNG inversion | backward compatibility |
| Hirata (2008) | — | **not implemented**, only cited |

## Direct-2D contract (#36)

`src/rp_decoders.py`: symmetrised `|R|` → classical 2-D MDS → closed-form O(2)
orientation (smaller-eigenvalue eigenvector of a 2×2 matrix per reflection
parity, sign chosen with the signed RP) → overlap recovery of the `N` samples.
Metadata-free: the PNG is centred on the median of its diagonal.
`_check_rp_decoders` Task 5 asserts that the closed-form equations printed in
the paper match the implementation; both RP gates pass at `ebf235d`.

## Hyperparameters (#37)

| Parameter | Value | Where |
|---|---|---|
| Window `T`, step | 128, 128 (no overlap) | `activity_data.get_time_setup` |
| Folds | 3, `StratifiedGroupKFold`, no shuffle | `generate_images`, `reconstruct_tseries` |
| Seed | 95 | `seeding.DEFAULT_SEED` |
| RP legacy threshold | percentile 75 of `|R|` (diagonal included) | `Reconstruct_RP(percentile=75.0)`, `CreateCostMatrix` |
| RP ablation ρ | 0.60 (Jaccard and GEO rows) | `scripts/_ablation_rp_2d.py::PERCENTILE_RHO` |
| RP orientation | closed form; grid baseline `n_theta = 720` | `rp_decoders` |
| GADF | α = 0.7, 5 pivots | `GADF.py` |
| MTF | Q = 4 bins | `MTF.py` |

## Legacy RP: what "our previous decoder" actually was (#35)

The roadmap assumes `ρ = 0.60` and a Jaccard weight on edges. Git history says
otherwise.

| Date | Commit | State of the code |
|---|---|---|
| 2024-05-27 | `5291c0b` "Consigue implementar bien RP" | Jaccard weight assigned **only to edges** (`CostM ≠ 0`), threshold = percentile 75, sklearn SMACOF MDS (`random_state=1`), `T = 129` |
| 2025-03-03 | `edf761f` "Improving performance of RP" | vectorised: `1 − J` assigned to **all pairs**; the edge loop survives as a comment |
| 2025-03 → 2025-08 | classifiers, cubic-spline post-step (`scripts/cubicspline.py`, since deleted) | all use the all-pairs version |
| 2026-05-21 | `6a02ad1` first manuscript (`doc/main.tex`) | text specifies **ρ = 0.75**, Steps 1–7 below |
| 2026-05-21 | `94dbf28` | `T = 128`; SMACOF replaced by classical MDS |
| 2026-07-09 | `5ba3271` `search_rp_rho.py` | **ρ = 0.60** appears for the first time, as a sweep result |

So `ρ = 0.60` is the July 2026 ablation value, not the historical one; the
first manuscript and all code before July 2026 use 0.75 (percentile 75).

### First-manuscript specification vs. code

| Step (`6a02ad1:doc/main.tex`) | Specification | Code (any version since 2025-03) |
|---|---|---|
| 1 | `τ` = ρ-quantile of **off-diagonal** state distances, ρ = 0.75 | percentile over the full `|R|`, diagonal zeros included |
| 1 (input) | distances `d_ij = ‖z_i − z_j‖₂` | PNG inverted with the **signal** range, an affine distortion of `d_ij` (`rp_decoders` docstring, `rp_range=None`) |
| 2–3 | edges where `c_ij = 1`; Jaccard weight **on each edge** | weight on every pair (graph almost complete) |
| 4 | Dijkstra | Dijkstra (`directed=True`; weights of 0 count as missing edges) |
| 5 | MDS, component of the largest eigenvalue | ≤2026-05: SMACOF 2-D, then the column at the argmax of the *covariance eigenvalues* (an index into eigenvectors used as a column index); since 2026-05: classical MDS, column of largest variance |
| 6.1 | append the mean (N−1 → N) | same |
| 6.2 | rescale **only if** the range is available | always rescales with the per-window signal range |
| 6.3–6.5 | `rec₂ = −rec₁`, keep the one whose signed RP is closer | `n ↦ 1 − n` then min–max rescale (equivalent after rescaling) |
| 7 | cubic-spline resampling to N | not in the decoder; was a separate script, now deleted; the code appends the mean instead |

Hirata et al. (2008) already weight edges by the Jaccard dissimilarity of
neighbourhoods and use shortest paths + MDS. Steps 1–5 are therefore
essentially Hirata on a quantile-thresholded continuous RP; Steps 6–7 and the
1-D projection are ours. Issues #64 and #65 must keep that boundary.

## Code vs. manuscript (#40)

1. The legacy RP row of the current supplement (`Legacy Jaccard one-dimensional
   baseline`) is the all-pairs, classical-MDS, ρ = 0.60 variant — none of the
   three historical configurations above.
2. The first manuscript's ρ = 0.75 and Step 7 (spline) do not match any code
   path that exists today.
3. `main3.tex` does not state ρ for the legacy baseline.

## What M4 inherits

- #65 should restore the edge-only Jaccard with SMACOF and ρ = 0.75 as a new
  variant, keep the current one under an explicit name, and report both.
- #64 should implement Hirata on a binary RP and list Steps 6–7 as our additions.
- #67 (GEO-2D) only needs integration and documentation.
