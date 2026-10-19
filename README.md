# Reconstructing Time Series from Image Encodings

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](pyproject.toml)

Reference implementation for a study on inverting four image-based encodings of
triaxial accelerometer windows — **RP, GASF, GADF, MTF** — back into signals, and
on testing whether those reconstructions remain usable in place of the originals.

Every decoder here is analytic. No generative model is trained: the question is
whether the stored image, under a declared encoder convention, carries enough
information to recover the signal.

| ID | Encoding | Decoder in this repo |
|----|----------|----------------------|
| `RP` | Recurrence Plot | Classical MDS on the recovered operator, then a closed-form O(2) orientation from a 2×2 eigenproblem. Metadata-free. |
| `GAF` | Gramian Angular **Summation** Field (GASF) | Diagonal inverse. `GAF` is the code identifier for GASF throughout. |
| `GADF` | Gramian Angular **Difference** Field | Closed-form multi-pivot decoder on the restricted branch (α = 0.7 < 1/√2). |
| `MTF` | Markov Transition Field | Fiedler-vector spectral seriation of the quantile bins. Reported as a lossy control. |

The submission manuscript is **[doc/main3.tex](doc/main3.tex)**, with
[doc/supplementary.tex](doc/supplementary.tex) alongside it.

---

## Headline results

Numbers below are the ones printed in `main3.tex`; each points at the table that
carries it, and every table is regenerated from stored artefacts (see
[Reproducing the paper numbers](#reproducing-the-paper-numbers)).

**Reconstruction fidelity**, pooled over 8,172 windows (`tab:pointwise_pooled`):

| Method | MAE (m/s²) | NMAE (×10⁻³) | Pearson |
|--------|-----------|--------------|---------|
| RP | 0.0142 ± 0.0156 | 1.0 ± 0.7 | 0.99998 |
| GADF | 0.0157 ± 0.0167 | 1.3 ± 0.5 | 0.99999 |
| GASF | 0.0205 ± 0.0199 | 1.5 ± 0.6 | 0.99988 |
| MTF | 3.94 ± 4.25 | 278 ± 80 | 0.014 |

**Downstream classification**, grouped subject-out, three folds
(`tab:subject_out_acc`, `tab:equivalence_ci`):

| Classifier | Original | RP | GASF | GADF | MTF |
|-----------|----------|------|------|------|------|
| CNN | 0.796 | 0.796 | 0.796 | 0.796 | 0.686 |
| GRU | 0.811 | 0.811 | 0.811 | 0.811 | 0.702 |
| RF | 0.786 | 0.786 | 0.787 | 0.787 | 0.689 |

What the study concludes:

- RP, GASF and GADF sit near the clean 8-bit round-trip floor and are
  **classification-equivalent to the original signal** within a ±0.01 margin.
  The claim rests on simultaneous subject-cluster bootstrap intervals (51
  subjects, 10⁴ joint resamples), not on a failure to reject a difference.
- MTF is equivalent to none of them. Its many-to-one quantisation is lossy by
  construction, which is why it serves as the negative control rather than as a
  competitor.
- Ordering inside the near-original trio is detectable but small. It survives
  aggregation to 51 subject means (RP 0.0137, GADF 0.0148, GASF 0.0199 mean MAE),
  so it is not an artefact of window-level pseudoreplication.

Scope: the decoders assume the encoder conventions declared in
`tab:decoder_contract`. This separates *"can this representation be inverted
under known conditions?"* from *"can a generator synthesise calibrated signals
from nothing?"* — only the first is addressed here.

---

## Repository layout

```text
src/                      library code
  activity_data.py        WISDM loading, windowing, fold assignment
  config.py               data directory resolution (TFG_DATA_DIR)
  paths.py                canonical stype -> folder mapping; artefact paths
  seeding.py              fixed seeds
  recurrence_plots.py     RP encoder
  rp_decoders.py          RP decoder family (production: direct2ds, closed form)
  GAF.py GADF.py          GASF / GADF encoders and decoders
  MTF.py MTF_v2.py        MTF encoder; Fiedler-vector decoder
  generate_images.py      encoding stage
  reconstruct_tseries.py  parallel image -> signal reconstruction
  evaluator.py            shared classifier evaluation and its three caches
  models.py results.py    model cache and metric persistence
  calculate_ts_errors.py  pointwise error metrics
  fidelity_metrics.py     NMAE / spectral / trend metrics

scripts/                  entry points, all run as `python -m scripts.<name>`
  run_pipeline.py         orchestrator; prefer this for full runs
  generate_images.py      stage 1: encode
  generate_time_series.py stage 2: reconstruct
  *validation.py          stage 3: one classifier each (Validation.py = SVM)
  summarize_results.py    walk results/ into a Markdown table
  audit_manuscripts.py    recompute paper tables straight from the arrays
  emit_classifier_tables.py  LaTeX bodies for the per-classifier tables
  rp_gadf_structure_*.py  supplementary RP/GADF region-overlap diagnostic
  plot_rp_gadf_structure_figures.py  its figure
  setup.sh setup.ps1      environment bootstrap
  _*.py _*.sh             paper statistics, ablations, figures and build gates

doc/                      manuscripts and their figures
  main3.tex               submission build (elsarticle, preprint 12pt)
  main2.tex               archived two-column camera-ready build
  supplementary.tex       supplementary material
  images/                 only the figures a manuscript includes
  workflow_overview.md    pipeline diagram (Mermaid source alongside)

docs/                     supporting notes (ablations, synthetic experiment,
                          baseline metrics)
results/_stats/           the stored statistics the paper prints
```

Generated artefacts — encoded images, reconstruction arrays, model caches,
per-run JSON — stay out of Git by design. See [Outputs](#outputs).

---

## Requirements

- Python `>=3.10,<3.12`. TensorFlow 2.11–2.15 publishes no wheels outside that
  range, and Keras 3 is excluded because the code uses `keras.preprocessing.image`.
- The WISDM Smartphone and Smartwatch Activity and Biometrics Dataset.
- Disk space for the generated image and reconstruction trees.

[pyproject.toml](pyproject.toml) is the authoritative dependency list.

---

## Setup

Both scripts install [`uv`](https://github.com/astral-sh/uv) locally without
admin rights, create `.venv/`, and run `pip install -e .`.

```bash
bash scripts/setup.sh          # Linux, macOS, WSL
bash scripts/setup.sh dev      # + ruff and pre-commit
```

```powershell
.\scripts\setup.ps1
.\scripts\setup.ps1 -Extras dev
```

Manually:

```bash
python3.10 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

The editable install is required: `src/` and `scripts/` are real packages, and
imports are written `from src.X import Y`. Never patch `sys.path`.

**Windows long paths.** TensorFlow 2.11 unpacks headers past the 260-character
limit. Either enable long paths, or put the virtualenv outside the repo with
`.\scripts\setup.ps1 -VenvDir "C:\Users\$env:USERNAME\tfg310"`.

---

## Data

Unpack WISDM so the raw files sit under `data/WISDM/wisdm-dataset/raw/`:

```bash
mkdir -p data/WISDM
curl -L -o /tmp/wisdm.zip https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip
unzip -q /tmp/wisdm.zip -d data/WISDM/
unzip -q data/WISDM/wisdm-dataset.zip -d data/WISDM/
find data/WISDM -name ".DS_Store" -delete     # these crash the loader
rm data/WISDM/wisdm-dataset.zip /tmp/wisdm.zip
```

To read from elsewhere, `export TFG_DATA_DIR=/absolute/path/to/WISDM/` — the
value must end with a separator, which `src/config.py` enforces.

Protocol: smartwatch accelerometer; Walking, Jogging, Stairs, Sitting, Standing;
51 subjects; windows of `N=128` samples (≈6.4 s at 20 Hz). RP embeds adjacent
states, so its image is 127×127 and the decoder pads back to 128. Splits are
`loto` (leave-one-trial-out) and `loso` (leave-one-subject-out, the primary
protocol in the paper).

---

## Running the pipeline

`run_pipeline.py` chains the three stages in one process so tracebacks stay
readable. Prefer it.

```bash
# Full run for one encoder and fold; --create-numpies on the first run only
python -m scripts.run_pipeline --stype GADF --sampling loto --fold 0 --n-folds 3 --create-numpies

# Re-validate using cached images and reconstructions
python -m scripts.run_pipeline --stype GAF --skip-encode --skip-reconstruct

# A subset of classifiers
python -m scripts.run_pipeline --stype MTF --classifiers knn,svc,randomforest
```

| Flag | Meaning |
|------|---------|
| `--stype RP\|GAF\|GADF\|MTF` | Encoding. `GAF` means GASF. |
| `--sampling loto\|loso` | Split protocol. |
| `--fold 0\|1\|2`, `--n-folds` | Fold selection. |
| `--create-numpies` | Rebuild the windowed cache from raw WISDM. First run per `(sampling, n_folds)`. |
| `--skip-encode` / `--skip-reconstruct` / `--skip-validate` | Reuse or stop after a stage. |
| `--train-new` | Retrain instead of loading the cached model. |
| `--force` | Recompute result JSON even if it exists. |

The stages also run standalone:

```bash
python -m scripts.generate_images      --image-type GADF --sampling loto --n-folds 3
python -m scripts.generate_time_series --image-type GADF --sampling loto --n-folds 3
python -m scripts.knnvalidation        --stype GADF --sampling loto --fold 0
python -m scripts.Validation           --stype GADF --sampling loto --fold 0   # SVM
```

**Reconstruction needs `OMP_NUM_THREADS=1`.** The process pool is per-window, so
multi-threaded BLAS inside each worker oversubscribes the cores: 2.3 it/s at
default threading against 16–17 it/s pinned, on a 24-core host.

---

## Outputs

```text
data/WISDM/plots/<encoder>/sampling_<loto|loso>/<n>-fold/fold-<k>/{train,test}/<label>/
data/WISDM/tseries/<encoder>/sampling_<loto|loso>/<n>-fold/fold-<k>/{train,test}/
models/
results/<stype>/<classifier>/<sampling>/fold-<k>.json
```

`src/paths.py` owns every path convention; `ALL_STYPES` and `STYPE_FOLDER` are
the only mapping from CLI identifier to folder name. All encodings read their
raw windows from the RP subtree.

Aggregate the result tree with:

```bash
python -m scripts.summarize_results --out docs/results/latest.md
```

`evaluate_run` in [src/evaluator.py](src/evaluator.py) carries three caches that
make a second run cheap — a trained-model cache, an original-report cache and a
skip-if-done guard. Change a classifier's hyperparameters and you must invalidate
them, with `--train-new --force` or by hand; otherwise the stale model is reused
silently. [CLAUDE.md](CLAUDE.md) lists the exact files.

---

## The paper

```bash
cd doc
latexmk -pdf main3.tex          # build this first
latexmk -pdf supplementary.tex  # resolves \externaldocument{main3} via main3.aux
```

`main2.tex` is the earlier two-column build, kept for reference; it is no longer
derived from the same source and the two are edited independently.

### Reproducing the paper numbers

Every printed figure has a stored source under `results/_stats/`:

| Artefact | Generator |
|----------|-----------|
| `stat_report_<sampling>.{md,json}` | `scripts/_stat_tests.py` (`TFG_STAT_SAMPLING=loso` for the subject-out family) |
| `per_subject_mae_<sampling>.json` | `scripts/_stat_per_subject.py` |
| `gadf_pivot_sweep.{json,tex}` | `scripts/_gadf_pivot_stats.py --n 1500` |
| `mtf_decoder_ablation.json` | `scripts/_mtf_controlled_ablation.py` |
| `robustness.json`, `fidelity_metrics.json` | `scripts/_make_robustness_tables.py`, `scripts/_make_fidelity_tables.py` |
| `doc/images/robustness_curves.pdf`, `equivalence_forest.pdf` | `scripts/_make_paper_figures.py` |
| `doc/images/reconstruction_grid.pdf` | `scripts/_regen_reconstruction_figs.py` |

Table rows come from the live `results/` tree via
`scripts/_emit_main2_numbers.py` and `scripts/_make_loso_tables.py`.
`results/_originals/` is a cache, not a source — read the Original column from
any encoder's fold JSON instead.

Checks to run before committing a paper change:

```bash
python scripts/_check_refs.py          # broken or unused \label and \ref
bash scripts/_check_build.sh           # strict build gate: undefined refs, overfull boxes
python -m scripts._check_rp_decoders   # RP decoder contract
python -m scripts._check_rp_production # production defaults the published numbers depend on
```

---

## Conventions

- `stype` strings are canonical. Use `ALL_STYPES` and `STYPE_FOLDER`, never a
  hardcoded mapping.
- Reconstruction arrays are axis-normalised on load: `_load_reconstruction`
  transposes `(N, C, T)` to `(N, T, C)`. Don't bake an axis order into callers.
- Parallel reconstruction must preserve sample order. The pool carries explicit
  indices and sorts before stacking; an earlier `as_completed` version produced
  label misalignment that surfaced as nonsense accuracies.
- There is no test suite. The only automated check is the import smoke test in
  `.github/workflows/smoke.yml`.
- Lint with `ruff check .` and `ruff format .` (configured in `pyproject.toml`).

---

## Troubleshooting

- **A validator seems to ignore your change** — it loaded a cached model. Run
  with `--train-new --force`.
- **Wrong reconstruction shape** — regenerate that fold; old `N=129` artefacts
  are incompatible with the `N=128` protocol.
- **`TFG_DATA_DIR` not picked up** — it must end with a path separator and point
  at the directory containing `wisdm-dataset/`.
- **TensorFlow install fails on Windows** — long paths; see Setup.

---

## Citation

```bibtex
@misc{tfg-tsd,
  title  = {Time-series reconstruction from image-based encodings on WISDM},
  author = {Garcia-Giralda, Adriano},
  year   = {2026}
}
```
