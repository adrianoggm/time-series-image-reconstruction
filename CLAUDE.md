# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Reference implementation of a TFG on reconstructing triaxial accelerometer time series (WISDM) from four image-based encodings — **RP, GASF, GADF, MTF** — and benchmarking the reconstructions against the original signals with six classifiers (KNN, SVM, RandomForest, LSTM, GRU, CNN). The paper draft (`doc/main2.tex`, Elsevier build) plus `doc/supplementary.tex` are part of the deliverable and are updated alongside the code. Build `main2.tex` first: the supplement resolves cross-references through `\externaldocument{main2}`.

## Setup

The dependency manifest is `pyproject.toml`. Python is pinned to **>=3.10,<3.12** because TensorFlow 2.11–2.15 wheels do not exist outside that range; Keras 3 (TF 2.16+) is intentionally excluded because the code uses `keras.preprocessing.image`.

```bash
bash scripts/setup.sh                # Linux/macOS/WSL, core deps
bash scripts/setup.sh hf diffusion   # + optional extras
.\scripts\setup.ps1                  # Windows PowerShell equivalent
```

Both scripts install [`uv`](https://github.com/astral-sh/uv) locally (no admin), create `.venv/`, and `pip install -e .`.

**Windows long-path gotcha** — TF 2.11 unpacks header files past Windows' 260-char limit. Either enable long paths (`Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1 -Type DWord` as admin) or place the venv outside the repo (`scripts/setup.ps1 -VenvDir "C:\Users\$env:USERNAME\tfg310"`). In practice, the maintainer runs the heavy pipeline in WSL2 because of this.

## Data layout

By default every script reads from `./data/WISDM/`. Override with `TFG_DATA_DIR=/abs/path/` (the value must end with a trailing separator — `src/config.py` enforces that). The downloaded WISDM archive ships with macOS `.DS_Store` files that crash `create_wisdm_dataset()`; delete them after unzip.

The pipeline materialises three nested artefact trees under `data/WISDM/`:
- `plots/<encoder_folder>/sampling_<loto|loso>/<n>-fold/fold-<k>/{train,test}/<label>/<subject>x<i>.png` — encoded images.
- `tseries/<encoder_folder>/sampling_<loto|loso>/<n>-fold/fold-<k>/{train,test}/X_all_<stype>.npy` — the aggregate reconstruction array.
- `tseries/recurrence_plot/.../{validation,training}_data.npy` — the per-fold raw window arrays. **All encodings read the raw windows from the RP subtree**, regardless of which encoder is being exercised; see `src/paths.py::windowed_npy`.

`STYPE_FOLDER` in [src/paths.py](src/paths.py) is the single source of truth mapping the CLI identifier (`RP|GAF|GADF|MTF`) to its folder name (`recurrence_plot|GAF|GADF|MTF`). `GAF` everywhere in the code means **GASF**; the difference variant has its own `GADF` identifier.

## Pipeline

The three pipeline stages run as `python -m scripts.<module>`. The orchestrator [scripts/run_pipeline.py](scripts/run_pipeline.py) chains them in-process so tracebacks stay readable; prefer it for full runs:

```bash
# Full pipeline on one encoder
python -m scripts.run_pipeline --stype GADF --sampling loto --fold 0 --n-folds 3

# Re-validate only (reuse cached images + reconstructions)
python -m scripts.run_pipeline --stype GAF --skip-encode --skip-reconstruct

# Subset of classifiers
python -m scripts.run_pipeline --stype MTF --classifiers knn,svc,randomforest
```

Or call each stage directly:
```bash
python -m scripts.generate_images       --image-type GAF --sampling loto --n-folds 3 --create-numpies
python -m scripts.generate_time_series  --image-type GAF --sampling loto --n-folds 3
python -m scripts.{knn,svm,cnn,gru,lstm,randomforest}validation  # Validation.py = SVM
```

`--create-numpies` rebuilds the windowed numpy cache from raw WISDM and is only needed on the first run per `(sampling, n_folds)` pair.

## Validator architecture (the part you have to read multiple files to understand)

Every validator (`scripts/knnvalidation.py`, `lstmvalidation.py`, etc.) is a thin shim that:
1. Builds a `ClassifierAdapter` (defined in [src/evaluator.py](src/evaluator.py)) wrapping its model-specific `build/fit/predict/save/load` callbacks and a `flatten_for_model` flag (sklearn classifiers want `(N, T*C)`, Keras ones want `(N, T, C)`).
2. Calls `evaluate_run(adapter=..., stype=..., ...)`.

`evaluate_run` is where the cross-cutting caching lives — adding a new classifier or a new encoding should never duplicate this logic:

- **E1 (model cache)** — `models/<clf>__<sampling>-fold-<k>-<n>f.<h5|joblib>` is fit **once on the originals** and reused across every encoding *within that (sampling, fold)*. Trained on the very first validator call; subsequent calls (any encoder) just predict. The bare `models/<clf>.<ext>` form in [src/models.py](src/models.py) is a legacy fallback only — `evaluate_run` always passes `sampling`/`fold`/`n_folds`, so new runs write the qualified filename. The split was added because LOTO fold-0 and LOSO fold-0 would otherwise collide.
- **E2 (original-report cache)** — `results/_originals/<classifier>/fold-<k>-<sampling>-<n>f.json` stores the classification report on the un-reconstructed validation set. Same logic: independent of the encoder, persisted and reused.
- **E3 (skip-if-done)** — if `results/<stype>/<classifier>/<sampling>/fold-<k>.json` exists, the call is a no-op unless `--force` is passed. (The bare `results/<stype>/<classifier>/fold-<k>.json` form is again a legacy fallback in `results_dir`; new callers always include the sampling segment.)

Implication: when you change a classifier's hyperparameters or training code, either pass `--train-new --force` to `run_pipeline.py`, or delete by hand:
- `models/<clf>__<sampling>-fold-<k>-<n>f.<ext>`
- `results/_originals/<clf>/fold-<k>-<sampling>-<n>f.json`
- `results/<stype>/<clf>/<sampling>/fold-<k>.json`

Otherwise the evaluator silently reuses the stale model.

**Reconstruction-array axis order is auto-normalised, not stype-specific.** [src/evaluator.py](src/evaluator.py)::`_load_reconstruction` inspects `arr.shape` and transposes `(N, C, T)` → `(N, T, C)` when the channel axis is in position 1. Every classifier downstream assumes `(N, T, C)` (or its flattened `(N, T*C)` form). Persist reconstructions however you like, but don't hardcode an axis convention in callers — let `_load_reconstruction` normalise.

## Reconstruction internals (parallel-order bug, fixed)

`src/reconstruct_tseries.py` reconstructs each window in parallel via `ProcessPoolExecutor`. **The aggregation must preserve sample order** — earlier versions used `as_completed`, which produced label/sample misalignment and showed up downstream as nonsense classifier accuracies (the historical RP-KNN 0.51 in baselines was this bug, not the algorithm). The current code carries `(index, recon, axis)` tuples through the pool and sorts by index before stacking. If you touch the parallel loop, preserve that invariant.

### RP production decoder contract

Production RP reconstruction is `Reconstruct_RP(..., method="direct2ds")` with **two defaults that the paper depends on**:

- **`rp_range=None` → metadata-free.** `src/rp_decoders.py::decode_window` inverts the PNG by centring on the diagonal median (`R = png - median(diag(png))`), since the RP diagonal is identically zero. Passing per-window `(R_min, R_max)` instead is the *operator-range oracle*, an ablation-only variant. `reconstruct_tseries._build_rp_range` is kept solely for that variant — production never calls it (it costs as much as computing the RP).
- **`orientation_method="closed_form"`.** The O(2) orientation comes from the smaller-eigenvalue eigenvector of a symmetric 2×2 matrix per reflection parity, giving four exact candidates; `"grid"` (n_theta=720) is the historical baseline kept for the ablation. Closed form is exact on the native float operator (NMAE ~9e-17 vs the grid's 3.5e-4) and ~2.2× faster.

Changing either default changes published numbers. Guard both with `python -m scripts._check_rp_decoders` and `python -m scripts._check_rp_production`.

**Parallel reconstruction needs `OMP_NUM_THREADS=1`.** The pool is per-window, so multi-threaded BLAS inside each worker oversubscribes the cores: measured 2.3 it/s at default threading vs 16–17 it/s pinned to one thread each (24-core host). `scripts/_regen_rp_6folds.sh` exports it.

RP-specific: `src/recurrence_plots.py::fix_rotationscale` returns `posi[:TIME_STEPS]` (not `[:128]`). The adjacent-state embedding `(x_j, x_{j+1})` produces `N-1` recovered samples; the function augments back to `N=TIME_STEPS` by appending the per-window mean. Don't hardcode 128.

MTF-specific: when `T % numstates == 0` (e.g. `T=128, Q=4`) there are no leftover indices, so `MTF.py` iterates `estadosfaltantes` instead of indexing `[0]` — guard that pattern if you generalise the bin count.

## Statistical analysis

Significance tests for paper claims live in [scripts/_stat_tests.py](scripts/_stat_tests.py): bootstrap 95% CIs for accuracy, paired McNemar (continuity-corrected via `chi2.sf` for tail precision), Wilcoxon signed-rank on per-window MAE with effect size `r = Z/√N`, Steiger's z for dependent correlations, and Holm-Bonferroni inside each family. Writes `results/_stats/stat_report_<sampling>.{md,json}` — pass `TFG_STAT_SAMPLING=loso` for the subject-out family (default `loto`).

Verification scripts (run them before committing paper changes):
```bash
python scripts/_check_refs.py            # broken/unused \label/\ref in doc/main2.tex + supplementary.tex
python scripts/_verify_error_tables.py   # pointwise tables vs .npy reconstructions (runs in WSL)
```

`scripts/_verify_paper_tables.py` and `scripts/_make_accuracy_tables.py` are **obsolete**: they read the deleted `doc/main.tex` and the stale `.results_snapshot/`. Use [scripts/_emit_main2_numbers.py](scripts/_emit_main2_numbers.py) instead — it emits the classification and equivalence table rows straight from the live `results/` tree and the regenerated stat reports. `_make_loso_tables.py` reads the live tree too.

### Every printed number needs a stored source

| Artefact under `results/_stats/` | Generator |
|---|---|
| `per_subject_mae_<sampling>.json` | `scripts/_stat_per_subject.py` — subject-clustered Wilcoxon on per-window MAE, Holm-corrected over the six encoder pairs |
| `gadf_pivot_sweep.{json,tex}` | `scripts/_gadf_pivot_stats.py --n 1500` |
| `mtf_decoder_ablation.json` | `scripts/_mtf_controlled_ablation.py` |

[scripts/_persist_paper_sources.sh](scripts/_persist_paper_sources.sh) regenerates all three in WSL and copies them back to the Windows checkout. These three used to print to stdout only, which left three supplementary tables untraceable.

[scripts/audit_manuscripts.py](scripts/audit_manuscripts.py) recomputes the pointwise and classification tables straight from the reconstruction arrays and writes `docs/audit/numerical_evidence.json`. Its `--compare-root` must name the **other** checkout; pointed at its own tree the cross-checkout diff is skipped rather than silently reporting "no differences".

**`results/_originals/` is a cache, not a source.** It is untracked and has gone stale before: its CNN entries disagreed with the `original` block of the fold JSONs, which is what the paper actually reports. Table generators must read the Original column from any encoder's `results/<stype>/<clf>/<sampling>/fold-<k>.json`, as `_emit_main2_numbers.py` and `_make_loso_tables.py` now do.

## Aggregating results

```bash
python -m scripts.summarize_results --out docs/results/latest.md
```

Walks `results/**/fold-*.json` (skipping `_originals/`) and emits a Markdown comparison table. The fields exposed in each JSON are produced by [src/results.py](src/results.py) via `save_run_metrics`.

## CI

`.github/workflows/smoke.yml` runs an import-only smoke test on (Ubuntu, Windows) × (3.10, 3.11) — it does not exercise the pipeline. It also asserts `get_time_setup('WISDM')` returns `(128, 128)` and that `DEFAULT_DATA_DIR` ends with a path separator. Both invariants are load-bearing across the codebase; if you change them, expect a cascade.

## Linting

Ruff is configured in [pyproject.toml](pyproject.toml) (line-length 120, target `py310`, rules `E,F,I,UP,B`, ignoring `E501,E402`). With the `dev` extras installed:

```bash
pip install -e ".[dev]"
ruff check .
ruff format .
```

There is no pytest suite — `src/test_unitarios.py` is a collection of utility helpers used inline by the reconstruction code, not actual tests. Don't add `pytest` to a workflow expecting tests to exist; the only automated check is the smoke import.

## Calling code in WSL from a Windows checkout

The maintainer keeps the repo on Windows (`C:\Users\...\TFG-Tratamiento-de-Series-Temporales`) but runs the pipeline in WSL2 (`/home/adrianogarcia/tfg/`). Sync with `rsync -a /mnt/c/Users/.../scripts/<file> /home/adrianogarcia/tfg/scripts/`. Avoid piping long commands through `wsl.exe -- bash -lc "..."` — PowerShell mangles quotes; write the command to a `.sh` file in `scripts/` and execute it via `bash /tmp/<copy>.sh` (strip CRs with `sed 's/\r$//'` first).

## Conventions worth knowing

- **Editable install required.** Both `src/` and `scripts/` are real packages (each with `__init__.py`); always import as `from src.X import Y` and `from scripts.X import Y`. Never sys.path-hack — `pip install -e .` puts them on the path correctly.
- **`evaluate_run` returns a dict and persists the JSON in the same call.** Don't re-implement persistence in a validator.
- **`stype` strings are the canonical identifiers.** Use `ALL_STYPES` and `STYPE_FOLDER` from `src/paths.py` instead of hardcoded mappings — that mistake is what made adding GADF a multi-file refactor the first time.
- **Don't rename the legacy `Validation.py`** (SVM); `run_pipeline.py` references it by module path.
