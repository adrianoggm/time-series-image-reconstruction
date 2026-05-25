# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Reference implementation of a TFG on reconstructing triaxial accelerometer time series (WISDM) from four image-based encodings — **RP, GASF, GADF, MTF** — and benchmarking the reconstructions against the original signals with six classifiers (KNN, SVM, RandomForest, LSTM, GRU, CNN). The paper draft (`doc/main.tex`) is part of the deliverable and is updated alongside the code.

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

- **E1 (model cache)** — `models/<classifier>.<h5|joblib>` is fit **once on the originals** and reused across every encoding. Trained on the very first validator call; subsequent calls (any encoder) just predict.
- **E2 (original-report cache)** — `results/_originals/<classifier>/fold-<k>-<sampling>-<n>f.json` stores the classification report on the un-reconstructed validation set. Same logic: independent of the encoder, persisted and reused.
- **E3 (skip-if-done)** — if `results/<stype>/<classifier>/fold-<k>.json` exists, the call is a no-op unless `--force` is passed.

Implication: when you change a classifier's hyperparameters or training code, delete its `models/<clf>.*` cache and the matching `results/_originals/<clf>/...` JSON or pass `--train-new` to `run_pipeline.py`; otherwise the evaluator silently reuses the stale model.

## Reconstruction internals (parallel-order bug, fixed)

`src/reconstruct_tseries.py` reconstructs each window in parallel via `ProcessPoolExecutor`. **The aggregation must preserve sample order** — earlier versions used `as_completed`, which produced label/sample misalignment and showed up downstream as nonsense classifier accuracies (the historical RP-KNN 0.51 in baselines was this bug, not the algorithm). The current code carries `(index, recon, axis)` tuples through the pool and sorts by index before stacking. If you touch the parallel loop, preserve that invariant.

RP-specific: `src/recurrence_plots.py::fix_rotationscale` returns `posi[:TIME_STEPS]` (not `[:128]`). The adjacent-state embedding `(x_j, x_{j+1})` produces `N-1` recovered samples; the function augments back to `N=TIME_STEPS` by appending the per-window mean. Don't hardcode 128.

MTF-specific: when `T % numstates == 0` (e.g. `T=128, Q=4`) there are no leftover indices, so `MTF.py` iterates `estadosfaltantes` instead of indexing `[0]` — guard that pattern if you generalise the bin count.

## Statistical analysis

Significance tests for paper claims live in [scripts/_stat_tests.py](scripts/_stat_tests.py): bootstrap 95% CIs for accuracy, paired McNemar (continuity-corrected via `chi2.sf` for tail precision), Wilcoxon signed-rank on per-window MAE with effect size `r = Z/√N`, Steiger's z for dependent correlations, and Holm-Bonferroni inside each family. Writes `results/_stats/stat_report.{md,json}`.

Verification scripts (run them before committing paper changes):
```bash
python scripts/_check_refs.py            # broken/unused \label/\ref in doc/main.tex
python scripts/_verify_paper_tables.py   # tab:*_classifiers accuracies vs JSON snapshot
python scripts/_verify_error_tables.py   # pointwise tables vs .npy reconstructions (runs in WSL)
```

## Aggregating results

```bash
python -m scripts.summarize_results --out docs/results/latest.md
```

Walks `results/**/fold-*.json` (skipping `_originals/`) and emits a Markdown comparison table. The fields exposed in each JSON are produced by [src/results.py](src/results.py) via `save_run_metrics`.

## CI

`.github/workflows/smoke.yml` runs an import-only smoke test on (Ubuntu, Windows) × (3.10, 3.11) — it does not exercise the pipeline. It also asserts `get_time_setup('WISDM')` returns `(128, 128)` and that `DEFAULT_DATA_DIR` ends with a path separator. Both invariants are load-bearing across the codebase; if you change them, expect a cascade.

## Calling code in WSL from a Windows checkout

The maintainer keeps the repo on Windows (`C:\Users\...\TFG-Tratamiento-de-Series-Temporales`) but runs the pipeline in WSL2 (`/home/adrianogarcia/tfg/`). Sync with `rsync -a /mnt/c/Users/.../scripts/<file> /home/adrianogarcia/tfg/scripts/`. Avoid piping long commands through `wsl.exe -- bash -lc "..."` — PowerShell mangles quotes; write the command to a `.sh` file in `scripts/` and execute it via `bash /tmp/<copy>.sh` (strip CRs with `sed 's/\r$//'` first).

## Conventions worth knowing

- **Editable install required.** Both `src/` and `scripts/` are real packages (each with `__init__.py`); always import as `from src.X import Y` and `from scripts.X import Y`. Never sys.path-hack — `pip install -e .` puts them on the path correctly.
- **`evaluate_run` returns a dict and persists the JSON in the same call.** Don't re-implement persistence in a validator.
- **`stype` strings are the canonical identifiers.** Use `ALL_STYPES` and `STYPE_FOLDER` from `src/paths.py` instead of hardcoded mappings — that mistake is what made adding GADF a multi-file refactor the first time.
- **Don't rename the legacy `Validation.py`** (SVM); `run_pipeline.py` references it by module path.
