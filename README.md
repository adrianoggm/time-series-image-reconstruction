# TFG Time-Series Reconstruction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](pyproject.toml)

This repository is a full reconstruction benchmark for image-based time-series encodings on WISDM. It does not stop at producing pretty RP/GAF/MTF images: it implements inverse maps back to triaxial accelerometer signals and tests whether those reconstructions are still usable as signals.

The core contribution is a controlled, apples-to-apples comparison of four encodings:

| ID | Encoding | Role in this repo |
|----|----------|-------------------|
| `RP` | Recurrence Plot | Signed RP encoder plus graph/MDS reconstruction. |
| `GAF` | Gramian Angular Summation Field, GASF | Diagonal inverse baseline; near-original reconstruction. |
| `GADF` | Gramian Angular Difference Field | Closed-form pivot-row decoder with scale metadata. |
| `MTF` | Markov Transition Field | Quantile-transition encoder plus Fiedler-vector reconstruction. |

The benchmark is deliberately multi-lens: visual inspection, pointwise reconstruction error, polynomial-trend similarity, and downstream classification with KNN, SVM, Random Forest, LSTM, GRU, and CNN.

The paper draft is [doc/main.tex](doc/main.tex). The workflow diagram source is [doc/workflow_overview.md](doc/workflow_overview.md).

---

## Why This Repo Matters

The headline result from [doc/main.tex](doc/main.tex) is that image encodings can be made genuinely reversible under a controlled protocol, but the word "best" depends on what the reconstructed signal is used for.

What is new here:

- A signed-RP decoder that uses the Lu-Tong sign convention to resolve the MDS reflection ambiguity and restores the `N -> N-1` adjacent-state shrinkage.
- A closed-form GADF pivot-row decoder, presented as the first explicit inverse map for GADF under restricted angular scaling and external scale metadata.
- A Fiedler-vector MTF decoder, presented as the first inverse map for MTF, replacing a brittle cosine-pivot heuristic with spectral bin ordering.
- A symmetric evaluation of RP, GASF, GADF, and MTF under the same `N=128` WISDM protocol, both LOTO and LOSO 3-fold, with 8,227 pooled paired test windows per protocol.

---

## Results That Sell the Benchmark

The table below is the compact version of the paper's Results and Discussion sections. The important finding is not just that GASF wins. The important finding is that the ranking changes depending on whether you care about waveform fidelity or classifier utility.

| Lens | Headline numbers | Ranking | Interpretation |
|------|------------------|---------|----------------|
| Pointwise reconstruction | Global MAE: GASF `0.0209`, GADF `1.47`, RP `2.95`, MTF `4.00` | GASF > GADF > RP > MTF | Strict waveform fidelity in native accelerometer units. |
| Polynomial trend | Mean Euclidean distance: GASF `0.00088`, GADF `0.10962`, RP `0.19815`, MTF `0.22844` | GASF > GADF > RP > MTF | Same ordering after compressing each signal into smooth trend coefficients. |
| LOTO classifiers | Mean accuracy across six classifiers: Original `0.836`, GASF `0.836`, GADF `0.764`, MTF `0.708`, RP `0.670` | Original ~= GASF > GADF > MTF > RP | MTF is less waveform-faithful than RP, but its coarse transition structure is more useful to most classifiers. |
| LOSO classifiers | Mean accuracy across six classifiers: Original `0.707`, GASF `0.707`, GADF `0.661`, MTF `0.624`, RP `0.600` | Original ~= GASF > GADF > MTF > RP | Subject-out evaluation is harder, but the classifier-side ranking is preserved. |

Main takeaways:

- GASF is effectively near-original: global MAE `0.0209`, Pearson `0.99988 +/- 0.00020`, and classifier accuracy statistically indistinguishable from the raw signal across all six classifiers under both LOTO and LOSO.
- GADF is the strongest non-GASF decoder: MAE `1.47 +/- 1.65`, Pearson `0.983 +/- 0.025`, about `2.0x` tighter than RP and `2.7x` tighter than MTF in MAE.
- GADF is the only reconstruction that strictly beats the raw signal in one setting: LOSO-KNN reaches `0.500 +/- 0.017` vs original `0.477 +/- 0.010` (`p_Holm = 6.4e-11`).
- RP and MTF expose the central rank inversion: RP is closer pointwise (`2.95` vs `4.00` MAE), but MTF beats RP downstream on five of the six classifiers under both LOTO and LOSO.
- The MTF decoder upgrade matters: at `Q=4`, the old cosine-pivot decoder has MAE `6.111`, plain Fiedler ordering drops it to `4.664`, and adding midpoint plus sign-bit metadata reaches `2.371`.
- `Q=4` is not arbitrary: in the paper's sweep it minimises MTF pointwise MAE, maximises pointwise Pearson, and gives the best KNN/CNN accuracies.

Practical recommendation:

- Use GASF when you want the most faithful controlled inverse.
- Use GADF as the best reconstruction-oriented GAF companion: closed-form, strong trend preservation, and useful smoothing under LOSO-KNN.
- Use MTF when downstream classification is the target and coarse transition dynamics are acceptable.
- Keep RP for recurrence/dynamical-systems diagnostics where recurrence geometry itself matters.

Scope note: these numbers do not prove unconstrained generative augmentation. The decoders assume controlled encoder-side metadata such as min/max scale, sign convention, quantisation branch, or equivalent calibration. That is exactly why the benchmark is useful: it separates "can this image representation be inverted under known conditions?" from "can a generator synthesize physically calibrated signals without extra metadata?"

Detailed result tables live in [doc/_3fold_tables.tex](doc/_3fold_tables.tex), [doc/_loso_tables.tex](doc/_loso_tables.tex), and the Results/Discussion sections of [doc/main.tex](doc/main.tex).

---

## Requirements

- Python `>=3.10,<3.12`
- WISDM Smartphone and Smartwatch Activity and Biometrics Dataset
- Enough disk space for generated images, reconstructed arrays, model caches, and JSON result files

TensorFlow is pinned below Keras 3 because this code still uses Keras 2 APIs. The authoritative dependency list is [pyproject.toml](pyproject.toml).

---

## Setup

Use the setup script for your platform.

Linux, macOS, or WSL:

```bash
bash scripts/setup.sh
bash scripts/setup.sh hf diffusion   # optional extras
```

Windows PowerShell:

```powershell
.\scripts\setup.ps1
.\scripts\setup.ps1 -Extras hf,diffusion
```

Manual setup:

```bash
python3.10 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e .
pip install -e ".[hf,diffusion,dev]" # optional
```

Windows note: TensorFlow can exceed the default 260-character path limit during installation. Enable long paths or place the virtual environment outside a deeply nested repository path, for example with `.\scripts\setup.ps1 -VenvDir "C:\Users\$env:USERNAME\tfg310"`.

---

## Data

Download the WISDM dataset and unpack it so the raw files are under:

```text
data/WISDM/wisdm-dataset/raw/
  phone/accel/
  phone/gyro/
  watch/accel/
  watch/gyro/
```

Linux or WSL example:

```bash
mkdir -p data/WISDM
curl -L -o /tmp/wisdm.zip https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip
unzip -q /tmp/wisdm.zip -d data/WISDM/
unzip -q data/WISDM/wisdm-dataset.zip -d data/WISDM/
find data/WISDM -name ".DS_Store" -delete
rm data/WISDM/wisdm-dataset.zip /tmp/wisdm.zip
```

By default scripts read from `./data/WISDM/`. To use another location:

```bash
export TFG_DATA_DIR=/absolute/path/to/WISDM/
```

`src/config.py` normalizes the path with a trailing separator because some legacy code still concatenates strings.

Dataset protocol:

- Sensor stream: WISDM smartwatch accelerometer.
- Activities: Walking, Jogging, Stairs, Sitting, Standing.
- Subjects: 51.
- Window length: `N=128` samples, about 6.4 seconds at 20 Hz.
- `RP` internally uses adjacent states, so its image is `127x127`; the decoder pads the recovered sequence back to 128 samples.
- `LOTO`: leave-one-trial-out.
- `LOSO`: leave-one-subject-out.

---

## Running the Pipeline

Use `scripts.run_pipeline` for normal work. It chains encoding, reconstruction, and validation in one Python process.

Run one encoder on one fold:

```bash
python -m scripts.run_pipeline --stype GADF --sampling loto --fold 0 --n-folds 3 --create-numpies
```

Reuse already generated images and reconstructions, and only evaluate classifiers:

```bash
python -m scripts.run_pipeline --stype GAF --sampling loto --fold 0 --skip-encode --skip-reconstruct
```

Evaluate a subset of classifiers:

```bash
python -m scripts.run_pipeline --stype MTF --sampling loso --fold 1 --classifiers knn,svc,randomforest
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--stype RP|GAF|GADF|MTF` | Encoding to process. `GAF` means GASF. |
| `--sampling loto|loso` | Split protocol. |
| `--fold 0|1|2` | Fold used for reconstruction/validation. |
| `--create-numpies` | Rebuild cached WISDM window arrays from raw files. Use on the first run. |
| `--skip-encode` | Reuse PNG images under `data/WISDM/plots/`. |
| `--skip-reconstruct` | Reuse reconstructed arrays under `data/WISDM/tseries/`. |
| `--skip-validate` | Stop after encoding/reconstruction. |
| `--train-new` | Force model retraining instead of loading cached models. |
| `--force` | Recompute result JSON files even if they already exist. |

Low-level stages are also available:

```bash
python -m scripts.generate_images --image-type GADF --sampling loto --n-folds 3 --create-numpies
python -m scripts.generate_time_series --image-type GADF --sampling loto --n-folds 3
python -m scripts.knnvalidation --stype GADF --sampling loto --fold 0
python -m scripts.Validation --stype GADF --sampling loto --fold 0  # SVM
```

---

## Outputs

Generated artifacts are intentionally kept out of Git.

```text
data/WISDM/plots/<encoding>/sampling_<loto|loso>/3-fold/fold-<k>/
data/WISDM/tseries/<encoding>/sampling_<loto|loso>/3-fold/fold-<k>/
models/
results/
```

Path conventions are centralized in [src/paths.py](src/paths.py):

- `ALL_STYPES = ("RP", "GAF", "GADF", "MTF")`
- `GAF` is the code identifier for GASF.
- Per-run metrics are written under `results/<stype>/<classifier>/<sampling>/fold-<k>.json`.
- Original-signal classifier reports are cached under `results/_originals/`.

---

## Paper Utilities

The paper tables are stored as LaTeX fragments:

- LOTO tables: [doc/_3fold_tables.tex](doc/_3fold_tables.tex)
- LOSO tables: [doc/_loso_tables.tex](doc/_loso_tables.tex)
- Main paper: [doc/main.tex](doc/main.tex)

Useful checks:

```bash
python scripts/_check_refs.py
python scripts/_verify_paper_tables.py
python scripts/_verify_error_tables.py
```

No command here compiles the paper; these scripts only check consistency between the paper text/tables and stored result artifacts.

---

## Repository Layout

```text
.
  src/
    activity_data.py          WISDM loading and windowing
    recurrence_plots.py       RP encoder/decoder
    GAF.py                    GASF encoder/decoder
    GADF.py                   GADF encoder/decoder
    MTF.py                    MTF encoder/decoder wrapper
    MTF_v2.py                 Fiedler-vector MTF decoder
    reconstruct_tseries.py    image-to-signal reconstruction entry point
    evaluator.py              shared classifier evaluation pipeline
    models.py                 model and original-report cache helpers
    paths.py                  canonical path mapping
    results.py                JSON metric persistence
  scripts/
    run_pipeline.py           preferred orchestration entry point
    generate_images.py        image encoding stage
    generate_time_series.py   reconstruction stage
    *validation.py            classifier validators
    _*.py / _*.sh             paper, audit, smoke, and batch helper scripts
  doc/
    main.tex                  paper draft
    _3fold_tables.tex         LOTO paper tables
    _loso_tables.tex          LOSO paper tables
    workflow_overview.md      pipeline diagram source
  docs/
    experiments/              supporting experiment notes
```

---

## Troubleshooting

- If a validator silently reuses old behavior, delete the relevant file under `models/` and `results/_originals/`, or run with `--train-new --force`.
- If reconstructed arrays have the wrong shape, regenerate the corresponding fold under `data/WISDM/tseries/`; old `N=129` artifacts are not compatible with the current `N=128` protocol.
- If Windows install fails inside TensorFlow, shorten the repository or virtualenv path and enable long paths.
- If `TFG_DATA_DIR` points outside the repo, make sure it points to the directory that contains `wisdm-dataset/`.

---

## Citation

```bibtex
@misc{tfg-tsd,
  title  = {Time-series reconstruction from image-based encodings on WISDM},
  author = {Garcia-Giralda, Adriano},
  year   = {2026}
}
```
