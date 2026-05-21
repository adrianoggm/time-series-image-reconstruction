# TFG: Time-series reconstruction from image-based encodings

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](pyproject.toml)

Reference implementation of the TFG *"Diseño e implementación de la reconstrucción de series temporales a partir de imágenes"* (paper: *Diff-TSD: Modelling Time-series Data Generation with Diffusion Models*).

The repo implements four image encodings of triaxial accelerometer windows from the WISDM dataset — **Recurrence Plots (RP)**, **Gramian Angular Summation Fields (GASF)**, **Gramian Angular Difference Fields (GADF)**, and **Markov Transition Fields (MTF)** — together with a dedicated inversion pipeline for each, and evaluates the reconstructed signals via visual, pointwise, trend-level, and downstream-classification metrics.

The full pipeline diagram lives at [doc/workflow_overview.md](doc/workflow_overview.md).

---

## Quickstart

### 1. Install

The repo declares its dependencies in [pyproject.toml](pyproject.toml). Two equivalent ways to bootstrap a working environment:

**Linux / macOS / WSL:**
```bash
bash scripts/setup.sh                # core
bash scripts/setup.sh hf diffusion   # + optional extras
```

**Windows PowerShell:**
```powershell
.\scripts\setup.ps1
.\scripts\setup.ps1 -Extras hf,diffusion
```

Both scripts:
- install [`uv`](https://github.com/astral-sh/uv) locally if missing (no admin needed);
- ensure Python 3.10 is available;
- create a `.venv/` at the repo root;
- install the project in editable mode with the selected extras.

If you prefer doing it by hand:
```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e .                    # core
pip install -e ".[hf,diffusion]"    # with extras
```

> **Windows long-path note.** TensorFlow 2.11 unpacks deeply-nested header files that exceed the 260-character path limit on stock Windows. Either enable long paths (`Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1 -Type DWord` from an admin PowerShell), or create the venv outside a long repo path: `.\scripts\setup.ps1 -VenvDir "C:\Users\$env:USERNAME\tfg310"`.

### 2. Get the WISDM dataset

Download the *Smartphone and Smartwatch Activity and Biometrics Dataset* (UCI #507) and place it so the layout becomes:
```
<repo>/data/WISDM/wisdm-dataset/
└── raw/
    ├── phone/{accel,gyro}/    # 51 .txt per subdir
    └── watch/{accel,gyro}/    # 51 .txt per subdir
```
One-liner (Linux/WSL):
```bash
curl -L -o /tmp/wisdm.zip https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip
unzip -q /tmp/wisdm.zip -d data/WISDM/
unzip -q data/WISDM/wisdm-dataset.zip -d data/WISDM/
find data/WISDM -name ".DS_Store" -delete   # macOS junk in the official archive
rm data/WISDM/wisdm-dataset.zip /tmp/wisdm.zip
```

By default every script looks for `./data/WISDM/`. Override with the **`TFG_DATA_DIR`** environment variable if your data lives elsewhere:
```bash
export TFG_DATA_DIR=/path/to/WISDM/
```

### 3. Run

Three pipeline stages, each callable as a `python -m` module:

```bash
# (1) encode windows as RP/GASF/GADF/MTF images. --create-numpies only the first run.
python -m scripts.generate_images   --image-type GAF --sampling loto --n-folds 3 --create-numpies

# (2) invert the images back into time series
python -m scripts.generate_time_series --image-type GAF --sampling loto --n-folds 3

# (3) evaluate (one classifier at a time)
python -m scripts.knnvalidation
python -m scripts.lstmvalidation
python -m scripts.cnnvalidation
python -m scripts.gruvalidation
python -m scripts.randomforestvalidation
python -m scripts.Validation        # SVM
```

`--image-type` accepts `RP`, `GAF` (GASF), `GADF`, `MTF`. `--sampling` accepts `loto` (Leave-One-Trial-Out) or `loso` (Leave-One-Subject-Out).

---

## Repo layout

```
.
├── src/                # encoders, decoders, data loading, evaluation helpers
│   ├── activity_data.py        # WISDM loader + windowing
│   ├── recurrence_plots.py     # RP encode/decode
│   ├── GAF.py / GADF.py        # GASF & GADF encode/decode
│   ├── MTF.py                  # MTF encode/decode
│   ├── reconstruct_tseries.py  # inversion entry point
│   └── config.py               # TFG_DATA_DIR resolution
├── scripts/            # CLI entry points (python -m scripts.X)
├── doc/                # paper figures + workflow diagrams
├── docs/               # additional notes
│   ├── results/baseline_metrics.md      # historical classification reports
│   └── experiments/synthetic.md         # LaTeX fragment for the synthetic-signal experiment
└── pyproject.toml      # canonical dependency list
```

## Dataset details

We use the WISDM smartwatch accelerometer stream sampled at 20 Hz, restricted to five non-hand-oriented activities — Walking, Jogging, Stairs, Sitting, Standing — for a total of 1 053 141 samples across 51 subjects. Windows are 129 samples wide (≈ 6.4 s + 1 sample, so RP yields a 128×128 image after the adjacent-state reduction). Two sampling strategies are supported:

- **LOTO** (Leave-One-Trial-Out) — keeps trials atomic; matches the prior literature.
- **LOSO** (Leave-One-Subject-Out) — stricter subject-wise eval; lower accuracy but more honest generalisation.

| Activity | Instances | Share |
|----------|-----------|-------|
| Standing | 216 529   | 20.6 % |
| Sitting  | 213 018   | 20.2 % |
| Walking  | 210 495   | 20.0 % |
| Stairs   | 207 312   | 19.7 % |
| Jogging  | 205 787   | 19.5 % |

3-fold split sizes per sampling approach:

| Fold | LOTO train / test | LOSO train / test |
|------|-------------------|-------------------|
| 1    | 5392 / 2672       | 5408 / 2688       |
| 2    | 5392 / 2688       | 5344 / 2768       |
| 3    | 5392 / 2672       | 5456 / 2640       |

The generated RP dataset is mirrored on Hugging Face — see `src/upload_recurrence_plots_to_huggingface.py`. A few example RP images per class live in [doc/](doc/).

---

## Reference results

Historical classification reports across decoders (RP / MTF / GAF / original) and classifiers (KNN, LSTM, CNN, GRU, RandomForest, SVC) are recorded in [docs/results/baseline_metrics.md](docs/results/baseline_metrics.md). They are a frozen snapshot from the original experiments and may drift from current code output.

A small synthetic 3-channel signal (cosine / sine / sine−cosine) is used as a first end-to-end sanity check; the LaTeX description lives at [docs/experiments/synthetic.md](docs/experiments/synthetic.md).

---

## Background execution (long runs)

```bash
nohup python -m scripts.generate_images --image-type GAF --sampling loto --n-folds 3 --create-numpies > generate_images.log 2>&1 &
ps aux | grep generate_images
kill <PID>      # to stop
```

---

## Citation

```bibtex
@misc{tfg-tsd,
  title  = {Time-series reconstruction from image-based encodings on WISDM},
  author = {García-Giralda, Adriano},
  year   = {2025}
}
```

References for the encoding techniques and the dataset itself live in the paper draft under `doc/main.tex`.
