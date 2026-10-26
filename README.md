# How Much of a Time Series Survives Its Image?

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](pyproject.toml)

Code, results and manuscripts for a comparative analysis of how well time series can be **reconstructed from
image-based representations**: recurrence plots (binary, unsigned and signed), Gramian angular summation and
difference fields (GASF/GADF), Markov transition fields (MTF), and the invertible-by-design XIRP as a reference.

Every image is stored as an 8-bit picture and decoded **from the image alone**. The minimum and maximum of each
channel are used only to return to physical units. Every decoder is analytic; no generative model is trained here.

The repository holds two studies:

| Study | Manuscript | Scope |
|---|---|---|
| **Cross-domain benchmark** (current) | [doc/infsci/main.tex](doc/infsci/main.tex) + [supplementary.tex](doc/infsci/supplementary.tex), prepared for *Information Sciences* | 7 datasets, 6 domains, 10 encoding–decoder combinations, fidelity, robustness, cost and classification utility |
| **Original study** (bachelor's thesis) | [doc/main3.tex](doc/main3.tex) + [doc/supplementary.tex](doc/supplementary.tex) | WISDM accelerometer only; RP, GASF, GADF, MTF; subject-out equivalence analysis |

---

## Encodings and decoders

| Encoding | What the image keeps | Decoder | Exact before storage? |
|---|---|---|---|
| Signed RP | distances and directional signs | **Geometric decoder** (this work): MDS, closed-form overlap orientation, sign selection | Yes, under the conditions of Proposition 1 |
| Signed RP | | Hirata et al. (graph geodesics) and Sipers et al. (spectral, circular RP), with our sign step | No / yes, respectively (Sipers with its frequency threshold) |
| Unsigned RP | distances | Geometric (without sign), Hirata, Sipers | Up to a global reflection |
| Binary RP | recurrence neighbourhoods | Hirata et al. | No (many-to-one) |
| GASF | angular sums | Diagonal inverse on [0, 1] | Yes |
| GADF | angular differences | **Restricted-branch, boundary-anchor decoder** (this work), α = 0.7 | Yes, on the restricted range |
| MTF | state transitions | Spectrally ordered representative (this work), Q = 4 | No (many-to-one) |
| XIRP | log returns and the series on the diagonal | Inversion by mean (Hellermann and Lessmann) | Yes |

All of them are registered in [src/representations.py](src/representations.py) behind one per-channel API:
`encode`, `to_uint8` / `from_uint8` and `decode`.

---

## Main results

All numbers are from the benchmark manuscript and are regenerated from the stored artefacts
(see [Reproducing the benchmark](#reproducing-the-benchmark)).

- **Exact decoding.** The signed RP with the geometric decoder, GASF, GADF and XIRP recover every one of the
  61,079 benchmark series to machine precision from the floating-point matrix. The largest median error per block
  is 1.4e-13 for the signed RP.
- **8-bit storage.** For these four, the remaining error is storage error. The signed RP stays between 0.0009 and
  0.0022 NMAE on six datasets and rises to 0.012–0.017 on Swarm, whose states are nearly collinear.
- **Conditioning.** None of the 54,040 series with σ₂/σ₁ ≥ 0.1 exceeds an 8-bit NMAE of 0.01. On PTB-XL, the signed
  RP error grows with the sampling rate, from 0.0009 at 50 Hz to 0.0076 at 500 Hz.
- **Decoder versus encoding.** On the same circular signed RP, the geometric decoder is 1.8 to 4.8 times more
  accurate than Sipers'. The unsigned RP keeps the spectrum but loses the orientation; the binary RP and MTF lose
  the waveform.
- **Classification utility.** Classifiers are trained on original windows and tested on reconstructions. The four
  exactly decodable combinations never lose more than 0.007 accuracy and are equivalent within ±0.01 in nearly every
  single-label combination (63 of 66 for the geometric decoder).
- **Cost.** The geometric decoder is 2.9 times faster than Hirata's at T = 256 and 5.2 times at T = 1024.

---

## Datasets

| Dataset | Domain | f<sub>s</sub> | Channels | Source | Default location |
|---|---|---|---|---|---|
| WISDM | activity (smartwatch) | 20 Hz | 3 | [UCI 507](https://archive.ics.uci.edu/dataset/507) | `data/WISDM/` or `TFG_DATA_DIR` |
| PAMAP2 | activity (IMU) | 100 Hz | 9 | [UCI 231](https://archive.ics.uci.edu/dataset/231) | `~/datasets/PAMAP2/PAMAP2_Dataset` |
| ESC-50 | environmental audio | 8 kHz (resampled) | 1 | [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50) | `~/datasets/ESC-50/ESC-50-master` |
| PTB-XL | electrocardiography | 100 Hz (500 Hz for the sampling-rate study) | 12 | [PhysioNet](https://physionet.org/content/ptb-xl/) | `~/datasets/PTB-XL` |
| Swarm MAG | geomagnetism | 1 Hz | 3 | ESA VirES, via [scripts/_swarm_download.py](scripts/_swarm_download.py) (needs a VirES token in `~/.viresclient.ini`) | `~/datasets/SWARM` |
| STEAD | seismology | 100 Hz | 3 | [STEAD](https://github.com/smousavi05/STEAD), chunks 1 and 2 | `~/datasets/STEAD` |
| EMG gestures | electromyography | 100 Hz | 8 | [UCI 481](https://archive.ics.uci.edu/dataset/481) | `~/datasets/EMG-Gestures/EMG_data_for_gestures-master` |

The adapters in [src/adapters.py](src/adapters.py) read each dataset and segment it through
[src/datasets.py](src/datasets.py). Windows never cross a label change or a time gap, and the grouped folds keep
every subject, patient, event, station or source recording on one side of the split (`group_kfold` asserts it).

The WISDM archive ships with macOS `.DS_Store` files that crash the loader; delete them after unzipping.
`TFG_DATA_DIR` must end with a path separator.

---

## Setup

Python `>=3.10,<3.12`: TensorFlow 2.11–2.15 has no wheels outside that range, and Keras 3 is excluded.
[pyproject.toml](pyproject.toml) is the dependency list.

```bash
bash scripts/setup.sh          # Linux, macOS, WSL (installs uv, creates .venv, pip install -e .)
bash scripts/setup.sh dev      # + ruff and pre-commit
```

```powershell
.\scripts\setup.ps1            # Windows
```

The editable install is required: `src/` and `scripts/` are packages, imported as `from src.X import Y`.
On Windows, TensorFlow needs long paths enabled, or a venv outside the repo
(`.\scripts\setup.ps1 -VenvDir "C:\Users\$env:USERNAME\tfg310"`). The heavy runs were done in WSL2.

---

## Reproducing the benchmark

Run every stage with `OMP_NUM_THREADS=1`: the stages parallelise per window, and multithreaded BLAS inside each
worker oversubscribes the cores.

```bash
export OMP_NUM_THREADS=1
python -m scripts._benchmark --n 500                 # fidelity, float and 8-bit, T = 64/128/256 (3000 windows for ESC-50)
python -m scripts._benchmark_stats                   # paired statistics over the stored blocks
python -m scripts._robustness_multi                  # Gaussian noise on the stored image
python -m scripts._sampling_rate                     # PTB-XL resampled from 500 Hz (add --duration 1.28 for fixed duration)
python -m scripts._cost_scaling                      # time and memory, T = 64 ... 1024
python -m scripts._rp_conditions                     # sigma2/sigma1 conditioning of the RP states
python -m scripts._hirata_threshold_sweep            # Hirata's recurrence-rate sensitivity
python -m scripts._sipers_pilot                      # Sipers vs geometric decoder on the same circular image
python -m scripts._sipers_calibration                # Sipers' threshold on windows outside every evaluated sample
python -m scripts._classification                    # 6 classifiers trained on originals, tested on reconstructions
python -m scripts._classification --out results/_classification_esc10 \
       --lengths 512,1024 --datasets esc10_2k,esc10_4k,esc10_8k   # audio: ESC-10, accuracy per clip
```

Checks behind specific claims of the manuscript:

```bash
python -m scripts._check_sign_identifiability        # hypotheses of Proposition 1 on every benchmark series
python -m scripts._check_image_only                  # the calibration enters every decoder only as the final affine map
python -m scripts._check_representations             # new API bit-identical to the legacy WISDM pipeline
```

Every stage is resumable and accepts `--only <technique>`: a new representation added to `REPS`
([scripts/_benchmark.py](scripts/_benchmark.py)) and `TECHS` ([scripts/_classification.py](scripts/_classification.py))
is computed on the stored windows and merged, without recomputing the published columns. Trained classifiers are kept
under `results/_classification/models/` (untracked, a few GB) and reused.

Then regenerate every table, quoted number and figure, and build both documents:

```bash
python -m scripts._emit_infsci_tables      # doc/infsci/generated/*.tex, including numbers.tex
python -m scripts._make_infsci_figures     # robustness and sampling-rate figure
python -m scripts._make_infsci_examples    # encoding collage and reconstruction grid
bash scripts/_build_infsci.sh              # main <-> supplementary through xr; reports warnings and '??'
```

No number in [doc/infsci/main.tex](doc/infsci/main.tex) is typed by hand except design parameters; they come from
`generated/numbers.tex`. The audit of the Sipers threshold, the proposition's hypotheses, image-only decoding and
the hand-written ranges is in [docs/audit/infsci_reproducibility.md](docs/audit/infsci_reproducibility.md).

### Stored results

| Path | Content |
|---|---|
| `results/_benchmark/` | per-dataset, per-length fidelity arrays and summaries |
| `results/_classification/` | per-classifier, per-fold accuracy on originals and reconstructions |
| `results/_stats/` | robustness, sampling rate, cost, conditioning, Hirata sweep, Sipers pilot and calibration, Proposition 1 check |
| `doc/infsci/generated/` | the tables and macros the manuscript inputs |

---

## The original WISDM study

The thesis pipeline encodes WISDM windows into PNG trees, reconstructs them and validates six classifiers under
trial-out (`loto`) and subject-out (`loso`) three-fold splits.

```bash
python -m scripts.run_pipeline --stype GADF --sampling loso --fold 0 --n-folds 3 --create-numpies
python -m scripts.run_pipeline --stype GAF --skip-encode --skip-reconstruct     # re-validate from cache
python -m scripts.summarize_results --out docs/results/latest.md
```

`GAF` is the code identifier for GASF. [src/evaluator.py](src/evaluator.py) caches trained models, original-data
reports and finished runs. After changing a classifier, pass `--train-new --force` or the stale model is reused;
[CLAUDE.md](CLAUDE.md) lists the cache files. Its statistics (`_stat_tests`, `_stat_per_subject`) and figures
(`_make_paper_figures`) write to `results/_stats/` and `doc/images/`; build with
`cd doc && latexmk -pdf main3.tex && latexmk -pdf supplementary.tex`.

---

## Repository layout

```text
src/
  datasets.py          dataset interface: recordings, windowing, grouped folds
  adapters.py          the seven dataset adapters
  representations.py   every encoding/decoder behind one API
  rp_decoders.py       geometric RP decoder (MDS + closed-form orientation)
  rp_hirata.py         Hirata et al. on the binary RP
  rp_sipers.py         Sipers et al. on the circular RP
  GAF.py GADF.py MTF.py XIRP.py recurrence_plots.py   per-encoding code
  activity_data.py evaluator.py reconstruct_tseries.py ...   original WISDM pipeline
scripts/               entry points, run as python -m scripts.<name> (CLAUDE.md lists every one)
doc/infsci/            Information Sciences manuscript, supplement and generated tables
doc/                   thesis manuscript (main3.tex), its supplement and figures
docs/audit/            reproducibility audits
results/               stored results (large intermediate arrays and models are untracked)
```

Lint with `ruff check .` (configured in `pyproject.toml`). There is no test suite; CI runs an import smoke test
(`.github/workflows/smoke.yml`) and the `_check_*` scripts above are the gates.

---

## Citation

```bibtex
@misc{garciagiralda2026survives,
  title  = {How Much of a Time Series Survives Its Image? A Cross-Domain Comparative Analysis of the
            Reconstruction Capacity of Image-Based Representations},
  author = {Garc{\'\i}a-Giralda Milena, Adriano and Garc{\'\i}a Moreno, Francisco and
            Jim{\'e}nez Madrid, Jos{\'e} Antonio and Berm{\'u}dez Edo, Mar{\'\i}a},
  year   = {2026},
  note   = {Manuscript in preparation. Code: https://github.com/adrianoggm/time-series-image-reconstruction}
}
```

Released under the [MIT License](LICENSE). The datasets keep their own licences.
