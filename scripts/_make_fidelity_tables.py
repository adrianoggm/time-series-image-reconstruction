"""Compact fidelity + spectral reconstruction table for the validation redesign
(doc/VALIDATION_REDESIGN.md, Blocks 1 & 2a), over all 8172 pooled windows.

Emits the main compact table
    Method | NMAE | NRMSE | Pearson | Delta-NMAE | PSD-shape distance | Dominant-freq error
and a JSON with the full metric set (raw + normalised delta-MAE, energy error,
band-shape difference). Range-normalised metrics make the four encoders
comparable regardless of each window's amplitude; MRE is dropped.

Reconstruction is deterministic and untrained, so the pooled set is the full
dataset (sampling-independent). Run inside the WSL venv:
    python scripts/_make_fidelity_tables.py > .fidelity_tables.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.fidelity_metrics import per_window_fidelity, per_window_spectral

DATA = Path("/home/adrianogarcia/tfg/data/WISDM")
FOLDS = (0, 1, 2)
SAMPLING = "sampling_loto"                     # pooled result is protocol-independent
RP_TREE = DATA / "tseries/recurrence_plot" / SAMPLING / "3-fold"
ENCODERS = [
    ("RP",   "tseries/recurrence_plot", "X_all_rec.npy"),
    ("GASF", "tseries/GAF",             "X_all_gaf.npy"),
    ("GADF", "tseries/GADF",            "X_all_gadf.npy"),
    ("MTF",  "tseries/MTF",             "X_all_mtf.npy"),
]
# Main-table columns: (metric key, header, decimals-for-non-MTF, decimals-for-MTF)
MAIN_COLS = [
    ("NMAE",       r"NMAE",                  4, 3),
    ("NRMSE",      r"NRMSE",                 4, 3),
    ("Pearson",    r"Pearson",               5, 3),
    ("dNMAE",      r"$\Delta$-NMAE",         4, 3),
    ("PSDShapeDist", r"PSD-shape dist.",     4, 3),
    ("DomFreqErr", r"Dom.-freq.\ err.\ (Hz)", 3, 3),
]


def _align(rec, shape):
    if rec.shape == shape:
        return rec
    if rec.shape == (shape[0], shape[2], shape[1]):
        return np.transpose(rec, (0, 2, 1))
    raise ValueError(f"cannot align {rec.shape} to {shape}")


def _load():
    orig = np.concatenate([np.load(RP_TREE / f"fold-{f}/test/validation_data.npy") for f in FOLDS])
    recs = {}
    for enc, sub, npy in ENCODERS:
        r = np.concatenate([np.load(DATA / sub / SAMPLING / "3-fold" / f"fold-{f}/test/{npy}") for f in FOLDS])
        recs[enc] = _align(r, orig.shape)
    return orig, recs


def main():
    orig, recs = _load()
    print(f"% pooled originals {orig.shape}, N={len(orig)}", file=sys.stderr)

    stats = {}
    for enc, _, _ in ENCODERS:
        m = {**per_window_fidelity(orig, recs[enc]), **per_window_spectral(orig, recs[enc])}
        stats[enc] = {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                          "median": float(np.nanmedian(v)),
                          "n_valid": int(np.isfinite(v).sum())} for k, v in m.items()}
        print(f"% {enc}: NMAE={stats[enc]['NMAE']['mean']:.4f} "
              f"Pearson={stats[enc]['Pearson']['mean']:.5f} "
              f"dMAE={stats[enc]['dMAE']['mean']:.4f} "
              f"PSD-shape={stats[enc]['PSDShapeDist']['mean']:.4f} "
              f"domHz={stats[enc]['DomFreqErr']['mean']:.3f}", file=sys.stderr)

    # ---- LaTeX main table ----
    out = [
        r"\begin{table}[H]", r"\centering",
        (r"\caption{Compact fidelity and spectral reconstruction metrics over all "
         r"$N=8172$ pooled windows (mean$\pm$std across windows, averaged over the "
         r"three axes). NMAE/NRMSE and $\Delta$-NMAE are normalised by each window's "
         r"amplitude range; PSD-shape distance compares unit-mass spectra, and "
         r"dominant-frequency error comes from "
         r"Welch periodograms ($f_s=20$\,Hz). This replaces the cubic-polynomial "
         r"coefficient distances.}"),
        r"\label{tab:fidelity_spectral}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(MAIN_COLS) + "}", r"\toprule",
        r"\textbf{Method} & " + " & ".join(fr"\textbf{{{h}}}" for _, h, _, _ in MAIN_COLS) + r" \\",
        r"\midrule",
    ]
    for enc, _, _ in ENCODERS:
        cells = [enc]
        for key, _h, dnl, dmtf in MAIN_COLS:
            d = dmtf if enc == "MTF" else dnl
            s = stats[enc][key]
            cells.append(fr"${s['mean']:.{d}f} \pm {s['std']:.{d}f}$")
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    print("\n".join(out))

    # Persist inside the repo, not /tmp: the paper's numbers must stay traceable to
    # an artefact that survives a reboot.
    out = Path(__file__).resolve().parents[1] / "results" / "_stats" / "fidelity_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"% wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
