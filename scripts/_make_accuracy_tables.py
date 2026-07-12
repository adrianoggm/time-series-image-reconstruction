"""Emit the subject-out (LOSO) headline + per-fold accuracy tables for the paper
from the JSON result snapshot, so every column is internally consistent and
reproducible after the RP-decoder fix.

    python -m scripts._make_accuracy_tables

Orig column is sourced from GADF's committed 'original' report (canonical; the
non-RP encoders share it). Each encoder column is that encoder's 'reconstructed'
accuracy. GASF is read from the GAF folder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / ".results_snapshot"
SAMPLING = "loso"
FOLDS = (0, 1, 2)
# (display, results-key)
CLFS = [("GRU", "gru"), ("KNN", "knn"), ("CNN", "cnn"),
        ("Random Forest", "randomforest"), ("Support Vector Machine", "svc"),
        ("LSTM", "lstm")]
# (column label, snapshot folder)  — GASF lives under GAF/
ENCS = [("GASF", "GAF"), ("GADF", "GADF"), ("RP", "RP"), ("MTF", "MTF")]
ORIG_FOLDER = "GADF"  # canonical source of the shared 'original' report


def _acc(folder, clf, fold, key):
    d = json.load(open(ROOT / folder / clf / SAMPLING / f"fold-{fold}.json"))["evaluations"]
    return d[key]["accuracy"]


def _fold_accs(folder, clf, key):
    return [_acc(folder, clf, f, key) for f in FOLDS]


def _dfmt(delta):
    return f"(+{delta:.3f})" if delta >= 0 else f"($-${abs(delta):.3f})"


def headline():
    print("%%%%% tab:gadf_summary_subject_out (regenerated) %%%%%")
    for disp, clf in CLFS:
        orig = _fold_accs(ORIG_FOLDER, clf, "original")
        om, os_ = np.mean(orig), np.std(orig)
        cells = [f"${om:.3f} \\pm {os_:.3f}$"]
        for _, folder in ENCS:
            rec = _fold_accs(folder, clf, "reconstructed")
            rm, rs = np.mean(rec), np.std(rec)
            cells.append(f"${rm:.3f} \\pm {rs:.3f}$ {_dfmt(rm - om)}")
        print(f"{disp} & " + " & ".join(cells) + r" \\")


def perfold():
    print("\n%%%%% tab:gadf_summary_perfold_subject_out (regenerated) %%%%%")
    for disp, clf in CLFS:
        blocks = [_fold_accs(ORIG_FOLDER, clf, "original")]
        for _, folder in ENCS:
            blocks.append(_fold_accs(folder, clf, "reconstructed"))
        cells = " & ".join(f"{v:.3f}" for blk in blocks for v in blk)
        print(f"{disp} & {cells} " + r"\\")


if __name__ == "__main__":
    headline()
    perfold()
