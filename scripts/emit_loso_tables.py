"""Emit the numeric row bodies of the three grouped subject-out (LOSO) summary
tables from the results JSONs: headline accuracies (+ Delta acc), per-fold
accuracies, and the LOTO-vs-subject-out side-by-side.

Column/source order matches the paper: Orig., GASF, GADF, RP, MTF.
"""

from __future__ import annotations

import json

import numpy as np

from src.paths import results_dir

CLFS = [("GRU", "gru"), ("KNN", "knn"), ("CNN", "cnn"),
        ("Random Forest", "randomforest"), ("Support Vector Machine", "svc"), ("LSTM", "lstm")]
SRCS = [("Orig.", None), ("GASF", "GAF"), ("GADF", "GADF"), ("RP", "RP"), ("MTF", "MTF")]


def _accs(clf, stype, sampling):
    st = "RP" if stype is None else stype
    which = "original" if stype is None else "reconstructed"
    out = []
    for f in (0, 1, 2):
        j = json.loads((results_dir(st, clf, sampling=sampling) / f"fold-{f}.json").read_text())
        out.append(j["evaluations"][which]["accuracy"])
    return np.array(out)


def headline_loso():
    rows = []
    for name, clf in CLFS:
        orig = _accs(clf, None, "loso")
        cells = [f"${orig.mean():.3f} \\pm {orig.std():.3f}$"]
        for _lbl, stype in SRCS[1:]:
            a = _accs(clf, stype, "loso")
            d = a.mean() - orig.mean()
            sign = "+" if d >= 0 else "$-$"
            cells.append(f"${a.mean():.3f} \\pm {a.std():.3f}$ ({sign}{abs(d):.3f})")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def perfold_loso():
    rows = []
    for name, clf in CLFS:
        cells = []
        for _lbl, stype in SRCS:
            a = _accs(clf, stype, "loso")
            cells += [f"{a[0]:.3f}", f"{a[1]:.3f}", f"{a[2]:.3f}"]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def loto_vs_loso():
    rows = []
    for name, clf in CLFS:
        cells = []
        for _lbl, stype in SRCS:
            lt = _accs(clf, stype, "loto").mean()
            lo = _accs(clf, stype, "loso").mean()
            cells += [f"{lt:.3f}", f"{lo:.3f}"]
        rows.append(f"{name} &  & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


BLOCKS = {
    "tab:gadf_summary_loso": headline_loso,
    "tab:gadf_summary_perfold_loso": perfold_loso,
    "tab:loto_vs_loso": loto_vs_loso,
}


def main():
    for label, fn in BLOCKS.items():
        print(f"%%%%% {label} %%%%%")
        print(fn())
        print()


if __name__ == "__main__":
    main()
