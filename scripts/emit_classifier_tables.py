"""Emit the LaTeX body of the per-classifier result tables from the results JSONs.

For a (classifier, sampling) it reads results/<stype>/<clf>/<sampling>/fold-{0,1,2}.json,
aggregates per-class precision/recall/F1 and the footer accuracy/macro/weighted rows
as mean +/- std across the three folds, and prints the table rows in the paper's
column order (Original, RP, MTF, GASF, GADF).

    python -m scripts.emit_classifier_tables --sampling loto
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.paths import results_dir

CLASSIFIERS = ["gru", "knn", "cnn", "randomforest", "svc", "lstm"]
# (paper column label, stype key). "Original" is read from any stype's "original" block.
COLS = [("Original", None), ("RP", "RP"), ("MTF", "MTF"), ("GASF", "GAF"), ("GADF", "GADF")]
CLASSES = ["0", "1", "2", "3", "4"]


def _load(clf, stype, sampling, fold):
    p = results_dir(stype, clf, sampling=sampling) / f"fold-{fold}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _cell(vals):
    a = np.asarray(vals, dtype=float)
    return f"{a.mean():.2f}$\\pm${a.std():.2f}"


def _reports(clf, sampling):
    """Return {source_label: [report_fold0, report_fold1, report_fold2]}."""
    out = {}
    for label, stype in COLS:
        reps = []
        for fold in (0, 1, 2):
            # Original block is identical across encoders; read it from RP.
            j = _load(clf, "RP" if stype is None else stype, sampling, fold)
            reps.append(j["evaluations"]["original" if stype is None else "reconstructed"])
        out[label] = reps
    return out


def emit(clf, sampling):
    reps = _reports(clf, sampling)
    lines = []
    # per-class rows
    for c in CLASSES:
        cells = []
        supp = None
        for label, _ in COLS:
            r = reps[label]
            prec = _cell([f[c]["precision"] for f in r])
            rec = _cell([f[c]["recall"] for f in r])
            f1 = _cell([f[c]["f1-score"] for f in r])
            supp = int(round(np.mean([f[c]["support"] for f in r])))
            cells.append(f"{prec} & {rec} & {f1} & {supp}")
        lines.append(f"{c} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    # footer: accuracy (with per-fold N), macro avg, weighted avg
    n_test = int(round(np.mean([sum(reps["Original"][f][c]["support"] for c in CLASSES) for f in range(3)])))
    acc_cells = []
    for label, _ in COLS:
        r = reps[label]
        acc_cells.append("\\multicolumn{4}{c" + ("|" if label != "GADF" else "") + "}{" +
                         _cell([f["accuracy"] for f in r]) + f" ({n_test})" + "}")
    lines.append("Accuracy & " + " & ".join(acc_cells) + r" \\")
    for avg in ("macro avg", "weighted avg"):
        cells = []
        for label, _ in COLS:
            r = reps[label]
            p = _cell([f[avg]["precision"] for f in r])
            rc = _cell([f[avg]["recall"] for f in r])
            f1 = _cell([f[avg]["f1-score"] for f in r])
            cells.append("\\multicolumn{4}{c" + ("|" if label != "GADF" else "") + "}{" +
                        f"{p}/{rc}/{f1}" + "}")
        name = "Macro Avg" if avg == "macro avg" else "Weighted Avg"
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampling", default="loto")
    ap.add_argument("--classifier", default=None, help="one of gru,knn,cnn,randomforest,svc,lstm; default all")
    args = ap.parse_args()
    clfs = [args.classifier] if args.classifier else CLASSIFIERS
    for clf in clfs:
        print(f"%%%%% {clf} {args.sampling} %%%%%")
        print(emit(clf, args.sampling))
        print()


if __name__ == "__main__":
    main()
