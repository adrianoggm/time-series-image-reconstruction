"""Generate the LOSO 3-fold LaTeX tables for ``doc/main.tex``.

Same structure as _make_paper_tables_3fold.py but reads from results/<stype>/<clf>/loso/.
Emits two tables:
  - tab:gadf_summary_loso          headline accuracies, mean+/-std across 3 LOSO folds
  - tab:gadf_summary_perfold_loso  the 3 individual fold accuracies

Plus a comparison table:
  - tab:loto_vs_loso               side-by-side LOTO vs LOSO means per (classifier, encoder)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "results"
ORIGINALS_DIR = ROOT / "_originals"

CLASSIFIERS = [
    ("gru",          "GRU"),
    ("knn",          "KNN"),
    ("cnn",          "CNN"),
    ("randomforest", "Random Forest"),
    ("svc",          "Support Vector Machine"),
    ("lstm",         "LSTM"),
]
STYPES = [("RP", "RP"), ("MTF", "MTF"), ("GAF", "GASF"), ("GADF", "GADF")]
FOLDS = [0, 1, 2]


def _agg(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _fmt_mean_std(m, s, n=3):
    return f"${m:.{n}f} \\pm {s:.{n}f}$"


def _load_recon(stype, clf, sampling, fold):
    p = ROOT / stype / clf / sampling / f"fold-{fold}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _load_orig(clf, sampling, fold):
    """LOSO/LOTO original-report caches share the same filename pattern."""
    p = ORIGINALS_DIR / clf / f"fold-{fold}-{sampling}-3f.json"
    return json.loads(p.read_text(encoding="utf-8"))


def build_summary(sampling: str, label_suffix: str, caption: str) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{" + caption + r"}",
        rf"\label{{tab:gadf_summary_{label_suffix}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\footnotesize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Classifier} & \textbf{Orig.} & \textbf{GASF} & \textbf{GADF} "
        r"& \textbf{RP} & \textbf{MTF} \\",
        r"\midrule",
    ]
    for clf_key, clf_label in CLASSIFIERS:
        orig_accs = [_load_orig(clf_key, sampling, f)["accuracy"] for f in FOLDS]
        om, os_ = _agg(orig_accs)
        row = [clf_label, _fmt_mean_std(om, os_)]
        cells_for_stype = {}
        for stype_key, stype_label in STYPES:
            rec_accs = [
                _load_recon(stype_key, clf_key, sampling, f)
                ["evaluations"]["reconstructed"]["accuracy"]
                for f in FOLDS
            ]
            rm, rs = _agg(rec_accs)
            delta_mean = rm - om
            sign = "+" if delta_mean >= 0 else "$-$"
            cells_for_stype[stype_label] = f"{_fmt_mean_std(rm, rs)} ({sign}{abs(delta_mean):.3f})"
        # Column order: Original | GASF | GADF | RP | MTF
        row.extend([cells_for_stype[c] for c in ("GASF", "GADF", "RP", "MTF")])
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_perfold(sampling: str, label_suffix: str, caption: str) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{" + caption + r"}",
        rf"\label{{tab:gadf_summary_perfold_{label_suffix}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\footnotesize",
        r"\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{\textbf{Orig.}} "
        r"& \multicolumn{3}{c|}{\textbf{GASF}} "
        r"& \multicolumn{3}{c|}{\textbf{GADF}} "
        r"& \multicolumn{3}{c|}{\textbf{RP}} "
        r"& \multicolumn{3}{c}{\textbf{MTF}} \\",
        r"\textbf{Classifier} & f0 & f1 & f2 & f0 & f1 & f2 & f0 & f1 & f2 "
        r"& f0 & f1 & f2 & f0 & f1 & f2 \\",
        r"\midrule",
    ]
    for clf_key, clf_label in CLASSIFIERS:
        orig = [f"{_load_orig(clf_key, sampling, f)['accuracy']:.3f}" for f in FOLDS]
        cells = [clf_label, *orig]
        for stype_key in ("GAF", "GADF", "RP", "MTF"):
            for f in FOLDS:
                acc = _load_recon(stype_key, clf_key, sampling, f)["evaluations"]["reconstructed"]["accuracy"]
                cells.append(f"{acc:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def build_loto_vs_loso() -> str:
    """One row per (classifier, encoder) showing LOTO mean and LOSO mean side by side."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{LOTO vs LOSO accuracies (mean across 3 folds). LOSO holds out "
        r"16--18 subjects per fold with zero overlap, so it is the stricter "
        r"subject-out generalisation test. The ranking $\text{GASF}\succ\text{GADF}\succ\text{RP}\succ\text{MTF}$ "
        r"is preserved under both sampling schemes; absolute accuracies are lower "
        r"under LOSO because the model has not seen any window from the test "
        r"subjects.}",
        r"\label{tab:loto_vs_loso}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\footnotesize",
        r"\begin{tabular}{ll|cc|cc|cc|cc|cc}",
        r"\toprule",
        r" & & \multicolumn{2}{c|}{\textbf{Orig.}} "
        r"& \multicolumn{2}{c|}{\textbf{GASF}} "
        r"& \multicolumn{2}{c|}{\textbf{GADF}} "
        r"& \multicolumn{2}{c|}{\textbf{RP}} "
        r"& \multicolumn{2}{c}{\textbf{MTF}} \\",
        r"\textbf{Classifier} & & LOTO & LOSO & LOTO & LOSO & LOTO & LOSO "
        r"& LOTO & LOSO & LOTO & LOSO \\",
        r"\midrule",
    ]
    for clf_key, clf_label in CLASSIFIERS:
        row_cells = [clf_label, ""]
        for sampling in ("loto", "loso"):
            orig = np.mean([_load_orig(clf_key, sampling, f)["accuracy"] for f in FOLDS])
            row_cells.append(f"{orig:.3f}")
            # break out, we'll rebuild row differently
        # Simpler: collect dict[col_pair] -> (loto, loso)
        cols = {}
        for sampling in ("loto", "loso"):
            cols[("Orig", sampling)] = float(np.mean(
                [_load_orig(clf_key, sampling, f)["accuracy"] for f in FOLDS]
            ))
            for stype_key in ("GAF", "GADF", "RP", "MTF"):
                cols[(stype_key, sampling)] = float(np.mean([
                    _load_recon(stype_key, clf_key, sampling, f)
                    ["evaluations"]["reconstructed"]["accuracy"]
                    for f in FOLDS
                ]))
        out = [clf_label, ""]
        for src in ("Orig", "GAF", "GADF", "RP", "MTF"):
            out.append(f"{cols[(src,'loto')]:.3f}")
            out.append(f"{cols[(src,'loso')]:.3f}")
        lines.append(" & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    print("% ====== LOSO 3-fold headline summary ======")
    print(build_summary(
        "loso", "loso",
        "Headline accuracies under the LOSO 3-fold protocol "
        "(leave-one-subject-out; mean $\\pm$ std across folds 0, 1, 2; "
        "$\\sim$2\\,742 test windows per fold drawn from 16--18 held-out "
        "subjects). $\\Delta$acc is the accuracy of the reconstructed signal "
        "minus the accuracy of the original signal under the same classifier, "
        "averaged over the same three folds.",
    ))
    print()
    print("% ====== LOSO per-fold breakdown ======")
    print(build_perfold(
        "loso", "loso",
        "Per-fold accuracies on the LOSO 3-fold protocol. Folds split the 51 "
        "WISDM subjects into disjoint groups of 17, 18 and 16 test subjects "
        "respectively; the test set of one fold never shares a subject with "
        "either of its training folds.",
    ))
    print()
    print("% ====== LOTO vs LOSO side-by-side ======")
    print(build_loto_vs_loso())


if __name__ == "__main__":
    main()
