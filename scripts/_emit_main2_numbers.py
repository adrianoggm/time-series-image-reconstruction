"""Emit the exact figures main2.tex's classification and equivalence tables need,
read from the LIVE results tree and the regenerated stat reports.

    python -m scripts._emit_main2_numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STATS = RES / "_stats"
FOLDS = (0, 1, 2)
ENC = [("RP", "RP"), ("GASF", "GAF"), ("GADF", "GADF"), ("MTF", "MTF")]
CLFS = [("CNN", "cnn"), ("GRU", "gru"), ("RF", "randomforest"),
        ("KNN", "knn"), ("SVM", "svc"), ("LSTM", "lstm")]


def _acc(folder, clf, sampling, fold, key):
    p = RES / folder / clf / sampling / f"fold-{fold}.json"
    ev = json.loads(p.read_text())["evaluations"]
    r = ev[key]["report"] if "report" in ev[key] else ev[key]
    return float(r["accuracy"])


def _mean(folder, clf, sampling, key):
    return float(np.mean([_acc(folder, clf, sampling, f, key) for f in FOLDS]))


def _ms(folder, clf, sampling, key):
    v = np.array([_acc(folder, clf, sampling, f, key) for f in FOLDS], float)
    return f"${v.mean():.3f}\\pm{v.std():.3f}$"


def main():
    for sampling in ("loso", "loto"):
        print(f"\n===== tab:subject_out_acc rows ({sampling.upper()}) =====")
        print("Classifier & Original & RP & GASF & GADF & MTF")
        for name, clf in CLFS:
            # Original is model-only and shared; read it from GADF as the paper does.
            orig = _mean("GADF", clf, sampling, "original")
            cells = [f"{orig:.3f}"]
            for _lbl, folder in ENC:
                cells.append(f"{_mean(folder, clf, sampling, 'reconstructed'):.3f}")
            print(f"{name} & " + " & ".join(cells) + r" \\")

    print("\n===== tab:gadf_summary_perfold_subject_out (LOSO, mean+-std) =====")
    for name, clf in [("KNN", "knn"), ("SVM", "svc"), ("RF", "randomforest"),
                      ("LSTM", "lstm"), ("GRU", "gru"), ("CNN", "cnn")]:
        cells = [_ms("GADF", clf, "loso", "original")]
        cells += [_ms(folder, clf, "loso", "reconstructed") for _lbl, folder in ENC]
        print(f"{name:<4} & " + " & ".join(cells) + r" \\")

    print("\n===== tab:window_vs_subject_out (W=loto, S=loso) =====")
    for name, clf in [("CNN", "cnn"), ("GRU", "gru"), ("RF", "randomforest")]:
        cells = [f"{_mean('GADF', clf, 'loto', 'original'):.3f}",
                 f"{_mean('GADF', clf, 'loso', 'original'):.3f}"]
        for _lbl, folder in ENC:
            cells.append(f"{_mean(folder, clf, 'loto', 'reconstructed'):.3f}")
            cells.append(f"{_mean(folder, clf, 'loso', 'reconstructed'):.3f}")
        print(f"{name:<4} & " + " & ".join(cells) + r" \\")

    for sampling in ("loso", "loto"):
        p = STATS / f"stat_report_{sampling}.json"
        if not p.exists():
            print(f"\n{sampling}: MISSING {p}")
            continue
        d = json.loads(p.read_text())
        ci = d["subject_cluster_bootstrap_simultaneous_ci_accuracy"]
        print(f"\n===== tab:equivalence_ci rows ({sampling.upper()}) =====")
        label = {"RP": "RP", "GAF": "GASF", "GADF": "GADF", "MTF": "MTF"}
        clf_label = {"knn": "KNN", "svc": "SVM", "randomforest": "RF",
                     "lstm": "LSTM", "gru": "GRU", "cnn": "CNN"}
        print(f"--- individual 24 intervals ({sampling}) ---")
        for _n, clf in [("KNN", "knn"), ("SVM", "svc"), ("RF", "randomforest"),
                        ("LSTM", "lstm"), ("GRU", "gru"), ("CNN", "cnn")]:
            cells = []
            for key in ("RP", "GAF", "GADF", "MTF"):
                e = next(x for x in ci if x["name"] == f"{clf}/Original-vs-{key}")
                cell = (f"{e['mean_delta_acc']:+.4f}\\ [{e['ci_lo']:+.4f},{e['ci_hi']:+.4f}]")
                cells.append(f"$\\mathbf{{{cell}}}$" if e["equivalent_0.01"] else f"${cell}$")
            print(f"{_n:<4} & " + " & ".join(cells) + r" \\")
        print(f"--- interval spans ({sampling}) ---")
        for key in ("RP", "GAF", "GADF", "MTF"):
            rows = [e for e in ci if e["name"].endswith(f"-vs-{key}")]
            print(f"{key:<4} span [{min(e['ci_lo'] for e in rows):+.4f},"
                  f"{max(e['ci_hi'] for e in rows):+.4f}]")
        for key in ("RP", "GAF", "GADF", "MTF"):
            rows = [e for e in ci if e["name"].endswith(f"-vs-{key}")]
            deltas = [e["mean_delta_acc"] for e in rows]
            # "Most boundary-near" = the interval whose furthest endpoint sits closest to
            # the margin. For an equivalent method that is the widest interval (closest
            # from inside); for MTF it is the narrowest (closest from outside).
            worst = min(rows, key=lambda e: abs(max(abs(e["ci_lo"]), abs(e["ci_hi"])) - 0.01))
            clf = clf_label[worst["name"].split("/")[0]]
            n01 = sum(e["equivalent_0.01"] for e in rows)
            n02 = sum(e["equivalent_0.02"] for e in rows)
            print(f"{label[key]:<4} & \\([{min(deltas):+.4f},{max(deltas):+.4f}]\\) & "
                  f"\\([{worst['ci_lo']:+.4f},{worst['ci_hi']:+.4f}]\\)~({clf}) & "
                  f"{n01}/{len(rows)} & {n02}/{len(rows)}" + r" \\")


if __name__ == "__main__":
    main()
