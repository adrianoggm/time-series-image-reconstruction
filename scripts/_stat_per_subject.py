#!/usr/bin/env python3
"""Subject-clustered pointwise test: aggregate per-window MAE to one value per
subject, then run paired Wilcoxon across subjects (N = #subjects, not #windows).

This is the cluster-aware companion to the window-level Wilcoxon in
_stat_tests.py: it treats each subject as the independent unit, so the p-values
cannot be inflated by intra-subject pseudoreplication.

    python -m scripts._stat_per_subject --sampling loto
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.config import DEFAULT_DATA_DIR

STATS_DIR = Path(__file__).resolve().parents[1] / "results" / "_stats"


def _holm(pairs):
    """Holm-Bonferroni over [(name, p), ...]; returns {name: adjusted p}."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m, running, adj = len(pairs), 0.0, {}
    for rank, idx in enumerate(order):
        a = min(1.0, max((m - rank) * pairs[idx][1], running))
        running = a
        adj[pairs[idx][0]] = a
    return adj

ENCS = {"GASF": ("GAF", "X_all_gaf.npy"), "GADF": ("GADF", "X_all_gadf.npy"),
        "RP": ("recurrence_plot", "X_all_rec.npy"), "MTF": ("MTF", "X_all_mtf.npy")}


def _align(rec, target):
    return rec if rec.shape == target else np.transpose(rec, (0, 2, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling", default="loto")
    ap.add_argument("--folds", default="0,1,2")
    args = ap.parse_args()
    folds = [int(f) for f in args.folds.split(",")]

    # per-window MAE and subject id, pooled across folds
    subj_all, mae_all = [], {e: [] for e in ENCS}
    for fold in folds:
        base = (f"{DEFAULT_DATA_DIR}tseries/recurrence_plot/sampling_{args.sampling}/"
                f"3-fold/fold-{fold}/test/")
        orig = np.load(base + "validation_data.npy")
        sj = np.load(base + "sj_validation_data.npy").reshape(-1)
        subj_all.append(sj)
        for e, (sub, npy) in ENCS.items():
            rec = _align(np.load(f"{DEFAULT_DATA_DIR}tseries/{sub}/sampling_{args.sampling}/"
                                 f"3-fold/fold-{fold}/test/{npy}"), orig.shape)
            mae_all[e].append(np.mean(np.abs(orig - rec), axis=(1, 2)))  # per-window MAE
    subj = np.concatenate(subj_all)
    mae = {e: np.concatenate(v) for e, v in mae_all.items()}

    subjects = np.unique(subj)
    # per-subject mean MAE
    per_subj = {e: np.array([mae[e][subj == s].mean() for s in subjects]) for e in ENCS}
    print(f"sampling={args.sampling}  subjects={len(subjects)}  windows={len(subj)}\n")
    print(f"{'encoder':6} {'per-subject mean MAE':>22}")
    for e in ENCS:
        print(f"{e:6} {per_subj[e].mean():10.4f}  (median {np.median(per_subj[e]):.4f})")

    print(f"\npaired Wilcoxon across {len(subjects)} subjects (per-subject mean MAE):")
    raw = []
    for a, b in itertools.combinations(ENCS, 2):
        da, db = per_subj[a], per_subj[b]
        try:
            _, p = wilcoxon(da, db)
        except ValueError:
            p = float("nan")
        raw.append((f"{a}-vs-{b}", float(p)))
    adj = _holm(raw)

    print(f"{'pair':16} {'p (raw)':>12} {'p (Holm)':>12} {'better':>8}")
    tests = {}
    for name, p in raw:
        a, b = name.split("-vs-")
        better = a if per_subj[a].mean() < per_subj[b].mean() else b
        tests[name] = {"p_raw": p, "p_holm": adj[name], "better": better,
                       "reject_0.05": bool(adj[name] <= 0.05)}
        print(f"{name:16} {p:12.3e} {adj[name]:12.3e} {better:>8}")

    # Persist: the manuscript quotes these means and the RP-vs-GADF p-value, so
    # they need an artefact rather than a console transcript.
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": {"sampling": args.sampling, "folds": folds,
                   "n_subjects": int(len(subjects)), "n_windows": int(len(subj)),
                   "unit": "per-subject mean of the per-window MAE (m/s^2)",
                   "family": "6 pairwise encoder comparisons, Holm-corrected"},
        "per_subject_mean_mae": {e: {"mean": float(per_subj[e].mean()),
                                     "median": float(np.median(per_subj[e])),
                                     "std": float(per_subj[e].std(ddof=0))}
                                 for e in ENCS},
        "wilcoxon_pairwise_holm": tests,
    }
    out = STATS_DIR / f"per_subject_mae_{args.sampling}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
