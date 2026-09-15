"""Statistical significance tests for the reconstruction-evaluation results.

Computes, for the three pooled folds at N=128:

  (1) Bootstrap 95% CI for accuracy of every (classifier, encoder) pair.
  (2) McNemar's test, Original vs each Reconstructed (24 tests).
  (3) McNemar's test, pairwise between encoders for each classifier (36 tests).
  (4) Wilcoxon signed-rank on per-window MAE between encoders (6 pairs).

All families of p-values are corrected with Holm--Bonferroni. The primary
equivalence result uses a joint subject-cluster bootstrap with a max-deviation
critical value, giving simultaneous 95% intervals over the 18 confirmatory
(six-classifier, three-near-original-reconstruction) comparisons within one
protocol. The six MTF negative-control comparisons form a separate family.

Output: a Markdown report under results/_stats/stat_report.md plus a
machine-readable summary at results/_stats/stat_report.json.

Run inside the WSL venv after the main pipeline has populated
``data/WISDM/tseries/.../X_all_*.npy`` and ``models/``.
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

from src.config import DEFAULT_DATA_DIR, REPO_ROOT
from src.models import model_path
from src.paths import windowed_npy, xall_path

OUT_DIR = REPO_ROOT / "results" / "_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Override via the env var TFG_STAT_SAMPLING ("loto" or "loso") so the script
# can run on either protocol without touching the source.
SAMPLING = os.environ.get("TFG_STAT_SAMPLING", "loto")
N_FOLDS = 3
FOLDS = (0, 1, 2)
# Primary equivalence margin: one accuracy percentage point. A difference this
# small does not alter the practical ranking and is well below the observed
# between-fold variation. Override only for an explicitly labelled sensitivity
# analysis.
EQUIV_MARGIN = float(os.environ.get("TFG_EQUIV_MARGIN", "0.01"))
# Backwards compatibility with code that imports FOLD as a scalar.
FOLD = 0

# (classifier-key, "flatten? True/False" → how the model expects X)
CLASSIFIERS = [
    ("knn",          True),
    ("svc",          True),
    ("randomforest", True),
    ("lstm",         False),
    ("gru",          False),
    ("cnn",          False),
]
ENCODERS = ["RP", "GAF", "GADF", "MTF"]   # filesystem keys; GAF == GASF


def _to_labels(y_cat):
    out = np.full(len(y_cat), -1)
    for i in range(len(y_cat)):
        for j in range(y_cat.shape[1]):
            if 0.9 <= y_cat[i, j] <= 1.1:
                out[i] = j
    return out


def _load_model(clf, fold=FOLD):
    p = model_path(clf, SAMPLING, fold, N_FOLDS)
    if not p.exists():
        raise FileNotFoundError(f"missing trained model: {p}. Run the pipeline first.")
    if p.suffix == ".h5":
        if clf == "cnn":
            from scripts.cnnvalidation import statistical_features_native
            return load_model(p, custom_objects={
                "statistical_features_native": statistical_features_native})
        return load_model(p)
    return joblib.load(p)


def _predict(model, X, flatten):
    if flatten:
        X = X.reshape(X.shape[0], -1)
    out = model.predict(X) if not hasattr(model, "predict_proba") else model.predict(X)
    if out.ndim == 2:                    # Keras softmax
        return out.argmax(axis=-1)
    return out.astype(int)


def _load_recon(stype, fold=FOLD):
    arr = np.load(xall_path(stype, SAMPLING, fold, N_FOLDS, "test", DEFAULT_DATA_DIR))
    if arr.ndim == 3 and arr.shape[1] in (1, 3) and arr.shape[1] != arr.shape[2]:
        arr = np.transpose(arr, (0, 2, 1))
    return arr


def _bootstrap_acc(correct: np.ndarray, n_boot=10_000, alpha=0.05, rng=None) -> tuple[float, float, float]:
    """Percentile bootstrap CI for accuracy (np.mean over a 0/1 array)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    N = len(correct)
    idx = rng.integers(0, N, size=(n_boot, N))
    means = correct[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(correct.mean()), float(lo), float(hi)


def _mcnemar(correct_a: np.ndarray, correct_b: np.ndarray):
    """Continuity-corrected McNemar test on paired 0/1 correctness vectors.

    The discordant pairs are compared with a chi-squared statistic carrying
    Yates' correction, evaluated through ``chi2.sf`` so the far tail keeps its
    precision. This is the test the manuscript reports; it is not a mid-p test.
    """
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "stat": 0.0}
    # Continuity-corrected chi-squared with df=1; use the survival function
    # so we keep numerical precision in the tail (p << 1e-16).
    stat = (abs(b - c) - 1) ** 2 / n
    p = stats.chi2.sf(stat, df=1)
    return {"b": b, "c": c, "p_value": float(p), "stat": float(stat)}


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Per-window paired difference. Returns p-value and effect size r = Z/sqrt(N).

    Z is computed analytically from the W statistic so it doesn't blow up to
    +inf when the asymptotic p-value underflows to 0.
    """
    diff = a - b
    nz_mask = diff != 0
    nz = diff[nz_mask]
    N = int(len(nz))
    if N == 0:
        return {"n_nonzero": 0, "p_value": 1.0, "effect_r": 0.0}
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                         correction=True, method="approx")
    # Analytic Z from W (Wilcoxon signed-rank, normal approximation with
    # continuity correction). W here is the sum of positive ranks.
    ranks = stats.rankdata(np.abs(nz))
    W_pos = float(ranks[nz > 0].sum())
    mu = N * (N + 1) / 4.0
    sigma = np.sqrt(N * (N + 1) * (2 * N + 1) / 24.0)
    cc = 0.5 if W_pos != mu else 0.0
    Z = (W_pos - mu - np.sign(W_pos - mu) * cc) / sigma
    return {"n_nonzero": N,
            "p_value": float(res.pvalue),
            "Z": float(Z),
            "effect_r": float(Z / np.sqrt(N))}


def _per_subject_acc(correct_vec: np.ndarray, subj: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    """Accuracy of one 0/1 correctness vector aggregated per subject -> (n_subjects,)."""
    return np.array([correct_vec[subj == s].mean() for s in uniq])


def _tost(diffs: np.ndarray, margin: float) -> dict:
    """Paired TOST equivalence test on per-subject accuracy differences
    (Original - reconstruction). Equivalence within +/-margin is declared when
    BOTH one-sided t-tests reject at alpha; p_tost = max(p_lower, p_upper).

    p_lower tests H0: mu <= -margin (reject => mu > -margin),
    p_upper tests H0: mu >=  margin (reject => mu <  margin)."""
    n = len(diffs)
    m = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        # No variance: difference is exactly m for every subject.
        equiv = abs(m) < margin
        return {"n": n, "mean_diff": m, "sd": 0.0, "p_lower": 0.0 if equiv else 1.0,
                "p_upper": 0.0 if equiv else 1.0, "p_tost": 0.0 if equiv else 1.0,
                "equivalent": bool(equiv), "margin": margin}
    se = sd / np.sqrt(n)
    df = n - 1
    t_lower = (m + margin) / se
    t_upper = (m - margin) / se
    p_lower = float(stats.t.sf(t_lower, df))   # P(T > t_lower)
    p_upper = float(stats.t.cdf(t_upper, df))  # P(T < t_upper)
    p_tost = max(p_lower, p_upper)
    return {"n": n, "mean_diff": m, "sd": sd, "p_lower": p_lower, "p_upper": p_upper,
            "p_tost": p_tost, "equivalent": bool(p_tost < 0.05), "margin": margin}


def _cluster_bootstrap_ci(diffs: np.ndarray, margins=(0.01, 0.02),
                          n_boot=10000, seed=0, alpha=0.05) -> dict:
    """Subject-cluster bootstrap 95% CI for the mean per-subject accuracy
    difference (Original - reconstruction). ``diffs`` holds one value per
    subject, so resampling the entries with replacement resamples whole
    subjects. Equivalence at margin m is declared when the whole CI lies inside
    [-m, +m] (the standard 'CI within the equivalence region' rule)."""
    rng = np.random.default_rng(seed)
    K = len(diffs)
    idx = rng.integers(0, K, size=(n_boot, K))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(diffs.mean()), "lo": float(lo), "hi": float(hi),
            "equivalent": {f"{m}": bool(lo >= -m and hi <= m) for m in margins}}


def _cluster_bootstrap_simultaneous(diffs_by_name: dict[str, np.ndarray],
                                    margins=(0.01, 0.02), n_boot=10_000,
                                    seed=0, family_alpha=0.05) -> dict[str, dict]:
    """Joint subject bootstrap with simultaneous max-deviation intervals.

    Every column is a comparison and every row is the same subject. A single
    subject-index resample is used across all columns, preserving dependence
    between classifiers and encodings. The 95th percentile of the largest
    absolute centred bootstrap deviation supplies one family-wise critical
    value for all intervals.
    """
    names = list(diffs_by_name)
    matrix = np.column_stack([np.asarray(diffs_by_name[n], dtype=float) for n in names])
    if len({len(diffs_by_name[n]) for n in names}) != 1:
        raise ValueError("all comparisons must contain the same ordered subjects")
    rng = np.random.default_rng(seed)
    n_subjects = matrix.shape[0]
    idx = rng.integers(0, n_subjects, size=(n_boot, n_subjects))
    boot_means = matrix[idx].mean(axis=1)
    means = matrix.mean(axis=0)
    critical = float(np.quantile(np.max(np.abs(boot_means - means), axis=1),
                                 1 - family_alpha))
    point_lo, point_hi = np.percentile(boot_means, [2.5, 97.5], axis=0)
    result = {}
    for j, name in enumerate(names):
        lo, hi = float(means[j] - critical), float(means[j] + critical)
        result[name] = {
            "mean": float(means[j]),
            "lo": lo, "hi": hi,
            "pointwise_lo": float(point_lo[j]), "pointwise_hi": float(point_hi[j]),
            "critical_max_deviation": critical,
            "family_size": len(names),
            "family_alpha": family_alpha,
            "equivalent": {f"{m}": bool(lo >= -m and hi <= m) for m in margins},
        }
    return result


def _holm(pvals: list[tuple[str, float]]) -> list[tuple[str, float, float, bool]]:
    """Holm-Bonferroni correction. Returns [(name, raw_p, adj_p, reject_at_0.05)]."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i][1])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        a = (m - rank) * pvals[idx][1]
        a = min(1.0, max(a, running))
        running = a
        adj[idx] = a
    return [(name, p, adj[i], adj[i] <= 0.05)
            for i, (name, p) in enumerate(pvals)]


def _fmt_p(p: float) -> str:
    """Format a p-value, using "<1e-300" when it underflowed."""
    if p < 1e-300:
        return "<1e-300"
    return f"{p:.3e}"


def main() -> None:
    print(f"Loading windows and labels for folds {FOLDS}...")
    # Concatenate fold outputs to compute paired window-level diagnostics.
    # Pairing is preserved, but windows from the same subject are dependent;
    # these pooled McNemar/Wilcoxon results are therefore secondary. The
    # subject-aggregated tests and joint subject-cluster bootstrap below supply
    # the primary inferential unit.
    X_val_parts, y_parts, recon_parts = [], [], {st: [] for st in ENCODERS}
    sj_parts: list[np.ndarray] = []
    per_fold_N: list[int] = []
    for f in FOLDS:
        Xv = np.load(windowed_npy("validation_data",   SAMPLING, f, N_FOLDS, DEFAULT_DATA_DIR))
        yc = np.load(windowed_npy("y_validation_data", SAMPLING, f, N_FOLDS, DEFAULT_DATA_DIR))
        sj = np.load(windowed_npy("sj_validation_data", SAMPLING, f, N_FOLDS, DEFAULT_DATA_DIR)).ravel()
        X_val_parts.append(Xv)
        y_parts.append(_to_labels(yc))
        sj_parts.append(sj)
        for st in ENCODERS:
            recon_parts[st].append(_load_recon(st, fold=f))
        per_fold_N.append(len(yc))
        print(f"  fold-{f}: {len(yc)} windows")

    X_val = np.concatenate(X_val_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    subj = np.concatenate(sj_parts, axis=0)          # subject id per pooled window (same order)
    recon = {st: np.concatenate(parts, axis=0) for st, parts in recon_parts.items()}
    N = len(y)
    print(f"  pooled test windows across folds: {N}  (per fold: {per_fold_N}); subjects: {len(np.unique(subj))}")

    # ---- 1) Predict every (classifier, source) combination once ----
    # Each fold uses its own trained model (models/<clf>__loto-fold-<f>-3f.<ext>);
    # we therefore predict fold-by-fold, then concatenate correctness vectors.
    print("\nLoading classifiers and predicting per fold...")
    correct = {}   # correct[clf][src] = int8 array of length N (pooled)
    for clf, flatten in CLASSIFIERS:
        per_src_parts: dict[str, list[np.ndarray]] = {"Original": []}
        for st in ENCODERS:
            per_src_parts[st] = []
        for fi, f in enumerate(FOLDS):
            m = _load_model(clf, fold=f)
            Xv = X_val_parts[fi]
            yf = y_parts[fi]
            for src, X in [("Original", Xv), *[(st, recon_parts[st][fi]) for st in ENCODERS]]:
                pred = _predict(m, X, flatten)
                per_src_parts[src].append((pred == yf).astype(np.int8))
        correct[clf] = {src: np.concatenate(parts, axis=0)
                        for src, parts in per_src_parts.items()}
        print(f"  {clf}: predicted on {len(correct[clf])} sources, pooled N={len(correct[clf]['Original'])}")

    # ---- 2) Mean accuracies per fold (instead of bootstrap CIs on pooled vector,
    #         which would understate variance) ----
    print("\nPer-fold accuracy table (acc = mean of fold accuracies, std = std across folds):")
    ci_table = {}   # reused as the "mean +/- std" carrier so downstream code stays
    for clf, _ in CLASSIFIERS:
        ci_table[clf] = {}
        # Per-fold accuracies for each source
        for src in ("Original", *ENCODERS):
            accs = []
            cursor = 0
            for nf in per_fold_N:
                slab = correct[clf][src][cursor:cursor + nf]
                accs.append(float(slab.mean()))
                cursor += nf
            arr = np.asarray(accs, dtype=float)
            ci_table[clf][src] = (float(arr.mean()), float(arr.std(ddof=0)),
                                  float(arr.min()), float(arr.max()))
    for clf, _ in CLASSIFIERS:
        for src in ("Original", *ENCODERS):
            m, s, lo, hi = ci_table[clf][src]
            print(f"  {clf:<12} {src:<10} acc={m:.3f} +/-{s:.3f}  [min={lo:.3f}, max={hi:.3f}]")

    # ---- 3) McNemar's: Original vs each Reconstructed ----
    print("\nMcNemar Original vs each Reconstructed (24 tests):")
    mcnemar_orig = []
    for clf, _ in CLASSIFIERS:
        for src in ENCODERS:
            r = _mcnemar(correct[clf]["Original"], correct[clf][src])
            mcnemar_orig.append((f"{clf}/Original-vs-{src}", r["p_value"]))
    mcnemar_orig_adj = _holm(mcnemar_orig)

    # ---- 4) McNemar's: pairwise between encoders ----
    print("McNemar pairwise between encoders (36 tests):")
    mcnemar_pair = []
    for clf, _ in CLASSIFIERS:
        for s1, s2 in combinations(ENCODERS, 2):
            r = _mcnemar(correct[clf][s1], correct[clf][s2])
            mcnemar_pair.append((f"{clf}/{s1}-vs-{s2}", r["p_value"]))
    mcnemar_pair_adj = _holm(mcnemar_pair)

    # ---- 5) Wilcoxon on per-window MAE, between encoders ----
    print("Wilcoxon on per-window MAE between encoders (6 pairs):")
    mae_perwindow = {
        st: np.abs(X_val - r).mean(axis=(1, 2)) for st, r in recon.items()
    }
    wilc_rows = []
    for s1, s2 in combinations(ENCODERS, 2):
        w = _wilcoxon(mae_perwindow[s1], mae_perwindow[s2])
        wilc_rows.append((f"MAE/{s1}-vs-{s2}", w["p_value"], w["effect_r"]))
    wilc_adj = _holm([(name, p) for name, p, _ in wilc_rows])

    # ---- 6) Subject-clustered classification tests (treat the N=51 subjects,
    #         not the windows, as the independent units -> addresses intra-subject
    #         dependence and the missing subject-clustered classifier analysis),
    #         plus a paired TOST equivalence test on per-subject Delta-accuracy
    #         (addresses the missing formal equivalence test). ----
    print("\nSubject-clustered classification (per-subject accuracy, 51 subjects)...")
    uniq = np.unique(subj)
    psa = {clf: {src: _per_subject_acc(correct[clf][src], subj, uniq)
                 for src in ("Original", *ENCODERS)}
           for clf, _ in CLASSIFIERS}

    # 6a) Subject-clustered Wilcoxon, Original vs each reconstruction (difference).
    subj_wilc = []
    for clf, _ in CLASSIFIERS:
        for src in ENCODERS:
            w = _wilcoxon(psa[clf]["Original"], psa[clf][src])
            subj_wilc.append((f"{clf}/Original-vs-{src}", w["p_value"], w.get("effect_r", 0.0)))
    subj_wilc_adj = _holm([(n, p) for n, p, _ in subj_wilc])

    # 6b) Paired TOST equivalence, Original vs each reconstruction, margin EQUIV_MARGIN.
    tost_rows = []
    for clf, _ in CLASSIFIERS:
        for src in ENCODERS:
            d = psa[clf]["Original"] - psa[clf][src]
            tost_rows.append((f"{clf}/Original-vs-{src}", _tost(d, EQUIV_MARGIN)))
    tost_adj = _holm([(n, t["p_tost"]) for n, t in tost_rows])
    print(f"  margin={EQUIV_MARGIN}; equivalent (Holm) for "
          f"{sum(1 for _, _, _, rj in tost_adj if rj)}/{len(tost_adj)} (classifier,encoder) pairs")
    # Per-encoder worst-case across classifiers (largest |per-subject mean Delta-acc|).
    for src in ENCODERS:
        worst = max(abs(t["mean_diff"]) for n, t in tost_rows if n.endswith(f"-vs-{src}"))
        allok = all(rj for (n, *_), (_, _, _, rj) in zip(tost_rows, tost_adj) if n.endswith(f"-vs-{src}"))
        print(f"    {src}: max|Delta-acc|={worst:.4f}  all-equivalent={allok}")

    # 6c/6d) Pairwise subject-clustered Wilcoxon + TOST between encoders (backs or
    #        refutes claims like "GADF is indistinguishable from GASF in classification").
    subj_wilc_pair, tost_pair = [], []
    for clf, _ in CLASSIFIERS:
        for s1, s2 in combinations(ENCODERS, 2):
            w = _wilcoxon(psa[clf][s1], psa[clf][s2])
            subj_wilc_pair.append((f"{clf}/{s1}-vs-{s2}", w["p_value"], w.get("effect_r", 0.0)))
            tost_pair.append((f"{clf}/{s1}-vs-{s2}", _tost(psa[clf][s1] - psa[clf][s2], EQUIV_MARGIN)))
    subj_wilc_pair_adj = _holm([(n, p) for n, p, _ in subj_wilc_pair])
    tost_pair_adj = _holm([(n, t["p_tost"]) for n, t in tost_pair])
    for pair in ("GAF-vs-GADF", "RP-vs-GAF", "RP-vs-GADF"):
        eq = sum(1 for (n, *_), (_, _, _, rj) in zip(tost_pair, tost_pair_adj)
                 if n.endswith(pair) and rj)
        diff = sum(1 for (n, *_), (_, _, _, rj) in zip(subj_wilc_pair, subj_wilc_pair_adj)
                   if n.endswith(pair) and rj)
        print(f"    pairwise {pair}: equivalent(TOST) {eq}/6 classifiers, "
              f"subject-clustered difference detected {diff}/6")

    # 6e) Joint subject-cluster bootstrap on per-subject Delta-accuracy. The
    #     confirmatory family contains 6 classifiers x 3 near-original methods =
    #     18 comparisons. The 6 MTF negative-control comparisons are a separate
    #     family. Simultaneous max-deviation intervals,
    #     not an envelope of unadjusted intervals, support the primary claim.
    print("\nSimultaneous subject-cluster bootstrap CI on per-subject Delta-accuracy "
          "(Original - recon):")
    boot_diffs = {}
    mtf_diffs = {}
    for clf, _ in CLASSIFIERS:
        for src in ENCODERS:
            d = psa[clf]["Original"] - psa[clf][src]
            target = mtf_diffs if src == "MTF" else boot_diffs
            target[f"{clf}/Original-vs-{src}"] = d
    boot_joint = _cluster_bootstrap_simultaneous(boot_diffs, margins=(0.01, 0.02))
    boot_mtf = _cluster_bootstrap_simultaneous(mtf_diffs, margins=(0.01, 0.02), seed=1)
    boot_rows = list(boot_joint.items()) + list(boot_mtf.items())
    for src in ENCODERS:
        rows = [ci for n, ci in boot_rows if n.endswith(f"-vs-{src}")]
        eq1 = sum(1 for ci in rows if ci["equivalent"]["0.01"])
        eq2 = sum(1 for ci in rows if ci["equivalent"]["0.02"])
        worst_hi = max(abs(ci["lo"]) if abs(ci["lo"]) > abs(ci["hi"]) else abs(ci["hi"]) for ci in rows)
        print(f"    {src}: equivalent at +/-0.01 {eq1}/6, at +/-0.02 {eq2}/6; "
              f"widest CI bound |{worst_hi:.4f}|")

    # ---- 7) Persist ----
    payload = {
        "n_test": int(N),
        "folds": list(FOLDS),
        "per_fold_n": per_fold_N,
        "perfold_accuracy_mean_std_min_max": {
            clf: {src: list(v) for src, v in d.items()} for clf, d in ci_table.items()
        },
        "mcnemar_orig_vs_recon_holm": [
            {"name": n, "p_raw": pr, "p_holm": pa, "reject_0.05": rj}
            for n, pr, pa, rj in mcnemar_orig_adj
        ],
        "mcnemar_pairwise_encoders_holm": [
            {"name": n, "p_raw": pr, "p_holm": pa, "reject_0.05": rj}
            for n, pr, pa, rj in mcnemar_pair_adj
        ],
        "wilcoxon_mae_perwindow_holm": [
            {"name": wilc_rows[i][0], "effect_r": wilc_rows[i][2],
             "p_raw": wilc_adj[i][1], "p_holm": wilc_adj[i][2],
             "reject_0.05": wilc_adj[i][3]}
            for i in range(len(wilc_rows))
        ],
        "n_subjects": int(len(uniq)),
        "subject_clustered_wilcoxon_accuracy_holm": [
            {"name": subj_wilc[i][0], "effect_r": subj_wilc[i][2],
             "p_raw": subj_wilc_adj[i][1], "p_holm": subj_wilc_adj[i][2],
             "reject_0.05": subj_wilc_adj[i][3]}
            for i in range(len(subj_wilc))
        ],
        "tost_equivalence_accuracy_holm": [
            {"name": tost_rows[i][0], "margin": EQUIV_MARGIN,
             "mean_delta_acc": tost_rows[i][1]["mean_diff"],
             "p_tost_raw": tost_adj[i][1], "p_tost_holm": tost_adj[i][2],
             "equivalent_0.05": tost_adj[i][3]}
            for i in range(len(tost_rows))
        ],
        "subject_clustered_wilcoxon_pairwise_holm": [
            {"name": subj_wilc_pair[i][0], "effect_r": subj_wilc_pair[i][2],
             "p_raw": subj_wilc_pair_adj[i][1], "p_holm": subj_wilc_pair_adj[i][2],
             "reject_0.05": subj_wilc_pair_adj[i][3]}
            for i in range(len(subj_wilc_pair))
        ],
        "tost_equivalence_pairwise_holm": [
            {"name": tost_pair[i][0], "margin": EQUIV_MARGIN,
             "mean_delta_acc": tost_pair[i][1]["mean_diff"],
             "p_tost_raw": tost_pair_adj[i][1], "p_tost_holm": tost_pair_adj[i][2],
             "equivalent_0.05": tost_pair_adj[i][3]}
            for i in range(len(tost_pair))
        ],
        "subject_cluster_bootstrap_simultaneous_ci_accuracy": [
            {"name": n, "mean_delta_acc": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
             "pointwise_ci_lo": ci["pointwise_lo"],
             "pointwise_ci_hi": ci["pointwise_hi"],
             "family_size": ci["family_size"],
             "critical_max_deviation": ci["critical_max_deviation"],
             "equivalent_0.01": ci["equivalent"]["0.01"],
             "equivalent_0.02": ci["equivalent"]["0.02"]}
            for n, ci in boot_rows
        ],
    }
    json_path = OUT_DIR / f"stat_report_{SAMPLING}.json"
    md_path   = OUT_DIR / f"stat_report_{SAMPLING}.md"
    # Also keep a sampling-less alias so existing readers/tooling still work.
    legacy_json = OUT_DIR / "stat_report.json"
    legacy_md   = OUT_DIR / "stat_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown view
    md = [f"# Statistical significance report ({SAMPLING.upper()} 3-fold pooled)", "",
          f"Window-level diagnostics use the pooled $N={N}$ test windows from",
          f"folds {list(FOLDS)} (per-fold sizes {per_fold_N}). Pairing is preserved,",
          "but windows from one subject are dependent; these tests are therefore",
          "secondary. Section 5 aggregates by subject and provides the primary",
          "equivalence inference. Raw p-values are corrected with Holm--Bonferroni",
          "inside each stated family at $\\alpha=0.05$.", ""]

    md.append("## 1. Per-fold accuracy summary (mean +/- std across 3 folds)")
    md.append("")
    md.append("| Classifier | Original | RP | MTF | GASF | GADF |")
    md.append("|---|---|---|---|---|---|")
    for clf, _ in CLASSIFIERS:
        row = [clf]
        for src in ("Original", "RP", "MTF", "GAF", "GADF"):
            m, s, lo, hi = ci_table[clf][src]
            row.append(f"{m:.3f}±{s:.3f} [{lo:.3f},{hi:.3f}]")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## 2. McNemar Original vs reconstruction (Holm)")
    md.append("| Comparison | p (raw) | p (Holm) | Reject H0? |")
    md.append("|---|---|---|---|")
    for name, pr, pa, rj in mcnemar_orig_adj:
        md.append(f"| {name} | {_fmt_p(pr)} | {_fmt_p(pa)} | {'yes' if rj else 'no'} |")
    md.append("")

    md.append("## 3. McNemar pairwise between encoders (Holm)")
    md.append("| Comparison | p (raw) | p (Holm) | Reject H0? |")
    md.append("|---|---|---|---|")
    for name, pr, pa, rj in mcnemar_pair_adj:
        md.append(f"| {name} | {_fmt_p(pr)} | {_fmt_p(pa)} | {'yes' if rj else 'no'} |")
    md.append("")

    md.append("## 4. Wilcoxon signed-rank on per-window MAE")
    md.append("| Comparison | effect r | p (raw) | p (Holm) | Reject H0? |")
    md.append("|---|---|---|---|---|")
    for i, (name, _p, rfx) in enumerate(wilc_rows):
        _, pr, pa, rj = wilc_adj[i]
        md.append(f"| {name} | {rfx:+.3f} | {_fmt_p(pr)} | {_fmt_p(pa)} | {'yes' if rj else 'no'} |")
    md.append("")

    md.append(f"## 5. Subject-clustered classification ({len(uniq)} subjects as units)")
    md.append("")
    md.append("### 5a. Subject-clustered Wilcoxon on per-subject accuracy, Original vs reconstruction")
    md.append("| Comparison | effect r | p (raw) | p (Holm) | Reject H0 (difference)? |")
    md.append("|---|---|---|---|---|")
    for i, (name, _p, rfx) in enumerate(subj_wilc):
        _, pr, pa, rj = subj_wilc_adj[i]
        md.append(f"| {name} | {rfx:+.3f} | {_fmt_p(pr)} | {_fmt_p(pa)} | {'yes' if rj else 'no'} |")
    md.append("")
    md.append(f"### 5b. TOST equivalence on per-subject $\\Delta$accuracy (margin $\\pm{EQUIV_MARGIN}$)")
    md.append("Equivalence within the margin is declared when the Holm-adjusted TOST p < 0.05.")
    md.append("| Comparison | mean $\\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |")
    md.append("|---|---|---|---|---|")
    for i, (name, t) in enumerate(tost_rows):
        _, pr, pa, rj = tost_adj[i]
        md.append(f"| {name} | {t['mean_diff']:+.4f} | {_fmt_p(pr)} | {_fmt_p(pa)} | {'yes' if rj else 'no'} |")
    md.append("")
    md.append("### 5c. Pairwise subject-clustered Wilcoxon on per-subject accuracy (between encoders)")
    md.append("| Comparison | effect r | p (raw) | p (Holm) | Reject H0 (difference)? |")
    md.append("|---|---|---|---|---|")
    for i, (name, _p, rfx) in enumerate(subj_wilc_pair):
        _, pr, pa, rj = subj_wilc_pair_adj[i]
        md.append(f"| {name} | {rfx:+.3f} | {_fmt_p(pr)} | {_fmt_p(pa)} | {'yes' if rj else 'no'} |")
    md.append("")
    md.append(f"### 5d. Pairwise TOST equivalence between encoders (margin $\\pm{EQUIV_MARGIN}$)")
    md.append("| Comparison | mean $\\Delta$acc | p TOST (raw) | p TOST (Holm) | Equivalent? |")
    md.append("|---|---|---|---|---|")
    for i, (name, t) in enumerate(tost_pair):
        _, pr, pa, rj = tost_pair_adj[i]
        md.append(f"| {name} | {t['mean_diff']:+.4f} | {_fmt_p(pr)} | {_fmt_p(pa)} | {'yes' if rj else 'no'} |")
    md.append("")
    md.append("### 5e. Simultaneous subject-cluster bootstrap 95% intervals on per-subject $\\Delta$accuracy (Original - reconstruction)")
    md.append("The primary family contains 18 comparisons (six classifiers by RP/GASF/GADF); "
              "the six MTF negative-control comparisons form a separate family. Within each "
              "family the same subject resamples are used jointly and a "
              "max-deviation critical value controls family-wise coverage. Equivalence is "
              "declared when the whole simultaneous interval lies inside the margin. "
              "Primary margin $\\pm0.01$; the less stringent $\\pm0.02$ margin is a sensitivity analysis.")
    md.append("| Comparison | mean $\\Delta$acc | simultaneous 95% interval | Equiv. $\\pm$0.01 | Equiv. $\\pm$0.02 |")
    md.append("|---|---|---|---|---|")
    for name, ci in boot_rows:
        md.append(f"| {name} | {ci['mean']:+.4f} | [{ci['lo']:+.4f}, {ci['hi']:+.4f}] | "
                  f"{'yes' if ci['equivalent']['0.01'] else 'no'} | "
                  f"{'yes' if ci['equivalent']['0.02'] else 'no'} |")
    md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")
    # Keep the legacy unsuffixed files mirrored to the latest run for any reader
    # that hard-codes the old path. Newer code should read the sampling-suffixed
    # variants.
    legacy_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    legacy_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nReport written to {md_path} (sampling={SAMPLING})")


if __name__ == "__main__":
    main()
