"""Compute pointwise + polynomial-trend reconstruction metrics for the four
encoders over ALL 8172 windows (the three LOTO test folds concatenated) and
emit LaTeX tables ready to drop into the paper.

Why pooled, not fold-0: the decoders are deterministic and untrained, so each
window's reconstruction is identical regardless of which fold it lands in (or
of the sampling protocol). The union of the three test folds is therefore the
full dataset, each window exactly once. Reporting over all 8172 windows is both
more complete than an arbitrary fold-0 subsample and consistent with the
paired significance tests (``_stat_tests.py``), which already pool the folds.

Outputs:
  - One consolidated pointwise table: 4 encoders x 4 metrics, each with
    mean +/- std, median [IQR], and subject-cluster bootstrap 95% CI.
  - Four per-axis supplementary tables (per-encoder MAE/MRE/MSE/Pearson).
  - One 4-column polynomial-trend summary (RP / GASF / GADF / MTF).

Run from inside the WSL venv:
  python scripts/_make_error_tables.py > .error_tables.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DATA = Path("/home/adrianogarcia/tfg/data/WISDM")
FOLDS = (0, 1, 2)
SAMPLING = "sampling_loto"          # pooled result is sampling-independent (see module docstring)
N_BOOT = 10_000
BOOT_SEED = 0

# Originals + per-window subject ids live under the recurrence_plot tree.
RP_TREE = DATA / "tseries/recurrence_plot" / SAMPLING / "3-fold"

ENCODERS = [
    ("RP",   "tseries/recurrence_plot", "X_all_rec.npy"),
    ("GASF", "tseries/GAF",             "X_all_gaf.npy"),
    ("GADF", "tseries/GADF",            "X_all_gadf.npy"),
    ("MTF",  "tseries/MTF",             "X_all_mtf.npy"),
]
METRIC_ORDER = ["MAE", "MRE", "MSE", "Pearson"]
METRIC_LABEL = {
    "MAE": "Mean Absolute Error",
    "MRE": "Mean Relative Error",
    "MSE": "Mean Squared Error",
    "Pearson": "Pearson Correlation Coefficient",
}
METRIC_SHORT = {"MAE": "MAE", "MRE": "MRE", "MSE": "MSE", "Pearson": "Pearson"}
# Per-fold table covers both 3-fold protocols; reconstruction is deterministic
# so the pooled column is protocol-independent (same 8172 windows repartitioned).
PERFOLD_SAMPLINGS = [("loto", "Window-level 3-fold"), ("loso", "Subject-out 3-fold")]


def _align(rec: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if rec.shape == target_shape:
        return rec
    if rec.shape == (target_shape[0], target_shape[2], target_shape[1]):
        return np.transpose(rec, (0, 2, 1))
    raise ValueError(f"cannot align rec of shape {rec.shape} to {target_shape}")


def _load_pooled():
    orig = np.concatenate(
        [np.load(RP_TREE / f"fold-{f}/test/validation_data.npy") for f in FOLDS], axis=0)
    subj = np.concatenate(
        [np.load(RP_TREE / f"fold-{f}/test/sj_validation_data.npy").ravel() for f in FOLDS], axis=0)
    recs = {}
    for enc, subdir, npy in ENCODERS:
        r = np.concatenate(
            [np.load(DATA / subdir / SAMPLING / "3-fold" / f"fold-{f}/test/{npy}") for f in FOLDS],
            axis=0)
        recs[enc] = _align(r, orig.shape)
    return orig, subj, recs


def per_window(orig: np.ndarray, rec: np.ndarray) -> dict:
    """Per-window arrays. Each metric -> (global (N,), per-axis (N,3)).
    Global is the per-window value averaged over the three axes, so its mean
    equals the historical 'Global Mean' and its dispersion is window-level."""
    diff = orig - rec
    abs_err = np.abs(diff)
    mae_ax = abs_err.mean(axis=1)
    mse_ax = (diff ** 2).mean(axis=1)
    safe = np.where(np.abs(orig) < 1.0, 1.0, np.abs(orig))
    mre_ax = (abs_err / safe).mean(axis=1)
    o_c = orig - orig.mean(axis=1, keepdims=True)
    r_c = rec - rec.mean(axis=1, keepdims=True)
    num = (o_c * r_c).sum(axis=1)
    den = np.linalg.norm(o_c, axis=1) * np.linalg.norm(r_c, axis=1)
    pear_ax = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    return {
        "MAE": (mae_ax.mean(axis=1), mae_ax),
        "MRE": (mre_ax.mean(axis=1), mre_ax),
        "MSE": (mse_ax.mean(axis=1), mse_ax),
        "Pearson": (pear_ax.mean(axis=1), pear_ax),
    }


def _summ(v):
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(mean=float(v.mean()), std=float(v.std()),
                median=float(med), q1=float(q1), q3=float(q3))


def _subject_ci(global_v, subj, rng):
    uniq = np.unique(subj)
    per_subj = np.array([global_v[subj == s].mean() for s in uniq])
    idx = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    boot = per_subj[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(per_subj.mean()), float(lo), float(hi)


def polynomial_trend(orig, rec):
    N, T, C = orig.shape
    t = np.arange(T)

    def fit(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-12:
            return np.zeros(4)
        return np.polyfit(t, (x - lo) / (hi - lo), deg=3)

    co = np.empty((N * C, 4)); cr = np.empty((N * C, 4)); k = 0
    for w in range(N):
        for c in range(C):
            co[k] = fit(orig[w, :, c]); cr[k] = fit(rec[w, :, c]); k += 1
    d = co - cr
    euclid = np.linalg.norm(d, axis=1)
    manh = np.abs(d).sum(axis=1)
    norms = np.linalg.norm(co, axis=1) * np.linalg.norm(cr, axis=1)
    cos = np.where(norms > 1e-12, (co * cr).sum(axis=1) / np.maximum(norms, 1e-12), 0.0)
    cov = np.cov(co.T)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    maha = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, inv, d), 0.0))
    return dict(Euclidean=float(euclid.mean()), Manhattan=float(manh.mean()),
                Cosine=float(cos.mean()), Mahalanobis=float(maha.mean()))


def perfold_global(sampling):
    """Per-fold global metric for one sampling protocol. Returns
    res[enc][metric] = {per_fold:[f0,f1,f2], cv_mean, cv_std, pooled}."""
    tree = DATA / "tseries/recurrence_plot" / f"sampling_{sampling}" / "3-fold"
    res = {}
    for enc, subdir, npy in ENCODERS:
        per = {m: [] for m in METRIC_ORDER}
        pooled_parts = {m: [] for m in METRIC_ORDER}
        for f in FOLDS:
            o = np.load(tree / f"fold-{f}/test/validation_data.npy")
            r = _align(np.load(DATA / subdir / f"sampling_{sampling}" / "3-fold" / f"fold-{f}/test/{npy}"), o.shape)
            pw = per_window(o, r)
            for m in METRIC_ORDER:
                gv = pw[m][0]
                per[m].append(float(gv.mean()))
                pooled_parts[m].append(gv)
        res[enc] = {m: {"per_fold": per[m],
                        "cv_mean": float(np.mean(per[m])), "cv_std": float(np.std(per[m])),
                        "pooled": float(np.concatenate(pooled_parts[m]).mean())}
                    for m in METRIC_ORDER}
    return res


# ---------- formatting ----------
def _dec(enc, metric):
    if metric == "Pearson":
        return 5
    return 2 if enc == "MTF" else 4


def _perfold_table(by_sampling):
    out = [
        r"\begin{table}[H]", r"\centering",
        (r"\caption{Per-fold reconstruction error for both 3-fold protocols "
         r"(physical units, m/s$^2$). Each cell is the per-window global metric "
         r"(averaged over the three axes) averaged over that fold's test windows. "
         r"Because the decoders are deterministic and untrained, a window "
         r"reconstructs identically in any fold; the two protocols only "
         r"repartition the same $8172$ windows, so the Pooled column is identical "
         r"for both and equals the full-dataset value of "
         r"Table~\ref{tab:pointwise_pooled}. The window-level folds are "
         r"near-identical stratified partitions (negligible fold-to-fold spread), "
         r"whereas the subject-out folds vary more because different subjects are "
         r"held out.}"),
        r"\label{tab:pointwise_perfold}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll ccc ccc c}", r"\toprule",
        (r" & & \multicolumn{3}{c}{\textbf{" + PERFOLD_SAMPLINGS[0][1] + r"}} & "
         r"\multicolumn{3}{c}{\textbf{" + PERFOLD_SAMPLINGS[1][1] + r"}} & \\"),
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        (r"\textbf{Encoder} & \textbf{Metric} & F0 & F1 & F2 & F0 & F1 & F2 & "
         r"\textbf{Pooled} \\"),
        r"\midrule",
    ]
    for ei, (enc, _, _) in enumerate(ENCODERS):
        for mi, m in enumerate(METRIC_ORDER):
            d = _dec(enc, m)
            cells = [fr"\textbf{{{enc}}}" if mi == 0 else "", METRIC_SHORT[m]]
            for s, _lbl in PERFOLD_SAMPLINGS:
                cells += [fr"${v:.{d}f}$" for v in by_sampling[s][enc][m]["per_fold"]]
            cells.append(fr"${by_sampling[PERFOLD_SAMPLINGS[0][0]][enc][m]['pooled']:.{d}f}$")
            out.append(" & ".join(cells) + r" \\")
        if ei < len(ENCODERS) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(out)


def _consolidated_table(stats, N, n_subj):
    out = [
        r"\begin{table}[H]", r"\centering",
        (r"\caption{Pointwise reconstruction errors over all $N=8172$ windows "
         r"(the three LOTO test folds pooled; physical units, m/s$^2$). Because "
         r"the decoders are deterministic and untrained, the pooled set is the "
         r"full dataset with each window scored once. Each metric is computed per "
         r"window (averaged over the three accelerometer axes); columns report the "
         r"across-window mean$\pm$std and median[IQR], and the subject-cluster "
         r"bootstrap 95\% CI of the grand per-subject mean ($51$ subjects, "
         r"$10^4$ resamples). Per-axis breakdowns are in "
         r"Appendix~\ref{app:per_axis_errors}.}"),
        r"\label{tab:pointwise_pooled}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccc}", r"\toprule",
        (r"\textbf{Encoder} & \textbf{Metric} & \textbf{Mean $\pm$ Std} & "
         r"\textbf{Median [IQR]} & \textbf{Per-subject mean [95\% CI]} \\"),
        r"\midrule",
    ]
    for ei, (enc, _, _) in enumerate(ENCODERS):
        for mi, m in enumerate(METRIC_ORDER):
            d = _dec(enc, m)
            s = stats[enc][m]
            enc_cell = fr"\textbf{{{enc}}}" if mi == 0 else ""
            mean = fr"${s['mean']:.{d}f} \pm {s['std']:.{d}f}$"
            med = fr"${s['median']:.{d}f}$ [{s['q1']:.{d}f}, {s['q3']:.{d}f}]"
            ci = fr"${s['subj_mean']:.{d}f}$ [{s['ci_lo']:.{d}f}, {s['ci_hi']:.{d}f}]"
            out.append(f"{enc_cell} & {METRIC_LABEL[m]} & {mean} & {med} & {ci} " + r"\\")
        if ei < len(ENCODERS) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(out)


def _per_axis_table(enc, stats_ax, label):
    out = [
        r"\begin{table}[H]", r"\centering",
        fr"\caption{{{enc} per-axis reconstruction errors over all $N=8172$ pooled "
        r"windows (physical units, m/s$^2$); mean$\pm$std across windows. The "
        r"Global column averages the three axes per window.}",
        fr"\label{{{label}}}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Metric} & \textbf{Dim 0} & \textbf{Dim 1} & \textbf{Dim 2} & \textbf{Global} \\",
        r"\midrule",
    ]
    for m in METRIC_ORDER:
        d = _dec(enc, m)
        cells = [METRIC_LABEL[m]]
        for c in range(3):
            mu, sd = stats_ax[m]["axis"][c]
            cells.append(fr"${mu:.{d}f} \pm {sd:.{d}f}$")
        gmu, gsd = stats_ax[m]["global"]
        cells.append(fr"${gmu:.{d}f} \pm {gsd:.{d}f}$")
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(out)


def _poly_table(poly):
    rows = []
    for metric, lbl in [("Euclidean", "Mean Euclidean Distance"),
                        ("Manhattan", "Mean Manhattan Distance"),
                        ("Cosine", "Mean Cosine Similarity"),
                        ("Mahalanobis", "Mean Mahalanobis Distance")]:
        cells = [lbl] + [f"{poly[e][metric]:.5f}" for e in ["RP", "GASF", "GADF", "MTF"]]
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[H]", r"\centering",
        r"\caption{Normalized polynomial-trend coefficient metrics over all "
        r"$N=8172$ pooled windows.}",
        r"\label{tab:normalized_metrics}", r"\resizebox{0.95\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Metric} & \textbf{RP} & \textbf{GASF} & \textbf{GADF} & \textbf{MTF} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])


def main():
    orig, subj, recs = _load_pooled()
    N = len(orig); n_subj = len(np.unique(subj))
    print(f"% pooled originals shape={orig.shape}, N={N}, subjects={n_subj}", file=sys.stderr)
    rng = np.random.default_rng(BOOT_SEED)

    stats = {}       # stats[enc][metric] = {mean,std,median,q1,q3,subj_mean,ci_lo,ci_hi}
    stats_ax = {}    # stats_ax[enc][metric] = {axis:[(mu,sd)x3], global:(mu,sd)}
    poly = {}
    for enc, _, _ in ENCODERS:
        pw = per_window(orig, recs[enc])
        stats[enc] = {}; stats_ax[enc] = {}
        for m, (gv, ax) in pw.items():
            s = _summ(gv)
            sm, lo, hi = _subject_ci(gv, subj, rng)
            s.update(subj_mean=sm, ci_lo=lo, ci_hi=hi)
            stats[enc][m] = s
            stats_ax[enc][m] = {
                "axis": [(float(ax[:, c].mean()), float(ax[:, c].std())) for c in range(3)],
                "global": (float(gv.mean()), float(gv.std())),
            }
        poly[enc] = polynomial_trend(orig, recs[enc])
        print(f"% {enc}: MAE mean={stats[enc]['MAE']['mean']:.4f} "
              f"median={stats[enc]['MAE']['median']:.4f} "
              f"subjCI=[{stats[enc]['MAE']['ci_lo']:.4f},{stats[enc]['MAE']['ci_hi']:.4f}]",
              file=sys.stderr)

    print("% ===== per-fold table (both protocols) =====")
    by_sampling = {s: perfold_global(s) for s, _ in PERFOLD_SAMPLINGS}
    print(_perfold_table(by_sampling)); print()

    print("% ===== consolidated pooled table =====")
    print(_consolidated_table(stats, N, n_subj)); print()
    print("% ===== per-axis supplementary tables =====")
    labels = {"RP": "tab:recurrence_plot", "MTF": "tab:markov_fields",
              "GASF": "tab:grammian_fields", "GADF": "tab:gadf_fields"}
    for enc in ["RP", "MTF", "GASF", "GADF"]:
        print(_per_axis_table(enc, stats_ax[enc], labels[enc])); print()
    print("% ===== polynomial-trend table =====")
    print(_poly_table(poly))


if __name__ == "__main__":
    main()
