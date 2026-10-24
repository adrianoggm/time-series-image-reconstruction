"""Paired statistics for the M7 benchmark (results/_benchmark/<dataset>_L<length>.npz).

For every dataset and window length, compares RP with each other representation
on the same channel series (8-bit condition by default):

  * median paired difference in NMAE (other - RP; positive means RP is better)
    with a 95% cluster-bootstrap interval that resamples whole groups
    (subject, patient, event, day...) rather than windows;
  * share of series where RP is better;
  * Wilcoxon signed-rank p-value, Holm-corrected within each dataset x length family.

Then a macro average across datasets (each dataset weighs the same, so the
largest one cannot dominate) of the per-dataset median differences.

    python -m scripts._benchmark_stats [--metric nmae] [--cond png8]
Writes results/_benchmark/stats.json and stats.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from scripts._benchmark import CONDS, LABELS, METRICS, MIN_WINDOWS, REPS, stored_reps

OUT = Path("results/_benchmark")
B = 2000


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def cluster_ci(diff, groups, rng):
    """95% bootstrap interval of the median of diff, resampling groups with replacement."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    meds = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        meds[b] = np.median(diff[np.concatenate([members[g] for g in pick])])
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metric", default="nmae", choices=METRICS)
    ap.add_argument("--cond", default="png8", choices=CONDS)
    args = ap.parse_args()
    mi, ci = METRICS.index(args.metric), CONDS.index(args.cond)
    rng = np.random.default_rng(0)

    stats, per_dataset = {"_design": {"metric": args.metric, "cond": args.cond, "bootstrap": B, "seed": 0,
                                      "reference": "RP", "doc": __doc__.split("\n\n")[1].strip()}}, {}
    for f in sorted(OUT.glob("*_L*.npz")):
        z = np.load(f, allow_pickle=False)
        vals, groups = z["values"], z["group"]                    # (n, reps, conds, C, metrics), (n,)
        if len(vals) < MIN_WINDOWS:                               # EMG at L = 256: 20 windows, gestures last ~3 s
            stats.setdefault("_not_reported", []).append(f.stem)
            continue
        C = vals.shape[3]
        g_series = np.repeat(groups, C)
        key = f.stem.replace("_L", "/L")
        names = stored_reps(z)
        others = [r for r in names if r != "RP"]
        entry, pvals = {}, []
        for other in others:
            a = vals[:, names.index("RP"), ci, :, mi].reshape(-1)
            b = vals[:, names.index(other), ci, :, mi].reshape(-1)
            ok = np.isfinite(a) & np.isfinite(b)
            d = b[ok] - a[ok]
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            pvals.append(p)
            entry[other] = {"n_series": int(ok.sum()), "n_groups": int(len(np.unique(g_series[ok]))),
                            "median_diff": float(np.median(d)), "ci95": cluster_ci(d, g_series[ok], rng),
                            "rp_better_share": float(np.mean(d > 0)), "wilcoxon_p": p}
        for other, padj in zip(others, holm(pvals), strict=True):
            entry[other]["wilcoxon_p_holm"] = float(padj)
        stats[key] = entry
        per_dataset.setdefault(key.split("/")[1], {})[key.split("/")[0]] = {o: entry[o]["median_diff"] for o in others}

    others = sorted({o for ds in per_dataset.values() for v in ds.values() for o in v},
                    key=lambda o: REPS.index(o))
    stats["_macro"] = {L: {o: float(np.mean([v[o] for v in ds.values()])) for o in others} | {"datasets": len(ds)}
                       for L, ds in sorted(per_dataset.items(), key=lambda kv: int(kv[0][1:]))}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")

    rows = [f"# M7 paired statistics: {args.metric} ({args.cond}), other minus RP\n",
            "Positive = RP better. Interval: 95% cluster bootstrap over groups. p: Wilcoxon, Holm within dataset x length.\n",
            "| Dataset | L | " + " | ".join(LABELS[o] for o in others) + " |", "|---|---|" + "---|" * len(others)]
    for key, e in stats.items():
        if key.startswith("_"):
            continue
        ds, L = key.split("/")
        cells = [f"{e[o]['median_diff']:+.4f} [{e[o]['ci95'][0]:+.4f}, {e[o]['ci95'][1]:+.4f}] "
                 f"({100 * e[o]['rp_better_share']:.0f}%, p={e[o]['wilcoxon_p_holm']:.1g})" for o in others]
        rows.append(f"| {ds} | {L[1:]} | " + " | ".join(cells) + " |")
    rows += ["", "Macro average of per-dataset median differences (each dataset weighs the same):", "",
             "| L | " + " | ".join(LABELS[o] for o in others) + " | datasets |", "|---|" + "---|" * (len(others) + 1)]
    for L, m in stats["_macro"].items():
        rows.append(f"| {L[1:]} | " + " | ".join(f"{m[o]:+.4f}" for o in others) + f" | {m['datasets']} |")
    (OUT / "stats.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
