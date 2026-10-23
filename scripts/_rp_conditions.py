"""When is the signed RP decoder exact? Sign information and degenerate series (roadmap #122, #127).

On the same seeded windows as scripts/_benchmark.py (read from its .npz files),
every channel series gets:

  informative   share of state pairs i < j whose signed-RP entry carries direction.
                The sign is +1 iff the displacement s_i - s_j lies within 45 degrees
                of (1, 1); (i, j) and (j, i) then get opposite signs. Displacements
                within 45 degrees of the anti-diagonal get -1 both ways and tell the
                decoder nothing about orientation.
  collinearity  sigma_2 / sigma_1 of the centred adjacent states (x_i, x_{i+1}): near 0
                when the states lie on a line (slow series, e.g. Swarm at 1 Hz), where
                the 2-D embedding is ill-conditioned.
  range         max - min (flat and tiny-amplitude series).

and is joined with its 8-bit RP (Direct-2D, signed) NMAE from the benchmark to
see which conditions the series with NMAE > 0.01 fall into.

    python -m scripts._rp_conditions
Writes results/_stats/rp_conditions.json and .md.
"""
import json
from pathlib import Path

import numpy as np

COLLINEAR = 1e-2      # sigma_2 / sigma_1 below this: states on (almost) a line
TINY = 1e-6           # range below this: amplitude at the floor of float32 data
BAD = 0.01            # 8-bit NMAE above this counts as "not explained by quantization alone"
OUT = Path("results/_stats")
BINS = (0.0, 0.01, 0.03, 0.1, 0.3, np.inf)   # sigma2/sigma1 bands for the bad-series rate (descriptive, no tuned cut)


def series_conditions(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    s = np.stack([x[:-1], x[1:]], axis=1)
    d = s[:, None, :] - s[None, :, :]
    iu = np.triu_indices(len(s), 1)
    d0, d1 = d[..., 0][iu], d[..., 1][iu]
    norm = np.hypot(d0, d1)
    moved = norm > 0
    cos = (d0[moved] + d1[moved]) / (np.sqrt(2) * norm[moved])
    informative = float(np.mean(np.abs(cos) >= np.sqrt(2) / 2)) if moved.any() else np.nan
    sv = np.linalg.svd(s - s.mean(0), compute_uv=False)
    collinearity = float(sv[1] / sv[0]) if sv[0] > 0 else np.nan
    return informative, collinearity, float(np.ptp(x))


def main():
    from scipy.stats import spearmanr

    from scripts._benchmark import CONDS, DATASETS, METRICS, REPS
    from src.datasets import get, make_windows

    ri, ci, mi = REPS.index("RP"), CONDS.index("png8"), METRICS.index("nmae")
    report = {"_design": {"collinear_below": COLLINEAR, "tiny_range_below": TINY, "bad_nmae_above": BAD,
                          "doc": __doc__.split("\n\n")[1].strip()}}
    pooled = np.zeros((len(BINS) - 1, 2), int)                 # (band, [series, bad])
    for name, cfg in DATASETS.items():
        adapter = get(name, **cfg)
        for L in (64, 128, 256):
            npz = np.load(Path("results/_benchmark") / f"{name}_L{L}.npz")
            w = make_windows(adapter, L, dtype=np.float32)
            X = w.X[npz["window_index"]]
            nmae = npz["values"][:, ri, ci, :, mi]                     # (windows, channels)
            rows = [(*series_conditions(X[k][:, c]), nmae[k, c]) for k in range(len(X)) for c in range(X.shape[2])]
            inf, col, rng, err = map(np.array, zip(*rows, strict=True))
            flat = rng == 0
            tiny = (rng > 0) & (rng < TINY)
            collinear = ~flat & (col < COLLINEAR)
            bad = np.isfinite(err) & (err > BAD)
            ok = np.isfinite(err) & np.isfinite(inf)
            rho = spearmanr(inf[ok], err[ok]).statistic if ok.sum() > 2 else np.nan
            report[f"{name}/L{L}"] = {
                "series": int(len(rows)),
                "informative_mean": float(np.nanmean(inf)), "informative_p10": float(np.nanpercentile(inf, 10)),
                "flat": int(flat.sum()), "tiny_range": int(tiny.sum()), "collinear": int(collinear.sum()),
                "rp_png8_nmae_above": int(bad.sum()),
                "bad_and_collinear": int((bad & collinear).sum()), "bad_and_tiny": int((bad & tiny).sum()),
                "bad_unexplained": int((bad & ~collinear & ~tiny).sum()),
                "spearman_informative_vs_nmae": float(rho),
                "spearman_collinearity_vs_nmae": float(spearmanr(col[ok], err[ok]).statistic) if ok.sum() > 2 else np.nan,
                "collinearity_median_bad": float(np.nanmedian(col[bad])) if bad.any() else np.nan,
                "collinearity_median_ok": float(np.nanmedian(col[ok & ~bad]))}
            band = np.digitize(col, BINS[1:-1])
            counts = [[int((ok & (band == b)).sum()), int((bad & (band == b)).sum())] for b in range(len(BINS) - 1)]
            report[f"{name}/L{L}"]["bands_series_bad"] = counts
            if len(X) >= 200:                                        # same rule as the benchmark tables
                pooled += np.array(counts)
            r = report[f"{name}/L{L}"]
            print(f"{name:13} L{L:<4} {r['series']:5} series | informative {r['informative_mean']:.3f} "
                  f"(p10 {r['informative_p10']:.3f}) | flat {r['flat']} tiny {r['tiny_range']} collinear "
                  f"{r['collinear']} | NMAE>{BAD}: {r['rp_png8_nmae_above']} (collinear {r['bad_and_collinear']}, "
                  f"tiny {r['bad_and_tiny']}, other {r['bad_unexplained']}) | rho inf {rho:+.2f} col "
                  f"{r['spearman_collinearity_vs_nmae']:+.2f} | col median bad/ok {r['collinearity_median_bad']:.3f}/"
                  f"{r['collinearity_median_ok']:.3f}", flush=True)
    edges = [f"[{a:g}, {b:g})" for a, b in zip(BINS[:-1], BINS[1:], strict=True)]
    report["_pooled_bands"] = {e: {"series": int(n), "bad": int(k), "bad_rate": float(k / n) if n else np.nan}
                               for e, (n, k) in zip(edges, pooled, strict=True)}
    for e, v in report["_pooled_bands"].items():
        print(f"sigma2/sigma1 {e:14} {v['series']:6} series  {v['bad']:5} bad  rate {v['bad_rate']:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rp_conditions.json").write_text(json.dumps(report, indent=1) + "\n")
    rows = ["# Signed RP: sign information and degenerate series", "",
            f"Same windows as the benchmark. Collinear: sigma2/sigma1 < {COLLINEAR}; tiny: range < {TINY}; "
            f"bad: 8-bit RP NMAE > {BAD}. rho: Spearman of 8-bit NMAE with the informative share and with sigma2/sigma1.", "",
            "| Dataset / L | Series | Informative (mean / p10) | Flat | Tiny | Collinear | NMAE > 0.01 | of which collinear / tiny / other | rho informative | rho sigma2/sigma1 | sigma2/sigma1 median bad / ok |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in report.items():
        if k.startswith("_"):
            continue
        rows.append(f"| {k} | {r['series']} | {r['informative_mean']:.3f} / {r['informative_p10']:.3f} | {r['flat']} | "
                    f"{r['tiny_range']} | {r['collinear']} | {r['rp_png8_nmae_above']} | {r['bad_and_collinear']} / "
                    f"{r['bad_and_tiny']} / {r['bad_unexplained']} | {r['spearman_informative_vs_nmae']:+.2f} | "
                    f"{r['spearman_collinearity_vs_nmae']:+.2f} | {r['collinearity_median_bad']:.3f} / "
                    f"{r['collinearity_median_ok']:.3f} |")
    rows += ["", "## Share of series with 8-bit NMAE > 0.01 by sigma2/sigma1 (all datasets pooled)", "",
             "| sigma2/sigma1 | Series | NMAE > 0.01 | Rate |", "|---|---|---|---|"]
    rows += [f"| {e} | {v['series']} | {v['bad']} | {v['bad_rate']:.3f} |" for e, v in report["_pooled_bands"].items()]
    (OUT / "rp_conditions.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
