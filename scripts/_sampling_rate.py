"""Sensitivity of every representation to the sampling rate (roadmap #87).

The same 3000 PTB-XL ECGs read at 500 Hz and resampled with an anti-aliased
polyphase filter to 250, 100 and 50 Hz, plus the dataset's native 100 Hz files
as a check of the resampling. At each rate and window length, the benchmark's
per-series metrics (scripts/_benchmark.py, float and 8-bit) on a seeded sample
of windows, and the conditioning sigma2/sigma1 of the RP states
(scripts/_rp_conditions.py): a higher rate makes consecutive samples closer, the
states more collinear and the signed-RP reconstruction harder.

    python -m scripts._sampling_rate --n 300 --workers 20
Writes results/_stats/sampling_rate.json and .md.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RATES = {"500": {"fs": 500}, "250": {"fs": 500, "resample_to": 250}, "100": {"fs": 500, "resample_to": 100},
         "50": {"fs": 500, "resample_to": 50}, "100_native": {"fs": 100}}
OUT = Path("results/_stats")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lengths", default="64,128,256")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    from scripts._benchmark import CONDS, LABELS, METRICS, REPS, summarise, window_row
    from scripts._rp_conditions import series_conditions
    from src.datasets import get, make_windows

    path = OUT / "sampling_rate.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["_design"] = {"rates": RATES, "windows": args.n, "seed": 0, "max_records": 3000,
                         "doc": __doc__.split("\n\n")[1].strip()}
    mi, ri, ci = METRICS.index("nmae"), REPS.index("RP"), CONDS.index("png8")
    for rate, cfg in RATES.items():
        adapter = get("ptbxl", max_records=3000, **cfg)
        for L in map(int, args.lengths.split(",")):
            key = f"{rate}Hz/L{L}"
            if key in report:
                continue
            w = make_windows(adapter, L, dtype=np.float32)
            idx = np.sort(np.random.default_rng(0).choice(len(w.X), size=min(args.n, len(w.X)), replace=False))
            with ProcessPoolExecutor(args.workers) as ex:
                vals = np.stack(list(ex.map(window_row, [w.X[i] for i in idx], chunksize=2)))
            col = np.array([series_conditions(w.X[i][:, c])[1] for i in idx for c in range(w.X.shape[2])])
            err = vals[:, ri, ci, :, mi].reshape(-1)
            report[key] = {"fs": float(adapter.spec.fs), "window_seconds": L / adapter.spec.fs,
                           "windows": int(len(idx)), "collinearity_median": float(np.nanmedian(col)),
                           "share_below_0.1": float(np.nanmean(col < 0.1)),
                           "rp_png8_nmae_above_0.01": float(np.mean(err[np.isfinite(err)] > 0.01)),
                           **summarise(vals)}
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=1, default=float) + "\n")
            r = report[key]
            print(f"{key:16} {r['window_seconds']:.2f} s | sigma2/sigma1 median {r['collinearity_median']:.3f}, "
                  f"<0.1 {r['share_below_0.1']:.2f} | 8-bit NMAE " + "  ".join(
                      f"{x} {r[f'{x}/png8']['nmae']['mean']:.4f}" for x in REPS if "nmae" in r[f"{x}/png8"]), flush=True)
    rows = ["# Sampling rate (PTB-XL, anti-aliased resampling from 500 Hz)", "",
            "Mean 8-bit NMAE; sigma2/sigma1: median conditioning of the RP states and share below 0.1.", "",
            "| Rate / L | Seconds | sigma2/sigma1 (median, share < 0.1) | " + " | ".join(LABELS[x] for x in REPS) + " |",
            "|---|---|---|" + "---|" * len(REPS)]
    for key, r in report.items():
        if key.startswith("_"):
            continue
        rows.append(f"| {key} | {r['window_seconds']:.2f} | {r['collinearity_median']:.3f}, {r['share_below_0.1']:.2f} | "
                    + " | ".join(f"{r[f'{x}/png8']['nmae']['mean']:.4f}" if "nmae" in r[f"{x}/png8"] else "—"
                                 for x in REPS) + " |")
    (OUT / "sampling_rate.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
