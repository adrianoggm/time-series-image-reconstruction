"""RP comparison matrix (docs/experiments/rp_signed_comparison.md), per dataset and storage condition.

Methods, all on the same windows and channels:
  hirata          Hirata RPR on the conventional binary RP (rate 0.75), orientation as returned
  hirata_oracle   the same with evaluation-only oracle orientation (a bound, not a result)
  legacy_signed   our previous decoder (src/rp_legacy.py) on the signed RP
  direct_unsigned Direct-2D on |R| only (no sign available)
  direct_unsigned_oracle  the same with oracle orientation
  direct_signed   Direct-2D on the signed RP (production decoder)
  geo_signed      GEO-2D (continuous-weight geodesics, rho 0.60) on the signed RP

Conditions: "float" hands each decoder the exact matrix; "png8" first stores it as
an 8-bit image and decodes metadata-free — the signed RP with the production
normalisation (NormalizeMatrix) and diagonal centring, the unsigned RP as
|R| / max|R|. Every series is rescaled to the channel's own min-max, the
calibration all methods share.

    python -m scripts._rp_matrix --datasets wisdm,pamap2 --n 200 --workers 12
Writes results/_stats/rp_matrix.json (summary) and rp_matrix_series.npz (per series, for paired tests).
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

METHODS = ("hirata", "hirata_oracle", "legacy_signed", "direct_unsigned", "direct_unsigned_oracle",
           "direct_signed", "geo_signed")
CONDITIONS = ("float", "png8")
METRICS = ("nmae", "nrmse", "pearson", "orientation_ok", "seconds")


def _metrics(rec, x):
    rng = x.max() - x.min()
    refl = x.max() + x.min() - rec                      # the same reconstruction, reflected
    err, err_refl = np.abs(rec - x).mean(), np.abs(refl - x).mean()
    pearson = np.corrcoef(rec, x)[0, 1] if rec.std() > 1e-12 else 0.0
    return [err / rng, np.sqrt(np.mean((rec - x) ** 2)) / rng, pearson, float(err <= err_refl)]


def _oracle(rec, x):
    refl = x.max() + x.min() - rec
    return refl if np.abs(refl - x).mean() < np.abs(rec - x).mean() else rec


def series_row(x):
    """All methods x conditions for one channel series -> array (conditions, methods, metrics)."""
    from src.recurrence_plots import NormalizeMatrix, varRP2
    from src.rp_decoders import decode_channel
    from src.rp_hirata import binary_rp, eps_for_rate, states, to_series
    from src.rp_legacy import decode_channel as legacy

    T, mm = len(x), (x.max(), x.min())
    R = varRP2(x, TIME_STEPS=T)
    out = np.full((len(CONDITIONS), len(METHODS), len(METRICS)), np.nan)
    if x.max() - x.min() < 1e-12:
        return out                                        # flat channel: nothing to reconstruct, left as NaN
    for ci, cond in enumerate(CONDITIONS):
        if cond == "float":
            R_s, D_u = R, np.abs(R)
        else:
            png_s = np.round(NormalizeMatrix(R) * 255)
            R_s = png_s - np.median(np.diag(png_s))
            D_u = np.round(np.abs(R) / np.abs(R).max() * 255)
        recs = {}
        t0 = time.perf_counter()
        Y, _ = states(binary_rp(D_u, eps_for_rate(D_u, 0.75)))
        recs["hirata"] = (to_series(Y, mm), time.perf_counter() - t0)
        recs["hirata_oracle"] = (_oracle(recs["hirata"][0], x), recs["hirata"][1])
        t0 = time.perf_counter()
        recs["legacy_signed"] = (legacy(R_s, mm, scale_invariant=True, n_out=T)[0], time.perf_counter() - t0)
        t0 = time.perf_counter()
        du = decode_channel(D_u, mm, source="direct", signed=False)
        recs["direct_unsigned"] = (du, time.perf_counter() - t0)
        recs["direct_unsigned_oracle"] = (_oracle(du, x), recs["direct_unsigned"][1])
        t0 = time.perf_counter()
        recs["direct_signed"] = (decode_channel(R_s, mm, source="direct", signed=True, scale_invariant=True),
                                 time.perf_counter() - t0)
        t0 = time.perf_counter()
        recs["geo_signed"] = (decode_channel(R_s, mm, source="geo", rho=0.60, signed=True, scale_invariant=True),
                              time.perf_counter() - t0)
        for mi, m in enumerate(METHODS):
            rec, secs = recs[m]
            out[ci, mi] = _metrics(np.asarray(rec, dtype=float)[:T], x) + [secs]
    return out


def main():
    from scripts._benchmark import DATASETS

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n", type=int, default=300, help="windows per dataset and length, sampled with seed 0")
    ap.add_argument("--lengths", default="64,128,256")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    from src.datasets import get, make_windows

    summary = {"methods": METHODS, "conditions": CONDITIONS, "metrics": METRICS, "n_windows": args.n,
               "lengths": args.lengths, "datasets": DATASETS, "doc": "docs/experiments/rp_signed_comparison.md"}
    arrays = {}
    for ds in args.datasets.split(","):
        adapter = get(ds, **DATASETS[ds])
        for L in map(int, args.lengths.split(",")):
            name = f"{ds}/L{L}"
            w = make_windows(adapter, L, dtype=np.float32)
            idx = np.sort(np.random.default_rng(0).choice(len(w.X), size=min(args.n, len(w.X)), replace=False))
            series = [w.X[i, :, c].astype(float) for i in idx for c in range(w.X.shape[2])]
            t0 = time.perf_counter()
            with ProcessPoolExecutor(args.workers) as ex:
                res = np.stack(list(ex.map(series_row, series, chunksize=4)))  # (S, cond, method, metric)
            arrays[f"{name}/values"] = res
            arrays[f"{name}/window"] = np.repeat(idx, w.X.shape[2])
            arrays[f"{name}/channel"] = np.tile(np.arange(w.X.shape[2]), len(idx))
            arrays[f"{name}/group"] = np.repeat(w.groups[idx], w.X.shape[2])
            summary[name] = {"series": len(series), "flat_series": int(np.isnan(res[:, 0, 0, 0]).sum()),
                             "wall_seconds": time.perf_counter() - t0}
            _report(summary, name, res)
    out = Path("results/_stats")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rp_matrix.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    np.savez_compressed(out / "rp_matrix_series.npz", **arrays)
    print(f"\nwrote {out / 'rp_matrix.json'} and rp_matrix_series.npz")


def _report(summary, name, res):
    """Summarise (series, conditions, methods, metrics) into summary[name] and print it."""
    for ci, cond in enumerate(CONDITIONS):
        for mi, m in enumerate(METHODS):
            v = res[:, ci, mi]
            summary[name][f"{cond}/{m}"] = {
                "nmae_mean": np.nanmean(v[:, 0]), "nmae_median": np.nanmedian(v[:, 0]),
                "nmae_p95": np.nanpercentile(v[:, 0], 95), "nrmse_mean": np.nanmean(v[:, 1]),
                "pearson_mean": np.nanmean(v[:, 2]), "orientation_ok": np.nanmean(v[:, 3]),
                "ms_per_series": 1000 * np.nanmean(v[:, 4])}
    print(f"\n{name}: {summary[name]['series']} channel series, {summary[name]['wall_seconds']:.0f} s", flush=True)
    print(f"{'condition/method':32} {'NMAE mean':>10} {'median':>9} {'p95':>8} {'Pearson':>8} {'orient':>7} {'ms':>7}")
    for cond in CONDITIONS:
        for m in METHODS:
            s = summary[name][f"{cond}/{m}"]
            print(f"{cond + '/' + m:32} {s['nmae_mean']:10.4f} {s['nmae_median']:9.4f} {s['nmae_p95']:8.4f} "
                  f"{s['pearson_mean']:8.3f} {s['orientation_ok']:7.3f} {s['ms_per_series']:7.1f}", flush=True)


if __name__ == "__main__":
    main()
