"""M7 benchmark: every representation and RP decoder on the seven datasets at 64, 128 and 256 samples.

For each dataset and window length the same seeded sample of windows goes
through the RP decoder x encoding grid (Hirata's classical reconstruction and our
Direct-2D decoder, each on the unsigned and on the signed RP), GASF, GADF, MTF and XIRP
(src/representations.py), each decoded from the float operator and from its
8-bit image. Per channel series it
stores dimensionless fidelity and spectral metrics plus encode/decode time:

  nmae, nrmse        |r - x| and RMSE over the series' own range
  pearson            correlation with the original
  dnmae              MAE of first differences over the range (local dynamics)
  psd_shape          L1 distance between Welch PSDs normalised to unit mass
  fdom_err           |dominant-frequency bin difference| as a fraction of Nyquist
  energy_err         |total power difference| / original power
  enc_ms, dec_ms     time to encode / decode this series (single thread)

Constant (flat) series are skipped and counted; a decoder exception is
recorded as a failure, never dropped. Peak memory per representation is
measured once per length with tracemalloc.

    python -m scripts._benchmark --lengths 64,128,256 --n 300 --workers 20
Writes results/_benchmark/<dataset>_L<length>.npz, summary.json and summary.md.
"""
import argparse
import json
import os
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

REPS = ("RP_HIRATA", "RP_HIRATA_SIGNED", "RP_UNSIGNED", "RP", "GASF", "GADF", "MTF", "XIRP",   # RP = Direct-2D, signed
        "RP_SIPERS", "RP_SIPERS_SIGNED")                       # added later: --only fills them into existing results
CONDS = ("float", "png8")
LABELS = {"RP_HIRATA": "Hirata · RP sin signo", "RP_HIRATA_SIGNED": "Hirata · Signed RP",
          "RP_UNSIGNED": "RP nueva · RP sin signo", "RP": "RP nueva · Signed RP",
          "GASF": "GASF", "GADF": "GADF", "MTF": "MTF", "XIRP": "XIRP",
          "RP_SIPERS": "Sipers · RP sin signo (circular)", "RP_SIPERS_SIGNED": "Sipers · Signed RP (circular)"}
METRICS = ("nmae", "nrmse", "pearson", "dnmae", "psd_shape", "fdom_err", "energy_err", "enc_ms", "dec_ms")
# Dataset-specific settings live here and nowhere else.
DATASETS = {
    "wisdm": {},
    "pamap2": {},
    "esc50": {"target_fs": 8000},          # 128 samples at 44.1 kHz would be 2.9 ms of sound
    "ptbxl": {"fs": 100, "max_records": 3000},
    "swarm": {},
    "stead": {"n_per_class": 1000},
    "emg_gestures": {},
}
OUT = Path("results/_benchmark")
MIN_WINDOWS = 200     # blocks with fewer windows stay in summary.json but out of the tables and statistics


def series_metrics(x: np.ndarray, r: np.ndarray) -> list[float]:
    from scipy.signal import welch

    rng = float(x.max() - x.min())
    d = r - x
    pearson = float(np.corrcoef(x, r)[0, 1]) if r.std() > 1e-12 else np.nan
    dn = float(np.mean(np.abs(np.diff(r) - np.diff(x)))) / rng
    nper = max(16, len(x) // 2)
    _, px = welch(x, nperseg=nper)
    _, pr = welch(r, nperseg=nper)
    shape = float(np.abs(px / (px.sum() + 1e-300) - pr / (pr.sum() + 1e-300)).sum())
    fdom = abs(int(np.argmax(px)) - int(np.argmax(pr))) / (len(px) - 1)
    energy = abs(float(px.sum() - pr.sum())) / (float(px.sum()) + 1e-300)
    return [float(np.mean(np.abs(d))) / rng, float(np.sqrt(np.mean(d ** 2))) / rng, pearson, dn, shape, fdom, energy]


def window_row(window: np.ndarray, reps: tuple = REPS) -> np.ndarray:
    """(T, C) window -> (reps, conds, C, metrics); NaN for flat channels, inf in nmae for failures."""
    from src.representations import REPRESENTATIONS

    T, C = window.shape
    out = np.full((len(reps), len(CONDS), C, len(METRICS)), np.nan)
    for c in range(C):
        x = window[:, c].astype(float)
        if x.max() - x.min() < 1e-12:
            continue
        mm = (float(x.max()), float(x.min()))
        for ri, name in enumerate(reps):
            rep = REPRESENTATIONS[name]
            try:
                t0 = time.perf_counter()
                op = rep.encode(x, mm)
                enc = time.perf_counter() - t0
                for ci, cond in enumerate(CONDS):
                    t0 = time.perf_counter()
                    img = None if cond == "float" else rep.to_uint8(op)
                    t_store = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    rec = np.asarray(rep.decode(op if img is None else rep.from_uint8(img), mm), dtype=float)
                    dec = time.perf_counter() - t0
                    if rec.shape != x.shape or not np.isfinite(rec).all():
                        raise ValueError(f"bad output {rec.shape}")
                    out[ri, ci, c] = series_metrics(x, rec) + [1000 * (enc + t_store), 1000 * dec]
            except Exception:
                out[ri, :, c, 0] = np.inf                       # failure marker, counted in the summary
    return out


def peak_memory_mb(name: str, x: np.ndarray) -> float:
    from src.representations import REPRESENTATIONS

    rep, mm = REPRESENTATIONS[name], (float(x.max()), float(x.min()))
    tracemalloc.start()
    rep.decode(rep.from_uint8(rep.to_uint8(rep.encode(x, mm))), mm)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 2 ** 20


def summarise(vals: np.ndarray, reps: tuple = REPS) -> dict:
    """vals: (n_windows, reps, conds, C, metrics) -> nested summary."""
    s = {}
    flat = np.moveaxis(vals, 3, 1).reshape(-1, len(reps), len(CONDS), len(METRICS))   # (series, reps, conds, m)
    for ri, rep in enumerate(reps):
        for ci, cond in enumerate(CONDS):
            v = flat[:, ri, ci]
            fail = np.isinf(v[:, 0])
            ok = np.isfinite(v[:, 0])
            entry = {"series": int(ok.sum()), "failures": int(fail.sum()),
                     "flat_skipped": int((~ok & ~fail).sum())}
            for mi, m in enumerate(METRICS):
                col = v[ok, mi]
                col = col[np.isfinite(col)]
                if col.size:
                    entry[m] = {"mean": float(col.mean()), "median": float(np.median(col)),
                                "p95": float(np.percentile(col, 95))}
            s[f"{rep}/{cond}"] = entry
    return s


def stored_reps(npz) -> tuple:
    """Representation order of a stored values array (files written before the names were stored hold the first 8)."""
    return tuple(npz["reps"]) if "reps" in npz.files else REPS[:npz["values"].shape[1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--lengths", default="64,128,256")
    ap.add_argument("--n", type=int, default=300, help="windows per dataset and length (seed 0)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--only", default="", help="compute only these representations, on the stored windows, and merge "
                                               "them into the existing results; the others are left untouched")
    args = ap.parse_args()

    from src.datasets import get, make_windows

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    n_windows = summary.get("_design", {}).get("n_windows", args.n) if args.only else args.n
    summary["_design"] = {"representations": REPS, "conditions": CONDS, "metrics": METRICS, "datasets": DATASETS,
                          "n_windows": n_windows, "seed": 0, "doc": __doc__.split("\n\n")[1].strip()}
    only = tuple(args.only.split(",")) if args.only else REPS
    for name in args.datasets.split(","):
        adapter = get(name, **DATASETS[name])
        for L in map(int, args.lengths.split(",")):
            t0 = time.perf_counter()
            w = make_windows(adapter, L, dtype=np.float32)
            key = f"{name}/L{L}"
            if args.only:
                old = np.load(OUT / f"{name}_L{L}.npz")
                idx = old["window_index"]
            else:
                idx = np.sort(np.random.default_rng(0).choice(len(w.X), size=min(args.n, len(w.X)), replace=False))
            with ProcessPoolExecutor(args.workers) as ex:
                new = np.stack(list(ex.map(partial(window_row, reps=only), [w.X[i] for i in idx], chunksize=2)))
            probe = next(w.X[i][:, c].astype(float) for i in idx for c in range(w.X.shape[2])
                         if np.ptp(w.X[i][:, c]) > 0)
            mem = {rep: peak_memory_mb(rep, probe) for rep in only}
            if args.only:
                have = stored_reps(old)
                vals = np.stack([new[:, only.index(r)] if r in only else old["values"][:, have.index(r)]
                                 for r in REPS], axis=1)
                mem = summary[key]["peak_memory_mb"] | mem
                wall = summary[key]["wall_s"] + time.perf_counter() - t0
            else:
                vals, wall = new, time.perf_counter() - t0
            np.savez_compressed(OUT / f"{name}_L{L}.npz", values=vals, reps=np.array(REPS), window_index=idx,
                                rec_id=w.rec_id[idx], start=w.start[idx], group=w.groups[idx], y=w.y[idx])
            summary[key] = {"fs": w.meta["fs"], "channels": w.meta["channels"], "duration_s": w.meta["duration_s"],
                            "windows_available": int(len(w.X)), "windows_used": int(len(idx)),
                            "peak_memory_mb": mem, "wall_s": wall, **summarise(vals)}
            summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
            line = "  ".join(f"{r} {summary[key][f'{r}/float']['nmae']['mean']:.1e}/"
                             f"{summary[key][f'{r}/png8']['nmae']['mean']:.4f}" for r in REPS
                             if 'nmae' in summary[key][f'{r}/png8'])
            print(f"{key:20} {len(idx)} win x {w.X.shape[2]} ch, {summary[key]['wall_s']:.0f} s | NMAE float/8-bit {line}",
                  flush=True)
    write_markdown(summary)


def write_markdown(summary: dict) -> None:
    rows = ["| Dataset | L | " + " | ".join(f"{LABELS[r]} 8-bit" for r in REPS) + " | " + " | ".join(f"{LABELS[r]} ms" for r in REPS) + " |",
            "|---|---|" + "---|" * (2 * len(REPS))]
    for key, s in summary.items():
        if key.startswith("_"):
            continue
        if s["windows_used"] < MIN_WINDOWS:
            continue
        ds, L = key.split("/")
        nm = [f"{s[f'{r}/png8']['nmae']['mean']:.4f}" if "nmae" in s.get(f"{r}/png8", {}) else "—" for r in REPS]
        ms = [f"{s[f'{r}/png8']['dec_ms']['mean']:.1f}" if "dec_ms" in s.get(f"{r}/png8", {}) else "—" for r in REPS]
        rows.append(f"| {ds} | {L[1:]} | " + " | ".join(nm) + " | " + " | ".join(ms) + " |")
    small = [k for k, s in summary.items() if not k.startswith("_") and s["windows_used"] < MIN_WINDOWS]
    note = f"Not reported (fewer than {MIN_WINDOWS} windows): {', '.join(small)}.\n\n" if small else ""
    (OUT / "summary.md").write_text("# M7 benchmark: mean NMAE (8-bit) and decode time per series\n\n" + note
                                    + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
