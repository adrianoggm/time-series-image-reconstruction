"""Robustness of every representation to noise on the stored image, on the seven datasets (roadmap #128).

The protocol of the WISDM study (doc/main3.tex, robustness): each channel's 8-bit
image gets zero-mean Gaussian noise of sigma in {2, 4, 8, 16} intensity levels on
every pixel, is clipped to [0, 255] and rounded, and is decoded; two independent
replicates per level, averaged within the series. sigma = 0 is the clean 8-bit
round trip. Windows: a seeded subset of the benchmark windows
(results/_benchmark/<dataset>_L<length>.npz). Error: range-normalised MAE (NMAE)
against the original series; per (dataset, length, technique, sigma) the mean
and a 95% bootstrap interval resampling whole groups. A local-stability stress
test, not a model of generated-image defects.

    python -m scripts._robustness_multi --n 200 --workers 20
Writes results/_stats/robustness_multi.json and .md.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIGMAS = (0.0, 2.0, 4.0, 8.0, 16.0)
REPLICATES = 2
B = 2000
OUT = Path("results/_stats")


def window_noise(args) -> np.ndarray:
    """(window (T, C), seed) -> NMAE (techniques, C, sigmas); NaN for flat channels and failures."""
    from threadpoolctl import threadpool_limits

    from scripts._benchmark import REPS
    from src.representations import REPRESENTATIONS

    window, seed, reps = (*args, REPS) if len(args) == 2 else args
    rng = np.random.default_rng(seed)
    T, C = window.shape
    out = np.full((len(reps), C, len(SIGMAS)), np.nan)
    with threadpool_limits(1):
        for c in range(C):
            x = window[:, c].astype(float)
            span = x.max() - x.min()
            if span < 1e-12:
                continue
            mm = (float(x.max()), float(x.min()))
            for k, name in enumerate(reps):
                rep = REPRESENTATIONS[name]
                try:
                    img = rep.to_uint8(rep.encode(x, mm)).astype(float)
                    for s, sigma in enumerate(SIGMAS):
                        errs = []
                        for _ in range(1 if sigma == 0 else REPLICATES):
                            noisy = img if sigma == 0 else np.clip(np.round(img + rng.normal(0, sigma, img.shape)), 0, 255)
                            r = rep.decode(rep.from_uint8(noisy.astype(np.uint8)), mm)
                            errs.append(np.mean(np.abs(np.asarray(r, float) - x)) / span)
                        out[k, c, s] = np.mean(errs)
                except Exception:
                    out[k, c, :] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--lengths", default="64,128,256")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--only", default="", help="add only these techniques to the existing blocks; the noise of "
                                               "the added ones comes from the same per-window seeds")
    args = ap.parse_args()

    from scripts._benchmark import DATASETS, LABELS, MIN_WINDOWS, REPS
    from src.datasets import get, make_windows

    path = OUT / "robustness_multi.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["_design"] = {"sigmas": SIGMAS, "replicates": REPLICATES, "windows": args.n, "bootstrap": B, "seed": 0,
                         "techniques": REPS, "doc": __doc__.split("\n\n")[1].strip()}
    rng = np.random.default_rng(0)
    only = tuple(args.only.split(",")) if args.only else REPS
    for name, cfg in DATASETS.items():
        adapter = get(name, **cfg)
        for L in map(int, args.lengths.split(",")):
            key = f"{name}/L{L}"
            if key in report and all(r in report[key] for r in only):
                continue
            npz = np.load(Path("results/_benchmark") / f"{name}_L{L}.npz")
            if len(npz["window_index"]) < MIN_WINDOWS:
                continue
            pick = np.sort(np.random.default_rng(0).choice(len(npz["window_index"]), size=args.n, replace=False))
            w = make_windows(adapter, L, dtype=np.float32)
            idx = npz["window_index"][pick]
            with ProcessPoolExecutor(args.workers) as ex:
                vals = np.stack(list(ex.map(window_noise, [(w.X[i], int(i), only) for i in idx], chunksize=2)))
            groups = np.repeat(w.groups[idx], vals.shape[2])          # one group label per channel series
            entry = report.get(key, {})
            for k, rep in enumerate(only):
                e = vals[:, k].reshape(-1, len(SIGMAS))               # (series, sigmas)
                ok = np.isfinite(e).all(axis=1)
                g = groups[ok]
                uniq, inv = np.unique(g, return_inverse=True)
                members = [np.flatnonzero(inv == j) for j in range(len(uniq))]
                curve = {}
                for s, sigma in enumerate(SIGMAS):
                    v = e[ok, s]
                    boots = [v[np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])].mean()
                             for _ in range(B)]
                    curve[str(sigma)] = {"mean": float(v.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                                           float(np.percentile(boots, 97.5))]}
                slope = float(np.polyfit(SIGMAS, [curve[str(s)]["mean"] for s in SIGMAS], 1)[0])
                entry[rep] = {"series": int(ok.sum()), "failed": int((~ok).sum()), "nmae": curve,
                              "slope_per_level": slope}
            report[key] = entry
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=1) + "\n")
            print(f"{key:20} " + "  ".join(f"{r} {entry[r]['nmae']['0.0']['mean']:.4f}->{entry[r]['nmae']['16.0']['mean']:.4f}"
                                           for r in only), flush=True)
    rows = ["# Robustness to Gaussian noise on the stored 8-bit image", "",
            "Mean NMAE at sigma = 0 / 4 / 16 intensity levels (two replicates per level).", "",
            "| Dataset / L | " + " | ".join(LABELS[r] for r in REPS) + " |", "|---|" + "---|" * len(REPS)]
    for key, entry in report.items():
        if key.startswith("_"):
            continue
        rows.append(f"| {key} | " + " | ".join(
            f"{entry[r]['nmae']['0.0']['mean']:.4f} / {entry[r]['nmae']['4.0']['mean']:.4f} / "
            f"{entry[r]['nmae']['16.0']['mean']:.4f}" if r in entry else "—" for r in REPS) + " |")
    (OUT / "robustness_multi.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
