"""How much orientation information does the signed RP carry? (docs/experiments/rp_signed_comparison.md)

For a displacement d = s_i - s_j the sign is +1 iff the angle to v = (1,1)/sqrt2 is
<= pi/4. Reflecting the signal (x -> -x) maps d to -d; the pair separates the two
orientations only if sign(d) != sign(-d), i.e. if d lies within pi/4 of the +-v
axis. Per window and channel this reports, in float precision:

* informative_frac  share of off-diagonal pairs whose sign separates x from -x
* informative_mass  the same share weighted by |d|, i.e. by its weight in |R|
* margin            ||R(x) - R(-x)||_F / ||R(x)||_F, how far apart the signed RPs of
                    the two orientations are; 0 means the sign cannot tell them apart

    python -m scripts._rp_sign_information --datasets wisdm,pamap2 --n 500
Writes results/_stats/rp_sign_information.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.datasets import get, make_windows
from src.recurrence_plots import varRP2

COS = np.cos(np.pi / 4)


def channel_stats(x: np.ndarray) -> tuple[float, float, float]:
    s = np.stack([x[:-1], x[1:]], axis=1)
    d = s[:, None, :] - s[None, :, :]
    norm = np.linalg.norm(d, axis=-1)
    off = ~np.eye(len(s), dtype=bool) & (norm > 0)
    cosang = (d @ (np.ones(2) / np.sqrt(2)))[off] / norm[off]
    informative = (cosang >= COS) | (cosang <= -COS)          # d within pi/4 of +v or -v
    R, Rm = varRP2(x, TIME_STEPS=len(x)), varRP2(-x, TIME_STEPS=len(x))
    margin = float(np.linalg.norm(R - Rm) / (np.linalg.norm(R) + 1e-12))
    return float(informative.mean()), float((norm[off] * informative).sum() / norm[off].sum()), margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="wisdm,pamap2")
    ap.add_argument("--n", type=int, default=500, help="windows sampled per dataset (seed 0)")
    ap.add_argument("--length", type=int, default=128)
    args = ap.parse_args()

    out = {"definition": __doc__.split("\n\n")[1].strip(), "n_per_dataset": args.n, "length": args.length}
    for name in args.datasets.split(","):
        w = make_windows(get(name), args.length)
        idx = np.random.default_rng(0).choice(len(w.X), size=min(args.n, len(w.X)), replace=False)
        rows = np.array([channel_stats(w.X[i, :, c]) for i in idx for c in range(w.X.shape[2])])
        frac, mass, margin = rows.T
        out[name] = {
            "windows": int(len(idx)), "channels": int(w.X.shape[2]),
            "informative_frac": {"mean": frac.mean(), "p05": np.quantile(frac, 0.05), "p50": np.median(frac)},
            "informative_mass": {"mean": mass.mean(), "p05": np.quantile(mass, 0.05), "p50": np.median(mass)},
            "margin": {"mean": margin.mean(), "p05": np.quantile(margin, 0.05), "min": margin.min()},
            "channel_series_with_margin_below_0.01": int((margin < 0.01).sum()),
        }
        o = out[name]
        print(f"{name}: {o['windows']} windows x {o['channels']} ch | informative pairs "
              f"{o['informative_frac']['mean']:.3f} (p05 {o['informative_frac']['p05']:.3f}) | by |d| "
              f"{o['informative_mass']['mean']:.3f} | margin mean {o['margin']['mean']:.3f}, "
              f"min {o['margin']['min']:.4f}, <0.01 in {o['channel_series_with_margin_below_0.01']}")
    path = Path("results/_stats/rp_sign_information.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
