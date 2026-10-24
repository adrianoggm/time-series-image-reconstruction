"""Sign identifiability of the signed RP (review of the manuscript, 2026-10-24).

The signed RP of x and of its reflection -x coincide exactly when no pair of
states carries direction, i.e. every displacement s_i - s_j lies within pi/4 of
the anti-diagonal. Then the decoder cannot choose the reflection, even though
the states have rank two. This script

1. checks the reviewer's counterexample x = (1, -1.05, 1.10, ...) on the
   production encoder and decoder, and
2. counts, on the benchmark windows, the channel series with no informative
   pair (ambiguous) and with fewer than 1% informative pairs.

    python -m scripts._check_sign_identifiability
Writes results/_stats/sign_identifiability.json.
"""
import json
from pathlib import Path

import numpy as np


def informative_share(x):
    from scripts._rp_conditions import series_conditions

    return series_conditions(x)[0]


def main():
    from scripts._benchmark import DATASETS, MIN_WINDOWS
    from src.datasets import get, make_windows
    from src.recurrence_plots import varRP2
    from src.representations import REPRESENTATIONS

    x = np.array([1, -1.05, 1.10, -1.15, 1.20, -1.25, 1.30, -1.35])
    same = bool(np.array_equal(varRP2(x), varRP2(-x)))
    rep = REPRESENTATIONS["RP"]
    mm = (float(x.max()), float(x.min()))
    rec = rep.decode(rep.encode(x, mm), mm)
    err = float(np.mean(np.abs(rec - x)) / np.ptp(x))
    mm_r = (float((-x).max()), float((-x).min()))
    rec_r = rep.decode(rep.encode(-x, mm_r), mm_r)
    err_r = float(np.mean(np.abs(rec_r + x)) / np.ptp(x))
    print(f"counterexample: signed RP(x) == signed RP(-x): {same}; informative share {informative_share(x):.3f}; "
          f"float NMAE x {err:.3f}, -x {err_r:.3f}")
    report = {"counterexample": {"x": x.tolist(), "identical_signed_rp": same, "informative_share": informative_share(x),
                                 "float_nmae_x": err, "float_nmae_minus_x": err_r}}
    total = zero = low = 0
    for name, cfg in DATASETS.items():
        adapter = get(name, **cfg)
        for L in (64, 128, 256):
            npz = np.load(Path("results/_benchmark") / f"{name}_L{L}.npz")
            if len(npz["window_index"]) < MIN_WINDOWS:
                continue
            w = make_windows(adapter, L, dtype=np.float32)
            shares = np.array([informative_share(w.X[i][:, c]) for i in npz["window_index"]
                               for c in range(w.X.shape[2]) if np.ptp(w.X[i][:, c]) > 0])
            report[f"{name}/L{L}"] = {"series": int(len(shares)), "no_informative_pair": int((shares == 0).sum()),
                                      "below_1pct": int((shares < 0.01).sum()), "min_share": float(shares.min())}
            total, zero, low = total + len(shares), zero + int((shares == 0).sum()), low + int((shares < 0.01).sum())
            print(f"{name}/L{L}: {report[f'{name}/L{L}']}", flush=True)
    report["_total"] = {"series": total, "no_informative_pair": zero, "below_1pct": low}
    print("total", report["_total"])
    Path("results/_stats/sign_identifiability.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
