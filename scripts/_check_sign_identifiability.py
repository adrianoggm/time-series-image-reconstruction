"""Hypotheses of Proposition 1 on every benchmark series (reviews of the manuscript, 2026-10-24 and 10-26).

Proposition 1 (doc/infsci/main.tex) gives sufficient conditions for the geometric decoder to recover a
series exactly from its floating-point signed RP:
  (a) the centred states (x_i, x_{i+1}) span the plane: sigma_2 / sigma_1 > 0;
  (b) no relation (R) x_i + x_{i+2} = kappa, nor (F) alpha (x_{i+2} - x_i) = beta x_{i+1} + kappa with
      (alpha, beta) != (0, 0), holds for all i;
  (c) at least one pair of states is sign-informative (displacement within pi/4 of +-(1, 1)).
This script

1. checks the counterexample x = (1, -1.05, 1.10, ...) on the production encoder and decoder, and
2. evaluates (a), (b) and (c) on every channel series of the benchmark windows, with a relative tolerance
   TOL: (a) sigma_2/sigma_1 > TOL; (R) fails when the range of x_i + x_{i+2} exceeds TOL times the range of
   x; (F) fails when the smallest singular value of the column-normalised matrix [x_{i+2} - x_i, x_{i+1}, 1]
   exceeds TOL times the largest. It also reports the sufficient rank-three criterion on the centred triples.

    python -m scripts._check_sign_identifiability
Writes results/_stats/sign_identifiability.json.
"""
import json
from pathlib import Path

import numpy as np

TOL = 1e-9


def informative_share(x):
    from scripts._rp_conditions import series_conditions

    return series_conditions(x)[0]


def hypotheses(x: np.ndarray) -> tuple[float, float, float, float]:
    """Margins of (a), (R), (F) and of the rank-three criterion; each condition holds when its margin > TOL."""
    x = np.asarray(x, dtype=float)
    s = np.stack([x[:-1], x[1:]], axis=1)
    sv = np.linalg.svd(s - s.mean(0), compute_uv=False)
    a = sv[1] / sv[0]
    r = np.ptp(x[:-2] + x[2:]) / np.ptp(x)
    A = np.column_stack([x[2:] - x[:-2], x[1:-1], np.ones(len(x) - 2)])
    A = A / np.maximum(np.linalg.norm(A, axis=0), 1e-300)
    fa = np.linalg.svd(A, compute_uv=False)
    f = fa[-1] / fa[0]
    t = np.stack([x[:-2], x[1:-1], x[2:]], axis=1)
    tv = np.linalg.svd(t - t.mean(0), compute_uv=False)
    return float(a), float(r), float(f), float(tv[2] / tv[0])


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
          f"float NMAE x {err:.3f}, -x {err_r:.3f}; hypotheses margins {hypotheses(x)}")
    report = {"_design": {"tolerance": TOL, "doc": __doc__.split("\n\n")[1].strip()},
              "counterexample": {"x": x.tolist(), "identical_signed_rp": same, "informative_share": informative_share(x),
                                 "float_nmae_x": err, "float_nmae_minus_x": err_r,
                                 "margins_a_R_F_rank3": hypotheses(x)}}
    tot = {"series": 0, "no_informative_pair": 0, "below_1pct": 0, "fails_a": 0, "fails_R": 0, "fails_F": 0,
           "fails_rank3": 0, "fails_any_abc": 0}
    mins = {"min_share": 1.0, "min_a": 1.0, "min_R": np.inf, "min_F": 1.0, "min_rank3": 1.0}
    for name, cfg in DATASETS.items():
        adapter = get(name, **cfg)
        for L in (64, 128, 256):
            npz = np.load(Path("results/_benchmark") / f"{name}_L{L}.npz")
            if len(npz["window_index"]) < MIN_WINDOWS:
                continue
            w = make_windows(adapter, L, dtype=np.float32)
            series = [w.X[i][:, c].astype(float) for i in npz["window_index"] for c in range(w.X.shape[2])
                      if np.ptp(w.X[i][:, c].astype(float)) >= 1e-12]   # same flat rule as _benchmark.window_row
            shares = np.array([informative_share(s) for s in series])
            h = np.array([hypotheses(s) for s in series])          # (n, 4): a, R, F, rank3
            fa, fr, ff, f3 = h[:, 0] <= TOL, h[:, 1] <= TOL, h[:, 2] <= TOL, h[:, 3] <= TOL
            anyf = fa | fr | ff | (shares == 0)
            entry = {"series": len(series), "no_informative_pair": int((shares == 0).sum()),
                     "below_1pct": int((shares < 0.01).sum()), "min_share": float(shares.min()),
                     "fails_a": int(fa.sum()), "fails_R": int(fr.sum()), "fails_F": int(ff.sum()),
                     "fails_rank3": int(f3.sum()), "fails_any_abc": int(anyf.sum()),
                     "min_a": float(h[:, 0].min()), "min_R": float(h[:, 1].min()), "min_F": float(h[:, 2].min()),
                     "min_rank3": float(h[:, 3].min())}
            report[f"{name}/L{L}"] = entry
            for k in tot:
                tot[k] += entry[k]
            for k in mins:
                mins[k] = min(mins[k], entry[k])
            print(f"{name}/L{L}: {entry}", flush=True)
    report["_total"] = tot | {k: float(v) for k, v in mins.items()}
    print("total", report["_total"])
    Path("results/_stats/sign_identifiability.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
