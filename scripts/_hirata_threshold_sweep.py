"""Recurrence-rate sweep for Hirata's reconstruction (review of the manuscript, 2026-10-24).

Hirata et al. do not prescribe the threshold; their 2016 restatement only notes
that the method still works with up to 70% of the binary map set to 1. Instead
of defending one rate, the baseline is run at several recurrence rates on the
stored 8-bit images (unsigned image, and signed image with the sign used for the
reflection) and the best rate is reported per dataset, which favours the baseline.

    python -m scripts._hirata_threshold_sweep --n 100 --length 128 --workers 20
Writes results/_stats/hirata_threshold_sweep.json and .md.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RATES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.75)
OUT = Path("results/_stats")


def window_sweep(window) -> np.ndarray:
    """(T, C) -> NMAE (2 encodings, len(RATES), C); NaN for flat channels."""
    from threadpoolctl import threadpool_limits

    from src.representations import REPRESENTATIONS
    from src.rp_decoders import signed_rp_error
    from src.rp_hirata import binary_rp, eps_for_rate, states, to_series

    uns, sig = REPRESENTATIONS["RP_HIRATA"], REPRESENTATIONS["RP_HIRATA_SIGNED"]
    out = np.full((2, len(RATES), window.shape[1]), np.nan)
    with threadpool_limits(1):
        for c in range(window.shape[1]):
            x = window[:, c].astype(float)
            span = np.ptp(x)
            if span < 1e-12:
                continue
            mm = (float(x.max()), float(x.min()))
            D = np.abs(uns.from_uint8(uns.to_uint8(uns.encode(x, mm))))
            R = sig.from_uint8(sig.to_uint8(sig.encode(x, mm)))
            Ds = np.abs(R)
            for k, rate in enumerate(RATES):
                try:
                    r = to_series(states(binary_rp(D, eps_for_rate(D, rate)))[0], mm, "as_is")
                    out[0, k, c] = np.mean(np.abs(r - x)) / span
                    r = to_series(states(binary_rp(Ds, eps_for_rate(Ds, rate)))[0], mm, "as_is")
                    m = mm[0] + mm[1] - r
                    r = m if signed_rp_error(R, m, True) < signed_rp_error(R, r, True) else r
                    out[1, k, c] = np.mean(np.abs(r - x)) / span
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    from scripts._benchmark import DATASETS
    from src.datasets import get, make_windows

    report = {"_design": {"rates": RATES, "windows": args.n, "length": args.length, "seed": 0,
                          "doc": __doc__.split("\n\n")[1].strip()}}
    for name, cfg in DATASETS.items():
        npz = np.load(Path("results/_benchmark") / f"{name}_L{args.length}.npz")
        idx = npz["window_index"][np.sort(np.random.default_rng(0).choice(len(npz["window_index"]),
                                                                          size=min(args.n, len(npz["window_index"])),
                                                                          replace=False))]
        w = make_windows(get(name, **cfg), args.length, dtype=np.float32)
        with ProcessPoolExecutor(args.workers) as ex:
            vals = np.stack(list(ex.map(window_sweep, [w.X[i] for i in idx], chunksize=2)))   # (n, 2, rates, C)
        entry = {}
        for e, enc in enumerate(("unsigned", "signed")):
            means = [float(np.nanmean(vals[:, e, k])) for k in range(len(RATES))]
            best = int(np.argmin(means))
            entry[enc] = {"nmae_by_rate": dict(zip(map(str, RATES), means, strict=True)),
                          "best_rate": RATES[best], "best_nmae": means[best]}
        report[name] = entry
        print(f"{name:13} " + " | ".join(f"{enc}: best rate {entry[enc]['best_rate']} NMAE {entry[enc]['best_nmae']:.3f}"
                                         f" (0.75: {entry[enc]['nmae_by_rate']['0.75']:.3f})"
                                         for enc in ("unsigned", "signed")), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hirata_threshold_sweep.json").write_text(json.dumps(report, indent=1) + "\n")
    rows = ["# Hirata's reconstruction: 8-bit NMAE against the recurrence rate", "",
            f"T = {args.length}, {args.n} windows per dataset (benchmark windows), every channel.", "",
            "| Dataset | Encoding | " + " | ".join(str(r) for r in RATES) + " | Best |", "|---|---|" + "---|" * (len(RATES) + 1)]
    for name, entry in report.items():
        if name.startswith("_"):
            continue
        for enc in ("unsigned", "signed"):
            e = entry[enc]
            rows.append(f"| {name} | {enc} | " + " | ".join(f"{v:.3f}" for v in e["nmae_by_rate"].values())
                        + f" | {e['best_rate']} |")
    (OUT / "hirata_threshold_sweep.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
