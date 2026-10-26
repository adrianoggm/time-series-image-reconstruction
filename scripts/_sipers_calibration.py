"""Independent calibration of the frequency threshold of Sipers et al.'s decoder (review, 2026-10-26).

The threshold tol = 1e-4 in src/rp_sipers.py was first chosen with scripts/_sipers_pilot.py on the first 100
benchmark windows of each dataset at T = 128, which are also evaluated. This script repeats the choice on
calibration windows that no evaluation uses at T = 128: windows outside both the fidelity sample
(results/_benchmark/<dataset>_L128.npz) and the classification sample (scripts/_classification.py: up to
6000 windows, seed 0; ESC-10 uses other audio settings and is not affected). They are drawn with seed 1 and
may come from the same recordings as evaluated windows.

Criterion, as in the pilot: 8-bit NMAE of Sipers' decoder on the circular unsigned RP with the reflection
fixed using the original series (the method is exact only up to sign), averaged over the channel series of
each dataset and then over datasets with equal weight. The signed variant, whose reflection is chosen from the
image, is reported as a second criterion. If the chosen tol is 1e-4, the evaluation stands as it is.

    python -m scripts._sipers_calibration --n 100        (needs the datasets: run in WSL)
Writes results/_stats/sipers_calibration.{json,md}.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts._sipers_pilot import DATASETS, TOLS, nmae, pick_sign, storage
from src.rp_sipers import circular_rp, sipers_decode, to_range

OUT = Path("results/_stats/sipers_calibration")
CLS_MAX_WINDOWS = 6000          # scripts/_classification.py default; its sample is drawn with seed 0


def evaluated_windows(name: str, X: np.ndarray, classified: bool) -> set[int]:
    """Indices of the windows evaluated at T = 128: the fidelity sample and, for labelled datasets, the
    classification sample, drawn exactly as scripts/_classification.py does (non-flat windows, seed 0)."""
    used = set(np.load(Path("results/_benchmark") / f"{name}_L128.npz")["window_index"].tolist())
    if classified:
        audible = np.flatnonzero(X.std(axis=(1, 2)) > 0.0)
        used |= set(np.random.default_rng(0).choice(audible, size=min(CLS_MAX_WINDOWS, len(audible)),
                                                    replace=False).tolist())
    return used


def series_sweep(x: np.ndarray) -> dict:
    mm = (float(x.max()), float(x.min()))
    R = circular_rp(x)
    D8, R8 = storage(np.abs(R), False), storage(R, True)
    S8 = 0.5 * (np.abs(R8) + np.abs(R8).T)
    out = {}
    for t in TOLS:
        xh = sipers_decode(D8, tol=t)
        out[f"unsigned_oracle/{t:g}"] = min(nmae(x, to_range(xh, mm)), nmae(x, to_range(-xh, mm)))
        out[f"signed/{t:g}"] = nmae(x, pick_sign(R8, sipers_decode(S8, tol=t), mm))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    from scripts._benchmark import DATASETS as CFG
    from scripts._classification import DATASETS as CLS_CFG
    from src.datasets import get, make_windows

    report = {"_design": {"T": 128, "windows": args.n, "seed": 1, "tols": TOLS, "doc": __doc__.split("\n\n")[1]}}
    for ds in DATASETS:
        X = make_windows(get(ds, **CFG[ds]), 128, dtype=np.float32).X
        if CLS_CFG.get(ds, None) is not None and CLS_CFG[ds] != CFG[ds]:
            raise SystemExit(f"{ds}: classification and benchmark use different adapter settings")
        used = evaluated_windows(ds, X, ds in CLS_CFG)
        free = np.array(sorted(set(range(len(X))) - used))
        if len(free) == 0:
            report[ds] = {"calibration_windows": 0}
            print(f"{ds:13} no window outside the evaluated samples", flush=True)
            continue
        pick = np.sort(np.random.default_rng(1).choice(free, size=min(args.n, len(free)), replace=False))
        rows = [series_sweep(X[i][:, c].astype(float)) for i in pick for c in range(X.shape[2])
                if np.ptp(X[i][:, c]) > 0]
        report[ds] = {"calibration_windows": int(len(pick)), "series": len(rows),
                      "overlap_with_evaluated": int(len(set(pick.tolist()) & used))}
        report[ds] |= {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"{ds:13} {len(pick)} windows ({len(rows)} series) | " + "  ".join(
            f"{t:g}: {report[ds][f'unsigned_oracle/{t:g}']:.4f}" for t in TOLS), flush=True)
    done = [d for d in DATASETS if report[d].get("series")]
    for crit in ("unsigned_oracle", "signed"):
        means = {f"{t:g}": float(np.mean([report[d][f"{crit}/{t:g}"] for d in done])) for t in TOLS}
        report[f"_choice_{crit}"] = {"mean_over_datasets": means, "chosen": min(means, key=means.get),
                                     "datasets": done}
        print(crit, "->", report[f"_choice_{crit}"]["chosen"], means, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    rows = ["# Sipers et al.: frequency threshold chosen on calibration windows (T = 128)", "",
            "Windows outside every evaluated sample at T = 128. 8-bit NMAE, mean over channel series.", "",
            "| Dataset | Windows | Criterion | " + " | ".join(f"tol {t:g}" for t in TOLS) + " |",
            "|---|---|---|" + "---|" * len(TOLS)]
    for d in done:
        for crit in ("unsigned_oracle", "signed"):
            rows.append(f"| {d} | {report[d]['calibration_windows']} | {crit} | "
                        + " | ".join(f"{report[d][f'{crit}/{t:g}']:.4f}" for t in TOLS) + " |")
    for crit in ("unsigned_oracle", "signed"):
        c = report[f"_choice_{crit}"]
        rows.append(f"| **mean** | | {crit} | " + " | ".join(f"{c['mean_over_datasets'][f'{t:g}']:.4f}" for t in TOLS)
                    + f" | chosen {c['chosen']} |")
    OUT.with_suffix(".md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
