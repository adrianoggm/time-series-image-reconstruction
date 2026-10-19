"""Metadata-free (diagonal-centred) RP inversion versus the operator-range oracle,
on the same 150 activity-balanced protocol windows, with the closed-form orientation
in both arms. Reports NMAE, physical MAE and Pearson at sigma=0 and sigma=8, which is
what Table~\\ref{tab:supp_rp_metadatafree} of the Supplementary Material needs.
The noisy condition uses the same two draws per window in both arms and averages
them within each window, matching the declared robustness protocol.

    python -m scripts._ablation_rp_metadata --workers 20

Writes results/_stats/rp_metadata_ablation.{json,tex}.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.GAF import normalize_to_zero_one
from src.recurrence_plots import varRP2
from src.rp_decoders import decode_window

OUT = Path(__file__).resolve().parents[1] / "results" / "_stats"
SIGMA = 8.0
N_REPS = 2
MODES = ("oracle", "diagonal")


def _nmae(orig, rec):
    rng = np.ptp(orig, axis=0)
    ok = rng > 1e-8
    return float((np.abs(orig - rec).mean(0)[ok] / rng[ok]).mean())


def _corr(orig, rec):
    """Mean per-axis Pearson over the three channels."""
    out = []
    for c in range(orig.shape[1]):
        a = orig[:, c] - orig[:, c].mean()
        b = rec[:, c] - rec[:, c].mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        if d > 0:
            out.append(float(a @ b / d))
    return float(np.mean(out))


def _job(args):
    wi, win, seed = args
    mats = [varRP2(win[:, c].astype(float), TIME_STEPS=win.shape[0]) for c in range(3)]
    png = np.stack([np.rint(np.clip(normalize_to_zero_one(m) * 255, 0, 255)) for m in mats], -1)
    rng = np.random.default_rng(seed)
    noisy = [
        np.rint(np.clip(png + rng.normal(0, SIGMA, png.shape), 0, 255))
        for _ in range(N_REPS)
    ]
    mm = ([float(win[:, c].max()) for c in range(3)], [float(win[:, c].min()) for c in range(3)])
    ranges = [(float(m.min()), float(m.max())) for m in mats]

    rows = []
    for mode in MODES:
        rp_range = ranges if mode == "oracle" else "diagonal"
        images = [("sigma0", 0, png)] + [
            ("sigma8", rep, img) for rep, img in enumerate(noisy)
        ]
        for cond, rep, img in images:
            rec = np.asarray(decode_window(img, mm, source="direct", rp_range=rp_range)[0])
            rec = rec[: win.shape[0], :3]
            rows.append(dict(window=int(wi), mode=mode, cond=cond, rep=int(rep),
                             nmae=_nmae(win, rec),
                             mae=float(np.abs(win - rec).mean()),
                             pearson=_corr(win, rec)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-windows", type=int, default=150)
    ap.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 4))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from scripts._make_robustness_tables import LABELS, ORIG, SUBJECTS, _balanced_indices
    idx = _balanced_indices(LABELS, SUBJECTS, a.n_windows)
    jobs = [(int(i), ORIG[i], a.seed + 7919 * int(i)) for i in idx]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in tqdm(ex.map(_job, jobs), total=len(jobs), desc="windows"):
            rows.extend(out)

    summary = {}
    for mode in MODES:
        for cond in ("sigma0", "sigma8"):
            sel = [r for r in rows if r["mode"] == mode and r["cond"] == cond]
            by_window = {}
            for r in sel:
                by_window.setdefault(r["window"], []).append(r)
            summary[f"{mode}/{cond}"] = {
                k: float(np.mean([
                    np.mean([r[k] for r in reps]) for reps in by_window.values()
                ])) for k in ("nmae", "mae", "pearson")
            }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rp_metadata_ablation.json").write_text(json.dumps(
        {"design": {"n_windows": len(idx), "sigma": SIGMA,
                    "noise_replicates": N_REPS, "seed": a.seed,
                    "orientation": "closed_form", "sample_indices": idx.tolist()},
         "summary": summary, "rows": rows}, indent=2))

    label = {"oracle": "operator-range oracle", "diagonal": "metadata-free (diagonal)"}
    tex = [r"\begin{tabular}{llccc}", r"\toprule",
           r"Noise & Decoder & NMAE & MAE (m/s$^2$) & Pearson \\", r"\midrule"]
    for ci, cond in enumerate(("sigma0", "sigma8")):
        head = r"$\sigma=0$" if cond == "sigma0" else r"$\sigma=8$"
        for mi, mode in enumerate(MODES):
            s = summary[f"{mode}/{cond}"]
            tex.append(f"{head if mi == 0 else '           '} & {label[mode]:<26} & "
                       f"${s['nmae']:.4f}$ & ${s['mae']:.4f}$ & ${s['pearson']:.5f}$" + r" \\")
        del ci
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "rp_metadata_ablation.tex").write_text("\n".join(tex) + "\n")

    print(f"{'arm':>28} {'NMAE':>9} {'MAE':>9} {'Pearson':>9}")
    for k, s in summary.items():
        print(f"{k:>28} {s['nmae']:>9.5f} {s['mae']:>9.5f} {s['pearson']:>9.6f}")
    print("\n" + "\n".join(tex))


if __name__ == "__main__":
    main()
