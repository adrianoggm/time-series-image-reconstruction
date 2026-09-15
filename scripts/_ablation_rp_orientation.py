"""Phase 4: grid-720 baseline vs closed-form orientation for the RP decoder.

Runs on the SAME activity-balanced 150-window protocol as the robustness block
(``scripts/_make_robustness_tables._balanced_indices``, same seed), so the two
tables describe the same sample. Both variants are driven through the metadata-free
production inversion (diagonal centring), so the only thing that differs is how
the O(2) orientation is chosen.

Conditions: native-float RP, clean 8-bit PNG, and perturbed PNG at
sigma in {2,4,8,16} with 2 independent noise draws per level.

Per axis-series: NMAE, Pearson, overlap residual, signed-RP residual, decode time,
and whether the signed-RP selection picked the NMAE-optimal candidate.
Per window: RP cycle/re-encoding residual.

    python -m scripts._ablation_rp_orientation --n-windows 150 --workers 20

Writes results/_stats/rp_orientation_ablation.{json,md,tex}.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.GAF import normalize_to_zero_one
from src.recurrence_plots import varRP2
from src.rp_decoders import (
    _rescale,
    align_states,
    closed_form_orientation_candidates,
    distance_matrix,
    embed_2d,
    orientation_candidates,
    reconstruct_signal_from_states,
    signed_rp_error,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "_stats"
SIGMAS = (2.0, 4.0, 8.0, 16.0)
CONDITIONS = ("float", "png") + tuple(f"png_s{int(s)}" for s in SIGMAS)
VARIANTS = ("closed_form", "grid720")
TOP_K = 8
N_THETA = 720
EPS = 1e-12


def _to_img(mats, quantize):
    """Encoder storage normalisation: per channel min-max -> [0,255]."""
    chans = [normalize_to_zero_one(m) * 255 for m in mats]
    if quantize:
        chans = [np.rint(np.clip(c, 0, 255)) for c in chans]
    return np.stack(chans, axis=-1)


def _decode_axis(R_obs, minmax_c, variant, truth, n_steps):
    """Decode one channel through the production primitives, returning the chosen
    reconstruction plus the diagnostics the ablation needs.

    Mirrors ``rp_decoders.decode_channel(dims=2, recover='overlap', signed=True,
    scale_invariant=True)`` and additionally records the winning candidate's
    overlap residual, its signed-RP residual, and whether that candidate is the
    one that actually minimises NMAE against the ground truth (a sign/reflection
    selection failure when it is not).
    """
    t0 = time.perf_counter()
    dmat = distance_matrix(R_obs, source="direct")
    Y, _ = embed_2d(dmat)
    if variant == "closed_form":
        Qs = closed_form_orientation_candidates(Y)
    else:
        Qs = orientation_candidates(n_theta=N_THETA)
    cands = sorted((align_states(Y, Q) + (j,) for j, Q in enumerate(Qs)), key=lambda c: c[2])

    best = None
    scored = []
    for a, b, e_ov, cid in cands[:TOP_K]:
        x = _rescale(reconstruct_signal_from_states(a, b), minmax_c)[:n_steps]
        e_sig = signed_rp_error(R_obs, x, scale_invariant=True)
        scored.append((e_sig, e_ov, cid, x))
        if best is None or e_sig < best[0]:
            best = (e_sig, e_ov, cid, x)
    dt = time.perf_counter() - t0

    rng = float(np.ptp(truth)) or 1.0
    nmae_of = {cid: float(np.abs(x - truth).mean() / rng) for _e, _o, cid, x in scored}
    oracle_cid = min(nmae_of, key=nmae_of.get)
    e_sig, e_ov, cid, x = best
    return {
        "rec": x,
        "nmae": nmae_of[cid],
        "pearson": _corr(x, truth),
        "e_ov": float(e_ov),
        "e_sig": float(e_sig),
        "time": dt,
        "sel_fail": bool(nmae_of[cid] > nmae_of[oracle_cid] + 1e-9),
        "sel_regret": float(nmae_of[cid] - nmae_of[oracle_cid]),
    }


def _corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def _window_job(args):
    """One window, all conditions x variants x 3 axes. Worker (picklable)."""
    wi, win, seed = args
    n_steps = win.shape[0]
    mats = [varRP2(win[:, c].astype(float), TIME_STEPS=n_steps) for c in range(3)]
    img_f = _to_img(mats, quantize=False)
    img_q = _to_img(mats, quantize=True)
    rng = np.random.default_rng(seed)

    images = {"float": [img_f], "png": [img_q]}
    for s in SIGMAS:
        images[f"png_s{int(s)}"] = [
            np.rint(np.clip(img_q + rng.normal(0, s, img_q.shape), 0, 255))
            for _ in range(2)
        ]

    rows = []
    for cond, imgs in images.items():
        for rep, img in enumerate(imgs):
            for variant in VARIANTS:
                recs = []
                for c in range(3):
                    png_c = img[:, :, c].astype(float)
                    R_obs = png_c - float(np.median(np.diag(png_c)))
                    mm_c = (float(win[:, c].max()), float(win[:, c].min()))
                    d = _decode_axis(R_obs, mm_c, variant, win[:, c], n_steps)
                    recs.append(d.pop("rec"))
                    rows.append(dict(window=int(wi), axis=c, condition=cond, rep=rep,
                                     variant=variant, **d))
                rec = np.stack(recs, axis=1)
                num = den = 0.0
                for c in range(3):
                    R1 = varRP2(rec[:, c].astype(float), TIME_STEPS=n_steps)
                    num += float(np.linalg.norm(R1 - mats[c]))
                    den += float(np.linalg.norm(mats[c]))
                for r in rows[-3:]:
                    r["cycle"] = num / (den + EPS)
    return rows


def _stats(vals):
    v = np.asarray(vals, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-windows", type=int, default=150)
    ap.add_argument("--workers", type=int, default=min(20, (os.cpu_count() or 4)))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # Same protocol sample as the robustness block (same helper, same seed).
    from scripts._make_robustness_tables import LABELS, ORIG, SUBJECTS, _balanced_indices, _class_ids
    idx = _balanced_indices(LABELS, SUBJECTS, a.n_windows)
    wins = ORIG[idx]
    subj = SUBJECTS[idx]
    labs = _class_ids(LABELS)[idx]
    print(f"{len(wins)} windows ({len(wins) * 3} axis-series) from {len(np.unique(subj))} subjects; "
          f"activities {dict(zip(*np.unique(labs, return_counts=True)))}")

    jobs = [(int(i), wins[j], a.seed + 7919 * int(i)) for j, i in enumerate(idx)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in tqdm(ex.map(_window_job, jobs), total=len(jobs), desc="windows"):
            rows.extend(out)

    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for cond in CONDITIONS:
            sel = [r for r in rows if r["variant"] == variant and r["condition"] == cond]
            summary[variant][cond] = {
                "nmae": _stats([r["nmae"] for r in sel]),
                "pearson_mean": float(np.mean([r["pearson"] for r in sel])),
                "overlap_residual_mean": float(np.mean([r["e_ov"] for r in sel])),
                "signed_rp_residual_mean": float(np.mean([r["e_sig"] for r in sel])),
                "cycle_residual_mean": float(np.mean([r["cycle"] for r in sel])),
                "time_per_axis_s": float(np.mean([r["time"] for r in sel])),
                "selection_failures": int(sum(r["sel_fail"] for r in sel)),
                "selection_regret_mean": float(np.mean([r["sel_regret"] for r in sel])),
                "n_axis_series": len(sel),
            }

    design = {"n_windows": len(wins), "n_subjects": int(len(np.unique(subj))),
              "activity_counts": {str(k): int((labs == k).sum()) for k in np.unique(labs)},
              "sample_indices": idx.tolist(), "sigmas": list(SIGMAS), "noise_replicates": 2,
              "seed": a.seed, "top_k": TOP_K, "n_theta": N_THETA,
              "png_inversion": "diagonal (metadata-free)", "fold": "loto/3-fold/fold-0/test"}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rp_orientation_ablation.json").write_text(
        json.dumps({"design": design, "summary": summary, "rows": rows}, indent=2))

    # --- Markdown ----------------------------------------------------------
    md = [f"# RP orientation ablation: grid-{N_THETA} vs closed form", "",
          f"{design['n_windows']} activity-balanced windows "
          f"({design['n_windows'] * 3} axis-series) from {design['n_subjects']} subjects, "
          f"LOTO fold-0 test split. Metadata-free PNG inversion in both variants; "
          f"{design['noise_replicates']} noise draws per sigma; seed {design['seed']}.", "",
          "| Condition | Variant | NMAE mean | std | median | p95 | max | Pearson | "
          "Overlap res. | Signed-RP res. | Cycle res. | decode s/axis | Sel. fails |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        for variant in VARIANTS:
            s = summary[variant][cond]
            n = s["nmae"]
            md.append(f"| {cond} | {variant} | {n['mean']:.6f} | {n['std']:.6f} | "
                      f"{n['median']:.6f} | {n['p95']:.6f} | {n['max']:.6f} | "
                      f"{s['pearson_mean']:.5f} | {s['overlap_residual_mean']:.3e} | "
                      f"{s['signed_rp_residual_mean']:.5f} | {s['cycle_residual_mean']:.5f} | "
                      f"{s['time_per_axis_s']:.3f} | {s['selection_failures']}/{s['n_axis_series']} |")
    (OUT / "rp_orientation_ablation.md").write_text("\n".join(md) + "\n")
    (OUT / "rp_orientation_ablation_summary.json").write_text(
        json.dumps({"design": design, "summary": summary}, indent=2))

    # --- LaTeX (compact: float / clean PNG / sigma=8, plus cost) ------------
    tex = [r"\begin{table}[!htbp]", r"\centering",
           (r"\caption{RP orientation search: grid baseline versus closed-form solution, on "
            + str(design["n_windows"]) + r" activity-balanced windows ("
            + str(design["n_windows"] * 3) + r" axis-series) from "
            + str(design["n_subjects"]) + r" subjects. Both variants use the metadata-free "
            r"diagonal inversion, so only the orientation step differs. NMAE is the "
             r"range-normalised mean absolute error; the overlap residual is the "
             r"normalised translation-corrected state-overlap diagnostic, while the "
             r"closed form minimises its unnormalised quadratic numerator; "
             r"time is total decoding time per axis-series, including MDS.}"),
           r"\label{tab:rp_orientation_ablation}",
           r"\begin{tabular}{llccccc}", r"\toprule",
           r"Condition & Orientation & NMAE (mean$\pm$std) & NMAE p95 & Overlap res. & "
           r"Cycle res. & decode s/axis \\", r"\midrule"]
    label = {"float": "Native float", "png": "Clean 8-bit PNG", "png_s8": r"PNG $\sigma=8$"}
    for cond in ("float", "png", "png_s8"):
        for variant in VARIANTS:
            s = summary[variant][cond]
            n = s["nmae"]
            vname = r"Grid ($n_\theta{=}720$)" if variant == "grid720" else "Closed form"
            tex.append(f"{label[cond] if variant == VARIANTS[0] else ''} & {vname} & "
                       f"{n['mean']:.5f}$\\pm${n['std']:.5f} & {n['p95']:.5f} & "
                       f"{s['overlap_residual_mean']:.2e} & {s['cycle_residual_mean']:.4f} & "
                       f"{s['time_per_axis_s']:.3f}" + r" \\")
        if cond != "png_s8":
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "rp_orientation_ablation.tex").write_text("\n".join(tex) + "\n")

    print("\n".join(md[4:]))
    print(f"\nwrote {OUT / 'rp_orientation_ablation.json'} (+ .md, .tex)")


if __name__ == "__main__":
    main()
