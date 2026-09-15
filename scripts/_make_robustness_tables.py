"""Robustness block (doc/VALIDATION_REDESIGN.md, Block 2b): perturbation curves
and cycle consistency for the four decoders, computed by driving the real
encoders/decoders end to end (encode -> 8-bit PNG -> add pixel noise -> decode).

SELF-VERIFICATION GATE: at sigma=0 (the exact 8-bit path), every sampled
reconstruction must agree with the corresponding cached pipeline reconstruction
to within 5% of the original axis range. A method that fails on any sampled
window is reported as UNVERIFIED.

Run inside the WSL venv:
    python scripts/_make_robustness_tables.py > .robustness_tables.tex
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

from src.GADF import GADF_to_TS, Reconstruct_GADF, varGADF
from src.GAF import GAF_to_TS, Reconstruct_GAF, normalize_to_zero_one, varGAF
from src.MTF import Reconstruct_MTF, varMTF2   # varMTF2 == pipeline encoder (pyts MarkovTransitionField)
from src.recurrence_plots import Reconstruct_RP, varRP2

DATA = Path("/home/adrianogarcia/tfg/data/WISDM")
ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = ROOT / "results" / "_stats"
OUTPUT_JSON = Path(os.environ.get("TFG_ROBUST_OUTPUT", STATS_DIR / "robustness.json"))
PARTIAL_JSON = OUTPUT_JSON.with_suffix(".partial.json")
FOLD0 = "sampling_loto/3-fold/fold-0/test"
ORIG = np.load(DATA / "tseries/recurrence_plot" / FOLD0 / "validation_data.npy")
LABELS = np.load(DATA / "tseries/recurrence_plot" / FOLD0 / "y_validation_data.npy")
SUBJECTS = np.load(DATA / "tseries/recurrence_plot" / FOLD0 / "sj_validation_data.npy").reshape(-1)
# Stored (pipeline) reconstruction per method, used as the sigma=0 gate target.
STORED_NPY = {"RP": "tseries/recurrence_plot/X_all_rec.npy", "GASF": "tseries/GAF/X_all_gaf.npy",
              "GADF": "tseries/GADF/X_all_gadf.npy", "MTF": "tseries/MTF/X_all_mtf.npy"}
N_WINDOWS = int(os.environ.get("TFG_ROBUST_N", "150"))
# "diagonal" = metadata-free production inversion; "oracle" = per-window operator
# range (Supplementary Material upper bound). See _encode.
RP_RANGE_MODE = os.environ.get("TFG_RP_RANGE", "diagonal")
N_REPS = int(os.environ.get("TFG_ROBUST_REPS", "2"))
SIGMAS = (0.0, 2.0, 4.0, 8.0, 16.0)          # pixel-noise std, out of 255
# sigma=0 round-trip must reproduce the pipeline's stored reconstruction on the
# SAME windows: mean |mine - stored| / range must stay below this.
GATE_TOL = 0.05
SEED = int(os.environ.get("TFG_ROBUST_SEED", "0"))
RNG = np.random.default_rng(SEED)
_EPS = 1e-8


def _class_ids(y):
    """Return integer activity labels from scalar or one-hot arrays."""
    y = np.asarray(y)
    return np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)


def _balanced_indices(labels, subjects, n):
    """Activity-balanced sample spread across subjects within each activity."""
    labels = _class_ids(labels)
    classes = np.unique(labels)
    base, remainder = divmod(n, len(classes))
    selected = []
    for ci, cls in enumerate(classes):
        quota = base + (ci < remainder)
        pools = {}
        for subj in np.unique(subjects[labels == cls]):
            idx = np.flatnonzero((labels == cls) & (subjects == subj))
            pools[subj] = list(RNG.permutation(idx))
        order = list(RNG.permutation(list(pools)))
        while quota > 0 and order:
            next_order = []
            for subj in order:
                if pools[subj] and quota > 0:
                    selected.append(pools[subj].pop())
                    quota -= 1
                if pools[subj]:
                    next_order.append(subj)
            order = next_order
        if quota:
            raise ValueError(f"not enough windows for activity {cls}")
    return np.asarray(selected, dtype=int)


def _load_stored(method):
    arr = np.load(DATA / STORED_NPY[method].replace("X_all", FOLD0 + "/X_all"))
    if arr.ndim == 3 and arr.shape[1] in (1, 3) and arr.shape[1] != arr.shape[2]:
        arr = np.transpose(arr, (0, 2, 1))
    return arr


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _encode(method, win):
    """Return (list of 3 matrices, RP PNG-inversion mode else None).

    RP's second element is what the decoder needs to invert the stored image.
    Default is the string "diagonal" — the metadata-free production mode, where the
    RP's zero-valued diagonal locates R=0 in the 8-bit range. Setting
    TFG_RP_RANGE=oracle instead hands the decoder the per-window (Rmin,Rmax)
    operator range, the upper-bound variant reported in the Supplementary Material.
    """
    if method == "GADF":
        M = [varGADF(win, d, 128) for d in ("x", "y", "z")]
        return M, None
    if method == "GASF":
        M = [varGAF(win, d, 128) for d in ("x", "y", "z")]
        return M, None
    if method == "MTF":
        M = [np.nan_to_num(_quiet(varMTF2, win, d, 128), nan=0.0) for d in ("x", "y", "z")]
        return M, None
    if method == "RP":
        M = [varRP2(win[:, c]) for c in range(3)]
        if RP_RANGE_MODE == "oracle":
            return M, [(float(m.min()), float(m.max())) for m in M]
        return M, "diagonal"
    raise ValueError(method)


def _to_png(method, M, quantize=True):
    """Image (H,W,3) float in [0,255] per the encoder's storage normalisation.
    quantize=False skips the 8-bit rounding, giving the float (algorithm-only)
    path so its error can be separated from PNG quantisation."""
    rnd = np.round if quantize else (lambda a: a)
    if method == "MTF":
        chans = [rnd(np.clip(m, 0, 1) * 255) for m in M]             # already [0,1]
    else:
        chans = [rnd(normalize_to_zero_one(m) * 255) for m in M]     # min-max -> [0,1]
    return np.stack(chans, axis=-1)


def _decode(method, img, win, rp_range):
    maxs = [float(win[:, c].max()) for c in range(3)]
    mins = [float(win[:, c].min()) for c in range(3)]
    dct = {0: [maxs, mins]}
    if method == "GADF":
        rec = np.array(Reconstruct_GADF(img, dct, 0))
    elif method == "GASF":
        rec = np.array(Reconstruct_GAF(img, dct, 0))
    elif method == "MTF":
        rec = np.array(_quiet(Reconstruct_MTF, img, dct, 0))
    elif method == "RP":
        # Production settings: metadata-free diagonal inversion + closed-form
        # orientation (decode_window's default orientation_method).
        from src.rp_decoders import decode_window
        rec_tc, _ = decode_window(img, [maxs, mins], source="direct", dims=2,
                                  recover="overlap", signed=True, rp_range=rp_range)
        rec = np.asarray(rec_tc)
    else:
        raise ValueError(method)
    if rec.shape == (3, win.shape[0]):
        rec = rec.T
    return rec[: win.shape[0], :3]


def _decode_float_operator(method, matrices, win, rp_range):
    """Decode before 8-bit quantisation, using each operator's native matrix.

    GASF and GADF can consume their float matrices directly. RP still passes through
    its image-coordinate map, but un-rounded: the map is a positive affine rescale,
    and diagonal centring plus the scale-invariant signed comparison make the
    decoder insensitive to it, so no quantisation is introduced. MTF remains a
    representative decoder even for a float MTF matrix, so its intrinsic loss is
    retained.
    """
    maxs = [float(win[:, c].max()) for c in range(3)]
    mins = [float(win[:, c].min()) for c in range(3)]
    dct = [maxs, mins]
    if method == "GADF":
        return np.column_stack([GADF_to_TS(matrices[c], c, dct) for c in range(3)])
    if method == "GASF":
        return np.column_stack([GAF_to_TS(matrices[c], c, dct) for c in range(3)])
    return _decode(method, _to_png(method, matrices, quantize=False), win, rp_range)


def _nmae(orig, rec):
    rng = orig.max(0) - orig.min(0)                       # (3,)
    valid = rng > 1e-8
    if not valid.any():
        return float("nan")
    return float((np.abs(orig - rec).mean(0)[valid] / rng[valid]).mean())


def _mae(orig, rec):
    return float(np.abs(orig - rec).mean())


def _cluster_bootstrap_ci(values, subjects, n_boot=10_000):
    """Percentile CI obtained by resampling sampled subjects as clusters."""
    uniq = np.unique(subjects)
    sums = np.array([values[subjects == s].sum() for s in uniq])
    counts = np.array([(subjects == s).sum() for s in uniq])
    rng = np.random.default_rng(SEED + 104729)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _summary(values, subjects):
    values = np.asarray(values, dtype=float)
    lo, hi = _cluster_bootstrap_ci(values, subjects)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "ci95": [lo, hi]}


def run_method(method, wins, stored, subjects):
    curve = {s: [] for s in SIGMAS}
    mae0, cyc, agree, float_nmae = [], [], [], []
    for wi, win in enumerate(wins):
        try:
            M, rp_range = _encode(method, win)
            png = _to_png(method, M)
        except Exception as e:                    # noqa: BLE001
            return {"verified": False, "error": f"encode: {e}"}
        # Float (algorithm-only) path: decode the un-rounded image, no PNG step.
        try:
            rec_f = _decode_float_operator(method, M, win, rp_range)
            float_nmae.append(_nmae(win, rec_f))
        except Exception:                         # noqa: BLE001
            pass
        for s in SIGMAS:
            rep_errors, clean_rec = [], None
            for _ in range(1 if s == 0 else N_REPS):
                if s == 0:
                    noisy = png
                else:
                    # Independent noise for every pixel and channel, clipped and
                    # rounded back to the 8-bit intensity grid before decoding.
                    noisy = np.rint(np.clip(png + RNG.normal(0, s, png.shape), 0, 255))
                try:
                    rec = _decode(method, noisy, win, rp_range)
                except Exception as e:            # noqa: BLE001
                    return {"verified": False, "error": f"decode s={s}: {e}"}
                rep_errors.append(_nmae(win, rec))
                if s == 0:
                    clean_rec = rec
            curve[s].append(float(np.mean(rep_errors)))
            if s == 0:
                mae0.append(_mae(win, clean_rec))
                # g_mw compares this clean reconstruction with the cached
                # reconstruction for the same method and window.
                rng = win.max(0) - win.min(0)
                valid = rng > 1e-8
                agree.append(float((np.abs(clean_rec - stored[wi]).mean(0)[valid]
                                    / rng[valid]).mean()))
                try:                              # cycle consistency: re-encode clean recon
                    M2, _ = _encode(method, clean_rec)
                    num = sum(np.linalg.norm(a - b) for a, b in zip(M2, M))
                    den = sum(np.linalg.norm(a) for a in M) + _EPS
                    cyc.append(float(num / den))
                except Exception:                 # noqa: BLE001
                    pass
    agree_mean = float(np.mean(agree))
    agree_max = float(np.max(agree))
    return {
        "verified": agree_max <= GATE_TOL,
        "gate_agreement_mean": agree_mean, "gate_agreement_max": agree_max,
        "mae_sigma0": float(np.mean(mae0)),
        "nmae_float": _summary(float_nmae, subjects) if float_nmae else None,
        "nmae_curve": {f"{s}": _summary(curve[s], subjects) for s in SIGMAS},
        "cycle_consistency": _summary(cyc, subjects) if cyc else None,
    }


def main():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    sample_idx = _balanced_indices(LABELS, SUBJECTS, N_WINDOWS)
    wins = ORIG[sample_idx]
    sample_subjects = SUBJECTS[sample_idx]
    sample_labels = _class_ids(LABELS)[sample_idx]
    print(f"% robustness on {len(wins)} activity-balanced windows from "
          f"{len(np.unique(sample_subjects))} subjects; sigmas={SIGMAS}; "
          f"noise_replicates={N_REPS}; seed={SEED}", file=sys.stderr)
    res = {}
    for m in ("RP", "GASF", "GADF", "MTF"):
        try:
            stored = _load_stored(m)[sample_idx]
        except Exception as e:                    # noqa: BLE001
            res[m] = {"verified": False, "error": f"stored load: {e}"}
            print(f"% {m}: UNVERIFIED (stored load: {e})", file=sys.stderr)
            continue
        res[m] = run_method(m, wins, stored, sample_subjects)
        r = res[m]
        if r["verified"]:
            c = r["nmae_curve"]
            print(f"% {m}: VERIFIED (max gate agreement {r['gate_agreement_max']:.4f} <= {GATE_TOL}); "
                  f"NMAE {c['0.0']['mean']:.4f}->{c['16.0']['mean']:.4f}; "
                  f"cycle={r['cycle_consistency']['mean']:.4f}",
                  file=sys.stderr)
        else:
            reason = r.get("error") or f"max gate agreement {r.get('gate_agreement_max'):.4f} > {GATE_TOL}"
            print(f"% {m}: UNVERIFIED ({reason})", file=sys.stderr)
        # Persist after every method so a long RP run is not lost if the caller's
        # output channel closes before the final LaTeX emission.
        PARTIAL_JSON.write_text(json.dumps(res, indent=2))

    # Compact LaTeX perturbation table for the main article (verified methods only).
    ver = [m for m in ("RP", "GASF", "GADF", "MTF") if res[m]["verified"]]
    out = [r"\begin{table}[!htbp]", r"\centering",
           (r"\caption{Robustness block: reconstruction NMAE as increasing Gaussian "
             r"pixel noise (std, out of 255) is added to the 8-bit image before "
             r"decoding. Values are "
             r"mean$\pm$std across " + str(len(wins)) + r" activity-balanced windows, "
             r"averaging " + str(N_REPS) + r" independent noise draws per nonzero level.}"),
           r"\label{tab:robustness}",
           r"\begin{tabular}{l" + "c" * len(SIGMAS) + "}", r"\toprule",
           r"Method & " + " & ".join(fr"$\sigma={int(s)}$" for s in SIGMAS)
           + r" \\", r"\midrule"]
    for m in ver:
        c = res[m]["nmae_curve"]
        cells = [m] + [f"{c[f'{s}']['mean']:.4f}$\\pm${c[f'{s}']['std']:.4f}" for s in SIGMAS]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("\n".join(out))

    # Error-source decomposition table (float / 8-bit / perturbed), NMAE.
    dec = [r"\begin{table}[!htbp]", r"\centering",
           (r"\caption{Reconstruction error by source (NMAE, " + str(len(wins)) + r" windows): "
             r"native float operator, clean 8-bit PNG round trip, and perturbed PNG "
             r"at $\sigma=8$. Quantisation is the dominant clean-round-trip source "
             r"for the near-original methods; MTF loss is already present in float.}"),
           r"\label{tab:error_sources}",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"Method & Float (algorithm) & 8-bit PNG & Perturbed ($\sigma=8$) \\", r"\midrule"]
    for m in ver:
        r = res[m]
        if r["nmae_float"] is None:
            fl = "--"
        else:
            float_mean = r["nmae_float"]["mean"]
            if float_mean < 1e-12:
                mantissa, exponent = f"{float_mean:.1e}".split("e")
                fl = f"${mantissa}\\times10^{{{int(exponent)}}}$"
            else:
                fl = f"{float_mean:.4f}"
        dec.append(f"{m} & {fl} & {r['nmae_curve']['0.0']['mean']:.4f} & {r['nmae_curve']['8.0']['mean']:.4f}" + r" \\")
    dec += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print()
    print("\n".join(dec))

    res["_design"] = {"sample_indices": sample_idx.tolist(), "n_windows": len(wins),
                      "n_subjects": int(len(np.unique(sample_subjects))),
                      "activity_counts": {str(k): int((sample_labels == k).sum())
                                          for k in np.unique(sample_labels)},
                      "noise_replicates": N_REPS, "seed": SEED,
                      "clipped_to_8bit": True, "rerounded_to_integer": True,
                      # In "oracle" mode the sigma=0 gate legitimately differs from the
                      # stored production reconstruction, which is metadata-free.
                      "rp_range_mode": RP_RANGE_MODE}
    OUTPUT_JSON.write_text(json.dumps(res, indent=2))
    for m in ver:
        print(f"% {m}: float={res[m]['nmae_float']['mean']}", file=sys.stderr)
    print(f"% verified methods: {ver}", file=sys.stderr)


if __name__ == "__main__":
    main()
