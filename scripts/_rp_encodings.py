"""RP encoding x decoder grid: what the encoding adds versus what the decoder adds.

Encodings of the same channel series (adjacent states s_i = (x_i, x_{i+1})):
  binary    conventional RP: 1 if ||s_i - s_j|| < eps (recurrence rate 0.75), stored losslessly as 0/255
  unsigned  continuous distances |R| = ||s_i - s_j||, 8-bit image of |R| / max|R|
  signed    our signed RP R = sigma_ij ||s_i - s_j||, 8-bit with the production normalisation

Decoders:
  hirata    Hirata et al. RPR (graph of the binary RP, Eq. 2 weights, shortest paths, MDS) + our 1-D post-processing
  previous  our previous decoder (src/rp_legacy.py): Jaccard graph, Dijkstra, SMACOF 1-D, signed check
  direct2d  our decoder: continuous distances (geodesics of the binary RP when only that exists), 2-D MDS,
            closed-form orientation and overlap recovery; the sign picks the reflection when available

Hirata thresholds whatever it receives, so it only has a binary column. On the
binary RP "previous" uses the binary graph directly (its own quantile threshold
would make every pair recurrent). Without a sign the reflection cannot be
resolved: those cells report the reconstruction as returned and, separately, an
evaluation-only oracle that picks the better of x and its reflection.

    python -m scripts._rp_encodings --lengths 64,128,256 --n 300 --workers 20
Writes results/_stats/rp_encodings.json and rp_encodings.md.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CELLS = (("binary", "hirata"), ("binary", "previous"), ("binary", "direct2d"),
         ("unsigned", "previous"), ("unsigned", "direct2d"),
         ("signed", "previous"), ("signed", "direct2d"))
CONDS = ("float", "png8")
FIELDS = ("nmae", "nmae_oracle", "orientation_ok", "ms")


def _rescale(v, mm):
    lo, hi = float(v.min()), float(v.max())
    return np.full_like(v, 0.5 * (mm[0] + mm[1])) if hi - lo < 1e-300 else mm[1] + (v - lo) / (hi - lo) * (mm[0] - mm[1])


def decode(enc, dec, M, mm):
    from src.rp_decoders import decode_channel
    from src.rp_hirata import geodesics, states, to_series
    from src.rp_legacy import decode_channel as previous
    from src.rp_legacy import project_1d

    n = M.shape[0] + 1
    if dec == "hirata":
        return to_series(states(M)[0], mm, "as_is")
    if dec == "previous":
        if enc == "binary":
            proj = project_1d(geodesics(M)[0], "smacof")
            return _rescale(np.append(proj, proj.mean()), mm)
        return previous(M, mm, signed=(enc == "signed"), scale_invariant=True, n_out=n)[0]
    if enc == "binary":
        return decode_channel(M, mm, source="binary_geodesic", dims=2, recover="overlap", signed=False)
    return decode_channel(M, mm, source="direct", dims=2, recover="overlap", signed=(enc == "signed"),
                          scale_invariant=True)


def series_row(x):
    from src.recurrence_plots import NormalizeMatrix, varRP2
    from src.rp_hirata import binary_rp, eps_for_rate

    out = np.full((len(CONDS), len(CELLS), len(FIELDS)), np.nan)
    if x.max() - x.min() < 1e-12:
        return out
    mm, rng = (float(x.max()), float(x.min())), float(x.max() - x.min())
    R = varRP2(x, TIME_STEPS=len(x))
    D = np.abs(R)
    mats = {"float": {"binary": binary_rp(D, eps_for_rate(D, 0.75)), "unsigned": D, "signed": R}}
    png_s = np.round(NormalizeMatrix(R) * 255)
    mats["png8"] = {"binary": mats["float"]["binary"],                 # 0/255 storage is lossless
                    "unsigned": np.round(D / D.max() * 255),
                    "signed": png_s - np.median(np.diag(png_s))}
    for ci, cond in enumerate(CONDS):
        for k, (enc, dec) in enumerate(CELLS):
            t0 = time.perf_counter()
            try:
                r = np.asarray(decode(enc, dec, mats[cond][enc], mm), dtype=float)[:len(x)]
            except Exception:
                continue                                          # stays NaN: counted as a failure
            ms = 1000 * (time.perf_counter() - t0)
            refl = mm[0] + mm[1] - r
            e, e_refl = np.abs(r - x).mean() / rng, np.abs(refl - x).mean() / rng
            out[ci, k] = [e, min(e, e_refl), float(e <= e_refl), ms]
    return out


def main():
    from scripts._benchmark import DATASETS
    from src.datasets import get, make_windows

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--lengths", default="64,128,256")
    ap.add_argument("--n", type=int, default=300, help="windows per dataset and length (seed 0)")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    out = Path("results/_stats")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rp_encodings.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary["_design"] = {"cells": CELLS, "conditions": CONDS, "fields": FIELDS, "n_windows": args.n, "seed": 0,
                          "datasets": DATASETS, "doc": __doc__.split("\n\n")[1].strip()}
    for ds in args.datasets.split(","):
        adapter = get(ds, **DATASETS[ds])
        for L in map(int, args.lengths.split(",")):
            w = make_windows(adapter, L, dtype=np.float32)
            idx = np.sort(np.random.default_rng(0).choice(len(w.X), size=min(args.n, len(w.X)), replace=False))
            series = [w.X[i, :, c].astype(float) for i in idx for c in range(w.X.shape[2])]
            t0 = time.perf_counter()
            with ProcessPoolExecutor(args.workers) as ex:
                res = np.stack(list(ex.map(series_row, series, chunksize=4)))      # (S, conds, cells, fields)
            key = f"{ds}/L{L}"
            entry = {"series": len(series), "wall_s": time.perf_counter() - t0}
            for ci, cond in enumerate(CONDS):
                for k, (enc, dec) in enumerate(CELLS):
                    v = res[:, ci, k]
                    ok = np.isfinite(v[:, 0])
                    entry[f"{cond}/{enc}/{dec}"] = {"n": int(ok.sum()), "nmae_mean": float(np.mean(v[ok, 0])),
                                                    "nmae_median": float(np.median(v[ok, 0])),
                                                    "nmae_oracle_mean": float(np.mean(v[ok, 1])),
                                                    "orientation_ok": float(np.mean(v[ok, 2])),
                                                    "ms_mean": float(np.mean(v[ok, 3]))}
            summary[key] = entry
            path.write_text(json.dumps(summary, indent=1) + "\n")
            print(f"{key:20} {len(series)} series, {entry['wall_s']:.0f} s | 8-bit NMAE " + "  ".join(
                f"{e[:3]}/{d[:4]} {entry[f'png8/{e}/{d}']['nmae_mean']:.4f}" for e, d in CELLS), flush=True)
    write_md(summary, out / "rp_encodings.md")


def write_md(summary, path):
    rows = ["# RP: encoding x decoder (mean NMAE; oracle orientation in brackets for unsigned encodings)", ""]
    for cond in CONDS:
        rows += [f"## {cond}", "", "| Dataset | L | " + " | ".join(f"{e} / {d}" for e, d in CELLS) + " |",
                 "|---|---|" + "---|" * len(CELLS)]
        for key, e in summary.items():
            if key.startswith("_"):
                continue
            ds, L = key.split("/")
            cells = []
            for enc, dec in CELLS:
                s = e[f"{cond}/{enc}/{dec}"]
                cells.append(f"{s['nmae_mean']:.4f}" + (f" [{s['nmae_oracle_mean']:.4f}]" if enc != "signed" else ""))
            rows.append(f"| {ds} | {L[1:]} | " + " | ".join(cells) + " |")
        rows.append("")
    path.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
