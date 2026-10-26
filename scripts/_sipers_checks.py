"""Methodological checks of the Sipers comparison (encoder vs decoder, threshold, exactness, same pixels).

On the first 100 benchmark windows (T = 128, every channel) of four datasets:

* encoder vs decoder, 8-bit signed RP: geometric decoder on the benchmark (non-circular) RP, geometric
  decoder on the circular RP, Sipers on the circular RP. The first two differ only in the encoding, the
  last two only in the decoder;
* conditioning sigma2/sigma1 of the centred states without and with the wrap state (x_T, x_1);
* floating point: Sipers' error with tol = 0 (method alone: numerical error) and tol = 1e-4 (truncation),
  and at 8 bits with tol = 0 (storage alone) and tol = 1e-4 (both);
* the two decoders of the paired comparison receive byte-identical images (asserted per series), and the
  sign is chosen from the image only.

    python -m scripts._sipers_checks        (needs the datasets: run in WSL)
Writes results/_stats/sipers_checks.json.
"""
import json
from pathlib import Path

import numpy as np

from scripts._sipers_pilot import geometric_decode, nmae, pick_sign, storage
from src.rp_sipers import TOL, circular_rp, sipers_decode, to_range

DATASETS = ("wisdm", "ptbxl", "swarm", "stead")
OUT = Path("results/_stats/sipers_checks.json")


def ratio(states):
    sv = np.linalg.svd(states - states.mean(0), compute_uv=False)
    return float(sv[1] / sv[0]) if sv[0] > 0 else np.nan


def series_checks(x):
    from src.representations import REPRESENTATIONS

    mm = (float(x.max()), float(x.min()))
    rep = REPRESENTATIONS["RP"]
    bench = float(nmae(x, rep.decode(rep.from_uint8(rep.to_uint8(rep.encode(x, mm))), mm)))
    R = circular_rp(x)
    img = storage(R, signed=True)                                # one pixel matrix for both decoders
    D = 0.5 * (np.abs(img) + np.abs(img).T)
    h = hash(img.tobytes())
    geo = geometric_decode(D)
    sip = sipers_decode(D)
    assert hash(img.tobytes()) == h                              # neither decoder modified the image
    out = {"geo_noncircular_8bit": bench,
           "geo_circular_8bit": nmae(x, pick_sign(img, geo, mm)),
           "sipers_circular_8bit": nmae(x, pick_sign(img, sip, mm)),
           "sipers_circular_8bit_tol0": nmae(x, pick_sign(img, sipers_decode(D, tol=0.0), mm))}
    Df = np.abs(R)
    for tol, key in ((0.0, "float_tol0"), (TOL, "float_tol")):
        xh = sipers_decode(Df, tol=tol)
        out[f"sipers_{key}"] = min(nmae(x, to_range(xh, mm)), nmae(x, to_range(-xh, mm)))
    s = np.column_stack([x[:-1], x[1:]])
    out["ratio_noncircular"] = ratio(s)
    out["ratio_circular"] = ratio(np.vstack([s, [x[-1], x[0]]]))
    return out


def main():
    from scripts._benchmark import DATASETS as CFG
    from src.datasets import get, make_windows

    report = {"_design": {"T": 128, "windows": 100, "tol": TOL, "doc": __doc__.split("\n\n")[1]}}
    for ds in DATASETS:
        idx = np.load(Path("results/_benchmark") / f"{ds}_L128.npz")["window_index"][:100]
        X = make_windows(get(ds, **CFG[ds]), 128, dtype=np.float32).X
        rows = [series_checks(X[i][:, c].astype(float)) for i in idx for c in range(X.shape[2])
                if np.ptp(X[i][:, c]) > 0]
        report[ds] = {"series": len(rows)}
        for k in rows[0]:
            v = np.array([r[k] for r in rows])
            report[ds][k] = {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)), "max": float(np.nanmax(v))}
        print(ds, {k: round(v["mean"], 5) for k, v in report[ds].items() if isinstance(v, dict)}, flush=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
