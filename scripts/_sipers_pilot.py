"""Sipers et al. (2017) against our geometric decoder on the SAME circular RP, and its node-threshold sweep.

The benchmark evaluates Sipers' Fourier-domain reconstruction (src/rp_sipers.py) on the circular variant
of the RP that it requires, while our geometric decoder runs on the benchmark's non-circular RP. This
script removes that difference: both decoders receive the identical circular image (T states), stored
with the production 8-bit maps, so any gap is due to the decoder.

* unsigned: both are exact only up to sign; NMAE with the oracle sign (geometry alone) and as returned.
* signed: both decode |R| and keep the one of +x, -x whose circular signed RP is closer to the image.
* sweep: NMAE (unsigned, oracle sign) of the Sipers decoder against the node threshold tol.

    python -m scripts._sipers_pilot --n 100        (needs the datasets: run in WSL)
Writes results/_stats/sipers_pilot.{json,md}.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.rp_sipers import circular_rp, sipers_decode, to_range

DATASETS = ("wisdm", "pamap2", "esc50", "ptbxl", "swarm", "stead", "emg_gestures")
TOLS = (0.01, 1e-3, 1e-4, 1e-6, 0.0)
OUT = Path("results/_stats/sipers_pilot")


def geometric_decode(D):
    from src.rp_decoders import decode_channel

    x = decode_channel(np.abs(D), (1.0, 0.0), source="direct", dims=2, recover="overlap", signed=False,
                       orientation_method="closed_form")
    return np.asarray(x)[:D.shape[0]]                           # N states give N+1 samples, the last wraps


def nmae(x, y):
    return float(np.mean(np.abs(x - y)) / np.ptp(x))


def pick_sign(R_obs, xh, mm):
    best = None
    for cand in (to_range(xh, mm), to_range(-xh, mm)):
        Rc = circular_rp(cand)
        e = np.linalg.norm(Rc / np.linalg.norm(Rc) - R_obs / np.linalg.norm(R_obs))
        best = (e, cand) if best is None or e < best[0] else best
    return best[1]


def storage(R, signed):
    from src.representations import _rp_from_uint8, _rp_to_uint8, _rp_unsigned_from_uint8, _rp_unsigned_to_uint8

    return _rp_from_uint8(_rp_to_uint8(R)) if signed else _rp_unsigned_from_uint8(_rp_unsigned_to_uint8(R))


def run_series(x):
    mm = (float(x.max()), float(x.min()))
    R = circular_rp(x)
    out = {}
    for cond in ("float", "png8"):
        for enc in ("unsigned", "signed"):
            img = (np.abs(R) if enc == "unsigned" else R) if cond == "float" else storage(R, enc == "signed")
            D = 0.5 * (np.abs(img) + np.abs(img).T)
            for name, dec in (("sipers", sipers_decode), ("geometric", geometric_decode)):
                t0 = time.perf_counter()
                xh = dec(D)
                ms = 1e3 * (time.perf_counter() - t0)
                if enc == "unsigned":
                    out[f"{cond}/{enc}/{name}"] = {
                        "nmae": nmae(x, to_range(xh, mm)), "ms": ms,
                        "nmae_oracle": min(nmae(x, to_range(xh, mm)), nmae(x, to_range(-xh, mm)))}
                else:
                    out[f"{cond}/{enc}/{name}"] = {"nmae": nmae(x, pick_sign(img, xh, mm)), "ms": ms}
        D = np.abs(R)
        for t in TOLS:
            M = D if cond == "float" else storage(D, False)
            xh = sipers_decode(M, tol=t)
            out[f"{cond}/sweep/{t:g}"] = {"nmae_oracle": min(nmae(x, to_range(xh, mm)), nmae(x, to_range(-xh, mm)))}
    return out


def self_check():
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = np.cumsum(rng.normal(size=128))
        xh = sipers_decode(np.abs(circular_rp(x)), tol=0.0)
        mm = (x.max(), x.min())
        e = min(nmae(x, to_range(xh, mm)), nmae(x, to_range(-xh, mm)))
        assert e < 1e-9, f"Sipers decoder not exact in float: {e}"
    from src.recurrence_plots import varRP2
    x = rng.normal(size=64)
    assert np.allclose(circular_rp(x)[:63, :63], varRP2(x, TIME_STEPS=64))   # same encoding off the wrap
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="windows per dataset (T = 128)")
    args = ap.parse_args()
    self_check()
    from scripts._benchmark import DATASETS as CFG
    from src.datasets import get, make_windows

    summary = {"_design": {"T": 128, "windows": args.n, "M": 2, "delay": 1, "tols": TOLS, "doc": __doc__.split("\n\n")[1]}}
    for ds in DATASETS:
        idx = np.load(Path("results/_benchmark") / f"{ds}_L128.npz")["window_index"][:args.n]
        X = make_windows(get(ds, **CFG[ds]), 128, dtype=np.float32).X
        rows = [run_series(X[i][:, c].astype(float)) for i in idx for c in range(X.shape[2]) if np.ptp(X[i][:, c]) > 0]
        summary[ds] = {"series": len(rows)} | {k: {m: float(np.mean([r[k][m] for r in rows])) for m in rows[0][k]}
                                               for k in rows[0]}
        s = summary[ds]
        print(f"{ds:13} {len(rows):5} series | 8-bit oracle: Sipers {s['png8/unsigned/sipers']['nmae_oracle']:.4f} "
              f"geometric {s['png8/unsigned/geometric']['nmae_oracle']:.4f} | 8-bit signed: Sipers "
              f"{s['png8/signed/sipers']['nmae']:.4f} geometric {s['png8/signed/geometric']['nmae']:.4f}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    lines = ["# Sipers et al. (2017) vs geometric decoder on the same circular RP (T = 128)", "",
             "Mean NMAE. Unsigned: oracle sign [as returned]. Signed: sign picked from the image.", "",
             "| Dataset | Cond. | Unsigned, Sipers | Unsigned, geometric | Signed, Sipers | Signed, geometric | "
             + " | ".join(f"tol {t:g}" for t in TOLS) + " |", "|---|---|---|---|---|---|" + "---|" * len(TOLS)]
    for ds in DATASETS:
        for cond in ("float", "png8"):
            s = summary[ds]
            u_s, u_g = s[f"{cond}/unsigned/sipers"], s[f"{cond}/unsigned/geometric"]
            lines.append(f"| {ds} | {cond} | {u_s['nmae_oracle']:.4f} [{u_s['nmae']:.3f}] | {u_g['nmae_oracle']:.4f} "
                         f"[{u_g['nmae']:.3f}] | {s[f'{cond}/signed/sipers']['nmae']:.4f} | "
                         f"{s[f'{cond}/signed/geometric']['nmae']:.4f} | "
                         + " | ".join(f"{s[f'{cond}/sweep/{t:g}']['nmae_oracle']:.4f}" for t in TOLS) + " |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
