"""Cost of each representation against window length (roadmap #90).

Real WISDM series (first channel of seeded windows) at 64 to 1024 samples. Per
representation and length: median single-thread time to encode, store as 8 bits
and decode, peak traced memory of one encode+decode, and the log-log slope of
time against length (empirical complexity exponent). Cost against the number of
channels is linear by construction, since every channel is encoded and decoded
on its own. Run it on an idle machine: timings under load are not comparable.

    python -m scripts._cost_scaling --n 20
Writes results/_stats/cost_scaling.json and .md.
"""
import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

LENGTHS = (64, 128, 256, 512, 1024)
OUT = Path("results/_stats")


def run(rep, x, mm):
    return rep.decode(rep.from_uint8(rep.to_uint8(rep.encode(x, mm))), mm)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    from threadpoolctl import threadpool_limits

    from scripts._benchmark import LABELS, REPS
    from src.datasets import get, make_windows
    from src.representations import REPRESENTATIONS

    adapter = get("wisdm")
    report = {"_design": {"lengths": LENGTHS, "series_per_length": args.n, "dataset": "wisdm", "channel": 0,
                          "doc": __doc__.split("\n\n")[1].strip()}}
    with threadpool_limits(1):
        for L in LENGTHS:
            w = make_windows(adapter, L)
            idx = np.random.default_rng(0).choice(len(w.X), size=min(args.n, len(w.X)), replace=False)
            series = [w.X[i][:, 0].astype(float) for i in idx if np.ptp(w.X[i][:, 0]) > 0]
            for name in REPS:
                rep = REPRESENTATIONS[name]
                times = []
                for x in series:
                    mm = (float(x.max()), float(x.min()))
                    t0 = time.perf_counter()
                    run(rep, x, mm)
                    times.append((time.perf_counter() - t0) * 1e3)
                tracemalloc.start()
                run(rep, series[0], (float(series[0].max()), float(series[0].min())))
                peak = tracemalloc.get_traced_memory()[1] / 2**20
                tracemalloc.stop()
                report.setdefault(name, {})[str(L)] = {"ms_median": float(np.median(times)), "peak_mb": peak,
                                                       "series": len(series)}
                print(f"L{L:<5} {name:17} {np.median(times):9.1f} ms  {peak:8.1f} MB", flush=True)
    for name in REPS:
        ms = [report[name][str(L)]["ms_median"] for L in LENGTHS]
        report[name]["time_exponent"] = float(np.polyfit(np.log(LENGTHS), np.log(ms), 1)[0])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_scaling.json").write_text(json.dumps(report, indent=1) + "\n")
    rows = ["# Cost against window length (WISDM, one channel, single thread)", "",
            "Median ms to encode + store 8 bits + decode; peak traced MB; exponent = log-log slope of time.", "",
            "| Technique | " + " | ".join(f"L = {L}" for L in LENGTHS) + " | Exponent |",
            "|---|" + "---|" * (len(LENGTHS) + 1)]
    for name in REPS:
        cells = [f"{report[name][str(L)]['ms_median']:.1f} ms / {report[name][str(L)]['peak_mb']:.0f} MB" for L in LENGTHS]
        rows.append(f"| {LABELS[name]} | " + " | ".join(cells) + f" | {report[name]['time_exponent']:.2f} |")
    (OUT / "cost_scaling.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
