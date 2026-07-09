"""Publication figure: RP reconstruction error vs the recurrence-rate parameter rho.

Renders the sensitivity curve computed by ``scripts.search_rp_rho``: MAE and RMSE
(left axis) and Pearson correlation (right axis) against rho, shading the flat MAE
plateau and marking the chosen rho and the original rho=0.75. Reads the grid CSV
so the figure regenerates from data (no hard-coded numbers).

    python -m scripts.plot_rho_sensitivity \
        --csv results/_rho_search/rho_search_loto-fold-0-3f.csv \
        --out doc/images/rp_rho_sensitivity.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="grid CSV from search_rp_rho")
    ap.add_argument("--out", required=True, help="output image path (.png/.pdf)")
    ap.add_argument("--chosen", type=float, default=0.60, help="selected operating rho")
    ap.add_argument("--baseline", type=float, default=0.75, help="original rho")
    ap.add_argument("--plateau", default="0.58,0.61", help="lo,hi rho of the MAE plateau to shade")
    args = ap.parse_args()

    rows = []
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["rho"]), float(r["mae"]), float(r["rmse"]), float(r["pearson"])))
    rows.sort(key=lambda t: t[0])
    rho, mae, rmse, pear = (list(c) for c in zip(*rows))
    lo, hi = (float(x) for x in args.plateau.split(","))

    fig, axL = plt.subplots(figsize=(6.4, 4.2))
    axR = axL.twinx()

    axL.axvspan(lo, hi, color="0.88", zorder=0, label="MAE plateau")

    lm, = axL.plot(rho, mae, "o-", color="#1f77b4", label="MAE")
    lr, = axL.plot(rho, rmse, "s--", color="#ff7f0e", label="RMSE")
    lp, = axR.plot(rho, pear, "^:", color="#2ca02c", label="Pearson $r$")

    vc = axL.axvline(args.chosen, color="black", lw=1.2, ls="-")
    vb = axL.axvline(args.baseline, color="0.5", lw=1.2, ls="--")

    axL.set_xlabel(r"recurrence rate $\rho$")
    axL.set_ylabel("MAE / RMSE (raw signal units)")
    axR.set_ylabel(r"Pearson correlation $r$")
    axL.set_xticks([round(v, 2) for v in rho])
    axL.tick_params(axis="x", labelsize=8)

    handles = [lm, lr, lp, vc, vb]
    labels = ["MAE", "RMSE", r"Pearson $r$",
              rf"chosen $\rho={args.chosen:g}$", rf"original $\rho={args.baseline:g}$"]
    axL.legend(handles, labels, loc="upper center", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
