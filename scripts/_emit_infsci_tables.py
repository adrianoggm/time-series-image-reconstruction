"""Tables and numbers of the Information Sciences manuscript, generated from the stored artefacts (#96).

Reads results/_benchmark/{summary,stats}.json, results/_stats/{rp_encodings,
rp_conditions,cost_scaling}.json and results/_classification{,_esc10}/summary.json,
and writes doc/infsci/generated/: one .tex per table and numbers.tex with the
figures quoted in the text as macros, so no printed number is typed by hand.

    python -m scripts._emit_infsci_tables
"""
import json
from pathlib import Path

import numpy as np

OUT = Path("doc/infsci/generated")
REPS = ("RP_HIRATA", "RP_HIRATA_SIGNED", "RP_UNSIGNED", "RP", "GASF", "GADF", "MTF", "XIRP")
SHORT = {"RP_HIRATA": "Hirata\\,/\\,uns.", "RP_HIRATA_SIGNED": "Hirata\\,/\\,sig.", "RP_UNSIGNED": "Ours\\,/\\,uns.",
         "RP": "Ours\\,/\\,sig.", "GASF": "GASF", "GADF": "GADF", "MTF": "MTF", "XIRP": "XIRP"}
NAMES = {"wisdm": "WISDM", "pamap2": "PAMAP2", "esc50": "ESC-50", "ptbxl": "PTB-XL", "swarm": "Swarm",
         "stead": "STEAD", "emg_gestures": "EMG"}
CLFS = ("knn", "svm", "rf", "lstm", "gru", "cnn")
MIN_WINDOWS = 200


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt(v, d=4):
    return f"{v:.{d}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bench, stats = load("results/_benchmark/summary.json"), load("results/_benchmark/stats.json")
    enc, cond = load("results/_stats/rp_encodings.json"), load("results/_stats/rp_conditions.json")
    cost = load("results/_stats/cost_scaling.json")
    cls, esc = load("results/_classification/summary.json"), load("results/_classification_esc10/summary.json")
    blocks = {k: v for k, v in bench.items() if not k.startswith("_") and v["windows_used"] >= MIN_WINDOWS}
    nums = {}

    # Table: 8-bit NMAE of every decoder, L = 128.
    rows = [r"\begin{tabular}{@{}l" + "r" * len(REPS) + "@{}}", r"\toprule",
            "Dataset & " + " & ".join(SHORT[r] for r in REPS) + r" \\", r"\midrule"]
    for ds, name in NAMES.items():
        s = bench[f"{ds}/L128"]
        rows.append(f"{name} & " + " & ".join(fmt(s[f'{r}/png8']['nmae']['mean']) for r in REPS) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_fidelity.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Exactness in floating point and the 8-bit range of each exact decoder.
    for r, macro in (("RP", "RP"), ("GASF", "GASF"), ("GADF", "GADF"), ("XIRP", "XIRP")):
        nums[f"floatMax{macro}"] = f"{max(v[f'{r}/float']['nmae']['median'] for v in blocks.values()):.1e}"
    # the one degenerate ESC-50 window (126 zeros, amplitude ~3e-9) lifts that block's mean; the medians do not move
    nums["floatBlocksExactRP"] = str(sum(v["RP/float"]["nmae"]["mean"] < 1e-12 for v in blocks.values()))
    rp8 = [v["RP/png8"]["nmae"]["mean"] for k, v in blocks.items() if not k.startswith("swarm")]
    nums["rpEightMin"], nums["rpEightMax"] = fmt(min(rp8)), fmt(max(rp8))
    sw = [v["RP/png8"]["nmae"]["mean"] for k, v in blocks.items() if k.startswith("swarm")]
    nums["rpEightSwarmMin"], nums["rpEightSwarmMax"] = fmt(min(sw), 3), fmt(max(sw), 3)
    nums["benchSeries"] = f"{sum(v['RP/png8']['series'] for v in blocks.values()):,}".replace(",", "\\,")
    nums["benchBlocks"] = str(len(blocks))
    for r in REPS[:3] + ("MTF",):
        nums[f"macroDiff{r.replace('_', '')}"] = fmt(stats["_macro"]["L128"][r], 3)

    # Table: RP decoder x encoding, 8-bit, L = 128, with the orientation oracle on the unsigned RP.
    rows = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
            r"& \multicolumn{2}{c}{Hirata} & \multicolumn{3}{c}{Geometric decoder (ours)} \\",
            r"\cmidrule(lr){2-3}\cmidrule(l){4-6}",
            r"Dataset & Unsigned & Signed & Unsigned & Unsigned, oracle & Signed \\", r"\midrule"]
    for ds, name in NAMES.items():
        s, e = bench[f"{ds}/L128"], enc[f"{ds}/L128"]["png8/unsigned/direct2d"]
        rows.append(f"{name} & {fmt(s['RP_HIRATA/png8']['nmae']['mean'])} & {fmt(s['RP_HIRATA_SIGNED/png8']['nmae']['mean'])}"
                    f" & {fmt(s['RP_UNSIGNED/png8']['nmae']['mean'])} & {fmt(e['nmae_oracle_mean'])}"
                    f" & {fmt(s['RP/png8']['nmae']['mean'])} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_rp.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    bin_float = [enc[k][f"float/binary/{d}"]["nmae_oracle_mean"] for k in enc if not k.startswith("_")
                 for d in ("hirata", "direct2d")]
    nums["binaryOracleMin"], nums["binaryOracleMax"] = fmt(min(bin_float), 2), fmt(max(bin_float), 2)

    # Conditioning of the RP states.
    bands = cond["_pooled_bands"]
    good = [v for e, v in bands.items() if e.startswith("[0.1") or e.startswith("[0.3")]
    nums["condGoodSeries"] = f"{sum(v['series'] for v in good):,}".replace(",", "\\,")
    nums["condGoodBad"] = str(sum(v["bad"] for v in good))
    nums["condWorstRate"] = f"{100 * bands['[0, 0.01)']['bad_rate']:.0f}"

    # Table: cost against length.
    Ls = ("64", "256", "1024")
    rows = [r"\begin{tabular}{@{}l" + "r" * (len(Ls) + 2) + "@{}}", r"\toprule",
            "Decoder & " + " & ".join(f"$T={L}$" for L in Ls) + r" & Exponent & MB at 1024 \\", r"\midrule"]
    for r in REPS:
        c = cost[r]
        rows.append(f"{SHORT[r]} & " + " & ".join(f"{c[L]['ms_median']:.1f}" for L in Ls)
                    + f" & {c['time_exponent']:.2f} & {c['1024']['peak_mb']:.0f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_cost.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for L, name in (("256", "speedupMid"), ("1024", "speedupLong")):      # macro names cannot hold digits
        nums[name] = f"{cost['RP_HIRATA_SIGNED'][L]['ms_median'] / cost['RP'][L]['ms_median']:.1f}"

    # Table: classification utility, drop from the original.
    def drops(summary, key, skip):
        out = {r: [] for r in REPS}
        eq = {r: 0 for r in REPS}
        n = n_eq = 0
        for k, b in summary.items():
            if k.startswith("_") or b["windows"] < MIN_WINDOWS or skip(k):
                continue
            m = "macro_f1" if b.get("multilabel") else key
            for c in CLFS:
                n += 1
                n_eq += not b.get("multilabel")
                for r in REPS:
                    out[r].append(b[c]["original"][m] - b[c][r][m])
                    eq[r] += bool(b[c][r].get("equivalent_0.01"))
        return out, eq, n, n_eq

    main_d, main_eq, n_main, n_main_eq = drops(cls, "accuracy", lambda k: k.startswith("esc50"))
    esc_d, esc_eq, n_esc, _ = drops(esc, "clip_accuracy", lambda k: False)
    rows = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r"& \multicolumn{4}{c}{Six datasets, $T\in\{64,128,256\}$} & \multicolumn{2}{c}{ESC-10, per clip} \\",
            r"\cmidrule(lr){2-5}\cmidrule(l){6-7}",
            r"Decoder & Mean drop & Max drop & $|\Delta|\le0.01$ & Equivalent & Mean drop & Equivalent \\", r"\midrule"]
    for r in REPS:
        d, e = np.array(main_d[r]), np.array(esc_d[r])
        rows.append(f"{SHORT[r]} & {d.mean():.3f} & {d.max():.3f} & {100 * np.mean(np.abs(d) <= 0.01):.0f}\\% & "
                    f"{main_eq[r]}/{n_main_eq} & {e.mean():.3f} & {esc_eq[r]}/{n_esc} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_utility.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    nums["clsCombos"] = str(n_main)
    nums["clsCombosEq"] = str(n_main_eq)
    nums["clsWindows"] = f"{sum(b['windows'] for k, b in cls.items() if not k.startswith('_')):,}".replace(",", "\\,")
    nums["escWindows"] = f"{sum(b['windows'] for k, b in esc.items() if not k.startswith('_')):,}".replace(",", "\\,")
    nums["escCombos"] = str(n_esc)

    # Robustness (T = 128, mean over datasets) and sampling rate (PTB-XL, T = 128).
    rob, sr = load("results/_stats/robustness_multi.json"), load("results/_stats/sampling_rate.json")
    rb = [k for k in rob if k.endswith("/L128")]

    def noisy(r, s):
        return float(np.mean([rob[k][r]["nmae"][s]["mean"] for k in rb]))

    for r, name in (("RP", "RP"), ("GASF", "GASF"), ("GADF", "GADF"), ("XIRP", "XIRP")):
        nums[f"robHigh{name}"] = fmt(noisy(r, "16.0"), 3)
    nums["robDatasetsRPBetter"] = str(sum(rob[k]["RP"]["nmae"]["16.0"]["mean"] < min(
        rob[k]["GASF"]["nmae"]["16.0"]["mean"], rob[k]["GADF"]["nmae"]["16.0"]["mean"]) for k in rb))
    nums["robDatasets"] = str(len(rb))
    lossy = [abs(noisy(r, "16.0") - noisy(r, "0.0")) for r in ("RP_HIRATA", "RP_HIRATA_SIGNED", "RP_UNSIGNED", "MTF")]
    nums["robLossyShift"] = fmt(max(lossy), 3)
    for rate in ("500", "250", "100", "50"):
        b = sr[f"{rate}Hz/L128"]
        word = {"500": "Fivehundred", "250": "Twofifty", "100": "Hundred", "50": "Fifty"}[rate]
        nums[f"srRP{word}"] = fmt(b["RP/png8"]["nmae"]["mean"])
        nums[f"srCol{word}"] = fmt(b["collinearity_median"], 2)
        nums[f"srShare{word}"] = f"{100 * b['share_below_0.1']:.0f}"
    gas = [sr[f"{r}Hz/L128"][f"{x}/png8"]["nmae"]["mean"] for r in ("500", "250", "100", "50") for x in ("GASF", "GADF")]
    nums["srGafMin"], nums["srGafMax"] = fmt(min(gas)), fmt(max(gas))
    nat, res = sr["100_nativeHz/L128"]["RP/png8"]["nmae"]["mean"], sr["100Hz/L128"]["RP/png8"]["nmae"]["mean"]
    nums["srNativeRP"], nums["srResampledRP"] = fmt(nat), fmt(res)

    (OUT / "numbers.tex").write_text("% generated by scripts/_emit_infsci_tables.py; do not edit\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items()), encoding="utf-8")
    print("\n".join(f"{k} = {v}" for k, v in nums.items()))


if __name__ == "__main__":
    main()
