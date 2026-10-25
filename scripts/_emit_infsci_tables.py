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
REPS = ("RP_HIRATA", "RP_HIRATA_SIGNED", "RP_SIPERS", "RP_SIPERS_SIGNED", "RP_UNSIGNED", "RP", "GASF", "GADF", "MTF",
        "XIRP")
SHORT = {"RP_HIRATA": "Hirata\\,/\\,uns.", "RP_HIRATA_SIGNED": "Hirata\\,/\\,sig.", "RP_UNSIGNED": "Ours\\,/\\,uns.",
         "RP": "Ours\\,/\\,sig.", "GASF": "GASF", "GADF": "GADF", "MTF": "MTF", "XIRP": "XIRP",
         "RP_SIPERS": "Sipers\\,/\\,uns.", "RP_SIPERS_SIGNED": "Sipers\\,/\\,sig."}
EXACT = ("RP", "GASF", "GADF", "XIRP")
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

    # Table: 8-bit NMAE of every decoder, real reconstructions only (no orientation oracle); L = 128 in the
    # manuscript, the other lengths in the supplement.
    cols = ("RP_HIRATA", "RP_HIRATA_SIGNED", "RP_SIPERS", "RP_SIPERS_SIGNED", "RP_UNSIGNED", "RP",
            "GASF", "GADF", "MTF", "XIRP")
    for L in ("64", "128", "256"):
        rows = [r"\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", r"\toprule",
                r"& \multicolumn{2}{c}{Hirata} & \multicolumn{2}{c}{Sipers} & \multicolumn{2}{c}{Geometric (ours)}"
                r" & & & & \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
                r"Dataset & Uns. & Sig. & Uns. & Sig. & Uns. & Sig. & GASF & GADF & MTF & XIRP \\",
                r"\midrule"]
        for ds, name in NAMES.items():
            k = f"{ds}/L{L}"
            if k not in blocks:
                rows.append(f"{name} & " + " & ".join(["--"] * len(cols)) + r" \\")
                continue
            s = bench[k]
            cells = [fmt(s[f"{r}/png8"]["nmae"]["mean"]) if f"{r}/png8" in s else "--" for r in cols]
            rows.append(f"{name} & " + " & ".join(cells) + r" \\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        (OUT / ("tab_fidelity.tex" if L == "128" else f"tab_fidelity_L{L}.tex")).write_text(
            "\n".join(rows) + "\n", encoding="utf-8")

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
    for r in ("RP_HIRATA", "RP_HIRATA_SIGNED", "RP_UNSIGNED", "MTF", "RP_SIPERS", "RP_SIPERS_SIGNED"):
        if r in stats["_macro"]["L128"]:
            nums[f"macroDiff{r.replace('_', '')}"] = fmt(stats["_macro"]["L128"][r], 4 if "SIPERS" in r else 3)
    # Sipers et al. on the circular RP: exactness in floating point and 8-bit range with the sign
    if all("RP_SIPERS_SIGNED/png8" in v for v in blocks.values()):
        nums["floatMaxSIPERS"] = f"{max(v['RP_SIPERS_SIGNED/float']['nmae']['median'] for v in blocks.values()):.1e}"
        sp = [v["RP_SIPERS_SIGNED/png8"]["nmae"]["mean"] for v in blocks.values()]
        nums["sipEightMin"], nums["sipEightMax"] = fmt(min(sp)), fmt(max(sp))
        nums["sipBetterBlocks"] = str(sum(v["RP_SIPERS_SIGNED/png8"]["nmae"]["mean"] < v["RP/png8"]["nmae"]["mean"]
                                          for v in blocks.values()))

    bin_float = [enc[k][f"float/binary/{d}"]["nmae_oracle_mean"] for k in enc if not k.startswith("_")
                 for d in ("hirata", "direct2d")]
    nums["binaryOracleMin"], nums["binaryOracleMax"] = fmt(min(bin_float), 2), fmt(max(bin_float), 2)
    # same binary input, two decoders: how far apart they stay
    nums["binaryGapMax"] = fmt(max(abs(enc[k]["float/binary/hirata"]["nmae_oracle_mean"]
                                       - enc[k]["float/binary/direct2d"]["nmae_oracle_mean"])
                                   for k in enc if not k.startswith("_")), 2)

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
    for r in (r for r in REPS if r in cost):
        c = cost[r]
        rows.append(f"{SHORT[r]} & " + " & ".join(f"{c[L]['ms_median']:.1f}" for L in Ls)
                    + f" & {c['time_exponent']:.2f} & {c['1024']['peak_mb']:.0f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_cost.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for L, name in (("256", "speedupMid"), ("1024", "speedupLong")):      # macro names cannot hold digits
        nums[name] = f"{cost['RP_HIRATA_SIGNED'][L]['ms_median'] / cost['RP'][L]['ms_median']:.1f}"

    # Table: classification utility, drop from the original. Techniques added later (--only) carry the original
    # score of the refit they were evaluated with (ref_original); the drop is taken against that one.
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
                    if r in b[c]:
                        out[r].append(b[c][r].get("ref_original", b[c]["original"])[m] - b[c][r][m])
                        eq[r] += bool(b[c][r].get("equivalent_0.01"))
        return out, eq, n, n_eq

    main_d, main_eq, n_main, n_main_eq = drops(cls, "accuracy", lambda k: k.startswith("esc50"))
    esc_d, esc_eq, n_esc, _ = drops(esc, "clip_accuracy", lambda k: False)
    rows = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r"& \multicolumn{4}{c}{Five datasets, $T\in\{64,128,256\}$} & \multicolumn{2}{c}{ESC-10, per clip} \\",
            r"\cmidrule(lr){2-5}\cmidrule(l){6-7}",
            r"Decoder & Mean drop & Max drop & $|\Delta|\le0.01$ & Equivalent & Mean drop & Equivalent \\", r"\midrule"]
    for r in REPS:
        d, e = np.array(main_d[r]), np.array(esc_d[r])
        if len(d) < n_main or len(e) < n_esc:                   # still being computed: leave it out
            continue
        rows.append(f"{SHORT[r]} & {d.mean():.3f} & {d.max():.3f} & {100 * np.mean(np.abs(d) <= 0.01):.0f}\\% & "
                    f"{main_eq[r]}/{n_main_eq} & {e.mean():.3f} & {esc_eq[r]}/{n_esc} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_utility.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    nums["clsCombos"] = str(n_main)
    nums["clsCombosEq"] = str(n_main_eq)
    main_blocks = [b for k, b in cls.items()
                   if not k.startswith("_") and not k.startswith("esc50") and b["windows"] >= MIN_WINDOWS]
    nums["clsWindows"] = f"{sum(b['windows'] for b in main_blocks):,}".replace(",", "\\,")
    # Reference level: accuracy (single-label) and macro-F1 (PTB-XL) on the original windows.
    acc = [b[c]["original"]["accuracy"] for b in main_blocks if not b.get("multilabel") for c in CLFS]
    f1 = [b[c]["original"]["macro_f1"] for b in main_blocks if b.get("multilabel") for c in CLFS]
    nums["origAccMin"], nums["origAccMax"] = fmt(min(acc), 2), fmt(max(acc), 2)
    nums["origFoneMin"], nums["origFoneMax"] = fmt(min(f1), 2), fmt(max(f1), 2)
    nums["escWindows"] = f"{sum(b['windows'] for k, b in esc.items() if not k.startswith('_')):,}".replace(",", "\\,")
    nums["escCombos"] = str(n_esc)
    nums["clsMaxDropExact"] = fmt(max(max(main_d[r]) for r in EXACT), 3)
    nums["escMaxDropExact"] = fmt(max(max(esc_d[r]) for r in EXACT), 3)
    if main_d.get("RP_SIPERS_SIGNED"):                       # Sipers with the sign, added later with --only
        nums["sipersSigMaxDrop"] = fmt(max(main_d["RP_SIPERS_SIGNED"]), 3)
        nums["sipersSigEq"] = str(main_eq["RP_SIPERS_SIGNED"])
        nums["sipersSigEscEq"] = str(esc_eq["RP_SIPERS_SIGNED"])
    nums["clsEqRP"] = str(main_eq["RP"])
    if len(main_d["RP_SIPERS_SIGNED"]) == n_main and len(esc_d["RP_SIPERS_SIGNED"]) == n_esc:
        nums["clsMaxDropSipers"] = fmt(max(main_d["RP_SIPERS_SIGNED"]), 3)
        nums["clsEqSipers"] = str(main_eq["RP_SIPERS_SIGNED"])
        # does a refit on the originals reproduce the published original scores?
        refit = [abs(b[c]["original_refit"][m] - b[c]["original"][m]) for s in (cls, esc) for k, b in s.items()
                 if not k.startswith("_") for c in CLFS if "original_refit" in b[c] for m in b[c]["original"]]
        nums["refitMaxDiff"] = fmt(max(refit), 3)

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
    if all("RP_SIPERS_SIGNED" in rob[k] for k in rb):
        nums["robHighSIPERS"] = fmt(noisy("RP_SIPERS_SIGNED", "16.0"), 3)
        nums["robDatasetsRPBetterSipers"] = str(sum(rob[k]["RP"]["nmae"]["16.0"]["mean"]
                                                    < rob[k]["RP_SIPERS_SIGNED"]["nmae"]["16.0"]["mean"] for k in rb))
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
    for rate, word in (("500", "Fivehundred"), ("50", "Fifty")):          # constant duration (1.28 s) control
        b = sr[f"{rate}Hz/1.28s"]
        nums[f"srDurRP{word}"] = fmt(b["RP/png8"]["nmae"]["mean"])
        nums[f"srDurCol{word}"] = fmt(b["collinearity_median"], 2)
    nat, res = sr["100_nativeHz/L128"]["RP/png8"]["nmae"]["mean"], sr["100Hz/L128"]["RP/png8"]["nmae"]["mean"]
    nums["srNativeRP"], nums["srResampledRP"] = fmt(nat), fmt(res)

    # Temporal and spectral fidelity by technique: range over the seven datasets at T = 128.
    ks = [k for k in blocks if k.endswith("/L128")]
    cols = (("pearson", "Pearson", 3), ("psd_shape", "PSD shape", 4), ("energy_err", "Energy", 3))
    rows = [r"\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", r"\toprule",
            "Decoder & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for r in (r for r in REPS if all(f"{r}/png8" in bench[k] for k in ks)):
        cells = []
        for m, _, d in cols:
            v = [bench[k][f"{r}/png8"][m]["mean"] for k in ks]
            cells.append(f"{min(v):.{d}f}--{max(v):.{d}f}")
        rows.append(f"{SHORT[r]} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_spectral.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Reflection-invariant evidence for RQ2: the spectrum does not change under x -> -x, so it shows what each
    # decoder keeps of the waveform without knowing the orientation (no oracle needed).
    def span(r, m, d):
        v = [bench[k][f"{r}/png8"][m]["mean"] for k in ks]
        return fmt(min(v), d), fmt(max(v), d)

    nums["psdGeoUnsMin"], nums["psdGeoUnsMax"] = span("RP_UNSIGNED", "psd_shape", 4)
    nums["psdGeoSigMin"], nums["psdGeoSigMax"] = span("RP", "psd_shape", 4)
    nums["psdHirMin"], nums["psdHirMax"] = span("RP_HIRATA", "psd_shape", 2)
    nums["pearsonUnsAbsMax"] = fmt(max(abs(bench[k]["RP_UNSIGNED/png8"]["pearson"]["mean"]) for k in ks), 2)

    # Sign identifiability on the benchmark windows, and Hirata's recurrence-rate sweep.
    sid = load("results/_stats/sign_identifiability.json")
    nums["signSeries"] = f"{sid['_total']['series']:,}".replace(",", "\\,")
    nums["signAmbiguous"] = str(sid["_total"]["no_informative_pair"])
    nums["signMinShare"] = f"{100 * min(v['min_share'] for k, v in sid.items() if '/' in k):.0f}"
    sw = load("results/_stats/hirata_threshold_sweep.json")
    gains = {enc: max(v[enc]["nmae_by_rate"]["0.75"] - v[enc]["best_nmae"] for k, v in sw.items() if not k.startswith("_"))
             for enc in ("unsigned", "signed")}
    nums["hirataGainUns"], nums["hirataGainSig"] = fmt(gains["unsigned"], 3), fmt(gains["signed"], 3)
    best = [v[enc]["best_nmae"] for k, v in sw.items() if not k.startswith("_") for enc in ("unsigned", "signed")]
    nums["hirataBestMin"], nums["hirataBestMax"] = fmt(min(best), 2), fmt(max(best), 2)

    # Supplement: Hirata's recurrence-rate sweep.
    rates = list(sw["wisdm"]["unsigned"]["nmae_by_rate"])
    rows = [r"\begin{tabular}{@{}ll" + "r" * len(rates) + "r@{}}", r"\toprule",
            "Dataset & Encoding & " + " & ".join(rates) + r" & Best \\", r"\midrule"]
    for ds, name in NAMES.items():
        for e in ("unsigned", "signed"):
            v = sw[ds][e]
            rows.append(f"{name if e == 'unsigned' else ''} & {e} & "
                        + " & ".join(fmt(v["nmae_by_rate"][r], 3) for r in rates) + f" & {v['best_rate']} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_hirata_sweep.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Supplement: sampling rate at a fixed length (128 samples) and at a fixed duration (1.28 s).
    srs = ("RP", "RP_SIPERS_SIGNED", "GASF", "GADF", "XIRP")
    rows = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
            r"Window & Rate & Seconds & $\sigma_2/\sigma_1$ & Share $<0.1$ & Signed RP & Sipers\,/\,sig. & GASF & GADF"
            r" & XIRP \\", r"\midrule"]
    for win in ("L128", "1.28s"):
        for rate in ("500", "250", "100", "100_native", "50"):
            b = sr[f"{rate}Hz/{win}"]
            rows.append(("128 samples" if win == "L128" else "1.28 s") * (rate == "500")
                        + f" & {rate.replace('_native', ' (native)')} Hz & {b['window_seconds']:.2f}"
                        + f" & {b['collinearity_median']:.2f} & {100 * b['share_below_0.1']:.0f}\\%"
                        + "".join(f" & {fmt(b[f'{r}/png8']['nmae']['mean'])}" if f"{r}/png8" in b else " & --"
                                  for r in srs) + r" \\")
        if win == "L128":
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_sampling.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Supplement: Sipers et al. and our decoder on the same circular RP, and Sipers' node-threshold sweep.
    pilot_path = Path("results/_stats/sipers_pilot.json")
    if pilot_path.exists():
        pilot = load(pilot_path)
        tols = [f"{t:g}" for t in pilot["_design"]["tols"]]
        rows = [r"\begin{tabular}{@{}lrrrr" + "r" * len(tols) + "@{}}", r"\toprule",
                r"& \multicolumn{2}{c}{Unsigned, oracle sign} & \multicolumn{2}{c}{Signed}"
                r" & \multicolumn{" + str(len(tols)) + r"}{c}{Sipers, unsigned, oracle sign, node threshold} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-" + str(5 + len(tols)) + "}",
                "Dataset & Sipers & Geometric & Sipers & Geometric & " + " & ".join(tols) + r" \\", r"\midrule"]
        gaps = []
        for ds, name in NAMES.items():
            if ds not in pilot:
                continue
            p = pilot[ds]
            g = [p["png8/unsigned/sipers"]["nmae_oracle"], p["png8/unsigned/geometric"]["nmae_oracle"],
                 p["png8/signed/sipers"]["nmae"], p["png8/signed/geometric"]["nmae"]]
            gaps.append(g[2] / g[3])
            rows.append(f"{name} & " + " & ".join(fmt(v) for v in g) + " & "
                        + " & ".join(fmt(p[f"png8/sweep/{t}"]["nmae_oracle"]) for t in tols) + r" \\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        (OUT / "tab_sipers_pilot.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
        nums["pilotRatioMin"], nums["pilotRatioMax"] = fmt(min(gaps), 1), fmt(max(gaps), 1)
        fl = [pilot[ds]["float/sweep/0.01"]["nmae_oracle"] for ds in NAMES if ds in pilot]
        nums["sipPaperTolMin"], nums["sipPaperTolMax"] = fmt(min(fl), 3), fmt(max(fl), 3)

    # Encoder vs decoder on the same windows (scripts/_sipers_checks.py): Swarm, 8-bit signed RP.
    checks_path = Path("results/_stats/sipers_checks.json")
    if checks_path.exists():
        ch = load(checks_path)
        sw = ch["swarm"]
        nums["swGeoNonCirc"], nums["swGeoCirc"] = fmt(sw["geo_noncircular_8bit"]["mean"]), fmt(sw["geo_circular_8bit"]["mean"])
        nums["swSipCirc"] = fmt(sw["sipers_circular_8bit"]["mean"])
        nums["swRatioNonCirc"] = fmt(sw["ratio_noncircular"]["median"])
        nums["swRatioCirc"] = fmt(sw["ratio_circular"]["median"], 2)
        others = [d for d in ch if not d.startswith("_") and d != "swarm"]
        nums["encGapMax"] = fmt(max(abs(ch[d]["geo_circular_8bit"]["mean"] - ch[d]["geo_noncircular_8bit"]["mean"])
                                    for d in others), 4)
        clean = [d for d in ch if not d.startswith("_") and d != "ptbxl"]
        nums["sipZeroMaxClean"] = f"{max(ch[d]['sipers_float_tol0']['max'] for d in clean):.0e}"
        nums["sipZeroMaxPtb"] = f"{ch['ptbxl']['sipers_float_tol0']['max']:.1e}"
        nums["sipZeroPtbSeries"] = f"{ch['ptbxl']['series']:,}".replace(",", "\\,")

    (OUT / "numbers.tex").write_text("% generated by scripts/_emit_infsci_tables.py; do not edit\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items()), encoding="utf-8")
    print("\n".join(f"{k} = {v}" for k, v in nums.items()))


if __name__ == "__main__":
    main()
