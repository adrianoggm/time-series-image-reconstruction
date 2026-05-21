#!/usr/bin/env python3
"""Aggregate every `results/<stype>/<classifier>/fold-N.json` into a table.

Prints a Markdown comparison of original vs reconstructed accuracy / macro-F1
across encodings and classifiers, and optionally writes it to a file.

Example
-------
    python -m scripts.summarize_results
    python -m scripts.summarize_results --out docs/results/latest.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import REPO_ROOT
from src.paths import ALL_STYPES


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for jf in sorted(results_root.rglob("fold-*.json")):
        # `_originals/` holds per-classifier caches that don't follow the
        # `<stype>/<classifier>/fold-N.json` shape — skip them here, the data
        # they cache is already inlined in every <stype> JSON.
        if "_originals" in jf.parts:
            continue
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  ! skip {jf}: {e}")
            continue
        if "stype" not in payload or "classifier" not in payload:
            continue
        orig = payload.get("evaluations", {}).get("original", {})
        rec  = payload.get("evaluations", {}).get("reconstructed", {})
        rows.append({
            "stype":      payload.get("stype"),
            "classifier": payload.get("classifier"),
            "fold":       payload.get("fold"),
            "sampling":   payload.get("sampling"),
            "orig_acc":   orig.get("accuracy"),
            "orig_f1":    _safe_get(orig, "macro avg", "f1-score"),
            "rec_acc":    rec.get("accuracy"),
            "rec_f1":     _safe_get(rec, "macro avg", "f1-score"),
            "delta_acc":  None if rec.get("accuracy") is None or orig.get("accuracy") is None
                          else rec["accuracy"] - orig["accuracy"],
            "_path":      str(jf.relative_to(results_root.parent)),
        })
    return rows


def _fmt(x):
    return "—" if x is None else f"{x:.3f}"


def render_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_(no results found)_\n"
    lines = ["# Validation results summary", "",
             "| Encoding | Classifier | Fold | Sampling | Orig acc | Orig F1 | Rec acc | Rec F1 | Δacc |",
             "|---|---|---|---|---|---|---|---|---|"]
    # sort by stype, then classifier, then fold
    rows.sort(key=lambda r: (str(r["stype"]), str(r["classifier"]), r["fold"] or 0))
    for r in rows:
        lines.append("| {stype} | {classifier} | {fold} | {sampling} | "
                     "{orig_acc} | {orig_f1} | {rec_acc} | {rec_f1} | {delta_acc} |".format(
            stype=r["stype"], classifier=r["classifier"], fold=r["fold"],
            sampling=r["sampling"], orig_acc=_fmt(r["orig_acc"]),
            orig_f1=_fmt(r["orig_f1"]), rec_acc=_fmt(r["rec_acc"]),
            rec_f1=_fmt(r["rec_f1"]), delta_acc=_fmt(r["delta_acc"]),
        ))
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default=str(REPO_ROOT / "results"),
                   help="folder containing per-classifier subfolders with fold-*.json files")
    p.add_argument("--out", default=None, help="write the rendered Markdown to this path")
    args = p.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        print(f"no results yet at {root} — run a validator with default --save-results first")
        return

    rows = collect(root)
    md = render_markdown(rows)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
