"""Splice freshly-emitted classifier-table bodies into a supplied TeX file.

Replaces the data region (5 class rows + midrule + 3 footer rows) of each
tab:*_classifiers table with the body emitted by emit_classifier_tables.
Refuses to write unless all six tables matched.

    python -m scripts._apply_classifier_tables /mnt/c/.../doc/supplementary.tex
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from scripts.emit_classifier_tables import emit

LABELS = {
    "gru": "gru_classifiers", "knn": "knn_classifiers", "cnn": "cnn_classifiers",
    "randomforest": "rf_classifiers", "svc": "svm_classifiers", "lstm": "lstm_classifiers",
}


def main():
    tex_path = Path(sys.argv[1])
    sampling = sys.argv[2] if len(sys.argv) > 2 else "loto"
    text = tex_path.read_text(encoding="utf-8")
    changed = 0
    for clf, label in LABELS.items():
        body = emit(clf, sampling)
        pat = re.compile(
            r"(\\label\{tab:" + re.escape(label) + r"\}.*?Class\b.*?\\\\\s*\\midrule\s*)(.*?)(\s*\\bottomrule)",
            re.DOTALL,
        )
        text, k = pat.subn(
            lambda m: m.group(1) + body + "\n" + m.group(3).lstrip(),
            text,
            count=1,
        )
        if k == 1:
            changed += 1
            print(f"OK spliced tab:{label}")
        else:
            print(f"WARN: tab:{label} not matched")

    if changed == len(LABELS):
        tex_path.write_text(text, encoding="utf-8")
        print(f"Wrote {tex_path} ({changed}/{len(LABELS)} tables)")
    else:
        print(f"ABORT: {changed}/{len(LABELS)} matched; {tex_path} NOT modified")
        sys.exit(1)


if __name__ == "__main__":
    main()
