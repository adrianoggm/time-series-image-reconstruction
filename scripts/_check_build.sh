#!/usr/bin/env bash
# Strict gate: rebuild each document until latexmk reports it stable, then fail
# loudly if the FINAL pass still has an undefined reference or citation.
set -uo pipefail
DOC=/mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales/doc
cd "$DOC"
status=0

for f in main2 main3 supplementary; do
  echo "########## $f ##########"
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$f.tex" \
    > "/tmp/$f.check.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  BUILD FAILED (latexmk exit=$rc)"
    grep -E '^\./|^!|Fatal' "/tmp/$f.check.log" | head -15
    status=1
    continue
  fi
  # A stable build re-runs nothing; ask latexmk what it would still do.
  latexmk -pdf -interaction=nonstopmode "$f.tex" > "/tmp/$f.stable.log" 2>&1
  undef=$(grep -cE "LaTeX Warning: (Reference|Citation) .* undefined" "/tmp/$f.stable.log")
  over=$(grep -cE 'Overfull \\hbox \([0-9]{2,}' "/tmp/$f.stable.log")
  pages=$(pdfinfo "$f.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  echo "  ok  pages=$pages  undefined=$undef  overfull(>=10pt)=$over"
  [ "$undef" -eq 0 ] || { echo "  --- undefined ---"; grep -E "LaTeX Warning: (Reference|Citation) .* undefined" "/tmp/$f.stable.log" | sort -u | head -10; status=1; }
done

echo
[ "$status" -eq 0 ] && echo "ALL BUILDS CLEAN" || echo "BUILD GATE FAILED"
exit $status
