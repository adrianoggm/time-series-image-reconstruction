#!/usr/bin/env bash
# Phase 11: build main2.pdf then supplementary.pdf (the latter needs main2.aux via
# \externaldocument). Runs latexmk from WSL directly against the Windows checkout.
set -uo pipefail
DOC=/mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales/doc
cd "$DOC"

for f in main2 main3 supplementary; do
  echo "########## $f ##########"
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$f.tex" \
    > "/tmp/$f.build.log" 2>&1
  rc=$?
  echo "latexmk exit=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "--- errors ---"
    grep -E '^\./|^!|Error|Fatal' "/tmp/$f.build.log" | head -30
  fi
  echo "--- warnings worth seeing ---"
  grep -Ei 'Overfull \\hbox \([0-9]{2,}' "/tmp/$f.build.log" | head -10
  grep -Ei 'undefined (control sequence|reference|citation)|multiply defined|LaTeX Warning: Reference|LaTeX Warning: Citation' \
    "/tmp/$f.build.log" | sort -u | head -20
  ls -l "$f.pdf" 2>/dev/null
  echo
done

echo "########## page counts ##########"
for f in main2 main3 supplementary; do
  printf '%-16s ' "$f.pdf"
  pdfinfo "$f.pdf" 2>/dev/null | grep -i '^Pages' || echo "(pdfinfo unavailable)"
done
