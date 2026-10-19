#!/usr/bin/env bash
# Strict gate for the three LaTeX documents.
#
# Two passes on purpose. The first builds to stability, so the .aux files settle
# and first-pass "undefined reference" noise disappears. The second is a FORCED
# single pass (latexmk -g) over those settled aux files, and it is the one we
# inspect: a plain second run would compile nothing and its log would be empty,
# which silently reported every document as clean.
set -uo pipefail
DOC=/mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales/doc
cd "$DOC"
status=0
OVERFULL_PT=${OVERFULL_PT:-10}   # report overfull boxes at or above this many points

for f in main2 main3 supplementary; do
  echo "########## $f ##########"
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$f.tex" \
    > "/tmp/$f.build.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  BUILD FAILED (latexmk exit=$rc)"
    grep -E '^\./|^!|Fatal' "/tmp/$f.build.log" | head -15
    status=1
    continue
  fi

  latexmk -pdf -g -interaction=nonstopmode -file-line-error "$f.tex" \
    > "/tmp/$f.final.log" 2>&1
  log="/tmp/$f.final.log"

  undef=$(grep -cE "LaTeX Warning: (Reference|Citation) .* undefined" "$log")
  # Overfull boxes at or above the threshold, in points.
  over=$(grep -oE "Overfull \\\\hbox \(([0-9]+)\.[0-9]+pt too wide\)" "$log" \
         | grep -oE "\(([0-9]+)\." | tr -d '(.' \
         | awk -v t="$OVERFULL_PT" '$1>=t' | wc -l)
  pages=$(pdfinfo "$f.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  echo "  pages=$pages  undefined=$undef  overfull(>=${OVERFULL_PT}pt)=$over"

  if [ "$undef" -ne 0 ]; then
    echo "  --- undefined ---"
    grep -E "LaTeX Warning: (Reference|Citation) .* undefined" "$log" | sort -u | head -10
    status=1
  fi
  if [ "$over" -ne 0 ]; then
    echo "  --- overfull ---"
    grep -E "Overfull \\\\hbox \([0-9]{2,}" "$log" | head -10
    status=1
  fi
done

echo
[ "$status" -eq 0 ] && echo "ALL BUILDS CLEAN" || echo "BUILD GATE FAILED"
exit $status
