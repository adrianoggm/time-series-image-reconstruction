#!/usr/bin/env bash
# Post-build verification: undefined refs/citations, bad boxes, and label presence.
set -uo pipefail
DOC=/mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales/doc
cd "$DOC"

for f in main2 supplementary; do
  echo "########## $f.log ##########"
  if [ ! -f "$f.log" ]; then echo "no log"; continue; fi
  echo "-- undefined refs / citations --"
  grep -Ei 'Warning: Reference|Warning: Citation|There were undefined' "$f.log" | sort -u | head -20 \
    || true
  echo "-- overfull hboxes > 10pt --"
  grep -Eo 'Overfull \\hbox \([0-9]+\.[0-9]+pt' "$f.log" \
    | awk -F'[(]' '{gsub("pt","",$2); if ($2+0 > 10) print $0}' | sort -u | head -15 || true
  echo "-- rerun requested? --"
  grep -Ei 'Rerun to get|Label\(s\) may have changed' "$f.log" | sort -u || echo "no rerun needed"
  echo
done

echo "########## new labels present in supplementary.aux ##########"
grep -c 'supp_rp_orientation' supplementary.aux || echo 0
echo "########## PDF timestamps vs sources ##########"
ls -l --time-style=+%H:%M:%S main2.tex main2.pdf supplementary.tex supplementary.pdf
