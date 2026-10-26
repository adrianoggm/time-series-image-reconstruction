#!/usr/bin/env bash
# Build the Information Sciences manuscript and its supplement, which cross-reference each other through xr:
# main -> supplementary -> main -> supplementary, each with latexmk; then report warnings.
set -u
cd "$(dirname "$0")/../doc/infsci"
for f in main supplementary main supplementary; do
  latexmk -g -pdf -interaction=nonstopmode -halt-on-error "$f.tex" > "/tmp/infsci_$f.log" 2>&1 || { echo "$f failed"; tail -20 "/tmp/infsci_$f.log"; exit 1; }
done
for f in main supplementary; do
  pdflatex -interaction=nonstopmode "$f.tex" > "/tmp/infsci_$f.final.log" 2>&1
  echo "== $f: $(grep -a 'Output written' "/tmp/infsci_$f.final.log")"
  grep -a -E 'undefined|Undefined|multiply defined|Float too large|Overfull|Underfull' "/tmp/infsci_$f.final.log" | sort | uniq -c
  echo "unresolved '??' in text: $(pdftotext "$f.pdf" - | grep -c '??')"
done
