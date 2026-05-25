#!/usr/bin/env bash
# Helper for the LOTO 3-fold image generation across RP/GADF/MTF.
# Run from WSL after the GAF pass has cached the .npy windows.
set -u
cd ~/tfg
source .venv/bin/activate
export TF_CPP_MIN_LOG_LEVEL=3
LOG=/tmp/gen_all.log
: > "$LOG"
echo "=== START $(date) ===" >> "$LOG"
for enc in RP GADF MTF; do
  echo "" >> "$LOG"
  echo "=== $enc START $(date) ===" >> "$LOG"
  /usr/bin/time -f "elapsed=%E rss=%MkB" python -m scripts.generate_images \
    --image-type "$enc" --sampling loto --n-folds 3 >> "$LOG" 2>&1
  rc=$?
  echo "=== $enc END $(date) exit=$rc ===" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "FAILED at $enc with exit=$rc" >> "$LOG"
    exit "$rc"
  fi
done
echo "" >> "$LOG"
echo "=== ALL DONE $(date) ===" >> "$LOG"
