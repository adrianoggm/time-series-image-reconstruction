#!/usr/bin/env bash
# Run the GADF time-series reconstruction on the LOTO 3-fold split.
set -u
cd ~/tfg
source .venv/bin/activate
export TF_CPP_MIN_LOG_LEVEL=3
LOG=/tmp/rec_gadf.log
: > "$LOG"
echo "=== START $(date) ===" >> "$LOG"
/usr/bin/time -f "elapsed=%E rss=%MkB" python -m scripts.generate_time_series \
  --image-type GADF --sampling loto --n-folds 3 >> "$LOG" 2>&1
rc=$?
echo "=== END $(date) exit=$rc ===" >> "$LOG"
exit "$rc"
