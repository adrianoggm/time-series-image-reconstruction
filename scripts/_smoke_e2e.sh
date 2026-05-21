#!/usr/bin/env bash
# End-to-end smoke for the DX refactor: run knn validator with --stype GADF,
# verify a JSON result file is produced, then summarize.
set -e
cd ~/tfg
source .venv/bin/activate
export TF_CPP_MIN_LOG_LEVEL=3

echo "=== knnvalidation --stype GADF (default: save_results=True) ==="
python -m scripts.knnvalidation --stype GADF 2>&1 | tail -8

echo ""
echo "=== results dir contents ==="
find results -type f 2>/dev/null | sort

echo ""
echo "=== summarize_results ==="
python -m scripts.summarize_results
