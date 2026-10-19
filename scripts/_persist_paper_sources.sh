#!/usr/bin/env bash
# Regenerate, and persist, the three paper figures that had no stored source:
# the GADF pivot sweep, the per-subject MAE Wilcoxon and the controlled MTF
# decoder comparison. Also rebuilds the LOSO tables now that their Original
# column no longer reads the unversioned results/_originals cache.
#
# Run from Windows:  wsl.exe -e bash -c 'bash /home/adrianogarcia/run.sh scripts/_persist_paper_sources.sh'
set -uo pipefail
WIN=/mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales
WSL=/home/adrianogarcia/tfg
cd "$WSL"
export OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
PY=.venv/bin/python

for f in scripts/_gadf_pivot_stats.py scripts/_stat_per_subject.py \
         scripts/_mtf_controlled_ablation.py scripts/_make_loso_tables.py; do
  sed 's/\r$//' "$WIN/$f" > "$WSL/$f"
done

echo "== GADF pivot sweep"
$PY -m scripts._gadf_pivot_stats --n 1500 2>&1 | grep -vE "tensorflow|dlerror|libnv|cudart|warn\(" | tail -12
echo "== per-subject MAE Wilcoxon (loto)"
$PY -m scripts._stat_per_subject --sampling loto 2>&1 | grep -vE "tensorflow|dlerror|libnv|cudart|warn\(" | tail -14
echo "== controlled MTF decoder comparison"
$PY -m scripts._mtf_controlled_ablation 2>&1 | grep -vE "tensorflow|dlerror|libnv|cudart|warn\(|UserWarning|quantiles are equal" | tail -10
echo "== LOSO tables"
$PY -m scripts._make_loso_tables > "$WSL/results/_stats/loso_tables.tex" 2>/dev/null
grep -E "^CNN|Random Forest" "$WSL/results/_stats/loso_tables.tex" | head -2

echo "== copying artefacts back to the Windows checkout"
for f in gadf_pivot_sweep.json gadf_pivot_sweep.tex per_subject_mae_loto.json \
         mtf_decoder_ablation.json loso_tables.tex; do
  if [ -f "$WSL/results/_stats/$f" ]; then
    cp "$WSL/results/_stats/$f" "$WIN/results/_stats/$f" && echo "  ok  $f"
  else
    echo "  MISSING $f"
  fi
done
