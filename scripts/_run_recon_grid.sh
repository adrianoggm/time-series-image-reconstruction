#!/usr/bin/env bash
# Regenerate the reconstruction figures, including the composite 2x2 grid the
# main manuscript uses, from the WISDM window and the production decoders.
set -uo pipefail
WIN=/mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales
WSL=/home/adrianogarcia/tfg
sed 's/\r$//' "$WIN/scripts/_regen_reconstruction_figs.py" > "$WSL/scripts/_regen_reconstruction_figs.py"
cd "$WSL"
export OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
.venv/bin/python -m scripts._regen_reconstruction_figs 2>&1 \
  | grep -vE "tensorflow|dlerror|libnv|cudart|warn\(|UserWarning|quantiles are equal" | tail -20
