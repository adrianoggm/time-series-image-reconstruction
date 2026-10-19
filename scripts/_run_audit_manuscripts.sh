#!/usr/bin/env bash
# Read-only manuscript audit, run inside the WSL checkout (where data/ and the
# full results/ tree live). Syncs only the audit script and the two manuscripts,
# runs the numeric emitters + audit_manuscripts.py, and copies the evidence back
# to the Windows scratchpad. Launch from Windows with:
#   wsl.exe bash /home/adrianogarcia/run.sh scripts/_run_audit_manuscripts.sh
set -uo pipefail
WIN=/mnt/c/Users/adriano.garcia/Desktop/Developer/TFG-Tratamiento-de-Series-Temporales
WSL=/home/adrianogarcia/tfg
OUT=${AUDIT_OUT:-/mnt/c/Users/adriano.garcia/AppData/Local/Temp/claude/c--Users-adriano-garcia-Desktop-Developer-TFG-Tratamiento-de-Series-Temporales/113d57dc-ae2c-4256-86c9-723423df6f26/scratchpad/wsl_audit}
mkdir -p "$OUT" "$WSL/doc" "$WSL/docs/audit"
for f in scripts/audit_manuscripts.py doc/main2.tex doc/supplementary.tex; do
  sed 's/\r$//' "$WIN/$f" > "$WSL/$f"
done
cd "$WSL"
export OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
PY=.venv/bin/python
echo "== emit_main2_numbers";       $PY -m scripts._emit_main2_numbers            > "$OUT/emit_main2.txt" 2>&1;            echo "exit=$?"
echo "== emit_classifier_tables";   $PY -m scripts.emit_classifier_tables --sampling loto > "$OUT/clf_tables_loto.txt" 2>&1; echo "exit=$?"
echo "== stat_per_subject loto";    $PY -m scripts._stat_per_subject --sampling loto > "$OUT/stat_per_subject_loto.txt" 2>&1; echo "exit=$?"
echo "== audit_manuscripts";        $PY -m scripts.audit_manuscripts --data "$WSL/data/WISDM/" --results "$WSL/results" --out "$WSL/docs/audit" > "$OUT/audit_manuscripts.txt" 2>&1; echo "exit=$?"
cp "$WSL/docs/audit/numerical_evidence.json" "$OUT/" 2>/dev/null || echo "no evidence json"
echo "== tail audit"; tail -n 40 "$OUT/audit_manuscripts.txt"
