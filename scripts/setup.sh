#!/usr/bin/env bash
# One-shot environment bootstrap for Linux / macOS / WSL.
#
# Creates a Python 3.10 venv at ./.venv and installs the project (editable).
# Uses `uv` if available (10-100× faster); falls back to standard pip otherwise.
#
# Usage:   bash scripts/setup.sh                # core deps
#          bash scripts/setup.sh dev             # core + linting tools
#          bash scripts/setup.sh all            # everything
#          TFG_PYTHON=3.11 bash scripts/setup.sh

set -euo pipefail

PYTHON_VERSION="${TFG_PYTHON:-3.10}"
VENV_DIR="${TFG_VENV:-.venv}"
EXTRAS=("$@")

# --- locate repo root so the script works from any cwd -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- pick a Python interpreter ------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    HAVE_UV=1
else
    HAVE_UV=0
    echo "==> 'uv' not found. Bootstrapping it locally (no sudo needed)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    HAVE_UV=1
fi

echo "==> Ensuring Python $PYTHON_VERSION is available via uv..."
uv python install "$PYTHON_VERSION"

# --- create venv --------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "==> Reusing existing venv at $VENV_DIR"
else
    echo "==> Creating venv at $VENV_DIR (Python $PYTHON_VERSION)"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

# --- assemble install spec ----------------------------------------------------
SPEC="."
if [ "${#EXTRAS[@]}" -gt 0 ]; then
    JOINED=$(IFS=, ; echo "${EXTRAS[*]}")
    SPEC=".[$JOINED]"
fi

echo "==> Installing $SPEC (editable)"
uv pip install --python "$VENV_DIR/bin/python" -e "$SPEC"

cat <<EOF

==> Done.

Activate the venv with:
    source $VENV_DIR/bin/activate

Then verify imports:
    python -c "from src.activity_data import get_time_setup; print(get_time_setup('WISDM'))"

Dataset path defaults to ./data/WISDM/ (override with TFG_DATA_DIR env var).
EOF
