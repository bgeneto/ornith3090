#!/usr/bin/env bash
# Install NeMo AutoModel into a *separate* venv so the serving stack stays frozen.
#
#   bash docker/install_dflash2_train.sh
#
# Docker:  /cache/venv-dflash2   (survives compose recreate; HOME is /cache)
# Host:    $REPO/venv-dflash2
#
# The train venv sees /app/venv (torch, vLLM, compressed-tensors) through a .pth
# file. `pip install nemo-automodel` into /app/venv would downgrade transformers
# 5.15 → 5.8.1 and break serving. There is no system `pip` in this image:
# use /app/venv/bin/python -m pip, or source docker/env.sh.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

if [ -x "$REPO/venv/bin/python" ]; then
  SERVE_PY="$REPO/venv/bin/python"
elif [ -x /app/venv/bin/python ]; then
  SERVE_PY=/app/venv/bin/python
else
  echo "no serving venv at $REPO/venv or /app/venv" >&2
  exit 1
fi

if [ -f /.dockerenv ]; then
  VENV=${DFLASH2_TRAIN_VENV:-/cache/venv-dflash2}
else
  VENV=${DFLASH2_TRAIN_VENV:-$REPO/venv-dflash2}
fi
REQ="$DIR/requirements-dflash2-train.txt"
MARKER="$VENV/.dflash2-train-ready"

ok() {
  "$VENV/bin/python" -c "from nemo_automodel.recipes.llm.train_dflash2 import TrainDFlash2Recipe" 2>/dev/null
}

if [ -x "$VENV/bin/python" ] && ok; then
  echo "DFlash2 train venv ready: $VENV"
  "$VENV/bin/python" -c "import nemo_automodel; print('  nemo_automodel', getattr(nemo_automodel,'__version__','ok'))"
  exit 0
fi

echo "== creating $VENV (serving venv is not modified)"
"$SERVE_PY" -m venv "$VENV"
PYVER="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
SITE="$("$SERVE_PY" -c 'import site; print(site.getsitepackages()[0])')"
echo "$SITE" > "$VENV/lib/python${PYVER}/site-packages/ornith-serve.pth"
echo "  inherit serving site-packages: $SITE"

"$VENV/bin/python" -m pip install -U pip
# Install NeMo without pulling a second torch (CUDA torch comes from the .pth).
"$VENV/bin/python" -m pip install --no-deps "nemo-automodel>=0.5.0"
DEPS=$(mktemp)
trap 'rm -f "$DEPS"' EXIT
grep -vE '^(#|$|nemo-automodel)' "$REQ" > "$DEPS"
"$VENV/bin/python" -m pip install -r "$DEPS" --ignore-installed
if ! "$VENV/bin/python" -m pip install 'flashoptim>=0.1.3'; then
  echo "note: flashoptim skipped (optional; smoke training does not need it)" >&2
fi

if ! ok; then
  echo "install finished but TrainDFlash2Recipe still does not import." >&2
  echo "Try: $VENV/bin/python -c 'import nemo_automodel; print(nemo_automodel.__file__)'" >&2
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$MARKER"
echo "DFlash2 train venv ready: $VENV"
echo "Launch: bash drafter/train_dflash2.sh --smoke"
echo "     or: docker compose --profile train run --rm train --smoke"
