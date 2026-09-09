#!/usr/bin/env bash
# Train an Ornith-1.5-9B DFlash2 drafter (NeMo AutoModel TrainDFlash2Recipe).
#
#   bash drafter/train_dflash2.sh --smoke      # 1 GPU (the serving container)
#   bash drafter/train_dflash2.sh --cloud      # 2+ GPU yaml (tp_size 2)
#   bash drafter/train_dflash2.sh --full-attn  # taps 3/11/19/27/31
#   bash drafter/train_dflash2.sh --dry-run    # print the command only
#
# Docker — the image has no system `pip`, and NeMo must not go into /app/venv:
#   source docker/env.sh
#   docker compose --profile single stop      # one GPU; don't train beside vLLM
#   bash drafter/train_dflash2.sh --smoke     # auto-installs /cache/venv-dflash2
#   docker compose --profile train run --rm train --smoke
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

SMOKE=0
DRY=0
FULLATTN=0
CLOUD=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --dry-run|--dry) DRY=1 ;;
    --full-attn|--full-attn-taps) FULLATTN=1 ;;
    --cloud) CLOUD=1 ;;
  esac
done

SERVE_PY=""
if [ -x "$REPO/venv/bin/python" ]; then
  SERVE_PY="$REPO/venv/bin/python"
elif [ -x /app/venv/bin/python ]; then
  SERVE_PY=/app/venv/bin/python
fi

has_nemo() {
  local py=$1
  [ -n "$py" ] && [ -x "$py" ] && \
    "$py" -c "from nemo_automodel.recipes.llm.train_dflash2 import TrainDFlash2Recipe" 2>/dev/null
}

pick_train_python() {
  if [ -n "${TRAIN_PYTHON:-}" ]; then
    echo "$TRAIN_PYTHON"
    return
  fi
  local candidates=()
  [ -n "${DFLASH2_TRAIN_VENV:-}" ] && candidates+=("$DFLASH2_TRAIN_VENV/bin/python")
  candidates+=(/cache/venv-dflash2/bin/python)
  candidates+=("$REPO/venv-dflash2/bin/python")
  [ -n "$SERVE_PY" ] && candidates+=("$SERVE_PY")
  local c
  for c in "${candidates[@]}"; do
    if has_nemo "$c"; then
      echo "$c"
      return
    fi
  done
}

PYTHON="$(pick_train_python || true)"

if [ "$DRY" != 1 ] && ! has_nemo "${PYTHON:-}"; then
  echo "nemo_automodel is not in the serving venv (and must not be — it would" >&2
  echo "downgrade transformers and break vLLM). Install a separate train venv:" >&2
  echo "  source docker/env.sh                 # docker exec: no system pip" >&2
  echo "  bash docker/install_dflash2_train.sh  # writes /cache/venv-dflash2" >&2
  if [ -f /.dockerenv ] || [ "${INSTALL_DFLASH2_TRAIN:-0}" = 1 ]; then
    bash "$REPO/docker/install_dflash2_train.sh"
    PYTHON="$(pick_train_python || true)"
  fi
fi

if [ "$DRY" != 1 ] && ! has_nemo "${PYTHON:-}"; then
  echo "Still no TrainDFlash2Recipe. Fallback: bash drafter/train_dflash2_speculators.sh" >&2
  exit 1
fi

if has_nemo "${PYTHON:-}"; then
  export PATH="$(dirname "$PYTHON"):$PATH"
fi
[ -n "$SERVE_PY" ] && export PATH="$(dirname "$SERVE_PY"):$PATH"

NGPU=1
if [ -n "$SERVE_PY" ]; then
  NGPU="$("$SERVE_PY" -c "import torch; print(max(1, int(torch.cuda.device_count())))" 2>/dev/null || echo 1)"
fi
NPROC=${NPROC:-$NGPU}

if [ "$CLOUD" != 1 ] && [ "$SMOKE" != 1 ] && [ "$FULLATTN" != 1 ] && [ "$NPROC" -lt 2 ]; then
  echo "[train_dflash2] ${NPROC} GPU: using --smoke (cloud yaml needs tp_size 2). Pass --cloud to override."
  SMOKE=1
fi

CFG="$DIR/ornith_dflash2.yaml"
if [ "$SMOKE" = 1 ]; then
  CFG="$DIR/ornith_dflash2_smoke.yaml"
  NPROC=1
  if [ "$DRY" != 1 ]; then
    "${PYTHON:-${SERVE_PY:-python3}}" "$DIR/convert_gen_to_chat.py" --smoke-only
  fi
elif [ "$FULLATTN" = 1 ]; then
  CFG="$DIR/ornith_dflash2_fullattn.yaml"
fi

PY="${PYTHON:-${SERVE_PY:-python3}}"
CMD=("$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC"
     -m nemo_automodel.recipes.llm.train_dflash2
     -c "$CFG")
echo "python: $PY"
echo "${CMD[*]}"
if [ "$DRY" = 1 ]; then
  exit 0
fi
exec "${CMD[@]}"
