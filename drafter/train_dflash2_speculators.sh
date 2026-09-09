#!/usr/bin/env bash
# Fallback trainer if NeMo AutoModel will not load the AutoRound teacher.
# Uses vLLM speculators --speculator-type dflash2. Dry-run a checkpoint into
# vLLM (drafter/smoke_dflash2.py --load) before a long run so you do not train
# a DFlash1-shaped graph (is_causal must stay false).
#
#   bash drafter/train_dflash2_speculators.sh --dry-run
#   bash drafter/train_dflash2_speculators.sh
#
# Requires: pip install speculators  (and a CUDA venv that can load Ornith).
# Data: drafter/data/chat.jsonl (convert_gen_to_chat.py) or chat_smoke.jsonl.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"
if [ -x "$REPO/venv/bin/python" ]; then
  export PATH="$REPO/venv/bin:$PATH"
  PYTHON="$REPO/venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

DRY=0
SMOKE=0
for a in "$@"; do
  case "$a" in
    --dry-run|--dry) DRY=1 ;;
    --smoke) SMOKE=1 ;;
  esac
done

DATA="$DIR/data/chat.jsonl"
[ "$SMOKE" = 1 ] && DATA="$DIR/data/chat_smoke.jsonl"
[ "$SMOKE" = 1 ] && "$PYTHON" "$DIR/convert_gen_to_chat.py" --smoke-only
[ -f "$DATA" ] || { echo "missing $DATA — run drafter/convert_gen_to_chat.py" >&2; exit 1; }

TEACHER=${TEACHER:-$REPO/models/Ornith-1.5-9B-MixedInt4-AutoRound}
OUT=${OUT:-$REPO/drafter/runs/dflash2_speculators}
# Geometry must match dflash2_const.py / ornith_dflash2.yaml
CMD=("$PYTHON" -m speculators.train
     --speculator-type dflash2
     --target-model "$TEACHER"
     --data "$DATA"
     --output-dir "$OUT"
     --block-size 8
     --mask-token-id 248077
     --target-layer-ids 1,8,15,22,29
     --num-hidden-layers 5
     --trust-remote-code)
echo "${CMD[*]}"
echo "If the CLI flags differ on your speculators pin, match vLLM's"
echo "examples/train/dflash2_qwen3_8b_sharegpt_online_5k.sh and keep"
echo "hidden=4096 taps=1,8,15,22,29 mask=248077 is_causal=false."
if [ "$DRY" = 1 ]; then
  exit 0
fi
if ! "$PYTHON" -c "import speculators" 2>/dev/null; then
  echo "speculators is not installed. Prefer: bash drafter/train_dflash2.sh --smoke" >&2
  echo "  $PYTHON -m pip install speculators   # still not into a serving-only constraint" >&2
  echo "There is no system pip in the Docker image; source docker/env.sh first." >&2
  exit 1
fi
exec "${CMD[@]}"
