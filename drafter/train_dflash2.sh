#!/usr/bin/env bash
# Train an Ornith-1.5-9B DFlash2 drafter (NeMo AutoModel TrainDFlash2Recipe).
#
#   bash drafter/train_dflash2.sh              # cloud yaml, nproc from NPROC (default 2)
#   bash drafter/train_dflash2.sh --smoke      # 3090 yaml, 1 GPU, chat_smoke.jsonl
#   bash drafter/train_dflash2.sh --full-attn  # taps 3/11/19/27/31 if GDN aux hiddens starve fc
#   bash drafter/train_dflash2.sh --dry-run    # print the command only
#
# Teacher: models/Ornith-1.5-9B-MixedInt4-AutoRound (the serving INT4 checkpoint).
# Data: drafter/data/chat.jsonl from convert_gen_to_chat.py (or chat_smoke.jsonl).
#
# Requires: pip install "nemo-automodel" (or NVIDIA NeMo AutoModel from source)
# with a CUDA torch that can load the teacher. If AutoRound load fails, dump
# aux hiddens with vLLM and train offline — see drafter/README.md.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"
if [ -x "$REPO/venv/bin/python" ]; then
  PYTHON="$REPO/venv/bin/python"
  export PATH="$REPO/venv/bin:$PATH"
else
  PYTHON="${PYTHON:-python3}"
fi

SMOKE=0
DRY=0
FULLATTN=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --dry-run|--dry) DRY=1 ;;
    --full-attn|--full-attn-taps) FULLATTN=1 ;;
  esac
done

CFG="$DIR/ornith_dflash2.yaml"
NPROC=${NPROC:-2}
if [ "$SMOKE" = 1 ]; then
  CFG="$DIR/ornith_dflash2_smoke.yaml"
  NPROC=1
  "$PYTHON" "$DIR/convert_gen_to_chat.py" --smoke-only
elif [ "$FULLATTN" = 1 ]; then
  CFG="$DIR/ornith_dflash2_fullattn.yaml"
fi

CMD=(torchrun --standalone --nproc_per_node="$NPROC"
     -m nemo_automodel.recipes.llm.train_dflash2
     -c "$CFG")
echo "${CMD[*]}"
if [ "$DRY" = 1 ]; then
  exit 0
fi

if ! "$PYTHON" -c "import nemo_automodel" 2>/dev/null; then
  echo "nemo_automodel is not installed." >&2
  echo "  pip install nemo-automodel   # or clone NVIDIA-NeMo/Automodel" >&2
  echo "Fallback: bash drafter/train_dflash2_speculators.sh" >&2
  echo "Then rerun: bash drafter/train_dflash2.sh${SMOKE:+ --smoke}" >&2
  exit 1
fi
exec "${CMD[@]}"
