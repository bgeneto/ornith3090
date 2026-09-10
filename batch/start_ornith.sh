#!/bin/bash
# Ornith-1.5-9B on a single RTX 3090 — BATCH / THROUGHPUT mode.
#
# Optimized configuration adapted from syv-ai/qwen38-27b-rtx3090 for Ornith-1.5-9B:
#  - High concurrency (MAX_SEQS=64 default)
#  - FP16 recurrent Gated DeltaNet state (--mamba-ssm-cache-dtype float16)
#  - Marlin INT8 activation tensor-core GEMMs (W4A8) for decode and prefill
#  - Hybrid prefix caching (--enable-prefix-caching --mamba-cache-mode align)
#  - INT8 group-128 quantized lm_head and embed_tokens
#
# Usage:
#   bash batch/start_ornith.sh
#   MAX_SEQS=32 KV=fp8 bash batch/start_ornith.sh
#   KV=int8pth bash batch/start_ornith.sh
#   ENABLE_THINKING=1 bash batch/start_ornith.sh
#   PREFILL_ATTN=int8 bash batch/start_ornith.sh   # int8-QK prefill attn (with INT8_ACT)

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

export FLASHINFER_DISABLE_VERSION_CHECK=1

if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[start_ornith_batch] removing stale offload region $f"; rm -f "$f"; }
  done
fi

MODEL=${MODEL:-$REPO/models/Ornith-1.5-9B-MixedInt4-AutoRound}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-ornith-1.5-9b}
PORT=${PORT:-18020}
HOST=${HOST:-0.0.0.0}
MAX_SEQS=${MAX_SEQS:-64}
API_SERVERS=${API_SERVERS:-1}
GPU_UTIL=${GPU_UTIL:-0.95}

KV=${KV:-fp8}
case "$KV" in
  int8pth)
    MAX_LEN=${MAX_LEN:-150000}
    KV_ARGS="--kv-cache-dtype int8_per_token_head --attention-backend TRITON_ATTN"
    ;;
  int4pth)
    MAX_LEN=${MAX_LEN:-262144}
    KV_ARGS="--kv-cache-dtype int4_per_token_head --attention-backend TRITON_ATTN"
    ;;
  kvarn)
    MAX_LEN=${MAX_LEN:-262144}
    KV_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
    export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.25}
    ;;
  fp8)
    MAX_LEN=${MAX_LEN:-150000}
    KV_ARGS="--kv-cache-dtype fp8"
    ;;
  *)
    echo "KV must be fp8, int8pth, int4pth, or kvarn (got: $KV)" >&2
    exit 1
    ;;
esac

# INT8 activations for Marlin W4A8 tensor-core execution
INT8_ACT=${INT8_ACT-int8}
INT8_LAYERS=${INT8_LAYERS-mlp|linear_attn|self_attn}
[ -n "$INT8_ACT" ] && export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
[ -n "$INT8_ACT" ] && [ -n "$INT8_LAYERS" ] && export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS
# PREFILL_ATTN=int8: int8-QK Triton prefill on the 8 hd256 full-attn layers
# (16q/4kv). Same translation as single-user/start_ornith.sh. Empty = FA2.
# Prefill-only; quantized KV falls through. Pair with INT8_ACT.
PREFILL_ATTN=${PREFILL_ATTN-}
[ -n "$PREFILL_ATTN" ] && export VLLM_PREFILL_ATTN=$PREFILL_ATTN

EXTRA_ARGS=${EXTRA_ARGS:-}
PREFIX_CACHE=${PREFIX_CACHE:-1}
if [ "$PREFIX_CACHE" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
fi

# CPU offload (optional for constrained VRAM e.g. 8GB GPUs)
if [ -n "${CPU_OFFLOAD_GB:-}" ] && [ "$CPU_OFFLOAD_GB" != "0" ]; then
  EXTRA_ARGS="--cpu-offload-gb $CPU_OFFLOAD_GB ${EXTRA_ARGS}"
fi

# Sleep mode (vLLM --enable-sleep-mode). Default 1: park GPU after VLLM_IDLE_TIMEOUT
# (90s) and auto-wake on the next inference request via docker/vllm-idle-proxy.py.
# Level is not a serve CLI arg in 0.28: 0 = off; 1/2 enable the flag and
# /sleep + /wake_up (VLLM_SERVER_DEV_MODE=1). 1 offloads weights to CPU; 2 discards them.
SLEEP_LEVEL=${SLEEP_LEVEL:-1}
export SLEEP_LEVEL
export VLLM_IDLE_TIMEOUT=${VLLM_IDLE_TIMEOUT:-90}
SLEEP_ARGS=()
case "$SLEEP_LEVEL" in
  0|"") ;;
  1|2)
    SLEEP_ARGS=(--enable-sleep-mode)
    export VLLM_SERVER_DEV_MODE=1
    ;;
  *)
    echo "SLEEP_LEVEL must be 0, 1, or 2 (got: $SLEEP_LEVEL)" >&2
    exit 1
    ;;
esac

TOOL_PARSER=${TOOL_PARSER:-qwen3_xml}
TOOL_ARGS=()
if [ "${TOOLS:-1}" = "1" ]; then
  TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
fi

# Ornith was trained thinking-off. Default 0: template thinking off + greedy
# 0.0/0.80/20. ENABLE_THINKING=1 turns thinking on (1.0/0.95/20). Per-request
# chat_template_kwargs still override the server default.
ENABLE_THINKING=${ENABLE_THINKING:-0}
case "$ENABLE_THINKING" in
  1|true|TRUE|yes|on) THINK_JSON=true ;;
  0|false|FALSE|no|off|"") THINK_JSON=false ;;
  *)
    echo "ENABLE_THINKING must be 0 or 1 (got: $ENABLE_THINKING)" >&2
    exit 1
    ;;
esac
THINK_ARGS=(--default-chat-template-kwargs "{\"enable_thinking\": ${THINK_JSON}}")
if [ "$THINK_JSON" = true ]; then
  THINK_ARGS+=(--override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}')
else
  THINK_ARGS+=(--override-generation-config '{"temperature":0.0,"top_p":0.80,"top_k":20}')
fi

if [ "${VISION:-0}" = "1" ]; then
  VISION_ARGS='--limit-mm-per-prompt {"image":{"count":1}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'
  [ "${VISION_OFFLOAD:-1}" = "1" ] && export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
else
  VISION_ARGS="--language-model-only"
fi

if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
  ALLOC_DEFAULT=expandable_segments:False
else
  ALLOC_DEFAULT=expandable_segments:True
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-$ALLOC_DEFAULT}

if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

export PATH="$REPO/venv/bin:$PATH"

if [ "$SLEEP_LEVEL" = "1" ] || [ "$SLEEP_LEVEL" = "2" ]; then
  export VLLM_IDLE_BIND_HOST="$HOST"
  export VLLM_IDLE_BIND_PORT="$PORT"
  HOST=127.0.0.1
  PORT=${VLLM_ENGINE_PORT:-$((VLLM_IDLE_BIND_PORT + 1))}
  export VLLM_ENGINE_PORT="$PORT"
  export VLLM_IDLE_UPSTREAM="http://127.0.0.1:${PORT}"
fi

echo "=== Starting Ornith-1.5-9B batch server ==="
echo "Model:        $MODEL"
echo "Served as:    $SERVED_MODEL_NAME"
if [ "$SLEEP_LEVEL" = "1" ] || [ "$SLEEP_LEVEL" = "2" ]; then
  echo "Port:         $VLLM_IDLE_BIND_PORT"
  echo "Idle proxy:   ${VLLM_IDLE_BIND_HOST}:${VLLM_IDLE_BIND_PORT} -> 127.0.0.1:${PORT} (timeout ${VLLM_IDLE_TIMEOUT}s)"
else
  echo "Port:         $PORT"
fi
echo "Max Seqs:     $MAX_SEQS"
echo "Context:      $MAX_LEN"
echo "KV Cache:     $KV"
echo "Thinking:     $THINK_JSON (ENABLE_THINKING=$ENABLE_THINKING)"
echo "Sleep level:  $SLEEP_LEVEL"
echo "=========================================="

exec bash "$REPO/docker/run_vllm.sh" venv/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" --port "$PORT" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --api-server-count "$API_SERVERS" \
  ${VISION_ARGS} \
  $KV_ARGS \
  --mamba-ssm-cache-dtype float16 \
  --max-num-batched-tokens 2048 \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  "${THINK_ARGS[@]}" \
  "${TOOL_ARGS[@]}" \
  "${SLEEP_ARGS[@]}" \
  ${EXTRA_ARGS}
