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
PORT=${PORT:-18020}
HOST=${HOST:-0.0.0.0}
MAX_SEQS=${MAX_SEQS:-64}
API_SERVERS=${API_SERVERS:-1}
GPU_UTIL=${GPU_UTIL:-0.95}

KV=${KV:-fp8}
if [ "$KV" = "int4pth" ]; then
  MAX_LEN=${MAX_LEN:-262144}
  KV_ARGS="--kv-cache-dtype int4_per_token_head --attention-backend TRITON_ATTN"
elif [ "$KV" = "kvarn" ]; then
  MAX_LEN=${MAX_LEN:-262144}
  KV_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
  export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.25}
else
  # Default: fp8 KV via FlashInfer
  MAX_LEN=${MAX_LEN:-150000}
  KV_ARGS="--kv-cache-dtype fp8"
fi

# INT8 activations for Marlin W4A8 tensor-core execution
INT8_ACT=${INT8_ACT-int8}
INT8_LAYERS=${INT8_LAYERS-mlp|linear_attn|self_attn}
[ -n "$INT8_ACT" ] && export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
[ -n "$INT8_ACT" ] && [ -n "$INT8_LAYERS" ] && export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS

EXTRA_ARGS=${EXTRA_ARGS:-}
PREFIX_CACHE=${PREFIX_CACHE:-1}
if [ "$PREFIX_CACHE" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
fi

# CPU offload (optional for constrained VRAM e.g. 8GB GPUs)
if [ -n "${CPU_OFFLOAD_GB:-}" ] && [ "$CPU_OFFLOAD_GB" != "0" ]; then
  EXTRA_ARGS="--cpu-offload-gb $CPU_OFFLOAD_GB ${EXTRA_ARGS}"
fi

TOOL_PARSER=${TOOL_PARSER:-qwen3_xml}
TOOL_ARGS=()
if [ "${TOOLS:-1}" = "1" ]; then
  TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
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

echo "=== Starting Ornith-1.5-9B batch server ==="
echo "Model:        $MODEL"
echo "Port:         $PORT"
echo "Max Seqs:     $MAX_SEQS"
echo "Context:      $MAX_LEN"
echo "KV Cache:     $KV"
echo "=========================================="

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name ornith-1.5-9b \
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
  "${TOOL_ARGS[@]}" \
  ${EXTRA_ARGS}
