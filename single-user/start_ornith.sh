#!/bin/bash
# Ornith-1.5-9B on a single RTX 3090 — SINGLE USER / LOW LATENCY mode.
#
# Optimized configuration adapted from syv-ai/qwen38-27b-rtx3090 for Ornith-1.5-9B:
#  - Native MTP speculative decoding (k=4 draft tokens) with calibrated INT8/INT4 draft head
#  - 40k reduced draft vocabulary for fast ~0.5ms draft proposal
#  - INT8 group-128 quantized lm_head (speeds up 4096 -> 248k logits projection)
#  - INT8 group-128 quantized embed_tokens (frees ~1 GB VRAM)
#  - FP16 recurrent Gated DeltaNet state (--mamba-ssm-cache-dtype float16)
#  - Triton split-KV verification attention (patches/spec-decode-attn.patch)
#  - Sort-free top-k / fast multi-block softmax sampler (patches/sampler-small-topk-fast-softmax.patch)
#  - Hybrid prefix caching (--enable-prefix-caching --mamba-cache-mode align)
#  - Optional Marlin INT8 activation tensor-core GEMMs for prefill (INT8_ACT=int8)
#
# Measured baseline on RTX 3090:
#   Stock BF16: ~44-50 tok/s
#   Pilcothink INT4: ~90-120 tok/s
#   With INT8 lm_head + MTP k=4 + split-KV + fast sampler: ~150-190+ tok/s
#
# Usage:
#   bash single-user/start_ornith.sh
#   PREFIX_CACHE=1 DRAFT_TOKENS=4 bash single-user/start_ornith.sh
#   INT8_ACT=int8 bash single-user/start_ornith.sh   # prefill boost

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

# flashinfer version check bypass
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Unlink stale offload mmap regions if any
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[start_ornith] removing stale offload region $f"; rm -f "$f"; }
  done
fi

MODEL=${MODEL:-$REPO/models/Ornith-1.5-9B-MixedInt4-AutoRound}
PORT=${PORT:-18020}
HOST=${HOST:-0.0.0.0}
GPU_UTIL=${GPU_UTIL:-0.95}
MAX_SEQS=${MAX_SEQS:-8}
API_SERVERS=${API_SERVERS:-1}
CTX=${CTX:-fast}
SPEC=${SPEC:-mtp}

# Prefill int8 activations (Marlin W4A8)
INT8_ACT=${INT8_ACT-}
INT8_LAYERS=${INT8_LAYERS-mlp|linear_attn|self_attn}
[ -n "$INT8_ACT" ] && export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
[ -n "$INT8_ACT" ] && [ -n "$INT8_LAYERS" ] && export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS

# Context & Attention settings
if [ "$CTX" = "fast" ]; then
  MAX_LEN=${MAX_LEN:-65536}
  DRAFT_TOKENS=${DRAFT_TOKENS:-4}
  ATTN_ARGS="--attention-backend FLASH_ATTN --kv-cache-dtype bfloat16"
  export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
elif [ "$CTX" = "huge" ]; then
  MAX_LEN=${MAX_LEN:-262144}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
  export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.15}
else
  # CTX=long: fp8 KV
  MAX_LEN=${MAX_LEN:-150000}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--kv-cache-dtype fp8"
fi

# Speculative Decoding configuration
SPEC_ARGS=()
CG=${CG:-32}
if [ "$SPEC" = "mtp" ]; then
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  SPEC_ARGS=(--speculative-config "$SPEC_CFG")
  export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
  export MTP_DRAFT_VOCAB=${MTP_DRAFT_VOCAB:-0}
elif [ "$SPEC" = "off" ] || [ "$SPEC" = "none" ]; then
  SPEC_ARGS=()
else
  echo "Unsupported SPEC=$SPEC for Ornith-1.5-9B (use 'mtp' or 'off')" >&2
  exit 1
fi

# Prefix Caching (defaults to ON for agent / code workflows)
EXTRA_ARGS=${EXTRA_ARGS:-}
PREFIX_CACHE=${PREFIX_CACHE:-1}
if [ "$PREFIX_CACHE" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
fi

# CPU offload (optional for constrained VRAM e.g. 8GB GPUs)
if [ -n "${CPU_OFFLOAD_GB:-}" ] && [ "$CPU_OFFLOAD_GB" != "0" ]; then
  EXTRA_ARGS="--cpu-offload-gb $CPU_OFFLOAD_GB ${EXTRA_ARGS}"
fi

# Tool & Reasoning parser configuration
# Ornith uses Qwen3 XML format (<tool_call><function=...><parameter=...>)
TOOL_PARSER=${TOOL_PARSER:-qwen3_xml}
TOOL_ARGS=()
if [ "${TOOLS:-1}" = "1" ]; then
  TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
fi

# Vision tower configuration
# Ornith has 27 vision blocks; default to language model only for maximum speed/KV pool
if [ "${VISION:-0}" = "1" ]; then
  VISION_ARGS='--limit-mm-per-prompt {"image":{"count":1}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'
  [ "${VISION_OFFLOAD:-1}" = "1" ] && export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
else
  VISION_ARGS="--language-model-only"
fi

# Memory allocation settings
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
  ALLOC_DEFAULT=expandable_segments:False
else
  ALLOC_DEFAULT=expandable_segments:True
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-$ALLOC_DEFAULT}
export VLLM_USE_FLASHINFER_SAMPLER=0

# API key
if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

export PATH="$REPO/venv/bin:$PATH"

echo "=== Starting Ornith-1.5-9B single-user server ==="
echo "Model:        $MODEL"
echo "Port:         $PORT"
echo "Context:      $MAX_LEN tokens (mode: $CTX)"
echo "Speculation:  $SPEC (draft tokens: $DRAFT_TOKENS)"
echo "Prefix Cache: $PREFIX_CACHE"
echo "==============================================="

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name ornith-1.5-9b \
  --host "$HOST" --port "$PORT" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --api-server-count "$API_SERVERS" \
  ${VISION_ARGS} \
  $ATTN_ARGS \
  --mamba-ssm-cache-dtype float16 \
  --max-num-batched-tokens 2048 \
  "${SPEC_ARGS[@]}" \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  "${TOOL_ARGS[@]}" \
  ${EXTRA_ARGS}
