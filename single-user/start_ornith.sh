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
#  - Optional DFlash2 block drafter (SPEC=dflash2) after training
#    models/Ornith-1.5-9B-DFlash2-W4A16 — not the Qwen 27B checkpoint
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
#   KV=int8pth bash single-user/start_ornith.sh    # Triton int8 KV, ~150k, k=4
#   ENABLE_THINKING=1 bash single-user/start_ornith.sh
#   SPEC=dflash2 bash single-user/start_ornith.sh   # after models/Ornith-1.5-9B-DFlash2-W4A16

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
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-ornith-1.5-9b}
PORT=${PORT:-18020}
HOST=${HOST:-0.0.0.0}
GPU_UTIL=${GPU_UTIL:-0.95}
MAX_SEQS=${MAX_SEQS:-8}
API_SERVERS=${API_SERVERS:-1}
CTX=${CTX:-fast}
SPEC=${SPEC:-mtp}
KV=${KV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
# Remember caller MAX_LEN / DRAFT_TOKENS: later profiles pick their own default
# and must not clobber an explicit MAX_LEN=8192 (same trap as start_qwen #25 item 13).
USER_MAX_LEN=${MAX_LEN:-}
USER_DRAFT_TOKENS=${DRAFT_TOKENS:-}

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
ASYNC_ARGS=()
if [ "$SPEC" = "dflash2" ] && [ "$CTX" = "long" ]; then
  ATTN_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
  export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
elif [ "$SPEC" = "dflash2" ] && [ "$CTX" = "huge" ]; then
  export VLLM_SPEC_DECODE_ATTN=0
elif [ "$SPEC" = "dflash2" ] && [ "$CTX" != "fast" ]; then
  echo "SPEC=dflash2 supports CTX=fast (bf16), CTX=long (int8), CTX=huge (KVarN); CTX=$CTX keeps SPEC=mtp" >&2
  SPEC=mtp
fi

# KV=int8pth: Triton int8 per-token-head (FlashInfer-free long context). FA2 cannot
# store quantized KV on sm86. Same bytes as fp8, ~2x the CTX=fast pool; split-KV
# verify stays on so MTP can keep k=4 (CTX=long fp8/FlashInfer cannot).
case "$KV" in
  "") ;;
  fp8)
    if [ "$CTX" = "huge" ]; then
      echo "[start_ornith] KV=fp8 is incompatible with CTX=huge (KVarN). Use CTX=long." >&2
      exit 1
    fi
    ATTN_ARGS="--kv-cache-dtype fp8"
    if [ -z "$USER_MAX_LEN" ] && [ "$SPEC" != "dflash2" ]; then
      MAX_LEN=150000
    fi
    if [ "$SPEC" = "mtp" ] && [ -z "$USER_DRAFT_TOKENS" ]; then
      DRAFT_TOKENS=3
    fi
    echo "[start_ornith] KV=fp8: FlashInfer fp8 KV, MAX_LEN=$MAX_LEN, drafts=$DRAFT_TOKENS"
    ;;
  int8pth)
    if [ "$CTX" = "huge" ]; then
      echo "[start_ornith] KV=int8pth is incompatible with CTX=huge (KVarN). Use CTX=long or CTX=fast." >&2
      exit 1
    fi
    ATTN_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
    export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
    if [ -z "$USER_MAX_LEN" ] && [ "$SPEC" != "dflash2" ]; then
      MAX_LEN=150000
    fi
    if [ "$SPEC" = "mtp" ] && [ -z "$USER_DRAFT_TOKENS" ]; then
      DRAFT_TOKENS=4
    fi
    echo "[start_ornith] KV=int8pth: Triton int8 per-token-head KV, split-KV verify on, MAX_LEN=$MAX_LEN"
    ;;
  *)
    echo "KV must be empty (follow CTX), fp8, or int8pth (got: $KV)" >&2
    exit 1
    ;;
esac

if [ "$SPEC" = "mtp" ]; then
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  SPEC_ARGS=(--speculative-config "$SPEC_CFG")
  export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
  export MTP_DRAFT_VOCAB=${MTP_DRAFT_VOCAB:-1}
elif [ "$SPEC" = "off" ] || [ "$SPEC" = "none" ]; then
  SPEC_ARGS=()
elif [ "$SPEC" = "dflash2" ]; then
  if [ -z "${DRAFT:-}" ]; then
    for d in Ornith-1.5-9B-DFlash2-W4A16 Ornith-1.5-9B-DFlash2; do
      cfg="$REPO/models/$d/config.json"
      [ -f "$cfg" ] || continue
      if [ -f "$REPO/models/$d/model.safetensors" ] || [ -f "$REPO/models/$d/model.safetensors.index.json" ]; then
        DRAFT=$REPO/models/$d
        break
      fi
    done
  fi
  [ -n "${DRAFT:-}" ] || {
    echo "SPEC=dflash2 needs an Ornith DFlash2 drafter (not Qwen3.8-27B-DFlash2)." >&2
    echo "Train: bash drafter/train_dflash2.sh   Quantize: drafter/quant_dflash2.py" >&2
    echo "Or:    venv/bin/python prepare/fetch_dflash2.py" >&2
    exit 1
  }
  if [ ! -f "$DRAFT/model.safetensors" ] && [ ! -f "$DRAFT/model.safetensors.index.json" ]; then
    echo "SPEC=dflash2: $DRAFT has config.json but no weights. Train or fetch W4A16." >&2
    exit 1
  fi
  python3 - "$DRAFT" <<'PY' || exit 1
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
h = c.get("hidden_size"); n = c.get("num_target_layers")
taps = (c.get("dflash_config") or {}).get("target_layer_ids") or []
if h != 4096 or (n is not None and n != 32) or (taps and max(taps) >= 32):
    sys.stderr.write(
        f"refusing {sys.argv[1]}: hidden={h} num_target_layers={n} taps={taps}\n"
        "Ornith-1.5-9B needs hidden 4096, 32 target layers, taps < 32.\n"
        "Do not fetch syvai/Qwen3.8-27B-DFlash2-W4A16.\n")
    sys.exit(1)
if (c.get("architectures") or [None])[0] != "DFlash2DraftModel":
    sys.stderr.write(f"refusing {sys.argv[1]}: architectures={c.get('architectures')}\n")
    sys.exit(1)
if c.get("is_causal") is not False:
    sys.stderr.write(f"refusing {sys.argv[1]}: is_causal={c.get('is_causal')} (must be false for DFlash2)\n")
    sys.exit(1)
PY
  export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}
  # V2 runner UVA on WSL2. Leave unset on bare metal unless the caller wants it.
  if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
    export VLLM_WSL2_ENABLE_PIN_MEMORY=${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}
  fi
  DRAFT_TOKENS=${DFLASH_TOKENS:-7}
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  SPEC_ARGS=(--speculative-config "$SPEC_CFG")
  export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
  if [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ]; then
    ASYNC_SCHED=${ASYNC_SCHED:-0}
  fi
  if [ "$CTX" = "huge" ]; then
    MAX_SEQS=${MAX_SEQS:-2}
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      MAX_LEN=${DFLASH_MAX_LEN:-${USER_MAX_LEN:-221184}}
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
    else
      MAX_LEN=${DFLASH_MAX_LEN:-${USER_MAX_LEN:-245760}}
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
    fi
  elif [ "$CTX" = "long" ]; then
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-${USER_MAX_LEN:-131072}}
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
    else
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
    fi
  elif [ "$DRAFT_TOKENS" -gt 7 ]; then
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-${USER_MAX_LEN:-57344}}
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
  else
    MAX_LEN=${DFLASH_MAX_LEN:-${USER_MAX_LEN:-65536}}
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
  fi
  MAX_SEQS=${MAX_SEQS:-8}
  if [ $((MAX_SEQS * (DRAFT_TOKENS + 1))) -gt 64 ]; then
    CG=${CG:-64}
  else
    CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1)))}
  fi
  # Ornith weights are ~8.5 GB; do not copy the 27B 5.2 GiB KV_MEM pin. Size the
  # pool from GPU_UTIL unless the caller sets KV_MEM explicitly.
  if [ -n "${KV_MEM:-}" ]; then
    EXTRA_ARGS="--kv-cache-memory=$KV_MEM ${EXTRA_ARGS}"
  fi
  [ "${ASYNC_SCHED:-1}" = 1 ] && ASYNC_ARGS=(--async-scheduling)
else
  echo "Unsupported SPEC=$SPEC for Ornith-1.5-9B (use 'mtp', 'dflash2', or 'off')" >&2
  exit 1
fi

# Prefix Caching (defaults to ON for agent / code workflows)
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

# Tool & Reasoning parser configuration
# Ornith uses Qwen3 XML format (<tool_call><function=...><parameter=...>)
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

if [ "$SLEEP_LEVEL" = "1" ] || [ "$SLEEP_LEVEL" = "2" ]; then
  export VLLM_IDLE_BIND_HOST="$HOST"
  export VLLM_IDLE_BIND_PORT="$PORT"
  HOST=127.0.0.1
  PORT=${VLLM_ENGINE_PORT:-$((VLLM_IDLE_BIND_PORT + 1))}
  export VLLM_ENGINE_PORT="$PORT"
  export VLLM_IDLE_UPSTREAM="http://127.0.0.1:${PORT}"
fi

echo "=== Starting Ornith-1.5-9B single-user server ==="
echo "Model:        $MODEL"
echo "Served as:    $SERVED_MODEL_NAME"
if [ "$SLEEP_LEVEL" = "1" ] || [ "$SLEEP_LEVEL" = "2" ]; then
  echo "Port:         $VLLM_IDLE_BIND_PORT"
  echo "Idle proxy:   ${VLLM_IDLE_BIND_HOST}:${VLLM_IDLE_BIND_PORT} -> 127.0.0.1:${PORT} (timeout ${VLLM_IDLE_TIMEOUT}s)"
else
  echo "Port:         $PORT"
fi
echo "Context:      $MAX_LEN tokens (mode: $CTX${KV:+, KV=$KV})"
echo "Speculation:  $SPEC (draft tokens: $DRAFT_TOKENS)"
[ "$SPEC" = "dflash2" ] && echo "Drafter:      $DRAFT"
echo "Prefix Cache: $PREFIX_CACHE"
echo "Thinking:     $THINK_JSON (ENABLE_THINKING=$ENABLE_THINKING)"
echo "GPU util:     $GPU_UTIL"
echo "Sleep level:  $SLEEP_LEVEL"
echo "==============================================="

exec bash "$REPO/docker/run_vllm.sh" venv/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
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
  "${THINK_ARGS[@]}" \
  "${TOOL_ARGS[@]}" \
  "${SLEEP_ARGS[@]}" \
  "${ASYNC_ARGS[@]}" \
  ${EXTRA_ARGS}
