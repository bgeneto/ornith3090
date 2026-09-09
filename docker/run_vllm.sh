#!/bin/bash
# Supervise vLLM + the idle/wake proxy. Used by batch/ and single-user start_ornith.sh.
#
# SLEEP_LEVEL=0 (or empty): exec the engine on the public socket, no proxy.
# SLEEP_LEVEL=1|2: vLLM is already rewritten to 127.0.0.1:ENGINE_PORT; this
# script binds the proxy on VLLM_IDLE_BIND_HOST:VLLM_IDLE_BIND_PORT.
#
# If the engine dies before it ever answers /health (typical WSL2 CuMemAllocator
# crash at --enable-sleep-mode), retry once without sleep mode on the public
# port so compose up still serves.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
PROXY_PY="$DIR/vllm-idle-proxy.py"

if [ $# -lt 1 ]; then
  echo "usage: $0 <vllm serve ...>" >&2
  exit 2
fi

SLEEP_LEVEL=${SLEEP_LEVEL:-1}

if [ "$SLEEP_LEVEL" = "0" ] || [ -z "$SLEEP_LEVEL" ]; then
  exec "$@"
fi

if [ ! -f "$PROXY_PY" ]; then
  echo "[run_vllm] missing $PROXY_PY" >&2
  exit 1
fi

if [ -x "$REPO/venv/bin/python" ]; then
  PY="$REPO/venv/bin/python"
else
  PY="$(command -v python3)"
fi

BIND_HOST=${VLLM_IDLE_BIND_HOST:-0.0.0.0}
BIND_PORT=${VLLM_IDLE_BIND_PORT:-8000}
ENGINE_PORT=${VLLM_ENGINE_PORT:-$((BIND_PORT + 1))}
UPSTREAM=${VLLM_IDLE_UPSTREAM:-http://127.0.0.1:${ENGINE_PORT}}

export SLEEP_LEVEL
export VLLM_IDLE_TIMEOUT=${VLLM_IDLE_TIMEOUT:-90}
export VLLM_IDLE_BIND_HOST="$BIND_HOST"
export VLLM_IDLE_BIND_PORT="$BIND_PORT"
export VLLM_ENGINE_PORT="$ENGINE_PORT"
export VLLM_IDLE_UPSTREAM="$UPSTREAM"

VLLM_PID=""
PROXY_PID=""
ENGINE_READY=0
CLEANING=0

kill_pid() {
  local pid="${1:-}"
  [ -n "$pid" ] || return 0
  kill -TERM "$pid" 2>/dev/null || return 0
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.5
  done
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
  [ "$CLEANING" = "0" ] || return 0
  CLEANING=1
  trap - TERM INT EXIT
  kill_pid "${PROXY_PID:-}"
  kill_pid "${VLLM_PID:-}"
  wait "${PROXY_PID:-}" 2>/dev/null || true
  wait "${VLLM_PID:-}" 2>/dev/null || true
}

build_fallback_argv() {
  FALLBACK_ARGV=()
  local skip=""
  local a
  for a in "$@"; do
    if [ "$skip" = "host" ]; then
      FALLBACK_ARGV+=("$BIND_HOST")
      skip=""
      continue
    fi
    if [ "$skip" = "port" ]; then
      FALLBACK_ARGV+=("$BIND_PORT")
      skip=""
      continue
    fi
    if [ "$a" = "--enable-sleep-mode" ]; then
      continue
    fi
    if [ "$a" = "--host" ]; then
      FALLBACK_ARGV+=("$a")
      skip="host"
      continue
    fi
    if [ "$a" = "--port" ]; then
      FALLBACK_ARGV+=("$a")
      skip="port"
      continue
    fi
    FALLBACK_ARGV+=("$a")
  done
}

engine_health_ok() {
  "$PY" -c '
import os, sys, urllib.request
url = os.environ["VLLM_IDLE_UPSTREAM"].rstrip("/") + "/health"
try:
    urllib.request.urlopen(url, timeout=1)
except Exception:
    sys.exit(1)
'
}

echo "[run_vllm] engine ${UPSTREAM} ; public ${BIND_HOST}:${BIND_PORT} ; idle ${VLLM_IDLE_TIMEOUT}s level ${SLEEP_LEVEL}"

"$@" &
VLLM_PID=$!
sleep 0.3
if ! kill -0 "$VLLM_PID" 2>/dev/null; then
  status=0
  wait "$VLLM_PID" || status=$?
  echo "[run_vllm] vLLM exited immediately (status $status) with sleep mode enabled" >&2
  cleanup
  echo "[run_vllm] retrying without --enable-sleep-mode on ${BIND_HOST}:${BIND_PORT}" >&2
  unset VLLM_SERVER_DEV_MODE
  build_fallback_argv "$@"
  exec "${FALLBACK_ARGV[@]}"
fi

"$PY" "$PROXY_PY" &
PROXY_PID=$!
sleep 0.3
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
  echo "[run_vllm] idle proxy exited immediately" >&2
  cleanup
  exit 1
fi

trap 'cleanup; exit 143' TERM INT
trap cleanup EXIT

while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    status=0
    wait "$VLLM_PID" || status=$?
    if [ "$ENGINE_READY" = "0" ]; then
      echo "[run_vllm] vLLM died before /health (status $status); sleep mode likely failed" >&2
      CLEANING=0
      cleanup
      echo "[run_vllm] retrying without --enable-sleep-mode on ${BIND_HOST}:${BIND_PORT}" >&2
      unset VLLM_SERVER_DEV_MODE
      unset SLEEP_LEVEL
      build_fallback_argv "$@"
      exec "${FALLBACK_ARGV[@]}"
    fi
    echo "[run_vllm] vLLM exited (status $status)" >&2
    CLEANING=0
    cleanup
    exit "$status"
  fi
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    status=0
    wait "$PROXY_PID" || status=$?
    echo "[run_vllm] idle proxy exited (status $status)" >&2
    CLEANING=0
    cleanup
    exit "$status"
  fi
  if [ "$ENGINE_READY" = "0" ] && engine_health_ok; then
    ENGINE_READY=1
    echo "[run_vllm] engine /health 200"
  fi
  sleep 1
done
