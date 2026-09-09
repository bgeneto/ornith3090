#!/bin/bash
# Compare Ornith MTP k=4 vs DFlash2 after a trained drafter is served.
# Requires a running server on PORT (default 18020).
#
#   SPEC=mtp DRAFT_TOKENS=4 bash single-user/start_ornith.sh
#   bash bench/dflash2_vs_mtp.sh mtp
#   # restart with SPEC=dflash2, then:
#   bash bench/dflash2_vs_mtp.sh dflash2
#   SPEC=dflash2 DFLASH_TOKENS=15 ...  then:
#   bash bench/dflash2_vs_mtp.sh dflash2-copy
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"
TAG=${1:-dflash2}
PORT=${PORT:-18020}
export PATH="$REPO/venv/bin:$PATH"

curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || {
  echo "no server on :$PORT — start with SPEC=mtp or SPEC=dflash2" >&2
  exit 1
}

echo "== C1 chat  (bench/run_benchmarks.sh single)  tag=$TAG"
bash "$HERE/run_benchmarks.sh" single || true

if [ "$TAG" != "mtp" ]; then
  echo "== copy/quote suite (bench/labd_bench.py)  tag=$TAG"
  python "$HERE/labd_bench.py" "$TAG" --ctx 20000 || true
fi

echo "done. Keep MTP as default unless C1 tok/step and tok/s beat DRAFT_TOKENS=4."
