#!/usr/bin/env bash
set -euo pipefail

# Validate the patches whose order and hunk metadata are part of the vLLM 0.28.0
# contract. GNU patch is intentionally permissive about offsets/fuzz; git apply
# is the stricter format check that catches a hand-edited hunk header immediately.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SOURCE=${1:?usage: bash patches/check_vllm_series.sh /path/to/vllm-v0.28.0}
VLLM_SOURCE=$(cd -- "$VLLM_SOURCE" && pwd)

git -C "$VLLM_SOURCE" rev-parse --is-inside-work-tree >/dev/null

# The patches address files relative to the installed vllm package, not to the
# checkout root. `git apply` resolves a patch path against the REPOSITORY root
# and silently skips ("Skipped patch '...'") anything outside the subdirectory
# it runs in -- so running it from the package directory checked nothing at all
# and reported OK. Run from the repository root and name the prefix explicitly.
GIT_ROOT=$(git -C "$VLLM_SOURCE" rev-parse --show-toplevel)
PREFIX=${VLLM_SOURCE#"$GIT_ROOT"/}
if [ "$PREFIX" = "$VLLM_SOURCE" ]; then PREFIX=.; fi

PATCHES=(
  vllm-pr50021-gdn-spec-bounds.patch
  dflash2-lookup-drafting.patch
  dflash2-ngram-chains.patch
  dflash2-prewarm.patch
  dflash2-z-adaptive-emitted.patch
)

apply() {  # $1 = extra git-apply flags, $2 = patch path
  git -C "$GIT_ROOT" apply $1 --whitespace=error -p1 --directory="$PREFIX" < "$2"
}

for name in "${PATCHES[@]}"; do
  patch="$HERE/patches/$name"
  echo "== git apply --check $name"
  apply --check "$patch"
  apply "" "$patch"
  # A patch that touched nothing would pass --check vacuously; require a change.
  if git -C "$GIT_ROOT" diff --quiet; then
    echo "ERROR: $name applied but changed nothing -- the paths did not resolve." >&2
    exit 1
  fi
done

git -C "$GIT_ROOT" diff --check
echo "patch integrity: OK"
