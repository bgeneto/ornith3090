#!/bin/bash
# One-shot model preparation, idempotent: the README's Setup steps (the scripts
# in prepare/) against /app/models (a bind mount / volume), each skipped when its
# result is already there. On the CPU (no GPU needed). ~19.5 GB download + a few minutes of
# requantization; a fast-variant download of ~1 GB unless FAST_VARIANT=0, and the
# ~1 GB W4A16 DFlash2 drafter (SPEC=dflash2) unless DFLASH2=0.
#
#   docker compose run --rm prepare      (also runs automatically before single/batch)
set -e
cd /app
export PATH=/app/venv/bin:$PATH
BASE=${BASE_MODEL_DIR:-/app/models/Ornith-1.5-9B-MixedInt4-AutoRound}
HF_REPO=${HF_REPO:-Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound}

# Index can list lm_head.weight while the shard still has weight_packed (a
# previous restore copied bak-quant but not the 5 GB .bak). Peek the shard
# header, not just the index.
python - "$BASE" <<'EOF'
import os, sys, json
sys.path.insert(0, "/app/prepare")
sys.path.insert(0, os.path.join(os.getcwd(), "prepare"))
from checkpoint_io import autoround_needs_bf16_restore, restore_autoround_bf16
d = sys.argv[1].rstrip("/") + "/"
cfg_path = os.path.join(d, "config.json")
idx_path = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(cfg_path) and os.path.exists(idx_path):
    try:
        qc = json.load(open(cfg_path)).get("quantization_config", {})
        idx = json.load(open(idx_path))
        if qc.get("quant_method") == "auto-round":
            why = autoround_needs_bf16_restore(d, idx)
            if why:
                print(f"prepare: restoring AutoRound BF16 backups ({why})")
                restore_autoround_bf16(d)
                print("prepare: restore complete.")
    except Exception as e:
        print(f"prepare: restore check warning: {e}")
        import traceback; traceback.print_exc()
EOF

state() {  # prints the steps still to do
python - "$BASE" <<'EOF'
import json, os, sys
d = sys.argv[1].rstrip("/") + "/"
todo = []
# tokenizer.json belongs in this list: without it transformers builds an empty
# vocabulary rather than failing, and the dir stays servable-looking all the way to
# "ReasoningConfig: failed to tokenize reasoning strings" at startup.
if not all(os.path.exists(d + f) for f in
           ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")):
    print("download"); sys.exit()
idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
if any(not os.path.exists(d + f) for f in set(idx.values())):
    print("download"); sys.exit()
if not os.path.exists(d + "model_extra_tensors.safetensors"):
    print("download"); sys.exit()

c = json.load(open(d + "config.json"))
qc = c.get("quantization_config", {})
quant_method = qc.get("quant_method", "")
# AutoRound: INT8 heads must be AutoGPTQ (qweight). compressed-tensors
# weight_packed is restored above and must not be rewritten.
# Other checkpoints keep the original pack-quantized recipe.
if quant_method == "auto-round":
    if "lm_head.qweight" not in idx: todo.append("lm_head")
    if not any(k.endswith("embed_tokens.qweight") for k in idx): todo.append("embed")
else:
    if "lm_head.weight_packed" not in idx: todo.append("lm_head")
    if not any(k.endswith("embed_tokens.weight_packed") for k in idx): todo.append("embed")
if ("mtp.layers.0.mlp.down_proj.weight_packed" not in idx and
    "mtp.layers.0.mlp.down_proj.qweight" not in idx and
    "mtp.layers.0.mlp.down_proj.weight" in idx):
    todo.append("mtp")
if (("mtp.draft_lm_head.weight_packed" not in idx and "mtp.draft_lm_head.qweight" not in idx)
        or not os.path.exists(d + "mtp_draft_vocab_ids.pt")):
    todo.append("draft")

if os.environ.get("FAST_VARIANT", "0") != "0" and not os.path.exists(d[:-1] + "-fast/model.safetensors.index.json"):
    todo.append("fast")
if os.environ.get("DFLASH2", "0") != "0" and not os.path.exists(os.path.dirname(d[:-1]) + "/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"):
    todo.append("dflash2")
print(" ".join(todo))
EOF
}

TODO=$(state)
if [ "$TODO" = "download" ]; then
  echo "== downloading $HF_REPO -> $BASE (~8.8 GB, resumable)"
  export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
  HF_ARGS=()
  [ -n "${HF_TOKEN:-}" ] && HF_ARGS+=(--token "$HF_TOKEN")
  hf download "$HF_REPO" --local-dir "$BASE" "${HF_ARGS[@]}"
  TODO=$(state)
fi
[ "$TODO" = "download" ] && { echo "prepare: download incomplete (shards missing after hf download)"; exit 1; }
for step in $TODO; do
  case $step in
    lm_head) echo "== quant_lm_head.py (int8 lm_head)";      python prepare/quant_lm_head.py "$BASE" ;;
    embed)   echo "== quant_embed.py (int8 embeddings)";     python prepare/quant_embed.py "$BASE" ;;
    mtp)     echo "== quant_mtp.py (int8 MTP module)";       python prepare/quant_mtp.py "$BASE" ;;
    draft)   echo "== build_draft_vocab.py (40k draft head)"
             python prepare/build_draft_vocab.py "$BASE" --ids prepare/draft_vocab_ids.json ;;
    fast)    echo "== fetch_fast_variant.py"
             python prepare/fetch_fast_variant.py "$BASE" "$BASE-fast" ;;
    dflash2) echo "== fetch_dflash2.py"
             python prepare/fetch_dflash2.py "$(dirname "$BASE")/Qwen3.8-27B-DFlash2-W4A16" \
               || echo "prepare: DFlash2 drafter not fetched (optional; DFLASH2=0 silences this)" ;;
  esac
done
LEFT=$(state | sed 's/\bdflash2\b//')
[ -z "${LEFT// /}" ] || { echo "prepare: steps still missing after run: $LEFT"; exit 1; }
echo "prepare: model ready at $BASE$([ "${FAST_VARIANT:-0}" != 0 ] && echo " (+ $BASE-fast)")"
