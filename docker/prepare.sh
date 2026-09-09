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

# If corrupted compressed-tensors head packings are present from previous runs, restore pristine AutoRound files:
python - "$BASE" <<'EOF'
import os, sys, shutil, json
d = sys.argv[1].rstrip("/") + "/"
idx_path = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx_path):
    try:
        idx = json.load(open(idx_path)).get("weight_map", {})
        if "lm_head.weight_packed" in idx:
            print("prepare: detected incompatible lm_head.weight_packed for AutoRound; restoring pristine files...")
            for src, dst in [
                ("config.json.bak-quant", "config.json"),
                ("model.safetensors.index.json.bak-quant", "model.safetensors.index.json"),
                ("model-00001-of-00002.safetensors.bak", "model-00001-of-00002.safetensors"),
                ("model_extra_tensors.safetensors.bak-draft", "model_extra_tensors.safetensors"),
            ]:
                src_f = os.path.join(d, src)
                dst_f = os.path.join(d, dst)
                if os.path.exists(src_f):
                    shutil.copy2(src_f, dst_f)
                    print(f"  restored {dst}")
            for rm_f in ["mtp_draft_vocab_ids.pt", "model-00001-of-00002.safetensors.bak_embed"]:
                p = os.path.join(d, rm_f)
                if os.path.exists(p):
                    os.remove(p)
            print("prepare: restore complete.")
    except Exception as e:
        print(f"prepare: restore check warning: {e}")
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
# AutoRound models (like Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound) keep lm_head
# and embed_tokens in BF16. Compressed-tensors pack-quantization is incompatible with
# vLLM's AutoRoundConfig.
if quant_method != "auto-round":
    if "lm_head.weight_packed" not in idx: todo.append("lm_head")
    if not any(k.endswith("embed_tokens.weight_packed") for k in idx): todo.append("embed")
    if ("mtp.layers.0.mlp.down_proj.weight_packed" not in idx and
        "mtp.layers.0.mlp.down_proj.qweight" not in idx and
        "mtp.layers.0.mlp.down_proj.weight" in idx):
        todo.append("mtp")
    if "mtp.draft_lm_head.weight_packed" not in idx or not os.path.exists(d + "mtp_draft_vocab_ids.pt"):
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
