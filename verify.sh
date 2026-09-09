#!/bin/bash
# Check that this repo is installed the way the README numbers assume:
# venv + vLLM version, every compatible patch applied, the model requantized (lm_head,
# embed_tokens, MTP module, draft head), keys/files present, and — if a
# server is running — that it answers and which backend/pool it came up with.
#
#   bash verify.sh            # everything
#   bash verify.sh --no-server
#   bash verify.sh --install  # only the install (venv, vLLM, patches, KVarN): no GPU,
#                             # model or server checks — what the Docker build runs
# Exit code: 0 all PASS (WARNs allowed), 1 if anything FAILs.
# PY=/path/to/python overrides the interpreter (default: this repo's venv).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
NOSRV=0; INSTALL=0
for a in "$@"; do case "$a" in --no-server) NOSRV=1;; --install) INSTALL=1; NOSRV=1;; esac; done
FAILS=0
ok()   { printf "  PASS  %s\n" "$1"; }
warn() { printf "  WARN  %s\n" "$1"; }
fail() { printf "  FAIL  %s\n" "$1"; FAILS=$((FAILS+1)); }
MODEL=${MODEL:-$HERE/models/Ornith-1.5-9B-MixedInt4-AutoRound}
PY=${PY:-$HERE/venv/bin/python}

echo "== environment"
[ -x "$PY" ] && ok "python: $PY" || { fail "no $PY (see README Setup)"; exit 1; }
VER=$($PY -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -n1)
[ "$VER" = "0.28.0" ] && ok "vllm $VER" || warn "vllm ${VER:-missing} (patches were written against 0.28.0)"
SP=$($PY -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -n1)
[ -n "$SP" ] && [ -d "$SP" ] && ok "vllm package at $SP" || { fail "cannot import vllm with $PY"; exit 1; }
if [ $INSTALL = 0 ]; then
$PY - <<'EOF' 2>/dev/null || fail "torch cannot see a CUDA GPU"
import torch; assert torch.cuda.is_available()
p=torch.cuda.get_device_properties(0)
print(f"  PASS  GPU: {p.name}, {p.total_memory/2**30:.1f} GiB, sm{p.major}{p.minor}, torch {torch.__version__}")
EOF
command -v nvidia-smi >/dev/null && { PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1); ok "power limit ${PL} W (README numbers are at 250 W)"; }
fi
for t in triton compressed_tensors; do $PY -c "import $t" 2>/dev/null && ok "python module $t" || fail "python module $t missing"; done
# a bare `import flashinfer` passes while vLLM still falls back to torch.topk:
# has_flashinfer() additionally wants nvcc on PATH or the flashinfer-cubin
# package (#35). Test what the server will actually use.
export FLASHINFER_DISABLE_VERSION_CHECK=1  # cubin publishes 0.6.13 vs python 0.6.16.post3; the launchers export this too
$PY -c "from vllm.utils.flashinfer import has_flashinfer; assert has_flashinfer()" 2>/dev/null \
  && ok "flashinfer usable by vLLM (nvcc or flashinfer-cubin present)" \
  || fail "flashinfer unusable: DFlash2 selector will run torch.topk at ~half speed. pip install flashinfer-python flashinfer-cubin==0.6.13 (#35)" 

echo "== vLLM patches (patches/*.patch)"
# A later patch can rewrite the region an earlier one added -- both still apply, in
# order, but the earlier one's lines are no longer in the tree, so neither check
# above can see it. The later patch declares "Supersedes: <basename>" in its header;
# that only counts if the later patch is itself applied (#67 over #57).
superseded_by() {
  local target="$1" q
  for q in patches/*.patch; do
    grep -q "^Supersedes: $target\$" "$q" || continue
    patch -p1 -R --dry-run -s -d "$SP" < "$q" >/dev/null 2>&1 || $PY patches/_check_applied.py "$q" "$SP" 2>/dev/null || continue
    basename "$q"; return 0
  done
  return 1
}
# The reverse dry-run is exact, but two patches touching the same file (the DFlash2 pair)
# can no longer be reversed individually once both are applied; then look for their content.
for p in patches/*.patch; do
  if [ "$(basename "$p")" = "dflash2-backport.patch" ]; then
    ok "dflash2-backport.patch retired (DFlash2 is native in vLLM 0.28.0)"
    continue
  fi
  if patch -p1 -R --dry-run -s -d "$SP" < "$p" >/dev/null 2>&1; then ok "$(basename $p) applied"
  elif $PY patches/_check_applied.py "$p" "$SP" 2>/dev/null; then ok "$(basename $p) applied (content check; hunks overlap another patch)"
  elif s=$(superseded_by "$(basename $p)"); then ok "$(basename $p) applied (superseded by $s, which is applied)"
  elif patch -p1 -N --dry-run -s -d "$SP" < "$p" >/dev/null 2>&1; then fail "$(basename $p) NOT applied (patch -p1 -d $SP < $p)"
  else fail "$(basename $p) neither applied nor applicable — vLLM version mismatch?"; fi
done
grep -q "VLLM_MARLIN_INT8_INCLUDE_RE" "$SP/envs.py" 2>/dev/null && ok "int8 layer-select env vars registered in envs.py" || fail "envs.py lacks VLLM_MARLIN_INT8_INCLUDE_RE"

echo "== KVarN (optional, kvarn/)"
if [ -f "$SP/v1/attention/backends/kvarn_attn.py" ]; then
  if patch -p1 -R --dry-run -s -d "$SP" < kvarn/kvarn-0.28.0.patch >/dev/null 2>&1; then
    $PY -c "from vllm.v1.attention.backends.registry import AttentionBackendEnum; AttentionBackendEnum.KVARN.get_class()" 2>/dev/null && ok "KVarN backend importable, patch applied (KV=kvarn / CTX=huge available)" || fail "KVarN files present but backend does not import"
  else fail "KVarN modules present but kvarn-0.28.0.patch not applied (bash kvarn/install.sh)"; fi
  if $PY patches/_check_applied.py kvarn/kvarn-v2-runner-0.28.0.patch "$SP" >/dev/null 2>&1; then
    ok "kvarn-v2-runner-0.28.0.patch applied (SPEC=dflash2 + CTX=huge available)"
  else warn "kvarn-v2-runner-0.28.0.patch not applied (re-run bash kvarn/install.sh for DFlash2 at 240k)"; fi
else warn "KVarN not installed (optional; bash kvarn/install.sh for 262k context)"; fi

if [ $INSTALL = 0 ]; then
echo "== model at $MODEL"
if [ ! -f "$MODEL/config.json" ]; then fail "model not found (README Setup: hf download)"; else
$PY - "$MODEL" <<'EOF'
import json, os, sys
d = sys.argv[1].rstrip("/") + "/"
c = json.load(open(d + "config.json"))
qc = c.get("quantization_config", {})
groups = qc.get("config_groups", {})
ign = set(qc.get("ignore", []))
idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
F = 0
def ok(m): print("  PASS ", m)
def fail(m):
    global F
    print("  FAIL ", m); F += 1
# lm_head / embed
quant_method = qc.get("quant_method", "")
if quant_method == "auto-round":
    if "lm_head.weight" in idx:
        ok("lm_head in BF16 (native AutoRound format, full precision)")
    elif "lm_head.weight_packed" in idx:
        fail("lm_head has weight_packed which is incompatible with AutoRound quant_method")
    else:
        fail("lm_head.weight missing from index")

    if any(k.endswith("embed_tokens.weight") for k in idx):
        ok("embed_tokens in BF16 (native AutoRound format)")
    elif any(k.endswith("embed_tokens.weight_packed") for k in idx):
        fail("embed_tokens has weight_packed which is incompatible with AutoRound quant_method")
    else:
        fail("embed_tokens missing from index")
else:
    has_lm_group = any(g.get("targets") == ["re:.*lm_head$"] and g.get("weights", {}).get("num_bits") == 8 for g in groups.values()) or qc.get("extra_config", {}).get("lm_head", {}).get("bits") == 8
    if "lm_head.weight_packed" in idx and has_lm_group: ok("lm_head requantized to int8 (prepare/quant_lm_head.py)")
    else: fail("lm_head not requantized: run prepare/quant_lm_head.py")

    has_emb_group = any(g.get("targets") == ["re:.*embed_tokens$"] and g.get("weights", {}).get("num_bits") == 8 for g in groups.values()) or any("embed_tokens" in k and v.get("bits") == 8 for k, v in qc.get("extra_config", {}).items())
    if any(k.endswith("embed_tokens.weight_packed") for k in idx) and has_emb_group: ok("embed_tokens requantized to int8 (prepare/quant_embed.py)")
    else: fail("embed_tokens not requantized: run prepare/quant_embed.py")

mtp_quant = ("mtp.layers.0.mlp.down_proj.weight_packed" in idx and "mtp.layers.0.mlp.down_proj" not in ign) or ("mtp.layers.0.mlp.down_proj.qweight" in idx)
if mtp_quant: ok("MTP draft module quantized (INT4 AutoRound/GPTQ)")
else: print("  WARN  MTP module still bf16 (prepare/quant_mtp.py) — single-user mode is slower without it")

if "mtp.draft_lm_head.weight_packed" in idx and os.path.exists(d + "mtp_draft_vocab_ids.pt"):
    ok("40k-token draft head present (prepare/build_draft_vocab.py)")
else:
    ok("MTP drafts with full shared lm_head (native exact speculation)")
missing = [f for f in set(idx.values()) if not os.path.exists(d + f)]
if missing: fail(f"safetensors shards missing: {missing}")
else: ok(f"{len(set(idx.values()))} safetensors shards present")
sys.exit(1 if F else 0)
EOF
[ $? -ne 0 ] && FAILS=$((FAILS+1))
fi

echo "== single-user fast variant (optional)"
if [ -d "$HERE/models/Ornith-1.5-9B-MixedInt4-AutoRound-fast" ]; then ok "fast variant present (int4-GPTQ lm_head/MTP, own-output draft vocab)"; else warn "no models/Ornith-1.5-9B-MixedInt4-AutoRound-fast (single-user runs with base AutoRound + INT8 lm_head)"; fi

if [ $INSTALL = 0 ]; then
# A served model dir with no tokenizer.json is not an error to transformers: it hands
# back a Qwen2Tokenizer with a 1-token vocabulary that encodes everything to []. The
# server then dies far downstream on "ReasoningConfig: failed to tokenize reasoning
# strings", which names neither the dir nor the tokenizer. Encode <think> here instead.
echo "== tokenizers (every dir we serve --model from)"
$PY - "$MODEL" "$HERE/models/Ornith-1.5-9B-MixedInt4-AutoRound-fast" <<'EOF'
import os, sys
F = 0
for d in sys.argv[1:]:
    if not os.path.isfile(os.path.join(d, "config.json")):
        continue
    name = os.path.basename(d.rstrip("/"))
    try:
        from transformers import AutoTokenizer
        ids = AutoTokenizer.from_pretrained(d).encode("<think>", add_special_tokens=False)
    except Exception as e:
        print(f"  FAIL  {name}: tokenizer will not load ({type(e).__name__}: {str(e)[:70]})"); F += 1; continue
    if ids:
        print(f"  PASS  {name}: tokenizer loads, <think> -> {ids}")
    else:
        print(f"  FAIL  {name}: no usable tokenizer in the dir (encodes everything to []). "
              f"Copy tokenizer.json and tokenizer_config.json in from the base model dir, "
              f"or re-run the download. Serving this dir fails with "
              f"'ReasoningConfig: failed to tokenize reasoning strings'.")
        F += 1
sys.exit(1 if F else 0)
EOF
[ $? -ne 0 ] && FAILS=$((FAILS+1))
fi
echo "== single-user DFlash2 drafter (optional, SPEC=dflash2, Phase 2)"
if [ -f "$HERE/models/Ornith-1.5-9B-DFlash2-W4A16/config.json" ]; then
  $PY -c "import json,sys; c=json.load(open('$HERE/models/Ornith-1.5-9B-DFlash2-W4A16/config.json')); assert c['architectures']==['DFlash2DraftModel'] and c['quantization_config']['quant_method']=='compressed-tensors'" 2>/dev/null && ok "DFlash2 drafter present, W4A16 (models/Ornith-1.5-9B-DFlash2-W4A16)" || fail "models/Ornith-1.5-9B-DFlash2-W4A16 is not a quantized DFlash2DraftModel checkpoint"
  [ -f "$SP/model_executor/models/qwen3_dflash2.py" ] || fail "DFlash2 drafter present but vLLM 0.28.0 native DFlash2 support is missing"
else warn "no Ornith DFlash2 drafter (Phase 1 uses native MTP k=4; DFlash2 requires 32-layer Ornith retrained drafter)"; fi

echo "== keys / units"
# A key is optional: with neither api_key.txt nor VLLM_API_KEY the launchers export
# nothing and vLLM serves unauthenticated, which is a fine way to run this locally.
# Worth a WARN rather than silence only because both launchers bind 0.0.0.0.
[ -s api_key.txt ] || [ -n "${VLLM_API_KEY:-}" ] && ok "API key configured (api_key.txt or VLLM_API_KEY)" \
  || warn "no API key — the server will accept any request, and it listens on 0.0.0.0. Fine behind a firewall; otherwise: openssl rand -hex 24 > api_key.txt"
if [ -f /.dockerenv ]; then :; elif systemctl --user is-active ornith-serving >/dev/null 2>&1 || systemctl --user is-active qwen-serving >/dev/null 2>&1; then ok "systemd user unit active"; else warn "serving unit not active (fine if you launch the scripts by hand)"; fi
fi  # INSTALL

if [ $NOSRV = 0 ]; then
  echo "== live server (127.0.0.1:${PORT:-18020})"
  PORT=${PORT:-18020}
  if curl -sf -o /dev/null http://127.0.0.1:$PORT/health; then
    ok "/health 200"
    KEY=${VLLM_API_KEY:-$(cat api_key.txt 2>/dev/null)}
    SERVED_NAME=$(curl -s http://127.0.0.1:$PORT/v1/models -H "Authorization: Bearer $KEY" | $PY -c 'import json,sys; m=json.load(sys.stdin).get("data", [{}]); print(m[0].get("id", "ornith-1.5-9b"))' 2>/dev/null || echo "ornith-1.5-9b")
    R=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Respond with exactly the single word OK and nothing else.\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
    echo "$R" | grep -qi "ok" && ok "chat completion answers ('$(echo "$R" | $PY -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null)')" || fail "chat completion wrong/failed: $(echo "$R" | head -c 200)"
    LOG=${LOG:-$HERE/ornith.log}
    [ ! -f "$LOG" ] && [ -f "$HERE/qwen.log" ] && LOG="$HERE/qwen.log"
    if [ -f "$LOG" ]; then
      grep -oE "Using [A-Z_]+ attention backend" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -q "MarlinLinearKernel" "$LOG" && ok "Marlin kernels in use" || true
      grep -oE "capping max_num_seqs [0-9]+ -> [0-9]+" "$LOG" | tail -1 | sed 's/^/  INFO  KVarN /'
    fi
  else warn "no server on :$PORT (start batch/start_ornith.sh or single-user/start_ornith.sh, or pass --no-server)"; fi
fi
echo
[ $FAILS = 0 ] && echo "verify: OK ($FAILS failures)" || echo "verify: $FAILS FAILURE(S)"
exit $([ $FAILS = 0 ] && echo 0 || echo 1)
