"""Verify Ornith DFlash2 config + optional vLLM load.

Default (no GPU): check teacher vs draft geometry, print GDN vs full-attn tap kinds,
assert fc in_features == 5*4096 and mask_token_id is not audio/pad/eos.

  python drafter/smoke_dflash2.py
  python drafter/smoke_dflash2.py --init          # write random-init checkpoint first
  python drafter/smoke_dflash2.py --load          # load draft+teacher in vLLM (needs GPU)
"""
from __future__ import annotations

import json, os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dflash2_const import (
    FULL_ATTENTION_LAYERS, HIDDEN_SIZE, MASK_TOKEN_ID, NUM_TARGET_LAYERS,
    TARGET_LAYER_IDS, TARGET_LAYER_IDS_FULL_ATTN, VOCAB_SIZE,
    assert_teacher_compatible, fc_in_features, layer_kind,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=os.path.join(REPO, "models", "Ornith-1.5-9B-MixedInt4-AutoRound"))
parser.add_argument("--draft", default=os.path.join(REPO, "models", "Ornith-1.5-9B-DFlash2"))
parser.add_argument("--init", action="store_true")
parser.add_argument("--load", action="store_true")
parser.add_argument("--full-attn-taps", action="store_true")
args = parser.parse_args()

taps = TARGET_LAYER_IDS_FULL_ATTN if args.full_attn_taps else TARGET_LAYER_IDS
assert_teacher_compatible(args.model, taps)

print("teacher", args.model)
print(f"  hidden={HIDDEN_SIZE} layers={NUM_TARGET_LAYERS} vocab={VOCAB_SIZE}")
print("  full-attention layers:", FULL_ATTENTION_LAYERS)
print("taps", taps)
for i in taps:
    print(f"  layer {i}: {layer_kind(i)}")
n_lin = sum(1 for i in taps if layer_kind(i) == "linear_attention")
if n_lin:
    print(f"note: {n_lin}/{len(taps)} taps are GDN linear-attention. If vLLM aux-hidden "
          "hooks skip those layers, fc starves — retry --full-attn-taps ([3,11,19,27,31]).")
print(f"fc in_features={fc_in_features(len(taps))} (expect {len(taps)*HIDDEN_SIZE})")
print(f"mask_token_id={MASK_TOKEN_ID}")
if MASK_TOKEN_ID in (248044, 248046, 248070):
    raise SystemExit("mask_token_id collides with pad/eos/audio_start")

if args.init:
    import subprocess
    cmd = [sys.executable, os.path.join(HERE, "init_dflash2.py"),
           "--model", args.model, "--dst", args.draft]
    if args.full_attn_taps:
        cmd.append("--full-attn-taps")
    subprocess.check_call(cmd)

cfg_path = os.path.join(args.draft, "config.json")
if not os.path.isfile(cfg_path):
    cfg_path = os.path.join(HERE, "ornith_dflash2.json")
if not os.path.isfile(cfg_path):
    print(f"no {args.draft}/config.json or drafter/ornith_dflash2.json; run with --init")
    sys.exit(0 if not args.load else 1)

cfg = json.load(open(cfg_path))
dcfg = cfg.get("dflash_config") or {}
assert cfg.get("architectures") == ["DFlash2DraftModel"], cfg.get("architectures")
assert cfg.get("is_causal") is False, "is_causal must be false (else vLLM drafts as DFlash1)"
assert cfg.get("hidden_size") == HIDDEN_SIZE
assert cfg.get("num_target_layers") == NUM_TARGET_LAYERS
assert dcfg.get("mask_token_id") == MASK_TOKEN_ID
on_disk = dcfg.get("target_layer_ids")
if on_disk != taps:
    print(f"WARNING: on-disk taps {on_disk} != requested {taps}")
    if args.load:
        raise SystemExit("refusing --load with tap mismatch; pass matching --full-attn-taps or re-init")
else:
    assert on_disk == taps
assert dcfg.get("block_size") == 8
print("config.json ok:", cfg_path)

if not args.load:
    print("skip vLLM load (pass --load to boot draft+teacher)")
    sys.exit(0)

st = os.path.join(args.draft, "model.safetensors")
idx = os.path.join(args.draft, "model.safetensors.index.json")
if not os.path.isfile(st) and not os.path.isfile(idx):
    print(f"no draft weights under {args.draft} (need model.safetensors)")
    print("train first, or: python drafter/init_dflash2.py  # random-init smoke only")
    sys.exit(1)

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams

llm = LLM(
    model=args.model,
    served_model_name="ornith-1.5-9b",
    speculative_config={
        "method": "dflash",
        "model": args.draft,
        "num_speculative_tokens": 7,
        "draft_sample_method": "probabilistic",
    },
    enforce_eager=True,
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.90")),
    max_model_len=int(os.environ.get("MAX_LEN", "2048")),
    max_num_seqs=1,
    language_model_only=True,
    enable_prefix_caching=False,
    mamba_ssm_cache_dtype="float16",
)
core = llm.llm_engine.engine_core
core = getattr(core, "engine_core", core)
runner = core.model_executor.driver_worker.worker.model_runner

def find_draft(root):
    seen, queue = set(), [root]
    while queue:
        obj = queue.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, __import__("torch").nn.Module) and "DFlash" in type(obj).__name__:
            return obj
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                queue.append(v)
    return None

import torch
draft = find_draft(runner)
assert draft is not None, "DFlash draft model not found under the runner"
print("loaded", type(draft).__name__)
fc = getattr(getattr(draft, "model", draft), "fc", None)
if fc is not None:
    inn = getattr(fc, "input_size", None) or getattr(fc, "in_features", None)
    out = getattr(fc, "output_size", None) or getattr(fc, "out_features", None)
    print(f"fc {out} x {inn}")
    assert inn == len(taps) * HIDDEN_SIZE, (inn, len(taps) * HIDDEN_SIZE)
    assert out == HIDDEN_SIZE

sp = SamplingParams(max_tokens=8, temperature=0)
outs = llm.generate(["Say hello in one word."], sp, use_tqdm=False)
print("sample:", outs[0].outputs[0].text[:80])
print("vLLM DFlash2 smoke ok")
