"""Initialize a randomly-weighted Ornith DFlash2DraftModel checkpoint for vLLM smoke loads.

This is NOT a trained drafter. Acceptance will be noise. Use it to verify:
  - config.json shapes (fc 20480→4096, taps in 0..31, is_causal false)
  - vLLM DFlash2DraftModel load next to Ornith
  - aux-hidden concat width 5*4096

  python drafter/init_dflash2.py
  python drafter/init_dflash2.py --full-attn-taps   # ablation taps 3/11/19/27/31
  python drafter/init_dflash2.py --dst models/Ornith-1.5-9B-DFlash2

Writes config.json + model.safetensors and copies tokenizer files from the teacher.
"""
from __future__ import annotations

import os, sys, shutil, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dflash2_const import (
    BLOCK_SIZE, CONV_GROUP_SIZE, CONV_KERNEL_SIZE, HEAD_DIM, HIDDEN_SIZE,
    INTERMEDIATE_SIZE, MASK_TOKEN_ID, NUM_ATTENTION_HEADS, NUM_HIDDEN_LAYERS,
    NUM_KEY_VALUE_HEADS, TARGET_LAYER_IDS, TARGET_LAYER_IDS_FULL_ATTN,
    VOCAB_SIZE, assert_teacher_compatible, fc_in_features, kernel_projection_out,
    layer_kind, write_config,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=os.path.join(REPO, "models", "Ornith-1.5-9B-MixedInt4-AutoRound"))
parser.add_argument("--dst", default=os.path.join(REPO, "models", "Ornith-1.5-9B-DFlash2"))
parser.add_argument("--full-attn-taps", action="store_true")
parser.add_argument("--config-only", action="store_true",
                    help="write config.json and tokenizer copies; skip random weights")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

taps = TARGET_LAYER_IDS_FULL_ATTN if args.full_attn_taps else TARGET_LAYER_IDS
assert_teacher_compatible(args.model, taps)
cfg_path = write_config(args.dst, taps)
print("config", cfg_path, "taps", taps)
for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "preprocessor_config.json", "processor_config.json", "chat_template.jinja"):
    src = os.path.join(args.model, name)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(args.dst, name))
if args.config_only:
    if not args.full_attn_taps:
        shutil.copy2(cfg_path, os.path.join(HERE, "ornith_dflash2.json"))
    print("config-only; pass without --config-only to write random bf16 weights")
    sys.exit(0)

import torch
from safetensors.torch import save_file

g = torch.Generator().manual_seed(args.seed)


def t(*shape):
    w = torch.empty(*shape, dtype=torch.bfloat16)
    torch.nn.init.trunc_normal_(w, mean=0.0, std=0.02, a=-0.04, b=0.04, generator=g)
    return w.contiguous()


def ones(*shape):
    return torch.ones(*shape, dtype=torch.bfloat16)


H, I, V = HIDDEN_SIZE, INTERMEDIATE_SIZE, VOCAB_SIZE
Q = NUM_ATTENTION_HEADS * HEAD_DIM
KV = NUM_KEY_VALUE_HEADS * HEAD_DIM
fc_in = fc_in_features(len(taps))
kproj = kernel_projection_out()
tensors = {
    "embed_tokens.weight": t(V, H),
    "lm_head.weight": t(V, H),
    "fc.weight": t(H, fc_in),
    "hidden_norm.weight": ones(H),
    "norm.weight": ones(H),
    "candidate_selector.hidden_projection.weight": t(256, H),
    "candidate_selector.predecessor_codebook": t(V, 256),
    "candidate_selector.successor_codebook": t(V, 256),
}
for i in range(NUM_HIDDEN_LAYERS):
    p = f"layers.{i}"
    tensors.update({
        f"{p}.self_attn.q_proj.weight": t(Q, H),
        f"{p}.self_attn.k_proj.weight": t(KV, H),
        f"{p}.self_attn.v_proj.weight": t(KV, H),
        f"{p}.self_attn.o_proj.weight": t(H, Q),
        f"{p}.self_attn.q_norm.weight": ones(HEAD_DIM),
        f"{p}.self_attn.k_norm.weight": ones(HEAD_DIM),
        f"{p}.mlp.gate_proj.weight": t(I, H),
        f"{p}.mlp.up_proj.weight": t(I, H),
        f"{p}.mlp.down_proj.weight": t(H, I),
        f"{p}.input_layernorm.weight": ones(H),
        f"{p}.post_attention_layernorm.weight": ones(H),
        f"{p}.attention_conv.base_kernel": t(2, CONV_KERNEL_SIZE, H),
        f"{p}.attention_conv.kernel_projection.weight": t(kproj, H),
        f"{p}.mlp_conv.base_kernel": t(2, CONV_KERNEL_SIZE, H),
        f"{p}.mlp_conv.kernel_projection.weight": t(kproj, H),
    })

out = os.path.join(args.dst, "model.safetensors")
save_file(tensors, out, metadata={"format": "pt"})

nbytes = os.path.getsize(out)
print(f"wrote {args.dst}")
print(f"  taps={taps} ({', '.join(layer_kind(i) for i in taps)})")
print(f"  fc {H} x {fc_in}  mask_token_id={MASK_TOKEN_ID}  block_size={BLOCK_SIZE}")
print(f"  conv_group_size={CONV_GROUP_SIZE} kernel_projection_out={kproj}")
print(f"  tensors={len(tensors)}  {nbytes / 2**30:.2f} GiB")
print("this is random init — train before measuring acceptance")
