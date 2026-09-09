"""Requantize the token embedding table to int8 (group-128, symmetric),
in place. Companion to quant_lm_head.py — run that one first.

Two on-disk layouts, chosen from quantization_config.quant_method:

  auto-round  -> AutoGPTQ tensors (qweight / scales / qzeros / g_idx).
                Needs patches/inc-gptq-embed.patch so vLLM INC actually
                builds a quantized VocabParallelEmbedding (stock 0.28.0
                leaves embed_tokens unquantized and looks for .weight).
  anything else -> compressed-tensors pack-quantized (weight_packed / ...),
                which needs patches/qwen3_5-embed-quant.patch.

Usage: python prepare/quant_embed.py /path/to/model
"""

import copy
import json
import os
import sys

from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_io import (
    classify_embed,
    classify_lm_head,
    restore_shard_from_bak,
    safetensors_keys,
    stream_replace,
)
from gptq_pack import extra_bits8, is_autoround, pack_ct_from_open, pack_gptq_from_open

GROUP = 128
BITS = 8

d = sys.argv[1].rstrip("/") + "/"

idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
c = json.load(open(d + "config.json"))
qc = c.setdefault("quantization_config", {})
autoround = is_autoround(qc)

emb_q = next((k for k in wm if k.endswith("embed_tokens.qweight")), None)
emb_p = next((k for k in wm if k.endswith("embed_tokens.weight_packed")), None)
key = next((k for k in wm if k.endswith("embed_tokens.weight")), None)
shard = (
    wm.get(key)
    if key
    else wm.get(emb_q)
    if emb_q
    else wm.get(emb_p)
    if emb_p
    else "model-00001-of-00002.safetensors"
)

sk = safetensors_keys(d + shard) if os.path.exists(d + shard) else []
kind = classify_embed(sk)
if kind == "gptq":
    base = next(k[: -len(".qweight")] for k in sk if k.endswith("embed_tokens.qweight"))
    if key:
        del wm[key]
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        wm.pop(f"{base}.{s}", None)
    for s in ("qweight", "scales", "qzeros", "g_idx"):
        wm[f"{base}.{s}"] = shard
    json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)
    extra = qc.setdefault("extra_config", {})
    extra.update(extra_bits8("embed_tokens", base, "model.language_model.embed_tokens"))
    json.dump(c, open(d + "config.json", "w"), indent=2)
    print(f"embed_tokens.qweight already in {shard}; repaired the safetensors index")
    sys.exit(0)
if kind == "packed":
    if not autoround:
        print("embed_tokens already pack-quantized in the shard; nothing to do")
        sys.exit(0)
    if classify_lm_head(sk) == "gptq":
        sys.exit(
            f"{shard} has embed_tokens.weight_packed but lm_head is already GPTQ. "
            "Restore model-*.safetensors.bak and re-run quant_lm_head.py then this script."
        )
    print(f"{shard} still has embed_tokens.weight_packed; restoring .bak")
    if not restore_shard_from_bak(d, shard):
        sys.exit(
            "embed_tokens.weight_packed is compressed-tensors packing and is "
            "incompatible with AutoRound/INC. Restore the .bak shard "
            "(and config.json.bak-quant if needed), then re-run this script."
        )
    sk = safetensors_keys(d + shard)
    kind = classify_embed(sk)
if kind != "bf16":
    sys.exit(
        "embed_tokens.weight missing from "
        f"{shard} (layout={kind}). Restore model-*.safetensors.bak."
    )
key = next(k for k in sk if k.endswith("embed_tokens.weight"))
wm[key] = shard

print(f"{key} lives in {shard}  (layout: {'AutoGPTQ' if autoround else 'compressed-tensors'})")

base = key[: -len(".weight")]
with safe_open(d + shard, framework="pt") as f:
    if autoround:
        qweight, scales, qzeros, g_idx, err = pack_gptq_from_open(f, key, BITS, GROUP)
        add = {
            base + ".qweight": qweight,
            base + ".scales": scales,
            base + ".qzeros": qzeros,
            base + ".g_idx": g_idx,
        }
        packed_keys = ("qweight", "scales", "qzeros", "g_idx")
        print(
            f"round-trip relative error: {err:.4f}\n"
            f"wrote AutoGPTQ embed_tokens: qweight {tuple(qweight.shape)} {qweight.dtype}, "
            f"scales {tuple(scales.shape)} {scales.dtype}"
        )
    else:
        import torch

        packed, scale, shape, err = pack_ct_from_open(
            f, key, BITS, GROUP, scale_dtype=torch.bfloat16
        )
        add = {
            base + ".weight_packed": packed,
            base + ".weight_scale": scale,
            base + ".weight_shape": shape,
        }
        packed_keys = ("weight_packed", "weight_scale", "weight_shape")
        print(f"round-trip relative error: {err:.4f}")
assert err < 0.01, "quantization error too high, aborting"

stream_replace(d + shard, drop={key}, add=add)
del add
del wm[key]
for s in packed_keys:
    wm[f"{base}.{s}"] = shard
json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)

if autoround:
    extra = qc.setdefault("extra_config", {})
    extra.update(extra_bits8("embed_tokens", base, "model.language_model.embed_tokens"))
else:
    if "config_groups" not in qc:
        qc["config_groups"] = {}
    if "group_1" in qc["config_groups"]:
        g2 = copy.deepcopy(qc["config_groups"]["group_1"])
        g2["targets"] = ["re:.*embed_tokens$"]
        g2["weights"]["num_bits"] = BITS
    elif "group_0" in qc["config_groups"]:
        g2 = copy.deepcopy(qc["config_groups"]["group_0"])
        g2["targets"] = ["re:.*embed_tokens$"]
        g2["weights"]["num_bits"] = BITS
        g2["weights"]["symmetric"] = True
        g2["weights"]["zp_dtype"] = None
    else:
        g2 = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["re:.*embed_tokens$"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": GROUP,
                "num_bits": BITS,
                "observer": "memoryless_minmax",
                "observer_kwargs": {},
                "scale_dtype": None,
                "strategy": "group",
                "symmetric": True,
                "type": "int",
                "zp_dtype": None,
            },
        }
    qc["config_groups"]["group_2"] = g2
    if "extra_config" in qc and isinstance(qc["extra_config"], dict):
        qc["extra_config"][base] = {"bits": BITS}

json.dump(c, open(d + "config.json", "w"), indent=2)
print("done")
