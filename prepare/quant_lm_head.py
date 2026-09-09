"""Requantize lm_head to int8 (group-128, symmetric) in place.

Two on-disk layouts, chosen from quantization_config.quant_method:

  auto-round  -> AutoGPTQ tensors (qweight / scales / qzeros / g_idx).
                Required for Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound:
                vLLM INC builds ParallelLMHead as AutoGPTQLinearMethod.
  anything else -> compressed-tensors pack-quantized (weight_packed / ...),
                the original Qwen3.8-27B W4A16 path.

Usage: python prepare/quant_lm_head.py /path/to/model

Rewrites the shard containing lm_head.weight, the safetensors index and
config.json. Backups are written next to the originals (.bak / .bak-quant).
"""

import copy
import json
import os
import shutil
import sys

from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_io import classify_lm_head, restore_shard_from_bak, safetensors_keys, stream_rewrite, stream_replace
from gptq_pack import extra_bits8, is_autoround, pack_ct_from_open, pack_gptq_from_open

GROUP = 128
BITS = 8
KEY = "lm_head.weight"

d = sys.argv[1].rstrip("/") + "/"

idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
c = json.load(open(d + "config.json"))
qc = c.setdefault("quantization_config", {})
autoround = is_autoround(qc)

shard = (
    wm.get(KEY)
    or wm.get("lm_head.qweight")
    or wm.get("lm_head.weight_packed")
    or "model-00001-of-00002.safetensors"
)

def _sync_gptq_index():
    if KEY in wm:
        del wm[KEY]
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        wm.pop(f"lm_head.{s}", None)
    for s in ("qweight", "scales", "qzeros", "g_idx"):
        wm[f"lm_head.{s}"] = shard
    json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)
    extra = qc.setdefault("extra_config", {})
    extra.update(extra_bits8("lm_head"))
    json.dump(c, open(d + "config.json", "w"), indent=2)


if "lm_head.qweight" in wm:
    print("lm_head already GPTQ-quantized (lm_head.qweight present); nothing to do")
    sys.exit(0)

sk = safetensors_keys(d + shard) if os.path.exists(d + shard) else []
kind = classify_lm_head(sk)
if kind == "gptq":
    print(f"lm_head.qweight already in {shard}; repairing the safetensors index")
    _sync_gptq_index()
    sys.exit(0)
if kind == "packed" and not autoround:
    print("lm_head already pack-quantized in the shard; nothing to do")
    sys.exit(0)
if kind != "bf16":
    print(f"{shard} lm_head layout is {kind} (index lists {KEY!r}); restoring .bak")
    if not restore_shard_from_bak(d, shard):
        sys.exit(
            f"{KEY} missing from {shard} (lm_head layout={kind}, keys="
            f"{[k for k in sk if k.startswith('lm_head.')] or 'none'}). "
            "Restore model-*.safetensors.bak and the *.bak-quant index/config."
        )
    sk = safetensors_keys(d + shard)
    kind = classify_lm_head(sk)
if kind != "bf16":
    sys.exit(
        f"{KEY} still missing from {shard} after restore (layout={kind})."
    )
if KEY not in wm:
    wm[KEY] = shard

print(f"{KEY} lives in {shard}  (layout: {'AutoGPTQ' if autoround else 'compressed-tensors'})")

# Row-chunked quant + byte-copy of the rest of the shard. Loading the 5 GB
# file into a dict (then a float32 copy of lm_head) is what OOM-killed Docker.
with safe_open(d + shard, framework="pt") as f:
    if autoround:
        qweight, scales, qzeros, g_idx, err = pack_gptq_from_open(f, KEY, BITS, GROUP)
        add = {
            "lm_head.qweight": qweight,
            "lm_head.scales": scales,
            "lm_head.qzeros": qzeros,
            "lm_head.g_idx": g_idx,
        }
        packed_keys = ("qweight", "scales", "qzeros", "g_idx")
        print(
            f"round-trip relative error: {err:.4f}\n"
            f"wrote AutoGPTQ lm_head: qweight {tuple(qweight.shape)} {qweight.dtype}, "
            f"scales {tuple(scales.shape)} {scales.dtype}"
        )
    else:
        packed, scale, shape, err = pack_ct_from_open(f, KEY, BITS, GROUP)
        add = {
            "lm_head.weight_packed": packed,
            "lm_head.weight_scale": scale,
            "lm_head.weight_shape": shape,
        }
        packed_keys = ("weight_packed", "weight_scale", "weight_shape")
        print(f"round-trip relative error: {err:.4f}")
assert err < 0.01, "quantization error too high, aborting"

bak = d + shard + ".bak"
if not os.path.exists(bak):
    os.replace(d + shard, bak)
    stream_rewrite(bak, d + shard + ".tmp-quant", drop={KEY}, add=add)
    os.replace(d + shard + ".tmp-quant", d + shard)
    try:
        os.chmod(d + shard, 0o644)
    except OSError:
        pass
else:
    stream_replace(d + shard, drop={KEY}, add=add)
del add

del wm[KEY]
for s in packed_keys:
    wm[f"lm_head.{s}"] = shard

shutil.copy(d + "model.safetensors.index.json", d + "model.safetensors.index.json.bak-quant")
json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)

shutil.copy(d + "config.json", d + "config.json.bak-quant")
if "ignore" in qc:
    qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head"]
    for m in (
        "mtp.fc",
        "mtp.layers.0.mlp.down_proj",
        "mtp.layers.0.mlp.gate_proj",
        "mtp.layers.0.mlp.up_proj",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.self_attn.k_proj",
        "mtp.layers.0.self_attn.v_proj",
        "mtp.layers.0.self_attn.o_proj",
    ):
        if m not in qc["ignore"]:
            qc["ignore"].append(m)

if autoround:
    extra = qc.setdefault("extra_config", {})
    extra.update(extra_bits8("lm_head"))
else:
    if "config_groups" not in qc:
        qc["config_groups"] = {}
    if "group_0" in qc["config_groups"]:
        g1 = copy.deepcopy(qc["config_groups"]["group_0"])
        g1["targets"] = ["re:.*lm_head$"]
        g1["weights"]["num_bits"] = BITS
        g1["weights"]["symmetric"] = True
        g1["weights"]["zp_dtype"] = None
    else:
        g1 = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["re:.*lm_head$"],
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
    qc["config_groups"]["group_1"] = g1
    if "extra_config" in qc and isinstance(qc["extra_config"], dict):
        qc["extra_config"]["lm_head"] = {"bits": BITS}

json.dump(c, open(d + "config.json", "w"), indent=2)
print("done")
