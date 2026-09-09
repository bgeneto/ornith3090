"""Requantize the MTP (multi-token-prediction) draft module to int8 or int4
(group-128, symmetric) in compressed-tensors pack-quantized format, in place.

The published W4A16 quant leaves the whole `mtp.*` module in bf16 (~850 MB:
mtp.fc plus one full decoder layer). In single-user mode that module runs once
per draft token, so at 4 drafts/step it is read four times per step; int8
halves that traffic, int4 quarters it. The draft head only steers speculation
(acceptance rate) — the sampled distribution stays exact either way — so this
is a pure speed knob. Measured acceptance change: int8 none.

Usage: python prepare/quant_mtp.py /path/to/Qwen3.8-27B-W4A16-AutoRound [--bits 8|4] [--keep-fc]
--keep-fc leaves mtp.fc (the 10240->5120 input projection, 105 MB) in bf16.

Rewrites model_extra_tensors.safetensors, the safetensors index and
config.json (backups next to the originals: .bak-mtp).
"""

import copy
import json
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

GROUP = 128
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 8
QMAX = 2 ** (BITS - 1) - 1
KEEP_FC = "--keep-fc" in sys.argv or "--quant-fc" not in sys.argv
ALL_MTP_LINEARS = [
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
]

d = sys.argv[1].rstrip("/") + "/"
idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]

# Check if mtp.layers.0 is already quantized (e.g. Pilcothink MixedInt4 AutoRound)
already_quant = any(
    f"{m}.qweight" in wm or f"{m}.weight_packed" in wm
    for m in ALL_MTP_LINEARS
)

MTP_LINEARS = []
if not already_quant:
    for m in ALL_MTP_LINEARS:
        if f"{m}.weight" in wm:
            MTP_LINEARS.append(m)
else:
    print("MTP decoder layer (mtp.layers.0.*) is already quantized in this checkpoint.")

if "--quant-fc" in sys.argv and "mtp.fc.weight" in wm:
    MTP_LINEARS.append("mtp.fc")
elif "mtp.fc.weight" in wm and not already_quant and not KEEP_FC:
    MTP_LINEARS.append("mtp.fc")

if not MTP_LINEARS:
    print("No unquantized MTP linears to quantize (mtp.fc kept at BF16). MTP ready.")
    sys.exit(0)

shards = {wm[m + ".weight"] for m in MTP_LINEARS}
assert len(shards) == 1, f"mtp weights span several shards: {shards}"
shard = shards.pop()
print(f"mtp linears live in {shard}, quantizing {len(MTP_LINEARS)} layers to int{BITS} g{GROUP}")

tensors = {}
with safe_open(d + shard, framework="pt") as f:
    meta = f.metadata()
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

for m in MTP_LINEARS:
    w = tensors.pop(m + ".weight").to(torch.float32)
    out_f, in_f = w.shape
    assert in_f % GROUP == 0, (m, w.shape)
    g = w.reshape(out_f, in_f // GROUP, GROUP)
    scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
    q = torch.clamp(torch.round(g / scale), -QMAX - 1, QMAX).to(torch.int8).reshape(out_f, in_f)
    deq = (q.reshape(out_f, -1, GROUP).to(torch.float32) * scale).reshape(out_f, in_f)
    err = ((deq - w).norm() / w.norm()).item()
    print(f"  {m}: {tuple(w.shape)} round-trip rel error {err:.4f}")
    tensors[m + ".weight_packed"] = pack_to_int32(q, BITS, packed_dim=1).contiguous()
    tensors[m + ".weight_scale"] = scale.squeeze(-1).to(torch.float16).contiguous()
    tensors[m + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
    del wm[m + ".weight"]
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        wm[f"{m}.{s}"] = shard

shutil.copy(d + shard, d + shard + ".bak-mtp")
save_file(tensors, d + shard, metadata=meta or {"format": "pt"})
shutil.copy(d + "model.safetensors.index.json", d + "model.safetensors.index.json.bak-mtp")
json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)

c = json.load(open(d + "config.json"))
shutil.copy(d + "config.json", d + "config.json.bak-mtp")
qc = c.setdefault("quantization_config", {})
if "ignore" in qc:
    qc["ignore"] = [i for i in qc["ignore"] if i not in MTP_LINEARS]

if "config_groups" not in qc:
    qc["config_groups"] = {}

if "group_0" in qc["config_groups"]:
    g = copy.deepcopy(qc["config_groups"]["group_0"])
    g["targets"] = ["re:^mtp\\.layers\\..*"] if KEEP_FC else ["re:^mtp\\..*"]
    g["weights"]["num_bits"] = BITS
    g["weights"]["symmetric"] = True
    g["weights"]["zp_dtype"] = None
else:
    g = {
        "format": "pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "targets": ["re:^mtp\\.layers\\..*"] if KEEP_FC else ["re:^mtp\\..*"],
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
qc["config_groups"]["group_3"] = g
json.dump(c, open(d + "config.json", "w"), indent=2)
print("done")
