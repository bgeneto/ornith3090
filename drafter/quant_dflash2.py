"""Quantize an Ornith DFlash2 drafter to W4A16 compressed-tensors (Marlin) with GPTQ.

Calibrate on Hessians from capture_dflash2.py (drafter's own inputs on real traffic).

  python drafter/quant_dflash2.py <src_dir> <dst_dir> <hessians.pt> [--bits 4] [--fc-bits 4|8|16] [--rtn]
                                  [--blend-ctx-kv]

Widths are read from config.json (Ornith: hidden 4096, fc 20480×4096, 5 layers).
Leave convs, candidate selector, and norms in bf16 (vLLM builds them with
quant_config=None). Default recipe does NOT blend ctx_kv into k/v Hessians
(that cost ~7% greedy acceptance on the 27B stack). Pass --blend-ctx-kv to
experiment; the ctx_kv tensor must then be inside hessians.pt.
"""
import json, os, sys, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_utils import gptq_quantize, rtn_quantize, dequant

S = sys.argv[1].rstrip("/") + "/"; D = sys.argv[2].rstrip("/") + "/"; HP = sys.argv[3]
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 4
FC_BITS = int(sys.argv[sys.argv.index("--fc-bits") + 1]) if "--fc-bits" in sys.argv else 4
RTN = "--rtn" in sys.argv
BLEND_CTX = "--blend-ctx-kv" in sys.argv
GROUP = 128
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = json.load(open(S + "config.json"))
L = cfg["num_hidden_layers"]
H = cfg.get("hidden_size")
print(f"draft layers={L} hidden={H} vocab={cfg.get('vocab_size')} "
      f"taps={cfg.get('dflash_config', {}).get('target_layer_ids')}")

# checkpoint name -> (hessian key in capture file)
LIN = {}
for i in range(L):
    for p in ["q_proj", "k_proj", "v_proj"]:
        LIN[f"layers.{i}.self_attn.{p}"] = f"layers.{i}.self_attn.qkv_proj"
    LIN[f"layers.{i}.self_attn.o_proj"] = f"layers.{i}.self_attn.o_proj"
    for p in ["gate_proj", "up_proj"]:
        LIN[f"layers.{i}.mlp.{p}"] = f"layers.{i}.mlp.gate_up_proj"
    LIN[f"layers.{i}.mlp.down_proj"] = f"layers.{i}.mlp.down_proj"
if FC_BITS < 16:
    LIN["fc"] = "fc"


def _open_weights(src):
    st = src + "model.safetensors"
    if os.path.isfile(st):
        with safe_open(st, "pt") as f:
            return f.metadata(), {k: f.get_tensor(k) for k in f.keys()}
    # sharded
    idx = json.load(open(src + "model.safetensors.index.json"))
    tensors, meta = {}, None
    for shard in sorted(set(idx["weight_map"].values())):
        with safe_open(src + shard, "pt") as f:
            meta = f.metadata() or meta
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    return meta, tensors


HS = torch.load(HP, map_location="cpu") if not RTN else {}
if BLEND_CTX and "ctx_kv" not in HS:
    ctxp = os.path.join(os.path.dirname(HP), "hessians_ctx_kv.pt")
    if os.path.isfile(ctxp):
        HS.update(torch.load(ctxp, map_location="cpu"))
        print("loaded ctx_kv from", ctxp)
if not BLEND_CTX and "ctx_kv" in HS:
    print("ignoring ctx_kv Hessian (pass --blend-ctx-kv to mix into k/v)")
    HS = {k: v for k, v in HS.items() if k != "ctx_kv"}

meta, tensors = _open_weights(S)
out = {}
t0 = time.time()
stats = []
for name, w in tensors.items():
    base = name[: -len(".weight")] if name.endswith(".weight") else None
    if base in LIN:
        W = w.to(dev)
        bits = FC_BITS if base == "fc" else BITS
        if RTN or LIN[base] not in HS:
            if not RTN:
                print(f"  !! no Hessian for {LIN[base]}, RTN fallback")
            q, s = rtn_quantize(W, bits, GROUP)
        else:
            Hm = HS[LIN[base]]["H"].to(dev)
            if BLEND_CTX and base.endswith((".k_proj", ".v_proj")) and "ctx_kv" in HS:
                nq, Hc, nc = HS[LIN[base]]["n"], HS["ctx_kv"]["H"].to(dev), HS["ctx_kv"]["n"]
                Hm = (nq * Hm + nc * Hc) / (nq + nc)
                del Hc
            q, s = gptq_quantize(W, Hm, bits=bits, group=GROUP, blocksize=GROUP)
            del Hm
        rel = ((dequant(q, s) - W.float()).norm() / W.float().norm()).item()
        stats.append((base, bits, rel))
        out[base + ".weight_packed"] = pack_to_int32(q.cpu(), bits, packed_dim=1).contiguous()
        out[base + ".weight_scale"] = s.to(torch.float16).cpu().contiguous()
        out[base + ".weight_shape"] = torch.tensor(list(W.shape), dtype=torch.int64)
        del W, q, s
        if dev == "cuda":
            torch.cuda.empty_cache()
    else:
        out[name] = w
for base, bits, rel in stats:
    if base.startswith("layers.0.") or not base.startswith("layers."):
        print(f"  int{bits} {base}: rel err {rel:.4f}")
print(f"quantized {len(stats)} matrices in {time.time()-t0:.0f}s; "
      f"mean rel err {sum(r for _,_,r in stats)/len(stats):.4f}")
expect = 7 * L + (1 if FC_BITS < 16 else 0)  # q,k,v,o,gate,up,down per layer + fc
if len(stats) != expect:
    print(f"WARNING: expected {expect} quantized linears for {L} layers (fc_bits={FC_BITS}), got {len(stats)}")

os.makedirs(D, exist_ok=True)
save_file(out, D + "model.safetensors", metadata=meta or {"format": "pt"})
for f in os.listdir(S):
    if f not in ("model.safetensors", "config.json") and not f.startswith("."):
        p = os.path.join(S, f)
        if os.path.isfile(p):
            shutil.copy(p, D + f)
ignore = ["re:.*kernel_projection$", "re:.*candidate_selector.*", "re:.*hidden_projection$"]
if FC_BITS >= 16:
    ignore.append("re:.*\\.fc$")
groups = {"group_0": {"format": "pack-quantized", "input_activations": None, "output_activations": None,
                      "targets": ["Linear"],
                      "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": GROUP,
                                  "num_bits": BITS, "observer": "memoryless_minmax", "observer_kwargs": {},
                                  "scale_dtype": None, "strategy": "group", "symmetric": True, "type": "int",
                                  "zp_dtype": None}}}
if FC_BITS < 16 and FC_BITS != BITS:
    groups["group_1"] = json.loads(json.dumps(groups["group_0"]))
    groups["group_1"]["targets"] = ["re:.*\\.fc$"]
    groups["group_1"]["weights"]["num_bits"] = FC_BITS
cfg["quantization_config"] = {"config_groups": groups, "format": "pack-quantized", "global_compression_ratio": None,
                              "ignore": ignore, "kv_cache_scheme": None, "quant_method": "compressed-tensors",
                              "quantization_status": "compressed", "sparsity_config": {}, "transform_config": {},
                              "version": "0.17.0"}
json.dump(cfg, open(D + "config.json", "w"), indent=2)
sz = os.path.getsize(D + "model.safetensors") / 2**30
print(f"wrote {D} ({sz:.2f} GiB)")
