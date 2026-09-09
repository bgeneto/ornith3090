"""Build a vocab-truncated draft head for MTP speculative decoding.

The MTP drafter has to run the 248k-row lm_head once per draft token, and at
4-6 drafts per step that head read (1.3 GB int8) dominates the draft cost.
Speculative decoding stays exact no matter what the drafter proposes, so the
drafter can use a head restricted to the N most frequent tokens: tokens
outside the shortlist are simply never drafted (that position gets rejected
and the target's own sample is used, as always).

This script counts token frequencies over a text corpus (Danish + English +
code + the model's own outputs), picks the top N ids (plus special tokens),
slices those rows out of the already-int8-quantized lm_head, and stores them
as `mtp.draft_lm_head.*` in model_extra_tensors.safetensors, plus the id map in
`mtp_draft_vocab_ids.pt`. Needs the matching vLLM patch
(patches/qwen3_5-mtp-draft-vocab.patch) to be used.

Usage (from the repo root):
  python prepare/build_draft_vocab.py /path/to/model --ids prepare/draft_vocab_ids.json  # shipped id list
  python prepare/build_draft_vocab.py /path/to/model --n 40960 --corpus f1 f2 ...        # or count your own
Corpus files: .txt/.jsonl (uses "prompt"/"response"/"messages"/"text" fields)/.parquet(text)/.py
The shipped draft_vocab_ids.json was counted over Danish web text (fineweb-2),
English Wikipedia, Python source and the model's own chat outputs (8.8M tokens);
held-out coverage 95%.
"""
import glob, json, os, sys, shutil, collections
import torch
from safetensors import safe_open
from safetensors.torch import save_file

d = sys.argv[1].rstrip("/") + "/" if len(sys.argv) > 1 else "models/Ornith-1.5-9B-MixedInt4-AutoRound/"
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40960
corpus = sys.argv[sys.argv.index("--corpus") + 1:] if "--corpus" in sys.argv else []
ids_file = sys.argv[sys.argv.index("--ids") + 1] if "--ids" in sys.argv else None

if not ids_file and not corpus:
    # Check default shipped ids file if available
    default_ids = os.path.join(os.path.dirname(__file__), "draft_vocab_ids.json")
    if os.path.isfile(default_ids):
        ids_file = default_ids
        print(f"No corpus or --ids specified; using shipped default {ids_file}")
    else:
        print("Usage: python prepare/build_draft_vocab.py <model_dir> [--ids draft_vocab_ids.json] [--n 40960 --corpus file1 file2 ...]")
        sys.exit(1)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(d)

def texts_from(path, limit_bytes=20_000_000):
    n = 0
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for t in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            yield t; n += len(t)
            if n > limit_bytes: return
    elif path.endswith(".jsonl"):
        for line in open(path):
            try: r = json.loads(line)
            except Exception: continue
            parts = []
            for k in ("prompt", "response", "text"):
                if isinstance(r.get(k), str): parts.append(r[k])
            if isinstance(r.get("messages"), list):
                parts += [m.get("content", "") for m in r["messages"] if isinstance(m.get("content"), str)]
            t = "\n".join(parts); yield t; n += len(t)
            if n > limit_bytes: return
    else:
        t = open(path, errors="ignore").read(); yield t

counts = collections.Counter()
held = collections.Counter()
total = 0
if ids_file:
    ids = sorted(set(json.load(open(ids_file))))
    print(f"using {len(ids)} ids from {ids_file}")
    corpus = []
for i, path in enumerate(corpus):
    for j, t in enumerate(texts_from(path)):
        ids = tok(t, add_special_tokens=False).input_ids
        (held if j % 10 == 0 else counts).update(ids)
        total += len(ids)
print(f"corpus tokens: {total}")

special = set(tok.all_special_ids)
for name in (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
):
    tid = tok.convert_tokens_to_ids(name)
    if isinstance(tid, int) and tid >= 0:
        special.add(tid)

if not ids_file:
    top = [t for t, _ in counts.most_common() if t not in special][: N - len(special)]
    ids = sorted(set(top) | special)
    cover = sum(c for t, c in held.items() if t in set(ids)) / max(1, sum(held.values()))
    print(f"draft vocab: {len(ids)} ids, held-out token coverage {cover*100:.2f}%")
    for n_try in (16384, 32768, 40960, 49152, 65536):
        s = set(t for t, _ in counts.most_common(n_try)) | special
        c = sum(c for t, c in held.items() if t in s) / max(1, sum(held.values()))
        print(f"  coverage at N={n_try}: {c*100:.2f}%")
    json.dump(ids, open(d + "draft_vocab_ids.json", "w"))
    print(f"id list written to {d}draft_vocab_ids.json (copy it next to this script to reuse)")
else:
    ids = sorted(set(ids) | special)
    print(f"final draft vocab with special tokens: {len(ids)} ids")


def _lcm(a, b):
    x, y = int(a), int(b)
    while y:
        x, y = y, x % y
    return int(a) * int(b) // x


def _align_pack(ids, pack, vocab_size, pad_to=64):
    """Align the draft vocab to GPTQ packing and vLLM ParallelLMHead padding.

    AutoGPTQ qzeros pack along the vocab dim (pack = 32/bits). vLLM pads
    ParallelLMHead to DEFAULT_VOCAB_PADDING_SIZE (64), but GPTQ qweight stores
    vocab on dim 1 while the embedding weight_loader only pads dim 0. A length
    of 40968 therefore becomes an allocated 41024-wide Marlin N and crashes
    at load. Align to lcm(pack, 64) so checkpoint and head agree.
    """
    align = _lcm(pack, pad_to) if pad_to else pack
    have = set(ids)
    t = 0
    while len(have) % align:
        if t not in have and t < vocab_size:
            have.add(t)
        t += 1
        if t > vocab_size + align:
            break
    out = sorted(have)
    if len(out) != len(ids):
        print(f"padded draft vocab {len(ids)} -> {len(out)} ids (align={align})")
    if len(out) % align:
        raise SystemExit(f"draft vocab {len(out)} still not aligned to {align}")
    return out


# slice lm_head rows (AutoGPTQ qweight, or compressed-tensors weight_packed)
idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
ids_t = torch.tensor(ids, dtype=torch.int64)
extra = "model_extra_tensors.safetensors"
tensors = {}
meta = None
if os.path.exists(d + extra):
    with safe_open(d + extra, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    if not os.path.exists(d + extra + ".bak-draft"):
        shutil.copy(d + extra, d + extra + ".bak-draft")

if "lm_head.qweight" in wm:
    head_shard = wm["lm_head.qweight"]
    with safe_open(d + head_shard, framework="pt") as f:
        qw = f.get_tensor("lm_head.qweight")   # [K/pack, V]
        sc = f.get_tensor("lm_head.scales")    # [K/group, V]
        gi = f.get_tensor("lm_head.g_idx") if "lm_head.g_idx" in f.keys() else None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gptq_pack import pack_constant_qzeros
    ids = _align_pack(ids, pack=4, vocab_size=int(qw.shape[1]))
    ids_t = torch.tensor(ids, dtype=torch.int64)
    sub_w = qw.index_select(1, ids_t).contiguous()
    sub_s = sc.index_select(1, ids_t).contiguous()
    n_groups = int(sub_s.shape[0])
    sub_z = pack_constant_qzeros(len(ids), n_groups, bits=8)
    tensors["mtp.draft_lm_head.qweight"] = sub_w
    tensors["mtp.draft_lm_head.scales"] = sub_s
    tensors["mtp.draft_lm_head.qzeros"] = sub_z
    if gi is not None:
        tensors["mtp.draft_lm_head.g_idx"] = gi.contiguous()
    print(f"draft head (AutoGPTQ): qweight {tuple(sub_w.shape)} {sub_w.dtype}, "
          f"scales {tuple(sub_s.shape)} {sub_s.dtype}, "
          f"{(sub_w.numel()*4 + sub_s.numel()*2)/1e6:.0f} MB")
    for s in ("qweight", "scales", "qzeros") + (("g_idx",) if gi is not None else ()):
        wm[f"mtp.draft_lm_head.{s}"] = extra
    cfg_path = d + "config.json"
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        extra_cfg = cfg.setdefault("quantization_config", {}).setdefault("extra_config", {})
        extra_cfg["draft_lm_head"] = {"bits": 8, "group_size": 128, "sym": True}
        json.dump(cfg, open(cfg_path, "w"), indent=2)
else:
    head_shard = wm["lm_head.weight_packed"]
    with safe_open(d + head_shard, framework="pt") as f:
        wp = f.get_tensor("lm_head.weight_packed")   # [vocab, K/8] int32
        ws = f.get_tensor("lm_head.weight_scale")    # [vocab, K/group]
        shape = f.get_tensor("lm_head.weight_shape")
    ids = _align_pack(ids, pack=1, vocab_size=int(wp.shape[0]))
    ids_t = torch.tensor(ids, dtype=torch.int64)
    sub_p = wp.index_select(0, ids_t).contiguous()
    sub_s = ws.index_select(0, ids_t).contiguous()
    sub_shape = torch.tensor([len(ids), int(shape[1])], dtype=torch.int64)
    print(f"draft head: packed {tuple(sub_p.shape)} {sub_p.dtype}, scales {tuple(sub_s.shape)} {sub_s.dtype}, "
          f"{(sub_p.numel()*4 + sub_s.numel()*2)/1e6:.0f} MB")
    tensors["mtp.draft_lm_head.weight_packed"] = sub_p
    tensors["mtp.draft_lm_head.weight_scale"] = sub_s
    tensors["mtp.draft_lm_head.weight_shape"] = sub_shape
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        wm[f"mtp.draft_lm_head.{s}"] = extra

# The three quant_*.py scripts leave the base model's extras in this file; a
# single-shard export processed by prepare/quant_heads_stream.py has no extras
# file at all (#37) -- the draft head becomes its first content.
save_file(tensors, d + extra, metadata=meta or {"format": "pt"})
json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)
torch.save(ids_t, d + "mtp_draft_vocab_ids.pt")
print("done")
