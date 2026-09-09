"""Safetensors header helpers (no torch). Used by prepare.sh and the quant scripts.

A previous AutoRound run can leave the index pointing at `lm_head.weight`
while the shard still has compressed-tensors `weight_packed` (or vice versa).
Peeking at the on-disk header is enough to detect that and restore `.bak`.
"""

from __future__ import annotations

import json
import os
import shutil
import struct


def safetensors_keys(path: str) -> list[str]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return list(hdr)


def copy_replace(src: str, dst: str) -> None:
    """Copy then replace, so a killed copy cannot truncate `dst`."""
    tmp = dst + ".tmp-restore"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    try:
        os.chmod(dst, 0o644)
    except OSError:
        pass


def restore_shard_from_bak(d: str, shard: str) -> str | None:
    """Copy `{shard}.bak` over `{shard}` if the backup exists. Returns the bak path."""
    bak = d + shard + ".bak"
    if not os.path.exists(bak):
        return None
    print(f"  restoring {shard} from {bak} ({os.path.getsize(bak)} bytes)")
    copy_replace(bak, d + shard)
    return bak


def classify_lm_head(keys) -> str:
    keys = set(keys)
    if "lm_head.qweight" in keys:
        return "gptq"
    if "lm_head.weight_packed" in keys:
        return "packed"
    if "lm_head.weight" in keys:
        return "bf16"
    return "missing"


def classify_embed(keys) -> str:
    keys = set(keys)
    if any(k.endswith("embed_tokens.qweight") for k in keys):
        return "gptq"
    if any(k.endswith("embed_tokens.weight_packed") for k in keys):
        return "packed"
    if any(k.endswith("embed_tokens.weight") for k in keys):
        return "bf16"
    return "missing"


def autoround_needs_bf16_restore(d: str, idx: dict) -> str | None:
    """Why an AutoRound dir should restore `.bak` files, or None if it shouldn't.

    Never restore if the shard already has AutoGPTQ `qweight` — that would
    throw away a successful requant.
    """
    wm = idx.get("weight_map", idx)
    mixed = [
        k
        for k in wm
        if k.endswith("weight_packed")
        and (
            k.startswith("lm_head.")
            or k.endswith("embed_tokens.weight_packed")
            or "draft_lm_head" in k
        )
    ]
    shard = wm.get("lm_head.weight") or wm.get("lm_head.qweight") or wm.get(
        "lm_head.weight_packed"
    )
    if not shard:
        shard = "model-00001-of-00002.safetensors"
    path = d + shard
    if not os.path.exists(path):
        return f"index mixed {mixed}" if mixed else None
    try:
        keys = safetensors_keys(path)
    except OSError as e:
        return f"cannot read {shard}: {e}"
    lm = classify_lm_head(keys)
    emb = classify_embed(keys)
    if lm == "gptq" or emb == "gptq":
        return None
    if mixed:
        return "index lists compressed-tensors weight_packed: " + ", ".join(mixed)
    if lm in ("packed", "missing") or emb == "packed":
        return (
            f"{shard} has lm_head={lm} embed={emb} but the index lists "
            f"lm_head.weight={('lm_head.weight' in wm)}"
        )
    return None


def restore_autoround_bf16(d: str) -> bool:
    """Restore config/index/shard backups. Returns True if anything was copied."""
    copied = False
    has_full_bak = os.path.exists(d + "model-00001-of-00002.safetensors.bak")
    # shard .bak last so it wins over .bak_embed. Skip the 4 GB intermediate
    # copy when the original 5 GB BF16 backup is present.
    for src, dst in (
        ("config.json.bak-quant", "config.json"),
        ("model.safetensors.index.json.bak-quant", "model.safetensors.index.json"),
        ("model-00001-of-00002.safetensors.bak_embed", "model-00001-of-00002.safetensors"),
        ("model-00001-of-00002.safetensors.bak", "model-00001-of-00002.safetensors"),
        ("model_extra_tensors.safetensors.bak-draft", "model_extra_tensors.safetensors"),
    ):
        if src.endswith(".bak_embed") and has_full_bak:
            continue
        src_f, dst_f = d + src, d + dst
        if os.path.exists(src_f):
            copy_replace(src_f, dst_f)
            print(f"  restored {dst} from {src}")
            copied = True
    for rm_f in ("mtp_draft_vocab_ids.pt",):
        p = d + rm_f
        if os.path.exists(p):
            os.remove(p)
    return copied


def stream_rewrite(src: str, dst: str, drop: set[str], add: dict) -> int:
    """Copy `src` to `dst`, omitting names in `drop` and appending `add`.

    Untouched tensors are copied as raw bytes so a 5 GB shard does not have to
    sit in RAM. `add` values are CPU torch tensors.
    """
    import torch

    dtype_str = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.int8: "I8",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.uint8: "U8",
    }
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    data_start = 8 + n
    meta = hdr.pop("__metadata__", None)
    keep = [
        (k, v)
        for k, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
        if k not in drop
    ]

    new_hdr, off = {}, 0
    if meta is not None:
        new_hdr["__metadata__"] = meta
    plan = []
    for k, v in keep:
        b0, b1 = v["data_offsets"]
        size = b1 - b0
        new_hdr[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + size]}
        plan.append(("copy", data_start + b0, size))
        off += size
    for k, t in add.items():
        t = t.contiguous().cpu()
        size = t.numel() * t.element_size()
        new_hdr[k] = {
            "dtype": dtype_str[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [off, off + size],
        }
        plan.append(("write", t, size))
        off += size

    blob = json.dumps(new_hdr, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        for kind, a, size in plan:
            if kind == "copy":
                fi.seek(a)
                left = size
                while left:
                    chunk = fi.read(min(left, 64 << 20))
                    if not chunk:
                        raise OSError(f"short read in {src}")
                    fo.write(chunk)
                    left -= len(chunk)
            else:
                fo.write(memoryview(a.contiguous().view(torch.uint8).numpy()))
    try:
        os.chmod(dst, 0o644)
    except OSError:
        pass
    return off


def stream_replace(src: str, drop: set[str], add: dict) -> None:
    """Rewrite `src` in place via a temp file (safe if the process is killed)."""
    tmp = src + ".tmp-quant"
    try:
        stream_rewrite(src, tmp, drop, add)
        os.replace(tmp, src)
        try:
            os.chmod(src, 0o644)
        except OSError:
            pass
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
