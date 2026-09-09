"""AutoGPTQ packing used by AutoRound (`packing_format: auto_round:auto_gptq`).

vLLM INC builds ParallelLMHead / (with our patch) VocabParallelEmbedding as
AutoGPTQ layers. Those expect:

  qweight  [in // pack_factor, out]  int32   packed along the input dim
  scales   [in // group, out]        fp16
  qzeros   [in // group, out // pack_factor] int32   packed along the output dim
  g_idx    [in]                      int32   i // group (no act-order)

Signed codes in [-2^(b-1), 2^(b-1)-1] are stored as unsigned with bias
2^(b-1) (uint8b128 / uint4b8). Symmetric Marlin ignores qzeros; we still
write a constant zero-point so the loader's registered parameter is present.
"""

from __future__ import annotations

import torch

def pack_gptq(
    q_signed: torch.Tensor,
    scale: torch.Tensor,
    bits: int = 8,
    group: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a signed group-wise matrix into AutoGPTQ tensors.

    q_signed: [out, in] int8
    scale:    [out, in // group] float
    """
    if bits not in (4, 8):
        raise ValueError(f"unsupported bits={bits}")
    out_f, in_f = q_signed.shape
    pack = 32 // bits
    if in_f % group or in_f % pack or out_f % pack:
        raise ValueError(
            f"shape {tuple(q_signed.shape)} is not aligned to group={group} pack={pack}"
        )
    zp = 1 << (bits - 1)
    maxq = (1 << bits) - 1
    q_uint = (q_signed.to(torch.int32) + zp).clamp(0, maxq)

    # pack along input: [out, in/pack, pack] -> [out, in/pack] -> [in/pack, out]
    shifts = torch.arange(pack, dtype=torch.int32) * bits
    qweight = (
        (q_uint.reshape(out_f, in_f // pack, pack) << shifts)
        .sum(dim=-1)
        .to(torch.int32)
        .t()
        .contiguous()
    )

    # scale was [out, groups]; GPTQ stores [groups, out]
    if scale.ndim == 3:
        scale = scale.squeeze(-1)
    scales = scale.to(torch.float16).t().contiguous()

    zeros = torch.full((in_f // group, out_f), zp, dtype=torch.int32)
    qzeros = (
        (zeros.reshape(in_f // group, out_f // pack, pack) << shifts)
        .sum(dim=-1)
        .to(torch.int32)
        .contiguous()
    )
    g_idx = torch.arange(in_f, dtype=torch.int32) // group
    return qweight, scales, qzeros, g_idx


def unpack_gptq_qweight(
    qweight: torch.Tensor, bits: int = 8
) -> torch.Tensor:
    """Return unsigned codes [out, in] from GPTQ qweight [in/pack, out]."""
    pack = 32 // bits
    maxq = (1 << bits) - 1
    shifts = torch.arange(pack, dtype=torch.int32, device=qweight.device) * bits
    u = (qweight.unsqueeze(1) >> shifts.view(1, pack, 1)) & maxq
    # u: [in/pack, pack, out] -> [in, out] -> [out, in]
    return u.reshape(-1, qweight.shape[1]).t().contiguous()


def pack_constant_qzeros(out_f: int, n_groups: int, bits: int = 8) -> torch.Tensor:
    """Packed qzeros filled with the symmetric zero-point, shape [groups, out/pack]."""
    pack = 32 // bits
    zp = 1 << (bits - 1)
    if out_f % pack:
        raise ValueError(f"out_f={out_f} not aligned to pack={pack}")
    shifts = torch.arange(pack, dtype=torch.int32) * bits
    zeros = torch.full((n_groups, out_f), zp, dtype=torch.int32)
    return (
        (zeros.reshape(n_groups, out_f // pack, pack) << shifts)
        .sum(dim=-1)
        .to(torch.int32)
        .contiguous()
    )


def is_autoround(qc: dict) -> bool:
    return qc.get("quant_method") == "auto-round"


def extra_bits8(*names: str) -> dict:
    return {n: {"bits": 8, "group_size": 128, "sym": True} for n in names}


def _row_getter(f, key: str):
    """Prefer safetensors slicing so the full BF16 matrix never sits in RAM."""
    if hasattr(f, "get_slice"):
        sl = f.get_slice(key)
        shape = tuple(sl.get_shape())
        return shape, lambda lo, hi: sl[lo:hi, :]
    w = f.get_tensor(key)
    return tuple(w.shape), lambda lo, hi: w[lo:hi]


def pack_gptq_from_open(
    f,
    key: str,
    bits: int = 8,
    group: int = 128,
    rows: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Group-wise RTN + AutoGPTQ pack, row-chunked. Returns (qweight, scales, qzeros, g_idx, rel_err)."""
    (out_f, in_f), get_rows = _row_getter(f, key)
    qmax = (1 << (bits - 1)) - 1
    pack = 32 // bits
    qweight = torch.empty((in_f // pack, out_f), dtype=torch.int32)
    scales_out = torch.empty((in_f // group, out_f), dtype=torch.float16)
    col = 0
    num = den = 0.0
    for lo in range(0, out_f, rows):
        hi = min(lo + rows, out_f)
        chunk = get_rows(lo, hi).to(torch.float32)
        nrows = chunk.shape[0]
        g = chunk.reshape(nrows, in_f // group, group)
        scale = torch.clamp(g.abs().amax(dim=-1) / qmax, min=1e-10)
        q = torch.clamp(torch.round(g / scale[..., None]), -qmax - 1, qmax).to(torch.int8)
        q = q.reshape(nrows, in_f)
        deq = (q.float().reshape(nrows, -1, group) * scale[..., None]).reshape(nrows, in_f)
        num += (deq - chunk).pow(2).sum().item()
        den += chunk.pow(2).sum().item()
        qw, sc, _, _ = pack_gptq(q, scale, bits, group)
        qweight[:, col : col + nrows] = qw
        scales_out[:, col : col + nrows] = sc
        col += nrows
        del chunk, g, q, deq, qw, sc, scale
    qzeros = pack_constant_qzeros(out_f, in_f // group, bits)
    g_idx = torch.arange(in_f, dtype=torch.int32) // group
    return qweight, scales_out, qzeros, g_idx, (num / den) ** 0.5


def pack_ct_from_open(
    f,
    key: str,
    bits: int = 8,
    group: int = 128,
    rows: int = 4096,
    scale_dtype=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Group-wise RTN + compressed-tensors pack, row-chunked."""
    from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

    if scale_dtype is None:
        scale_dtype = torch.float16
    (out_f, in_f), get_rows = _row_getter(f, key)
    qmax = (1 << (bits - 1)) - 1
    packed_parts, scale_parts = [], []
    num = den = 0.0
    for lo in range(0, out_f, rows):
        hi = min(lo + rows, out_f)
        chunk = get_rows(lo, hi).to(torch.float32)
        nrows = chunk.shape[0]
        g = chunk.reshape(nrows, in_f // group, group)
        s = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / qmax, min=1e-10)
        q = torch.clamp(torch.round(g / s), -qmax - 1, qmax).to(torch.int8)
        deq = (q.to(torch.float32) * s).reshape(nrows, in_f)
        num += (deq - chunk).pow(2).sum().item()
        den += chunk.pow(2).sum().item()
        packed_parts.append(pack_to_int32(q.reshape(nrows, in_f), bits, packed_dim=1).contiguous())
        scale_parts.append(s.squeeze(-1).contiguous())
        del chunk, g, q, deq, s
    packed = torch.cat(packed_parts, dim=0).contiguous()
    scale = torch.cat(scale_parts, dim=0).to(scale_dtype).contiguous()
    shape = torch.tensor([out_f, in_f], dtype=torch.int64)
    return packed, scale, shape, (num / den) ** 0.5


if __name__ == "__main__":
    torch.manual_seed(0)
    out_f, in_f, group, bits = 256, 128, 128, 8
    w = torch.randn(out_f, in_f)
    g = w.reshape(out_f, in_f // group, group)
    scale = torch.clamp(g.abs().amax(dim=-1) / 127, min=1e-10)
    q = torch.clamp(torch.round(g / scale[..., None]), -128, 127).to(torch.int8).reshape(out_f, in_f)
    qw, sc, qz, gi = pack_gptq(q, scale, bits, group)
    assert qw.shape == (in_f // (32 // bits), out_f)
    assert sc.shape == (in_f // group, out_f)
    assert qz.shape == (in_f // group, out_f // (32 // bits))
    assert gi.tolist() == [0] * in_f
    u = unpack_gptq_qweight(qw, bits)
    assert torch.equal(u, q.to(torch.int32) + 128)
    deq = (q.float().reshape(out_f, -1, group) * scale[..., None]).reshape(out_f, in_f)
    err = ((deq - w).norm() / w.norm()).item()
    print(f"pack self-test ok  rel_err={err:.4f}  qweight={tuple(qw.shape)}")
