# prepare/ — one-time model preparation for Ornith-1.5-9B

The published Mixed INT4 AutoRound checkpoint of Ornith-1.5-9B (`Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound`)
has its body linears and MTP layer quantized, but leaves `lm_head` and `embed_tokens` in BF16 (each ~2.03 GB).
These scripts optimize the model in place on the CPU, reclaiming ~2.0 GB of VRAM and accelerating generation.
`docker compose run --rm prepare` runs them automatically.

On AutoRound checkpoints (`quant_method: auto-round`) the scripts write AutoGPTQ
tensors (`qweight` / `scales` / `qzeros` / `g_idx`). Compressed-tensors
`weight_packed` is incompatible with vLLM INC and will fail the loader;
`verify.sh` now FAILs that mixed state instead of treating `weight_packed` as
success. `embed_tokens` also needs `patches/inc-gptq-embed.patch`. Other
(compressed-tensors) checkpoints still get `weight_packed` as before.

Run from the repo root, in order — `quant_lm_head.py` first, because `build_draft_vocab.py` slices its rows:

```bash
V=venv/bin/python; M=models/Ornith-1.5-9B-MixedInt4-AutoRound
$V prepare/fetch_ornith.py             # download Pilcothink INT4 checkpoint (~8.8 GB; uses HF_TOKEN if set)
$V prepare/quant_lm_head.py $M         # lm_head -> int8 group-128, in place: ~1.02 GB freed, speeds up decode
$V prepare/quant_embed.py   $M         # embed_tokens likewise (untied): another ~1.02 GB freed
$V prepare/quant_mtp.py     $M         # verifies MTP layers (already INT4 in Pilcothink)
$V prepare/build_draft_vocab.py $M --ids prepare/draft_vocab_ids.json
```

`build_draft_vocab.py` writes a 40,960-row slice of `lm_head` for the MTP drafter to score instead of the full
248k vocabulary; `draft_vocab_ids.json` is the id list, and `--corpus` counts your own instead. It needs
[patches/qwen3_5-mtp-draft-vocab.patch](../patches/qwen3_5-mtp-draft-vocab.patch).

## A different checkpoint

`quant_heads_stream.py` does the work of `quant_lm_head.py` + `quant_embed.py` +
`quant_mtp.py` in one pass, for checkpoints those three cannot open: **single-shard**
ones (they read a shard into RAM whole; the uncensored build ships one 18.6 GB
`model.safetensors`) and **asymmetric AWQ** bodies (they clone `config_groups.group_0`
onto the symmetric tensors they write, so vLLM then looks for a `weight_zero_point`
that does not exist). Same math, same output tensors, peak RSS well under the shard
size (9.7 GB measured on the 18.6 GB example here -- still not a low-RAM tool).

```bash
$V prepare/fetch_thirdparty.py                          # ~18.6 GB (or: fetch_thirdparty.py <hf-repo>)
$V prepare/quant_heads_stream.py models/Qwen3.8-27B-Uncensored-W4A16
$V prepare/build_draft_vocab.py  models/Qwen3.8-27B-Uncensored-W4A16 \
  --ids prepare/draft_vocab_ids.json
```

Then `MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 bash single-user/start_qwen.sh`.
That leftover path is the 27B launcher. For Ornith, `SPEC=dflash2` needs
`models/Ornith-1.5-9B-DFlash2-W4A16` from `drafter/` (not `syvai/Qwen3.8-27B-DFlash2-W4A16`).
`--mtp-bits 4` and `--keep-fc` exist for experimenting with the draft module; the
defaults (int8, `mtp.fc` quantized) are what was verified.

The two `fetch_*` scripts only download: the fast variant is the int4-GPTQ lm_head and
drafter plus a draft vocabulary counted over the model's own outputs, and
`fetch_dflash2.py` installs an **Ornith** DFlash2 W4A16 dir (it refuses a 5120-d /
64-layer Qwen 27B checkpoint). Rebuild from [drafter/](../drafter/).

`bash verify.sh --no-server` checks every step above against the model dir and names
the script to run for whatever is missing. Each in-place script backs up what it
rewrites next to the original (`.bak*`), so a step can be undone without re-downloading
19.5 GB. Why each one is worth doing, with measurements:
[docs/optimizations.md](../docs/optimizations.md).
