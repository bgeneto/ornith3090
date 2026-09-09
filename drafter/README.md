# drafter/ — Ornith-1.5-9B MTP draft vocabulary & DFlash2 training

Tooling to (1) build the native MTP draft vocabulary and (2) train / quantize an
**Ornith-specific** DFlash2 block drafter.

The published [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
checkpoint **cannot** be used here. It was trained to consume hidden states from
layers 5/19/33/47/61 of a 64-layer / 5120-d model (`fc` is 25600→5120).
Ornith-1.5-9B has 32 layers and hidden size 4096.

## DFlash2 pipeline (Phase 2)

Geometry is pinned in [`dflash2_const.py`](dflash2_const.py):

| knob | value |
|---|---|
| `hidden_size` / `num_target_layers` | 4096 / 32 |
| `target_layer_ids` | **1, 8, 15, 22, 29** (paper-uniform). Ablation: 3, 11, 19, 27, 31 (full-attn only) |
| `fc` | 20480 → 4096 |
| `block_size` / drafts | 8 / 7 |
| draft stack | 5 Qwen3 sliding-window layers, `head_dim` 128, `intermediate_size` 12288, SW=2048 |
| conv / selector | k=2, g=16, rank=256, topk=16 |
| `mask_token_id` | **248077** (unused reserved). Not 248070 (`<|audio_start|>`), not pad/eos |
| `is_causal` | **false** (otherwise vLLM silently drafts as DFlash1) |

```bash
V=venv/bin/python

# 0. Self-distillation mix (same as the MTP vocab)
$V drafter/collect_prompts.py          # drafter/data/prompts.jsonl
VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_MARLIN_INT8_INCLUDE_RE=mlp $V drafter/gen_data.py
$V drafter/convert_gen_to_chat.py      # drafter/data/chat.jsonl  (+ chat_smoke.jsonl)

# 1. Config + shape smoke (no GPU)
$V drafter/init_dflash2.py --config-only
$V drafter/smoke_dflash2.py            # checks taps vs GDN/full-attn, fc width, mask id
# with GPU + a weight checkpoint:
# $V drafter/smoke_dflash2.py --init --load

# 2. Train (NeMo AutoModel TrainDFlash2Recipe; teacher = serving INT4 Ornith)
bash drafter/train_dflash2.sh --smoke  # 3090 / 1 GPU, sdpa, not a quality run
bash drafter/train_dflash2.sh --cloud  # 2–4 GPU, drafter/ornith_dflash2.yaml

# Docker: there is no system `pip`. Do not install NeMo into /app/venv.
#   source docker/env.sh
#   docker compose --profile single stop
#   docker compose --profile train run --rm train --smoke
# Copy the consolidated safetensors to models/Ornith-1.5-9B-DFlash2

# 3. GPTQ W4A16 for the 3090 (do not blend ctx_kv into k/v Hessians)
MODEL=models/Ornith-1.5-9B-MixedInt4-AutoRound \
DRAFT=models/Ornith-1.5-9B-DFlash2 \
  $V drafter/capture_dflash2.py
$V drafter/quant_dflash2.py models/Ornith-1.5-9B-DFlash2 \
    models/Ornith-1.5-9B-DFlash2-W4A16 drafter/runs/dflash2/hessians.pt

# 4. Serve (MTP stays the default until C1 wins)
SPEC=dflash2 bash single-user/start_ornith.sh
bash bench/dflash2_vs_mtp.sh dflash2
```

If NeMo cannot load the AutoRound teacher, capture aux hiddens with in-process
vLLM (same idea as `capture.py`, five layer tensors) and train offline.
Fallback trainer: `bash drafter/train_dflash2_speculators.sh` (vLLM speculators
`--speculator-type dflash2`). Dry-run the checkpoint in vLLM
(`smoke_dflash2.py --load`) before a long job so you do not train a DFlash1-shaped graph.

Selector codebooks stay `2 × 248320 × 256` bf16 (~254 MB) even on 9B; W4A16 of
the linears still leaves ~0.8–1.1 GB.

## MTP draft vocabulary

The native MTP drafter runs an `lm_head` projection for each draft token. In a 248k-vocabulary model, projecting $4096 \to 248,320$ repeatedly dominates draft latency.
`prepare/build_draft_vocab.py` slices a 40,960-row draft head (`mtp.draft_lm_head.*`) from the INT8 `lm_head`.

- A token outside the draft vocabulary can never be drafted (guaranteed rejection).
- Counting over **Ornith-1.5-9B's own generated outputs** (Python, TypeScript, Shell, Docker, Portuguese, technical English, tool calls) reaches **97.5%+** coverage (96%+ on code).

```bash
V=venv/bin/python
$V drafter/collect_prompts.py
VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_MARLIN_INT8_INCLUDE_RE=mlp $V drafter/gen_data.py
$V prepare/build_draft_vocab.py models/Ornith-1.5-9B-MixedInt4-AutoRound --corpus drafter/data/gen.jsonl
```

Speculative decoding remains exact: the target verifies every token with the full logits projection.
