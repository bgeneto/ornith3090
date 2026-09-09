# drafter/ — Ornith-1.5-9B MTP draft vocabulary & self-distillation

Tooling to build and calibrate the Ornith-1.5-9B native MTP draft module and draft vocabulary.

> [!NOTE]
> This repository focuses exclusively on **Ornith-1.5-9B with native MTP speculative decoding**.
> DFlash2 is not used or trained here, because Ornith-1.5-9B's 32-layer architecture has much lower overhead than 27B models and native MTP ($k=3/4$) provides exceptional single-user throughput (~150–190+ tok/s).

## Why the Draft Vocabulary Matters

The native MTP drafter runs an `lm_head` projection for each draft token. In a 248k-vocabulary model, projecting $4096 \to 248,320$ repeatedly dominates draft latency.
`prepare/build_draft_vocab.py` slices a 40,960-row draft head (`mtp.draft_lm_head.*`) from the INT8 `lm_head`.

Crucially:
- A token outside the draft vocabulary can never be drafted, meaning an out-of-vocab token is a guaranteed rejection that truncates the speculation chain.
- Counting token frequency over generic web text only yields ~92% coverage (~83% on code).
- Counting token frequency over **Ornith-1.5-9B's own generated outputs** across the target workloads (Python, TypeScript, Shell, Docker, Portuguese, technical English, tool calls) achieves **97.5%+ coverage** (96%+ on code), unlocking the full ~150–190+ tok/s decode potential.

## Complete Pipeline for Ornith-Specific Draft Vocabulary

```bash
V=venv/bin/python

# 1. Collect diverse prompts across code, tools, technical English, Portuguese, and math
$V drafter/collect_prompts.py
# Produces drafter/data/prompts.jsonl

# 2. Generate model outputs offline with Ornith-1.5-9B (resumable)
VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_MARLIN_INT8_INCLUDE_RE=mlp $V drafter/gen_data.py
# Produces drafter/data/gen.jsonl (containing millions of generated token IDs)

# 3. Count token frequencies, extract top 40k tokens + special control tokens, and slice the draft head
$V prepare/build_draft_vocab.py models/Ornith-1.5-9B-MixedInt4-AutoRound --corpus drafter/data/gen.jsonl
# Produces mtp_draft_vocab_ids.pt and adds mtp.draft_lm_head.* to model_extra_tensors.safetensors
```

Speculative decoding remains 100% exact: the target model always verifies each token using the full logits projection; the draft vocabulary only accelerates proposal generation.
