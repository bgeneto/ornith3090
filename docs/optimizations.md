# What this repo does for Ornith-1.5-9B that stock vLLM doesn't

Optimizations applied to [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) (starting from [Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound](https://huggingface.co/Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound)) on a single RTX 3090 GPU (24 GB). For what each one is worth in tokens per second, see the benchmark sections in the [main README](../README.md).

[← back to the main README](../README.md)

## Why Ornith-1.5-9B is an Ideal Target

Ornith-1.5-9B shares the same Qwen3.5 hybrid architecture as Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`), but with key architectural properties that make it exceptionally well suited for 24 GB consumer GPUs:
- **32 total layers**: 24 Gated DeltaNet (linear attention) layers + 8 full attention layers (interval 4: layers 3, 7, 11, 15, 19, 23, 27, 31).
- **Hidden size 4096**, intermediate size 12288, head dimension 256.
- **Large 248,320-token vocabulary** with untied embeddings.
- **Native MTP**: `mtp_num_hidden_layers = 1`.
- **Pre-quantized baseline**: `Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound` already performed mixed INT4 AutoRound quantization on transformer layers and MTP body layers.
- **Abundant VRAM headroom**: The quantized model weights take only ~8.5 GB (compared to ~17 GB on 27B), leaving ~15.5 GB of free VRAM on an RTX 3090 for KV cache, recurrent states, and large CUDA graph memory pools without risking OOM.

## The Short Version

1. **Both embedding matrices requantized to INT8** (`prepare/quant_lm_head.py`, `prepare/quant_embed.py`)
   — The base checkpoint leaves two ~2.03 GB BF16 matrices (`lm_head` and `embed_tokens`). Requantizing both to INT8 group-128 recovers **~2.03 GB VRAM** and directly reduces generation latency on the 248k-way logits projection.
2. **Quantized embedding patches** (`patches/qwen3_5-embed-quant.patch`, `patches/inc-gptq-embed.patch`)
   — Wires `quant_config` into `Qwen3_5ForConditionalGeneration` so it actually uses INT8 `embed_tokens`, and teaches vLLM INC to gather AutoGPTQ `qweight` (stock 0.28.0 only does that for `ParallelLMHead`).
3. **16-bit GDN recurrent state** (`--mamba-ssm-cache-dtype float16`)
   — Halves recurrent memory traffic and footprint across 24 DeltaNet layers while preserving perplexity.
4. **Native MTP speculative decoding ($k=3/4$)**
   — Uses Ornith's native 1-layer MTP head (already INT4 AutoRound quantized) to propose 3–4 draft tokens per forward pass.
5. **Reduced MTP draft vocabulary** (`prepare/build_draft_vocab.py`, `patches/qwen3_5-mtp-draft-vocab.patch`)
   — Slices a ~40k-token draft head so the drafter doesn't evaluate all 248k rows on every draft step.
6. **Split-KV verification kernel** (`patches/spec-decode-attn.patch`)
   — Triton kernel that splits the KV sequence across thread blocks during multi-query verification, fully utilizing the 3090's 82 SMs.
7. **Sort-free sampler** (`patches/sampler-small-topk-fast-softmax.patch`)
   — Avoids full 248k vocabulary sorting when sampling with top-$k \le 64$ (Ornith recommends top-$k = 20$).
8. **W4A8 INT8 activations for prefill** (`VLLM_MARLIN_INPUT_DTYPE=int8`)
   — Accelerates compute-bound prefill on INT8 tensor cores with AutoRound negative scale fixes.
9. **Hybrid prefix caching** (`--enable-prefix-caching`)
   — Reuses KV cache and resumes GDN recurrent state across turns for coding agent workflows.
10. **Phased approach for DFlash2**
   — DFlash2 is deferred to Phase 2 because the Qwen3.8 drafter targets 64 layers (extracting hidden states at layers 5, 19, 33, 47, 61), whereas Ornith has 32 layers. Native MTP has lower relative overhead on 9B and targets **~150–190+ tok/s**.

## In Full

1. **Both embedding matrices requantized.**
   Ornith-1.5-9B has untied embeddings with a vocabulary of 248,320 and hidden size of 4096. A single BF16 matrix of $248,320 \times 4096$ contains ~1.017 billion parameters (~2.03 GB). Untied embeddings mean two separate matrices:
   - `model.language_model.embed_tokens.weight` (~2.03 GB BF16)
   - `lm_head.weight` (~2.03 GB BF16)
   Totaling ~4.07 GB. `prepare/quant_lm_head.py` and `prepare/quant_embed.py` convert both to INT8 group-128 in place, saving **~2.03 GB VRAM**. Crucially, quantizing `lm_head` also speeds up decode steps because the large 248k projection occurs on every generated token.
2. **Quantized embedding patches.**
   vLLM provides an optimized dequant-on-gather kernel for quantized embedding tables, but `qwen3_5.py` did not connect it. `patches/qwen3_5-embed-quant.patch` hooks this up for compressed-tensors `weight_packed`. AutoRound checkpoints write AutoGPTQ `qweight` instead, which needs `patches/inc-gptq-embed.patch` so INC actually builds a quantized `VocabParallelEmbedding`.
3. **16-bit GDN recurrent state.**
   24 of Ornith's 32 layers are Gated DeltaNet with fixed recurrent state per sequence. Stock configuration requests FP32 (`"mamba_ssm_dtype": "float32"`). Using `--mamba-ssm-cache-dtype float16` cuts the memory footprint and bandwidth in half with identical perplexity.
4. **W4A8 / INT8 tensor cores for GEMMs with AutoRound fixes.**
   AutoRound symmetric quantization produces group scales with ~50% negative values. Upstream Marlin kernels read scales as unsigned, corrupting output. `patches/marlin-int8-negative-scales.patch` folds the sign into the INT4 weights at load time, allowing `VLLM_MARLIN_INPUT_DTYPE=int8` to deliver +25–30% prefill throughput.
5. **Native MTP draft module and reduced draft vocabulary.**
   Pilcothink already quantized the MTP transformer layers (`mtp.layers.0.*`) to INT4 AutoRound in `model_extra_tensors.safetensors`. To prevent the MTP draft step from evaluating the full 248k vocabulary, `prepare/build_draft_vocab.py` slices a 40k-token draft head (`mtp.draft_lm_head.*`), guided by `prepare/draft_vocab_ids.json` or custom output frequencies.
6. **Multi-query verify optimizations (Split-KV and sort-free sampler).**
   During MTP verification, multiple draft tokens are evaluated simultaneously. FlashAttention-2 leaves many SMs idle on consumer GPUs when verifying small batches of query tokens. `patches/spec-decode-attn.patch` splits KV attention across thread blocks. In addition, `patches/sampler-small-topk-fast-softmax.patch` replaces expensive 248k full-vocabulary sorting with a fast path for top-$k \le 64$ (perfect for Ornith's recommended top-$k = 20$).
7. **Hybrid prefix caching.**
   Ornith blends 24 linear-attention GDN layers and 8 standard attention layers. Prefix caching (`--enable-prefix-caching`) caches both attention KV blocks and GDN recurrent states at chunk boundaries, slashing time-to-first-token on multi-turn conversations from 20+ seconds to <1 second.

### int8 prefill for single-user mode (`INT8_ACT=int8`)

Single-user mode stayed W4A16 because int8 activations buy nothing at batch
size 1 *decode* — true, and beside the point for prefill, which is
compute-bound at every concurrency. A torch profile of a 4k prefill on the
dflash2 stack is 79% Marlin GEMM time with 15 ms of GPU idle out of 2.05 s, so
the GEMM dtype is the whole game. `INT8_ACT=int8` (default layer set
`mlp|linear_attn|self_attn`) borrows batch mode's W4A8 path for every linear
except the int8-weight lm_head/embed and the MTP module:

| prefill tok/s (dflash2 k=15, PC=1, seeded protocol) | 1k | 4k | 16k | 51k |
|---|---|---|---|---|
| W4A16 (mode default) | 1,437 | 1,494 | 1,410 | 1,200 |
| `INT8_ACT=int8 INT8_LAYERS=mlp` | 1,638 | 1,696 | 1,587 | 1,320 |
| `INT8_ACT=int8` (all linears) | **1,845** | **1,937** | **1,791** | **1,423** |

Decode is unchanged (122±5 vs 121 tok/s C1 over repeats, 3.2 tok/step both
ways) and quality is the documented int8 trade: GSM8K 95.0% against the fast
variant's 96.5, perplexity +4.1% (mostly prose, code flat), IFBench flat on
the batch-mode precedent. The 51k row gains least because the 16
full-attention layers grow quadratically to ~40% of prefill there, and FA2 at
head_dim 256 has no faster sm86 alternative (FlashInfer measured within 1.5%,
and it costs the split-KV verify path at decode).

**int8-QK prefill attention** (`PREFILL_ATTN=int8`,
`patches/triton-prefill-attn-int8.patch`) attacks what is left after the GEMMs: the
16 full-attention layers, whose head_dim of 256 pins FA2 at 54-57 TFLOPS on
sm86 (85% of the card's practical fp16 mma rate — no fp16 rewrite can win).
A Triton kernel runs QK^T on int8 tensor cores at 2x the fp16 rate,
SageAttention-style: K is smoothed by its per-head channel mean (softmax-
invariant, so exact up to int8 rounding — cos > 0.99999 vs fp32 at 4-51k),
gathered and quantized once per layer-chunk into contiguous int8 scratch;
Q rows carry per-row scales; P.V stays bf16. On the attention itself it is
1.27x FA2 at 4k rising to 1.35x at 51k; end-to-end with `INT8_ACT` it adds
+2.7% at 16k and +5.3% at 51k (1,839 / 1,498 tok/s), decode unchanged.
Alone it adds almost nothing, and that is structural rather than a defect:
prefill without the int8 GEMMs is GEMM-dominated, so a faster attention has
less to win. Measured standalone on the reference box (`bench/prefill_ab.sh`,
two interleaved arms per condition, each reproducing to 0.1%): 1,167 / 1,126 /
1,012 tok/s at 4k / 16k / 51k against 1,164 / 1,114 / 980 stock — +0.3%,
+1.1%, +3.3%, the gain growing with context exactly as the attention share
does. A WSL2 3090 measured the same standalone cell 1-6% negative
([#62](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/62)), so treat
`PREFILL_ATTN` alone as noise-to-slightly-positive and pair it with
`INT8_ACT`, where the same kernel is worth +1.8% at 16k and +4.5% at 51k on
top (1,826 / 1,491 against 1,793 / 1,427).
Prefill-only by construction: the branch fires for single-request prefill
chunks on the exact serving geometry and falls through to FA2 otherwise.

Things this campaign measured that did NOT pay, so nobody re-walks them:

- **The benchmark harness was mismeasuring prefill.** `vllm bench serve`
  defaults to `--seed 0`, so every call replays the same prompts; with
  `--enable-prefix-caching` that hands later calls silent partial prefix hits
  whose size depends on the server's pool geometry. The old protocol read one
  config 15-20% low and another 3× high (a 16k prefill "measured" at 4.0 s
  that cold costs 11.2 s). `bench/run_benchmarks.sh` now seeds every call;
  the tables above are the seeded numbers, and older published prefill rows
  are not comparable.
- **`SPEC=off` prefills ~20% slower than `SPEC=dflash2`** — removing the
  drafter demotes the server from the V2 model runner to V1. The drafter's
  own prefill cost on the V2 runner is nil (6 kernel launches in a 4k
  profile); the "~15% TTFT" figure that used to circulate here predates the
  fused context-KV precompute.
- **`PREFIX_CACHE=0` prefills ~20% slower than `PREFIX_CACHE=1`** on this
  stack, align mode ruled out as the cause (PC=0 + `--mamba-cache-mode align`
  measures the same as plain PC=0). Do not turn the cache off "for speed".
- **Bigger prefill chunks do not boot** under the pinned `KV_MEM`:
  `--max-num-batched-tokens` 4096 and 8192 both inflate the profiled
  activation peak past the transient floor (engine init fails). 2048 stays.
- **Marlin tile tuning for the W4A8 GEMMs washes out end-to-end.** The
  standalone sweep (min-of-rounds, burst clocks) shows +2-20% per GEMM over
  stock tiles at M=2048 — and exactly +0.4% end-to-end, because sustained
  250 W throttling flattens the differences the bursts show.
  `patches/marlin-tune-table.patch` ships the wiring anyway (off by default,
  `VLLM_MARLIN_TUNE=1`) for cards running without a power cap.
- `INT8_LAYERS="mlp|linear_attn"` (the GDN-only middle point) crashes at
  first forward — an inductor codegen bug with the mixed set on this
  torch/vllm pin. Use `mlp` or the full default.

### DFlash2 (`SPEC=dflash2`, Phase 2)

> [!NOTE]
> **Phase 1 vs. Phase 2 (DFlash2 on Ornith-1.5-9B)**:
> The existing `syvai/Qwen3.8-27B-DFlash2-W4A16` drafter cannot be used with Ornith-1.5-9B because it was trained to consume hidden states from layers 5/19/33/47/61 of the 64-layer 27B model, whereas Ornith-1.5-9B has only 32 layers.
> On a 9B target, drafter overhead is proportionally much larger, so native MTP ($k=3/4$) with INT4 AutoRound weights and INT8 logits is expected to achieve ~150–190+ tok/s. Training an Ornith-specific DFlash2 drafter via vLLM `speculators` is planned as a Phase 2 extension. The reference notes below document how DFlash2 integration was engineered on this stack.

The one lever left after all of the above is acceptance, and Qwen's MTP head
is a single-layer chain drafter at its ceiling. [DFlash2](https://inco.ai/blog/dflash2/)
(Inco, Aug 2026) is a different drafter for this exact target:
5 Qwen3-style layers that predict the whole 7-token block in one
non-autoregressive pass from the target's layer hidden states,
plus a selector that walks a coherent path through 16 candidates per slot. On
the bf16 model it reports 4.80 tokens per step vs 4.28 for MTP at the same block
size. What it took to make it pay on a 24 GB card, in order:

1. **Native support plus the repo port.** vLLM's support is [PR #52816](https://github.com/vllm-project/vllm/pull/52816)
   and is native in v0.28.0 on the V2 model runner. The old
   `patches/dflash2-backport.patch` is retired on this tag; v0.28.0's native
   implementation is layered with `patches/dflash2-lookup-drafting.patch` and
   `patches/dflash2-ngram-chains.patch`, which carry the quantized candidate
   head, context lookup, and drafter-free chain extensions. The DFlash2 port
   also shares the target's *quantized* lm_head (upstream refuses), and the V2
   sampler now takes our sort-free small-k top-k/top-p path. MTP mode is untouched (re-measured:
   110.7 / 113.4 tok/s, 73,777-token pool).
2. **The drafter itself is 1.92B parameters, 3.85 GB in bf16** — read once per
   step, that is +5 ms on a 3090 and no gain (106 / 112 tok/s, measured), and it
   leaves a 21k-token KV pool. `drafter/capture_dflash2.py` hooks the drafter's
   own linear layers inside vLLM on 400 real prompts (~290k rows per layer, plus
   the context-KV precompute's input distribution for the k/v rows) and
   `drafter/quant_dflash2.py` GPTQ-quantizes the 36 matrices to W4A16
   compressed-tensors (Marlin): **1.19 GB**, shipped as
   [syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)
   (`prepare/fetch_dflash2.py`). int4 costs ~5% acceptance at default sampling (3.2 vs
   3.4 tokens per step) and nothing at greedy; keeping `fc` in bf16 did not
   recover it.
3. **Result** (`bench/run_benchmarks.sh single`, fast variant target): 26.5 ms
   per step vs MTP's 24.8, 3.14-3.34 tokens per step vs 2.8-2.9 → **117.8 tok/s
   at default sampling and 125.7 greedy at C1** (MTP: 111-115 / 115-124), with
   the best runs of this drafter reading 133.8 / 138.5, and a higher decode rate
   at C2-C8. Same output distribution by construction (perplexity 8.094,
   GSM8K 96.0-96.5%).

4. **Getting the context back to 64k** took a second patch
   (`patches/hybrid-kv-groups-v2-cudagraph.patch`), because the first version of
   this mode capped out at 40k. vLLM sizes a hybrid model's KV groups by the
   *smallest* bucket of same-type layers — with the drafter that is its 5
   sliding-window layers, so the target's 16 attention layers were padded to 20
   and its 48 GDN layers to 50: 25% more pool for every token of context, to pad
   the layers that were not the problem. Sliding-window groups only ever hold
   window-many blocks, so padding *them* costs ~7 MB per request instead. That
   takes the pool from 105 to 78 KB per token (MTP: 75), i.e. 45,383 tokens at
   40k → 69,758 at 64k. The same patch makes the V2 runner's CUDA-graph memory
   explicit (`VLLM_V2_CUDAGRAPH_MEM_MIB`): upstream it returns 0, so ~1.2 GiB of
   graphs lands on top of `--gpu-memory-utilization` — ask for 0.93, run at 0.98.
   Since the runner's profiled activation peak also swings ~1 GiB between starts,
   this mode pins the pool by bytes (`KV_MEM`, 5.2 GiB) rather than by
   utilization, and start-up is then deterministic (69,758 tokens twice over).

### Drafting from the context (`LOOKUP=1`)

The drafter reads a 2,048-token window. A long-context assistant spends much of its output
*reproducing* things — quoting a document, listing commands it was shown, rewriting a
paragraph while keeping the code — and those tokens are sitting verbatim in the prompt, tens
of thousands of tokens beyond what the drafter can see. `patches/dflash2-lookup-drafting.patch`
scans the request's own token history (the buffer vLLM already keeps) for the most recent
occurrence of the longest suffix of what has been generated so far, and proposes the tokens
that followed it — one Triton program per request, batch-size independent, with an
`NMIN`-token reject test before any candidate is extended. It stays lossless: greedy
verification never reads the draft distribution, and every position the lookup filled gets a
point mass on the proposed token, which is a legal proposal for vLLM's rejection sampler
(acceptance becomes p(x), residual computed from the same buffer).

Four things decide what that is worth.

**The verify block no longer has to be the drafter's block.** `dflash_config.block_size` is a
property of the checkpoint — 8 = one anchor plus the 7 mask tokens DFlash2 was trained for —
and vLLM made it the target's verify length as well, so a verbatim copy could never exceed 8
tokens per step. It sat on that ceiling: 7.83 of 8 accepted while reproducing a document's
first 60 lines. The drafter now keeps its own block while the target verifies a longer one
(`DFLASH_TOKENS`), and the positions past the drafter's block are filled from the context. They cost the drafter nothing — no extra mask tokens, no extra pass of its
candidate head — which is the point: the context is a free source of drafts, the drafter is
not.

**The long block is only scheduled while a copy is running.** Each extra verify position
costs about 1 ms of attention at 25k context, so the speculator reports per step how many of
its proposals the scheduler should actually put up for verification: the drafter's 7
normally, the whole block when (a) the lookup has a match with enough tokens left to fill
the tail, and (b) the step that just finished emitted at least a full short block's worth of
tokens — twice in a row. A single saturated step happens inside ordinary prose and the block
it buys is wasted; two in a row is a copy. The flags are read from a pinned copy that landed
asynchronously, one step stale: reading them synchronously is a device synchronise on every
decode step and measured 5%, more than the long block is worth on most work. vLLM only feeds
the draft count back to the scheduler on the synchronous scheduling path, so this mode runs
`--no-async-scheduling`; at batch 1 that costs under 1%.

Against the same server with the long block disabled, the trigger is a gain on every task in
the suite: +55% reproducing a document, +10% rewriting one, +2-3% on prose.

**The proposal is fused with the drafter's, not substituted for it.** A match of at least
`VLLM_DFLASH2_LOOKUP_NSTRONG` (8) tokens is taken on its own; a shorter one only if the
drafter independently proposed the same first `_AGREE` (2) tokens. Two independent sources
agreeing is the cheap confidence signal — the drafter looked at the hidden state, the lookup
looked at the text — and it is what stops a coincidental 6-token match from costing
acceptance on prose, which the all-or-nothing first version did.

**A match may overlap the suffix it matched**, so a repeating pattern (a list marker, an
indent, a fence) is proposed from its own period instead of missed.

Measured at 25k context, greedy, against the same server with the previous version
(tokens per step / decode tok/s):

| | no lookup | default (`DFLASH_TOKENS=7`) | `DFLASH_TOKENS=15` |
|---|---|---|---|
| reproduce the first 60 lines verbatim | 4.72 / 159 | 7.83 / 260 | **14.97 / 381** |
| shorten this, keep the commands | 2.70 / 90 | 3.19 / 107 | **3.50 / 113** |
| quote and explain | 3.01 / 101 | 3.21 / 107 | **3.35 / 110** |
| reproduce every command | 4.62 / 153 | 5.23 / 173 | 5.32 / 166 |
| free-form summary / Q&A | 2.15 / 72 | 2.08 / 69 | **2.13 / 71** / 2.01 / 67 |
| C1, 8 short chat prompts | 3.22 / 126 | 3.33 / 131 | **3.42 / 133** |

So the long block is worth **+47%** where the model reproduces its context, and a few percent
on most other work once it is only scheduled while a copy is actually running. It ships as a
mode — `DFLASH_TOKENS=15`, for a coding assistant applying edits or a RAG front-end quoting
sources — because it also costs 4 request slots instead of 8 and 56k of context instead of
64k. Quality is unchanged: GSM8K 96.5% (200 questions, greedy) with the lookup on, the
same as without it, and 7 of 9 long greedy prompts come back token-identical against the
same server with `LOOKUP=0` (the two that differ are near-tie flips, gotcha 14).

The mode also costs 4 request slots instead of 8 and 56k context instead of 64k, because
`--mamba-cache-mode align` reserves recurrent-state pages per slot per speculative block.

Pair it with `PREFIX_CACHE=1` (vLLM's hybrid prefix caching, opt-in upstream): a follow-up
turn on a 24k-token document costs **~1 s instead of ~23 s** because the attention KV is
reused and the recurrent state resumes from the last cached block boundary, for one extra
state page per request (~16% of the KV pool). Prefill stops dominating a chat, and then
drafting from the context is what makes the decode fast.

`PREFIX_CACHE=1` works in **batch mode** too, and matters just as much there: 64 requests
sharing one 5,820-token system prompt take 222 s without it and **16.9 s** with it (median
latency 94.9 s → 8.0 s), for ~14% of the KV pool and no change on workloads without a shared
prefix. If your API backend sends the same instructions with every request, this is the
single biggest thing in this repo for you. (The other three changes on this page —
lookup drafting, the KV-group fix and the V2 graph accounting — only fire on the DFlash2
path and are inert in batch mode.)

Its limits are memory- and window-shaped: each *resident* request holds 1+7
recurrent-state slots — 0.88 GiB, 15.8% of the `CTX=fast` pool, before it stores any
context — so six requests are resident and the seventh is preempted, against eight for
MTP's 0.44 GiB; from 8 concurrent long generations MTP's e2e is higher again (309 vs
235 tok/s). The drafter's 2,048-token attention window separately loses acceptance on
long-context tasks (2.3-2.6 vs 2.6-3.0 tokens per step at 12-36k, where MTP is 5-10%
ahead e2e). `CTX=long`/`huge` stay MTP.

It is not only memory-shaped, though, and that sentence used to say it was. Per-stream
decode falls hard with concurrency — 137 / 97 / 46 tok/s at 1 / 2 / 4 distinct
4k-token streams — because every resident request adds ~7 ms to the forward pass
(25.9 → 49.1 ms). Aggregate throughput still rises (137 → 309 tok/s) and nothing is
preempted, so the verify step is batching; it is latency per user that goes. MTP does
the same thing (126 / 103 / 46 per stream, ~5 ms per resident), so this is the hybrid
model plus speculation on one 3090, not something DFlash2 does wrong — what is
DFlash2's own is running out of state pages at five residents where MTP holds eight
and reaches 383 tok/s aggregate at C8. For **one person** with normal context — what single-user mode is
for — it is the fastest config in this repo. Full table in
[single-user/README.md](../single-user/README.md).
