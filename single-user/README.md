# Single-user mode for Ornith-1.5-9B

For one person (or a handful) chatting with the model: coding assistant,
local chat UI, agentic tool workflows, or anything where you're watching tokens stream in.
Launched with `bash single-user/start_ornith.sh`.

The difference from batch mode is native MTP speculative decoding. Ornith-1.5-9B ships
a native multi-token-prediction head (`mtp_num_hidden_layers = 1`) and the Pilcothink
checkpoint pre-quantizes its transformer layers to INT4 AutoRound. The model drafts ahead
and verifies the drafts in a single forward pass.
Speculative decoding is exact: the sampled distribution is identical to non-speculative
generation, with only decode latency improving.

## Benchmarks

Realistic chat prompts (8 mixed English/Danish/code tasks in
[bench/prompts_real.jsonl](../bench/prompts_real.jsonl), 1,024-token answers),
`vllm bench serve --dataset-name custom`, RTX 3090 at 250 W:

> These are vLLM 0.27.1 baseline measurements. Re-benchmark on a GPU after the
> v0.28.0 upgrade before using the figures for capacity planning.

**`CTX=fast` + fast variant (the default; 64k context)**, as reproduced by
`bash bench/run_benchmarks.sh single`:

| Cohort | decode, model-default sampling (T 1.0, top-p 0.95, top-k 20) | decode, greedy | tokens per step | e2e (default / greedy) | mean TTFT |
|---|---|---|---|---|---|
| C1 | **111.1 tok/s** | **120.0 tok/s** | 2.75 / 2.90 | 108.7 / 116.9 | 164 ms |
| C2 | 191.8 tok/s | 199.4 tok/s | 2.80 / 2.87 | 174.9 / 177.0 | 223 ms |
| C4 | 268.5 tok/s | 280.9 tok/s | 2.79 / 2.90 | 226.2 / 252.4 | 340 ms |
| C8 | 407.3 tok/s | 414.3 tok/s | 2.88 / 2.88 | 328.7 / 334.6 | 1,005 ms |

Decode throughput — C × 1000 / mean TPOT — is the number to read: it is the rate
you feel once generation starts. The e2e column includes prefill and the tail of
the slowest request, so it is lower by construction.
Per-position draft acceptance at C1: 74% / 50% / 34% / 24% (T 1.0), 77% / 55%
/ 40% / 30% (greedy). The best C1 repeats read 119 / 120 tok/s decode: greedy
generation is deterministic for a given server and request order, but a
different drafter config or a prefix-cache hit changes the text at near-ties
and with it the acceptance, so expect ±3-5% between runs. Quality of the fast
variant: perplexity 8.095 vs 8.045 for the base requantization (+0.6%,
en/da/code), GSM8K 96.5% (200 questions, greedy), same as the base.

**`CTX=long` (150k context, FlashInfer/fp8 KV, k=3)** with the fast variant:
95.3 / 100.3 tok/s at C1 (T default / greedy, 2.58 / 2.61 tokens per step).
With the base requantization and the earlier draft vocabulary, all cohorts:

| Cohort | decode, model-default sampling | decode, greedy | tokens per step | e2e (default / greedy) | mean TTFT |
|---|---|---|---|---|---|
| C1 | 84.7 tok/s | 89.3 tok/s | 2.46 / 2.43 | 83.4 / 87.1 | 179 ms |
| C2 | 168.2 tok/s | 177.6 tok/s | 2.47 / 2.42 | 146.6 / 160.3 | 233 ms |
| C4 | 289.2 tok/s | 303.7 tok/s | 2.46 / 2.47 | 256.8 / 256.0 | 358 ms |
| C8 | 409.0 tok/s | 450.5 tok/s | 2.37 / 2.50 | 327.9 / 364.2 | 1,069 ms |

Four drafts win up to two concurrent users; from C4 up the three-draft config
is ahead (rejected drafts cost more when the verify batch is bigger), so for
a shared box `CTX=long` or `DRAFT_TOKENS=3` is the better single-user config.
That trade is MTP's; do not carry it into `SPEC=dflash2`, whose concurrency
behaviour is different and is measured two sections down.
Batch mode does 45-46 tok/s single-stream on the same prompts, and overtakes
this mode from C8 up.

**`SPEC=dflash2` — Ornith-trained block drafter (Phase 2).** The Qwen3.8-27B
DFlash2 checkpoint cannot be used here (32 vs 64 layers, 4096 vs 5120 hidden,
taps 5/19/33/47/61 are OOB). Train against Ornith
([drafter/README.md](../drafter/README.md)), quantize to
`models/Ornith-1.5-9B-DFlash2-W4A16`, then:

```bash
venv/bin/python prepare/fetch_dflash2.py      # local trained dir or DFLASH2_REPO
SPEC=dflash2 bash single-user/start_ornith.sh
bash bench/dflash2_vs_mtp.sh dflash2          # C1 vs MTP; labd copy suite
```

Keep `SPEC=mtp` as the default until those benches beat `DRAFT_TOKENS=4`.
On 9B the extra ~1 GB drafter read is a larger fraction of step time than on
27B; copy/quote (`LOOKUP=1`, `DFLASH_TOKENS=15`) is the likely win, chat C1
is not promised. Do not copy 27B tok/s tables into this file.

Geometry: hidden 4096, taps 1/8/15/22/29 (full-attn ablation 3/11/19/27/31),
`fc` 20480→4096, `block_size` 8, `mask_token_id` 248077, `is_causal: false`.
The pool is sized from `GPU_UTIL` (set `KV_MEM=` only if V2 activation jitter
returns). `draft_sample_method` is `probabilistic` (required for lossless
sampling). WSL2: `VLLM_WSL2_ENABLE_PIN_MEMORY=1`.

### Chat with a long document: prefix caching, and drafting from the context


Two things matter once the prompt is long and the same document comes back every turn.

**`PREFIX_CACHE=1`** turns on vLLM's hybrid prefix caching (`--enable-prefix-caching
--mamba-cache-mode align`), which upstream keeps opt-in for hybrid models. The attention
KV of the shared prefix is reused *and* the recurrent (GDN) state resumes from the last
cached block boundary, so a follow-up turn does not re-run the document:

| 4-turn chat, 24k-token document, greedy | turn 1 | turn 2 | turn 3 | turn 4 |
|---|---|---|---|---|
| default | 22.9 s | 23.1 s | 22.8 s | 22.9 s |
| `PREFIX_CACHE=1` | 23.5 s | **1.15 s** | **0.85 s** | **0.89 s** |

Same answers, token for token (the state resume is exact, not approximate); a control run
that changes the prefix every turn pays the full 23 s again. It costs one extra recurrent
state page per request — the KV pool goes 86,727 → 72,475 tokens (MTP) / 68,605 (DFlash2) —
and draft acceptance is unaffected (2.23 / 2.03 / 2.28 tokens per step with the cache vs
2.27 / 1.80 / 1.96 without). Worth it for anything conversational; leave it off if you serve
unrelated one-shot prompts and want the pool.

**Lookup-augmented drafting** (`LOOKUP=1`, on by default for `SPEC=dflash2`) fixes the other
half. With prefill nearly free, long-context chat is decode-bound, and that is exactly where
the block drafter is weakest: it sees a 2,048-token window, so when the model quotes or
reproduces part of a 25k document, the drafter is guessing at text that is sitting verbatim
in the prompt. `patches/dflash2-lookup-drafting.patch` scans the request's own token history
for the most recent occurrence of the longest suffix of what has been generated and proposes
the tokens that followed it.

> The tok/s and tokens/step tables in this lookup section were measured on the
> **Qwen3.8-27B** fork this repo was derived from. They are not Ornith-1.5-9B
> numbers. Re-measure with `bash bench/dflash2_vs_mtp.sh dflash2-copy` after a
> trained `models/Ornith-1.5-9B-DFlash2-W4A16` exists. Do not quote 27B copy
> cells (≈380 tok/s) as Ornith results.

Those tokens cost the drafter nothing, which is why the verify block no longer has to be
the drafter's block. `DFLASH_TOKENS=15` — *reproduction mode* — has the target verify 16
tokens per step while the DFlash2 checkpoint keeps drafting the 7 it was trained for, and
the lookup fills the rest. That ceiling was binding: reproducing a document's first 60 lines
accepted 7.83 of 8 drafts per step, so the copy ran at the block size, not at what the
context could support. The long block is only *scheduled* while the lookup is firing — each
extra verify position costs about 1 ms of attention at 25k context — so ordinary steps still
verify 8 tokens. Lossless either way: greedy never reads the draft distribution, and every
position the lookup filled gets a point mass on the proposed token, which is a legal
proposal for the rejection sampler.

The long block is only *scheduled* while a copy is actually running. The lookup's flag says
a request has something to put in the tail; the trigger adds that the step that just
finished emitted at least a full short block's worth of tokens, twice in a row. One
saturated step happens in the middle of ordinary prose — a quoted phrase, a repeated list
marker — and the block it would buy is then wasted; two in a row is a copy.

Leaving is not the mirror of entering. The flag drops out for reasons that have nothing to do
with the copy ending — a line the lookup cannot match, or a flag copy that had not landed yet
— and dropping the long block immediately costs the two steps needed to earn it back, so it
is held for `VLLM_DFLASH2_LOOKUP_STICKY` (3) steps more. That is not a tuning knob for the
average; it is what makes the mode reproducible. Six consecutive runs of the same prompt on
one server, tokens per step / decode tok/s:

| 25k-token document, greedy | hold off | hold on (default) |
|---|---|---|
| reproduce the first 60 lines verbatim | 13.92 ×6 / 362 | **15.21 ×6 / 379 (+5%)** |
| "quote and explain" | 2.86 ×6 / 93 | **2.87–3.44 / 105 mean (+12%)** |
| free-form summary | 2.13 / 71.1 | 2.13 / 71.1 |
| free-form Q&A | 2.02 / 68.3 | 2.02 / 68.2 |

Free-form prose is untouched, which is what the entry condition guarantees: it never produces
two saturated steps in a row, so the hold is never armed. Gating the hold on "the request is
still emitting a full block a step" — which ought to tell a late flag from a finished copy —
removes the whole effect instead, because by the time the flag drops the step it describes
was not saturated either.

**The hold only applies with one request in flight**, and that is not caution. The counter is
one number for the whole batch, and unlike the entry condition it keeps the long block on
through steps where the flags say no — so with several requests, which block length a copying
request gets starts to depend on when the others arrived. Different block length, different
rounding, different greedy text: `bench/labd_soak.py` caught a verbatim copy coming out
differently in two rounds of an otherwise identical four-way batch, and reproducibly did not
with the hold off. Per-request block lengths would fix it properly, but the V2 runner's
CUDA-graph dispatch needs one query length for the whole batch
(`cudagraph_utils.py: get_uniform_token_count`), so a ragged batch loses its decode graphs
and pays more than the hold is worth.

| 25k-token document, greedy | default (`DFLASH_TOKENS=7`) | `DFLASH_TOKENS=15` |
|---|---|---|
| reproduce the first 60 lines verbatim | 7.83 / 260 | **14.97 / 381 (+47%)** |
| "shorten this, keep the commands" | 3.19 / 107 | **3.50 / 113 (+6%)** |
| "quote and explain" | 3.21 / 107 | **3.35 / 110 (+3%)** |
| free-form summary | 2.08 / 69 | **2.13 / 71 (+2%)** |
| free-form Q&A | 2.02 / 68 | 2.01 / 67 |
| "reproduce every command, verbatim" | 5.23 / 173 | 5.32 / 166 |
| C1, the 8 short prompts above | 3.33 / 131 | **3.42 / 133 (+2%)** |
| the six-task suite | 3.12 / 104 | **3.32 / 108 (+3%)** |

(tokens per step / decode tok/s, one server session.) Tokens per step is up on every task;
decode is up on five of seven, and the two that are not are inside the ±3-5% that greedy
text divergence produces between any two drafter configurations here.

Three things had to be true for that. The long block is only *scheduled* while a copy is
running — the lookup has something for the tail and the step that just finished emitted at
least a full short block's worth of tokens, two steps in a row to start, and three more of
coasting before it stops when there is one request in flight. The flag is read from a pinned
copy that landed asynchronously; reading it synchronously is a device synchronise per decode
step and costs 5%. And decode CUDA graphs are captured for
*both* block lengths: `decode_query_len` only described the long one, so every short step —
the common case — fell back to piecewise and paid 8% (27.9 ms against 25.9 ms for the same
8-token step on a 7-slot server).

Reproduction mode also costs KV pool per request slot rather than per token
(`--mamba-cache-mode align` reserves state pages per slot per speculative block), so it runs
4 slots and 56k of context instead of 8 and 64k.

Quality is unchanged: GSM8K 96.5% (200 questions, greedy) with the lookup on, the same as
without it, and 96.0% with the hold — one question, which is what a 200-question sample
resolves. `bench/labd_soak.py` is the check that matters for the parts of this that are
decided for the whole batch: it runs a verbatim copy inside a mixed four-way batch and
insists it comes out the same in every round. Greedy *text* matches on 7 of 9 long prompts against the same server with
`LOOKUP=0`; the two that differ are near-tie flips of the kind any drafter change produces
here (the block is one chunk through the recurrent layers, so changing what is in it changes
the last bits of the logits — gotcha 14), not a change of distribution: the lookup's
positions carry a point mass, which is a legal proposal for the rejection sampler, and
greedy verification never reads the draft distribution at all.

The same server measured on random tokens (256 in, 1,024 out) reads anywhere from
35 to 151 tok/s depending on what the model makes of the noise — which is why the
tables above use real prompts. For comparison against another engine on this card,
[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) publishes 71.00 tok/s decode
at C1 on its own protocol (short real prompts, thinking on); the caveats are in the
[main README](../README.md#vs-ninfer-3090).

### How the draft got cheap

Every draft token is one pass through the MTP module plus a full lm_head
projection, k times per step. As shipped that is ~3 ms per draft on this card
(850 MB bf16 module + 1.3 GB int8 head), so MTP-2 was the sweet spot and MTP-3
already lost. Three changes, measured on the prompts above (T = default /
greedy):

| config | tok/s | tokens/step | acceptance pos 0 |
|---|---|---|---|
| no speculation (batch mode) | 46 / 46 | 1.0 | — |
| MTP-2 as shipped: bf16 module, full head, fp32 state | 66 / 79 | 2.1 / 2.4 | 65% / 80% |
| MTP-4: int8 module (`prepare/quant_mtp.py`), 40k-token draft head (`prepare/build_draft_vocab.py`), fp16 state | 78 / 99 | 2.2 / 2.7 | 58% / 70% |
| + `draft_sample_method: probabilistic` | 90 / 98 | 2.6 / 2.7 | 69% / 70% |
| same, k=3 (`CTX=long`) | 84 / 89 | 2.5 / 2.4 | 69% / 71% |
| same, k=6 | 76 / 94 | 2.3 / 2.7 | |
| k=4, full 248k head instead of 40k | 85 / 91 | 2.85 / 3.0 | 74% / 76% |
| k=4, bf16 `mtp.fc` (rest int8) | 88 / 96 | 2.6 / 2.6 | 67% / 70% |
| + sampler patch + split-KV verify attention | 93 / 99 | 2.6 / 2.6 | 69% / 70% |
| + draft vocabulary counted over the model's own outputs | 107 / 109 | 2.9 / 2.9 | 74% / 74% |
| + GPTQ-int4 lm_head, int4 draft head | 109 / 112 | 2.8 / 2.8 | 73% / 73% |
| **+ GPTQ-int4 MTP module (fast variant, shipped)** | **~114 / ~124** | 2.8 / 3.0 | 74% / 77% |
| same, k=5 | 106 / 105 | 3.0 / 2.9 | |
| same, 49k draft vocab | 109 / 115 | 2.7 / 2.8 | |
| same, `draft_sample_method: greedy` | 97 / 124 | 2.3 / 3.0 | |

The cheap drafter loses a few points of acceptance to int8/int4 (GPTQ with a
calibration set from the model's own hidden states loses none) and wins
because a step dropped from ~32 to ~24 ms while carrying more drafts. The
truncated vocabulary only costs acceptance if it's the wrong vocabulary: the
list counted over the model's own outputs (97.5% coverage of what it
generates) is the largest single step in the table; the earlier web-text list
(92%) had been silently capping acceptance at every position. Probabilistic
drafting samples the draft from the MTP distribution instead of taking its
argmax, which is what rejection sampling wants at temperature > 0; at greedy
it changes nothing. See [drafter/README.md](../drafter/README.md) for how the
vocabulary and the calibrated int4 tensors are made (and for the fine-tuning
attempt that did not help).

k=4 is the fastest but not the default: on the FlashInfer attention backend
(the only one that supports fp8 KV on Ampere, and fp8 KV is what makes 150k
context fit) the vLLM 0.28.0 FlashInfer path dies with an illegal memory access as soon as one
request finishes while another is mid-generation with 4 drafts (with or
without our patches; the vendored PR #50021 bounds fix does not cure it;
club-3090 sees the same "n=4 eventually dies, n=3 stable" on their rigs, and
vLLM has a family of open MTP illegal-memory-access reports on Qwen3.5/3.6,
e.g. [#40756](https://github.com/vllm-project/vllm/issues/40756),
[#36498](https://github.com/vllm-project/vllm/issues/36498)). The same k=4
config on the FlashAttention backend (bf16 KV) runs clean at C2/C4, so the
bug is in the FlashInfer spec-decode path. Hence two configs:

- `CTX=fast` (default): FlashAttention, bf16 KV, **~64k context**, k=4, split-KV
  verify attention → ~114 / ~124 tok/s with the fast variant
- `CTX=long`: FlashInfer, fp8 KV, **150k context**, k=3 → 95 / 100 tok/s with
  the fast variant (84 / 89 with the base requantization)

k=3 passed every concurrency soak we ran (C2/C4/C8 with staggered finishes,
100k-token prompt, 4×6k-token generations); if you see the crash anyway,
`DRAFT_TOKENS=2` costs ~5% and is the most conservative setting.

Why not 150? The verify pass alone reads ~13 GB of weights (~17 ms at what
this card actually delivers on 16-92 MB reads) plus ~4 ms
of drafts and sampling, and Qwen's MTP head agrees with the target on ~75-77%
of first drafts on real text once it can propose the right tokens, so ~3
accepted tokens per ~24 ms step is where a single-layer chain drafter tops
out. Random-token benchmarks that show 150+ are measuring how repetitive
noise is. Fine-tuning the head on the model's own outputs did not raise its
top-1 agreement (`drafter/README.md`); a tree drafter would, but the
DeltaNet layers can't verify a tree.

## Setup

Do the [common setup](../README.md#setup) first (venv, model download,
requantization, draft head, the fast variant via `prepare/fetch_fast_variant.py`, vLLM
patches; `bash verify.sh --no-server` checks all of it). Then:

```bash
bash single-user/start_ornith.sh                # MTP (64k context)
venv/bin/python prepare/fetch_dflash2.py      # Ornith DFlash2 (after training)
SPEC=dflash2 bash single-user/start_ornith.sh # DFlash2 (needs trained W4A16 drafter)
bash bench/run_benchmarks.sh single           # MTP / DFlash2 C1 tables
bash bench/dflash2_vs_mtp.sh dflash2          # plus copy/quote (labd)
```

If you are the only person on the card, this is the fastest configuration here —
the block drafter, a verify block the context fills, and the document you
already sent kept across turns:

```bash
SPEC=dflash2 DFLASH_TOKENS=15 PREFIX_CACHE=1 bash single-user/start_ornith.sh
```

Reproduction mode: 4 request slots and 56k of context instead of 8 and 64k.
Measure it with `bench/labd_bench.py` after a trained Ornith drafter exists;
do not quote the 27B 382 tok/s copy cell as an Ornith number.

Or in Docker (image build, model prep and the same knobs via `.env` — see the
[docs/docker.md](../docs/docker.md)):

```bash
docker compose --profile single up -d
```

Or as a service:

```bash
mkdir -p ~/.config/systemd/user
cp single-user/qwen-serving.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qwen-serving
loginctl enable-linger $USER
```

Point your chat client at `http://<host>:18020/v1` with the key from
`api_key.txt`. Works with anything that speaks the OpenAI API, tool calling
included (`tools` + `tool_choice: "auto"` come back as `tool_calls`).

## Knobs

| var | default | notes |
|---|---|---|
| `MODEL` | `models/Ornith-1.5-9B-MixedInt4-AutoRound` | serving INT4 teacher + MTP |
| `CTX` | `fast` | `fast`: bf16 KV / FlashAttention / 64k / 4 drafts / split-KV attention. `long`: fp8 KV / FlashInfer / 150k / 3 drafts, ~15% slower at C1, faster from C4 up. `huge`: KVarN 4/2-bit KV / 200k / 3 drafts (needs `bash kvarn/install.sh`; docs/long-context.md) — buys 1.7x the pool for **half the decode rate past ~100k** (32.0 vs 68.1 tok/s at 112k), so take it when the request would not otherwise fit, not for speed |
| `KV` | (follow `CTX`) | `int8pth`: Triton `int8_per_token_head` (~150k, k=4, split-KV on). FlashInfer-free long context; FA2 cannot store quantized KV on a 3090. `fp8`: force FlashInfer fp8 (same as `CTX=long`). Incompatible with `CTX=huge`. `SPEC=dflash2 CTX=long` already uses int8 |
| `ENABLE_THINKING` | `0` | `0`: `--default-chat-template-kwargs={"enable_thinking":false}` and greedy `0.0/0.80/20`. `1`: thinking on (`1.0/0.95/20`). Per-request `chat_template_kwargs` still wins |
| `INT8_ACT` | (off) | `int8`: W4A8 Marlin GEMMs for prefill (`VLLM_MARLIN_INPUT_DTYPE`). Decode at C1 unchanged. Pair with `PREFILL_ATTN` |
| `PREFILL_ATTN` | (off) | `int8`: int8-QK Triton prefill on the 8 hd256 full-attn layers (`patches/triton-prefill-attn-int8.patch` + `triton-prefill-attn-ornith-heads.patch`). Companion to `INT8_ACT`; empty keeps FA2. Quantized KV falls through. `fp16` is the unquantized kernel (debug) |
| `PREFIX_CACHE` | 0 | 1 = reuse a shared prompt prefix across requests (`--enable-prefix-caching --mamba-cache-mode align`): 20x faster follow-up turns, ~16% smaller KV pool |
| `LOOKUP` | 1 (`SPEC=dflash2`) | draft from the request's own context when it repeats itself (`patches/dflash2-lookup-drafting.patch`), and fill the verify positions the drafter's block does not reach. `VLLM_DFLASH2_LOOKUP_NMIN` (6) is the shortest suffix that may match, `_NMAX` (12) the longest — the kernel prefers the longest match and breaks ties by recency, so a higher cap makes it choose an older long match over a newer short one, which is the worse predictor — `_NSTRONG` (6) the match length trusted on its own, `_AGREE` (0) how many tokens the drafter must independently agree on for a shorter match to be taken, `_NMIN_TAIL` (4) the same for positions the drafter never proposed, `_ADAPTIVE` (1 = ask the scheduler for the long block only while a copy is running, 0 = always long), `_LONGMIN` (6) the match length that counts as a fillable tail, `_STICKY` (3) steps to hold the long block after the flag drops, with one request in flight only — copies do not end when the flag says so, and re-entry costs two steps, but the counter is batch-wide and holding it across a mixed batch makes the block length depend on when the other requests arrived — `_CHEAP_CTX` (0 = off) a context length below which the long block is taken unconditionally |
| `SPEC` | `mtp` | `dflash2`: Ornith DFlash2 block drafter (`prepare/fetch_dflash2.py`; V2 runner). `DRAFT` overrides the drafter dir (must be hidden 4096 / 32-layer taps). `DFLASH_TOKENS` (7) is the *verify* block — the checkpoint always proposes the 7 it was trained for; 15 is reproduction mode. Size the KV pool from `GPU_UTIL` unless you set `KV_MEM`. `VLLM_DFLASH2_DRAFT_TOPK_TOPP=0` disables proposal truncation, `VLLM_DFLASH2_TORCH_TOPK=1` avoids FlashInfer top-k JIT |
| `DRAFT_TOKENS` | 4 (3 for `CTX=long`/`huge`; 4 with `KV=int8pth`) | speculative depth; 5 and 6 are slower |
| `SPEC_ATTN` | 1 (`CTX=fast` and `KV=int8pth`) | split-KV Triton attention for the verify step (`patches/spec-decode-attn.patch` / `spec-decode-int8-kv.patch`); 0 = FlashAttention-2 / unified Triton |
| `DRAFT_SAMPLE` | `probabilistic` | `greedy` drafts: same speed at T=0, ~15% slower at T>0 |
| `MAX_SEQS` | 8 | how many requests are *admitted*, not how many the pool can hold: each resident request needs k+1 recurrent-state slots (0.88 GiB at DFlash2 k=7 — seven residents with short prompts, five at 4k, two at 16k), and the launcher prints the number at boot |
| `MAX_LEN` | 65536 (`fast`) / 150000 (`long`) | 150k needs `GPU_UTIL` 0.93 |
| `GPU_UTIL` | 0.93 | soak-tested with a 100k prompt and 4×6k-token generations; batch mode's 0.972 OOMs in the MTP path (docs/gotchas.md, gotcha 4) |
| `MTP_DRAFT_VOCAB` | 1 | set 0 to draft with the full lm_head (more acceptance, slower per draft) |
| `TOOLS` | 1 | tool/function calling (`--enable-auto-tool-choice --tool-call-parser`). `TOOL_PARSER` (`qwen3_coder`) must match the XML call format this model's chat template emits — `hermes` parses the JSON a Qwen model does *not* produce here, and fails silently. 0 = off, and `tool_choice: "auto"` then 400s |
| `VISION` | 0 | 1 keeps the vision tower instead of `--language-model-only` (0.858 GiB of BF16 weights on this checkpoint), for a client that sends images: one image per prompt and a 2048-image-token pixel cap, both overridable from `EXTRA_ARGS` |
| `VISION_OFFLOAD` | 1 | with `VISION=1`, keeps the tower's weights in pinned host RAM and copies each module to the GPU for its own forward (`patches/vision-tower-cpu-offload.patch`). **On 24 GB, `SPEC=dflash2` + `VISION=1` does not boot with this off** — the tower is 0.85 GiB of the ~1.1 GiB transient margin, and graph capture OOMs allocating the 960 MiB split-KV verify buffer with 787 MiB free. With it on, the same config comes up at the full 69,758-token pool and reads images. Costs 296 → 333 ms of encode per 8192-patch image, output bit-exact. 0 only on a card with headroom to spare. `VLLM_VISION_CPU_OFFLOAD_GB` (default 1) is the budget in GiB |
| `REQ_METRICS` | 0 | 1 = `--enable-per-request-metrics --enable-force-include-usage`: per-request timing fields in every response and `usage` on every request, the fields llama-swap's dashboard reads ([#51](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/51)). `prompt_tokens_details.cached_tokens` is always on. Not compatible with `--disable-log-stats` in `EXTRA_ARGS`; vLLM's per-request spec-decode summary flag is nightly-only, not in 0.27.1 |
| `PORT` | 18020 | |
| `SLEEP_LEVEL` | 1 | 1 = park GPU after `VLLM_IDLE_TIMEOUT` (90s) and auto-wake on the next chat request (`docker/vllm-idle-proxy.py`). 0 = off. 2 = discard weights |

## Switching modes

Only one mode can run at a time (one GPU). Swap by replacing the unit file:

```bash
systemctl --user stop qwen-serving
cp batch/qwen-serving.service ~/.config/systemd/user/   # or single-user/
systemctl --user daemon-reload
systemctl --user start qwen-serving
```
