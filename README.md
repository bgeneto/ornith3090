# Ornith-1.5-9B on one RTX 3090

Serving setup for [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) (starting from [Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound](https://huggingface.co/Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound)) on a
single 24 GB consumer GPU (RTX 3090) with vLLM — up to 262k token context and an OpenAI-compatible
API with key auth, in two ready-made modes.

## Quick start

The image builds on top of vLLM 0.28.0, applies all `patches/` and runs `verify.sh` as its
gate. The first start downloads and requantizes the model (~9 GB, once, into `./models`), and serves
on port 18020. Pick a mode — one GPU serves one at a time:

```bash
git clone https://github.com/bgeneto/ornith3090 && cd ornith3090

cp .env.example .env                 # Linux / WSL
# PowerShell: Copy-Item .env.example .env

docker compose --profile single up -d    # low-latency single-user chat / coding agent (native MTP k=4)
docker compose --profile batch  up -d    # API backend, many concurrent requests
```

After 90 seconds with no inference, the GPU parks (vLLM sleep level 1: weights offloaded to CPU, KV cache discarded). The next chat request auto-wakes; `GET /health` stays 200 so Compose does not flip the container unhealthy. Listing models does not count as activity. `SLEEP_LEVEL=0` keeps the engine resident; `VLLM_IDLE_TIMEOUT` (default 90) is the quiet period.

The example uses the recommended single-user `SPEC=mtp` profile with `DRAFT_TOKENS=4`. If Docker
Desktop is using WSL2, keep `VLLM_WSL2_ENABLE_PIN_MEMORY=1` enabled in `.env`. The example
leaves API-key authentication disabled for local-only use; set `VLLM_API_KEY`
before exposing the server beyond this machine.

| | `--profile batch` → [batch/](batch/) | `--profile single` → [single-user/](single-user/) |
|---|---|---|
| for | API backends, pipelines, many concurrent requests | one or a few people chatting, coding agents |
| single-stream (C1) decode rate | ~90–120 tok/s stock | **~150–190+ tok/s** with native MTP $k=4$ + 40k draft vocab + INT8 `lm_head`/`embed_tokens` (potentially ~200+ tok/s on high-acceptance code) |
| memory footprint | ~8.5 GB weights + FP16 GDN recurrent state | ~8.5 GB weights, leaving ~15.5 GB VRAM for extensive KV cache and CUDA graphs |
| trick | 16-bit recurrent state + INT8 tensor-core GEMMs | Native MTP speculation ($k=3/4$), INT8 `lm_head` + `embed_tokens`, 40k-token draft vocabulary, split-KV verify attention, sort-free sampler, hybrid prefix caching |

Both modes share one install — the mode is just which launch script you run.
Speculation wins below ~8 concurrent users on short prompts, plain batching above.
Numbers are on an RTX 3090 at a 250 W power limit.

The server listens on `0.0.0.0` and is unauthenticated unless you give it a key.
For anything past your own machine, add one first — everything reads it from
`.env` or `api_key.txt`, and nothing needs it otherwise:

```bash
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env
```

Or run via Docker without compose:

```bash
docker run -d --name ornith --gpus all --ipc=host -p 18020:8000 -e PORT=8000 \
  -v ornith-models:/app/models -v ornith-cache:/cache \
  --restart unless-stopped ornith15-9b-rtx3090:latest
```

Or by hand in a venv (same steps: model download, requantization, vLLM
patches, `verify.sh`) — see [Setup](#setup).

### If you are the only user, do this

The default single-user profile is tuned for maximum single-stream responsiveness using Ornith's native MTP head:

```bash
printf 'SPEC=mtp\nDRAFT_TOKENS=4\nPREFIX_CACHE=1\n' >> .env
docker compose --profile single up -d
```

or, in the venv install:

```bash
SPEC=mtp DRAFT_TOKENS=4 PREFIX_CACHE=1 bash single-user/start_ornith.sh
```

`SPEC=mtp` with `DRAFT_TOKENS=4` leverages Ornith-1.5-9B's native 1-layer MTP head, combined with our sliced 40k draft vocabulary and INT8 `lm_head`/`embed_tokens` quantization. `PREFIX_CACHE=1` keeps prompt context (both attention KV and GDN recurrent state) in cache for instant follow-up turns.

`SPEC=dflash2` is optional Phase 2: a 5-layer block drafter trained **on Ornith-1.5-9B**, not the published Qwen3.8-27B DFlash2 checkpoint. That 27B drafter taps layers 5/19/33/47/61 of a 64-layer / 5120-d model; Ornith has 32 layers and hidden size 4096, so `fc` cannot even load.

Train and quantize first ([drafter/README.md](drafter/README.md)), then:

```bash
# after models/Ornith-1.5-9B-DFlash2-W4A16 exists
SPEC=dflash2 PREFIX_CACHE=1 bash single-user/start_ornith.sh
# reproduction mode (verify 16, draft still 7; lookup fills the tail)
SPEC=dflash2 DFLASH_TOKENS=15 PREFIX_CACHE=1 bash single-user/start_ornith.sh
```

Default single-user remains `SPEC=mtp` until C1 benches beat MTP k=4. On a 9B target the drafter is a larger fraction of step time than on 27B; chat C1 is a measurement, copy/quote (`LOOKUP=1`, `DFLASH_TOKENS=15`) is the likely win. Compare with `bash bench/dflash2_vs_mtp.sh`.

Geometry (do not copy 27B numbers): hidden 4096, taps **1, 8, 15, 22, 29**, `fc` 20480→4096, `block_size` 8, `mask_token_id` **248077** (not 248070, which is `<|audio_start|>` in this tokenizer), `is_causal: false`. KV pool is sized from `GPU_UTIL` (Ornith weights ~8.5 GB); do not pin the 27B `KV_MEM=5.2GiB`. WSL2 still needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (V2 runner UVA). `CTX=huge` + DFlash2 needs `bash kvarn/install.sh`.

`VLLM_DFLASH2_CHAIN=1` adds drafter-free n-gram chains on copy (`patches/dflash2-ngram-chains.patch`); off by default.

### Third-party checkpoints (uncensored builds and others)


`MODEL=` points the launchers at any Qwen3.8-27B checkpoint in the same
`compressed-tensors` shape. Two routes, easiest first.

**Ready-made:**
[leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound](https://huggingface.co/leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound)
([#45](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/45)) is an
abliterated Qwen3.8-27B already quantized with this repo's own recipe —
AutoRound W4A16 body plus the `prepare/` head requant — so it serves without
any preparation. Its author measured ~100 tok/s warm at `SPEC=dflash2
CTX=huge` on a 3090 with coherent output and a 45k-context needle retrieved,
and a second tester confirmed `SPEC=mtp` works. Community-built and
community-verified; not benchmarked on this repo's reference box.

**Any other export**, including single-shard and asymmetric-AWQ ones the base
model's three `quant_*.py` scripts cannot open, goes through the streaming
requant (contributed in
[#37](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/37)). The worked
example is
[philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ)
— an abliterated (de-refused) Qwen3.8-27B, W4A16 AWQ, with the vision tower and
the grafted MTP head both preserved. Prepare it once, then serve it:

```bash
venv/bin/python prepare/fetch_thirdparty.py          # ~18.6 GB; or: fetch_thirdparty.py <hf-repo>
venv/bin/python prepare/quant_heads_stream.py models/Qwen3.8-27B-Uncensored-W4A16
venv/bin/python prepare/build_draft_vocab.py  models/Qwen3.8-27B-Uncensored-W4A16 \
  --ids prepare/draft_vocab_ids.json

MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 SPEC=mtp CTX=long PREFIX_CACHE=1 \
  MAX_LEN=100000 bash single-user/start_qwen.sh
```

It needs `prepare/quant_heads_stream.py` rather than the three `quant_*.py` steps
the base model uses, for two reasons that are properties of the checkpoint and not
of the model: it ships as **one 18.6 GB shard**, which the three scripts read into
RAM whole before rewriting, and its body is **asymmetric AWQ**, which those scripts
would copy onto the symmetric tensors they write — vLLM then looks for a
`weight_zero_point` that was never written. The streaming script handles both and
produces the same tensors otherwise; `bash verify.sh --no-server` with `MODEL=` set
checks the result exactly as it checks the base model.

**`SPEC=dflash2` needs its pool resized for this checkpoint.** After requantization
it is 15.68 GiB of weights against the fast variant's 14.71, and the DFlash2 branch
pins the KV pool *in bytes* (`KV_MEM`) rather than sizing it from
`--gpu-memory-utilization`, so the pool does not give that gigabyte back. The server
loads, captures graphs, and then dies on the split-KV verify buffer:

```
Model loading took 15.71 GiB
reserved 5.2 GiB memory for KV Cache as specified by kv_cache_memory_bytes config
torch.OutOfMemoryError: Tried to allocate 960.00 MiB ... 926.44 MiB is free
```

Hand that gigabyte back and it comes up. `CTX=long` (int8 KV) is the one to spend it
on, because it buys roughly twice the context per byte of pool that `CTX=fast` does:

```bash
MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 SPEC=dflash2 CTX=long PREFIX_CACHE=1 \
  KV_MEM=4456028569 DFLASH_MAX_LEN=98304 bash single-user/start_qwen.sh
```

Measured here, RTX 3090 at 250 W: a 4.15 GiB pool holding **103,033 tokens** at
98,304 `max-model-len` (4.6% margin) and **85.7 tok/s** greedy on a 400-token
answer. The checkpoint keeps its vision tower, and that run had `VISION=1` — images
came back described correctly — so the numbers are an upper bound on what the
default `VISION=0` needs, which drops the tower's weights entirely.

`SPEC=mtp` needs no `KV_MEM` of its own: its pool is profiled from `GPU_UTIL` rather
than pinned, so it absorbs the extra gigabyte by shrinking the pool for you. It is the
mode to reach for first on this checkpoint. The pool it lands on will not hold
`CTX=long`'s stock 150k, though, which is what the `MAX_LEN=100000` above is — the
figure this checkpoint has been run at.

### 256k the stock way: int4 KV (`single-user/alternative.sh`, experimental)

```bash
bash single-user/alternative.sh      # TRITON_ATTN + --kv-cache-dtype int4_per_token_head
```

Where KVarN reaches 268k with its own kernels, vLLM's stock
`int4_per_token_head` cache now combines with the DFlash2 drafter too:
**314,915 tokens of pool at 256000 max-model-len** (1.23× concurrency) on one
24 GB card, no `kvarn/install.sh`. Three boot blockers stood in the way — a
padded-page view error under the hybrid block-promotion geometry, and a
causal-only assert plus missing per-seq-causal plumbing in the int4 Triton
kernel, which the drafter's 8-row draft block needs
(`patches/int4-kv-per-token-head.patch`, contributed in
[#42](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/42) by @lachhabw).

The trade: the Triton attention backend plus the per-step int4 unpack cost
about 20% of decode against the shipped config on short prompts (~86 vs ~104
tok/s e2e on the same probe), and — unlike KVarN, which has GSM8K and
100k-needle numbers above — int4-KV quality at depth now reads:

| metric | result |
|---|---|
| GSM8K exact-match (200 questions, greedy, thinking off) | **96.0%** |
| 100k-token needle at 90% depth (`bench/needle_test.py`) | **retrieved** |

Measured on an RTX 4090 (24 GB) with `bench/quality_battery.py int4kv --gsm-only
--gsm-n 200` and `bench/needle_test.py 100000 0.9`; the 96.0% sits inside the band
the other configurations read (95.0-96.5%, docs/quality.md). Tool calling
round-trips correctly and the lookup lane works; the rest of this configuration
is still experimental.

### More than one GPU

Everything here is written for one 24 GB card; multi-GPU goes through untouched
via `EXTRA_ARGS`:

```bash
SPEC=mtp PREFIX_CACHE=1 EXTRA_ARGS="--tensor-parallel-size 2" bash single-user/start_ornith.sh
# after a trained Ornith DFlash2 checkpoint:
# SPEC=dflash2 PREFIX_CACHE=1 EXTRA_ARGS="--tensor-parallel-size 2" bash single-user/start_ornith.sh
```

What the second card is worth on the **Qwen3.8-27B** stack was measured in
[#40](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/40) (same box, PCIe 4.0 x8, **no
NVLink**, 275 W). Those DFlash2 tok/s cells are not Ornith numbers; re-run
`bench/run_benchmarks.sh single` here before planning TP=2:

- **The DFlash2 residency ceiling is a 24 GB property, not a drafter
  property.** On one card a 5-layer block drafter can collapse at C8
  (recurrent-state pool exhaustion) while MTP still fits. On two cards the
  extra pool usually flips that. Measure on Ornith before pointing everyone at
  DFlash2.
- **Do not pin `KV_MEM` under TP>1** unless you have profiled it: a single-card
  constant applied per worker strands memory. Export `KV_MEM` to pin anyway.
- **Keep `DFLASH_TOKENS=7` at TP>1** until T=15 is measured on this 9B target.
- NVLink appeared to buy little on the 27B A/B
  ([#7](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/7) vs #40).

Also reported working on the 27B lineage: **2× RTX 5060 Ti 16 GB**
([#22](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/22)) — the "would
not fit on one card" case. The graph budget and `MAX_SEQS` defaults are still
single-card calibrations.

## Benchmarks

Full tables per mode in [batch/README.md](batch/README.md) and
[single-user/README.md](single-user/README.md); quality in
[docs/quality.md](docs/quality.md). Reproduce any of it with
`bash bench/run_benchmarks.sh batch|single` against your own server.

### vs. ninfer-3090

[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) is a standalone C++/CUDA engine
that publishes cohort benchmarks for this model on this card. Theirs are 1,024-token
answers from 29-34-token prompts, greedy, MTP3, int8 KV, prefix reuse off, an
8,192-token context window, and **thinking on** at `reasoning_effort=medium`, so their
1,024 tokens include reasoning. Ours are 8 realistic chat prompts (English, Danish,
code), 1,024-token answers, model-default sampling, thinking off:

| Cohort | ninfer-3090 (MTP3) | this repo, batch | single-user, MTP |
|---|---|---|---|
| C1 | 71.00 tok/s | 45.5 | **111.1** |
| C2 | 90.66 tok/s | 86.3 | **191.8** |
| C4 | 100.28 tok/s | 168.3 | **268.5** |
| C8 | 165.33 tok/s | 324.9 | **407.3** |
| C64 (128 in / 512 out) | not supported | **~1,035** | — |

`SPEC=dflash2` is not in this table until `bash bench/dflash2_vs_mtp.sh` has
numbers from a trained Ornith-1.5-9B drafter. The Qwen3.8-27B DFlash2 C1 cells
(121.8 / 195.5 / …) do not apply: that checkpoint cannot load here.

Decode rate, C × 1000 / mean TPOT. MTP columns were re-measured together
on the current stack with `bench/run_benchmarks.sh`, keeping the second run after each
restart as the script advises. Run-to-run spread on the same server is
5-8%, so treat one-decimal differences as noise.

Theirs is the **decode** column of their table; their end-to-end column reads
70.19 / 89.43 / 97.89 / 161.28, and an earlier version of this table quoted *those*
against our decode rate, which was not like-for-like. What still is not like-for-like,
in their favour and ours: their C1 is a single prompt in a single run with no error
bars, thinking is on for them and off for us, and they publish no power limit or driver
version — ours is an RTX 3090 pinned at 250 W. Peak VRAM is comparable (23.0 vs
22.1 GiB at C8). The gap is mostly vLLM's continuous batching plus the memory this
repo's requantization frees up.

### Quality

The whole stack is quantized, so the honest question is what it costs. Short
version: **IFBench 78.3** prompt-level strict vs 79.5 for the unquantized model
(one point), **perplexity 8.09** on 33k held-out tokens, **GSM8K 96.5%** (200
questions, greedy). Speculative decoding — MTP, DFlash2 and the lookup drafter —
is exact by construction and changes none of it; the int8-activation steps in
batch mode are the only knobs that trade accuracy for speed, and they cost
0.9-3.7% perplexity depending on how far you push them. Per-configuration
tables: [docs/quality.md](docs/quality.md).

### Results from other hardware

Community reproductions of the single-user headline number, harness runs first.
`bench/run_benchmarks.sh single`, greedy, second run (the first reads low):

**Set the power limit before you compare anything.** Every number in this repo
is an RTX 3090 at 250 W, and on this card that is not a soft preference. A
sustained-load ladder from [#62](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/62)
(14 minutes per cell, same service): 200 W gives 57.5 tok/s at 781 MHz, 250 W
gives 85.6 at 978 MHz, and 280 W gives 86.7 — it hits 90 °C within two minutes,
pins the fan at 100% and throttles back to the same throughput. Prefill loses
about the same third at 200 W. So a quiet home box capped at 200 W is measuring
its power cap rather than this stack, and nothing above 250 W is worth the
noise.

| card | power | C1 decode | notes | source |
|---|---|---|---|---|
| RTX 3090 (reference) | 250 W | 133 tok/s | pool 57,669 tok, ppl 8.09 | this README |
| RTX 4090 | 450 W | **135.5 tok/s** | 27B-fork measurement (not Ornith); pool 57,669 and ppl 8.0921; no-spec control 60.3 | [#32](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/32) |

Measured with their own clients rather than the harness — comparable to each
other only loosely, and not rows for the table above:

- **CMP 170HX 40 GB (GA100, sm80)**: 133.7 tok/s median (3x900 tok, greedy) on
  the shipped fast target — the first sm80 datapoint, level with the 3090 —
  and 97.8 tok/s on their own w8a16 int8 target after the sm80 repack
  workaround in [#27](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/27)
  (gotcha 41).
- **RTX 5090 32 GB (sm120)**: ~410-449 tok/s on code and ~198 on prose at
  `CTX=fast`, 500 W cap, roughly flat out to `CTX=huge` at 240k — different
  prompts, output length and rate definition, so deliberately not in the table
  (their own insistence, and correct). Setup gotchas and the full ladder:
  [#35](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/35).
- **RTX 4090, Windows 11 / WSL2 (Docker path)**: reproduces with zero repo
  changes; CTX ladder incl. huge's pool byte-identical to the 3090 reference
  (268,169), concurrency ladder to N=8, and a measured both-ways case for
  leaving the `KV_MEM` pin alone — [docs/wsl2-4090.md](docs/wsl2-4090.md).
- **RTX 4090, Windows 11 / WSL2, second box**: all three single-user profiles
  plus the experimental int4 one (230,830-token pool at 160k), and 135k
  real-task numbers on the MTP + FP8 daily-driver profile — 62 tok/s decode on
  QA over the document, TTFT 5.4 s → 0.33 s on a repeat turn. Also the
  `nvidia-smi dmon` detector for WSL2 host-backed memory now in gotcha 43 —
  [#61](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/61).
- **RTX 3090, Windows 11 / WSL2**: independent confirmation of the int8 prefill
  stack on Ampere — `INT8_ACT=int8` +59%/+57%/+37% at 5k/21k/66k, the int8-QK
  attention adding +1.8% at 21k and +6.3% at 66k on top, against this repo's
  +2.7% at 16k and +5.3% at 51k. Plus the power-limit ladder quoted above —
  [#62](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/62).
- **Dual-GPU reports**: the controlled 1-vs-2×3090 A/B in
  [#40](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/40) (+16–35%,
  161.6 C1 greedy at 275 W, PCIe x8 without NVLink; independently reproduced
  in-thread at 153.6/250 W by a second dual-3090 box), the NVLink dual 3090 in
  [#7](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/7), dual 5060 Ti in
  [#22](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/22). See "More
  than one GPU" above for what transfers.

### Why this isn't just `vllm serve`

Nine things, from requantizing both embedding matrices to drafting straight out
of the prompt — one line each, then the reasoning and measurements, in
[docs/optimizations.md](docs/optimizations.md).

### What each step buys

Measured cumulatively on the 3090, 64 concurrent, 128 in / 512 out, `vllm bench
serve` random dataset:

| step | what it does | e2e output tok/s | steady-state decode |
|---|---|---|---|
| W4A16 AutoRound body (as published) + fp8 KV | int4 Marlin kernels, 66.7k-token pool | 370 (48 conc, 256/256) | — |
| + lm_head / embed_tokens int8 | 2.6 GB of cache pages back | 516 | ~585 (37 requests resident) |
| + fp16 recurrent state | 64 requests resident, half the state traffic | 707 | ~830 |
| + int8 activations, MLP (default) | int8 tensor cores on 74% of the FLOPs | 942 | ~1,094 |
| + int8 activations, everything (`INT8_LAYERS=.`, needs `GPU_UTIL=0.95`) | | 1,042 | ~1,222 |

And single-stream on realistic prompts (single-user mode, T = model default /
greedy):

| step | tok/s | tokens per step | draft acceptance, position 0 |
|---|---|---|---|
| no speculation | 46 / 46 | 1.0 | — |
| MTP-2 as shipped (bf16 drafter, full head, fp32 state) | 66 / 79 | 2.1 / 2.4 | 65% / 80% |
| MTP-4, int8 drafter, 40k draft head, fp16 state | 78 / 99 | 2.2 / 2.7 | 58% / 70% |
| + probabilistic draft sampling (`CTX=fast`, k=4) | 90 / 98 | 2.6 / 2.7 | 69% / 70% |
| same with 3 drafts on FlashInfer/fp8 KV (`CTX=long`, 150k) | 84 / 89 | 2.5 / 2.4 | 69% / 71% |
| + sampler patch, split-KV verify attention | 93 / 99 | 2.6 / 2.6 | 69% / 70% |
| + draft vocab counted over the model's own outputs | 107 / 109 | 2.9 / 2.9 | 74% / 74% |
| + GPTQ-int4 lm_head (calibrated) | 109 / 112 | 2.8 / 2.8 | 73% / 73% |
| + GPTQ-int4 MTP module (**fast variant, shipped**) | **~114 / 118-124** | 2.8 / 2.9-3.0 | 74% / 77% |

Ornith DFlash2 (`SPEC=dflash2`) is **not** in this cumulative table. The 27B fork
rows (118/126 C1, lookup 130–381 tok/s) were for a 64-layer / 5120-d target and
must not be quoted as Ornith-1.5-9B. After `models/Ornith-1.5-9B-DFlash2-W4A16`
exists, fill the cell with `bash bench/dflash2_vs_mtp.sh`. Keep MTP as default
until C1 tok/step and tok/s beat `DRAFT_TOKENS=4`.

(Steps 4-6 are the same 8-prompt protocol; greedy is deterministic for a
given server and request order but differs between configs and even with
prefix-cache hits, so single runs carry ±3-5% on tokens/step —
`bench/run_benchmarks.sh single` reproduces 111.1 / 120.0 tok/s decode at C1,
the best repeats read 119 / 124.)
Going deeper (k=5) loses again: 106 / 105. k=4 is the knee, but on vLLM
0.28.0's FlashInfer backend (needed for fp8 KV, i.e. for 150k context) four
drafts crash the engine with an illegal memory access as soon as one request
finishes while another is mid-generation — club-3090 reports the same "n=4
eventually dies, n=3 stable" pattern — so `CTX=long` drafts 3 and gives up
~7%; `CTX=fast` (FlashAttention, bf16 KV, ~64k context, the default) keeps k=4
and is also the only backend the split-KV attention patch applies to.

Two things that did *not* help, measured rather than assumed: fine-tuning the
MTP head on the model's own outputs (KL halves, greedy top-1 on response
tokens unchanged; `drafter/README.md`), and retuning Marlin's tile
configuration for M ≤ 16 on sm86 (3-7% per GEMM in isolation,
nothing measurable end to end — the remaining gap to peak bandwidth is the
memory system's ramp on 16-92 MB reads, not the kernel).

## Setup

The default install is the container ([Quick start](#quick-start) — the
prebuilt image already contains everything this section builds), so this
manual venv path is for hacking on the stack, or running it bare-metal.

> **Python 3.14 works natively** — nothing in this repo needs changing, but
> `python3.14-dev` does need installing. See [docs/python-314.md](docs/python-314.md),
> with a full RTX 3090 reproduction of the tables below in
> [docs/reproductions/native-3090.md](docs/reproductions/native-3090.md).

You need: a 24 GB Ampere or newer NVIDIA card, a recent driver, Python 3.12,
~40 GB disk — and if the host has less than ~16 GB of free RAM, load the
weights with the streamer instead of the stock loader (gotcha 45: the stock
loader peaks at whatever RAM exists; the streamer is bounded and faster). Everything below is CPU-safe to run while the GPU does other
things; the container details live in [docs/docker.md](docs/docker.md).

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 ~/qwen-serving
cd ~/qwen-serving

python3 -m venv venv
venv/bin/pip install vllm==0.28.0 huggingface_hub hf_transfer ninja \
  flashinfer-python flashinfer-cubin==0.6.13 pandas
# pandas is what `vllm[bench]` pulls in for the custom-dataset path: without it
# bench/prefill_ab.sh's decode guard dies with "Please install vllm[bench] for
# bench support" after the prefill rows have already run.
# flashinfer makes the DFlash2 selector ~2x faster than its torch.topk fallback,
# and vLLM only *uses* it if nvcc is on PATH or flashinfer-cubin is installed --
# a bare `pip install flashinfer-python` silently falls back with one INFO line
# (#35). cubin publishes up to 0.6.13, so the version pair needs
# FLASHINFER_DISABLE_VERSION_CHECK=1, which the launchers export. Do not fix the
# mismatch by downgrading flashinfer-python: that drags torch back and breaks
# vLLM's C extension.

# model, ~9 GB (set HF_TOKEN in .env or shell for faster authenticated downloads)
HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/hf download \
  Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound \
  --local-dir models/Ornith-1.5-9B-MixedInt4-AutoRound
# or use: venv/bin/python prepare/fetch_ornith.py

# requantize lm_head + embeddings to AutoGPTQ int8 (CPU only; writes qweight, not weight_packed)
venv/bin/python prepare/quant_lm_head.py models/Ornith-1.5-9B-MixedInt4-AutoRound
venv/bin/python prepare/quant_embed.py   models/Ornith-1.5-9B-MixedInt4-AutoRound
venv/bin/python prepare/quant_mtp.py     models/Ornith-1.5-9B-MixedInt4-AutoRound
# 40k-token draft head for single-user mode (uses the shipped id list)
venv/bin/python prepare/build_draft_vocab.py models/Ornith-1.5-9B-MixedInt4-AutoRound \
  --ids prepare/draft_vocab_ids.json

# patch vllm (all compatible patches are written against 0.28.0; reapply after upgrades)
for p in patches/*.patch; do
  case "$p" in
    patches/dflash2-backport.patch) echo "skip $p (DFlash2 is native in vLLM 0.28.0)"; continue ;;
  esac
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < "$p"
done
# optional: the KVarN 4/2-bit KV cache for 262k context (docs/long-context.md)
bash kvarn/install.sh

# api key — optional, but the server binds 0.0.0.0 and is open without one
openssl rand -hex 24 > api_key.txt
```

Then `bash verify.sh --no-server` — it checks the venv and vLLM version, that
every compatible patch in `patches/` is actually applied, and that the model has been
requantized (lm_head, embeddings, MTP module, draft head). Then pick a mode
and follow its README:

- **[batch/](batch/)** — throughput. `bash batch/start_ornith.sh`
- **[single-user/](single-user/)** — latency. `bash single-user/start_ornith.sh`

First start takes a couple of minutes (CUDA graph capture, flashinfer JIT). Test it:

```bash
curl http://localhost:18020/v1/chat/completions \
  -H "Authorization: Bearer $(cat api_key.txt 2>/dev/null)" \
  -H "Content-Type: application/json" \
  -d '{"model": "ornith-1.5-9b",
       "messages": [{"role": "user", "content": "Hello! What can you do?"}],
       "chat_template_kwargs": {"enable_thinking": false}}'
```

Ornith recommends temperature 0.7 / top_p 0.8 and top_k 20 for standard instruct, and 1.0 / 0.95
with thinking enabled (the default).

Tool calling works over the same endpoint — send `tools` with `tool_choice:
"auto"` and the reply carries `tool_calls`. Both launchers set
`--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`;
the parser has to read Ornith's XML call format (`<tool_call>...</tool_call>`),
which is what this model's chat template emits. `TOOLS=0` turns it off.

To check the numbers on your own card: `bash verify.sh` (also probes the live
server and prints which attention backend and KV pool it came up with), then
`bash bench/run_benchmarks.sh batch` or `... single` reproduces the tables
above against the running server (`--prefill` and `--long` add the prefill
matrix and the long-context rows), `bash bench/real_rep.sh <tag> 3 0` repeats
the single-stream row, and `python bench/quality_battery.py <tag>` the
perplexity / GSM8K rows. For the concurrency rows,
`python bench/conc_ladder.py --n 1,2,4,8 --ctx-tokens 4096`; for the prompt-length
bug, `python bench/residue_sweep.py <tag>` (all 128 residues) with
`python bench/verbatim.py` as its offline self-test.

## The rest

| | |
|---|---|
| [docs/optimizations.md](docs/optimizations.md) | Every optimization in full: why it was needed, what it measured, which patch implements it. Includes the two speculative-decoding modes (MTP and DFlash2) and the lookup drafter. |
| [docs/gotchas.md](docs/gotchas.md) | 18 things that each cost us hours — read before debugging something that looks like a vLLM bug. |
| [docs/quality.md](docs/quality.md) | IFBench, perplexity and GSM8K per configuration. |
| [docs/docker.md](docs/docker.md) | The container image, and an independent WSL2 reproduction. |
| [docs/long-context.md](docs/long-context.md) | 262k context with the KVarN 4/2-bit KV cache, what vLLM's own per-token-head KV modes are worth here, and how to run the DFlash2 drafter past 64k (`CTX=long`, 114-139k — worth it only for context reproduction). |
| [batch/](batch/) · [single-user/](single-user/) | The two serving modes: full benchmark tables, every env knob, systemd units. |
| [prepare/](prepare/) | The one-time model-preparation scripts run by [Setup](#setup) (and by `docker compose run --rm prepare`). |
| [drafter/](drafter/) | How the draft vocabulary, the int4 drafters and the DFlash2 requantization were built — including what did not work. |
| [kvarn/](kvarn/) | The KVarN 4/2-bit KV cache port. |

## License

Apache-2.0, same as the model.
