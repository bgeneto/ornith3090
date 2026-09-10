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
| decode (Ornith-1.5-9B) | throughput-oriented, no speculation | **~201 tok/s** single-stream after 2k with `KV=int8pth` + INT8 activations (**~243 tok/s** aggregate / **~166 tok/s** per request at 2 streams); **~220 tok/s** prose / **300+ tok/s** coding on bf16 KV |
| memory footprint | ~8.5 GB weights + FP16 GDN recurrent state | ~8.5 GB weights, leaving ~15.5 GB VRAM for extensive KV cache and CUDA graphs |
| trick | 16-bit recurrent state + INT8 tensor-core GEMMs | Native MTP speculation ($k=3/4$), INT8 `lm_head` + `embed_tokens`, 40k-token draft vocabulary, split-KV verify attention, sort-free sampler, hybrid prefix caching |

Both modes share one install — the mode is just which launch script you run.
Speculation is the win for one or a few concurrent users; plain batching is the
many-request path. Numbers are on an RTX 3090.

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

Default single-user remains `SPEC=mtp` until single-stream benches beat MTP k=4. On a 9B target the drafter is a larger fraction of step time than on 27B; chat decode is a measurement, copy/quote (`LOOKUP=1`, `DFLASH_TOKENS=15`) is the likely win. Compare with `bash bench/dflash2_vs_mtp.sh`.

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
  property.** On one card a 5-layer block drafter can collapse at 8 concurrent
  requests (recurrent-state pool exhaustion) while MTP still fits. On two
  cards the extra pool usually flips that. Measure on Ornith before pointing
  everyone at DFlash2.
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

Measured here on **Ornith-1.5-9B**, one RTX 3090, single-user profile. `ppN` is
prompt processing of N tokens; `tg128` is 128 generated tokens after that prompt
(the decode rate you feel once the KV is warm). Mean ± spread across repeats.
Quality in [docs/quality.md](docs/quality.md).

C1–C8 cohort tables from the Qwen3.8-27B fork this repo was derived from do **not**
apply to this 9B model and are not reproduced here.

Config unless noted: `INT8_ACT=int8`, `PREFILL_ATTN=int8`, `KV=int8pth` (Triton
int8 per-token-head KV). TTFR / e2e TTFT are time-to-first-token including engine
overhead; est. PPT is the prompt-processing portion of that.

### Single stream

| model | test | tok/s | peak tok/s | TTFR (ms) | est. PPT (ms) | e2e TTFT (ms) |
|---|---|---:|---:|---:|---:|---:|
| Ornith-1.5-9B | pp2048 | 4033.72 ± 72.79 | — | 520.36 ± 21.80 | 465.43 ± 21.80 | 520.36 ± 21.80 |
| Ornith-1.5-9B | tg128 (after 2,048) | 201.38 ± 9.03 | 202.97 ± 9.08 | — | — | — |
| Ornith-1.5-9B | pp8192 | 3300.44 ± 85.49 | — | 2328.44 ± 57.26 | 2273.52 ± 57.26 | 2328.44 ± 57.26 |
| Ornith-1.5-9B | tg128 (after 8,192) | 157.84 ± 1.73 | 159.10 ± 1.75 | — | — | — |

Drop `KV=int8pth` and the cache stays bf16 (`CTX=fast`): about **220 tok/s** on
creative / prose, and **300+ tok/s** on deterministic / coding. Int8 KV is the
FlashInfer-free longer-context path, not a decode-speed win — generation falls as
the cached prompt grows (~201 tok/s after 2k, ~158 tok/s after 8k).

### Two concurrent streams

Same knobs, two requests in flight. `tok/s (total)` is aggregate decode across both
streams; `tok/s (req)` is per request. Prefill is a shared resource, so
time-to-first-token is roughly 2× the single-stream figure. Aggregate decode rises
(~243 tok/s after 2k) while per-request decode falls (~166 tok/s).

| model | test | tok/s (total) | tok/s (req) | peak tok/s | peak tok/s (req) | TTFR (ms) | est. PPT (ms) | e2e TTFT (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ornith-1.5-9B | pp2048 (c2) | 3205.51 ± 219.04 | 2182.39 ± 644.57 | — | — | 985.01 ± 230.29 | 928.67 ± 230.29 | 985.01 ± 230.29 |
| Ornith-1.5-9B | tg128 (c2, after 2,048) | 242.59 ± 41.44 | 166.27 ± 40.84 | 266.77 ± 29.14 | 173.33 ± 34.38 | — | — | — |
| Ornith-1.5-9B | pp8192 (c2) | 2386.21 ± 179.53 | 1229.76 ± 97.01 | — | — | 6117.31 ± 515.00 | 6060.97 ± 515.00 | 6117.31 ± 515.00 |
| Ornith-1.5-9B | tg128 (c2, after 8,192) | 193.46 ± 22.61 | 118.63 ± 21.65 | 245.67 ± 23.84 | 132.74 ± 15.95 | — | — | — |

### Quality

The whole stack is quantized, so the honest question is what it costs. Short
version: **IFBench 78.3** prompt-level strict vs 79.5 for the unquantized model
(one point), **perplexity 8.09** on 33k held-out tokens, **GSM8K 96.5%** (200
questions, greedy). Speculative decoding — MTP, DFlash2 and the lookup drafter —
is exact by construction and changes none of it; the int8-activation steps in
batch mode are the only knobs that trade accuracy for speed, and they cost
0.9-3.7% perplexity depending on how far you push them. Per-configuration
tables: [docs/quality.md](docs/quality.md).

### Why this isn't just `vllm serve`

Nine things, from requantizing both embedding matrices to drafting straight out
of the prompt — one line each, then the reasoning and measurements, in
[docs/optimizations.md](docs/optimizations.md).

On vLLM 0.28.0's FlashInfer backend (needed for fp8 KV, i.e. for 150k context)
four MTP drafts crash the engine with an illegal memory access as soon as one
request finishes while another is mid-generation — club-3090 reports the same
"n=4 eventually dies, n=3 stable" pattern — so `CTX=long` drafts 3; `CTX=fast`
(FlashAttention, bf16 KV, ~64k context, the default) keeps k=4 and is also the
only backend the split-KV attention patch applies to. Keep `SPEC=mtp` as the
default until a trained Ornith DFlash2 checkpoint beats it on this 9B target
(`bash bench/dflash2_vs_mtp.sh`).

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

Ornith was trained thinking-off. The launchers default `ENABLE_THINKING=0`
(`--default-chat-template-kwargs={"enable_thinking":false}` and greedy
`0.0/0.80/20`). Set `ENABLE_THINKING=1` for thinking-on (`1.0/0.95/20`).
Clients can still send `chat_template_kwargs` per request.

Tool calling works over the same endpoint — send `tools` with `tool_choice:
"auto"` and the reply carries `tool_calls`. Both launchers set
`--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`;
the parser has to read Ornith's XML call format (`<tool_call>...</tool_call>`),
which is what this model's chat template emits. `TOOLS=0` turns it off.

To check the numbers on your own card: `bash verify.sh` (also probes the live
server and prints which attention backend and KV pool it came up with). The
Ornith-1.5-9B figures in [Benchmarks](#benchmarks) are the ones to compare against.
`bash bench/run_benchmarks.sh batch` or `... single`, `bash bench/real_rep.sh
<tag> 3 0`, and `python bench/quality_battery.py <tag>` still exist for local
harness runs; `python bench/conc_ladder.py --n 1,2,4,8 --ctx-tokens 4096` for
concurrency; `python bench/residue_sweep.py <tag>` (all 128 residues) with
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
