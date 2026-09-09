"""Capture GPTQ Hessians for the Ornith DFlash2 drafter's linear layers from the
drafter's OWN inputs on real traffic: vLLM in-process (no engine multiprocessing),
eager mode (CUDA graphs/torch.compile would bypass module hooks), the bf16 drafter
doing speculative decoding on our self-distillation prompts (drafter/data/gen.jsonl).

Widths come from the live modules, not 27B literals (5120 / 17408 / 25600). On
Ornith the draft hidden is 4096, down_proj is 12288, fc is 20480.

Inputs with K <= SMALL_K (default: draft hidden_size) accumulate into fp32 Hessians
on the GPU; wider ones (down_proj, fc) dump bf16 rows to memmaps and reduce after
the engine is torn down.

  MODEL=models/Ornith-1.5-9B-MixedInt4-AutoRound \
  DRAFT=models/Ornith-1.5-9B-DFlash2 \
    python drafter/capture_dflash2.py [--prompts 400] [--max-tokens 384] [--rows 250000]
  -> drafter/runs/dflash2/hessians.pt  {module_key: {"H": fp32 [K,K], "n": rows}}

ctx_kv is captured into hessians_ctx_kv.pt only. Do NOT merge it into k/v Hessians
for the shipped recipe (measured 7% worse greedy acceptance on the 27B stack).
"""
import os, sys, json, time, glob
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
sys.path.insert(0, HERE)
from gptq_utils import accumulate_hessian
from dflash2_const import HIDDEN_SIZE, NUM_HIDDEN_LAYERS

TARGET = os.environ.get("MODEL", os.path.join(REPO, "models", "Ornith-1.5-9B-MixedInt4-AutoRound"))
DRAFT = os.environ.get("DRAFT", os.path.join(REPO, "models", "Ornith-1.5-9B-DFlash2"))
OUT = os.environ.get("OUT", os.path.join(HERE, "runs", "dflash2"))
K_SPEC = int(os.environ.get("K", 7))
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
NPROMPTS, MAXTOK, ROWCAP = arg("--prompts", 400), arg("--max-tokens", 384), arg("--rows", 250000)
HOOKED = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj", "fc")
os.makedirs(OUT, exist_ok=True)


def _draft_hidden():
    cfg = os.path.join(DRAFT, "config.json")
    if os.path.isfile(cfg):
        return int(json.load(open(cfg)).get("hidden_size", HIDDEN_SIZE))
    return HIDDEN_SIZE


def _num_layers():
    cfg = os.path.join(DRAFT, "config.json")
    if os.path.isfile(cfg):
        return int(json.load(open(cfg)).get("num_hidden_layers", NUM_HIDDEN_LAYERS))
    return NUM_HIDDEN_LAYERS


SMALL_K = int(os.environ.get("SMALL_K", _draft_hidden()))


def find_draft_model(root):
    """BFS over attributes for an nn.Module whose class name contains 'DFlash'."""
    seen, queue = set(), [(root, "runner", 0)]
    while queue:
        obj, path, depth = queue.pop(0)
        if id(obj) in seen or depth > 4:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module) and "DFlash" in type(obj).__name__ and hasattr(obj, "model"):
            return obj, path
        if isinstance(obj, torch.nn.Module):
            continue
        d = getattr(obj, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k.startswith("__"):
                continue
            queue.append((v, f"{path}.{k}", depth + 1))
    return None, None


def _module_in_features(mod):
    return (getattr(mod, "input_size", None)
            or getattr(mod, "input_size_per_partition", None)
            or getattr(mod, "in_features", None))


def main():
    from vllm import LLM, SamplingParams
    recs = [json.loads(l) for l in open(os.path.join(HERE, "data", "gen.jsonl"))][:NPROMPTS]
    print(f"{len(recs)} prompts, max_tokens {MAXTOK}, k={K_SPEC}, SMALL_K={SMALL_K}", flush=True)

    llm = LLM(
        model=TARGET, served_model_name="ornith-1.5-9b",
        speculative_config={"method": "dflash", "model": DRAFT, "num_speculative_tokens": K_SPEC,
                            "draft_sample_method": "probabilistic"},
        enforce_eager=True,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.95)),
        max_model_len=int(os.environ.get("MAX_LEN", 4096)), max_num_seqs=int(os.environ.get("MAX_SEQS", 8)),
        max_num_batched_tokens=2048, attention_backend="FLASH_ATTN", kv_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float16", language_model_only=True, enable_prefix_caching=False,
    )
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    runner = core.model_executor.driver_worker.worker.model_runner
    draft, path = find_draft_model(runner)
    assert draft is not None, "could not find the DFlash draft model under the model runner"
    print("draft model at", path, type(draft).__name__, flush=True)

    stats = {}
    handles = []
    n_layers = _num_layers()
    for name, mod in draft.named_modules():
        key = next((h for h in HOOKED if name.endswith(h)), None)
        if key is None or not isinstance(mod, torch.nn.Module) or not hasattr(mod, "forward"):
            continue
        if key == "fc" and not (name == "fc" or name.endswith(".fc")):
            continue
        mkey = name[len("model."):] if name.startswith("model.") else name
        K = _module_in_features(mod)
        if K is None:
            continue
        if K <= SMALL_K:
            stats[mkey] = {"H": torch.zeros(K, K, device="cuda", dtype=torch.float32), "n": 0, "K": K}
        else:
            mm = np.lib.format.open_memmap(os.path.join(OUT, f"rows_{mkey}.npy"), mode="w+",
                                           dtype=np.uint16, shape=(ROWCAP, K))
            stats[mkey] = {"mm": mm, "n": 0, "K": K}

        def pre_hook(m, args, mkey=mkey):
            x = args[0]
            X = x.reshape(-1, x.shape[-1])
            st = stats[mkey]
            if "H" in st:
                st["H"], st["n"] = accumulate_hessian(st["H"], X, st["n"])
            else:
                n = st["n"]
                if n < ROWCAP:
                    take = min(X.shape[0], ROWCAP - n)
                    st["mm"][n:n + take] = X[:take].to(torch.bfloat16).cpu().view(torch.uint16).numpy()
                    st["n"] = n + take
        handles.append(mod.register_forward_pre_hook(pre_hook))

    # Context-KV precompute bypasses qkv_proj.forward. Captured separately and NOT
    # blended into k/v Hessians by default (quant_dflash2.py).
    from vllm import _custom_ops as ops
    dm = draft.model
    hid = int(getattr(dm.fc, "output_size", None) or getattr(dm.fc, "out_features", None) or SMALL_K)
    stats["ctx_kv"] = {"H": torch.zeros(hid, hid, device="cuda", dtype=torch.float32), "n": 0, "K": hid}
    orig_pck = dm._project_context_kv

    def _project_context_kv(context_states, *a, **kw):
        normed = torch.empty_like(context_states)
        ops.rms_norm(normed, context_states, dm._hidden_norm_weight, dm._rms_norm_eps)
        st = stats["ctx_kv"]
        st["H"], st["n"] = accumulate_hessian(st["H"], normed.reshape(-1, normed.shape[-1]), st["n"])
        return orig_pck(context_states, *a, **kw)
    dm._project_context_kv = _project_context_kv
    print(f"hooked {len(stats)} modules: {sorted(stats)}", flush=True)
    expect = 4 * n_layers + 2  # qkv, o, gate_up, down per layer + fc + ctx_kv
    assert len(stats) == expect, f"expected {expect} hooked modules for {n_layers} layers, got {len(stats)}"

    sp = SamplingParams(max_tokens=MAXTOK, temperature=1.0, top_p=0.95, top_k=20)
    t0 = time.time()
    CH = 32
    qkv0 = "layers.0.self_attn.qkv_proj"
    down0 = "layers.0.mlp.down_proj"
    for i in range(0, len(recs), CH):
        batch = recs[i:i + CH]
        outs = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in batch], sp, use_tqdm=False)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        n_small = stats.get(qkv0, {}).get("n", 0)
        n_wide = stats.get(down0, {}).get("n", 0)
        print(f"[{i + len(batch)}/{len(recs)}] {ntok} new tokens; rows small={n_small} wide={n_wide} "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    for h in handles:
        h.remove()
    dm._project_context_kv = orig_pck
    res, ctx = {}, None
    for k, st in stats.items():
        if k == "ctx_kv":
            ctx = {"H": st["H"].cpu(), "n": st["n"]}
            continue
        if "H" in st:
            res[k] = {"H": st["H"].cpu(), "n": st["n"]}
        else:
            st["mm"].flush()
    torch.save(res, os.path.join(OUT, "hessians_small.pt"))
    if ctx is not None:
        torch.save({"ctx_kv": ctx}, os.path.join(OUT, "hessians_ctx_kv.pt"))
    json.dump({k: {"n": st["n"], "K": st["K"]} for k, st in stats.items() if "H" not in st},
              open(os.path.join(OUT, "rows_meta.json"), "w"))
    print("capture done; re-executing for the Hessian reduction", flush=True)
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), "--reduce-only"])


def reduce_wide():
    """Phase 2: X^T X for the dumped wide inputs on the (now free) GPU."""
    res = torch.load(os.path.join(OUT, "hessians_small.pt"))
    meta = json.load(open(os.path.join(OUT, "rows_meta.json")))
    for k, m in meta.items():
        n, K = m["n"], m["K"]
        mm = np.load(os.path.join(OUT, f"rows_{k}.npy"), mmap_mode="r")
        H = torch.zeros(K, K, device="cuda", dtype=torch.float32); seen = 0
        for a in range(0, n, 8192):
            X = torch.from_numpy(np.array(mm[a:min(n, a + 8192)])).view(torch.bfloat16).cuda()
            H, seen = accumulate_hessian(H, X, seen)
        res[k] = {"H": H.cpu(), "n": seen}
        del H; torch.cuda.empty_cache()
        print(f"reduced {k}: {seen} rows", flush=True)
    torch.save(res, os.path.join(OUT, "hessians.pt"))
    for f in glob.glob(os.path.join(OUT, "rows_*.npy")):
        os.remove(f)
    print("saved", os.path.join(OUT, "hessians.pt"), {k: v["n"] for k, v in res.items()})
    print("ctx_kv (not blended) at", os.path.join(OUT, "hessians_ctx_kv.pt"))


if __name__ == "__main__":
    if "--reduce-only" in sys.argv:
        reduce_wide()
    else:
        main()
