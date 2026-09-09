"""Install the Ornith-1.5-9B DFlash2 drafter for SPEC=dflash2.

This is NOT the Qwen3.8-27B DFlash2 checkpoint (hidden 5120, taps 5/19/33/47/61).
That checkpoint cannot load against Ornith's 32-layer / 4096-d tower.

  venv/bin/python prepare/fetch_dflash2.py [dst_dir]
  venv/bin/python prepare/fetch_dflash2.py --bf16 [dst_dir]
  DFLASH2_REPO=<hf-id> venv/bin/python prepare/fetch_dflash2.py
  venv/bin/python prepare/fetch_dflash2.py --src /path/to/trained/bf16

Priority: --src, then DFLASH2_REPO / --repo, then an already-local dst dir.
"""
import os, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BF16 = "--bf16" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
src = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else None
repo_flag = sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else None
REPO_ID = repo_flag or os.environ.get("DFLASH2_REPO") or ""
name = "Ornith-1.5-9B-DFlash2" + ("" if BF16 else "-W4A16")
D = args[0] if args else os.path.join(ROOT, "models", name)


def _ok(path):
    cfg = os.path.isfile(os.path.join(path, "config.json"))
    w = (os.path.isfile(os.path.join(path, "model.safetensors"))
         or os.path.isfile(os.path.join(path, "model.safetensors.index.json")))
    return cfg and w


def _check_ornith(path):
    import json
    c = json.load(open(os.path.join(path, "config.json")))
    arch = c.get("architectures") or []
    h = c.get("hidden_size")
    n = c.get("num_target_layers")
    taps = (c.get("dflash_config") or {}).get("target_layer_ids") or []
    if "DFlash2DraftModel" not in arch:
        print("WARNING:", path, "is not architectures=['DFlash2DraftModel']", arch)
    if h and h != 4096:
        sys.exit(f"refusing {path}: hidden_size={h} (Ornith needs 4096; Qwen 27B drafters are 5120)")
    if n and n != 32:
        sys.exit(f"refusing {path}: num_target_layers={n} (Ornith has 32)")
    if taps and max(taps) >= 32:
        sys.exit(f"refusing {path}: target_layer_ids {taps} out of range for 32 layers")


if src:
    src = os.path.abspath(src)
    Dabs = os.path.abspath(D)
    if src != Dabs:
        os.makedirs(D, exist_ok=True)
        for f in os.listdir(src):
            p = os.path.join(src, f)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(D, f))
    _check_ornith(D)
    print("copied drafter", src, "->", D)
elif REPO_ID:
    from huggingface_hub import snapshot_download
    os.makedirs(D, exist_ok=True)
    snapshot_download(REPO_ID, local_dir=D, allow_patterns=["*.json", "*.safetensors", "README.md"])
    _check_ornith(D)
    print("drafter ready:", D)
elif _ok(D):
    _check_ornith(D)
    print("drafter already present:", D)
else:
    print("No Ornith DFlash2 drafter at", D)
    print("Train one (NeMo TrainDFlash2Recipe):")
    print("  bash drafter/train_dflash2.sh          # 2-4 GPU")
    print("  bash drafter/train_dflash2.sh --smoke  # 3090 smoke")
    print("Then quantize:")
    print("  python drafter/capture_dflash2.py")
    print("  python drafter/quant_dflash2.py models/Ornith-1.5-9B-DFlash2 \\")
    print("      models/Ornith-1.5-9B-DFlash2-W4A16 drafter/runs/dflash2/hessians.pt")
    print("Or set DFLASH2_REPO=<hf-id> once a trained checkpoint is published.")
    sys.exit(1)
print("serve with: SPEC=dflash2 bash single-user/start_ornith.sh")
