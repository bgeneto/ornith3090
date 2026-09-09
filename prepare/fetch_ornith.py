"""Fetch Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound checkpoint.

Usage:
  venv/bin/python prepare/fetch_ornith.py [hf-repo] [dst_dir]
  # default: Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound -> models/Ornith-1.5-9B-MixedInt4-AutoRound

Downloads ~8.8 GB of weights (2 base shards + extra tensors + tokenizer/config files).
After download, requantize the untied heads to AutoGPTQ int8 (qweight, not weight_packed):
  venv/bin/python prepare/quant_lm_head.py models/Ornith-1.5-9B-MixedInt4-AutoRound
  venv/bin/python prepare/quant_embed.py   models/Ornith-1.5-9B-MixedInt4-AutoRound
  venv/bin/python prepare/build_draft_vocab.py models/Ornith-1.5-9B-MixedInt4-AutoRound --ids prepare/draft_vocab_ids.json
"""
import os
import sys
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_REPO = "Pilcothink/Ornith-1.5-9B-MixedInt4-AutoRound"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
REPO = args[0] if args else DEFAULT_REPO
default_dst = (
    "Ornith-1.5-9B-MixedInt4-AutoRound"
    if REPO == DEFAULT_REPO
    else REPO.split("/")[-1]
)
D = (args[1] if len(args) > 1 else os.path.join(ROOT, "models", default_dst)).rstrip("/")
os.makedirs(D, exist_ok=True)

# HF authentication token from environment or .env
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    env_file = os.path.join(ROOT, ".env")
    if os.path.isfile(env_file):
        for line in open(env_file):
            line = line.strip()
            if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    token = val
                break
token = token.strip() if token else None
if token:
    print("Using HF_TOKEN for authenticated high-speed download.")
else:
    print("HF_TOKEN not detected (unauthenticated download; set HF_TOKEN in .env for faster downloads).")

# Enable hf_transfer for high-speed parallel chunk downloads
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1") != "0":
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("Fast download backend enabled: hf_transfer (Rust parallel downloader).")
    except ImportError:
        pass

print(f"Downloading {REPO} to {D}...")
snapshot_download(
    REPO,
    local_dir=D,
    token=token,
    allow_patterns=[
        "*.json",
        "*.jinja",
        "*.txt",
        "*.safetensors",
        "README.md",
    ],
)

rel = os.path.relpath(D, ROOT)
print(f"\nCheckpoint downloaded: {rel}")
print(
    "Next steps to optimize Ornith-1.5-9B for RTX 3090:\n"
    f"  venv/bin/python prepare/quant_lm_head.py {rel}\n"
    f"  venv/bin/python prepare/quant_embed.py {rel}\n"
    f"  venv/bin/python prepare/build_draft_vocab.py {rel} --ids prepare/draft_vocab_ids.json\n"
    f"Then launch:\n"
    f"  bash single-user/start_ornith.sh\n"
)
