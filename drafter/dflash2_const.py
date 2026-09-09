"""Ornith-1.5-9B DFlash2 geometry. Shared by init, smoke, convert, and the NeMo yaml comments.

The Qwen3.8-27B drafter (hidden 5120, taps 5/19/33/47/61, mask 248070) cannot load against
this 32-layer / 4096-d hybrid target. 248070 is also <|audio_start|> in the Ornith tokenizer;
the mask id must be a reserved unused slot, not pad/eos/audio.
"""
from __future__ import annotations

import json
import os

# Teacher (Ornith-1.5-9B text tower)
NUM_TARGET_LAYERS = 32
HIDDEN_SIZE = 4096
VOCAB_SIZE = 248320
INTERMEDIATE_SIZE = 12288  # 9B FFN; smaller draft than the 27B's 17408
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10_000_000.0
MAX_POSITION_EMBEDDINGS = 262144

# Paper: uniform from 2nd layer to 3rd-to-last of a 32-layer stack (0-indexed).
TARGET_LAYER_IDS = [1, 8, 15, 22, 29]
# Ablation if GDN aux-hidden hooks are empty / weak: full-attention layers only.
TARGET_LAYER_IDS_FULL_ATTN = [3, 11, 19, 27, 31]

# Unused reserved id (tokenizer added_tokens stop at 248076 <|audio_pad|>).
# Do not use 248044 pad, 248046 eos, or 248070 <|audio_start|>.
MASK_TOKEN_ID = 248077
EOS_TOKEN_ID = 248044
PAD_TOKEN_ID = 248044

# Slim Qwen3 draft stack (not Qwen3.5 hybrid). Matches published DFlash2 layout, rescaled.
NUM_HIDDEN_LAYERS = 5
BLOCK_SIZE = 8  # 1 anchor + 7 mask tokens
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128  # 32 * 128 = 4096
SLIDING_WINDOW = 2048
CONV_KERNEL_SIZE = 2
CONV_GROUP_SIZE = 16
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16

ARCH = "DFlash2DraftModel"
MODEL_TYPE = "qwen3"

FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


def fc_in_features(n_taps: int | None = None) -> int:
    n = len(TARGET_LAYER_IDS) if n_taps is None else n_taps
    return n * HIDDEN_SIZE


def conv_num_groups() -> int:
    assert HIDDEN_SIZE % CONV_GROUP_SIZE == 0
    return HIDDEN_SIZE // CONV_GROUP_SIZE


def kernel_projection_out() -> int:
    # 2 sides * taps * groups
    return 2 * CONV_KERNEL_SIZE * conv_num_groups()


def layer_kind(idx: int) -> str:
    return "full_attention" if idx in FULL_ATTENTION_LAYERS else "linear_attention"


def draft_config_dict(target_layer_ids=None) -> dict:
    ids = list(target_layer_ids or TARGET_LAYER_IDS)
    return {
        "architectures": [ARCH],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": None,
        "is_causal": False,
        "dflash_config": {
            "block_size": BLOCK_SIZE,
            "conv_group_size": CONV_GROUP_SIZE,
            "conv_kernel_size": CONV_KERNEL_SIZE,
            "mask_token_id": MASK_TOKEN_ID,
            "selector_rank": SELECTOR_RANK,
            "selector_top_k": SELECTOR_TOP_K,
            "target_layer_ids": ids,
        },
        "dtype": "bfloat16",
        "eos_token_id": EOS_TOKEN_ID,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN_SIZE,
        "initializer_range": 0.02,
        "intermediate_size": INTERMEDIATE_SIZE,
        "layer_types": ["sliding_attention"] * NUM_HIDDEN_LAYERS,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "max_window_layers": NUM_HIDDEN_LAYERS,
        "model_type": MODEL_TYPE,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "num_target_layers": NUM_TARGET_LAYERS,
        "pad_token_id": PAD_TOKEN_ID,
        "rms_norm_eps": RMS_NORM_EPS,
        "rope_parameters": {"rope_theta": ROPE_THETA, "rope_type": "default"},
        "sliding_window": SLIDING_WINDOW,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": True,
        "vocab_size": VOCAB_SIZE,
    }


def write_config(dst_dir: str, target_layer_ids=None) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, "config.json")
    with open(path, "w") as f:
        json.dump(draft_config_dict(target_layer_ids), f, indent=2)
        f.write("\n")
    return path


def teacher_config(model_dir: str) -> dict:
    return json.load(open(os.path.join(model_dir, "config.json")))


def assert_mask_unused(model_dir: str) -> None:
    """248077 must stay a reserved unused id (not pad/eos/audio, not an added_token)."""
    tok = os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tok):
        return
    data = json.load(open(tok))
    used = {int(t["id"]) for t in (data.get("added_tokens") or [])}
    if MASK_TOKEN_ID in used:
        raise SystemExit(
            f"mask_token_id {MASK_TOKEN_ID} is already an added token "
            f"({next(t.get('content') for t in data['added_tokens'] if int(t['id'])==MASK_TOKEN_ID)})"
        )
    for bad, why in ((248044, "pad/eos"), (248046, "im_end"), (248070, "audio_start")):
        if MASK_TOKEN_ID == bad:
            raise SystemExit(f"mask_token_id {MASK_TOKEN_ID} collides with {why}")
    if MASK_TOKEN_ID >= VOCAB_SIZE or MASK_TOKEN_ID < 0:
        raise SystemExit(f"mask_token_id {MASK_TOKEN_ID} outside vocab {VOCAB_SIZE}")


def assert_teacher_compatible(model_dir: str, target_layer_ids=None) -> None:
    cfg = teacher_config(model_dir)
    tc = cfg.get("text_config", cfg)
    h = int(tc["hidden_size"])
    n = int(tc["num_hidden_layers"])
    v = int(tc.get("vocab_size", VOCAB_SIZE))
    if h != HIDDEN_SIZE:
        raise SystemExit(f"teacher hidden_size {h} != {HIDDEN_SIZE}")
    if n != NUM_TARGET_LAYERS:
        raise SystemExit(f"teacher num_hidden_layers {n} != {NUM_TARGET_LAYERS}")
    if v != VOCAB_SIZE:
        raise SystemExit(f"teacher vocab_size {v} != {VOCAB_SIZE}")
    ids = list(target_layer_ids or TARGET_LAYER_IDS)
    bad = [i for i in ids if i < 0 or i >= n]
    if bad:
        raise SystemExit(f"target_layer_ids out of range for {n} layers: {bad}")
    assert_mask_unused(model_dir)
