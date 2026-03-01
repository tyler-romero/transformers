#!/usr/bin/env python3
"""Convert an OLMo-3.2-Hybrid HuggingFace checkpoint to match the transformers main branch.

The upstream checkpoint (e.g. allenai/OLMo-3.2-Hybrid-7B-Instruct-SFT) uses
different naming conventions than the code on transformers `main`. This script
patches the config.json and renames state dict keys so the checkpoint loads
cleanly without any code changes.

Changes applied:
  config.json:
    - model_type:    olmo3_2_hybrid  -> olmo_hybrid
    - architectures: [Olmo3_2HybridForCausalLM] -> [OlmoHybridForCausalLM]
    - removes:       linear_use_gate, rope_theta  (not in OlmoHybridConfig)

  state_dict (linear attention layers only):
    - attention_layer_norm   -> input_layernorm
    - feedforward_layer_norm -> post_attention_layernorm

Usage:
    python convert_olmo_hybrid_hf_to_main.py \
        --input_dir allenai/OLMo-3.2-Hybrid-7B-Instruct-SFT \
        --output_dir /path/to/converted

    # Or from a local directory:
    python convert_olmo_hybrid_hf_to_main.py \
        --input_dir /path/to/original \
        --output_dir /path/to/converted
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


# Config keys that don't exist in the OlmoHybridConfig on main
UNSUPPORTED_CONFIG_KEYS = {"linear_use_gate", "rope_theta"}

# State dict renames for linear attention layers
LINEAR_NORM_RENAMES = {
    "attention_layer_norm": "input_layernorm",
    "feedforward_layer_norm": "post_attention_layernorm",
}


def convert_config(input_path: str, output_path: str) -> list[str]:
    """Patch config.json and return the layer_types list."""
    with open(input_path) as f:
        config = json.load(f)

    config["model_type"] = "olmo_hybrid"
    config["architectures"] = ["OlmoHybridForCausalLM"]

    for key in UNSUPPORTED_CONFIG_KEYS:
        config.pop(key, None)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return config["layer_types"]


def convert_state_dict(state_dict: dict, layer_types: list[str]) -> tuple[dict, list[str]]:
    """Rename norm keys in linear attention layers. Returns (new_dict, list of renames)."""
    linear_indices = {i for i, t in enumerate(layer_types) if t == "linear_attention"}
    new_state_dict = {}
    renames = []

    for key, value in state_dict.items():
        new_key = key
        for old_name, new_name in LINEAR_NORM_RENAMES.items():
            if old_name not in key:
                continue
            # Extract layer index from "model.layers.{idx}.{old_name}.weight"
            parts = key.split(".")
            try:
                layer_idx = int(parts[2])
            except (IndexError, ValueError):
                break
            if layer_idx in linear_indices:
                new_key = key.replace(old_name, new_name)
                renames.append(f"  {key} -> {new_key}")
            break

        new_state_dict[new_key] = value

    return new_state_dict, renames


def main():
    parser = argparse.ArgumentParser(
        description="Convert OLMo-3.2-Hybrid HF checkpoint to transformers main format"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="HuggingFace repo ID or local directory with the original checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the converted checkpoint",
    )
    args = parser.parse_args()

    # Resolve input
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Downloading {input_dir} from HuggingFace Hub...")
        input_dir = snapshot_download(input_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Convert config
    print("Converting config.json...")
    layer_types = convert_config(
        os.path.join(input_dir, "config.json"),
        os.path.join(args.output_dir, "config.json"),
    )

    # 2. Convert weights
    weights_path = os.path.join(input_dir, "model.safetensors")
    index_path = os.path.join(input_dir, "model.safetensors.index.json")

    if os.path.exists(weights_path):
        print("Loading model.safetensors...")
        state_dict = load_file(weights_path)
    elif os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state_dict = {}
        for shard in shard_files:
            print(f"Loading {shard}...")
            state_dict.update(load_file(os.path.join(input_dir, shard)))
    else:
        raise FileNotFoundError(f"No safetensors weights found in {input_dir}")

    print("Renaming state dict keys...")
    new_state_dict, renames = convert_state_dict(state_dict, layer_types)

    if renames:
        print(f"Renamed {len(renames)} keys:")
        for r in sorted(renames):
            print(r)

    print("Saving model.safetensors...")
    save_file(new_state_dict, os.path.join(args.output_dir, "model.safetensors"))

    # 3. Copy remaining files (tokenizer, etc.)
    skip = {"config.json", "model.safetensors", "model.safetensors.index.json"}
    for fname in os.listdir(input_dir):
        if fname in skip or fname.startswith("model-0"):
            continue
        src = os.path.join(input_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Copied {fname}")

    print(f"\nDone! Converted checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
