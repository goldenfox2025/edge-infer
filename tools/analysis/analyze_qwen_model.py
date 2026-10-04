#!/usr/bin/env python3
"""
Script to analyze the structure of the Qwen3-1.7B model.
This script prints the structure of the model weights stored in safetensor files.
"""

import os
import json
import sys
from pathlib import Path
from safetensors import safe_open
import torch
import numpy as np

MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(Path(__file__).resolve().parents[2] / "models" / "Qwen3-1.7B")
)
SAFETENSOR_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors"
]

def print_tensor_info(name, tensor):
    """Print information about a tensor."""
    shape_str = str(tensor.shape)
    dtype_str = str(tensor.dtype)
    size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)

    print(f"  {name}:")
    print(f"    Shape: {shape_str}")
    print(f"    Dtype: {dtype_str}")
    print(f"    Size: {size_mb:.2f} MB")

    # Print some statistics
    if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        try:
            tensor_np = tensor.cpu().float().numpy()
            print(f"    Min: {tensor_np.min():.6f}")
            print(f"    Max: {tensor_np.max():.6f}")
            print(f"    Mean: {tensor_np.mean():.6f}")
            print(f"    Std: {tensor_np.std():.6f}")
        except Exception as e:
            print(f"    Error computing statistics: {e}")
    print()

def analyze_model_structure():
    """Analyze and print the structure of the Qwen3-1.7B model."""
    print("=" * 80)
    print(f"ANALYZING QWEN3-1.7B MODEL AT: {MODEL_PATH}")
    print("=" * 80)

    # Load the model index to understand the structure
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]

    # Load the model configuration
    config_path = os.path.join(MODEL_PATH, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("\nMODEL CONFIGURATION:")
    print(f"Architecture: {config['architectures'][0]}")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Intermediate Size: {config['intermediate_size']}")
    print(f"Number of Attention Heads: {config['num_attention_heads']}")
    print(f"Number of Key-Value Heads: {config['num_key_value_heads']}")
    print(f"Number of Hidden Layers: {config['num_hidden_layers']}")
    print(f"Vocabulary Size: {config['vocab_size']}")
    print(f"Max Position Embeddings: {config['max_position_embeddings']}")
    print(f"Data Type: {config['torch_dtype']}")
    print("=" * 80)

    # Group weights by component
    components = {}
    for name in weight_map:
        parts = name.split('.')
        if len(parts) > 1:
            component = parts[0]
            if component not in components:
                components[component] = []
            components[component].append(name)

    # Print the model structure
    print("\nMODEL STRUCTURE:")

    # First, print the embedding layer
    print("\n--- EMBEDDING LAYER ---")
    embedding_key = "model.embed_tokens.weight"
    file_name = weight_map[embedding_key]
    file_path = os.path.join(MODEL_PATH, file_name)

    with safe_open(file_path, framework="pt") as f:
        tensor = f.get_tensor(embedding_key)
        print_tensor_info(embedding_key, tensor)

    # Print one representative layer (layer 0)
    print("\n--- REPRESENTATIVE LAYER (Layer 0) ---")
    layer_keys = [k for k in weight_map.keys() if "model.layers.0." in k]
    layer_keys.sort()

    for key in layer_keys:
        file_name = weight_map[key]
        file_path = os.path.join(MODEL_PATH, file_name)

        with safe_open(file_path, framework="pt") as f:
            tensor = f.get_tensor(key)
            print_tensor_info(key, tensor)

    # Print the final normalization layer
    print("\n--- FINAL NORMALIZATION LAYER ---")
    norm_key = "model.norm.weight"
    file_name = weight_map[norm_key]
    file_path = os.path.join(MODEL_PATH, file_name)

    with safe_open(file_path, framework="pt") as f:
        tensor = f.get_tensor(norm_key)
        print_tensor_info(norm_key, tensor)

    # Print the language model head
    print("\n--- LANGUAGE MODEL HEAD ---")
    lm_head_key = "lm_head.weight"
    file_name = weight_map[lm_head_key]
    file_path = os.path.join(MODEL_PATH, file_name)

    with safe_open(file_path, framework="pt") as f:
        tensor = f.get_tensor(lm_head_key)
        print_tensor_info(lm_head_key, tensor)

    # Print summary of all layers
    print("\n--- SUMMARY OF ALL LAYERS ---")
    num_layers = config['num_hidden_layers']
    print(f"Total number of layers: {num_layers}")

    # Print the first and last few layers to show the pattern
    layers_to_print = [0, 1, 2, num_layers-3, num_layers-2, num_layers-1]

    for layer_idx in layers_to_print:
        print(f"\nLayer {layer_idx}:")
        layer_prefix = f"model.layers.{layer_idx}."
        layer_keys = [k for k in weight_map.keys() if k.startswith(layer_prefix)]

        # Just print the names and shapes, not the full tensor info
        for key in sorted(layer_keys):
            file_name = weight_map[key]
            file_path = os.path.join(MODEL_PATH, file_name)

            with safe_open(file_path, framework="pt") as f:
                tensor = f.get_tensor(key)
                print(f"  {key}: {tensor.shape}, {tensor.dtype}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    try:
        analyze_model_structure()
    except Exception as e:
        print(f"Error analyzing model: {e}")
        sys.exit(1)
