#!/usr/bin/env python3
"""
Script to analyze the structure of the Qwen3-1.7B model.
This script prints the structure of the model weights stored in safetensor files,
showing which weights are in each of the two safetensor files.
For repeated layer structures, only one representative layer is printed in detail.
"""

import os
import json
import sys
from pathlib import Path
from safetensors import safe_open
import torch
import numpy as np
from collections import defaultdict

MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(Path(__file__).resolve().parents[2] / "models" / "Qwen3-1.7B")
)
SAFETENSOR_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors"
]

def format_size(size_bytes):
    """Format size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def analyze_model_structure():
    """Analyze and print the structure of the Qwen3-1.7B model."""
    print("=" * 80)
    print(f"QWEN3-1.7B MODEL STRUCTURE ANALYSIS")
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

    # Group weights by safetensor file
    weights_by_file = defaultdict(list)
    for name, file in weight_map.items():
        weights_by_file[file].append(name)

    # Print safetensor files information
    print("\nSAFETENSOR FILES:")
    for file_name in SAFETENSOR_FILES:
        file_path = os.path.join(MODEL_PATH, file_name)
        file_size = os.path.getsize(file_path)
        num_weights = len(weights_by_file[file_name])
        print(f"  {file_name}:")
        print(f"    Size: {format_size(file_size)}")
        print(f"    Number of weights: {num_weights}")

    # Categorize weights by type
    embedding_weights = []
    layer_weights = defaultdict(list)
    final_weights = []

    for name in weight_map.keys():
        if "model.embed_tokens" in name:
            embedding_weights.append(name)
        elif "model.layers." in name:
            layer_parts = name.split("model.layers.")[1].split(".")
            layer_num = int(layer_parts[0])
            component_type = ".".join(layer_parts[1:])
            layer_weights[component_type].append((layer_num, name))
        elif "model.norm" in name or "lm_head" in name:
            final_weights.append(name)

    # Print embedding weights
    print("\nEMBEDDING WEIGHTS:")
    for name in sorted(embedding_weights):
        file_name = weight_map[name]
        file_path = os.path.join(MODEL_PATH, file_name)

        with safe_open(file_path, framework="pt") as f:
            tensor = f.get_tensor(name)
            shape_str = str(tensor.shape)
            dtype_str = str(tensor.dtype)
            size_bytes = tensor.numel() * tensor.element_size()

            print(f"  {name}:")
            print(f"    File: {file_name}")
            print(f"    Shape: {shape_str}")
            print(f"    Dtype: {dtype_str}")
            print(f"    Size: {format_size(size_bytes)}")

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

    # Print layer weights (grouped by component type)
    print("\nLAYER WEIGHTS (by component type):")
    for component_type, weights in sorted(layer_weights.items()):
        print(f"\n  Component: {component_type}")

        # Sort by layer number
        weights.sort()

        # Get a representative weight (from layer 0 if available)
        rep_layer_num = None
        rep_name = None
        for layer_num, name in weights:
            if rep_layer_num is None or layer_num < rep_layer_num:
                rep_layer_num = layer_num
                rep_name = name

        # Print detailed info for the representative weight
        if rep_name:
            file_name = weight_map[rep_name]
            file_path = os.path.join(MODEL_PATH, file_name)

            with safe_open(file_path, framework="pt") as f:
                tensor = f.get_tensor(rep_name)
                shape_str = str(tensor.shape)
                dtype_str = str(tensor.dtype)
                size_bytes = tensor.numel() * tensor.element_size()

                print(f"    Representative weight (Layer {rep_layer_num}):")
                print(f"      {rep_name}")
                print(f"      File: {file_name}")
                print(f"      Shape: {shape_str}")
                print(f"      Dtype: {dtype_str}")
                print(f"      Size: {format_size(size_bytes)}")

                # Print some statistics
                if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    try:
                        tensor_np = tensor.cpu().float().numpy()
                        print(f"      Min: {tensor_np.min():.6f}")
                        print(f"      Max: {tensor_np.max():.6f}")
                        print(f"      Mean: {tensor_np.mean():.6f}")
                        print(f"      Std: {tensor_np.std():.6f}")
                    except Exception as e:
                        print(f"      Error computing statistics: {e}")

        # Print all layer numbers that have this component
        layer_nums = sorted(set(layer_num for layer_num, _ in weights))
        print(f"    Present in layers: {layer_nums}")

        # Print all weights for this component (just name and file)
        print(f"    All weights:")
        for layer_num, name in weights:
            file_name = weight_map[name]
            print(f"      Layer {layer_num}: {name} -> {file_name}")

    # Print final weights
    print("\nFINAL WEIGHTS:")
    for name in sorted(final_weights):
        file_name = weight_map[name]
        file_path = os.path.join(MODEL_PATH, file_name)

        with safe_open(file_path, framework="pt") as f:
            tensor = f.get_tensor(name)
            shape_str = str(tensor.shape)
            dtype_str = str(tensor.dtype)
            size_bytes = tensor.numel() * tensor.element_size()

            print(f"  {name}:")
            print(f"    File: {file_name}")
            print(f"    Shape: {shape_str}")
            print(f"    Dtype: {dtype_str}")
            print(f"    Size: {format_size(size_bytes)}")

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

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    try:
        analyze_model_structure()
    except Exception as e:
        print(f"Error analyzing model: {e}")
        sys.exit(1)
