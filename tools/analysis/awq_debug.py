#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from pathlib import Path
from safetensors import safe_open
import torch

def analyze_mapping(model_path):
    model_path = Path(model_path)
    print(f"Analyzing model weight mappings: {model_path}")

    config_path = model_path / "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            num_layers = config.get("num_hidden_layers", 0)
            print(f"Model layer count: {num_layers}")

    safetensors_path = model_path / "model.safetensors"
    if not os.path.exists(safetensors_path):
        print(f"Error: Weight file not found {safetensors_path}")
        return

    weight_mapping = {
        "self_attn.q_proj": "wq",
        "self_attn.k_proj": "wk",
        "self_attn.v_proj": "wv",
        "self_attn.o_proj": "wo",
        "mlp.gate_proj": "w_gate",
        "mlp.up_proj": "w_up",
        "mlp.down_proj": "w_down"
    }

    with safe_open(safetensors_path, framework="pt") as f:
        all_keys = list(f.keys())

        qweight_keys = [k for k in all_keys if ".qweight" in k]
        scales_keys = [k for k in all_keys if ".scales" in k]
        qzeros_keys = [k for k in all_keys if ".qzeros" in k]

        for layer in range(num_layers):
            print(f"\nlayer {layer} weights:")
            for weight_type, dst_prefix in weight_mapping.items():

                orig_pattern = f"model.layers.{layer}.{weight_type}"

                target_name = f"{dst_prefix}{layer}"

                qweight_found = any(k.startswith(orig_pattern) and k.endswith(".qweight") for k in qweight_keys)
                scales_found = any(k.startswith(orig_pattern) and k.endswith(".scales") for k in scales_keys)
                qzeros_found = any(k.startswith(orig_pattern) and k.endswith(".qzeros") for k in qzeros_keys)

                status = "✓" if (qweight_found and scales_found and qzeros_found) else "✗"

                print(f"  {orig_pattern} -> {target_name}: {status}")
                print(f" qweight: {'Found' if qweight_found else 'Not found'}")
                print(f" scales: {'Found' if scales_found else 'Not found'}")
                print(f" qzeros: {'Found' if qzeros_found else 'Not found'}")

                if qweight_found:
                    qweight_key = next(k for k in qweight_keys if k.startswith(orig_pattern) and k.endswith(".qweight"))
                    tensor = f.get_tensor(qweight_key)
                    print(f"Shape: {tensor.shape}, Type: {tensor.dtype}")

def simulate_processing(model_path):
    model_path = Path(model_path)
    print(f"\nSimulating weight processing: {model_path}")

    weight_keys = []
    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        weight_keys = list(f.keys())

    transformed_weights = {}

    for key in weight_keys:
        if "model.layers." in key:

            layer_parts = key.split(".")
            layer_idx = -1
            for i, part in enumerate(layer_parts):
                if part == "layers" and i+1 < len(layer_parts):
                    layer_idx = int(layer_parts[i+1])
                    break

            if layer_idx >= 0:

                if "self_attn" in key:
                    if "q_proj" in key:
                        if ".qweight" in key:
                            transformed_weights[f"wq{layer_idx}"] = key
                        elif ".scales" in key:
                            transformed_weights[f"wq{layer_idx}.scales"] = key
                        elif ".qzeros" in key:
                            transformed_weights[f"wq{layer_idx}.qzeros"] = key
                    elif "k_proj" in key:
                        if ".qweight" in key:
                            transformed_weights[f"wk{layer_idx}"] = key
                        elif ".scales" in key:
                            transformed_weights[f"wk{layer_idx}.scales"] = key
                        elif ".qzeros" in key:
                            transformed_weights[f"wk{layer_idx}.qzeros"] = key
                    elif "v_proj" in key:
                        if ".qweight" in key:
                            transformed_weights[f"wv{layer_idx}"] = key
                        elif ".scales" in key:
                            transformed_weights[f"wv{layer_idx}.scales"] = key
                        elif ".qzeros" in key:
                            transformed_weights[f"wv{layer_idx}.qzeros"] = key
                    elif "o_proj" in key:
                        if ".qweight" in key:
                            transformed_weights[f"wo{layer_idx}"] = key
                        elif ".scales" in key:
                            transformed_weights[f"wo{layer_idx}.scales"] = key
                        elif ".qzeros" in key:
                            transformed_weights[f"wo{layer_idx}.qzeros"] = key

    print("\no_proj Weight mapping:")
    for i in range(28):
        target_key = f"wo{i}"
        if target_key in transformed_weights:
            print(f"  {target_key} <- {transformed_weights[target_key]}")
        else:
            print(f"{target_key}: Mapping not found")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/analysis/awq_debug.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    analyze_mapping(model_path)
    simulate_processing(model_path)
