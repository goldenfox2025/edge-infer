#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from pathlib import Path
from safetensors import safe_open

def analyze_model(model_path):
    model_path = Path(model_path)
    print(f"Analyzing model: {model_path}")

    config_path = model_path / "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            print("Model configuration:")
            for key, value in config.items():
                if isinstance(value, dict):
                    print(f"  {key}: {{{len(value)} items}}")
                elif isinstance(value, list):
                    print(f"  {key}: [{len(value)} items]")
                else:
                    print(f"  {key}: {value}")

    safetensors_path = model_path / "model.safetensors"
    weight_keys = []

    if os.path.exists(safetensors_path):
        print(f"\nWeight file: {safetensors_path}")

        with safe_open(safetensors_path, framework="pt") as f:

            all_keys = f.keys()
            weight_keys = list(all_keys)

            print(f"Total weights: {len(weight_keys)}")

            attention_keys = [k for k in weight_keys if "self_attn" in k]
            mlp_keys = [k for k in weight_keys if "mlp" in k]
            embedding_keys = [k for k in weight_keys if "embed" in k]
            layernorm_keys = [k for k in weight_keys if "layernorm" in k]

            print(f"Attention weight count: {len(attention_keys)}")
            print(f"MLP Weight count: {len(mlp_keys)}")
            print(f"Embedding weight count: {len(embedding_keys)}")
            print(f"Layer-normalization weight count: {len(layernorm_keys)}")

            qweight_keys = [k for k in weight_keys if ".qweight" in k]
            scales_keys = [k for k in weight_keys if ".scales" in k]
            qzeros_keys = [k for k in weight_keys if ".qzeros" in k]

            print(f"\nquantized weights (.qweight): {len(qweight_keys)}")
            print(f"Quantization scales (.scales): {len(scales_keys)}")
            print(f"Quantization zero points (.qzeros): {len(qzeros_keys)}")

            o_proj_keys = [k for k in weight_keys if "self_attn.o_proj" in k]
            print(f"\nOutput projection weights: {len(o_proj_keys)}")

            layer_counts = set()
            for k in weight_keys:
                if "model.layers." in k:
                    parts = k.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i+1 < len(parts):
                            layer_counts.add(parts[i+1])

            print(f"Model layer count: {len(layer_counts)}")

            print("\nKey weight samples:")
            for prefix in ["model.layers.0.self_attn.o_proj", "model.embed_tokens"]:
                matching_keys = [k for k in weight_keys if k.startswith(prefix)]
                for k in matching_keys[:5]:
                    tensor = f.get_tensor(k)
                    print(f"{k}: Shape {tensor.shape}, Type {tensor.dtype}")

            print("\nAll weight keys:")
            for key in sorted(weight_keys):
                tensor = f.get_tensor(key)
                print(f"{key}: Shape {tensor.shape}, Type {tensor.dtype}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/analysis/model_analysis.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    analyze_model(model_path)
