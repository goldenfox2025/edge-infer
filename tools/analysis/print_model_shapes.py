#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import torch
from collections import OrderedDict
import json

def print_tensor_shapes(model_path):
    """Load a model and print all weight tensor shapes.

    Args:
        model_path: Path to the model directory."""
    print(f"Loading the model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        sys.exit(1)

    pytorch_files = [f for f in os.listdir(model_path) if f.endswith('.pt') or f.endswith('.bin') or f.endswith('.pth')]
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]

    try:

        weights = None

        if len(pytorch_files) > 0:
            weight_file = os.path.join(model_path, pytorch_files[0])
            print(f"Trying to load PyTorch Weight file: {weight_file}")
            try:
                weights = torch.load(weight_file, map_location='cpu')
            except Exception as e:
                print(f"Load directly PyTorch file failed: {e}")

        if weights is None and len(safetensors_files) > 0:
            try:
                from safetensors.torch import load_file
                weight_file = os.path.join(model_path, safetensors_files[0])
                print(f"Trying to load Safetensors Weight file: {weight_file}")
                weights = load_file(weight_file)
            except Exception as e:
                print(f"Load Safetensors file failed: {e}")

        if weights is None:
            try:
                from transformers import AutoModel
                print("Trying to use transformers load the model with the library")
                model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
                weights = model.state_dict()
            except Exception as e:
                print(f"Use transformers Loading failed: {e}")

        if weights is None:
            print("All loading methods failed, Check the model files")
            sys.exit(1)

        print("\n===================== Model weight shapes =====================")

        sorted_weights = OrderedDict()

        if isinstance(weights, dict):
            weight_dict = weights
            if 'state_dict' in weights:
                weight_dict = weights['state_dict']

            for key, tensor in weight_dict.items():
                sorted_weights[key] = tensor
        else:
            print(f"Warning: Model is not a standard dict format, instead {type(weights)}")
            sys.exit(1)

        for i, (key, tensor) in enumerate(sorted(sorted_weights.items())):
            print(f"{i+1:4d}. {key:80s} | Shape: {tuple(tensor.shape)}")

            if 'lm_head' in key or 'embed' in key:
                print(f"-> Note: The weight shape is [{tensor.shape[0]}, {tensor.shape[1]}]" +
                      f" ({'NK format' if tensor.shape[0] > tensor.shape[1] else 'KN format'})")

        print("\n===================== Statistics =====================")
        total_params = sum(p.numel() for p in sorted_weights.values())
        print(f"Weight count: {len(sorted_weights)}")
        print(f"Total parameters: {total_params:,}")
        print(f"Total parameters ( billions ): {total_params / 1e9:.2f}B")

        shapes_info = {k: list(v.shape) for k, v in sorted_weights.items()}
        with open(f"{os.path.basename(model_path)}_shapes.json", 'w') as f:
            json.dump(shapes_info, f, indent=2)

        print(f"\nShape information saved to: {os.path.basename(model_path)}_shapes.json")

    except Exception as e:
        print(f"Program failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model_path = os.environ.get(
        "MODEL_PATH", str(Path(__file__).resolve().parents[2] / "models" / "Qwen3-1.7B")
    )

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print_tensor_shapes(model_path)
