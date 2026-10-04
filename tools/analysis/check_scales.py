#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse
import os
import numpy as np
from pathlib import Path

def check_scales(model_path):
    """Inspect AWQ scale tensors and report detailed statistics."""
    print(f"Load the model: {model_path}")

    try:
        weights = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return

    print(f"Model loaded successfully, Total {len(weights)} keys")

    scales_keys = [k for k in weights.keys() if ".scales" in k]

    print(f"Found {len(scales_keys)} items scales weights")

    for key in sorted(scales_keys):
        scale = weights[key]
        if not isinstance(scale, torch.Tensor):
            print(f"{key}: is not Tensor Type")
            continue

        dtype = scale.dtype
        shape = scale.shape
        device = scale.device

        with torch.no_grad():
            scale_np = scale.float().cpu().numpy()
            total_elements = scale_np.size
            non_zero = np.count_nonzero(scale_np)
            non_zero_percent = (non_zero / total_elements) * 100

            min_val = scale_np.min() if total_elements > 0 else "N/A"
            max_val = scale_np.max() if total_elements > 0 else "N/A"
            mean_val = scale_np.mean() if total_elements > 0 else "N/A"

            almost_zero = np.sum(np.abs(scale_np) < 1e-6)

        print("\n" + "="*50)
        print(f"Key: {key}")
        print(f"Type: {dtype}, Shape: {shape}, Device: {device}")
        print(f"Total elements: {total_elements}")
        print(f"Non-zero elements: {non_zero} ({non_zero_percent:.4f}%)")
        print(f"Near-zero elements (<1e-6): {almost_zero} ({almost_zero/total_elements*100:.4f}%)")
        print(f"Minimum: {min_val}")
        print(f"Maximum: {max_val}")
        print(f"Mean: {mean_val}")

        print("Sample values:")

        if len(shape) == 2:
            first_row = scale_np[0, :min(10, shape[1])]
            print(f"First row, first 10 values: {first_row}")

            if shape[0] > 1:
                random_indices = np.random.choice(shape[0], min(5, shape[0]), replace=False)
                for idx in random_indices:
                    if idx == 0:
                        continue
                    row_sample = scale_np[idx, :min(10, shape[1])]
                    print(f"index {idx} row, first 10 values: {row_sample}")

        if total_elements > 0:
            small_vals_count = {}
            thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

            for t in thresholds:
                count = np.sum((scale_np > 0) & (scale_np < t))
                small_vals_count[t] = count

            print("Small-value distribution:")
            for t, count in small_vals_count.items():
                print(f"  0 < x < {t}: {count} ({count/total_elements*100:.4f}%)")

def main():
    parser = argparse.ArgumentParser(description="Check AWQ model weights scales Value")
    parser.add_argument("model_path", help="Path to the model weight file")
    args = parser.parse_args()

    check_scales(args.model_path)

if __name__ == "__main__":
    main()