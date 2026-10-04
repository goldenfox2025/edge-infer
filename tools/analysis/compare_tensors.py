#!/usr/bin/env python3
"""Compare binary tensor files and report detailed differences."""

import os
import struct
import numpy as np
import argparse
import glob

def read_tensor_from_binary(filename: str) -> np.ndarray:
    """Read tensor data from a binary file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File does not exist: {filename}")

    with open(filename, 'rb') as f:

        ndim = struct.unpack('Q', f.read(8))[0]  # size_t = uint64

        shape = []
        for _ in range(ndim):
            dim = struct.unpack('Q', f.read(8))[0]
            shape.append(dim)

        dtype_size = struct.unpack('Q', f.read(8))[0]

        if dtype_size == 4:
            dtype = np.float32
        elif dtype_size == 2:
            dtype = np.float16 # for bfloat16, we use float16 approximation
        else:
            raise ValueError(f"Unsupported data type size: {dtype_size}")

        total_elements = np.prod(shape)
        data = f.read(total_elements * dtype_size)

        if dtype_size == 2:
            # for bfloat16, requires special handling
            raw_data = np.frombuffer(data, dtype=np.uint16)
            # Simple bfloat16 to float32 Conversion ( not fully accurate, but sufficient for comparison )
            tensor_data = raw_data.astype(np.float32) / 256.0
        elif dtype_size == 4 and filename.endswith('input_token.bin'):

            tensor_data = np.frombuffer(data, dtype=np.uint32)
        else:
            tensor_data = np.frombuffer(data, dtype=dtype)

        return tensor_data.reshape(shape)

def compare_two_files(file1: str, file2: str):
    """Compare two tensor files."""
    print(f"=== Comparing files ===")
    print(f"file 1: {file1}")
    print(f"file 2: {file2}")
    print()

    try:
        tensor1 = read_tensor_from_binary(file1)
        tensor2 = read_tensor_from_binary(file2)
    except Exception as e:
        print(f"❌ Failed to read the file: {e}")
        return

    print(f"Tensor 1 Shape: {tensor1.shape}, Data type: {tensor1.dtype}")
    print(f"Tensor 2 Shape: {tensor2.shape}, Data type: {tensor2.dtype}")

    if tensor1.shape != tensor2.shape:
        print("❌ Tensor shapes do not match!")
        return

    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    diff = np.abs(flat1 - flat2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    rel_diff = diff / (np.abs(flat1) + 1e-10)
    max_rel_diff = np.max(rel_diff)

    is_close = np.allclose(flat1, flat2, rtol=1e-5, atol=1e-8)

    print(f"\n=== Difference statistics ===")
    print(f"Maximum absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Maximum relative difference: {max_rel_diff:.6e}")
    print(f"Within tolerance: {'✅ Yes' if is_close else '❌ No'}")
    print(f"Total elements: {len(flat1)}")

    max_diff_idx = np.argmax(diff)

    print(f"\n=== Position of the maximum difference ===")
    print(f"Position index: {max_diff_idx}")
    if len(tensor1.shape) > 1:
        multi_idx = np.unravel_index(max_diff_idx, tensor1.shape)
        print(f"Multidimensional index: {multi_idx}")
    print(f"Tensor 1 Value: {flat1[max_diff_idx]:.6f}")
    print(f"Tensor 2 Value: {flat2[max_diff_idx]:.6f}")
    print(f"Absolute difference: {diff[max_diff_idx]:.6e}")
    print(f"Relative difference: {rel_diff[max_diff_idx]:.6e}")

    print(f"\n=== Near the maximum difference 10 elements ===")
    start_idx = max(0, max_diff_idx - 5)
    end_idx = min(len(flat1), max_diff_idx + 6)

    print("Index \t\t Tensor 1\t\t Tensor 2\t\t Absolute difference \t Relative difference")
    print("-" * 80)
    for i in range(start_idx, end_idx):
        marker = " *** " if i == max_diff_idx else "     "
        print(f"{i:6d}{marker}\t{flat1[i]:12.6f}\t{flat2[i]:12.6f}\t{diff[i]:12.6e}\t{rel_diff[i]:12.6e}")

    print(f"\n=== first 10 element comparison ===")
    print("Index \t\t Tensor 1\t\t Tensor 2\t\t Absolute difference \t Relative difference")
    print("-" * 80)
    for i in range(min(10, len(flat1))):
        print(f"{i:6d}\t\t{flat1[i]:12.6f}\t{flat2[i]:12.6f}\t{diff[i]:12.6e}\t{rel_diff[i]:12.6e}")

    if max_diff > 1e-3:
        print(f"\n=== Statistics for elements with large differences ===")
        large_diff_mask = diff > 1e-3
        large_diff_count = np.sum(large_diff_mask)
        print(f"Difference > 1e-3 element count: {large_diff_count} ({large_diff_count/len(flat1)*100:.2f}%)")

        if large_diff_count > 0:
            large_diff_indices = np.where(large_diff_mask)[0]
            print(f"first 5 indices of elements with large differences: {large_diff_indices[:5].tolist()}")

def compare_all_files():
    """Automatically compare all corresponding files."""

    graph_files = glob.glob("debug_graph_*.bin")
    cuda_files = glob.glob("cuda/debug_cuda_*.bin")

    if not graph_files:
        print("Not found graph file (debug_graph_*.bin)")
        return

    if not cuda_files:
        print("Not found cuda file (cuda/debug_cuda_*.bin)")
        return

    print(f"Found {len(graph_files)} items graph files and {len(cuda_files)} items cuda file")

    graph_patterns = {}
    for f in graph_files:
        pattern = f.replace("debug_graph_", "").replace(".bin", "")
        graph_patterns[pattern] = f

    cuda_patterns = {}
    for f in cuda_files:
        pattern = f.replace("cuda/debug_cuda_", "").replace(".bin", "")
        cuda_patterns[pattern] = f

    common_patterns = set(graph_patterns.keys()) & set(cuda_patterns.keys())

    if not common_patterns:
        print("No matching file pairs found")
        print(f"Graph Pattern: {list(graph_patterns.keys())[:5]}...")
        print(f"CUDA Pattern: {list(cuda_patterns.keys())[:5]}...")
        return

    print(f"Found {len(common_patterns)} matched file pairs \n")

    for pattern in sorted(common_patterns):
        graph_file = graph_patterns[pattern]
        cuda_file = cuda_patterns[pattern]

        print("=" * 100)
        print(f"Comparison pattern: {pattern}")
        compare_two_files(graph_file, cuda_file)
        print()

def main():
    parser = argparse.ArgumentParser(description="Compare two binary tensor files")
    parser.add_argument('--file1', help="Path to the first file")
    parser.add_argument('--file2', help="Path to the second file")
    parser.add_argument('--auto', action='store_true', help="Automatically compare all matching files")

    args = parser.parse_args()

    if args.auto:
        compare_all_files()
    elif args.file1 and args.file2:
        compare_two_files(args.file1, args.file2)
    else:

        file1 = "debug_graph_embedding.bin"
        file2 = "cuda/debug_cuda_embedding.bin"

        if os.path.exists(file1) and os.path.exists(file2):
            compare_two_files(file1, file2)
        else:
            print(f"Default file does not exist:")
            print(f" {file1}: {'exists' if os.path.exists(file1) else 'does not exist'}")
            print(f" {file2}: {'exists' if os.path.exists(file2) else 'does not exist'}")
            print("\nUsage:")
            print("  python tools/analysis/compare_tensors.py --file1 file1.bin --file2 file2.bin")
            print("  python tools/analysis/compare_tensors.py --auto # Compare all matching files")

if __name__ == "__main__":
    main()
