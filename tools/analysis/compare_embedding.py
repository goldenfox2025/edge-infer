#!/usr/bin/env python3
"""Compare embedding vectors produced by CUDA and graph inference."""

import os
import struct
import numpy as np
import sys

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
            if filename.endswith('input_token.bin'):
                dtype = np.uint32
            else:
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
        else:
            tensor_data = np.frombuffer(data, dtype=dtype)

        return tensor_data.reshape(shape)

def compare_embedding_vectors(cuda_file: str, graph_file: str):
    """Compare embedding vectors from CUDA and graph inference."""
    print(f"=== Compare embedding Vector ===")
    print(f"CUDA file: {cuda_file}")
    print(f"graph files: {graph_file}")
    print()

    try:
        cuda_embedding = read_tensor_from_binary(cuda_file)
        graph_embedding = read_tensor_from_binary(graph_file)
    except Exception as e:
        print(f"❌ Failed to read the file: {e}")
        return False

    print(f"CUDA embedding Shape: {cuda_embedding.shape}, Data type: {cuda_embedding.dtype}")
    print(f"Graph embedding Shape: {graph_embedding.shape}, Data type: {graph_embedding.dtype}")

    if cuda_embedding.shape != graph_embedding.shape:
        print("❌ embedding Shapes do not match!")
        return False

    flat_cuda = cuda_embedding.flatten()
    flat_graph = graph_embedding.flatten()

    diff = np.abs(flat_cuda - flat_graph)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    rel_diff = diff / (np.abs(flat_cuda) + 1e-10)
    max_rel_diff = np.max(rel_diff)

    is_close = np.allclose(flat_cuda, flat_graph, rtol=1e-5, atol=1e-8)

    print(f"\n=== Difference statistics ===")
    print(f"Maximum absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Maximum relative difference: {max_rel_diff:.6e}")
    print(f"Within tolerance: {'✅ Yes' if is_close else '❌ No'}")
    print(f"Total elements: {len(flat_cuda)}")

    if is_close:
        print("\n\u2705 embedding vectors are effectively identical, Weight-table access works correctly")
        print("The problem may be in gather operation or subsequent processing")
        return True
    else:
        print("\n\u274c embedding vectors differ substantially, Weight-table access has a problem")

        max_diff_idx = np.argmax(diff)
        print(f"\n=== Position of the maximum difference ===")
        print(f"Position index: {max_diff_idx}")
        print(f"CUDA Value: {flat_cuda[max_diff_idx]:.6f}")
        print(f"Graph value: {flat_graph[max_diff_idx]:.6f}")
        print(f"Absolute difference: {diff[max_diff_idx]:.6e}")

        print(f"\n=== first 10 element comparison ===")
        print("Index \t\tCUDA\t\t Graph \t\t Absolute difference")
        print("-" * 60)
        for i in range(min(10, len(flat_cuda))):
            print(f"{i:6d}\t\t{flat_cuda[i]:12.6f}\t{flat_graph[i]:12.6f}\t{diff[i]:12.6e}")

        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python tools/analysis/compare_embedding.py <token_id> <token_id>")
        print("For example: python tools/analysis/compare_embedding.py 9707 1")
        print("This compares:")
        print("  cuda/debug_cuda_token_<token_id>_embedding.bin")
        print("  graph/debug_graph_token_<token_id>_embedding.bin")
        return

    token_id1 = sys.argv[1]
    token_id2 = sys.argv[2]

    cuda_file1 = f"cuda/debug_cuda_token_{token_id1}_embedding.bin"
    graph_file1 = f"graph/debug_graph_token_{token_id1}_embedding.bin"

    if os.path.exists(cuda_file1) and os.path.exists(graph_file1):
        print(f"🔍 Compare token {token_id1} of embedding Vector")
        result1 = compare_embedding_vectors(cuda_file1, graph_file1)
        print("\n" + "="*80 + "\n")
    else:
        print(f"❌ token {token_id1} file does not exist:")
        print(f" {cuda_file1}: {'exists' if os.path.exists(cuda_file1) else 'does not exist'}")
        print(f" {graph_file1}: {'exists' if os.path.exists(graph_file1) else 'does not exist'}")
        result1 = False

    if token_id1 != token_id2:
        cuda_file2 = f"cuda/debug_cuda_token_{token_id2}_embedding.bin"
        graph_file2 = f"graph/debug_graph_token_{token_id2}_embedding.bin"

        if os.path.exists(cuda_file2) and os.path.exists(graph_file2):
            print(f"🔍 Compare token {token_id2} of embedding Vector")
            result2 = compare_embedding_vectors(cuda_file2, graph_file2)
        else:
            print(f"❌ token {token_id2} file does not exist:")
            print(f" {cuda_file2}: {'exists' if os.path.exists(cuda_file2) else 'does not exist'}")
            print(f" {graph_file2}: {'exists' if os.path.exists(graph_file2) else 'does not exist'}")
            result2 = False
    else:
        result2 = True

    print("\n" + "="*80)
    print("🎯 Summary:")
    if result1 and result2:
        print("\u2705 All embedding vectors all match, Weight-table access works correctly")
        print("The problem is gather processing after the operation")
    else:
        print("\u274c embedding vectors do not match, Weight-table access has a problem")
        print("Check that graph inference uses the correct weight table")

if __name__ == "__main__":
    main()
