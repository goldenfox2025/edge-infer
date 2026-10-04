#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "operators/core/array_view.hpp"

namespace op::cuda {

// Borrowed contiguous row-major [vocab_size, features] device storage.
template <typename T>
struct EmbeddingTable {
  ArrayView<const T> values;
  std::size_t vocab_size = 0;
};

// input [rows,in_features], physical weight [out_features,in_features], optional
// bias [out_features], output [rows,out_features]. The caller supplies at least
// rows*out_features floats of device scratch. GEMM accumulates into float;
// bias is added before the final conversion to T, including for BF16.
// The borrowed handle must use host pointer mode and CUBLAS_DEFAULT_MATH, and
// must not be used concurrently. Its stream is set explicitly to stream.
template <typename T>
void linear(ArrayView<const T> input, ArrayView<const T> weight,
            ArrayView<const T> bias, ArrayView<T> output,
            ArrayView<float> accumulation, std::size_t rows,
            std::size_t in_features, std::size_t out_features,
            cublasHandle_t handle, cudaStream_t stream = nullptr);

// Host IDs are row-major [frames,tables.size]; every codebook may have its own
// vocabulary size. Validate all host IDs and table extents before launching.
// Accumulate all table lookups in float, then convert once to T. The caller
// supplies at least frames*features floats of device scratch. Host descriptors
// and IDs are consumed during this call and may be released on return.
template <typename T>
void sum_embeddings(ArrayView<const EmbeddingTable<T>> tables,
                    ArrayView<const uint32_t> host_ids, ArrayView<T> output,
                    ArrayView<float> accumulation, std::size_t frames,
                    std::size_t features, cudaStream_t stream = nullptr);

// Both primitives support float and BF16 only, allocate no operand/workspace
// storage, and return asynchronously. All device views belong to the current
// CUDA device and remain valid until stream completion. Operands and scratch
// must not overlap; output may exactly alias accumulation for float. Other
// output overlap is unsupported. Empty operations do not touch CUDA/cuBLAS.

extern template void linear<float>(ArrayView<const float>, ArrayView<const float>,
    ArrayView<const float>, ArrayView<float>, ArrayView<float>, std::size_t,
    std::size_t, std::size_t, cublasHandle_t, cudaStream_t);
extern template void linear<__nv_bfloat16>(ArrayView<const __nv_bfloat16>,
    ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>,
    ArrayView<__nv_bfloat16>, ArrayView<float>, std::size_t, std::size_t,
    std::size_t, cublasHandle_t, cudaStream_t);
extern template void sum_embeddings<float>(ArrayView<const EmbeddingTable<float>>,
    ArrayView<const uint32_t>, ArrayView<float>, ArrayView<float>, std::size_t,
    std::size_t, cudaStream_t);
extern template void sum_embeddings<__nv_bfloat16>(
    ArrayView<const EmbeddingTable<__nv_bfloat16>>, ArrayView<const uint32_t>,
    ArrayView<__nv_bfloat16>, ArrayView<float>, std::size_t, std::size_t,
    cudaStream_t);

}  // namespace op::cuda
