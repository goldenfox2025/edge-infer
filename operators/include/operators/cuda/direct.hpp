#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "operators/core/array_view.hpp"

namespace op::cuda {

// Direct launch functions for float and BF16 device buffers. They borrow
// contiguous storage and launch on the supplied stream. They allocate no
// operand/workspace storage, register no operators, copy no shared ownership,
// and use no virtual dispatch.
// Matching extents are checked; elementwise counts must fit a signed int.
// Exact input/output aliasing is supported;
// partial overlap is unsupported. CUDA launch/runtime overhead still applies.
// Buffers belong to the current CUDA device and remain valid until the supplied
// stream completes. Calls return asynchronously; synchronize or use events
// before consuming outputs or reusing storage from another stream.
template <typename T>
void add(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> output,
         cudaStream_t stream = nullptr);

template <typename T>
void multiply(ArrayView<const T> a, ArrayView<const T> b, ArrayView<T> output,
              cudaStream_t stream = nullptr);

template <typename T>
void silu(ArrayView<const T> input, ArrayView<T> output,
          cudaStream_t stream = nullptr);

// Preserve the unfused dtype staging: round SiLU to T, then multiply by up
// and round the product to T. Exact output=gate aliasing is supported.
template <typename T>
void silu_multiply(ArrayView<const T> gate, ArrayView<const T> up,
                   ArrayView<T> output, cudaStream_t stream = nullptr);

// Each row has feature_dim elements. Weights must not overlap output.
// The existing kernel caches at most ten values per thread at 1024 threads:
// nonempty inputs require 1 <= feature_dim <= 10240. Empty views do not launch.
template <typename T>
void rms_norm(ArrayView<const T> input, ArrayView<const T> weights,
              ArrayView<T> output, std::size_t rows,
              std::size_t feature_dim, float eps,
              cudaStream_t stream = nullptr);

extern template void add<float>(ArrayView<const float>, ArrayView<const float>, ArrayView<float>, cudaStream_t);
extern template void add<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, cudaStream_t);
extern template void multiply<float>(ArrayView<const float>, ArrayView<const float>, ArrayView<float>, cudaStream_t);
extern template void multiply<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, cudaStream_t);
extern template void silu<float>(ArrayView<const float>, ArrayView<float>, cudaStream_t);
extern template void silu<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, cudaStream_t);
extern template void silu_multiply<float>(ArrayView<const float>, ArrayView<const float>, ArrayView<float>, cudaStream_t);
extern template void silu_multiply<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, cudaStream_t);
extern template void rms_norm<float>(ArrayView<const float>, ArrayView<const float>, ArrayView<float>, std::size_t, std::size_t, float, cudaStream_t);
extern template void rms_norm<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, std::size_t, std::size_t, float, cudaStream_t);

}  // namespace op::cuda
