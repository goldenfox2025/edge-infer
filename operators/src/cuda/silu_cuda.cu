#include <cuda_runtime.h>

#include <stdexcept>
#include <limits>

#include "operators/cuda/direct.hpp"

namespace op {

// CUDA kernel for SiLU activation function
template <typename T>
__global__ void silu_kernel(T* output, const T* input, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    float x = static_cast<float>(input[idx]);
    output[idx] = static_cast<T>(x / (1.0f + expf(-x)));
  }
}

template <typename T>
__global__ void silu_multiply_kernel(const T* gate, const T* up, T* output, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    const float x = static_cast<float>(gate[idx]);
    // Keep the same rounded activation that the standalone SiLU kernel stores.
    const T activated = static_cast<T>(x / (1.0f + expf(-x)));
    // An explicit rounded multiply prevents fast-math reassociation through
    // the SiLU division; BF16 conversion preserves the intermediate rounding.
    output[idx] = static_cast<T>(__fmul_rn(static_cast<float>(activated),
                                          static_cast<float>(up[idx])));
  }
}

// Implementation of SiLU CUDA operator
template <typename T>
void cuda::silu(ArrayView<const T> input, ArrayView<T> output, cudaStream_t stream) {

  size_t total = input.size;
  if (output.size != total) {
    throw std::runtime_error("SiLU operator: input and output views must have the same size");
  }
  if (total == 0) {
    return;
  }
  if (total > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("SiLU CUDA view extent exceeds the kernel index range");
  }

  int threads_per_block = 256;
  int blocks = (total + threads_per_block - 1) / threads_per_block;

  silu_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
      output.data, input.data, total);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in SiLU kernel: " +
                             std::string(cudaGetErrorString(err)));
  }
}

template <typename T>
void cuda::silu_multiply(ArrayView<const T> gate, ArrayView<const T> up,
                         ArrayView<T> output, cudaStream_t stream) {
  const size_t total = gate.size;
  if (up.size != total || output.size != total) {
    throw std::runtime_error("SiLU multiply operator: input and output views must have the same size");
  }
  if (!total) return;
  if (total > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("SiLU multiply CUDA view extent exceeds the kernel index range");
  }
  constexpr int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  silu_multiply_kernel<T><<<blocks, threads, 0, stream>>>(gate.data, up.data, output.data,
                                                        static_cast<int>(total));
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error("CUDA error in SiLU multiply kernel: " +
                             std::string(cudaGetErrorString(status)));
  }
}


template void cuda::silu<float>(ArrayView<const float>, ArrayView<float>, cudaStream_t);
template void cuda::silu<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, cudaStream_t);
template void cuda::silu_multiply<float>(ArrayView<const float>, ArrayView<const float>, ArrayView<float>, cudaStream_t);
template void cuda::silu_multiply<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, cudaStream_t);


}  // namespace op
