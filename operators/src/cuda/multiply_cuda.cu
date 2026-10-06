#include <cuda_runtime.h>

#include <stdexcept>
#include <limits>

#include "operators/cuda/direct.hpp"
#include "elementwise_math.cuh"

namespace op {

template <typename T, int N>
struct Vec {
  T t[N];
};

// CUDA kernel for element-wise multiplication (vectorized version)
template <typename T>
__global__ void multiply_kernel(const T* input_a, const T* input_b, T* output,
                                int total) {

  constexpr int vec_unit = 16 / sizeof(T);
  typedef Vec<T, vec_unit> VecT;

  size_t total_vec = static_cast<size_t>(total) / vec_unit;
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  const VecT* input_a_vec = reinterpret_cast<const VecT*>(input_a);
  const VecT* input_b_vec = reinterpret_cast<const VecT*>(input_b);
  VecT* output_vec = reinterpret_cast<VecT*>(output);

  for (size_t i = tid; i < total_vec; i += stride) {
    VecT a_val = input_a_vec[i];
    VecT b_val = input_b_vec[i];
    VecT result;
#pragma unroll
    for (int j = 0; j < vec_unit; j++) {
      result.t[j] = static_cast<T>(cuda::detail::multiply_preserving_subnormals(
          static_cast<float>(a_val.t[j]), static_cast<float>(b_val.t[j])));
    }
    output_vec[i] = result;
  }

  // Handle the scalar tail after the complete vector blocks.
  size_t remaining = static_cast<size_t>(total) - total_vec * vec_unit;
  size_t offset = total_vec * vec_unit;
  for (size_t i = tid; i < remaining; i += stride) {
    output[offset + i] = static_cast<T>(cuda::detail::multiply_preserving_subnormals(
        static_cast<float>(input_a[offset + i]), static_cast<float>(input_b[offset + i])));
  }
}

// Implementation of Multiply CUDA operator
template <typename T>
void cuda::multiply(ArrayView<const T> input_a, ArrayView<const T> input_b,
                  ArrayView<T> output, cudaStream_t stream) {

  size_t total = input_a.size;

  if (input_b.size != total || output.size != total) {
    throw std::runtime_error(
        "Multiply operator: input and output views must have the same size");
  }

  if (total == 0) {
    return;
  }
  if (total > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Multiply CUDA view extent exceeds the kernel index range");
  }

  int threads_per_block = 256;
  int blocks = (total + threads_per_block - 1) / threads_per_block;

  multiply_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
      input_a.data, input_b.data, output.data, total);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in Multiply kernel: " +
                             std::string(cudaGetErrorString(err)));
  }
}



template void cuda::multiply<float>(ArrayView<const float>, ArrayView<const float>, ArrayView<float>, cudaStream_t);
template void cuda::multiply<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, cudaStream_t);


}  // namespace op
