#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "operators/cuda/direct.hpp"
#include "operators/cuda/execution.hpp"

namespace op {

// CUDA Kernel for element-wise addition
template <typename T>
__global__ void add_kernel(const T* input_a, const T* input_b, T* output, size_t total) {

  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (size_t i = idx; i < total; i += stride) {
    output[i] = input_a[i] + input_b[i];
  }
}

// CUDA Kernel for element-wise addition using float2 for better memory throughput
__global__ void add_kernel_float2(const float2* input_a, const float2* input_b,
                                 float2* output, size_t total_vec2) {

  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (size_t i = idx; i < total_vec2; i += stride) {
    float2 a_val = input_a[i];
    float2 b_val = input_b[i];
    float2 result;
    result.x = a_val.x + b_val.x;
    result.y = a_val.y + b_val.y;
    output[i] = result;
  }
}

// CUDA Kernel for element-wise addition using __nv_bfloat162 for better memory throughput
__global__ void add_kernel_bf162(const __nv_bfloat162* input_a, const __nv_bfloat162* input_b,
                                __nv_bfloat162* output, size_t total_vec2) {

  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (size_t i = idx; i < total_vec2; i += stride) {
#if __CUDA_ARCH__ >= 800 // Ampere (SM80) and newer.
    // Use paired BF16 addition on SM80 and newer.
    output[i] = __hadd2(input_a[i], input_b[i]);
#else
    // Use component-wise arithmetic on older architectures.
    __nv_bfloat162 a_val = input_a[i];
    __nv_bfloat162 b_val = input_b[i];
    __nv_bfloat162 result;
    result.x = a_val.x + b_val.x;
    result.y = a_val.y + b_val.y;
    output[i] = result;
#endif
  }
}

// Implementation of Add CUDA operator
template <typename T>
void cuda::add(ArrayView<const T> input_a, ArrayView<const T> input_b,
                  ArrayView<T> output, cudaStream_t stream) {

  size_t total = input_a.size;

  if (input_b.size != total || output.size != total) {
    throw std::runtime_error(
        "Add operator: input and output views must have the same size");
  }

  if (total == 0) {
    return;
  }
  if (total > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Add CUDA view extent exceeds the launch index range");
  }

  int threads_per_block = 256;

  int device;
  cudaError_t query_status = cudaGetDevice(&device);
  if (query_status != cudaSuccess) {
    throw std::runtime_error("Add CUDA device query failed: " +
                             std::string(cudaGetErrorString(query_status)));
  }
  int numSMs;
  query_status = cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
  if (query_status != cudaSuccess) {
    throw std::runtime_error("Add CUDA device attribute query failed: " +
                             std::string(cudaGetErrorString(query_status)));
  }

  // Paired loads require an even extent and aligned input/output pointers.
  bool use_vectorized = (total % 2 == 0) &&
                        (reinterpret_cast<uintptr_t>(input_a.data) % (2 * sizeof(T)) == 0) &&
                        (reinterpret_cast<uintptr_t>(input_b.data) % (2 * sizeof(T)) == 0) &&
                        (reinterpret_cast<uintptr_t>(output.data) % (2 * sizeof(T)) == 0);

  if (use_vectorized) {

    size_t total_vec2 = total / 2;
    int blocks = std::min((int)((total_vec2 + threads_per_block - 1) / threads_per_block), numSMs * 32);

    if constexpr (std::is_same_v<T, float>) {
      add_kernel_float2<<<blocks, threads_per_block, 0, stream>>>(
          reinterpret_cast<const float2*>(input_a.data),
          reinterpret_cast<const float2*>(input_b.data),
          reinterpret_cast<float2*>(output.data),
          total_vec2);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      add_kernel_bf162<<<blocks, threads_per_block, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat162*>(input_a.data),
          reinterpret_cast<const __nv_bfloat162*>(input_b.data),
          reinterpret_cast<__nv_bfloat162*>(output.data),
          total_vec2);
    } else {

      int blocks = std::min((int)((total + threads_per_block - 1) / threads_per_block), numSMs * 32);
      add_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
          input_a.data, input_b.data, output.data, total);
    }
  } else {

    int blocks = std::min((int)((total + threads_per_block - 1) / threads_per_block), numSMs * 32);
    add_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
        input_a.data, input_b.data, output.data, total);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error in Add operator: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("Add CUDA kernel launch failed");
  }
}



template <typename T>
void cuda::add(const ExecutionContext& context, ArrayView<const T> input_a,
                ArrayView<const T> input_b, ArrayView<T> output) {
  const size_t count = input_a.size;
  if (input_b.size != count || output.size != count ||
      count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      (count && (!input_a.data || !input_b.data || !output.data))) {
    throw std::invalid_argument("Direct add buffer extents or pointers are invalid");
  }
  if (!count) return;
  constexpr int threads = 256;
  const bool paired = count % 2 == 0 &&
      reinterpret_cast<uintptr_t>(input_a.data) % (2 * sizeof(T)) == 0 &&
      reinterpret_cast<uintptr_t>(input_b.data) % (2 * sizeof(T)) == 0 &&
      reinterpret_cast<uintptr_t>(output.data) % (2 * sizeof(T)) == 0;
  if (paired) {
    const size_t pairs = count / 2;
    const int blocks = std::min<int>((pairs + threads - 1) / threads,
                                     context.multiprocessors * 32);
    if constexpr (std::is_same_v<T, float>) {
      add_kernel_float2<<<blocks, threads, 0, context.stream>>>(
          reinterpret_cast<const float2*>(input_a.data),
          reinterpret_cast<const float2*>(input_b.data),
          reinterpret_cast<float2*>(output.data), pairs);
    } else {
      add_kernel_bf162<<<blocks, threads, 0, context.stream>>>(
          reinterpret_cast<const __nv_bfloat162*>(input_a.data),
          reinterpret_cast<const __nv_bfloat162*>(input_b.data),
          reinterpret_cast<__nv_bfloat162*>(output.data), pairs);
    }
  } else {
    const int blocks = std::min<int>((count + threads - 1) / threads,
                                     context.multiprocessors * 32);
    add_kernel<T><<<blocks, threads, 0, context.stream>>>(input_a.data, input_b.data, output.data, count);
  }
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template void cuda::add<float>(const cuda::ExecutionContext&, ArrayView<const float>, ArrayView<const float>, ArrayView<float>);
template void cuda::add<__nv_bfloat16>(const cuda::ExecutionContext&, ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>);
template void cuda::add<float>(ArrayView<const float>, ArrayView<const float>, ArrayView<float>, cudaStream_t);
template void cuda::add<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, cudaStream_t);


}  // namespace op
