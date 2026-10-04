#include <cuda_runtime.h>

#include <stdexcept>
#include <limits>

#include "operators/cuda/direct.hpp"
#include "operators/cuda/silu_cuda.cuh"

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
void SiluCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input,
                                    cudaStream_t stream) {
  cuda::silu<T>({input->data_ptr(), input->numel()},
                {output->data_ptr(), output->numel()}, stream);
}

template void cuda::silu<float>(ArrayView<const float>, ArrayView<float>, cudaStream_t);
template void cuda::silu<__nv_bfloat16>(ArrayView<const __nv_bfloat16>, ArrayView<__nv_bfloat16>, cudaStream_t);

template class SiluCUDAOperator<float>;
template class SiluCUDAOperator<__nv_bfloat16>;

}  // namespace op
