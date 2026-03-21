#include <cuda_runtime.h>

#include <stdexcept>

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
void SiluCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input,
                                     cudaStream_t stream) {
  // 获取输入张量的大小
  size_t total = input->numel();

  // 如果输入和输出不是同一个张量，需要复制数据
  if (output->data_ptr() != input->data_ptr()) {
    cudaMemcpyAsync(output->data_ptr(), input->data_ptr(), total * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
  }

  // 配置CUDA核函数的启动参数
  int threads_per_block = 256;  // 可以根据需要调整
  int blocks = (total + threads_per_block - 1) / threads_per_block;

  // 启动核函数
  silu_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
      output->data_ptr(), input->data_ptr(), total);

  // 错误检查
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in SiLU kernel: " +
                             std::string(cudaGetErrorString(err)));
  }
}

// 显式模板实例化
template class SiluCUDAOperator<float>;
template class SiluCUDAOperator<__nv_bfloat16>;

}  // namespace op
