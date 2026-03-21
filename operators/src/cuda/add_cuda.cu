#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "operators/cuda/add_cuda.cuh"

namespace op {

// CUDA Kernel for element-wise addition
template <typename T>
__global__ void add_kernel(const T* input_a, const T* input_b, T* output, size_t total) {
  // 计算全局线程索引
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  // 每个线程处理多个元素
  for (size_t i = idx; i < total; i += stride) {
    output[i] = input_a[i] + input_b[i];
  }
}

// CUDA Kernel for element-wise addition using float2 for better memory throughput
__global__ void add_kernel_float2(const float2* input_a, const float2* input_b, 
                                 float2* output, size_t total_vec2) {
  // 计算全局线程索引
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  // 每个线程处理多个float2元素
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
  // 计算全局线程索引
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  // 每个线程处理多个__nv_bfloat162元素
  for (size_t i = idx; i < total_vec2; i += stride) {
#if __CUDA_ARCH__ >= 800 // Ampere (SM 8.0) 及更高架构
    // 使用硬件加速的 2x BF16 加法指令
    output[i] = __hadd2(input_a[i], input_b[i]);
#else
    // 兼容旧架构的回退方法
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
void AddCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input_a,
                                   Tensor<T>* input_b, cudaStream_t stream) {
  // 获取输入张量的大小
  size_t total = input_a->numel();

  // 检查输入张量的大小是否一致
  if (input_b->numel() != total) {
    throw std::runtime_error(
        "Add operator: input tensors must have the same size");
  }

  // 如果张量为空，则无需操作
  if (total == 0) {
    return;
  }

  // 配置CUDA核函数的启动参数
  int threads_per_block = 256;  // 可以根据需要调整
  
  // 获取设备属性以优化网格大小
  int device;
  cudaGetDevice(&device);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
  
  // 检查是否可以使用向量化加载/存储
  bool use_vectorized = (total % 2 == 0) && 
                        (reinterpret_cast<uintptr_t>(input_a->data_ptr()) % (2 * sizeof(T)) == 0) &&
                        (reinterpret_cast<uintptr_t>(input_b->data_ptr()) % (2 * sizeof(T)) == 0) &&
                        (reinterpret_cast<uintptr_t>(output->data_ptr()) % (2 * sizeof(T)) == 0);

  if (use_vectorized) {
    // 使用向量化版本
    size_t total_vec2 = total / 2;
    int blocks = std::min((int)((total_vec2 + threads_per_block - 1) / threads_per_block), numSMs * 32);
    
    if constexpr (std::is_same_v<T, float>) {
      add_kernel_float2<<<blocks, threads_per_block, 0, stream>>>(
          reinterpret_cast<const float2*>(input_a->data_ptr()),
          reinterpret_cast<const float2*>(input_b->data_ptr()),
          reinterpret_cast<float2*>(output->data_ptr()),
          total_vec2);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      add_kernel_bf162<<<blocks, threads_per_block, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat162*>(input_a->data_ptr()),
          reinterpret_cast<const __nv_bfloat162*>(input_b->data_ptr()),
          reinterpret_cast<__nv_bfloat162*>(output->data_ptr()),
          total_vec2);
    } else {
      // 对于其他类型，使用标准版本
      int blocks = std::min((int)((total + threads_per_block - 1) / threads_per_block), numSMs * 32);
      add_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
          input_a->data_ptr(), input_b->data_ptr(), output->data_ptr(), total);
    }
  } else {
    // 使用标准版本
    int blocks = std::min((int)((total + threads_per_block - 1) / threads_per_block), numSMs * 32);
    add_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
        input_a->data_ptr(), input_b->data_ptr(), output->data_ptr(), total);
  }

  // 检查核函数启动是否成功
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error in Add operator: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("Add CUDA kernel launch failed");
  }
}

// 显式模板实例化
template class AddCUDAOperator<float>;
template class AddCUDAOperator<__nv_bfloat16>;

}  // namespace op
