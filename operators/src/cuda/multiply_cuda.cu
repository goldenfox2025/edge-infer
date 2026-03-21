#include <cuda_runtime.h>

#include <stdexcept>

#include "operators/cuda/multiply_cuda.cuh"

namespace op {

// 向量化类型定义，用于提高内存访问效率
template <typename T, int N>
struct Vec {
  T t[N];
};

// CUDA kernel for element-wise multiplication (vectorized version)
template <typename T>
__global__ void multiply_kernel(const T* input_a, const T* input_b, T* output,
                                int total) {
  // 计算每次载入的 T 元素个数
  constexpr int vec_unit = 16 / sizeof(T);
  typedef Vec<T, vec_unit> VecT;

  // 计算能整除的向量块数量
  int total_vec = total / vec_unit;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // 将 input_a, input_b, output 转换为向量化的指针
  const VecT* input_a_vec = reinterpret_cast<const VecT*>(input_a);
  const VecT* input_b_vec = reinterpret_cast<const VecT*>(input_b);
  VecT* output_vec = reinterpret_cast<VecT*>(output);

  // 处理完整的向量块
  for (int i = tid; i < total_vec; i += stride) {
    VecT a_val = input_a_vec[i];
    VecT b_val = input_b_vec[i];
    VecT result;
#pragma unroll
    for (int j = 0; j < vec_unit; j++) {
      result.t[j] = a_val.t[j] * b_val.t[j];
    }
    output_vec[i] = result;
  }

  // 处理尾部不足向量块的部分
  int remaining = total - total_vec * vec_unit;
  int offset = total_vec * vec_unit;
  for (int i = tid; i < remaining; i += stride) {
    output[offset + i] = input_a[offset + i] * input_b[offset + i];
  }
}

// Implementation of Multiply CUDA operator
template <typename T>
void MultiplyCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input_a,
                                         Tensor<T>* input_b,
                                         cudaStream_t stream) {
  // 获取输入张量的大小
  size_t total = input_a->numel();

  // 检查输入张量的大小是否一致
  if (input_b->numel() != total) {
    throw std::runtime_error(
        "Multiply operator: input tensors must have the same size");
  }

  // 配置CUDA核函数的启动参数
  int threads_per_block = 256;  // 可以根据需要调整
  int blocks = (total + threads_per_block - 1) / threads_per_block;

  // 启动核函数
  multiply_kernel<T><<<blocks, threads_per_block, 0, stream>>>(
      input_a->data_ptr(), input_b->data_ptr(), output->data_ptr(), total);

  // 错误检查
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in Multiply kernel: " +
                             std::string(cudaGetErrorString(err)));
  }
}

// 显式模板实例化
template class MultiplyCUDAOperator<float>;
template class MultiplyCUDAOperator<__nv_bfloat16>;

}  // namespace op
