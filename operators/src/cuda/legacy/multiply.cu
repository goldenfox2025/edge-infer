#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {

template <typename T>
__global__ void multiply_kernel_v3(const T *A, const T *B, T *out, int total) {
  // 计算每次载入的 T 元素个数
  constexpr int vec_unit = 16 / sizeof(T);
  typedef Vec<T, vec_unit> VecT;

  // 计算能整除的向量块数量
  int total_vec = total / vec_unit;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // 将 A, B, out 转换为向量化的指针
  const VecT *A_vec = reinterpret_cast<const VecT *>(A);
  const VecT *B_vec = reinterpret_cast<const VecT *>(B);
  VecT *out_vec = reinterpret_cast<VecT *>(out);

  // 处理完整的向量块
  for (int i = tid; i < total_vec; i += stride) {
    VecT a_val = A_vec[i];
    VecT b_val = B_vec[i];
    VecT result;
#pragma unroll
    for (int j = 0; j < vec_unit; j++) {
      result.t[j] = a_val.t[j] * b_val.t[j];
    }
    out_vec[i] = result;
  }

  // 处理尾部不足向量块的部分
  int remaining = total - total_vec * vec_unit;
  int offset = total_vec * vec_unit;
  for (int i = tid; i < remaining; i += stride) {
    out[offset + i] = A[offset + i] * B[offset + i];
  }
}

// --------------------------------------------------
// 模板化的逐元素乘法内核实现（版本 v2，使用 __ldg 进行只读内存加载优化）
// --------------------------------------------------
template <typename T>
__global__ void multiply_kernel_v2(const T *A, const T *B, T *out, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = __ldg(&A[idx]) * __ldg(&B[idx]);
  }
}

// --------------------------------------------------
// 模板化的逐元素乘法内核实现（版本 v1，不使用 __ldg，仅做对比）
// --------------------------------------------------
template <typename T>
__global__ void multiply_kernel_v1(const T *A, const T *B, T *out, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = A[idx] * B[idx];
  }
}

// --------------------------------------------------
// 模板化的 host 端逐元素乘法函数
// 输入张量 A 与 B 均为 Tensor<T> 类型，输出张量 output 用于存储结果。
// --------------------------------------------------
template <typename T>
void multiply(Tensor<T> *output, const Tensor<T> *A, const Tensor<T> *B,
              cudaStream_t stream) {
  size_t total = A->numel();
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  multiply_kernel_v3<T><<<blocks, threads, 0, stream>>>(
      A->data_ptr(), B->data_ptr(), output->data_ptr(), total);
  checkCudaError(cudaGetLastError());
  // if (stream == nullptr) {
  //   checkCudaError(cudaDeviceSynchronize());
  // }
}
template void multiply<float>(Tensor<float> *, const Tensor<float> *,
                              const Tensor<float> *, cudaStream_t);

template void multiply<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *,
                               const Tensor<nvbf16> *, cudaStream_t);
}  // namespace cuda_OP
