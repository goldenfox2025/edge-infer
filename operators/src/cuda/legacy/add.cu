#ifndef CUDA_ADD_OP_CUH // 建议文件名与内容匹配
#define CUDA_ADD_OP_CUH

#include <cuda_bf16.h>  // 提供 nv_bfloat16, nv_bfloat162, __hadd2 等
#include <cuda_runtime.h>
#include <math.h>

#include <algorithm>  // 用于 std::min, std::max
#include <cstdio>
#include <iostream>
#include <stdexcept>    // 用于 std::runtime_error
#include <type_traits>  // 用于 std::is_same_v
#include <vector>

// 假设 cudaOP.cuh 包含 Tensor 定义, checkCudaError 宏等
#include "cuda/legacy/legacy_cuda_api.cuh"
// 假设 nvbf16 是在 cudaOP.cuh 中定义的 __nv_bfloat16 的别名
// using nvbf16 = __nv_bfloat16; // 可能在 cudaOP.cuh 中

namespace cuda_OP {

// --------------------------------------------------
// Kernel v1: 基础版本 - 网格跨步循环 (Grid-Stride Loop)
// 功能: 执行基本的元素加法 C = A + B。
// --------------------------------------------------
template <typename T>
__global__ void add_kernel_v1(const T* A,      // 输入: 张量 A (设备指针)
                              const T* B,      // 输入: 张量 B (设备指针)
                              T* out,        // 输出: 张量 C (设备指针)
                              size_t total) { // 输入: 元素总数
  // 计算全局线程索引 (网格跨步)
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  // 每个线程处理多个元素
  for (size_t i = idx; i < total; i += stride) {
    // 标准加载和存储，依赖 L1/L2 缓存
    out[i] = A[i] + B[i];
  }
}

// --------------------------------------------------
// Kernel v2: 向量化版本 (x2) - 网格跨步循环
// 功能: 利用向量类型 (float2 或 nv_bfloat162) 一次处理两个元素。
// --------------------------------------------------

// Kernel v2: 针对 float 类型 (使用 float2)
__global__ void add_kernel_float_v2(
    const float2* A,           // 输入: 张量 A (设备指针, float2 类型)
    const float2* B,           // 输入: 张量 B (设备指针, float2 类型)
    float2* out,             // 输出: 张量 C (设备指针, float2 类型)
    size_t total_vec2)       // 输入: float2 元素的总数 (原总数 / 2)
{
  // 计算全局线程索引 (网格跨步，针对 float2)
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  // 每个线程处理多个 float2 元素
  for (size_t i = idx; i < total_vec2; i += stride) {
    float2 a_val = A[i]; // 加载一对 float
    float2 b_val = B[i]; // 加载一对 float
    float2 result;
    // 分别计算 .x 和 .y 分量
    result.x = a_val.x + b_val.x;
    result.y = a_val.y + b_val.y;
    out[i] = result;     // 存储一对 float
  }
}

// Kernel v2: 针对 bfloat16 类型 (使用 nv_bfloat162)
__global__ void add_kernel_bf16_v2(
    const nv_bfloat162* A, // 输入: 张量 A (设备指针, nv_bfloat162 类型)
    const nv_bfloat162* B, // 输入: 张量 B (设备指针, nv_bfloat162 类型)
    nv_bfloat162* out,   // 输出: 张量 C (设备指针, nv_bfloat162 类型)
    size_t total_vec2)     // 输入: nv_bfloat162 元素的总数 (原总数 / 2)
{
  // 计算全局线程索引 (网格跨步，针对 nv_bfloat162)
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  // 每个线程处理多个 nv_bfloat162 元素
  for (size_t i = idx; i < total_vec2; i += stride) {
#if __CUDA_ARCH__ >= 800 // Ampere (SM 8.0) 及更高架构
    // 使用硬件加速的 2x BF16 加法指令
    out[i] = __hadd2(A[i], B[i]);
#else
    // 兼容旧架构的回退方法 (效率较低)
    nv_bfloat162 a_val = A[i];
    nv_bfloat162 b_val = B[i];
    nv_bfloat162 result;
    // 直接相加 (可能涉及隐式转换)
    result.x = a_val.x + b_val.x;
    result.y = a_val.y + b_val.y;
    out[i] = result;
#endif
  }
}

// --------------------------------------------------
// 主机端函数: 执行张量加法 (C = A + B)
// 功能: 检查输入，选择最优核函数 (向量化或标量)，并启动核函数。
// --------------------------------------------------
template <typename T>
void add(Tensor<T>* output,     // 输出: 结果张量 C (指针)
         Tensor<T>* A,          // 输入: 张量 A (指针)
         Tensor<T>* B,          // 输入: 张量 B (指针)
         cudaStream_t stream) { // 输入: CUDA 流，用于异步执行
  // --- 基础检查 ---
  if (A->numel() != B->numel() || A->numel() != output->numel()) {
    throw std::runtime_error("张量加法要求形状匹配");
  }
  if (A->device() != Device::CUDA || B->device() != Device::CUDA ||
      output->device() != Device::CUDA) {
    throw std::runtime_error("所有张量必须在 CUDA 设备上");
  }

  size_t total = A->numel(); // 获取元素总数
  if (total == 0) {
    return; // 如果张量为空，则无需操作
  }

  // --- 核函数启动配置 ---
  int threads = 256; // 每个块的线程数 (常用值, 可调整)

  // --- 判断是否可以使用向量化核函数 (v2) ---
  bool can_vectorize = false;    // 标记类型是否支持向量化
  size_t vec_alignment = 0;      // 向量类型的内存对齐要求

  // 检查类型是否支持向量化
  if constexpr (std::is_same_v<T, float>) {
    vec_alignment = alignof(float2); // float2 的对齐要求 (通常是 8)
    can_vectorize = true;
  } else if constexpr (std::is_same_v<T, nvbf16>) { // 假设 nvbf16 是 __nv_bfloat16 的别名
    vec_alignment = alignof(nv_bfloat162); // nv_bfloat162 的对齐要求 (通常是 4)
    can_vectorize = true;
  }

  // 检查向量化的所有条件:
  // 1. 类型支持 (can_vectorize == true)
  // 2. 元素总数是偶数 (total % 2 == 0)
  // 3. 输入输出张量的数据指针都满足向量类型的对齐要求
  bool use_vectorized_kernel =
      can_vectorize && (total % 2 == 0) &&
      (reinterpret_cast<uintptr_t>(A->data_ptr()) % vec_alignment == 0) &&
      (reinterpret_cast<uintptr_t>(B->data_ptr()) % vec_alignment == 0) &&
      (reinterpret_cast<uintptr_t>(output->data_ptr()) % vec_alignment == 0);

  // --- 启动选择的核函数 ---
  if (use_vectorized_kernel) {
    // --- 启动向量化核函数 (v2) ---
    size_t total_vec2 = total / 2; // 向量元素的总数

    // 计算网格大小 (块的数量)
    // 使用启发式方法，目标是让足够多的块在 GPU 上并发执行以占满 SM
    int device;
    checkCudaError(cudaGetDevice(&device)); // 获取当前设备 ID
    int numSMs;
    checkCudaError(cudaDeviceGetAttribute(
        &numSMs, cudaDevAttrMultiProcessorCount, device)); // 获取 SM 数量
    // 目标块数: 覆盖所有元素所需的最小块数，但不超过 SM 数量的某个倍数 (例如 32 倍)
    int blocks = std::max(
        1, std::min((int)((total_vec2 + threads - 1) / threads), numSMs * 32));

    // 根据类型启动相应的向量化核函数
    if constexpr (std::is_same_v<T, float>) {
      add_kernel_float_v2<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const float2*>(A->data_ptr()), // 指针类型转换为 float2*
          reinterpret_cast<const float2*>(B->data_ptr()),
          reinterpret_cast<float2*>(output->data_ptr()),
          total_vec2);
    } else if constexpr (std::is_same_v<T, nvbf16>) {
      add_kernel_bf16_v2<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const nv_bfloat162*>(A->data_ptr()), // 指针类型转换为 nv_bfloat162*
          reinterpret_cast<const nv_bfloat162*>(B->data_ptr()),
          reinterpret_cast<nv_bfloat162*>(output->data_ptr()),
          total_vec2);
    }
  } else {
    // --- 回退到标量核函数 (v1) ---
    // 计算网格大小 (与上面类似，但基于总元素数 total)
    int device;
    checkCudaError(cudaGetDevice(&device));
    int numSMs;
    checkCudaError(cudaDeviceGetAttribute(
        &numSMs, cudaDevAttrMultiProcessorCount, device));
    int blocks = std::max(
        1, std::min((int)((total + threads - 1) / threads), numSMs * 32));

    // 启动标量核函数
    add_kernel_v1<T><<<blocks, threads, 0, stream>>>(
        A->data_ptr(), B->data_ptr(), output->data_ptr(), total);
  }

  // 检查核函数启动时可能发生的异步错误
  checkCudaError(cudaGetLastError());

}

// --- 显式实例化主机函数模板 ---
// 为支持的类型 (float 和 nvbf16) 生成函数代码
template void add<float>(Tensor<float>*, Tensor<float>*, Tensor<float>*, cudaStream_t);
template void add<nvbf16>(Tensor<nvbf16>*, Tensor<nvbf16>*, Tensor<nvbf16>*, cudaStream_t);

} // namespace cuda_OP

#endif // CUDA_ADD_OP_CUH