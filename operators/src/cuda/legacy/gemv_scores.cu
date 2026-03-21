#ifndef CUDA_GEMMV_OP_CUH  // 建议为文件定义一个唯一的头文件保护符
#define CUDA_GEMMV_OP_CUH

#include <cuda_bf16.h>  // 提供 nv_bfloat16, __float2bfloat16 等
#include <cuda_fp16.h>  // 提供 half, __float2half, __half2float, __hadd, half2 等
#include <stdint.h>  // 提供 int64_t 等标准整数类型

#include <cstdio>       // 提供 fprintf, stderr (用于可能的错误输出)
#include <stdexcept>    // 提供 std::runtime_error, std::invalid_argument
#include <string>       // 提供 std::string, std::to_string
#include <type_traits>  // 提供 std::is_same, std::conditional

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace cuda_OP {

// 设备函数：Warp 内求和归约 (使用 float 进行中间累加)
// 输入: val - 每个线程的局部 float 值
// 输出: warp 内所有线程 val 值的总和 (结果仅在 warp 内线程 0 中有效)
// 功能: 利用 __shfl_down_sync 指令在 warp 内高效地聚合求和。
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll  // 提示编译器展开循环
  // Butterfly reduction using shuffle down
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    // 从距离当前线程 offset 的下方线程获取值，并加到当前线程的 val 上
    // 0xFFFFFFFF 表示 warp 内所有线程都参与同步
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  // 循环结束后，warp 内线程 0 拥有该 warp 的总和
  return val;
}

//----------------------------------------------------------------------------//
// GEMMV Kernel: 计算矩阵-向量乘积 (针对 Attention Score 计算优化)           //
//----------------------------------------------------------------------------//
// T: 输入数据类型 (如 float, nv_bfloat16)
// type_acc: 累加和输出的数据类型 (通常是 float 或与 T 相同)
// block_size: 每个 CUDA 块的线程数 (必须是 warpSize 的倍数)
template <typename T, typename type_acc, int block_size>
static __global__ void gemv_s(
    const T* x,     // 输入: 矩阵 Q (设备指针)
    const T* y,     // 输入: 向量 K/V (设备指针)
    type_acc* dst,  // 输出: 结果 (设备指针)
    const int channel_ratio,  // 输入: Q 头数量 / KV 头数量的比率 (用于 GQA/MQA)
    const int stride_channel_x,  // 输入: 矩阵 x (Q) 的通道/头维度步长
    const int stride_channel_y,  // 输入: 向量 y (K/V) 的通道/头维度步长
    const int stride_channel_dst  // 输入: 输出 dst 的通道/头维度步长
) {
  // 计算块/线程索引
  const int64_t seq_idx = blockIdx.x;  // 当前处理的序列/行索引
  const int64_t channel = blockIdx.y;  // 当前处理的通道/头索引 (对应输出)
  const int tid = threadIdx.x;   // 块内线程 ID (0 到 block_size-1)
  constexpr int warp_size = 32;  // Warp 大小 (固定为 32)

  // --- 指针调整 ---
  // 根据块索引调整输入/输出指针，定位到当前块负责的数据区域
  // x 指针根据 seq_idx 和 channel (除以 ratio) 定位到对应的 Q 向量
  x += seq_idx * (gridDim.y / channel_ratio) *
           stride_channel_x +  // 定位到正确的 Q 序列行
       (channel / channel_ratio) * stride_channel_x;  // 定位到正确的 Q 头
  // y 指针根据 channel 定位到对应的 K/V 向量/头
  y += channel * stride_channel_y;
  // dst 指针根据 channel 定位到输出位置的起始点 (行由 seq_idx 决定)
  dst += channel * stride_channel_dst;

  // --- 共享内存 (用于块内归约) ---
  // 仅当块大小 > warp 大小时才需要额外的块内归约步骤
  __shared__ float smem[warp_size];  // 分配足够存储一个 warp 结果的共享内存
  if (block_size > warp_size) {
    // 如果块大于 warp，由第一个 warp 的线程初始化共享内存
    if (tid < warp_size) {
      smem[tid] = 0.0f;
    }
    __syncthreads();  // 确保共享内存初始化完成
  }

  // --- 点积计算 ---
  float sumf = 0.0f;  // 使用 float 进行累加，以获得更高精度

  // --- nv_bfloat16 类型优化分支 ---
  if constexpr (std::is_same<T, nv_bfloat16>::value) {
    // 定义向量加载单元包含的元素数 (16字节 / 单个元素大小)
    constexpr int vec_unit =
        16 / sizeof(T);  // 对于 nv_bfloat16 (2字节), vec_unit = 8
    // 将输入指针重新解释为自定义向量类型 Vec 的指针，以便进行 16 字节加载
    // (假设 Vec<T, vec_unit> 在 cudaOP.cuh 中定义，通常是一个包含 T t[vec_unit]
    // 的结构或联合)
    const Vec<T, vec_unit>* x_vec =
        reinterpret_cast<const Vec<T, vec_unit>*>(x);
    const Vec<T, vec_unit>* y_vec =
        reinterpret_cast<const Vec<T, vec_unit>*>(y);

    // 循环处理向量的各个部分 (网格跨步)
    // stride_channel_y 是向量 y 的总长度 (以 T 为单位)
    for (int64_t col_vec = tid; col_vec < stride_channel_y / vec_unit;
         col_vec += block_size) {
      Vec<T, vec_unit> xi = x_vec[col_vec];  // 加载 16 字节 (vec_unit 个 T)
      Vec<T, vec_unit> yi = y_vec[col_vec];  // 加载 16 字节 (vec_unit 个 T)
      // 在向量内部进行元素级的乘积累加
      for (int i = 0; i < vec_unit; ++i) {
        // 将 bfloat16 转换为 float 进行计算
        sumf += static_cast<float>(xi.t[i]) * static_cast<float>(yi.t[i]);
      }
    }
    // --- float 类型优化分支 ---
  } else if constexpr (std::is_same<T, float>::value) {
    // 对于 float 类型，使用 float4 进行向量化加载 (16 字节)
    constexpr int vec_unit = 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const float4* y_vec = reinterpret_cast<const float4*>(y);
    // stride_channel_y 是向量 y 的总长度 (以 float 为单位)
    for (int64_t col_vec = tid; col_vec < stride_channel_y / vec_unit;
         col_vec += block_size) {
      float4 xi = x_vec[col_vec];  // 加载 4 个 float
      float4 yi = y_vec[col_vec];  // 加载 4 个 float
      // 直接计算 4 对 float 的乘积累加
      sumf += xi.x * yi.x + xi.y * yi.y + xi.z * yi.z + xi.w * yi.w;
    }
    // --- 不支持的类型 ---
  } else {
    // 如果 T 不是 float 或 nv_bfloat16，编译时报错
    static_assert(std::is_same<T, void>::value,
                  "gemmv_s: unsupported data type T");
  }

  // --- Warp 内归约 ---
  // 对每个线程计算出的局部和 sumf，在 warp 内部进行求和
  sumf = warp_reduce_sum(sumf);  // 结果在每个 warp 的线程 0 中

  // --- 块内归约 (如果 block_size > warp_size) ---
  if (block_size > warp_size) {
    // 每个 warp 的线程 0 将其 warp 的结果写入共享内存
    if (tid % warp_size == 0) {
      smem[tid / warp_size] = sumf;
    }
    __syncthreads();  // 确保所有 warp 的结果都已写入共享内存

    // 由第一个 warp (tid < warp_size) 负责读取共享内存并进行最终归约
    if (tid < warp_size) {
      // 从共享内存加载对应 warp 的结果 (如果 tid 超出实际 warp 数量则加载 0)
      sumf = (tid < block_size / warp_size) ? smem[tid] : 0.0f;
      // 在第一个 warp 内部再次进行求和归约
      sumf = warp_reduce_sum(sumf);
    }
    // 其他 warp 的线程在此之后不再需要工作
  }

  // --- 写回结果 ---
  // 只有线程 0 (块内的最终归约结果持有者) 才执行写操作
  if (tid == 0) {
    // 将最终结果乘以缩放因子 (1 / sqrt(head_dim))，这里硬编码为 128
    // 注意: 硬编码 128 可能特定于某个模型配置，一般应作为参数传入或从配置读取
    const float scale_factor = rsqrtf(128.0f);  // 计算 1/sqrt(128)
    // 将计算结果 (float) 转换为目标输出类型 (type_acc) 并写入目标内存地址
    // 输出位置由 dst 基地址 + seq_idx (行偏移) 决定
    dst[seq_idx] = static_cast<type_acc>(sumf * scale_factor);
  }
}

//----------------------------------------------------------------------------//
// 主机端启动函数 (Launcher Function)                                         //
//----------------------------------------------------------------------------//
// T: 输入数据类型
// AccT: 累加和输出数据类型
template <typename T, typename AccT>
void launch_gemv_scores(
    const T* x,              // 输入: 矩阵 Q (设备指针)
    const T* y,              // 输入: 向量 K/V (设备指针)
    AccT* dst,               // 输出: 结果 (设备指针)
    const int channel_size,  // 输入: 输出通道/头的数量 (gridDim.y)
    const int channel_ratio,  // 输入: Q头/KV头比率 (用于核函数内指针计算)
    const int row_size,  // 输入: 需要处理的行数/序列长度 (gridDim.x)
    const int stride_channel_x,    // 输入: 矩阵 x 的通道步长
    const int stride_channel_y,    // 输入: 向量 y 的通道步长
    const int stride_channel_dst,  // 输入: 输出 dst 的通道步长
    cudaStream_t stream) {         // 输入: CUDA 流
  // --- 执行配置 ---
  // 网格维度: x 维度对应行数 (row_size), y 维度对应输出通道数 (channel_size)
  dim3 grid(row_size, channel_size);
  // 块维度: 设置每个块的线程数
  constexpr int block_size =
      128;  // 选择一个常用的块大小 (应为 warpSize 的倍数)
  dim3 block(block_size);

  // --- 启动核函数 ---
  gemv_s<T, AccT, block_size>
      <<<grid, block, 0, stream>>>(x, y, dst, channel_ratio, stride_channel_x,
                                   stride_channel_y, stride_channel_dst);

  // 注意: 这里不包含 cudaGetLastError 或 cudaStreamSynchronize
  // 调用者负责检查错误和同步流
}

// --- 显式模板实例化 ---
// 为支持的类型组合预编译核函数代码

// float 输入, float 输出
template void launch_gemv_scores<float, float>(
    const float* x, const float* y, float* dst, const int channel_size,
    const int channel_ratio, const int row_size, const int stride_channel_x,
    const int stride_channel_y, const int stride_channel_dst,
    cudaStream_t stream);

// nv_bfloat16 输入, nv_bfloat16 输出
template void launch_gemv_scores<nv_bfloat16, nv_bfloat16>(
    const nv_bfloat16* x, const nv_bfloat16* y, nv_bfloat16* dst,
    const int channel_size, const int channel_ratio, const int row_size,
    const int stride_channel_x, const int stride_channel_y,
    const int stride_channel_dst, cudaStream_t stream);

// nv_bfloat16 输入, float 输出 (累加和输出使用 float)
template void launch_gemv_scores<nv_bfloat16, float>(
    const nv_bfloat16* x, const nv_bfloat16* y, float* dst,
    const int channel_size, const int channel_ratio, const int row_size,
    const int stride_channel_x, const int stride_channel_y,
    const int stride_channel_dst, cudaStream_t stream);

}  // namespace cuda_OP

#endif  // CUDA_GEMMV_OP_CUH