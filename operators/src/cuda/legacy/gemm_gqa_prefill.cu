

#ifndef CUDA_GQA_GEMM_CUH
#define CUDA_GQA_GEMM_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>  // 如果 tensor.sizes() 返回 std::vector 则需要

#include "cuda/legacy/legacy_cuda_api.cuh"  // 假设 Tensor 类在此处定义

namespace cuda_OP {

/**
 * 分块 GQA GEMM 核函数 v2
 * 输入输出和 v1 保持一致
 * 输入布局:
 *   Q: [seq_len, n_q_heads, head_dim]
 *   K: [total_seq_len, n_kv_heads, head_dim]
 * 输出布局:
 *   scores: [seq_len, n_q_heads, total_seq_len]
 */
template <typename T,  // 数据类型 (float, half, nv_bfloat16)
          int BM,      // 块块维度 M (Q 序列长度)
          int BN,      // 块块维度 N (K 序列长度)
          int BK,      // 块块维度 K (头维度)
          int TM,      // 线程块维度 M
          int TN       // 线程块维度 N
          >
__global__ void gqa_gemm_kernel_v2(
    const T *__restrict__ Q,  // 查询输入 [seq_len, n_q_heads, head_dim]
    const T *__restrict__ K,  // 键输入   [total_seq_len, n_kv_heads, head_dim]
    T *__restrict__ scores,   // 输出    [seq_len, n_q_heads, total_seq_len]
    int seq_len,              // Q 序列长度
    int head_dim,             // 每个头的维度
    // int batch_size,       // 对于 3D 输入不直接使用，假定为 1
    int n_q_heads,      // 查询头的数量
    int n_kv_heads,     // 键/值头的数量
    int ratio,          // n_q_heads / n_kv_heads 的比例
    int total_seq_len,  // K 序列长度 (可以不同于 seq_len)
    float scale         // 缩放因子 (1.0f / sqrtf(head_dim))
) {
  // 每一个块负责BM*BN大小的结果
  // 每个线程块负责部分的序号
  int block_k_seq_idx = blockIdx.x;
  int block_q_seq_idx = blockIdx.y;

  // z维度可计算批次（本实现暂忽略）
  int q_head_idx = blockIdx.z;

  // 确定对应的 KV 头索引
  int kv_head_idx = q_head_idx / ratio;

  // 计算每个线程块负责部分的具体起始位置
  int block_q_seq_start = block_q_seq_idx * BM;
  int block_k_seq_start = block_k_seq_idx * BN;

  // 块内线程索引计算
  // 使用一维索引便于映射
  int tid = threadIdx.x;

  constexpr int THREADS_PER_BLOCK_N = BN / TN;

  // 将线性线程 ID 映射到块的 2D 线程坐标
  int thread_m_idx = tid / THREADS_PER_BLOCK_N;
  int thread_n_idx = tid % THREADS_PER_BLOCK_N;

  // 计算此线程在其块输出块内的起始行/列
  int thread_q_seq_start = block_q_seq_start + thread_m_idx * TM;
  int thread_k_seq_start = block_k_seq_start + thread_n_idx * TN;

  int ph = 1;
  __shared__ float smemQ[2][BM][BK + 1];
  __shared__ float smemK[2][BN][BK + 1];

  // 累加精度
  float accum[TM][TN] = {{0.0f}};

  // 计算得出 seqlen 维度对应的步长
  // head_dim 为 head 对应维度的步长
  int q_stride_seq = n_q_heads * head_dim;
  int q_stride_head = head_dim;

  int k_stride_seq = n_kv_heads * head_dim;
  int k_stride_head = head_dim;

  // 通过 head 维度的步长，索引得到每个 head 处理 的对象
  // 转换为 n_head seqlen dim 的维度安排
  // 相当于n_head为批次，计算一个二维矩阵的乘法
  const T *q_head_ptr = Q + q_head_idx * q_stride_head;
  const T *k_head_ptr = K + kv_head_idx * k_stride_head;

  // 这是加载 vec 的结果
  constexpr int vec_unit = 16 / sizeof(T);
  int k_tile_start = 0;

// 理论上，M 维度的线程数量为 BM/TM
// all_tid = BM*BN/(TM*TN)
// Q或者K的分块加载 需要确保BM*BK或者BN*BK的数据
// 1次向量化加载可以负责vec_unit个数据
// 以Q为例，BM*BK/(vec_unit*cnt)是需要的总线程数
// cnt是一个线程执行向量化加载次数
// TM=2 TN=2 BK=32 BM=32 BN=32
// 这样的配置下，总线程为256（16*16），需要加载32*32个数据。
// 每个线程加载1个float4，即8个数据。加载32*4次需要128个线程
// 我们考虑使用 grid - stride - loop，可以简单地分发任务
// 我们还应该考虑内存访问事务
// 缓存行大小是128字节。
// 尽量使得warp访问的全局内存是合并的
// 枚举需要处理的行，tid则负责索引行内
// --- 使用 Grid-Stride Loop (如果需要) 遍历 smemQ 的行 ---
// 这个循环是为了处理 BM 可能大于一个块一次能加载的行数的情况
#pragma unroll(1)
  for (int load_idx = tid; load_idx < BM * (BK / vec_unit);
       load_idx += blockDim.x) {
    // 1. 计算此向量在 smemQ 中的目标行和起始列
    int smem_q_row = load_idx / (BK / vec_unit);
    int vec_idx_in_row = load_idx % (BK / vec_unit);
    int smem_q_col_start = vec_idx_in_row * vec_unit;

    // 2. 计算对应的全局内存读取地址
    int global_q_seq = block_q_seq_start + smem_q_row;  // 全局行
    int global_q_dim_vec_start =
        k_tile_start + smem_q_col_start;  // 全局列 (向量起点)

    // 3. 严格边界检查 和 加载/写入
    if (global_q_seq < seq_len) {
      // 行有效，检查列
      if (global_q_dim_vec_start + vec_unit - 1 < head_dim) {
        // --- 情况 A: 完整向量在边界内，直接向量化加载 ---
        Vec<T, vec_unit> vq;
        vq.f4 = *reinterpret_cast<const float4 *>(
            &q_head_ptr[global_q_seq * q_stride_seq + global_q_dim_vec_start]);

// 水平写入 smemQ
#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          // 但加上更安全)
          if (smem_q_col_start + i < BK) {
            smemQ[ph & 1][smem_q_row][smem_q_col_start + i] =
                static_cast<float>(vq.t[i]);
          }
        }
      } else {
// --- 情况 B: 向量跨越 head_dim 边界，处理部分加载 ---
#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          int current_smem_col = smem_q_col_start + i;
          if (current_smem_col < BK) {  // 检查 smem 列边界
            int current_global_dim = k_tile_start + current_smem_col;
            if (current_global_dim < head_dim) {  // 检查全局列边界
              // 只加载和写入在 head_dim 内的部分
              T element =
                  q_head_ptr[global_q_seq * q_stride_seq + current_global_dim];
              smemQ[ph & 1][smem_q_row][current_smem_col] =
                  static_cast<float>(element);
            } else {
              // 超出 head_dim 的部分在 smem 中补零
              smemQ[ph & 1][smem_q_row][current_smem_col] = 0.0f;
            }
          }
        }
      }
    } else {
// --- 情况 C: 整行无效 (global_q_seq >= seq_len) ---
// 将 smemQ 中对应的向量位置全部置零
#pragma unroll
      for (int i = 0; i < vec_unit; ++i) {
        int current_smem_col = smem_q_col_start + i;
        if (current_smem_col < BK) {  // 检查 smem 列边界
          smemQ[ph & 1][smem_q_row][current_smem_col] = 0.0f;
        }
      }
    }
  }  // 结束 Q 加载循环

  // --- 开始加载 K 到 smemK ---
  // 使用与 Q 加载完全相同的逻辑结构，但替换为 K 的参数

  // VEC_LOADS_PER_ROW_K: K 的共享内存行需要多少向量加载 (BK 维度)

// const int THREADS_PER_ROW_LOAD_GROUP_K = blockDim.x / VEC_LOADS_PER_ROW_K; //
// 理想分组

// --- 使用 Grid-Stride Loop (如果需要) 遍历 smemK 的行 ---
#pragma unroll(1)
     // 循环遍历 K tile 中的所有 *向量* (注意维度是 BN)
  for (int load_idx = tid; load_idx < BN * (BK / vec_unit);
       load_idx += blockDim.x) {
    // 1. 计算此向量在 smemK 中的目标行和起始列
    int smem_k_row = load_idx / (BK / vec_unit);
    int vec_idx_in_row = load_idx % (BK / vec_unit);
    int smem_k_col_start = vec_idx_in_row * vec_unit;

    // 2. 计算对应的全局内存读取地址 (使用 K 的起始块和 stride)
    int global_k_seq = block_k_seq_start + smem_k_row;  // <<< 使用 K 的起始行
    int global_k_dim_vec_start = k_tile_start + smem_k_col_start;

    // 3. 严格边界检查 和 加载/写入 (注意使用 total_seq_len)
    if (global_k_seq < total_seq_len) {  // <<< 使用 K 的总长度
      if (global_k_dim_vec_start + vec_unit - 1 < head_dim) {
        // --- 情况 A: 完整向量加载 ---
        Vec<T, vec_unit> vk;
        vk.f4 = *reinterpret_cast<const float4 *>(
            &k_head_ptr[global_k_seq * k_stride_seq + global_k_dim_vec_start]);

// 水平写入 smemK
#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          if (smem_k_col_start + i < BK) {
            smemK[ph & 1][smem_k_row][smem_k_col_start + i] =
                static_cast<float>(vk.t[i]);
          }
        }
      } else {
// --- 情况 B: 部分向量加载 ---
#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          int current_smem_col = smem_k_col_start + i;
          if (current_smem_col < BK) {
            int current_global_dim = k_tile_start + current_smem_col;
            if (current_global_dim < head_dim) {
              T element =
                  k_head_ptr[global_k_seq * k_stride_seq + current_global_dim];
              smemK[ph & 1][smem_k_row][current_smem_col] =
                  static_cast<float>(element);
            } else {
              smemK[ph & 1][smem_k_row][current_smem_col] = 0.0f;
            }
          }
        }
      }
    } else {
// --- 情况 C: 整行无效 (for K) ---
#pragma unroll
      for (int i = 0; i < vec_unit; ++i) {
        int current_smem_col = smem_k_col_start + i;
        if (current_smem_col < BK) {
          smemK[ph & 1][smem_k_row][current_smem_col] = 0.0f;
        }
      }
    }
  }  // 结束 K 加载循环

  // 根据BK遍历 MK NK 的 K 维度。
  for (k_tile_start = BK; k_tile_start <= head_dim; k_tile_start += BK) {
    ph ^= 1;
#pragma unroll 1  // 通常不对 grid-stride loop 做完全展开
    for (int load_idx = tid; load_idx < BM * (BK / vec_unit);
         load_idx += blockDim.x) {
      // 1. 计算此向量在 smemQ 中的目标行和起始列
      int smem_q_row = load_idx / (BK / vec_unit);
      int vec_idx_in_row = load_idx % (BK / vec_unit);
      int smem_q_col_start = vec_idx_in_row * vec_unit;

      // 2. 计算对应的全局内存读取地址
      int global_q_seq = block_q_seq_start + smem_q_row;  // 全局行
      int global_q_dim_vec_start =
          k_tile_start + smem_q_col_start;  // 全局列 (向量起点)

      // 3. 严格边界检查 和 加载/写入
      if (global_q_seq < seq_len) {
        // 行有效，检查列
        if (global_q_dim_vec_start + vec_unit - 1 < head_dim) {
          // --- 情况 A: 完整向量在边界内，直接向量化加载 ---
          Vec<T, vec_unit> vq;
          vq.f4 = *reinterpret_cast<const float4 *>(
              &q_head_ptr[global_q_seq * q_stride_seq +
                          global_q_dim_vec_start]);

// 水平写入 smemQ
#pragma unroll
          for (int i = 0; i < vec_unit; ++i) {
            // 但加上更安全)
            if (smem_q_col_start + i < BK) {
              smemQ[ph & 1][smem_q_row][smem_q_col_start + i] =
                  static_cast<float>(vq.t[i]);
            }
          }
        } else {
// --- 情况 B: 向量跨越 head_dim 边界，处理部分加载 ---
#pragma unroll
          for (int i = 0; i < vec_unit; ++i) {
            int current_smem_col = smem_q_col_start + i;
            if (current_smem_col < BK) {  // 检查 smem 列边界
              int current_global_dim = k_tile_start + current_smem_col;
              if (current_global_dim < head_dim) {  // 检查全局列边界
                // 只加载和写入在 head_dim 内的部分
                T element = q_head_ptr[global_q_seq * q_stride_seq +
                                       current_global_dim];
                smemQ[ph & 1][smem_q_row][current_smem_col] =
                    static_cast<float>(element);
              } else {
                // 超出 head_dim 的部分在 smem 中补零
                smemQ[ph & 1][smem_q_row][current_smem_col] = 0.0f;
              }
            }
          }
        }
      } else {
// --- 情况 C: 整行无效 (global_q_seq >= seq_len) ---
// 将 smemQ 中对应的向量位置全部置零
#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          int current_smem_col = smem_q_col_start + i;
          smemQ[ph & 1][smem_q_row][current_smem_col] = 0.0f;
        }
      }
    }  // 结束 Q 加载循环

// --- 开始加载 K 到 smemK ---
// 使用与 Q 加载完全相同的逻辑结构，但替换为 K 的参数

// --- 使用 Grid-Stride Loop (如果需要) 遍历 smemK 的行 ---
#pragma unroll(1)
    // 循环遍历 K tile 中的所有 *向量* (注意维度是 BN)
    for (int load_idx = tid; load_idx < BN * (BK / vec_unit);
         load_idx += blockDim.x) {
      // 1. 计算此向量在 smemK 中的目标行和起始列
      int smem_k_row = load_idx / (BK / vec_unit);
      int vec_idx_in_row = load_idx % (BK / vec_unit);
      int smem_k_col_start = vec_idx_in_row * vec_unit;

      // 2. 计算对应的全局内存读取地址 (使用 K 的起始块和 stride)
      int global_k_seq = block_k_seq_start + smem_k_row;  // <<< 使用 K 的起始行
      int global_k_dim_vec_start = k_tile_start + smem_k_col_start;

      // 3. 严格边界检查 和 加载/写入 (注意使用 total_seq_len)
      if (global_k_seq < total_seq_len) {  // <<< 使用 K 的总长度
        if (global_k_dim_vec_start + vec_unit - 1 < head_dim) {
          // --- 情况 A: 完整向量加载 ---
          Vec<T, vec_unit> vk;
          vk.f4 = *reinterpret_cast<const float4 *>(
              &k_head_ptr[global_k_seq * k_stride_seq +
                          global_k_dim_vec_start]);

// 水平写入 smemK
#pragma unroll
          for (int i = 0; i < vec_unit; ++i) {
            if (smem_k_col_start + i < BK) {
              smemK[ph & 1][smem_k_row][smem_k_col_start + i] =
                  static_cast<float>(vk.t[i]);
            }
          }
        } else {
// --- 情况 B: 部分向量加载 ---
#pragma unroll
          for (int i = 0; i < vec_unit; ++i) {
            int current_smem_col = smem_k_col_start + i;
            if (current_smem_col < BK) {
              int current_global_dim = k_tile_start + current_smem_col;
              if (current_global_dim < head_dim) {
                T element = k_head_ptr[global_k_seq * k_stride_seq +
                                       current_global_dim];
                smemK[ph & 1][smem_k_row][current_smem_col] =
                    static_cast<float>(element);
              } else {
                smemK[ph & 1][smem_k_row][current_smem_col] = 0.0f;
              }
            }
          }
        }
      } else {
// --- 情况 C: 整行无效 (for K) ---
#pragma unroll
        for (int i = 0; i < vec_unit; ++i) {
          int current_smem_col = smem_k_col_start + i;
          if (current_smem_col < BK) {
            smemK[ph & 1][smem_k_row][current_smem_col] = 0.0f;
          }
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {  // 在块内遍历 K 维度
#pragma unroll
      for (int i = 0; i < TM; ++i) {  // 遍历线程的 M 维度
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          // smemQ 中的行，thread_m_idx 是一个 tid 映射到的对象
          // 由于一个线程负责 TM*TN 的结果， 因为映射到的对象步长为 TM 或者 TN
          int smem_q_row = thread_m_idx * TM + i;
          int smem_k_col = thread_n_idx * TN + j;
          accum[i][j] +=
              smemQ[(ph) ^ 1][smem_q_row][k] * smemK[(ph) ^ 1][smem_k_col][k];
        }
      }
    }
  }
  __syncthreads();
  // --- 缩放结果并写回全局内存 ---
  // scores 布局: [seq_len, n_q_heads, total_seq_len]
  int scores_stride_seq = n_q_heads * total_seq_len;
  int scores_stride_head = total_seq_len;

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      // 计算全局输出坐标
      int global_scores_q_seq = thread_q_seq_start + i;
      int global_scores_k_seq = thread_k_seq_start + j;

      // 写回前进行边界检查
      if (global_scores_q_seq < seq_len &&
          global_scores_k_seq < total_seq_len) {
        // 计算输出偏移量:
        // scores[global_scores_q_seq][q_head_idx][global_scores_k_seq]
        int scores_offset = global_scores_q_seq * scores_stride_seq +
                            q_head_idx * scores_stride_head +
                            global_scores_k_seq;
        // 应用缩放并转换回输出类型 T
        scores[scores_offset] = static_cast<T>(accum[i][j] * scale);
      }
    }
  }
}
/**
 * 分块 GQA GEMM 核函数
 *
 * 输入布局:
 *   Q: [seq_len, n_q_heads, head_dim]
 *   K: [total_seq_len, n_kv_heads, head_dim]
 * 输出布局:
 *   scores: [seq_len, n_q_heads, total_seq_len]
 *
 * - 每个线程块计算特定 Q 头对应的输出 Scores 矩阵的一个 BM x BN 的块 (tile)。
 * - 每个线程计算其线程块输出块内的一个 TM x TN 的子块 (sub-tile)。
 * - 使用共享内存: BM x BK 用于 Q, BK x BN 用于 K^T。
 * - 以 BK 为步长遍历 head_dim 维度。
 */
template <typename T,  // 数据类型 (float, half, nv_bfloat16)
          int BM,      // 块块维度 M (Q 序列长度)
          int BN,      // 块块维度 N (K 序列长度)
          int BK,      // 块块维度 K (头维度)
          int TM,      // 线程块维度 M
          int TN       // 线程块维度 N
          >
__global__ void gqa_gemm_kernel_v1(
    const T *__restrict__ Q,  // 查询输入 [seq_len, n_q_heads, head_dim]
    const T *__restrict__ K,  // 键输入   [total_seq_len, n_kv_heads, head_dim]
    T *__restrict__ scores,   // 输出    [seq_len, n_q_heads, total_seq_len]
    int seq_len,              // Q 序列长度
    int head_dim,             // 每个头的维度
    // int batch_size,       // 对于 3D 输入不直接使用，假定为 1
    int n_q_heads,      // 查询头的数量
    int n_kv_heads,     // 键/值头的数量
    int ratio,          // n_q_heads / n_kv_heads 的比例
    int total_seq_len,  // K 序列长度 (可以不同于 seq_len)
    float scale         // 缩放因子 (1.0f / sqrtf(head_dim))
) {
  // 每一个块负责BM*BN大小的结果
  // 每个线程块负责部分的序号
  int block_k_seq_idx = blockIdx.x;
  int block_q_seq_idx = blockIdx.y;

  // z维度可计算批次（本实现暂忽略）
  int q_head_idx = blockIdx.z;

  // 确定对应的 KV 头索引
  int kv_head_idx = q_head_idx / ratio;

  // 计算每个线程块负责部分的具体起始位置
  int block_q_seq_start = block_q_seq_idx * BM;
  int block_k_seq_start = block_k_seq_idx * BN;

  // 块内线程索引计算
  // 使用一维索引便于映射
  int tid = threadIdx.x;

  constexpr int THREADS_PER_BLOCK_N = BN / TN;

  // 将线性线程 ID 映射到块的 2D 线程坐标
  int thread_m_idx = tid / THREADS_PER_BLOCK_N;
  int thread_n_idx = tid % THREADS_PER_BLOCK_N;

  // 计算此线程在其块输出块内的起始行/列
  int thread_q_seq_start = block_q_seq_start + thread_m_idx * TM;
  int thread_k_seq_start = block_k_seq_start + thread_n_idx * TN;

  __shared__ float smemQ[BM][BK];
  __shared__ float smemK[BN][BK];

  // 累加精度
  float accum[TM][TN] = {{0.0f}};

  // 计算得出 seqlen 维度对应的步长
  // head_dim 为 head 对应维度的步长
  int q_stride_seq = n_q_heads * head_dim;
  int q_stride_head = head_dim;

  int k_stride_seq = n_kv_heads * head_dim;
  int k_stride_head = head_dim;

  // 通过 head 维度的步长，索引得到每个 head 处理 的对象
  // 转换为 n_head seqlen dim 的维度安排
  // 相当于n_head为批次，计算一个二维矩阵的乘法
  const T *q_head_ptr = Q + q_head_idx * q_stride_head;
  const T *k_head_ptr = K + kv_head_idx * k_stride_head;
  // 根据BK遍历 MK NK 的 K 维度。
  for (int k_tile_start = 0; k_tile_start < head_dim; k_tile_start += BK) {
#pragma unroll

    for (int load_idx = tid; load_idx < BM * BK; load_idx += blockDim.x) {
      // 一个线程定位到一个特定行和特定列
      // 通过循环确保所有数据加载成功
      int load_row = load_idx / BK;
      int load_col = load_idx % BK;
      int global_q_seq = block_q_seq_start + load_row;
      int global_q_dim = k_tile_start + load_col;
      if (global_q_seq < seq_len && global_q_dim < head_dim) {
        smemQ[load_row][load_col] = static_cast<float>(
            q_head_ptr[global_q_seq * q_stride_seq + global_q_dim]);
      } else {
        smemQ[load_row][load_col] = 0.0f;
      }
    }
#pragma unroll
    for (int load_idx = tid; load_idx < BN * BK; load_idx += blockDim.x) {
      // 一个线程定位到一个特定行和特定列
      // 通过循环确保所有数据加载成功
      int load_row = load_idx / BK;
      int load_col = load_idx % BK;
      int global_k_seq = block_k_seq_start + load_row;
      int global_k_dim = k_tile_start + load_col;
      if (global_k_seq < total_seq_len && global_k_dim < head_dim) {
        smemK[load_row][load_col] = static_cast<float>(
            k_head_ptr[global_k_seq * k_stride_seq + global_k_dim]);
      } else {
        smemK[load_row][load_col] = 0.0f;
      }
    }
    __syncthreads();

// --- 使用共享内存块计算矩阵乘法 ---
#pragma unroll
    for (int k = 0; k < BK; ++k) {  // 在块内遍历 K 维度
#pragma unroll
      for (int i = 0; i < TM; ++i) {  // 遍历线程的 M 维度
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          // smemQ 中的行，thread_m_idx 是一个 tid 映射到的对象
          // 由于一个线程负责 TM*TN 的结果， 因为映射到的对象步长为 TM 或者 TN
          int smem_q_row = thread_m_idx * TM + i;
          int smem_k_col = thread_n_idx * TN + j;
          accum[i][j] += smemQ[smem_q_row][k] * smemK[smem_k_col][k];
        }
      }
    }
    __syncthreads();
  }

  // --- 缩放结果并写回全局内存 ---
  // scores 布局: [seq_len, n_q_heads, total_seq_len]
  int scores_stride_seq = n_q_heads * total_seq_len;
  int scores_stride_head = total_seq_len;

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      // 计算全局输出坐标
      int global_scores_q_seq = thread_q_seq_start + i;
      int global_scores_k_seq = thread_k_seq_start + j;

      // 写回前进行边界检查
      if (global_scores_q_seq < seq_len &&
          global_scores_k_seq < total_seq_len) {
        // 计算输出偏移量:
        // scores[global_scores_q_seq][q_head_idx][global_scores_k_seq]
        int scores_offset = global_scores_q_seq * scores_stride_seq +
                            q_head_idx * scores_stride_head +
                            global_scores_k_seq;
        // 应用缩放并转换回输出类型 T
        scores[scores_offset] = static_cast<T>(accum[i][j] * scale);
      }
    }
  }
}

/**
 * 分块 GQA GEMM 核函数的启动器
 *
 * 配置并启动 gqa_gemm_tiled_kernel_3d。
 * 输入布局:
 *   Q: [seq_len, n_q_heads, head_dim]
 *   K: [total_seq_len, n_kv_heads, head_dim]
 * 输出布局:
 *   scores: [seq_len, n_q_heads, total_seq_len]
 */
template <typename T>
void launch_gqa_gemm(
    const Tensor<T> &Q,   // 查询 [seq_len, n_q_heads, head_dim]
    const Tensor<T> &K,   // 键   [total_seq_len, n_kv_heads, head_dim]
    Tensor<T> &scores,    // 输出 [seq_len, n_q_heads, total_seq_len]
    cudaStream_t stream)  // CUDA 流
{
  // --- 参数提取与验证 ---
  const auto &q_sizes = Q.sizes();
  const auto &k_sizes = K.sizes();
  const auto &scores_sizes = scores.sizes();

  // 基本检查是否为 3 维 - 如果 .sizes() 行为不同则需调整
  if (q_sizes.size() != 3 || k_sizes.size() != 3 || scores_sizes.size() != 3) {
    throw std::runtime_error(
        "输入/输出张量必须是基于 [seq, heads, "
        "dim/seq_k] 布局的 3 维张量");
  }

  // 根据指定的 3D 布局提取维度
  int seq_len = q_sizes[0];
  int n_q_heads = q_sizes[1];
  int head_dim = q_sizes[2];

  int total_seq_len = k_sizes[0];
  int n_kv_heads = k_sizes[1];
  if (k_sizes[2] != head_dim) {
    throw std::runtime_error("Q (" + std::to_string(head_dim) + ") 和 K (" +
                             std::to_string(k_sizes[2]) +
                             ") 之间的头维度不匹配");
  }

  // 在此上下文中，3D 张量的批处理大小隐式为 1
  // int batch_size = 1; // 核函数索引不再显式需要它

  // 检查 GQA 约束
  if (n_q_heads == 0 || n_kv_heads == 0) {
    throw std::runtime_error("头的数量不能为零。");
  }
  if (n_q_heads % n_kv_heads != 0) {
    throw std::runtime_error("n_q_heads (" + std::to_string(n_q_heads) +
                             ") 必须能被 n_kv_heads (" +
                             std::to_string(n_kv_heads) + ") 整除");
  }
  int ratio = n_q_heads / n_kv_heads;

  // 检查 scores 的形状
  if (scores_sizes[0] != seq_len || scores_sizes[1] != n_q_heads ||
      scores_sizes[2] != total_seq_len) {
    throw std::runtime_error(
        "Scores 张量形状不匹配。预期 [" + std::to_string(seq_len) + ", " +
        std::to_string(n_q_heads) + ", " + std::to_string(total_seq_len) +
        "], 得到 [" + std::to_string(scores_sizes[0]) + ", " +
        std::to_string(scores_sizes[1]) + ", " +
        std::to_string(scores_sizes[2]) + "]");
  }

  // 处理维度为零的情况，避免启动空的网格
  if (seq_len == 0 || total_seq_len == 0 || n_q_heads == 0 || head_dim == 0) {
    throw std::runtime_error("gqa_gemm数据有误");
  }

  // --- 分块参数选择 ---
  // 示例配置 (根据性能分析和目标 GPU 进行调整):
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int TM = 2;
  constexpr int TN = 2;

  // 一个线程负责M维度上的多少数据
  constexpr int THREADS_M = BM / TM;
  // 一个线程负责N维度上的多少数据
  constexpr int THREADS_N = BN / TN;
  constexpr int THREADS_PER_BLOCK = THREADS_M * THREADS_N;
  static_assert(THREADS_PER_BLOCK > 0 && THREADS_PER_BLOCK <= 1024,
                "每个块的线程数无效");

  static_assert(BM % TM == 0, "BM 必须能被 TM 整除");
  static_assert(BN % TN == 0, "BN 必须能被 TN 整除");

  // --- 核函数启动配置 ---
  dim3 blockDim(THREADS_PER_BLOCK);            // 1D 块维度
  dim3 gridDim((total_seq_len + BN - 1) / BN,  // K 序列维度上的块数 (grid.x)
               (seq_len + BM - 1) / BM,        // Q 序列维度上的块数 (grid.y)
               n_q_heads                       // Q 头维度上的块数 (grid.z)
  );

  // 预计算缩放因子
  static float scale = (head_dim > 0)
                           ? (1.0f / sqrtf(static_cast<float>(head_dim)))
                           : 1.0f;  // 避免 head_dim 为 0 时除以零

  // --- 启动核函数 ---
  // 使用新的核函数名称并传递与 3D 布局一致的参数
  gqa_gemm_kernel_v2<T, BM, BN, BK, TM, TN><<<gridDim, blockDim, 0, stream>>>(
      Q.data_ptr(), K.data_ptr(), scores.data_ptr(), seq_len, head_dim,
      // batch_size, // 核函数不再需要
      n_q_heads, n_kv_heads, ratio, total_seq_len, scale);

  // 核函数启动后检查 CUDA 错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // 在错误消息中包含 grid/block 维度通常很有帮助
    std::string error_msg = "launch_gqa_gemm 中的 CUDA 错误 (Grid: ";
    error_msg += std::to_string(gridDim.x) + "," + std::to_string(gridDim.y) +
                 "," + std::to_string(gridDim.z);
    error_msg += ", Block: " + std::to_string(blockDim.x) +
                 "): " + std::string(cudaGetErrorString(err));
    throw std::runtime_error(error_msg);
  }
}

// --- 显式模板实例化 ---
// 实例化新的 3D 启动器函数。

template void launch_gqa_gemm<float>(const Tensor<float> &Q,
                                     const Tensor<float> &K,
                                     Tensor<float> &scores,
                                     cudaStream_t stream);

template void launch_gqa_gemm<half>(const Tensor<half> &Q,
                                    const Tensor<half> &K, Tensor<half> &scores,
                                    cudaStream_t stream);

template void launch_gqa_gemm<nv_bfloat16>(const Tensor<nv_bfloat16> &Q,
                                           const Tensor<nv_bfloat16> &K,
                                           Tensor<nv_bfloat16> &scores,
                                           cudaStream_t stream);

}  // namespace cuda_OP

#endif  // CUDA_GQA_GEMM_TILED_CUH

// #ifndef CUDA_GQA_GEMM_CUH
// #define CUDA_GQA_GEMM_CUH

// #include <cuda_bf16.h> // 提供 nv_bfloat16, __float2bfloat16 等
// #include <cuda_fp16.h> // 提供 half, __float2half, __half2float, __hadd,
// half2 等 #include <stdint.h>    // 提供 int64_t 等标准整数类型

// #include <cstdio>    // 提供 fprintf, stderr (用于可能的错误输出)
// #include <stdexcept> // 提供 std::runtime_error, std::invalid_argument
// #include <string>    // 提供 std::string, std::to_string

// #include "cuda/legacy/legacy_cuda_api.cuh"

// namespace cuda_OP
// {
//     /**
//      * 支持GQA的批处理GEMM内核
//      *
//      * 这个内核处理批处理矩阵乘法，支持GQA等场景中的索引映射
//      * 每个线程块负责计算一个批次中的一个输出块
//      * 使用共享内存优化，提高内存访问效率
//      */
//     template <typename T, int BLOCK_SIZE = 16>
//     __global__ void gqa_gemm_kernel(
//         const T *Q,     // 查询矩阵 [batch_size, seq_len, n_q_heads,
//         head_dim] const T *K,     // 键矩阵 [batch_size, total_seq_len,
//         n_kv_heads, head_dim] T *scores,      // 输出分数矩阵 [batch_size,
//         seq_len, n_q_heads, total_seq_len] int seq_len,    // 序列长度 int
//         head_dim,   // 头维度 int batch_size, // 批次大小 int n_q_heads,  //
//         Q头数量 int n_kv_heads, // KV头数量 int ratio,      //
//         Q头与KV头的比例 (n_q_heads / n_kv_heads) int total_seq_len)
//     {
//         int batch_idx = blockIdx.z / n_q_heads;
//         int q_head_idx = blockIdx.z % n_q_heads;

//         // 计算对应的KV头索引
//         int kv_head_idx = q_head_idx / ratio;

//         // 计算当前线程负责的输出元素索引
//         int seq_pos = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//         int t_seq_pos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//         int scores_offset = batch_idx * total_seq_len * n_q_heads * seq_len +
//                             seq_pos * total_seq_len * n_q_heads +
//                             q_head_idx * total_seq_len + t_seq_pos;
//         const T *q = Q + batch_idx * n_q_heads * head_dim * seq_len +
//                      seq_pos * head_dim * n_q_heads + q_head_idx * head_dim;
//         const T *k = K + batch_idx * n_kv_heads * head_dim * total_seq_len +
//                      t_seq_pos * head_dim * n_kv_heads + kv_head_idx *
//                      head_dim;
//         float sum = 0.0f;
//         int tid_t = threadIdx.x;
//         int tid = threadIdx.y;
//         int tile = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
//         __shared__ float smemQ[16][16];
//         __shared__ float smemK[16][16];
//         for (int i = 0; i < tile; i++)
//         {
//             if (tid_t + i * BLOCK_SIZE < head_dim && tid + i * BLOCK_SIZE <
//             head_dim)
//             {
//                 smemQ[tid][tid_t] = static_cast<float>(q[tid_t + i *
//                 BLOCK_SIZE]); smemK[tid_t][tid] = static_cast<float>(k[tid +
//                 i * BLOCK_SIZE]);
//             }
//             __syncthreads();
//             for (int j = 0; j < BLOCK_SIZE; j++)
//             {
//                 sum += smemQ[tid][j] * smemK[tid_t][j];
//             }
//             __syncthreads();
//         }

//         sum *= rsqrtf(static_cast<float>(head_dim));

//         if (seq_pos < seq_len && t_seq_pos < total_seq_len)
//         {
//             scores[scores_offset] = static_cast<T>(sum);
//         }
//     }

//     /**
//      * 支持GQA的GEMM计算包装函数
//      *
//      * 这个函数专门处理GQA场景中的矩阵乘法，一个KV头对应多个Q头
//      * 本项目不支持batchsize 但算子先尝试适配一下各种情况
//      */
//     template <typename T>
//     void launch_gqa_gemm(
//         const Tensor<T> &Q, // 查询张量 [batch_size, n_q_heads, seq_len,
//         head_dim] const Tensor<T> &K, // 键张量 [batch_size, n_kv_heads,
//         seq_len, head_dim] Tensor<T>
//             &scores, // 输出分数张量 [batch_size, n_q_heads, seq_len,
//             seq_len]

//         cudaStream_t stream) // CUDA流
//     {
//         int batch_size = 1;

//         // 根据张量维度提取参数
//         // 张量布局为 [seq_len, n_heads, head_dim]
//         int seq_len = Q.sizes()[0];
//         int n_q_heads = Q.sizes()[1];
//         int head_dim = Q.sizes()[2];
//         int total_seq_len = K.sizes()[0];
//         int n_kv_heads = K.sizes()[1];

//         // 计算Q头与KV头的比例
//         if (n_q_heads % n_kv_heads != 0)
//         {
//             throw std::runtime_error("n_q_heads must be divisible by
//             n_kv_heads");
//         }
//         int ratio = n_q_heads / n_kv_heads;

//         // 检查scores的形状
//         if (scores.sizes()[0] != seq_len || scores.sizes()[1] != n_q_heads ||
//             scores.sizes()[2] != total_seq_len)
//         {
//             throw std::runtime_error("scores tensor shape mismatch");
//         }

//         // 配置内核启动参数
//         constexpr int BLOCK_SIZE = 16; // 可以根据GPU架构调整

//         // 计算网格维度
//         // x维度：K序列长度的块数
//         // y维度：Q序列长度的块数
//         // z维度：批次 * Q头数
//         dim3 grid(
//             (total_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, // x:
//             K序列长度的块数 (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE,       //
//             y: Q序列长度的块数 batch_size * n_q_heads // z: 批次 * Q头数
//         );

//         // 计算块维度
//         // 使用2D块，每个线程处理一个输出元素
//         dim3 block(BLOCK_SIZE, BLOCK_SIZE);

//         // 启动内核
//         //
//         注意：这里我们传递了所有必要的参数，包括序列长度、头维度、批次大小等
//         gqa_gemm_kernel<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(
//             Q.data_ptr(),      // 查询矩阵
//             K.data_ptr(),      // 键矩阵
//             scores.data_ptr(), // 输出分数矩阵
//             seq_len,           // Q序列长度
//             head_dim,          // 头维度
//             batch_size,        // 批次大小
//             n_q_heads,         // Q头数量
//             n_kv_heads,        // KV头数量
//             ratio,             // Q头与KV头的比例
//             total_seq_len      // K序列长度
//         );

//         // 检查CUDA错误
//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess)
//         {
//             throw std::runtime_error("CUDA error in launch_gqa_gemm: " +
//                                      std::string(cudaGetErrorString(err)));
//         }
//     }

//     // 显式实例化模板函数，以便在其他文件中使用
//     template void launch_gqa_gemm<float>(const Tensor<float> &Q,
//                                          const Tensor<float> &K,
//                                          Tensor<float> &scores,
//                                          cudaStream_t stream);

//     template void launch_gqa_gemm<half>(const Tensor<half> &Q,
//                                         const Tensor<half> &K, Tensor<half>
//                                         &scores, cudaStream_t stream);

//     template void launch_gqa_gemm<nv_bfloat16>(const Tensor<nv_bfloat16> &Q,
//                                                const Tensor<nv_bfloat16> &K,
//                                                Tensor<nv_bfloat16> &scores,
//                                                cudaStream_t stream);

// } // namespace cuda_OP

// #endif // CUDA_GQA_GEMM_CUH