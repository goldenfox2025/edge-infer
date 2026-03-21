#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>  // 提供 __nv_bfloat16 定义
#include <cuda_runtime.h>
#include <curand_kernel.h>  // 用于设备端随机数生成
#include <float.h>
#include <math.h>

#include <algorithm>  // min
#include <cstdio>     // printf
#include <cstring>    // memcpy
#include <iostream>
#include <stdexcept>
#include <vector>

#include "inference.hpp"
#include "tensor.hpp"
#define CUTLASS_CHECK(status)                                                                                        \
    {                                                                                                                \
        cutlass::Status error = status;                                                                              \
        if (error != cutlass::Status::kSuccess) {                                                                    \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                      \
        }                                                                                                            \
    }

// CUDA Error Check Macro
#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));                                          \
        }                                                                                               \
    } while (0)
template <typename T, int N>
union Vec {
    float4 f4;  // 实际载入 16 字节数据
    T t[N];     // 重解释为 N 个 T 类型元素
};

template <typename T, int N>
union Vec_2 {
    float2 f2;  // 实际载入 8 字节数据
    T t[N];     // 重解释为 N 个 T 类型元素
};

namespace cuda_OP {

// 定义支持的数据类型别名
template <typename T, typename ScaleType = float>
void matmul_quantized_gemv(           // Renamed wrapper
    const Tensor<T> &input,           // [M, K]
    const Tensor<int32_t> &qweight,   // [N, K/8] <- Expected GEMV layout
    const Tensor<ScaleType> &scales,  // [N, G_padded] (G_padded >= G)
    const Tensor<int32_t> &zeros,     // [N, G/8]
    int group_size,
    Tensor<T> *output,  // [M, N] - DIMS NOT CHECKED HERE
    cudaStream_t stream = nullptr, const Tensor<T> *bias = nullptr);
using nvbf16 = __nv_bfloat16;
void init_curand(curandState *d_states, unsigned long long seed, int offset, cudaStream_t stream = nullptr);
// 工具函数声明
void checkCudaError(cudaError_t err);

template <typename T>
void add_rms(Tensor<T> *output, Tensor<T> *input, const Tensor<T> *add_, const Tensor<T> *weight, float eps,
             cudaStream_t stream = nullptr);

// 从 embedding_table 中根据 input 索引取值写入 output
template <typename T>
void gather(Tensor<T> *output, const Tensor<uint32_t> *input, const Tensor<T> *embedding_table,
            cudaStream_t stream = nullptr);

// 均方根归一化算子：output = rms_norm(input, weight, eps)
template <typename T>
void rms_norm(Tensor<T> *output, const Tensor<T> *input, const Tensor<T> *weight, float eps,
              cudaStream_t stream = nullptr);

template <typename T, typename AccT = float>
void launch_gemv_scores(const T *x, const T *y, AccT *dst, const int channel_size, const int channel_ratio,
                        const int row_size, const int stride_channel_x, const int stride_channel_y,
                        const int stride_channel_dst, cudaStream_t stream = nullptr);

template <typename T>
void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> *C, cudaStream_t stream = nullptr,
            const Tensor<T> *bias = nullptr, int use_ = 1);

// rope 算子，用于位置编码
template <typename T>
void rope(Tensor<T> *tensor, size_t current_pos, float theta, cudaStream_t stream = nullptr);

// CUDA图优化版本：使用设备端固定内存的offset
template <typename T>
void rope_with_device_offset(Tensor<T> *tensor, const size_t *d_offset, float theta, cudaStream_t stream = nullptr);

// 使用预计算sin/cos缓存的RoPE版本：接受设备端offset和预计算的sin/cos缓存

template <typename T>
void rope_with_precomputed_cache(Tensor<T> *tensor, const size_t *d_offset, const Tensor<float> *sin_cos_cache,
                                 cudaStream_t stream = nullptr, int *offset_array = nullptr, int layer_index = 0,
                                 int n_layers = 28, int *pingpong_index = nullptr);

template <typename T>
void gemv_qkv(const Tensor<T> *A, const Tensor<T> *B, Tensor<T> *q, Tensor<T> *k, Tensor<T> *v, const Tensor<T> *bias,
              int *offset_array, int layer_index, size_t Q_len, size_t K_len, size_t V_len,
              cudaStream_t stream = nullptr, int n_layers = 28, int *pingpong_index = nullptr);

// Fused GEMV QKV + RoPE operation: compute Q,K,V and apply RoPE to Q,K in a single kernel
template <typename T>
void gemv_qkv_rope(const Tensor<T> *A, const Tensor<T> *B, Tensor<T> *q, Tensor<T> *k, Tensor<T> *v,
                   const Tensor<T> *bias, const size_t *d_offset, const Tensor<float> *sin_cos_cache, int *offset_array,
                   int layer_index, size_t Q_len, size_t K_len, size_t V_len, size_t n_heads, size_t n_kv_heads,
                   size_t head_dim, cudaStream_t stream = nullptr, int n_layers = 28, int *pingpong_index = nullptr);

// Fused GEMV MLP: compute gate_proj, up_proj, silu and multiply in a single kernel
template <typename T>
void gemv_mlp_fused(const Tensor<T> *hidden_states, const Tensor<T> *merged_mlp_weight, Tensor<T> *output,
                    cudaStream_t stream);

// Fused RoPE and KV Cache write operation for prefill.
// Applies RoPE to Q (in-place) and K, then writes K and V to their respective layer caches.
template <typename T>
void rope_kv_fused(Tensor<T> &q_tensor,          // Input/Output Q tensor, shape [seq_len, n_heads, head_dim]
                   const Tensor<T> &k_tensor,    // Input K tensor, shape [seq_len, n_kv_heads, head_dim]
                   const Tensor<T> &v_tensor,    // Input V tensor, shape [seq_len, n_kv_heads, head_dim]
                   Tensor<T> &k_cache_layer,     // K cache for a single layer, shape [max_len, n_kv_heads * head_dim]
                   Tensor<T> &v_cache_layer,     // V cache for a single layer, shape [max_len, n_kv_heads * head_dim]
                   const size_t *d_rope_offset,  // Device pointer to the past sequence length
                   const Tensor<float> &sin_cos_cache,  // Precomputed sin/cos table
                   cudaStream_t stream = nullptr);
template <typename T>
void softmax(Tensor<T> *output, const Tensor<T> *input, int dim, bool mask = true, int offset = 0,
             cudaStream_t stream = nullptr);
template <typename T>
void rope_qkv_precompute_with_write_kv(
    Tensor<T> *q_tensor,                             // 输入输出的Q张量（可选，如果为nullptr则不处理Q）
    const Tensor<T> &k_input,                        // 输入的K张量 [seq_len, n_kv_heads, head_dim]
    const Tensor<T> &v_input,                        // 输入的V张量 [seq_len, n_kv_heads, head_dim]
    const std::vector<Tensor<T> *> &k_cache_slices,  // K cache切片数组（用于获取基地址和计算偏移）
    const std::vector<Tensor<T> *> &v_cache_slices,  // V cache切片数组
    const size_t *d_offset,                          // RoPE offset
    const Tensor<float> *sin_cos_cache,              // 预计算的sin/cos缓存
    cudaStream_t stream                              // CUDA stream
);                                                   // silu 激活函数算子
template <typename T>
void silu(Tensor<T> *output, const Tensor<T> *input, cudaStream_t stream = nullptr);
template <typename T>
void silu_multiply(Tensor<T> *output, const Tensor<T> *input, const Tensor<T> *input2, cudaStream_t stream = nullptr);
// 支持步长访问的silu激活函数算子
// input可以是非连续张量，output必须是连续张量
template <typename T>
void silu_strided(Tensor<T> *output, const Tensor<T> *input, cudaStream_t stream = nullptr);

// 逐元素乘法算子
template <typename T>
void multiply(Tensor<T> *output, const Tensor<T> *A, const Tensor<T> *B, cudaStream_t stream = nullptr);

// 支持步长访问的逐元素乘法算子
// A和B可以是非连续张量，output必须是连续张量
template <typename T>
void multiply_strided(Tensor<T> *output, const Tensor<T> *A, const Tensor<T> *B, cudaStream_t stream = nullptr);

// 逐元素加法算子
template <typename T>
void add(Tensor<T> *output, Tensor<T> *A, Tensor<T> *B, cudaStream_t stream = 0);

template <typename T>
uint32_t *sample(Tensor<T> &&input, float temperature, float top_p, size_t top_k, curandState *d_states,
                 cudaStream_t stream = nullptr);

// 采样函数的变体，将结果写入指定的GPU内存位置
template <typename T>
void sample_to_fixed(Tensor<T> &&input, uint32_t *output_ptr, float temperature, float top_p, size_t top_k,
                     curandState *d_states, cudaStream_t stream = nullptr);

// 高效采样函数的变体，避免全量排序，将token和概率写入指定的GPU内存位置
template <typename T>
void fast_sample_to_fixed(Tensor<T> &&input, uint32_t *output_ptr, float *prob_ptr, float temperature, float top_p,
                          size_t top_k, curandState *d_states, cudaStream_t stream = nullptr);

// 批量采样函数的变体，将结果写入指定的GPU内存位置数组
template <typename T>
void sample_batch_to_fixed(Tensor<T> &&logits, uint32_t *output_ptr, float temperature, float top_p, size_t top_k,
                           curandState *d_states, cudaStream_t stream = nullptr);

// 计算注意力分数（多头注意力机制相关算子）
template <typename T>
void compute_attention_scores(const Tensor<T> &Q, const Tensor<T> &K, size_t n_q_h, size_t dqkv, Tensor<T> &att_scores,
                              size_t n_kv_h, cudaStream_t stream = nullptr);

// 计算注意力输出
template <typename T>
void compute_att_output(const Tensor<T> &att_probs, const Tensor<T> &V, size_t n_q_h, size_t dqkv,
                        Tensor<T> &att_output, size_t n_kv_h, cudaStream_t stream = nullptr);

// flahattention实现
template <typename T>
void flash_attention(Tensor<T> &Q, const Tensor<T> &&K1, const Tensor<T> &&K2, const Tensor<T> &&K3,
                     const Tensor<T> &&V1, const Tensor<T> &&V2, const Tensor<T> &&V3, Tensor<T> &att_output1,
                     Tensor<T> &att_output2, Tensor<T> &att_output3, cudaStream_t stream = nullptr);

// prefill 版本：计算注意力分数
template <typename T>
void compute_attention_scores_prefill(const Tensor<T> &Q, const Tensor<T> &K, Tensor<T> &att_scores, size_t dqkv,
                                      cudaStream_t stream = nullptr);

// prefill 版本：计算注意力输出
template <typename T>
void compute_att_output_prefill(const Tensor<T> &att_probs, const Tensor<T> &V, Tensor<T> &att_output, size_t n_q_h,
                                size_t dqkv, size_t total_seq_len, size_t n_kv_h, cudaStream_t stream = nullptr);

// 支持GQA的GEMM计算，用于prefill阶段的注意力分数计算
template <typename T>
void launch_gqa_gemm(const Tensor<T> &Q,  // 查询张量 [batch_size, n_q_heads, seq_len, head_dim]
                     const Tensor<T> &K,  // 键张量 [batch_size, n_kv_heads, seq_len, head_dim]
                     Tensor<T> &scores,   // 输出分数张量 [batch_size, n_q_heads, seq_len, seq_len]

                     cudaStream_t stream = nullptr);  // CUDA流

template <typename T>
void gather_fa(const Tensor<T> &T1, const Tensor<T> &T2, const Tensor<T> &T3, Tensor<T> &T5,
               cudaStream_t stream = nullptr);

// 可变分支数量的gather_fa实现
template <typename T>
void gather_fa_variable(const std::vector<Tensor<T>> &inputs, Tensor<T> &output, cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 1分支
template <typename T>
void gather_fa_specialized_1branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 2分支
template <typename T>
void gather_fa_specialized_2branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 3分支
template <typename T>
void gather_fa_specialized_3branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 4分支
template <typename T>
void gather_fa_specialized_4branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 特化版本的gather_fa实现 - 5分支
template <typename T>
void gather_fa_specialized_5branch(const std::vector<Tensor<T>> &inputs, Tensor<T> &output,
                                   cudaStream_t stream = nullptr);

// 可变分支数量的flash_attention实现
template <typename T>
void flash_attention_variable(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                              const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                              cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 1分支
template <typename T>
void flash_attention_specialized_1branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 2分支
template <typename T>
void flash_attention_specialized_2branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 3分支
template <typename T>
void flash_attention_specialized_3branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 4分支
template <typename T>
void flash_attention_specialized_4branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 特化版本的flash_attention实现 - 5分支
template <typename T>
void flash_attention_specialized_5branch(Tensor<T> &Q, const std::vector<Tensor<T>> &K_slices,
                                         const std::vector<Tensor<T>> &V_slices, std::vector<Tensor<T>> &outputs,
                                         cudaStream_t stream = nullptr);

// 包装函数：根据KV缓存长度动态选择分支数量
template <typename T>
void dynamic_flash_attention_wrapper(Tensor<T> &Q, const Tensor<T> &total_K, const Tensor<T> &total_V,
                                     Tensor<T> &att_output, int n_kv_heads, cudaStream_t stream = nullptr);

// CUDA图优化版本：使用固定内存地址和分段信息的flash attention
template <typename T>
void flash_attention_graph_fixed(Tensor<T> &Q, const Tensor<T> &total_K, const Tensor<T> &total_V, T **d_output_ptrs,
                                 int *d_segment_info, int n_kv_heads, cudaStream_t stream = nullptr,
                                 int *pingpong_index = nullptr);

// CUDA图优化版本：使用固定内存地址的gather_fa
template <typename T>
void gather_fa_graph_fixed(T **d_input_ptrs, Tensor<T> &output, int *d_segment_info, cudaStream_t stream = nullptr);

// AWQ量化矩阵乘法
template <typename T, typename ScaleType = float>
void matmul_quantized(const Tensor<T> &input, const Tensor<int32_t> &qweight, const Tensor<ScaleType> &scales,
                      const Tensor<int32_t> &zeros, int group_size, Tensor<T> *output, cudaStream_t stream = nullptr,
                      const Tensor<T> *bias = nullptr);

// 生成随机值数组
void generate_random_values(float *values, size_t count, curandState *states, cudaStream_t stream = nullptr);

// 获取指定token在logits中的概率
template <typename T>
float get_token_probability(const Tensor<T> &logits, int position, uint32_t token_id, cudaStream_t stream = nullptr);

// 从logits中采样一个token并返回其概率
template <typename T>
std::pair<uint32_t, float> sample_with_prob(Tensor<T> &&logits, float temperature, float top_p, size_t top_k,
                                            curandState *states, cudaStream_t stream = nullptr);

// 采样函数的变体，将token和概率写入指定的GPU内存位置
template <typename T>
void sample_to_fixed_with_prob(Tensor<T> &&input, uint32_t *token_ptr, float *prob_ptr, float temperature, float top_p,
                               size_t top_k, curandState *states, cudaStream_t stream = nullptr);

// 批量采样函数的变体，将token和概率写入指定的GPU内存位置
template <typename T>
void sample_batch_to_fixed_with_prob(Tensor<T> &&logits, uint32_t *token_ptr, float *prob_ptr, float temperature,
                                     float top_p, size_t top_k, curandState *states, cudaStream_t stream = nullptr);

// Flash Attention Prefill: 完整的flash attention实现，用于prefill阶段
template <typename T>
void flash_attention_prefill(const Tensor<T> &Q,  // Query张量 [seq_len, n_heads, head_dim]
                             const Tensor<T> &K,  // Key张量 [total_seq_len, n_kv_heads, head_dim]
                             const Tensor<T> &V,  // Value张量 [total_seq_len, n_kv_heads, head_dim]
                             Tensor<T> &output,   // 输出张量 [seq_len, n_heads, head_dim]
                             int n_heads,         // Query头数
                             int n_kv_heads,      // Key/Value头数
                             int head_dim,        // 头维度
                             int seq_len,         // Query序列长度
                             int total_seq_len,   // Key/Value总序列长度
                             int offset,          // Q在整个序列中的起始偏移量
                             cudaStream_t stream = nullptr);

// WMMA-optimized attention score computation (Q @ K^T)
template <typename T>
void compute_attention_scores_prefill_wmma(const Tensor<T> &Q, const Tensor<T> &K, Tensor<T> &att_scores,
                                           size_t head_dim, cudaStream_t stream = nullptr);

// WMMA-optimized attention output computation (attention_scores @ V)
template <typename T>
void compute_att_output_prefill_wmma(const Tensor<T> &att_probs, const Tensor<T> &V, Tensor<T> &att_output,
                                     size_t n_q_heads, size_t head_dim, size_t total_seq_len, size_t n_kv_heads,
                                     cudaStream_t stream = nullptr);

// cuBLAS-based GQA optimization for large-scale attention computation
template <typename T>
void compute_attention_scores_prefill_cublas(const Tensor<T> &Q, const Tensor<T> &K, Tensor<T> &att_scores,
                                             size_t head_dim, cudaStream_t stream = nullptr);

template <typename T>
void compute_att_output_prefill_cublas(const Tensor<T> &att_probs, const Tensor<T> &V, Tensor<T> &att_output,
                                       size_t n_q_heads, size_t head_dim, size_t total_seq_len, size_t n_kv_heads,
                                       cudaStream_t stream = nullptr);

// Adaptive functions that automatically choose the best implementation
template <typename T>
void compute_attention_scores_prefill_adaptive(const Tensor<T> &Q, const Tensor<T> &K, Tensor<T> &att_scores,
                                               size_t head_dim, cudaStream_t stream = nullptr);

template <typename T>
void compute_att_output_prefill_adaptive(const Tensor<T> &att_probs, const Tensor<T> &V, Tensor<T> &att_output,
                                         size_t n_q_heads, size_t head_dim, size_t total_seq_len, size_t n_kv_heads,
                                         cudaStream_t stream = nullptr);

}  // namespace cuda_OP
