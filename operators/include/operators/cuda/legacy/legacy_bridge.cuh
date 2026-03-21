#pragma once

#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>

#include "tensor.hpp"

namespace op::legacy {

template <typename T>
void gather(Tensor<T>* output, const Tensor<uint32_t>* input,
            const Tensor<T>* embedding_table, cudaStream_t stream = nullptr);

template <typename T>
void compute_attention_scores_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                                      Tensor<T>& att_scores, size_t head_dim,
                                      cudaStream_t stream = nullptr);

template <typename T>
void compute_attention_output_prefill(const Tensor<T>& att_scores,
                                      const Tensor<T>& V,
                                      Tensor<T>& att_output, size_t n_heads,
                                      size_t head_dim, size_t total_seq_len,
                                      size_t n_kv_heads,
                                      cudaStream_t stream = nullptr);

template <typename T>
void dynamic_flash_attention(Tensor<T>& Q, const Tensor<T>& K,
                             const Tensor<T>& V, Tensor<T>& output,
                             int n_kv_heads, cudaStream_t stream = nullptr);

template <typename T>
void flash_attention_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                             const Tensor<T>& V, Tensor<T>& output,
                             int n_heads, int n_kv_heads, int head_dim,
                             int seq_len, int total_seq_len, int offset,
                             cudaStream_t stream = nullptr);

template <typename T>
void softmax(Tensor<T>* output, const Tensor<T>* input, int dim, bool mask,
             int offset, cudaStream_t stream = nullptr);

template <typename T>
void matmul_quantized_gemv(const Tensor<T>& input,
                           const Tensor<int32_t>& qweight,
                           const Tensor<T>& scales,
                           const Tensor<int32_t>& qzeros, int group_size,
                           Tensor<T>* output, cudaStream_t stream = nullptr,
                           const Tensor<T>* bias = nullptr);

template <typename T>
void gemv_qkv_rope(Tensor<T>* hidden_states,
                   const Tensor<T>* merged_qkv_weight, Tensor<T>* q_buf,
                   Tensor<T>* k_buf, Tensor<T>* v_buf,
                   const Tensor<T>* merged_qkv_bias, size_t* d_rope_offset,
                   const Tensor<float>* rope_sin_cos_cache,
                   int* d_offset_array, int layer_idx, int q_dim, int k_dim,
                   int v_dim, int n_heads, int n_kv_heads, int head_dim,
                   cudaStream_t stream = nullptr, int n_layers = 0,
                   int* pingpong = nullptr);

template <typename T>
void rope_with_precomputed_cache(Tensor<T>* tensor, const size_t* d_offset,
                                 const Tensor<float>* rope_sin_cos_cache,
                                 cudaStream_t stream = nullptr,
                                 int* d_offset_array = nullptr,
                                 int layer_idx = 0, int n_layers = 0,
                                 int* pingpong = nullptr);

template <typename T>
void flash_attention_graph_fixed(Tensor<T>& q, const Tensor<T>& k,
                                 const Tensor<T>& v, T** d_output_ptrs,
                                 int* d_segment_info, int n_kv_heads,
                                 cudaStream_t stream = nullptr,
                                 int* pingpong = nullptr);

template <typename T>
void gather_fa_graph_fixed(T** d_output_ptrs, Tensor<T>& att_heads,
                           int* d_segment_info,
                           cudaStream_t stream = nullptr);

template <typename T>
void gemv_mlp_fused(Tensor<T>* hidden_states,
                    const Tensor<T>* merged_mlp_weight,
                    Tensor<T>* gate_buf_silu, cudaStream_t stream = nullptr);

}  // namespace op::legacy
