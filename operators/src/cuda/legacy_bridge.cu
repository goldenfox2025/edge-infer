#include "operators/cuda/legacy/legacy_bridge.cuh"

#include <cuda_bf16.h>

#include "cuda/legacy/legacy_cuda_api.cuh"

namespace op::legacy {

template <typename T>
void gather(Tensor<T>* output, const Tensor<uint32_t>* input,
            const Tensor<T>* embedding_table, cudaStream_t stream) {
  cuda_OP::gather(output, input, embedding_table, stream);
}

template <typename T>
void compute_attention_scores_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                                      Tensor<T>& att_scores, size_t head_dim,
                                      cudaStream_t stream) {
  cuda_OP::compute_attention_scores_prefill(Q, K, att_scores, head_dim,
                                            stream);
}

template <typename T>
void compute_attention_output_prefill(const Tensor<T>& att_scores,
                                      const Tensor<T>& V,
                                      Tensor<T>& att_output, size_t n_heads,
                                      size_t head_dim, size_t total_seq_len,
                                      size_t n_kv_heads,
                                      cudaStream_t stream) {
  cuda_OP::compute_att_output_prefill(att_scores, V, att_output, n_heads,
                                      head_dim, total_seq_len, n_kv_heads,
                                      stream);
}

template <typename T>
void dynamic_flash_attention(Tensor<T>& Q, const Tensor<T>& K,
                             const Tensor<T>& V, Tensor<T>& output,
                             int n_kv_heads, cudaStream_t stream) {
  cuda_OP::dynamic_flash_attention_wrapper(Q, K, V, output, n_kv_heads,
                                           stream);
}

template <typename T>
void flash_attention_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                             const Tensor<T>& V, Tensor<T>& output,
                             int n_heads, int n_kv_heads, int head_dim,
                             int seq_len, int total_seq_len, int offset,
                             cudaStream_t stream) {
  cuda_OP::flash_attention_prefill(Q, K, V, output, n_heads, n_kv_heads,
                                   head_dim, seq_len, total_seq_len, offset,
                                   stream);
}

template <typename T>
void softmax(Tensor<T>* output, const Tensor<T>* input, int dim, bool mask,
             int offset, cudaStream_t stream) {
  cuda_OP::softmax(output, input, dim, mask, offset, stream);
}

template <typename T>
void matmul_quantized_gemv(const Tensor<T>& input,
                           const Tensor<int32_t>& qweight,
                           const Tensor<T>& scales,
                           const Tensor<int32_t>& qzeros, int group_size,
                           Tensor<T>* output, cudaStream_t stream,
                           const Tensor<T>* bias) {
  cuda_OP::matmul_quantized_gemv(input, qweight, scales, qzeros, group_size,
                                 output, stream, bias);
}

template <typename T>
void gemv_qkv_rope(Tensor<T>* hidden_states,
                   const Tensor<T>* merged_qkv_weight, Tensor<T>* q_buf,
                   Tensor<T>* k_buf, Tensor<T>* v_buf,
                   const Tensor<T>* merged_qkv_bias, size_t* d_rope_offset,
                   const Tensor<float>* rope_sin_cos_cache,
                   int* d_offset_array, int layer_idx, int q_dim, int k_dim,
                   int v_dim, int n_heads, int n_kv_heads, int head_dim,
                   cudaStream_t stream, int n_layers, int* pingpong) {
  cuda_OP::gemv_qkv_rope(hidden_states, merged_qkv_weight, q_buf, k_buf, v_buf,
                         merged_qkv_bias, d_rope_offset, rope_sin_cos_cache,
                         d_offset_array, layer_idx, q_dim, k_dim, v_dim,
                         n_heads, n_kv_heads, head_dim, stream, n_layers,
                         pingpong);
}

template <typename T>
void rope_with_precomputed_cache(Tensor<T>* tensor, const size_t* d_offset,
                                 const Tensor<float>* rope_sin_cos_cache,
                                 cudaStream_t stream, int* d_offset_array,
                                 int layer_idx, int n_layers,
                                 int* pingpong) {
  cuda_OP::rope_with_precomputed_cache(tensor, d_offset, rope_sin_cos_cache,
                                       stream, d_offset_array, layer_idx,
                                       n_layers, pingpong);
}

template <typename T>
void flash_attention_graph_fixed(Tensor<T>& q, const Tensor<T>& k,
                                 const Tensor<T>& v, T** d_output_ptrs,
                                 int* d_segment_info, int n_kv_heads,
                                 cudaStream_t stream, int* pingpong) {
  cuda_OP::flash_attention_graph_fixed(q, k, v, d_output_ptrs, d_segment_info,
                                       n_kv_heads, stream, pingpong);
}

template <typename T>
void gather_fa_graph_fixed(T** d_output_ptrs, Tensor<T>& att_heads,
                           int* d_segment_info, cudaStream_t stream) {
  cuda_OP::gather_fa_graph_fixed(d_output_ptrs, att_heads, d_segment_info,
                                 stream);
}

template <typename T>
void gemv_mlp_fused(Tensor<T>* hidden_states,
                    const Tensor<T>* merged_mlp_weight,
                    Tensor<T>* gate_buf_silu, cudaStream_t stream) {
  cuda_OP::gemv_mlp_fused(hidden_states, merged_mlp_weight, gate_buf_silu,
                          stream);
}

#define INSTANTIATE_LEGACY_BRIDGE(T)                                          \
  template void gather<T>(Tensor<T>*, const Tensor<uint32_t>*,                \
                          const Tensor<T>*, cudaStream_t);                    \
  template void compute_attention_scores_prefill<T>(                          \
      const Tensor<T>&, const Tensor<T>&, Tensor<T>&, size_t, cudaStream_t); \
  template void compute_attention_output_prefill<T>(                          \
      const Tensor<T>&, const Tensor<T>&, Tensor<T>&, size_t, size_t, size_t,\
      size_t, cudaStream_t);                                                  \
  template void dynamic_flash_attention<T>(Tensor<T>&, const Tensor<T>&,      \
                                           const Tensor<T>&, Tensor<T>&, int, \
                                           cudaStream_t);                     \
  template void flash_attention_prefill<T>(                                   \
      const Tensor<T>&, const Tensor<T>&, const Tensor<T>&, Tensor<T>&, int, \
      int, int, int, int, int, cudaStream_t);                                \
  template void softmax<T>(Tensor<T>*, const Tensor<T>*, int, bool, int,      \
                           cudaStream_t);                                     \
  template void matmul_quantized_gemv<T>(                                     \
      const Tensor<T>&, const Tensor<int32_t>&, const Tensor<T>&,            \
      const Tensor<int32_t>&, int, Tensor<T>*, cudaStream_t,                  \
      const Tensor<T>*);                                                      \
  template void gemv_qkv_rope<T>(                                             \
      Tensor<T>*, const Tensor<T>*, Tensor<T>*, Tensor<T>*, Tensor<T>*,      \
      const Tensor<T>*, size_t*, const Tensor<float>*, int*, int, int, int,  \
      int, int, int, int, cudaStream_t, int, int*);                           \
  template void rope_with_precomputed_cache<T>(Tensor<T>*, const size_t*,     \
                                               const Tensor<float>*,          \
                                               cudaStream_t, int*, int, int,  \
                                               int*);                         \
  template void flash_attention_graph_fixed<T>(                               \
      Tensor<T>&, const Tensor<T>&, const Tensor<T>&, T**, int*, int,         \
      cudaStream_t, int*);                                                    \
  template void gather_fa_graph_fixed<T>(T**, Tensor<T>&, int*,               \
                                         cudaStream_t);                       \
  template void gemv_mlp_fused<T>(Tensor<T>*, const Tensor<T>*, Tensor<T>*,   \
                                  cudaStream_t)

INSTANTIATE_LEGACY_BRIDGE(float);
INSTANTIATE_LEGACY_BRIDGE(__nv_bfloat16);

#undef INSTANTIATE_LEGACY_BRIDGE

}  // namespace op::legacy
