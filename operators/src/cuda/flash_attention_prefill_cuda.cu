#include "operators/cuda/legacy/legacy_bridge.cuh"
#include "operators/cuda/flash_attention_prefill_cuda.cuh"

namespace op {

template <typename T>
void FlashAttentionPrefillCUDAOperator<T>::operator()(
    const Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
    Tensor<T>& output, int n_heads, int n_kv_heads, int head_dim, int seq_len,
    int total_seq_len, int offset, cudaStream_t stream) {
  legacy::flash_attention_prefill(Q, K, V, output, n_heads, n_kv_heads,
                                   head_dim, seq_len, total_seq_len, offset,
                                   stream);
}

template class FlashAttentionPrefillCUDAOperator<float>;
template class FlashAttentionPrefillCUDAOperator<__nv_bfloat16>;

}  // namespace op
