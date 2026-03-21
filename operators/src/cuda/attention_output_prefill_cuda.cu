#include "operators/cuda/legacy/legacy_bridge.cuh"
#include "operators/cuda/attention_output_prefill_cuda.cuh"

namespace op {

template <typename T>
void AttentionOutputPrefillCUDAOperator<T>::operator()(
    const Tensor<T>& att_scores, const Tensor<T>& V, Tensor<T>& att_output,
    size_t n_heads, size_t head_dim, size_t total_seq_len, size_t n_kv_heads,
    cudaStream_t stream) {
  legacy::compute_attention_output_prefill(att_scores, V, att_output, n_heads,
                                           head_dim, total_seq_len, n_kv_heads,
                                           stream);
}

template class AttentionOutputPrefillCUDAOperator<float>;
template class AttentionOutputPrefillCUDAOperator<__nv_bfloat16>;

}  // namespace op
