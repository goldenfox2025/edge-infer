#include "operators/cuda/legacy/legacy_bridge.cuh"
#include "operators/cuda/attention_scores_prefill_cuda.cuh"

namespace op {

template <typename T>
void AttentionScoresPrefillCUDAOperator<T>::operator()(
    const Tensor<T>& Q, const Tensor<T>& K, Tensor<T>& att_scores,
    size_t head_dim, cudaStream_t stream) {
  legacy::compute_attention_scores_prefill(Q, K, att_scores, head_dim,
                                            stream);
}

template class AttentionScoresPrefillCUDAOperator<float>;
template class AttentionScoresPrefillCUDAOperator<__nv_bfloat16>;

}  // namespace op
