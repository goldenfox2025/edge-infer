#include "operators/cuda/legacy/legacy_bridge.cuh"
#include "operators/cuda/dynamic_flash_attention_cuda.cuh"

namespace op {

template <typename T>
void DynamicFlashAttentionCUDAOperator<T>::operator()(Tensor<T>& Q,
                                                      const Tensor<T>& K,
                                                      const Tensor<T>& V,
                                                      Tensor<T>& output,
                                                      int n_kv_heads,
                                                      cudaStream_t stream) {
  if (workspace_.data_ptr()) {
    legacy::dynamic_flash_attention_with_workspace(Q, K, V, output,
                                                   n_kv_heads, workspace_, stream);
  } else {
    legacy::dynamic_flash_attention(Q, K, V, output, n_kv_heads, stream);
  }
}

template class DynamicFlashAttentionCUDAOperator<float>;
template class DynamicFlashAttentionCUDAOperator<__nv_bfloat16>;

}  // namespace op
