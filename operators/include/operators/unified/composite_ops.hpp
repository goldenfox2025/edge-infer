#pragma once

#include "tensor.hpp"

namespace op::detail {

template <typename T, typename Facade>
struct UnifiedCompositeOps {
  static void add_rms(Facade& ops, Tensor<T>* hidden_states,
                      Tensor<T>* residual, Tensor<T>* update,
                      Tensor<T>* norm_weight, float eps,
                      cudaStream_t stream = nullptr) {
    ops.add(residual, residual, update, stream);
    ops.rms_norm(hidden_states, residual, norm_weight, eps, stream);
  }

  static void silu_multiply(Facade& ops, Tensor<T>* output, Tensor<T>* input_a,
                            Tensor<T>* input_b,
                            cudaStream_t stream = nullptr) {
    ops.silu(output, input_a, stream);
    ops.multiply(output, output, input_b, stream);
  }

  static void dynamic_flash_attention_fallback(Facade& ops, Tensor<T>& Q,
                                               const Tensor<T>& K,
                                               const Tensor<T>& V,
                                               Tensor<T>& output,
                                               int n_kv_heads,
                                               cudaStream_t stream = nullptr) {
    Tensor<T> att_scores({Q.sizes()[0], Q.sizes()[1], K.sizes()[0]},
                         ops.device() == Device::CUDA ? Device::CUDA
                                                      : Device::CPU);
    ops.compute_attention_scores_prefill(Q, K, att_scores, Q.sizes()[2],
                                         stream);
    ops.softmax(&att_scores, &att_scores, 2, false, 0, stream);
    ops.compute_attention_output_prefill(att_scores, V, output, Q.sizes()[1],
                                         Q.sizes()[2], K.sizes()[0],
                                         n_kv_heads, stream);
  }

  static void flash_attention_prefill_fallback(
      Facade& ops, const Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
      Tensor<T>& output, int n_heads, int n_kv_heads, int head_dim,
      int seq_len, int total_seq_len, int offset,
      cudaStream_t stream = nullptr) {
    Tensor<T> att_scores({static_cast<size_t>(seq_len),
                          static_cast<size_t>(n_heads),
                          static_cast<size_t>(total_seq_len)},
                         ops.device() == Device::CUDA ? Device::CUDA
                                                      : Device::CPU);
    ops.compute_attention_scores_prefill(Q, K, att_scores, head_dim, stream);
    ops.softmax(&att_scores, &att_scores, 2, true, offset, stream);
    ops.compute_attention_output_prefill(att_scores, V, output, n_heads,
                                         head_dim, total_seq_len, n_kv_heads,
                                         stream);
  }
};

}  // namespace op::detail
