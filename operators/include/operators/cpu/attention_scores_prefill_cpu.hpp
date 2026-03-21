#pragma once

#include <cmath>
#include <stdexcept>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class AttentionScoresPrefillCPUOperator
    : public AttentionScoresPrefillOperator<T> {
 public:
  AttentionScoresPrefillCPUOperator() = default;
  ~AttentionScoresPrefillCPUOperator() override = default;

  void operator()(const Tensor<T>& Q, const Tensor<T>& K, Tensor<T>& att_scores,
                  size_t head_dim, cudaStream_t stream = nullptr) override {
    (void)stream;

    const auto& q_sizes = Q.sizes();
    const auto& k_sizes = K.sizes();
    if (q_sizes.size() != 3 || k_sizes.size() != 3) {
      throw std::runtime_error(
          "AttentionScoresPrefillCPUOperator expects 3D Q and K tensors");
    }

    const size_t seq_len = q_sizes[0];
    const size_t n_heads = q_sizes[1];
    const size_t total_seq_len = k_sizes[0];
    const size_t n_kv_heads = k_sizes[1];
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) {
      throw std::runtime_error("Invalid head mapping for CPU attention scores");
    }

    const size_t group_size = n_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
      for (size_t head_idx = 0; head_idx < n_heads; ++head_idx) {
        const size_t kv_head_idx = head_idx / group_size;
        for (size_t pos = 0; pos < total_seq_len; ++pos) {
          float score = 0.0f;
          const size_t q_base =
              (token_idx * n_heads + head_idx) * head_dim;
          const size_t k_base =
              (pos * n_kv_heads + kv_head_idx) * head_dim;
          for (size_t dim = 0; dim < head_dim; ++dim) {
            score += static_cast<float>(Q.data_ptr()[q_base + dim]) *
                     static_cast<float>(K.data_ptr()[k_base + dim]);
          }
          att_scores.data_ptr()[(token_idx * n_heads + head_idx) * total_seq_len +
                                pos] = static_cast<T>(score * scale);
        }
      }
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
