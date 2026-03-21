#pragma once

#include <stdexcept>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class AttentionOutputPrefillCPUOperator
    : public AttentionOutputPrefillOperator<T> {
 public:
  AttentionOutputPrefillCPUOperator() = default;
  ~AttentionOutputPrefillCPUOperator() override = default;

  void operator()(const Tensor<T>& att_scores, const Tensor<T>& V,
                  Tensor<T>& att_output, size_t n_heads, size_t head_dim,
                  size_t total_seq_len, size_t n_kv_heads,
                  cudaStream_t stream = nullptr) override {
    (void)stream;

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) {
      throw std::runtime_error("Invalid head mapping for CPU attention output");
    }

    const auto& output_sizes = att_output.sizes();
    if (output_sizes.size() != 3) {
      throw std::runtime_error(
          "AttentionOutputPrefillCPUOperator expects a 3D output tensor");
    }

    const size_t seq_len = output_sizes[0];
    const size_t group_size = n_heads / n_kv_heads;

    for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
      for (size_t head_idx = 0; head_idx < n_heads; ++head_idx) {
        const size_t kv_head_idx = head_idx / group_size;
        const size_t out_base =
            (token_idx * n_heads + head_idx) * head_dim;
        std::fill(att_output.data_ptr() + out_base,
                  att_output.data_ptr() + out_base + head_dim, static_cast<T>(0.0f));

        for (size_t pos = 0; pos < total_seq_len; ++pos) {
          const float weight =
              static_cast<float>(att_scores.data_ptr()[(token_idx * n_heads + head_idx) *
                                                           total_seq_len +
                                                       pos]);
          const size_t v_base =
              (pos * n_kv_heads + kv_head_idx) * head_dim;
          for (size_t dim = 0; dim < head_dim; ++dim) {
            const float accum =
                static_cast<float>(att_output.data_ptr()[out_base + dim]) +
                weight * static_cast<float>(V.data_ptr()[v_base + dim]);
            att_output.data_ptr()[out_base + dim] = static_cast<T>(accum);
          }
        }
      }
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
