#pragma once

#include <cmath>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class RopeCPUOperator : public RopeOperator<T> {
 public:
  RopeCPUOperator() = default;
  ~RopeCPUOperator() override = default;

  void operator()(Tensor<T>* x, size_t offset, float theta,
                  cudaStream_t stream = nullptr) override {
    // These compatibility adapters do not support BF16 on the CPU.
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "RoPE operator for __nv_bfloat16 not supported on CPU platform");
    } else {
      const auto& sizes = x->sizes();

      if (sizes.size() < 3) {
        throw std::runtime_error("rope: tensor must be at least 3D");
      }

      size_t seq_len, n_heads, head_dim;
      size_t batch_size = 1;

      if (sizes.size() == 3) {
        // 3D layout: [seq_len, n_heads, head_dim].
        seq_len = sizes[0];
        n_heads = sizes[1];
        head_dim = sizes[2];
      } else {
        // Leading dimensions are flattened into batch_size: [..., seq_len, n_heads, head_dim].
        // Flatten all leading dimensions into the batch count.
        for (size_t i = 0; i < sizes.size() - 3; ++i) {
          batch_size *= sizes[i];
        }
        seq_len = sizes[sizes.size() - 3];
        n_heads = sizes[sizes.size() - 2];
        head_dim = sizes[sizes.size() - 1];
      }

      const size_t dim_half = head_dim / 2;

      if (head_dim % 2 != 0) {
        throw std::runtime_error("rope: head_dim must be even");
      }

      for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
          for (size_t h = 0; h < n_heads; h++) {

            T* head_ptr = x->data_ptr() + (b * seq_len * n_heads * head_dim) +
                          (s * n_heads * head_dim) + (h * head_dim);

            for (size_t i = 0; i < dim_half; i++) {
              float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
              float val = (s + offset) * freq;
              float cos_val = cosf(val);
              float sin_val = sinf(val);

              float x0 = static_cast<float>(head_ptr[i]);
              float x1 = static_cast<float>(head_ptr[i + dim_half]);

              head_ptr[i] = static_cast<T>(x0 * cos_val - x1 * sin_val);
              head_ptr[i + dim_half] =
                  static_cast<T>(x0 * sin_val + x1 * cos_val);
            }
          }
        }
      }
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
