#pragma once

#include <cmath>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class RopeCPUOperator : public RopeOperator<T> {
 public:
  RopeCPUOperator() = default;
  ~RopeCPUOperator() override = default;

  // 实现CPU版本的RoPE - 使用一重指针
  void operator()(Tensor<T>* x, size_t offset, float theta,
                  cudaStream_t stream = nullptr) override {
    // 检查是否是BF16类型，CPU不支持BF16
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "RoPE operator for __nv_bfloat16 not supported on CPU platform");
    } else {
      const auto& sizes = x->sizes();

      // 检查张量维度
      if (sizes.size() < 3) {
        throw std::runtime_error("rope: tensor must be at least 3D");
      }

      // 确定序列长度、头数和头维度
      size_t seq_len, n_heads, head_dim;
      size_t batch_size = 1;

      // 根据张量维度确定批次大小和其他参数
      if (sizes.size() == 3) {
        // 3D张量: [seq_len, n_heads, head_dim]
        seq_len = sizes[0];
        n_heads = sizes[1];
        head_dim = sizes[2];
      } else {
        // 4D或更高维张量: [batch_size, seq_len, n_heads, head_dim]
        // 计算批次大小（除了最后三个维度外的所有维度的乘积）
        for (size_t i = 0; i < sizes.size() - 3; ++i) {
          batch_size *= sizes[i];
        }
        seq_len = sizes[sizes.size() - 3];
        n_heads = sizes[sizes.size() - 2];
        head_dim = sizes[sizes.size() - 1];
      }

      const size_t dim_half = head_dim / 2;

      // 检查head_dim是否为偶数
      if (head_dim % 2 != 0) {
        throw std::runtime_error("rope: head_dim must be even");
      }

      // 对每个批次样本应用RoPE
      for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
          for (size_t h = 0; h < n_heads; h++) {
            // 计算当前头的指针位置
            T* head_ptr = x->data_ptr() + (b * seq_len * n_heads * head_dim) +
                          (s * n_heads * head_dim) + (h * head_dim);

            // 对每个维度对应用旋转
            for (size_t i = 0; i < dim_half; i++) {
              float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
              float val = (s + offset) * freq;
              float cos_val = cosf(val);
              float sin_val = sinf(val);

              // 获取原始值并转换为float
              float x0 = static_cast<float>(head_ptr[i]);
              float x1 = static_cast<float>(head_ptr[i + dim_half]);

              // 应用旋转并转换回原始类型
              head_ptr[i] = static_cast<T>(x0 * cos_val - x1 * sin_val);
              head_ptr[i + dim_half] =
                  static_cast<T>(x0 * sin_val + x1 * cos_val);
            }
          }
        }
      }
    }
  }

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
