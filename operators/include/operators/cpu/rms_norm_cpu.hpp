#pragma once

#include <cmath>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class RmsNormCPUOperator : public RmsNormOperator<T> {
 public:
  RmsNormCPUOperator() = default;
  ~RmsNormCPUOperator() override = default;

  // 实现CPU版本的RMS Norm - 使用一重指针
  void operator()(Tensor<T>* output, Tensor<T>* input, Tensor<T>* weight,
                  float eps, cudaStream_t stream = nullptr) override {
    // 检查是否是BF16类型，CPU不支持BF16
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "RMS Norm operator for __nv_bfloat16 not supported on CPU platform");
    } else {
      // 获取输入张量的形状
      const auto& sizes = input->sizes();

      // 确定特征维度和批次维度
      size_t feature_dim = sizes.back();
      size_t batch_size = 1;

      // 计算批次大小（所有除最后一维外的维度的乘积）
      for (size_t i = 0; i < sizes.size() - 1; ++i) {
        batch_size *= sizes[i];
      }

      // 对每个批次样本进行RMS Norm
      for (size_t b = 0; b < batch_size; ++b) {
        // 计算均方根
        float sum_squares = 0.0f;  // 使用float而不是T来避免类型问题
        for (size_t j = 0; j < feature_dim; ++j) {
          float val =
              static_cast<float>(input->data_ptr()[b * feature_dim + j]);
          sum_squares += val * val;
        }
        float rms = sqrtf(sum_squares / feature_dim + eps);

        // 应用归一化和缩放
        for (size_t j = 0; j < feature_dim; ++j) {
          float normalized =
              static_cast<float>(input->data_ptr()[b * feature_dim + j]) / rms;
          float scaled = normalized * static_cast<float>(weight->data_ptr()[j]);
          output->data_ptr()[b * feature_dim + j] = static_cast<T>(scaled);
        }
      }
    }
  }

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
