#pragma once

#include <cmath>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class SiluCPUOperator : public SiluOperator<T> {
 public:
  SiluCPUOperator() = default;
  ~SiluCPUOperator() override = default;

  // 实现CPU版本的SiLU - 使用一重指针
  void operator()(Tensor<T>* output, Tensor<T>* input,
                  cudaStream_t stream = nullptr) override {
    // 检查是否是BF16类型，CPU不支持BF16
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "SiLU operator for __nv_bfloat16 not supported on CPU platform");
    } else {
      // 获取输入张量的大小
      size_t total = input->numel();

      // 对每个元素应用SiLU激活函数
      for (size_t i = 0; i < total; i++) {
        float val = static_cast<float>(input->data_ptr()[i]);
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        output->data_ptr()[i] = static_cast<T>(val / (1.0f + expf(-val)));
      }
    }
  }

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
