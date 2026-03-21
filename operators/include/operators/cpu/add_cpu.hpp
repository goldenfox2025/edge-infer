#pragma once

#include <cuda_bf16.h>

#include <type_traits>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class AddCPUOperator : public AddOperator<T> {
 public:
  AddCPUOperator() = default;
  ~AddCPUOperator() override = default;

  // 实现CPU版本的Add - 使用一重指针
  void operator()(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
                  cudaStream_t stream = nullptr) override {
    // 检查是否是BF16类型，CPU不支持BF16
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "Add operator for __nv_bfloat16 not supported on CPU platform");
    } else {
      // 获取输入张量的大小
      size_t total = input_a->numel();

      // 检查输入张量的大小是否一致
      if (input_b->numel() != total) {
        throw std::runtime_error(
            "Add operator: input tensors must have the same size");
      }

      // 逐元素相加
      for (size_t i = 0; i < total; i++) {
        output->data_ptr()[i] = input_a->data_ptr()[i] + input_b->data_ptr()[i];
      }
    }
  }

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
