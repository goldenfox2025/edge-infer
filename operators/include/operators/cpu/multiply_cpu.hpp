#pragma once

#include "operators/operator_base.hpp"
#include "operators/core/cpu_reference.hpp"

namespace op {

template <typename T>
class MultiplyCPUOperator : public MultiplyOperator<T> {
 public:
  MultiplyCPUOperator() = default;
  ~MultiplyCPUOperator() override = default;

  void operator()(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
                  cudaStream_t stream = nullptr) override {
    // These compatibility adapters do not support BF16 on the CPU.
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "Multiply operator for __nv_bfloat16 not supported on CPU platform");
    } else {

      size_t total = input_a->numel();

      if (input_b->numel() != total) {
        throw std::runtime_error(
            "Multiply operator: input tensors must have the same size");
      }

      cpu::multiply<T>({input_a->data_ptr(), total}, {input_b->data_ptr(), total},
                       {output->data_ptr(), total});
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
