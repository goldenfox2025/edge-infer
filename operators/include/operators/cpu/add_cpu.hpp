#pragma once

#include <cuda_bf16.h>

#include <type_traits>

#include "operators/operator_base.hpp"
#include "operators/core/cpu_reference.hpp"

namespace op {

template <typename T>
class AddCPUOperator : public AddOperator<T> {
 public:
  AddCPUOperator() = default;
  ~AddCPUOperator() override = default;

  void operator()(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
                  cudaStream_t stream = nullptr) override {
    // These compatibility adapters do not support BF16 on the CPU.
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "Add operator for __nv_bfloat16 not supported on CPU platform");
    } else {

      size_t total = input_a->numel();

      if (input_b->numel() != total) {
        throw std::runtime_error(
            "Add operator: input tensors must have the same size");
      }

      cpu::add<T>({input_a->data_ptr(), total}, {input_b->data_ptr(), total},
                  {output->data_ptr(), total});
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
