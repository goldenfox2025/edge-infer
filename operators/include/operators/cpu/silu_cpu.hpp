#pragma once

#include <cmath>

#include "operators/operator_base.hpp"
#include "operators/core/cpu_reference.hpp"

namespace op {

template <typename T>
class SiluCPUOperator : public SiluOperator<T> {
 public:
  SiluCPUOperator() = default;
  ~SiluCPUOperator() override = default;

  void operator()(Tensor<T>* output, Tensor<T>* input,
                  cudaStream_t stream = nullptr) override {
    // These compatibility adapters do not support BF16 on the CPU.
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "SiLU operator for __nv_bfloat16 not supported on CPU platform");
    } else {

      size_t total = input->numel();

      cpu::silu<T>({input->data_ptr(), total}, {output->data_ptr(), total});
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
