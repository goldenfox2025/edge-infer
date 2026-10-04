#pragma once

#include <cmath>

#include "operators/operator_base.hpp"
#include "operators/core/cpu_reference.hpp"

namespace op {

template <typename T>
class RmsNormCPUOperator : public RmsNormOperator<T> {
 public:
  RmsNormCPUOperator() = default;
  ~RmsNormCPUOperator() override = default;

  void operator()(Tensor<T>* output, Tensor<T>* input, Tensor<T>* weight,
                  float eps, cudaStream_t stream = nullptr) override {
    // These compatibility adapters do not support BF16 on the CPU.
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "RMS Norm operator for __nv_bfloat16 not supported on CPU platform");
    } else {

      const auto& sizes = input->sizes();

      size_t feature_dim = sizes.back();
      size_t batch_size = 1;

      // Flatten all dimensions except the last into the row count.
      for (size_t i = 0; i < sizes.size() - 1; ++i) {
        batch_size *= sizes[i];
      }

      cpu::rms_norm<T>({input->data_ptr(), input->numel()},
                       {weight->data_ptr(), weight->numel()},
                       {output->data_ptr(), output->numel()},
                       batch_size, feature_dim, eps);
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
