#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class MultiplyCUDAOperator : public MultiplyOperator<T> {
 public:
  MultiplyCUDAOperator() = default;
  ~MultiplyCUDAOperator() override = default;

  void operator()(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
                  cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
