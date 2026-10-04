#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class RmsNormCUDAOperator : public RmsNormOperator<T> {
 public:
  RmsNormCUDAOperator() = default;
  ~RmsNormCUDAOperator() override = default;

  void operator()(Tensor<T>* output, Tensor<T>* input, Tensor<T>* weight,
                  float eps, cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
