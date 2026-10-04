#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class SiluCUDAOperator : public SiluOperator<T> {
 public:
  SiluCUDAOperator() = default;
  ~SiluCUDAOperator() override = default;

  void operator()(Tensor<T>* output, Tensor<T>* input,
                  cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
