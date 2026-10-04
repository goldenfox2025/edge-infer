#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class RopeCUDAOperator : public RopeOperator<T> {
 public:
  RopeCUDAOperator() = default;
  ~RopeCUDAOperator() override = default;

  void operator()(Tensor<T>* x, size_t offset, float theta,
                  cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
