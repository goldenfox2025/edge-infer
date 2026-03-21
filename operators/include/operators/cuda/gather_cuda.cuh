#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class GatherCUDAOperator : public GatherOperator<T> {
 public:
  GatherCUDAOperator() = default;
  ~GatherCUDAOperator() override = default;

  void operator()(Tensor<T>* output, const Tensor<uint32_t>* input,
                  const Tensor<T>* embedding_table,
                  cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
