#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class SampleCUDAOperator : public SampleOperator<T> {
 public:
  SampleCUDAOperator() = default;
  ~SampleCUDAOperator() override = default;

  uint32_t* operator()(Tensor<T>&& logits, float temperature, float top_p,
                       size_t top_k, curandState* d_states,
                       cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
