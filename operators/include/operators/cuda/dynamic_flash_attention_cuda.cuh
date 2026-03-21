#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class DynamicFlashAttentionCUDAOperator
    : public DynamicFlashAttentionOperator<T> {
 public:
  DynamicFlashAttentionCUDAOperator() = default;
  ~DynamicFlashAttentionCUDAOperator() override = default;

  void operator()(Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                  Tensor<T>& output, int n_kv_heads,
                  cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
