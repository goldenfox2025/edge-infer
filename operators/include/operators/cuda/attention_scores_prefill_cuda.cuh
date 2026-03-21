#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class AttentionScoresPrefillCUDAOperator
    : public AttentionScoresPrefillOperator<T> {
 public:
  AttentionScoresPrefillCUDAOperator() = default;
  ~AttentionScoresPrefillCUDAOperator() override = default;

  void operator()(const Tensor<T>& Q, const Tensor<T>& K,
                  Tensor<T>& att_scores, size_t head_dim,
                  cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
