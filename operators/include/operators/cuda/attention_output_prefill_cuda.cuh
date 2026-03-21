#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class AttentionOutputPrefillCUDAOperator
    : public AttentionOutputPrefillOperator<T> {
 public:
  AttentionOutputPrefillCUDAOperator() = default;
  ~AttentionOutputPrefillCUDAOperator() override = default;

  void operator()(const Tensor<T>& att_scores, const Tensor<T>& V,
                  Tensor<T>& att_output, size_t n_heads, size_t head_dim,
                  size_t total_seq_len, size_t n_kv_heads,
                  cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
