#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class FlashAttentionPrefillCUDAOperator
    : public FlashAttentionPrefillOperator<T> {
 public:
  FlashAttentionPrefillCUDAOperator() = default;
  ~FlashAttentionPrefillCUDAOperator() override = default;

  void operator()(const Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                  Tensor<T>& output, int n_heads, int n_kv_heads,
                  int head_dim, int seq_len, int total_seq_len, int offset,
                  cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
