#pragma once

#include <cuda_runtime.h>
#include <utility>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class DynamicFlashAttentionCUDAOperator
    : public DynamicFlashAttentionOperator<T> {
 public:
  DynamicFlashAttentionCUDAOperator() = default;
  // Workspace holds up to five branch outputs. Its owner keeps it alive until
  // all work on the supplied stream completes.
  explicit DynamicFlashAttentionCUDAOperator(Tensor<T> workspace)
      : workspace_(std::move(workspace)) {}
  ~DynamicFlashAttentionCUDAOperator() override = default;

  void operator()(Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                  Tensor<T>& output, int n_kv_heads,
                  cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }

 private:
  Tensor<T> workspace_;
};

}  // namespace op
