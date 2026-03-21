#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class KvCacheWriteCUDAOperator : public KvCacheWriteOperator<T> {
 public:
  KvCacheWriteCUDAOperator() = default;
  ~KvCacheWriteCUDAOperator() override = default;

  void operator()(const Tensor<T>* src_k, const Tensor<T>* src_v,
                  Tensor<T>* dst_k_cache, Tensor<T>* dst_v_cache,
                  size_t offset, cudaStream_t stream = nullptr) override;

  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
