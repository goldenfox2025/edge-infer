#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class RmsNormCUDAOperator : public RmsNormOperator<T> {
 public:
  RmsNormCUDAOperator() = default;
  ~RmsNormCUDAOperator() override = default;

  // 实现CUDA版本的RMS Norm - 使用一重指针
  void operator()(Tensor<T>* output, Tensor<T>* input, Tensor<T>* weight,
                  float eps, cudaStream_t stream = nullptr) override;

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
